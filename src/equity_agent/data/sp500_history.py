"""Point-in-time S&P 500 membership — the survivorship-bias fix.

Reconstructs which tickers were in the index on any past date from two Wikipedia
tables: the *current* constituents and the *changes* log (added/removed with an
effective date). The trick is to walk the changes **backward** from today: a
change effective on date *D* (added *A*, removed *R*) means that just before *D*,
*A* was **not yet** a member and *R* **still was**. So to get membership as of a
query date *d*, undo every change effective after *d*.

This removes the **additions bias** completely (we only ever rank names that were
actually in the index on the rebalance date) and the **deletions bias** as far as
price history exists for dropped names (yfinance has many — but not all — delisted
tickers; the residual gap is reported, not hidden).

Maintenance helper (needs network + lxml), like :mod:`universe`.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from .universe import _HEADERS, WIKI_SP500_URL, fetch_sp500_symbols, to_yf_symbol

logger = logging.getLogger("equity_agent")


def _norm(x: object) -> str | None:
    if not isinstance(x, str):
        if pd.isna(x):
            return None
        x = str(x)
    sym = to_yf_symbol(x)
    return sym or None


_CHANGES_COLS = ["eff", "add", "add_sec", "rem", "rem_sec", "reason"]
# Wikipedia dropped the "Selected changes" table from the live page in 2026-07; this
# revision still carries it and is the reproducible fallback.
_FALLBACK_OLDID = 1360191682


def _changes_cache_path() -> Path:
    from ..config import PROJECT_ROOT, load_config

    d = PROJECT_ROOT / load_config().data_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / "sp500_changes.csv"


def _find_changes_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Pick the changes table by CONTENT (Added/Removed columns), not by position.

    Indexing ``tables[1]`` broke when the page was restructured; matching on the
    header keeps this working across edits.
    """
    for t in tables:
        header = " ".join(str(c) for c in t.columns)
        if "Added" in header and "Removed" in header and t.shape[1] >= 5:
            return t
    return None


def _parse_changes(raw: pd.DataFrame) -> pd.DataFrame:
    ch = raw.copy()
    ch.columns = _CHANGES_COLS
    ch = ch.iloc[1:].copy()  # drop the repeated sub-header row
    ch["eff"] = pd.to_datetime(ch["eff"], errors="coerce")
    ch = ch.dropna(subset=["eff"])
    ch["add"] = ch["add"].map(_norm)
    ch["rem"] = ch["rem"].map(_norm)
    return ch[["eff", "add", "rem"]].reset_index(drop=True)


def _fetch_tables(url: str, params: dict[str, str] | None, timeout: float) -> list[pd.DataFrame]:
    html = requests.get(url, params=params, headers=_HEADERS, timeout=timeout).text
    return pd.read_html(io.StringIO(html))


def fetch_sp500_changes(timeout: float = 30.0, use_cache: bool = True) -> pd.DataFrame:
    """S&P 500 changes log as ``eff`` (datetime), ``add``, ``rem`` (yfinance-normalized).

    Resolution order, so point-in-time research stays reproducible even when the
    upstream page changes: live page -> a known-good old revision -> local CSV cache.
    A successful fetch refreshes the cache.
    """
    cache = _changes_cache_path()

    for url, params in (
        (WIKI_SP500_URL, None),
        (
            "https://en.wikipedia.org/w/index.php",
            {"title": "List_of_S&P_500_companies", "oldid": str(_FALLBACK_OLDID)},
        ),
    ):
        try:
            table = _find_changes_table(_fetch_tables(url, params, timeout))
        except Exception as e:  # noqa: BLE001 - network/parse issues fall through
            logger.warning("changes fetch failed (%s): %s", url, e)
            continue
        if table is not None:
            parsed = _parse_changes(table)
            parsed.to_csv(cache, index=False)
            return parsed

    if use_cache and cache.exists():
        logger.warning("using cached S&P 500 changes (%s) - upstream unavailable", cache)
        cached = pd.read_csv(cache)
        cached["eff"] = pd.to_datetime(cached["eff"], errors="coerce")
        return cached.dropna(subset=["eff"])

    raise RuntimeError("could not obtain the S&P 500 changes table (live, revision, or cache)")


def ever_members(current: list[str], changes: pd.DataFrame, start: object) -> list[str]:
    """Union of all tickers that were members at some point on/after ``start``.

    Current members plus everything removed since ``start`` (added-then-removed
    names are captured by the removal). This is the price-data universe to ingest.
    """
    start_ts = pd.Timestamp(start)
    recent = changes[changes["eff"] >= start_ts]
    s = set(current)
    s.update(t for t in recent["rem"].tolist() if isinstance(t, str) and t)
    return sorted(s)


def members_asof(d: object, current: list[str], changes: pd.DataFrame) -> set[str]:
    """Set of index members as of date ``d`` (undo every change effective after ``d``)."""
    s = set(current)
    fut = changes[changes["eff"] > pd.Timestamp(d)]
    for add, rem in zip(fut["add"], fut["rem"], strict=False):
        if isinstance(add, str) and add:
            s.discard(add)
        if isinstance(rem, str) and rem:
            s.add(rem)
    return s


def membership_mask(
    index: pd.Index, current: list[str] | None = None, changes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Boolean (date x symbol) mask: True where the symbol was a member as-of that date.

    Computed once per constant-membership segment (between change dates) and
    broadcast across the segment's trading days, so it is cheap even over a wide
    universe. Columns are the union of all ever-members in the index's date range.
    """
    if current is None:
        current = fetch_sp500_symbols()
    if changes is None:
        changes = fetch_sp500_changes()

    dt = pd.DatetimeIndex(pd.to_datetime(index))
    lo, hi = dt.min(), dt.max()
    syms = ever_members(current, changes, lo)
    mask = pd.DataFrame(False, index=index, columns=syms)

    in_range = sorted({c for c in changes["eff"] if lo < c <= hi})
    # Membership from the latest in-range change date through the end = current set.
    s = set(current)
    boundaries: list[tuple[pd.Timestamp, frozenset[str]]] = []
    if in_range:
        boundaries.append((in_range[-1], frozenset(s)))
        for i in range(len(in_range) - 1, 0, -1):
            for add, rem in zip(*_changes_on(changes, in_range[i]), strict=True):
                if isinstance(add, str) and add:
                    s.discard(add)
                if isinstance(rem, str) and rem:
                    s.add(rem)
            boundaries.append((in_range[i - 1], frozenset(s)))
        for add, rem in zip(*_changes_on(changes, in_range[0]), strict=True):
            if isinstance(add, str) and add:
                s.discard(add)
            if isinstance(rem, str) and rem:
                s.add(rem)
    # The earliest segment (before the first in-range change) covers the start.
    boundaries.append((lo, frozenset(s)))
    boundaries.sort(key=lambda b: b[0])

    starts = [b[0] for b in boundaries]
    for i, (seg_start, members) in enumerate(boundaries):
        seg_end = starts[i + 1] if i + 1 < len(boundaries) else hi + pd.Timedelta(days=1)
        in_seg = (dt >= seg_start) & (dt < seg_end)
        cols = [c for c in members if c in mask.columns]
        if in_seg.any() and cols:
            mask.loc[in_seg, cols] = True
    return mask


def _changes_on(changes: pd.DataFrame, eff: pd.Timestamp) -> tuple[list, list]:
    rows = changes[changes["eff"] == eff]
    return list(rows["add"]), list(rows["rem"])


# --------------------------------------------------------------------------- #
# Generic index membership (S&P 400 mid-cap, S&P 600 small-cap, ...)
# --------------------------------------------------------------------------- #
# The reconstruction helpers below (ever_members / members_asof / membership_mask)
# are already index-agnostic — they take `current` and `changes` as arguments. Only
# the fetchers were hardcoded to the S&P 500 page, so these generalize them.
INDEX_PAGES = {
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    "sp500": WIKI_SP500_URL,
}


def _find_constituents_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """The constituents table is the one with a Symbol/Ticker column and many rows."""
    for t in tables:
        header = " ".join(str(c) for c in t.columns)
        if ("Symbol" in header or "Ticker" in header) and len(t) > 100:
            return t
    return None


def fetch_index_symbols(index: str, timeout: float = 30.0) -> list[str]:
    """Current constituents of a supported index, normalized for yfinance."""
    tables = _fetch_tables(INDEX_PAGES[index], None, timeout)
    table = _find_constituents_table(tables)
    if table is None:
        raise RuntimeError(f"no constituents table found for {index}")
    col = "Symbol" if "Symbol" in table.columns else "Ticker"
    return sorted({s for s in table[col].map(_norm).tolist() if s})


def fetch_index_changes(index: str, timeout: float = 30.0) -> pd.DataFrame:
    """Changes log for a supported index (same schema as fetch_sp500_changes).

    Cached to ``data/<index>_changes.csv`` so research stays reproducible if the
    upstream page is restructured (which already happened to the S&P 500 page).
    """
    cache = _changes_cache_path().with_name(f"{index}_changes.csv")
    try:
        table = _find_changes_table(_fetch_tables(INDEX_PAGES[index], None, timeout))
    except Exception as e:  # noqa: BLE001 - fall through to the cache
        logger.warning("[%s] changes fetch failed: %s", index, e)
        table = None
    if table is not None:
        parsed = _parse_changes(table)
        parsed.to_csv(cache, index=False)
        return parsed
    if cache.exists():
        logger.warning("[%s] using cached changes (%s)", index, cache)
        cached = pd.read_csv(cache)
        cached["eff"] = pd.to_datetime(cached["eff"], errors="coerce")
        return cached.dropna(subset=["eff"])
    raise RuntimeError(f"could not obtain the changes table for {index}")
