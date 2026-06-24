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

import pandas as pd
import requests

from .universe import _HEADERS, WIKI_SP500_URL, fetch_sp500_symbols, to_yf_symbol


def _norm(x: object) -> str | None:
    if not isinstance(x, str):
        if pd.isna(x):
            return None
        x = str(x)
    sym = to_yf_symbol(x)
    return sym or None


def fetch_sp500_changes(timeout: float = 30.0) -> pd.DataFrame:
    """Wikipedia S&P 500 changes log as columns ``eff`` (datetime), ``add``, ``rem``.

    Tickers are normalized for yfinance; rows without a parseable date are dropped.
    """
    html = requests.get(WIKI_SP500_URL, headers=_HEADERS, timeout=timeout).text
    ch = pd.read_html(io.StringIO(html))[1]
    ch.columns = ["eff", "add", "add_sec", "rem", "rem_sec", "reason"]
    ch = ch.iloc[1:].copy()  # drop the repeated sub-header row
    ch["eff"] = pd.to_datetime(ch["eff"], errors="coerce")
    ch = ch.dropna(subset=["eff"])
    ch["add"] = ch["add"].map(_norm)
    ch["rem"] = ch["rem"].map(_norm)
    return ch[["eff", "add", "rem"]].reset_index(drop=True)


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
