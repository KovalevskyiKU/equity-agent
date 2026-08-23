"""Point-in-time fundamentals from Finnhub ``financials-reported`` (annual, as-filed).

Design choices that keep this honest and free-tier-viable:

* **Source = Finnhub financials-reported.** FMP's free tier caps history to the
  last 5 quarters (useless for a decade backtest); Finnhub returns the full
  as-reported history for free.
* **freq=annual (10-K).** The quarterly as-reported feed mixes period contexts
  (interim filings carry *year-to-date* figures, not the discrete quarter), which
  silently triple-counts a naive TTM. Annual filings carry clean full-year flows.
  Annual fundamentals lagged to the filing date are the academically standard
  factor input (cf. Fama-French's annual book equity).
* **as-of = filedDate.** The 10-K for FY ending Dec 31 isn't public until it's
  filed weeks/months later; indexing by ``filed_date`` (not period end) is what
  prevents look-ahead.
* **EPS, not shares.** The diluted-EPS tag is reported consistently; the
  shares-outstanding tag is missing for many filers — so value uses EPS/price
  directly and we never depend on a flaky share count.

One request per symbol returns the whole history. Stored as one parquet per symbol
under ``data/fundamentals/``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import PROJECT_ROOT, get_settings, load_config

logger = logging.getLogger("equity_agent")

_URL = "https://finnhub.io/api/v1/stock/financials-reported"

# Ordered candidate us-gaap tags per field; first present wins (sector variants).
_CONCEPTS: dict[str, list[str]] = {
    "net_income": [
        "us-gaap_NetIncomeLoss",
        "us-gaap_ProfitLoss",
        "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "revenue": [
        "us-gaap_Revenues",
        "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap_RevenuesNetOfInterestExpense",
        "us-gaap_SalesRevenueNet",
        "us-gaap_SalesRevenueGoodsNet",
    ],
    "gross_profit": ["us-gaap_GrossProfit"],
    "equity": [
        "us-gaap_StockholdersEquity",
        "us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "eps": [
        "us-gaap_EarningsPerShareDiluted",
        "us-gaap_EarningsPerShareBasicAndDiluted",
        "us-gaap_EarningsPerShareBasic",
    ],
    # Added for the second research pass: assets power gross-profits-to-assets
    # (Novy-Marx) and asset growth; shares power net share issuance.
    "assets": ["us-gaap_Assets", "us-gaap_AssetsNet"],
    "shares": [
        "us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding",
        "us-gaap_WeightedAverageNumberOfSharesOutstandingBasic",
        "us-gaap_CommonStockSharesOutstanding",
        "dei_EntityCommonStockSharesOutstanding",
    ],
    "cash_flow_ops": ["us-gaap_NetCashProvidedByUsedInOperatingActivities"],
}
_SECTION = {
    "net_income": "ic",
    "revenue": "ic",
    "gross_profit": "ic",
    "eps": "ic",
    "equity": "bs",
    "assets": "bs",
    "shares": "ic",
    "cash_flow_ops": "cf",
}


def _fundamentals_dir() -> Path:
    d = PROJECT_ROOT / load_config().data_dir / "fundamentals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _num(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _extract(section: list[dict], candidates: list[str]) -> float:
    """First value whose concept matches a candidate tag (exact, then suffix match)."""
    by_concept = {c.get("concept"): c.get("value") for c in section}
    for tag in candidates:
        if tag in by_concept:
            return _num(by_concept[tag])
    for tag in candidates:  # suffix fallback (e.g. ifrs-full_ prefixes)
        bare = tag.split("_", 1)[-1]
        for concept, value in by_concept.items():
            if isinstance(concept, str) and concept.endswith(bare):
                return _num(value)
    return float("nan")


def parse_filings(payload: list[dict]) -> pd.DataFrame:
    """Finnhub filing list -> rows: filed_date, end_date + extracted annual fields."""
    rows: list[dict[str, object]] = []
    for f in payload:
        rep = f.get("report", {})
        row: dict[str, object] = {
            "filed_date": pd.to_datetime(f.get("filedDate"), errors="coerce"),
            "end_date": pd.to_datetime(f.get("endDate"), errors="coerce"),
        }
        for field, candidates in _CONCEPTS.items():
            row[field] = _extract(rep.get(_SECTION[field], []), candidates)
        rows.append(row)
    df = pd.DataFrame(rows).dropna(subset=["filed_date", "end_date"])
    if df.empty:
        return df
    # One row per fiscal year-end; keep the earliest filing (the original 10-K).
    df = df.sort_values("filed_date").drop_duplicates("end_date", keep="first")
    return df.reset_index(drop=True)


def fetch_financials(symbol: str, token: str, timeout: float = 30.0) -> pd.DataFrame:
    """Fetch and parse the full annual as-reported history for one symbol."""
    url = f"{_URL}?symbol={symbol}&freq=annual&token={token}"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        logger.warning("[%s] finnhub status %s", symbol, r.status_code)
        return pd.DataFrame()
    payload = r.json().get("data", [])
    return parse_filings(payload) if payload else pd.DataFrame()


def ingest_fundamentals(
    symbols: list[str], token: str | None = None, rate_per_min: float = 55.0
) -> dict[str, int]:
    """Fetch annual as-reported fundamentals for each symbol -> parquet. {symbol: rows}.

    Rate-limited under the Finnhub free tier (60/min). Re-running overwrites (full
    history each call), so it is safe to resume.
    """
    token = token or get_settings().finnhub_api_key
    if not token:
        raise RuntimeError("FINNHUB_API_KEY not set")
    out_dir = _fundamentals_dir()
    delay = 60.0 / rate_per_min
    counts: dict[str, int] = {}
    for i, sym in enumerate(symbols):
        try:
            df = fetch_financials(sym, token)
        except requests.RequestException as e:
            logger.warning("[%s] fundamentals fetch failed: %s", sym, e)
            df = pd.DataFrame()
        counts[sym] = len(df)
        if not df.empty:
            df.to_parquet(out_dir / f"{sym.replace('^', '_')}.parquet")
        if (i + 1) % 50 == 0:
            got = sum(1 for v in counts.values() if v > 0)
            logger.info("fundamentals %d/%d fetched (%d with data)", i + 1, len(symbols), got)
        time.sleep(delay)
    return counts


def load_fundamentals(symbol: str) -> pd.DataFrame:
    """Read a symbol's stored fundamentals (empty frame if not fetched)."""
    path = _fundamentals_dir() / f"{symbol.replace('^', '_')}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
