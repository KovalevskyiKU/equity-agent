"""S&P 500 constituent universe — point-in-time snapshot from Wikipedia.

This is a *maintenance helper*, not a runtime dependency: the traded universe
lives in ``config.yaml`` (config-driven). Use this to (re)generate that list and
to snapshot the constituents (with sector / date-added) for provenance.

SURVIVORSHIP-BIAS WARNING
-------------------------
``fetch_sp500_symbols`` returns *today's* index membership. Running a backtest
over 2015-2026 with today's members is survivorship-biased: companies that were
dropped or delisted along the way are absent, and the current members are
implicitly conditioned on having survived and stayed large enough to remain in
the index. This inflates factor/backtest results. Point-in-time constituents are
hard to get on free tiers, so we document the bias and the as-of date instead,
and prefer names with full-period history when interpreting results.

Needs network access and ``lxml`` (a dev/maintenance dependency, not required to
run the trader).
"""

from __future__ import annotations

import io

import pandas as pd
import requests

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {
    "User-Agent": "equity-agent/0.1 (quant research; https://github.com/KovalevskyiKU/equity-agent)"
}


def to_yf_symbol(symbol: str) -> str:
    """Normalize a ticker to yfinance's convention: class shares use ``-`` not ``.``.

    e.g. Wikipedia's ``BRK.B`` / ``BF.B`` -> yfinance's ``BRK-B`` / ``BF-B``.
    """
    return symbol.strip().upper().replace(".", "-")


def fetch_sp500_table(timeout: float = 30.0) -> pd.DataFrame:
    """Fetch the current S&P 500 constituents table from Wikipedia.

    Returns the raw table plus a ``yf_symbol`` column normalized for yfinance.
    Raises on network / parse failure (caller decides how to handle).
    """
    html = requests.get(WIKI_SP500_URL, headers=_HEADERS, timeout=timeout).text
    df = pd.read_html(io.StringIO(html))[0]
    df["yf_symbol"] = df["Symbol"].astype(str).map(to_yf_symbol)
    return df


def fetch_sp500_symbols(timeout: float = 30.0) -> list[str]:
    """Current S&P 500 tickers, normalized for yfinance, sorted and de-duplicated."""
    df = fetch_sp500_table(timeout=timeout)
    return sorted(dict.fromkeys(df["yf_symbol"].tolist()))


def sector_map(timeout: float = 30.0) -> dict[str, str]:
    """{yf_symbol: GICS sector} for current members (used for sector-neutralizing factors).

    Current sectors applied to history — sectors are stable enough that this is a
    minor approximation; dropped names not in the current table fall back to "Unknown".
    """
    df = fetch_sp500_table(timeout=timeout)
    return dict(zip(df["yf_symbol"], df["GICS Sector"], strict=False))
