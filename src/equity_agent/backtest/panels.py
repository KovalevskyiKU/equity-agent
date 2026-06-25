"""Build aligned open/close price panels (date x symbol) from stored bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..signals.feature_store import load_bars


def load_price_panels(
    symbols: list[str], total_return: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (open_px, close_px) date x symbol, aligned on the union calendar.

    Forward-filled per symbol (causal) so a symbol missing a day doesn't drop the
    whole row; leading gaps stay NaN and are treated as "not yet tradable".

    ``total_return=True`` returns **dividend-adjusted** prices: both open and close
    are scaled by ``adj_close / close`` (yfinance's cumulative adjustment factor), so
    close-to-close returns include reinvested dividends. Default is raw (price-return)
    to preserve existing behaviour; the honest benchmark comparison uses total return.
    """
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    for s in symbols:
        bars = load_bars(s)
        if bars.empty:
            continue
        o, c = bars["open"], bars["close"]
        if total_return and "adj_close" in bars:
            factor = (bars["adj_close"] / c).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
            o, c = o * factor, c * factor
        opens[s] = o
        closes[s] = c

    if not closes:
        return pd.DataFrame(), pd.DataFrame()

    open_df = pd.DataFrame(opens).sort_index()
    close_df = pd.DataFrame(closes).sort_index()
    idx = open_df.index.union(close_df.index)
    return open_df.reindex(idx).ffill(), close_df.reindex(idx).ffill()
