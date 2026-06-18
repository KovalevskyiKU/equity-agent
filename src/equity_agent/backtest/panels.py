"""Build aligned open/close price panels (date x symbol) from stored bars."""

from __future__ import annotations

import pandas as pd

from ..signals.feature_store import load_bars


def load_price_panels(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (open_px, close_px) date x symbol, aligned on the union calendar.

    Forward-filled per symbol (causal) so a symbol missing a day doesn't drop the
    whole row; leading gaps stay NaN and are treated as "not yet tradable".
    """
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    for s in symbols:
        bars = load_bars(s)
        if bars.empty:
            continue
        opens[s] = bars["open"]
        closes[s] = bars["close"]

    if not closes:
        return pd.DataFrame(), pd.DataFrame()

    open_df = pd.DataFrame(opens).sort_index()
    close_df = pd.DataFrame(closes).sort_index()
    idx = open_df.index.union(close_df.index)
    return open_df.reindex(idx).ffill(), close_df.reindex(idx).ffill()
