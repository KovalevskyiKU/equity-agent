"""Causal target-weight generators for the backtester.

Each returns a (date x symbol) weight matrix using only past/current data, so it
is safe to feed straight into the engine (which then lags execution by a day).
These are baselines/benchmarks and a sanity demo — the real allocator will be
the LLM decision engine (Phase 2), which emits weights into this same format.
"""

from __future__ import annotations

import pandas as pd


def single_asset(close_px: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """100% in one symbol from its first valid bar — the buy-and-hold benchmark."""
    w = pd.DataFrame(0.0, index=close_px.index, columns=close_px.columns)
    valid = close_px[symbol].notna()
    w.loc[valid, symbol] = 1.0
    return w


def buy_and_hold_equal(close_px: pd.DataFrame) -> pd.DataFrame:
    """Equal weight across all symbols (rebalanced daily)."""
    n = close_px.shape[1]
    valid = close_px.notna()
    return valid.astype(float) / n


def momentum_long_flat(close_px: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """DEMO ONLY: equal weight among symbols whose `lookback`-day return is positive.

    A sanity strategy to exercise the engine's turnover/fees — not a recommended
    signal (momentum was noise on the full sample; see PHASE1_FINDINGS).
    """
    up = close_px > close_px.shift(lookback)
    counts = up.sum(axis=1).replace(0, pd.NA)
    weights = up.astype(float).div(counts, axis=0).fillna(0.0)
    return weights
