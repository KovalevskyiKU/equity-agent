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


def vol_target_weights(
    close_px: pd.DataFrame,
    *,
    lookback: int = 20,
    target_vol: float = 0.20,
    max_weight: float = 0.34,
) -> pd.DataFrame:
    """Per-name volatility targeting (mechanical risk management baseline).

    weight = target_vol / trailing annualised vol, capped at max_weight. Higher
    recent volatility -> lower weight. Causal. This is the no-LLM yardstick the
    LLM agent must beat to justify itself, given the agent's defensive profile.
    """
    rets = close_px.pct_change()
    ann_vol = rets.rolling(lookback).std() * (252.0**0.5)
    weights = (target_vol / ann_vol).clip(upper=max_weight).fillna(0.0)
    # Cap total gross exposure at 100% (long-only, no leverage): scale a row down
    # only when its weights sum above 1. Without this, sizing each of N names
    # independently can sum to N*max_weight (e.g. 8*0.34 = 2.7x leverage).
    gross = weights.sum(axis=1)
    scale = (1.0 / gross.replace(0.0, 1.0)).clip(upper=1.0)
    return weights.mul(scale, axis=0)


def momentum_long_flat(close_px: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """DEMO ONLY: equal weight among symbols whose `lookback`-day return is positive.

    A sanity strategy to exercise the engine's turnover/fees — not a recommended
    signal (momentum was noise on the full sample; see PHASE1_FINDINGS).
    """
    up = close_px > close_px.shift(lookback)
    counts = up.sum(axis=1).replace(0, pd.NA)
    weights = up.astype(float).div(counts, axis=0).fillna(0.0)
    return weights
