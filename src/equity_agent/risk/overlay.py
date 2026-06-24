"""Volatility-targeting risk overlay — scale market exposure by recent vol.

A principled, opt-in risk control for the core: when realized market volatility is
high, hold less of the risk asset (the rest in cash); when it's calm, hold up to
full exposure. ``exposure = clip(target_vol / realized_vol, 0, max_exposure)``.

This is the overlay the research supported (shallower drawdowns in crash windows at
a small return cost). It is deliberately *continuous* — unlike a hard drawdown
kill-switch, which the research showed hurts by liquidating into V-shaped
recoveries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realized_vol(returns: pd.Series, lookback: int = 20) -> pd.Series:
    """Trailing annualized volatility of a daily-return series."""
    return returns.rolling(lookback).std() * np.sqrt(TRADING_DAYS)


def vol_target_exposure(
    returns: pd.Series,
    *,
    target_vol: float = 0.15,
    lookback: int = 20,
    max_exposure: float = 1.0,
) -> float:
    """Latest exposure scalar in ``[0, max_exposure]`` for a vol-target overlay.

    Returns ``max_exposure`` when there isn't enough history to estimate vol (fail
    open to full exposure rather than silently sitting in cash).
    """
    av = realized_vol(returns, lookback)
    if len(av) == 0:
        return max_exposure
    last = float(av.iloc[-1])
    if not np.isfinite(last) or last <= 0:
        return max_exposure
    return float(min(max_exposure, target_vol / last))
