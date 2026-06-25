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


def realized_vol(
    returns: pd.Series, lookback: int = 20, trading_days: int = TRADING_DAYS
) -> pd.Series:
    """Trailing annualized volatility of a return series (``trading_days`` = 365 for crypto)."""
    return returns.rolling(lookback).std() * np.sqrt(trading_days)


def vol_target_exposure(
    returns: pd.Series,
    *,
    target_vol: float = 0.15,
    lookback: int = 20,
    max_exposure: float = 1.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Latest exposure scalar in ``[0, max_exposure]`` for a vol-target overlay.

    Returns ``max_exposure`` when there isn't enough history to estimate vol (fail
    open to full exposure rather than silently sitting in cash).
    """
    av = realized_vol(returns, lookback, trading_days)
    if len(av) == 0:
        return max_exposure
    last = float(av.iloc[-1])
    if not np.isfinite(last) or last <= 0:
        return max_exposure
    return float(min(max_exposure, target_vol / last))


def vol_target_exposure_series(
    returns: pd.Series,
    *,
    target_vol: float = 0.15,
    lookback: int = 20,
    max_exposure: float = 1.0,
    band: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> pd.Series:
    """Full daily exposure series for a vol-target overlay (for backtesting).

    Warm-up (insufficient vol history) holds full exposure. ``band`` adds a no-trade
    buffer: exposure only moves when it would change by more than ``band``, which cuts
    the daily churn (and trading cost) of a naive vol-target. ``trading_days`` = 365
    for crypto.
    """
    av = realized_vol(returns, lookback, trading_days)
    raw = (target_vol / av).clip(upper=max_exposure)
    raw = raw.where(np.isfinite(raw)).fillna(max_exposure)
    if band <= 0:
        return raw
    out: list[float] = []
    cur = max_exposure
    for v in raw.to_numpy():
        if abs(float(v) - cur) > band:
            cur = float(v)
        out.append(cur)
    return pd.Series(out, index=raw.index)
