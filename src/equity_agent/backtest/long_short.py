"""Dollar-neutral long-short factor backtest — the high-power test of a factor.

A long-only top-quantile portfolio is mostly *market beta plus a small tilt*, so
comparing its Sharpe to the index is a low-power test of the factor itself. The
standard test is the **market-neutral long-short spread**: long the top quantile,
short the bottom, dollar-neutral, held between rebalances, **net of turnover costs**.

Long-only cash accounting doesn't apply here (there is no cash drag and shorts are
negative weights), so returns are computed directly from the weight panel rather
than through the long-only engine: ``r_t = sum(w_{t-1} * ret_t) - cost_t``.
Weights decided at the close of t are held from t+1 (no look-ahead).

Shorting is idealized: no borrow fee, no locate constraint, no margin call. Real
short books cost more, so treat this as an upper bound on the factor's payoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def long_short_weights(
    factor: pd.DataFrame, rebal_dates: pd.Index, q: float = 0.2, min_names: int = 30
) -> pd.DataFrame:
    """Dollar-neutral weights: +1/(2k) on the top-q, -1/(2k) on the bottom-q.

    Gross exposure is 1.0 (0.5 long + 0.5 short), net exposure 0. Rows between
    rebalances are NaN and forward-filled by the caller (position is held).
    """
    w = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    for d in rebal_dates:
        if d not in factor.index:
            continue
        row = factor.loc[d].dropna()
        if len(row) < min_names:
            continue
        k = max(1, int(round(len(row) * q)))
        top, bottom = row.nlargest(k).index, row.nsmallest(k).index
        w.loc[d] = 0.0
        w.loc[d, top] = 0.5 / k
        w.loc[d, bottom] = -0.5 / k
    return w


def long_short_returns(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    *,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> pd.Series:
    """Daily net returns of a held dollar-neutral book, charging turnover costs."""
    held = weights.ffill().fillna(0.0)
    rets = close.pct_change().reindex(columns=held.columns).fillna(0.0)
    gross = (held.shift(1).fillna(0.0) * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    cost = turnover * (fee_bps + slippage_bps) / 1e4
    return (gross - cost).rename("ls_return")


def long_short_turnover(weights: pd.DataFrame) -> float:
    """Total traded notional over the run (per unit of capital)."""
    held = weights.ffill().fillna(0.0)
    return float(held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1)).sum())
