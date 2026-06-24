"""Monitoring summary over the paper-trading equity curve.

Turns the stored ``EquitySnapshot`` series into the numbers you actually watch: net
equity, last-run P&L, drawdown from peak, rolling risk, and **tracking vs SPY**
(the core tracks the cap-weight benchmark, so a large gap is a red flag). Pure and
unit-tested; the CLI/dashboard are thin callers.
"""

from __future__ import annotations

import pandas as pd

from .backtest.metrics import return_summary


def monitor_summary(
    equity_curve: pd.DataFrame, spy_close: pd.Series | None = None
) -> dict[str, float | int]:
    """Monitoring metrics from an equity curve (columns ``ts``, ``equity``).

    Returns at least ``snapshots``/``equity``; richer stats (P&L, drawdown, Sharpe,
    excess vs SPY) appear once there are >= 2 snapshots. Degrades gracefully on
    sparse data rather than raising.
    """
    if equity_curve is None or equity_curve.empty:
        return {"snapshots": 0}

    eq = equity_curve.set_index("ts")["equity"].astype(float).sort_index()
    out: dict[str, float | int] = {"snapshots": int(len(eq)), "equity": float(eq.iloc[-1])}
    if len(eq) < 2:
        return out

    out["total_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    out["last_pnl"] = float(eq.iloc[-1] - eq.iloc[-2])
    rets = eq.pct_change().dropna()
    out["last_pnl_pct"] = float(rets.iloc[-1])
    m = return_summary(rets)
    for k in ("ann_vol", "sharpe", "max_drawdown"):
        out[k] = float(m[k])

    if spy_close is not None and not spy_close.empty:
        spy = spy_close.copy()
        spy.index = pd.to_datetime(spy.index)
        dates = pd.to_datetime(eq.index).normalize()
        spy_on = spy.reindex(dates, method="ffill")
        if spy_on.notna().sum() >= 2 and float(spy_on.iloc[0]) > 0:
            spy_ret = float(spy_on.iloc[-1] / spy_on.iloc[0] - 1.0)
            out["spy_return"] = spy_ret
            out["excess_vs_spy"] = out["total_return"] - spy_ret
    return out
