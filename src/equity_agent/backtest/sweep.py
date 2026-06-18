"""Rolling-window comparison of mechanical (no-LLM) strategies — quota-free.

For each rolling window over full history, backtest vol-target, the equal-weight
basket and SPY, collect Sharpe / max-drawdown / return, then aggregate. This
gives a robust, regime-spanning read on the risk-managed quant baseline (the
yardstick the LLM agent must beat) without spending any LLM quota.
"""

from __future__ import annotations

import pandas as pd

from .engine import BacktestConfig, run_backtest
from .metrics import return_summary
from .panels import load_price_panels
from .strategy import buy_and_hold_equal, single_asset, vol_target_weights

_TRADING_DAYS_PER_MONTH = 21


def _windows(n: int, window: int, step: int) -> list[tuple[int, int]]:
    """Rolling [start, end) index windows of length `window`, stepped by `step`."""
    out: list[tuple[int, int]] = []
    start = 0
    while start + window <= n:
        out.append((start, start + window))
        start += step
    return out


def run_sweep(
    universe: list[str],
    benchmark: str,
    *,
    window_months: int = 6,
    step_months: int = 2,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_window metrics, aggregate-by-strategy)."""
    open_u, close_u = load_price_panels(universe)
    open_b, close_b = load_price_panels([benchmark])
    if close_u.empty:
        raise RuntimeError("No price data; run `eqa ingest` first.")

    cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)
    dates = close_u.index
    window = window_months * _TRADING_DAYS_PER_MONTH
    step = step_months * _TRADING_DAYS_PER_MONTH

    rows: list[dict[str, object]] = []
    for a, b in _windows(len(dates), window, step):
        cal = dates[a:b]
        bcal = cal[cal.isin(close_b.index)]
        runs = {
            "voltgt": run_backtest(
                open_u.loc[cal], close_u.loc[cal], vol_target_weights(close_u.loc[cal]), cfg
            ),
            "basket": run_backtest(
                open_u.loc[cal], close_u.loc[cal], buy_and_hold_equal(close_u.loc[cal]), cfg
            ),
            "spy": run_backtest(
                open_b.loc[bcal], close_b.loc[bcal], single_asset(close_b.loc[bcal], benchmark), cfg
            ),
        }
        for name, res in runs.items():
            m = return_summary(res.returns)
            rows.append(
                {
                    "window_end": str(cal[-1]),
                    "strategy": name,
                    "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"],
                    "total_return": m["total_return"],
                }
            )

    per_window = pd.DataFrame(rows)
    agg = (
        per_window.groupby("strategy")
        .agg(
            median_sharpe=("sharpe", "median"),
            median_maxdd=("max_drawdown", "median"),
            median_return=("total_return", "median"),
            n_windows=("sharpe", "count"),
        )
        .reset_index()
    )
    return per_window, agg
