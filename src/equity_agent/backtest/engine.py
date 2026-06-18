"""Event-driven daily backtester (the chosen Phase 3 engine).

The strategy's job is to produce a causal target-weight matrix (date x symbol,
weights in [0, 1]); the engine simulates execution. The one rule that prevents
look-ahead: weights decided on the close of day *t* are executed at the **open
of day t+1**. The engine applies per-side fees and slippage and marks the
portfolio to market at each close.

This is deliberately separate from signal generation — the same engine will
backtest the quant baselines now and the LLM decision engine later.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    fee_bps: float = 1.0  # per-side commission, bps of traded notional
    slippage_bps: float = 5.0  # adverse fill vs the open price, bps


@dataclass
class BacktestResult:
    equity: pd.Series  # portfolio value at each close
    returns: pd.Series  # daily returns
    n_trades: int
    turnover: float  # total traded notional / mean equity


def run_backtest(
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Simulate a daily long/flat (or partial-weight) portfolio with next-open fills."""
    config = config or BacktestConfig()
    dates = close_px.index
    symbols = list(close_px.columns)

    # Weights decided at close of t-1 are the ones we hold from the open of t.
    exec_weights = target_weights.reindex(index=dates, columns=symbols).shift(1)

    cash = config.initial_cash
    shares = pd.Series(0.0, index=symbols)
    fee_rate = config.fee_bps / 1e4
    slip = config.slippage_bps / 1e4

    equity_curve: list[float] = []
    n_trades = 0
    traded_notional = 0.0

    for d in dates:
        o = open_px.loc[d]
        c = close_px.loc[d]
        w = exec_weights.loc[d]

        if w.notna().any():
            equity_open = cash + float((shares * o.fillna(0.0)).sum())
            for s in symbols:
                ws, os_ = w[s], o[s]
                if pd.isna(ws) or pd.isna(os_) or os_ <= 0:
                    continue
                desired = equity_open * float(ws) / float(os_)
                delta = desired - shares[s]
                if abs(delta) < 1e-9:
                    continue
                exec_price = float(os_) * (1.0 + slip * (1.0 if delta > 0 else -1.0))
                notional = delta * exec_price
                cash -= notional + abs(notional) * fee_rate
                shares[s] = desired
                n_trades += 1
                traded_notional += abs(notional)

        equity_curve.append(cash + float((shares * c.fillna(0.0)).sum()))

    equity = pd.Series(equity_curve, index=dates, name="equity")
    returns = equity.pct_change().fillna(0.0)
    mean_eq = float(equity.mean())
    turnover = traded_notional / mean_eq if mean_eq else float("nan")
    return BacktestResult(equity=equity, returns=returns, n_trades=n_trades, turnover=turnover)
