import numpy as np
import pandas as pd

from equity_agent.backtest import metrics


def test_total_return() -> None:
    r = pd.Series([0.1, 0.1])
    assert abs(metrics.total_return(r) - 0.21) < 1e-9


def test_max_drawdown() -> None:
    # equity: 1.1 -> 0.55 -> 0.605; trough at 0.55 from peak 1.1 = -50%.
    r = pd.Series([0.1, -0.5, 0.1])
    assert abs(metrics.max_drawdown(r) - (-0.5)) < 1e-9


def test_sharpe_sign_and_zero_variance() -> None:
    up = pd.Series([0.01, 0.02, 0.005, 0.015, 0.008])
    assert metrics.sharpe_ratio(up) > 0
    flat = pd.Series([0.01, 0.01, 0.01])
    assert np.isnan(metrics.sharpe_ratio(flat))  # zero variance


def test_trade_metrics() -> None:
    pnl = pd.Series([1.0, 1.0, -1.0])
    assert metrics.profit_factor(pnl) == 2.0
    assert abs(metrics.win_rate(pd.Series([1.0, -1.0, 1.0, 0.0])) - 0.5) < 1e-9
    assert metrics.profit_factor(pd.Series([1.0, 2.0])) == float("inf")  # no losses


def test_return_summary_keys() -> None:
    r = pd.Series(np.linspace(-0.01, 0.02, 300))
    s = metrics.return_summary(r)
    for key in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar"):
        assert key in s
