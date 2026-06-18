from datetime import date, timedelta

import pandas as pd

from equity_agent.backtest.engine import BacktestConfig, run_backtest
from equity_agent.backtest.metrics import max_drawdown
from equity_agent.risk.limits import cap_exposure


def test_cap_exposure_caps_per_name_and_gross() -> None:
    w = pd.DataFrame({"A": [0.5], "B": [0.5], "C": [0.5]})  # gross 1.5, names over 0.34
    capped = cap_exposure(w, max_per_name=0.34, max_gross=1.0)
    assert (capped <= 0.34 + 1e-9).to_numpy().all()
    assert abs(float(capped.iloc[0].sum()) - 1.0) < 1e-9  # scaled down to 100% gross


def _crash_panel() -> pd.DataFrame:
    prices = [100.0, 100.0, 95.0, 85.0, 72.0, 60.0, 55.0, 52.0]
    idx = [date(2024, 1, 1) + timedelta(days=i) for i in range(len(prices))]
    return pd.DataFrame({"X": prices}, index=idx)


def test_drawdown_circuit_breaker_limits_drawdown() -> None:
    px = _crash_panel()
    weights = pd.DataFrame({"X": [1.0] * len(px)}, index=px.index)

    no_stop = run_backtest(px, px, weights, BacktestConfig(initial_cash=1000.0))
    with_stop = run_backtest(
        px, px, weights, BacktestConfig(initial_cash=1000.0, max_drawdown_stop=0.15)
    )

    # The kill switch liquidates after the breach, so its drawdown is shallower
    # (less negative) than riding the crash fully invested.
    assert max_drawdown(with_stop.returns) > max_drawdown(no_stop.returns)
