from datetime import date, timedelta

import pandas as pd

from equity_agent.backtest.engine import BacktestConfig, run_backtest


def _panel(prices: list[float]) -> pd.DataFrame:
    idx = [date(2024, 1, 1) + timedelta(days=i) for i in range(len(prices))]
    return pd.DataFrame({"X": prices}, index=idx)


def test_no_lookahead_entry_is_delayed_one_day() -> None:
    # open == close; weight 1.0 every day, but execution can't happen on day 0.
    px = _panel([100.0, 110.0, 121.0])
    weights = pd.DataFrame({"X": [1.0, 1.0, 1.0]}, index=px.index)
    cfg = BacktestConfig(initial_cash=1000.0, fee_bps=0, slippage_bps=0)
    res = run_backtest(px, px, weights, cfg)

    # Day 0: no position yet -> equity unchanged.
    assert abs(res.equity.iloc[0] - 1000.0) < 1e-6
    # Entered at day-1 open (110), held to day-2 close (121): 1000 * 121/110 = 1100.
    assert abs(res.equity.iloc[-1] - 1100.0) < 1e-6
    assert res.n_trades >= 1


def test_fees_reduce_equity() -> None:
    px = _panel([100.0, 110.0, 121.0])
    weights = pd.DataFrame({"X": [1.0, 1.0, 1.0]}, index=px.index)
    free_cfg = BacktestConfig(initial_cash=1000.0, fee_bps=0, slippage_bps=0)
    costed_cfg = BacktestConfig(initial_cash=1000.0, fee_bps=50, slippage_bps=50)
    free = run_backtest(px, px, weights, free_cfg)
    costed = run_backtest(px, px, weights, costed_cfg)
    assert costed.equity.iloc[-1] < free.equity.iloc[-1]


def test_flat_strategy_holds_cash() -> None:
    px = _panel([100.0, 110.0, 121.0])
    weights = pd.DataFrame({"X": [0.0, 0.0, 0.0]}, index=px.index)
    res = run_backtest(px, px, weights, BacktestConfig(initial_cash=1000.0))
    assert (res.equity == 1000.0).all()
    assert res.n_trades == 0
