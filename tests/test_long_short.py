import numpy as np
import pandas as pd

from equity_agent.backtest.long_short import (
    long_short_returns,
    long_short_turnover,
    long_short_weights,
)


def _panel(n_days: int = 60, n_names: int = 10) -> tuple[pd.DatetimeIndex, list[str]]:
    return pd.date_range("2020-01-01", periods=n_days, freq="B"), [
        f"S{i:02d}" for i in range(n_names)
    ]


def test_weights_are_dollar_neutral_with_unit_gross() -> None:
    dates, syms = _panel()
    factor = pd.DataFrame(
        np.tile(np.arange(10, dtype=float), (len(dates), 1)), index=dates, columns=syms
    )
    w = long_short_weights(factor, pd.Index([dates[5]]), q=0.2, min_names=5)
    row = w.loc[dates[5]]
    assert abs(row.sum()) < 1e-12  # net zero
    assert abs(row.abs().sum() - 1.0) < 1e-12  # gross one
    assert (row[["S08", "S09"]] > 0).all()  # top quintile long
    assert (row[["S00", "S01"]] < 0).all()  # bottom quintile short


def test_returns_capture_the_spread_and_charge_costs() -> None:
    dates, syms = _panel(40, 10)
    factor = pd.DataFrame(
        np.tile(np.arange(10, dtype=float), (len(dates), 1)), index=dates, columns=syms
    )
    # High-factor names drift up, low-factor names drift down -> positive spread.
    drift = np.linspace(-0.002, 0.002, 10)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(np.tile(drift, (len(dates), 1)), axis=0)),
        index=dates, columns=syms,
    )
    w = long_short_weights(factor, pd.Index([dates[1]]), q=0.2, min_names=5)
    gross_free = long_short_returns(w, close, fee_bps=0, slippage_bps=0)
    with_costs = long_short_returns(w, close, fee_bps=10, slippage_bps=20)
    assert gross_free.sum() > 0  # long winners / short losers pays
    assert with_costs.sum() < gross_free.sum()  # costs reduce it


def test_market_move_cancels_out() -> None:
    """A pure market move (all names move together) nets to ~0 for a neutral book."""
    dates, syms = _panel(30, 10)
    factor = pd.DataFrame(
        np.tile(np.arange(10, dtype=float), (len(dates), 1)), index=dates, columns=syms
    )
    close = pd.DataFrame(
        np.tile(100 * np.exp(np.cumsum(np.full(len(dates), 0.01))), (10, 1)).T,
        index=dates, columns=syms,
    )
    r = long_short_returns(
        long_short_weights(factor, pd.Index([dates[1]]), q=0.2, min_names=5),
        close, fee_bps=0, slippage_bps=0,
    )
    assert abs(r.iloc[3:].sum()) < 1e-9


def test_turnover_is_positive() -> None:
    dates, syms = _panel(30, 10)
    factor = pd.DataFrame(
        np.tile(np.arange(10, dtype=float), (len(dates), 1)), index=dates, columns=syms
    )
    w = long_short_weights(factor, pd.Index([dates[1], dates[15]]), q=0.2, min_names=5)
    assert long_short_turnover(w) > 0
