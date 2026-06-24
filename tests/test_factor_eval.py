import numpy as np
import pandas as pd

from equity_agent.backtest.factor_portfolio import (
    equal_weight_monthly,
    long_short_spread,
    quantile_long_weights,
    spread_summary,
)
from equity_agent.research.factor_eval import (
    cross_sectional_ic,
    forward_return_panel,
    low_vol_factor,
    momentum_factor,
    month_end_dates,
    summarize_ic,
)


def _panel(periods: int = 60, n: int = 40) -> tuple[pd.DatetimeIndex, list[str]]:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    syms = [f"S{i:02d}" for i in range(n)]
    return dates, syms


def test_forward_return_panel() -> None:
    dates, syms = _panel(5, 2)
    close = pd.DataFrame(
        {"A": [1.0, 2.0, 4.0, 8.0, 16.0], "B": [10, 10, 10, 10, 10]},
        index=dates[:5],
    )
    fwd = forward_return_panel(close, 1)
    assert fwd["A"].iloc[0] == 1.0  # 2/1 - 1
    assert fwd["B"].iloc[0] == 0.0
    assert np.isnan(fwd["A"].iloc[-1])  # no bar after the last


def test_momentum_and_lowvol_are_causal_with_warmup() -> None:
    dates, syms = _panel(300, 3)
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), 3)), axis=0)),
        index=dates, columns=syms,
    )
    mom = momentum_factor(close, lookback=252, skip=21)
    # First `lookback` rows have no full window -> NaN (no look-ahead, needs warm-up).
    assert mom.iloc[:252].isna().all().all()
    assert mom.iloc[252:].notna().any().any()

    lv = low_vol_factor(close, lookback=126)
    assert lv.iloc[:126].isna().all().all()
    # Low-vol factor is the negative of realised vol -> non-positive where defined.
    assert (lv.dropna() <= 0).all().all()


def test_cross_sectional_ic_detects_signal_and_noise() -> None:
    dates, syms = _panel(40, 40)
    rng = np.random.default_rng(0)
    ranks = np.tile(np.arange(40, dtype=float), (len(dates), 1))
    factor = pd.DataFrame(ranks, index=dates, columns=syms)

    # Forward return monotonically tracks the factor cross-sectionally -> IC ~ +1.
    fwd_signal = factor + rng.normal(0, 0.1, factor.shape)
    tab = cross_sectional_ic(factor, fwd_signal, dates, min_names=10)
    assert len(tab) == len(dates)
    assert (tab["n_names"] == 40).all()
    s = summarize_ic(tab)
    assert s.mean_ic > 0.9
    assert s.hit_rate == 1.0

    # Independent noise -> mean IC near zero.
    fwd_noise = pd.DataFrame(rng.normal(0, 1, factor.shape), index=dates, columns=syms)
    s_noise = summarize_ic(cross_sectional_ic(factor, fwd_noise, dates, min_names=10))
    assert abs(s_noise.mean_ic) < 0.2


def test_cross_sectional_ic_respects_min_names() -> None:
    dates, syms = _panel(10, 5)
    factor = pd.DataFrame(1.0, index=dates, columns=syms)
    fwd = pd.DataFrame(1.0, index=dates, columns=syms)
    # Only 5 names but min_names=30 -> nothing qualifies.
    assert cross_sectional_ic(factor, fwd, dates, min_names=30).empty


def test_month_end_dates_picks_last_trading_day() -> None:
    idx = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28"])
    me = month_end_dates(idx)
    assert list(pd.to_datetime(me)) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]


def test_quantile_long_weights_selects_top_and_normalizes() -> None:
    dates, syms = _panel(5, 10)
    factor = pd.DataFrame(
        np.tile(np.arange(10, dtype=float), (len(dates), 1)), index=dates, columns=syms
    )
    rebal = pd.Index([dates[2]])
    w = quantile_long_weights(factor, rebal, q=0.2, min_names=5)
    row = w.loc[dates[2]]
    assert row.sum() == 1.0
    # Top 20% of 10 names = 2 names, the two highest-factor (S08, S09).
    held = set(row[row > 0].index)
    assert held == {"S08", "S09"}
    # Non-rebalance dates stay NaN (engine holds, no trade).
    assert w.loc[dates[0]].isna().all()


def test_equal_weight_monthly_uniform_over_rankable() -> None:
    dates, syms = _panel(5, 10)
    factor = pd.DataFrame(1.0, index=dates, columns=syms)
    rebal = pd.Index([dates[1]])
    w = equal_weight_monthly(factor, rebal, min_names=5)
    row = w.loc[dates[1]]
    assert np.isclose(row.sum(), 1.0)
    assert np.allclose(row.values, 0.1)


def test_long_short_spread_sign() -> None:
    dates, syms = _panel(45, 10)
    # Build a factor that genuinely predicts: high-factor names rise, low-factor fall.
    factor = pd.DataFrame(
        np.tile(np.arange(10, dtype=float), (len(dates), 1)), index=dates, columns=syms
    )
    # close grows fastest for high-index names.
    drift = np.linspace(-0.001, 0.001, 10)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(np.tile(drift, (len(dates), 1)), axis=0)),
        index=dates, columns=syms,
    )
    rebal = month_end_dates(dates)
    ls = long_short_spread(factor, close, rebal, q=0.2, min_names=5)
    assert not ls.empty
    summ = spread_summary(ls)
    assert summ["ann_return"] > 0  # long winners, short losers -> positive
