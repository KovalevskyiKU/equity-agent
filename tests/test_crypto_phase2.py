import numpy as np
import pandas as pd

from equity_agent.backtest.crypto import (
    trend_long_short_weights,
    walkforward_trend_weights,
)
from equity_agent.data.funding import carry_summary


def _btc_like() -> pd.DataFrame:
    # 3 years of daily data with a clear up-then-down path.
    idx = pd.date_range("2020-01-01", periods=1095, freq="D")
    up = np.linspace(100, 1000, 700)
    down = np.linspace(1000, 300, 395)
    return pd.DataFrame({"BTC-USD": np.concatenate([up, down])}, index=idx)


def test_long_short_weights_are_plus_minus_one() -> None:
    w = trend_long_short_weights(_btc_like(), "BTC-USD", fast=20, slow=100)["BTC-USD"]
    assert set(w.dropna().unique()) <= {1.0, -1.0}
    assert w.iloc[600] == 1.0  # uptrend -> long
    assert w.iloc[-1] == -1.0  # downtrend -> short


def test_walkforward_trend_no_lookahead_and_flat_warmup() -> None:
    w = walkforward_trend_weights(_btc_like(), "BTC-USD", min_train_days=365)["BTC-USD"]
    # First year has no prior training -> stays flat (cash).
    assert (w.iloc[:365] == 0.0).all()
    # Later it trades long/flat (0/1 only — this is the long-flat WF variant).
    assert set(w.unique()) <= {0.0, 1.0}
    assert (w.iloc[365:] == 1.0).any()


def test_carry_summary_positive_funding() -> None:
    idx = pd.date_range("2021-01-01", periods=900, freq="8h")
    funding = pd.DataFrame(
        {"funding_rate": [0.0001] * 900, "mark_price": [50000.0] * 900}, index=idx
    )
    s = carry_summary(funding, cost_bps_per_year=200.0)
    # 0.0001 * 3 * 365 = 0.1095 gross; net = gross - 0.02
    assert abs(s["ann_carry_gross"] - 0.1095) < 1e-6
    assert abs(s["ann_carry_net"] - (0.1095 - 0.02)) < 1e-6
    assert s["pct_positive"] == 1.0


def test_carry_summary_empty() -> None:
    s = carry_summary(pd.DataFrame())
    assert s["n"] == 0
