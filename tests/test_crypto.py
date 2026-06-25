import numpy as np
import pandas as pd

from equity_agent.backtest.crypto import run_crypto_comparison, trend_weights


def test_trend_weights_long_in_uptrend_flat_in_downtrend() -> None:
    # Rising then falling price -> long (1) while fast>slow, flat (0) after it rolls over.
    up = np.linspace(100, 200, 120)
    down = np.linspace(200, 100, 120)
    px = pd.Series(np.concatenate([up, down]))
    close = pd.DataFrame({"BTC-USD": px})
    w = trend_weights(close, "BTC-USD", fast=10, slow=50)["BTC-USD"]
    assert set(w.dropna().unique()) <= {0.0, 1.0}
    assert w.iloc[110] == 1.0  # established uptrend -> long
    assert w.iloc[-1] == 0.0  # established downtrend -> flat


def test_run_crypto_comparison_no_data(temp_db: None) -> None:
    assert run_crypto_comparison().empty
