from datetime import date, timedelta

import numpy as np
import pandas as pd

from equity_agent.backtest.strategy import vol_target_weights


def test_vol_target_lower_weight_for_higher_vol() -> None:
    n = 60
    idx = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(0)
    low_vol = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))
    high_vol = 100 * np.cumprod(1 + rng.normal(0, 0.03, n))
    df = pd.DataFrame({"LOW": low_vol, "HIGH": high_vol}, index=idx)

    w = vol_target_weights(df, lookback=20, max_weight=0.5)
    # The calmer name gets the larger target weight; nothing exceeds the cap.
    assert w["LOW"].iloc[-1] > w["HIGH"].iloc[-1]
    assert (w <= 0.5 + 1e-9).to_numpy().all()
