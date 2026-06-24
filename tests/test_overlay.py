import numpy as np
import pandas as pd

from equity_agent.risk.overlay import vol_target_exposure


def test_low_vol_gives_full_exposure() -> None:
    calm = pd.Series([0.0005] * 40)  # ~zero realized vol
    assert vol_target_exposure(calm, target_vol=0.15) == 1.0


def test_high_vol_reduces_exposure() -> None:
    rng = np.random.default_rng(0)
    turbulent = pd.Series(rng.normal(0.0, 0.04, 60))  # ~60%+ annualized vol
    e = vol_target_exposure(turbulent, target_vol=0.15)
    assert 0.0 < e < 1.0


def test_insufficient_history_fails_open() -> None:
    # Fewer points than the lookback -> can't estimate vol -> full exposure, not cash.
    assert vol_target_exposure(pd.Series([0.01, -0.01]), lookback=20) == 1.0


def test_exposure_is_capped_at_max() -> None:
    calm = pd.Series([0.001] * 40)
    assert vol_target_exposure(calm, target_vol=0.15, max_exposure=0.8) == 0.8
