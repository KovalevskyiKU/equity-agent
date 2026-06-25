import numpy as np
import pandas as pd

from equity_agent.risk.overlay import vol_target_exposure, vol_target_exposure_series


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


def test_exposure_series_warmup_and_reduction() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0, 0.04, 80))  # high vol
    s = vol_target_exposure_series(rets, target_vol=0.15)
    assert len(s) == 80
    assert (s.iloc[:19] == 1.0).all()  # warm-up (rolling-20 NaN) holds full exposure
    assert (s.iloc[19:] < 1.0).any()  # once vol is estimable, high vol reduces exposure


def test_exposure_band_reduces_changes() -> None:
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0, 0.02, 200))
    no_band = vol_target_exposure_series(rets, band=0.0)
    banded = vol_target_exposure_series(rets, band=0.1)
    assert banded.nunique() < no_band.nunique()  # fewer distinct levels -> less churn
