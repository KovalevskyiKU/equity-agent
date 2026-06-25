import numpy as np
import pandas as pd

from equity_agent.backtest.overlay_backtest import run_overlay_comparison, vol_target_index_weights


def test_vol_target_index_weights_long_only_capped() -> None:
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
    cb = pd.DataFrame({"SPY": close})
    w = vol_target_index_weights(cb, "SPY", target_vol=0.15, band=0.0)
    assert list(w.columns) == ["SPY"]
    assert (w["SPY"] >= 0).all() and (w["SPY"] <= 1.0 + 1e-9).all()


def test_run_overlay_comparison_no_data(temp_db: None) -> None:
    assert run_overlay_comparison().empty
