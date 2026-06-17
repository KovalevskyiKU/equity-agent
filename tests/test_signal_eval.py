import numpy as np
import pandas as pd

from equity_agent.research.signal_eval import forward_return, information_coefficient


def test_forward_return() -> None:
    close = pd.Series([1.0, 2.0, 4.0, 8.0])
    fwd = forward_return(close, 1)
    assert fwd.iloc[0] == 1.0  # 2/1 - 1
    assert fwd.iloc[1] == 1.0  # 4/2 - 1
    assert np.isnan(fwd.iloc[-1])  # no bar after the last


def test_ic_detects_signal_and_noise() -> None:
    rng = np.random.default_rng(0)
    n = 500
    target = pd.Series(rng.normal(0.0, 1.0, n))

    # Feature strongly aligned with the target -> high positive IC.
    informative = target + pd.Series(rng.normal(0.0, 0.3, n))
    r_info, t_info, n_info = information_coefficient(informative, target)
    assert n_info == n
    assert r_info > 0.7
    assert abs(t_info) > 5

    # Independent noise -> IC near zero.
    noise = pd.Series(rng.normal(0.0, 1.0, n))
    r_noise, _, _ = information_coefficient(noise, target)
    assert abs(r_noise) < 0.2


def test_ic_too_few_observations() -> None:
    feat = pd.Series([1.0, 2.0, 3.0])
    target = pd.Series([1.0, 2.0, 3.0])
    r, t, n = information_coefficient(feat, target)
    assert n == 3
    assert np.isnan(r)
