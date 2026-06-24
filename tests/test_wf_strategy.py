import numpy as np

from equity_agent.research.wf_strategy import _ridge


def test_ridge_recovers_linear_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (500, 3))
    true = np.array([1.0, -2.0, 0.5])
    y = x @ true
    xs = (x - x.mean(0)) / x.std(0)
    yc = y - y.mean()
    coef = _ridge(xs, yc, alpha=1e-6)
    pred = xs @ coef
    # with negligible regularisation, ridge ~ OLS and recovers the (scaled) signal
    assert np.corrcoef(pred, yc)[0, 1] > 0.99
