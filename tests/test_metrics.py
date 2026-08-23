import numpy as np
import pandas as pd

from equity_agent.backtest import metrics


def test_total_return() -> None:
    r = pd.Series([0.1, 0.1])
    assert abs(metrics.total_return(r) - 0.21) < 1e-9


def test_max_drawdown() -> None:
    # equity: 1.1 -> 0.55 -> 0.605; trough at 0.55 from peak 1.1 = -50%.
    r = pd.Series([0.1, -0.5, 0.1])
    assert abs(metrics.max_drawdown(r) - (-0.5)) < 1e-9


def test_sharpe_sign_and_zero_variance() -> None:
    up = pd.Series([0.01, 0.02, 0.005, 0.015, 0.008])
    assert metrics.sharpe_ratio(up) > 0
    flat = pd.Series([0.01, 0.01, 0.01])
    assert np.isnan(metrics.sharpe_ratio(flat))  # zero variance


def test_trade_metrics() -> None:
    pnl = pd.Series([1.0, 1.0, -1.0])
    assert metrics.profit_factor(pnl) == 2.0
    assert abs(metrics.win_rate(pd.Series([1.0, -1.0, 1.0, 0.0])) - 0.5) < 1e-9
    assert metrics.profit_factor(pd.Series([1.0, 2.0])) == float("inf")  # no losses


def test_return_summary_keys() -> None:
    r = pd.Series(np.linspace(-0.01, 0.02, 300))
    s = metrics.return_summary(r)
    for key in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar"):
        assert key in s


def test_capm_alpha_beta_recovers_known_alpha_and_beta() -> None:
    """Regression recovers a planted beta and alpha (noise demeaned so alpha is exact)."""
    import numpy as np
    import pandas as pd

    from equity_agent.backtest.metrics import capm_alpha_beta

    rng = np.random.default_rng(0)
    n = 2000
    mkt = pd.Series(rng.normal(0.0004, 0.01, n))
    noise = pd.Series(rng.normal(0, 0.004, n))
    noise = noise - noise.mean()  # demean so the planted alpha is exact
    planted_alpha_ann = 0.03
    strat = planted_alpha_ann / 252 + 0.6 * mkt + noise

    r = capm_alpha_beta(strat, mkt)
    assert abs(r["beta"] - 0.6) < 0.02
    assert abs(r["ann_alpha"] - planted_alpha_ann) < 0.005
    # POWER LESSON: a REAL 3%/yr alpha with ~6.3%/yr residual vol (IR ~0.48) over
    # ~8 years only reaches t ~1.3 — below the usual 2.0 bar. Detecting modest alpha
    # needs ~(2/IR)^2 years (~17y here). Our 11-year samples can only prove large edges.
    assert 1.0 < r["alpha_t"] < 2.0
    assert 0.5 < r["r2"] < 0.9


def test_capm_zero_alpha_when_strategy_is_pure_beta() -> None:
    import numpy as np
    import pandas as pd

    from equity_agent.backtest.metrics import capm_alpha_beta

    rng = np.random.default_rng(1)
    mkt = pd.Series(rng.normal(0.0004, 0.01, 1500))
    strat = 0.8 * mkt  # pure beta, no alpha, no noise
    r = capm_alpha_beta(strat, mkt)
    assert abs(r["beta"] - 0.8) < 1e-9
    assert abs(r["ann_alpha"]) < 1e-9
