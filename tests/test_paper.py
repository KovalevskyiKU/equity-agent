import numpy as np
import pandas as pd

from equity_agent.execution.paper_broker import get_positions, rebalance, reset_account
from equity_agent.execution.runner import compute_core_target


def test_paper_rebalance_and_reweight(temp_db: None) -> None:
    reset_account(1000.0)

    # Equal-weight 50/50 at prices 10 / 20 -> 50 shares A, 25 shares B, cash ~0.
    r1 = rebalance(
        {"A": 0.5, "B": 0.5}, {"A": 10.0, "B": 20.0},
        fee_bps=0, slippage_bps=0, starting_cash=1000.0,
    )
    pos = get_positions()
    assert abs(pos["A"] - 50.0) < 1e-6
    assert abs(pos["B"] - 25.0) < 1e-6
    assert abs(r1["equity"] - 1000.0) < 1e-6

    # Reweight to all-A: B sold to 0, A scaled up; equity unchanged at same prices.
    r2 = rebalance({"A": 1.0}, {"A": 10.0, "B": 20.0}, fee_bps=0, slippage_bps=0)
    pos2 = get_positions()
    assert "B" not in pos2
    assert abs(pos2["A"] - 100.0) < 1e-6
    assert abs(r2["equity"] - 1000.0) < 1e-6


def test_fees_reduce_paper_equity(temp_db: None) -> None:
    reset_account(1000.0)
    r = rebalance({"A": 1.0}, {"A": 10.0}, fee_bps=50, slippage_bps=50, starting_cash=1000.0)
    assert r["equity"] < 1000.0  # costs eat into equity


def test_core_target_spy_holds_benchmark() -> None:
    close_b = pd.DataFrame({"SPY": [400.0, 410.0]})
    target, prices = compute_core_target("spy", pd.DataFrame(), close_b, "SPY")
    assert target == {"SPY": 1.0}
    assert prices == {"SPY": 410.0}


def test_core_target_equal_weight_over_universe() -> None:
    close_u = pd.DataFrame({"A": [10.0, 11.0], "B": [20.0, 22.0]})
    target, prices = compute_core_target("equal_weight", close_u, pd.DataFrame(), "SPY")
    assert set(target) == {"A", "B"}
    assert abs(sum(target.values()) - 1.0) < 1e-9
    assert abs(target["A"] - 0.5) < 1e-9
    assert prices == {"A": 11.0, "B": 22.0}


def test_core_target_vol_target_is_long_only_and_capped() -> None:
    rng = np.random.default_rng(0)
    close_u = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 3)), axis=0)), columns=["A", "B", "C"]
    )
    target, _ = compute_core_target("vol_target", close_u, pd.DataFrame(), "SPY")
    assert all(w > 0 for w in target.values())
    assert sum(target.values()) <= 1.0 + 1e-9  # gross capped at 100%


def test_core_target_exposure_scales_toward_cash() -> None:
    close_b = pd.DataFrame({"SPY": [400.0, 410.0]})
    target, _ = compute_core_target("spy", pd.DataFrame(), close_b, "SPY", exposure=0.5)
    assert target == {"SPY": 0.5}  # half invested, half cash
