import numpy as np
import pandas as pd
import pytest

from equity_agent.execution.paper_broker import (
    cancel_order,
    check_pending_fills,
    get_open_orders,
    get_positions,
    place_limit_order,
    place_order,
    rebalance,
    reset_account,
)
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


def test_place_order_buy_then_sell(temp_db: None) -> None:
    reset_account(1000.0)
    r = place_order("AAA", "BUY", 5.0, 10.0, fee_bps=0, slippage_bps=0)
    assert r["filled"] == "ok"
    assert abs(get_positions()["AAA"] - 5.0) < 1e-9
    assert abs(float(r["cash"]) - 950.0) < 1e-6  # 1000 - 5*10

    s = place_order("AAA", "SELL", 5.0, 12.0, fee_bps=0, slippage_bps=0)
    assert "AAA" not in get_positions()
    assert abs(float(s["cash"]) - 1010.0) < 1e-6  # 950 + 5*12


def test_place_order_rejects_oversell(temp_db: None) -> None:
    reset_account(1000.0)
    place_order("AAA", "BUY", 2.0, 10.0, fee_bps=0, slippage_bps=0)
    with pytest.raises(ValueError):
        place_order("AAA", "SELL", 5.0, 10.0)  # no shorting on paper


def test_limit_order_fills_on_cross(temp_db: None) -> None:
    reset_account(10000.0)
    r = place_limit_order("AAA", "BUY", 2.0, 100.0)
    assert r["status"] == "open"
    assert len(get_open_orders()) == 1
    assert check_pending_fills({"AAA": 105.0}) == 0  # above limit -> no fill
    assert len(get_open_orders()) == 1
    assert check_pending_fills({"AAA": 98.0}) == 1  # at/below limit -> fill
    assert get_open_orders() == []
    assert abs(get_positions()["AAA"] - 2.0) < 1e-9


def test_cancel_open_order(temp_db: None) -> None:
    reset_account(10000.0)
    r = place_limit_order("BBB", "BUY", 1.0, 50.0)
    cancel_order(int(r["order_id"]))  # type: ignore[arg-type]
    assert get_open_orders() == []
    with pytest.raises(ValueError):
        cancel_order(int(r["order_id"]))  # type: ignore[arg-type]  # already cancelled
