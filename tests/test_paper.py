from equity_agent.execution.paper_broker import get_positions, rebalance, reset_account


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
