from equity_agent.execution.ibkr_broker import PlannedOrder, plan_orders


def test_plan_orders_from_flat_book() -> None:
    # $10k equity, target 100% SPY at $400 -> buy 25 shares (whole shares).
    orders = plan_orders({"SPY": 1.0}, {"SPY": 400.0}, 10_000.0, {})
    assert orders == [PlannedOrder("SPY", "BUY", 25.0, 400.0, 10_000.0)]


def test_plan_orders_diffs_current_positions() -> None:
    # Hold 10 SPY, want ~25 -> buy 15; hold 5 AAPL, want 0 -> sell 5.
    orders = plan_orders(
        {"SPY": 1.0}, {"SPY": 400.0, "AAPL": 100.0}, 10_000.0, {"SPY": 10.0, "AAPL": 5.0}
    )
    by_sym = {o.symbol: o for o in orders}
    assert by_sym["SPY"].side == "BUY" and by_sym["SPY"].qty == 15.0
    assert by_sym["AAPL"].side == "SELL" and by_sym["AAPL"].qty == 5.0


def test_plan_orders_skips_dust_and_bad_prices() -> None:
    # Already at target -> no order; missing/zero price -> skipped.
    assert plan_orders({"SPY": 1.0}, {"SPY": 400.0}, 10_000.0, {"SPY": 25.0}) == []
    assert plan_orders({"X": 1.0}, {"X": 0.0}, 10_000.0, {}) == []
    assert plan_orders({"Y": 1.0}, {}, 10_000.0, {}) == []


def test_plan_orders_whole_shares_floor() -> None:
    # $1000 / $399 = 2.5 -> floored to 2 shares.
    orders = plan_orders({"Z": 1.0}, {"Z": 399.0}, 1000.0, {})
    assert orders[0].qty == 2.0


def test_plan_orders_min_notional() -> None:
    # A 1-share, $5 delta is below a $50 min_notional -> skipped.
    orders = plan_orders({"W": 1.0}, {"W": 5.0}, 55.0, {"W": 10.0}, min_notional=50.0)
    assert orders == []
