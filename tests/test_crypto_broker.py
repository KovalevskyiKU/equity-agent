from equity_agent.execution.crypto_broker import BinanceBroker, base_asset, to_ccxt_symbol
from equity_agent.execution.orders import plan_orders


def test_symbol_helpers() -> None:
    assert to_ccxt_symbol("BTC-USD") == "BTC/USDT"
    assert base_asset("ETH-USD") == "ETH"


def test_plan_orders_fractional_for_crypto() -> None:
    # $1000 into BTC at $50k -> 0.02 BTC (fractional, not floored to whole units).
    orders = plan_orders({"BTC-USD": 1.0}, {"BTC-USD": 50_000.0}, 1000.0, {}, whole_shares=False)
    assert len(orders) == 1
    assert orders[0].side == "BUY"
    assert abs(orders[0].qty - 0.02) < 1e-9


class _FakeExchange:
    def __init__(self) -> None:
        self.orders: list[tuple] = []

    def fetch_balance(self) -> dict:
        return {"total": {"USDT": 1000.0, "BTC": 0.0}}

    def create_order(self, symbol: str, otype: str, side: str, qty: float) -> dict:
        self.orders.append((symbol, otype, side, qty))
        return {"id": "x"}


def test_rebalance_dry_run_plans_but_does_not_transmit() -> None:
    fake = _FakeExchange()
    broker = BinanceBroker(exchange=fake)
    orders = broker.rebalance({"BTC-USD": 1.0}, {"BTC-USD": 50_000.0}, execute=False)
    assert len(orders) == 1 and orders[0].side == "BUY"
    assert fake.orders == []  # nothing transmitted on a dry run


def test_rebalance_execute_transmits() -> None:
    fake = _FakeExchange()
    broker = BinanceBroker(exchange=fake)
    broker.rebalance({"BTC-USD": 1.0}, {"BTC-USD": 50_000.0}, execute=True)
    assert len(fake.orders) == 1
    assert fake.orders[0][0] == "BTC/USDT" and fake.orders[0][2] == "buy"
