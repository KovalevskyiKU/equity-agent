"""IBKR live/paper broker adapter (Phase 4) — same rebalance intent as paper_broker.

Split into a **pure, tested order planner** (`plan_orders`) and a thin live glue
class (`IBKRBroker`) that talks to TWS/Gateway via ``ib_insync`` (an optional
``[ibkr]`` dependency, imported lazily). The planner is what we can unit-test
without a gateway; the glue is deliberately minimal.

SAFETY: nothing here transmits an order unless the caller passes ``execute=True``.
The CLI defaults to a dry run (plan + print, no orders). Live trading is a
deliberate, user-initiated action against the user's own TWS/Gateway.
"""

from __future__ import annotations

import logging
from typing import Any

from .orders import PlannedOrder, plan_orders  # re-exported for back-compat

__all__ = ["IBKRBroker", "PlannedOrder", "plan_orders"]

logger = logging.getLogger("equity_agent")


class IBKRBroker:
    """Thin live adapter over ``ib_insync``. Glue only — order math is in plan_orders.

    Requires the ``[ibkr]`` extra (``pip install -e ".[ibkr]"``) and a running
    TWS/IB Gateway (paper port 7497, live 7496; Gateway 4002/4001).
    """

    def __init__(self, host: str, port: int, client_id: int, ib: Any = None) -> None:
        self.host, self.port, self.client_id = host, port, client_id
        self._ib: Any = ib  # injectable for tests; created on connect() otherwise

    def connect(self) -> None:
        if self._ib is None:
            from ib_insync import IB  # lazy: optional dependency

            self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id)

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()

    def net_liquidation(self) -> float:
        """Account net liquidation value (total equity) from the account summary."""
        for row in self._ib.accountSummary():
            if row.tag == "NetLiquidation":
                return float(row.value)
        raise RuntimeError("NetLiquidation not found in account summary")

    def positions(self) -> dict[str, float]:
        """Current positions as {symbol: qty} (stocks only)."""
        out: dict[str, float] = {}
        for p in self._ib.positions():
            out[p.contract.symbol] = out.get(p.contract.symbol, 0.0) + float(p.position)
        return out

    def rebalance(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        *,
        execute: bool = False,
        min_notional: float = 1.0,
    ) -> list[PlannedOrder]:
        """Plan orders to reach ``target_weights``; transmit them only if ``execute``."""
        equity = self.net_liquidation()
        orders = plan_orders(
            target_weights, prices, equity, self.positions(), min_notional=min_notional
        )
        if not execute:
            logger.info("IBKR dry-run: %d orders planned (not transmitted)", len(orders))
            return orders

        from ib_insync import MarketOrder, Stock  # lazy

        for o in orders:
            contract = Stock(o.symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            self._ib.placeOrder(contract, MarketOrder(o.side, o.qty))
            logger.info("IBKR order transmitted: %s %s %g", o.side, o.symbol, o.qty)
        return orders

    def place_order(
        self, symbol: str, side: str, qty: float, *, execute: bool = False
    ) -> PlannedOrder:
        """Place a single market order. Transmits only if ``execute`` (else a no-op plan)."""
        order = PlannedOrder(symbol, side.upper(), qty, 0.0, 0.0)
        if not execute:
            return order
        from ib_insync import MarketOrder, Stock  # lazy

        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        self._ib.placeOrder(contract, MarketOrder(order.side, qty))
        logger.info("IBKR order transmitted: %s %s %g", order.side, symbol, qty)
        return order

    def open_orders(self) -> list[dict[str, object]]:
        """Resting orders at the gateway (needs a connected client)."""
        return [
            {"id": t.order.orderId, "symbol": t.contract.symbol,
             "side": t.order.action, "qty": float(t.order.totalQuantity)}
            for t in self._ib.openTrades()
        ]

    def cancel_order(self, order_id: int) -> dict[str, object]:
        """Cancel a live order at the gateway."""
        for t in self._ib.openTrades():
            if t.order.orderId == order_id:
                self._ib.cancelOrder(t.order)
                return {"id": order_id, "status": "cancelled"}
        raise ValueError(f"order {order_id} not found")
