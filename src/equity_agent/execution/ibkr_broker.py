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
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("equity_agent")


@dataclass
class PlannedOrder:
    symbol: str
    side: str  # BUY | SELL
    qty: float
    est_price: float
    est_notional: float


def plan_orders(
    target_weights: dict[str, float],
    prices: dict[str, float],
    equity: float,
    current_positions: dict[str, float],
    *,
    min_notional: float = 1.0,
    whole_shares: bool = True,
) -> list[PlannedOrder]:
    """Diff current positions to target weights -> the orders that close the gap.

    Pure and deterministic (no IB dependency): ``desired_qty = equity * weight /
    price``, rounded to whole shares for US stocks, minus the current position.
    Orders below ``min_notional`` are skipped (no dust trades). This is the same
    intent as ``paper_broker.rebalance`` but emits orders instead of mutating a DB.
    """
    orders: list[PlannedOrder] = []
    for sym in sorted(set(target_weights) | set(current_positions)):
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        desired = equity * target_weights.get(sym, 0.0) / price
        if whole_shares:
            desired = float(int(desired))  # IBKR stocks: whole shares by default
        delta = desired - current_positions.get(sym, 0.0)
        notional = abs(delta) * price
        if notional < min_notional:
            continue
        orders.append(
            PlannedOrder(
                symbol=sym,
                side="BUY" if delta > 0 else "SELL",
                qty=abs(delta),
                est_price=price,
                est_notional=notional,
            )
        )
    return orders


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
