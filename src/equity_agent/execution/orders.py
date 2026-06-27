"""Shared, pure order planner — used by every broker adapter (IBKR, Binance, …).

Turning a target-weight book into the orders that close the gap is broker-agnostic
and the one piece worth unit-testing without a live connection. Each adapter adds
only the thin glue to fetch balances/prices and transmit.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    Pure and deterministic: ``desired_qty = equity * weight / price`` minus the
    current position. ``whole_shares=True`` rounds down (US stocks); set False for
    crypto (fractional). Orders below ``min_notional`` are skipped (no dust trades).
    """
    orders: list[PlannedOrder] = []
    for sym in sorted(set(target_weights) | set(current_positions)):
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        desired = equity * target_weights.get(sym, 0.0) / price
        if whole_shares:
            desired = float(int(desired))
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
