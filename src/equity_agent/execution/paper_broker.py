"""Paper-trading broker: stateful cash / positions / trades persisted in the DB.

Simulates fills at a given price with fees + slippage, rebalancing to target
weights — the same model as the backtest engine, but stateful across daily runs.
Long-only. A real IBKR broker (Phase 4 live) will implement the same `rebalance`
interface later.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import delete, select

from ..storage.db import session_scope
from ..storage.models import Account, EquitySnapshot, PendingOrder, Position, Trade

logger = logging.getLogger("equity_agent")


def reset_account(starting_cash: float) -> None:
    """Wipe paper state and start fresh with the given cash."""
    with session_scope() as s:
        for model in (Trade, Position, EquitySnapshot, PendingOrder, Account):
            s.execute(delete(model))
        s.add(Account(id=1, cash=starting_cash, starting_cash=starting_cash))


def get_positions() -> dict[str, float]:
    with session_scope() as s:
        return {p.symbol: p.qty for p in s.scalars(select(Position)).all()}


def rebalance(
    target_weights: dict[str, float],
    prices: dict[str, float],
    *,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
    starting_cash: float = 100_000.0,
    now: datetime | None = None,
) -> dict[str, float]:
    """Rebalance the paper portfolio to target weights at the given prices."""
    now = now or datetime.now(UTC)
    fee_rate = fee_bps / 1e4
    slip = slippage_bps / 1e4

    with session_scope() as s:
        acc = s.get(Account, 1)
        if acc is None:
            acc = Account(id=1, cash=starting_cash, starting_cash=starting_cash)
            s.add(acc)
        pos = {p.symbol: p for p in s.scalars(select(Position)).all()}

        equity = acc.cash + sum(p.qty * prices.get(sym, 0.0) for sym, p in pos.items())

        for sym in set(target_weights) | set(pos):
            price = prices.get(sym)
            if price is None or price <= 0:
                continue
            cur_qty = pos[sym].qty if sym in pos else 0.0
            desired_qty = equity * target_weights.get(sym, 0.0) / price
            delta = desired_qty - cur_qty
            if abs(delta * price) < 1e-6:
                continue

            exec_price = price * (1.0 + slip * (1.0 if delta > 0 else -1.0))
            notional = delta * exec_price
            fee = abs(notional) * fee_rate
            acc.cash -= notional + fee

            realized: float | None = None
            if sym in pos:
                p = pos[sym]
                if delta < 0:  # selling: realise P&L on the shares sold
                    realized = (exec_price - p.avg_price) * (-delta)
                new_qty = p.qty + delta
                if abs(new_qty) < 1e-9:
                    s.delete(p)
                    pos.pop(sym)
                else:
                    if delta > 0:  # buying more: update average cost
                        p.avg_price = (p.avg_price * p.qty + exec_price * delta) / new_qty
                    p.qty = new_qty
            elif delta > 0:
                pos[sym] = Position(symbol=sym, qty=delta, avg_price=exec_price, opened_at=now)
                s.add(pos[sym])

            s.add(
                Trade(
                    symbol=sym,
                    side="BUY" if delta > 0 else "SELL",
                    qty=abs(delta),
                    price=exec_price,
                    executed_at=now,
                    fees=fee,
                    pnl=realized,
                    reason="rebalance",
                )
            )

        acc.updated_at = now
        positions_value = sum(p.qty * prices.get(sym, 0.0) for sym, p in pos.items())
        s.add(
            EquitySnapshot(
                ts=now, cash=acc.cash, positions_value=positions_value,
                equity=acc.cash + positions_value,
            )
        )
        return {
            "cash": acc.cash,
            "positions_value": positions_value,
            "equity": acc.cash + positions_value,
            "n_positions": float(len(pos)),
        }


def place_order(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    *,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
    starting_cash: float = 100_000.0,
    now: datetime | None = None,
) -> dict[str, object]:
    """Execute a single manual market order against the paper account.

    ``side`` is BUY or SELL, ``qty`` > 0, ``price`` the reference price (slippage is
    applied adversely). Updates cash, the position (average cost on buys, realized
    P&L on sells), records a Trade(reason="manual"), and writes an equity snapshot.
    """
    side = side.upper()
    if side not in {"BUY", "SELL"} or qty <= 0 or price <= 0:
        raise ValueError("side must be BUY/SELL, qty>0, price>0")
    now = now or datetime.now(UTC)
    slip = slippage_bps / 1e4
    fee_rate = fee_bps / 1e4
    signed = qty if side == "BUY" else -qty
    exec_price = price * (1.0 + slip * (1.0 if signed > 0 else -1.0))

    with session_scope() as s:
        acc = s.get(Account, 1)
        if acc is None:
            acc = Account(id=1, cash=starting_cash, starting_cash=starting_cash)
            s.add(acc)
        pos = {p.symbol: p for p in s.scalars(select(Position)).all()}

        existing = pos.get(symbol)
        held = existing.qty if existing else 0.0
        if side == "SELL" and qty > held + 1e-9:
            raise ValueError(f"cannot sell {qty} {symbol}; only {held} held (no shorting on paper)")

        notional = signed * exec_price
        fee = abs(notional) * fee_rate
        acc.cash -= notional + fee

        realized: float | None = None
        if existing is None:
            pos[symbol] = Position(symbol=symbol, qty=qty, avg_price=exec_price, opened_at=now)
            s.add(pos[symbol])
        else:
            new_qty = existing.qty + signed
            if side == "SELL":
                realized = (exec_price - existing.avg_price) * qty
            elif new_qty > 0:
                cost = existing.avg_price * existing.qty + exec_price * qty
                existing.avg_price = cost / new_qty
            if abs(new_qty) < 1e-9:
                s.delete(existing)
                pos.pop(symbol)
            else:
                existing.qty = new_qty

        s.add(
            Trade(
                symbol=symbol, side=side, qty=qty, price=exec_price,
                executed_at=now, fees=fee, pnl=realized, reason="manual",
            )
        )
        acc.updated_at = now
        positions_value = sum(
            p.qty * (exec_price if sym == symbol else p.avg_price) for sym, p in pos.items()
        )
        equity = acc.cash + positions_value
        s.add(EquitySnapshot(ts=now, cash=acc.cash, positions_value=positions_value, equity=equity))
        return {
            "filled": "ok", "symbol": symbol, "side": side, "qty": qty,
            "exec_price": exec_price, "fee": fee, "cash": acc.cash, "equity": equity,
        }


def place_limit_order(
    symbol: str,
    side: str,
    qty: float,
    limit_price: float,
    kind: str = "limit",
    now: datetime | None = None,
) -> dict[str, object]:
    """Create a resting order (paper). ``kind`` = limit or stop; ``limit_price`` is the
    trigger. A limit fills at a favourable cross; a stop fills at an adverse/breakout
    cross (SELL stop-loss below, BUY stop-entry above)."""
    side = side.upper()
    kind = kind.lower()
    if side not in {"BUY", "SELL"} or qty <= 0 or limit_price <= 0 or kind not in {"limit", "stop"}:
        raise ValueError("side BUY/SELL, qty>0, price>0, kind limit|stop")
    with session_scope() as s:
        order = PendingOrder(
            symbol=symbol, side=side, qty=qty, limit_price=limit_price, kind=kind,
            status="open", created_at=now or datetime.now(UTC),
        )
        s.add(order)
        s.flush()
        return {
            "order_id": order.id, "status": "open", "symbol": symbol, "side": side,
            "qty": qty, "limit_price": limit_price, "kind": kind,
        }


def place_stop_order(
    symbol: str, side: str, qty: float, stop_price: float, now: datetime | None = None
) -> dict[str, object]:
    """Create a resting stop order (paper). SELL stop = stop-loss; BUY stop = breakout."""
    return place_limit_order(symbol, side, qty, stop_price, kind="stop", now=now)


def get_open_orders() -> list[dict[str, object]]:
    """Open (resting) limit/stop orders, newest first."""
    with session_scope() as s:
        stmt = select(PendingOrder).where(PendingOrder.status == "open")
        rows = s.scalars(stmt.order_by(PendingOrder.id.desc())).all()
        return [
            {
                "id": o.id, "symbol": o.symbol, "side": o.side, "qty": o.qty,
                "limit_price": o.limit_price, "kind": o.kind, "created_at": str(o.created_at),
            }
            for o in rows
        ]


def _crossed(kind: str, side: str, price: float, trigger: float) -> bool:
    """Whether a resting order triggers at ``price``.

    limit: BUY fills below (cheaper), SELL fills above (richer).
    stop:  BUY fills above (breakout), SELL fills below (stop-loss).
    """
    if kind == "stop":
        return price >= trigger if side == "BUY" else price <= trigger
    return price <= trigger if side == "BUY" else price >= trigger


def cancel_order(order_id: int) -> dict[str, object]:
    """Cancel an open limit order."""
    with session_scope() as s:
        o = s.get(PendingOrder, order_id)
        if o is None or o.status != "open":
            raise ValueError(f"order {order_id} not open")
        o.status = "cancelled"
        return {"id": order_id, "status": "cancelled"}


def check_pending_fills(prices: dict[str, float], now: datetime | None = None) -> int:
    """Fill open limit/stop orders whose trigger is crossed by ``prices``. Returns #filled.

    Fills at the trigger price (no slippage). An order that can't fill (e.g. oversell)
    is cancelled.
    """
    now = now or datetime.now(UTC)
    with session_scope() as s:
        opens = s.scalars(select(PendingOrder).where(PendingOrder.status == "open")).all()
        candidates = [
            (o.id, o.symbol, o.side, o.qty, o.limit_price)
            for o in opens
            if o.symbol in prices and _crossed(o.kind, o.side, prices[o.symbol], o.limit_price)
        ]

    filled = 0
    for oid, symbol, side, qty, limit_price in candidates:
        try:
            place_order(symbol, side, qty, limit_price, slippage_bps=0.0, now=now)
            status, fill_price = "filled", limit_price
            filled += 1
        except ValueError:
            status, fill_price = "cancelled", None
        with session_scope() as s2:
            o = s2.get(PendingOrder, oid)
            if o is not None and o.status == "open":
                o.status = status
                o.filled_at = now
                o.fill_price = fill_price
    return filled
