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
from ..storage.models import Account, EquitySnapshot, Position, Trade

logger = logging.getLogger("equity_agent")


def reset_account(starting_cash: float) -> None:
    """Wipe paper state and start fresh with the given cash."""
    with session_scope() as s:
        for model in (Trade, Position, EquitySnapshot, Account):
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
