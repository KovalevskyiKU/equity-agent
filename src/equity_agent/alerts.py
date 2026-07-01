"""Price / trend alerts — armed rules that fire when their condition first holds.

Kinds: ``above`` / ``below`` (price vs a level) and ``trend_up`` / ``trend_down``
(fast SMA above/below slow SMA). An armed alert becomes ``triggered`` the first time
its condition is met on a portfolio refresh; the cockpit surfaces triggered alerts.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from .storage.db import session_scope
from .storage.models import AlertRule

_KINDS = {"above", "below", "trend_up", "trend_down"}


def create_alert(symbol: str, kind: str, level: float | None = None) -> dict[str, object]:
    """Arm a new alert. ``level`` is required for above/below, ignored for trend_*."""
    kind = kind.lower()
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}")
    if kind in {"above", "below"} and (level is None or level <= 0):
        raise ValueError("a positive level is required for above/below alerts")
    with session_scope() as s:
        a = AlertRule(symbol=symbol, kind=kind, level=level, status="armed")
        s.add(a)
        s.flush()
        return {"id": a.id, "symbol": symbol, "kind": kind, "level": level, "status": "armed"}


def list_alerts() -> list[dict[str, object]]:
    """All alerts, newest first (armed + triggered)."""
    with session_scope() as s:
        rows = s.scalars(select(AlertRule).order_by(AlertRule.id.desc())).all()
        return [
            {
                "id": a.id, "symbol": a.symbol, "kind": a.kind, "level": a.level,
                "status": a.status,
                "triggered_at": str(a.triggered_at) if a.triggered_at else None,
            }
            for a in rows
        ]


def delete_alert(alert_id: int) -> dict[str, object]:
    with session_scope() as s:
        a = s.get(AlertRule, alert_id)
        if a is None:
            raise ValueError(f"alert {alert_id} not found")
        s.delete(a)
        return {"id": alert_id, "deleted": True}


def check_alerts(
    prices: dict[str, float], trends: dict[str, str], now: datetime | None = None
) -> int:
    """Mark armed alerts triggered when their condition holds. Returns #triggered."""
    now = now or datetime.now(UTC)
    fired = 0
    with session_scope() as s:
        armed = s.scalars(select(AlertRule).where(AlertRule.status == "armed")).all()
        for a in armed:
            px = prices.get(a.symbol)
            tr = trends.get(a.symbol)
            hit = (
                (a.kind == "above" and px is not None and a.level is not None and px >= a.level)
                or (a.kind == "below" and px is not None and a.level is not None and px <= a.level)
                or (a.kind == "trend_up" and tr == "up")
                or (a.kind == "trend_down" and tr == "down")
            )
            if hit:
                a.status = "triggered"
                a.triggered_at = now
                fired += 1
    return fired
