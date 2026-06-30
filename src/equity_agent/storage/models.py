"""ORM models — the system of record for data, decisions and trades.

Every table that stores a model judgement (NewsItem, Decision) keeps the source
text and the model name, so every decision is auditable after the fact.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Instrument(Base):
    __tablename__ = "instruments"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120))
    asset_type: Mapped[str] = mapped_column(String(20), default="equity")
    sector: Mapped[str | None] = mapped_column(String(60))
    role: Mapped[str] = mapped_column(String(20), default="traded")  # traded|benchmark|regime
    active: Mapped[bool] = mapped_column(default=True)


class DailyBar(Base):
    __tablename__ = "daily_bars"
    __table_args__ = (UniqueConstraint("symbol", "ts", name="uq_daily_bar"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    ts: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20), default="yfinance")


class NewsItem(Base):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str | None] = mapped_column(String(20), index=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    source: Mapped[str] = mapped_column(String(60))
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(600), unique=True)
    # Filled by the LLM scorer (Phase 1). published_at gates look-ahead in backtests.
    sentiment: Mapped[float | None] = mapped_column(Float)
    impact: Mapped[float | None] = mapped_column(Float)
    llm_model: Mapped[str | None] = mapped_column(String(60))
    llm_rationale: Mapped[str | None] = mapped_column(Text)


class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = (UniqueConstraint("symbol", "asof_date", name="uq_decision"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    asof_date: Mapped[date] = mapped_column(Date, index=True)
    action: Mapped[str] = mapped_column(String(10))  # BUY|SELL|HOLD
    conviction: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    rationale: Mapped[str | None] = mapped_column(Text)
    signals_json: Mapped[str | None] = mapped_column(Text)  # raw per-signal inputs
    model: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    trades: Mapped[list[Trade]] = relationship(back_populates="decision")


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    qty: Mapped[float] = mapped_column(Float)
    avg_price: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    take_profit: Mapped[float | None] = mapped_column(Float)


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(10))  # BUY|SELL
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    pnl: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(String(40))
    decision_id: Mapped[int | None] = mapped_column(ForeignKey("decisions.id"))

    decision: Mapped[Decision | None] = relationship(back_populates="trades")


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)


class PendingOrder(Base):
    """A resting limit order (paper). Fills when the last price crosses the limit."""

    __tablename__ = "pending_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(10))  # BUY|SELL
    qty: Mapped[float] = mapped_column(Float)
    limit_price: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(12), default="open")  # open|filled|cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fill_price: Mapped[float | None] = mapped_column(Float)


class Account(Base):
    """Single-row paper-trading cash account (id always 1)."""

    __tablename__ = "account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float)
    starting_cash: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
