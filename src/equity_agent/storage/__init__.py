"""Storage layer: SQLAlchemy ORM (system of record) over any DATABASE_URL.

SQLite for dev, Postgres for prod — swapped by the ``DATABASE_URL`` env var only.
A Parquet feature-store (Phase 1) will live alongside this for columnar reads.
"""

from .db import get_engine, init_db, session_scope
from .models import (
    Account,
    Base,
    DailyBar,
    Decision,
    EquitySnapshot,
    Instrument,
    NewsItem,
    Position,
    Trade,
)

__all__ = [
    "Account",
    "Base",
    "DailyBar",
    "Decision",
    "EquitySnapshot",
    "Instrument",
    "NewsItem",
    "Position",
    "Trade",
    "get_engine",
    "init_db",
    "session_scope",
]
