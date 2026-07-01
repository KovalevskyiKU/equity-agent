"""Engine + session management. One source of truth, swappable by DATABASE_URL."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from ..config import PROJECT_ROOT, get_settings
from .models import Base

logger = logging.getLogger("equity_agent")

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _ensure_sqlite_dir(url: str) -> None:
    """Create the parent directory for a file-based SQLite DB if needed."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return
    db_path = url[len(prefix) :]
    if not db_path or db_path == ":memory:":
        return
    path = Path(db_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> Engine:
    global _engine, _SessionFactory
    if _engine is None:
        url = get_settings().database_url
        _ensure_sqlite_dir(url)
        _engine = create_engine(url, future=True)
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def _ensure_columns(engine: Engine) -> None:
    """Additive migration: ADD COLUMN for model columns missing from existing tables.

    ``create_all`` creates missing tables but never alters existing ones, so a new
    column on an existing table (e.g. PendingOrder.kind) would 500 at runtime. This
    adds those columns non-destructively (nullable or with the model default). Only
    handles additive changes — no drops/renames/type changes.
    """
    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {c["name"] for c in inspector.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing:
                continue
            coltype = col.type.compile(engine.dialect)
            clause = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {coltype}'
            default = None
            d = col.default
            if d is not None and hasattr(d, "arg") and not d.is_callable:
                default = d.arg
            if isinstance(default, str):
                clause += f" DEFAULT '{default}'"
            elif isinstance(default, (int, float)):
                clause += f" DEFAULT {default}"
            with engine.begin() as conn:
                conn.execute(text(clause))
            logger.info("migrated: added column %s.%s", table.name, col.name)


def init_db() -> None:
    """Create missing tables and add any missing columns (additive migration)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    _ensure_columns(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commit on success, rollback on error."""
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
