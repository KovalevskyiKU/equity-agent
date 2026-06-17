"""Shared fixtures. The ``temp_db`` fixture isolates each test on its own SQLite file."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from equity_agent import config
from equity_agent.storage import db


@pytest.fixture()
def temp_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file.as_posix()}")

    # Reset cached settings + engine so the temp URL takes effect.
    config.get_settings.cache_clear()
    db._engine = None
    db._SessionFactory = None

    db.init_db()
    yield

    db._engine = None
    db._SessionFactory = None
    config.get_settings.cache_clear()
