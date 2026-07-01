from datetime import date

from sqlalchemy import select

from equity_agent.storage import session_scope
from equity_agent.storage.models import DailyBar, Instrument


def test_instrument_and_bar_roundtrip(temp_db: None) -> None:
    with session_scope() as session:
        session.add(Instrument(symbol="AAPL", role="traded"))
        session.add(
            DailyBar(
                symbol="AAPL",
                ts=date(2024, 1, 2),
                open=10.0,
                high=11.0,
                low=9.5,
                close=10.5,
                adj_close=10.5,
                volume=1000.0,
            )
        )

    with session_scope() as session:
        bars = session.scalars(select(DailyBar).where(DailyBar.symbol == "AAPL")).all()
        assert len(bars) == 1
        assert bars[0].close == 10.5
        inst = session.get(Instrument, "AAPL")
        assert inst is not None and inst.role == "traded"


def test_ensure_columns_additive_migration(temp_db: None) -> None:
    """init_db adds a model column missing from an existing table (no data loss)."""
    from sqlalchemy import inspect, text

    from equity_agent.storage.db import get_engine, init_db

    eng = get_engine()
    with eng.begin() as c:
        c.execute(text("DROP TABLE pending_orders"))
        # Recreate the old schema WITHOUT the `kind` column, with a row.
        c.execute(
            text(
                "CREATE TABLE pending_orders (id INTEGER PRIMARY KEY, symbol VARCHAR, "
                "side VARCHAR, qty FLOAT, limit_price FLOAT, status VARCHAR, "
                "created_at DATETIME, filled_at DATETIME, fill_price FLOAT)"
            )
        )
        c.execute(text("INSERT INTO pending_orders (id, symbol) VALUES (1, 'AAA')"))

    init_db()  # should ADD COLUMN kind DEFAULT 'limit'
    cols = {col["name"] for col in inspect(eng).get_columns("pending_orders")}
    assert "kind" in cols
