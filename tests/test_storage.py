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
