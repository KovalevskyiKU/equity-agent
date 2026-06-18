from datetime import UTC, datetime

from equity_agent.signals.sentiment import get_daily_sentiment
from equity_agent.storage import session_scope
from equity_agent.storage.models import NewsItem


def test_daily_sentiment_is_impact_weighted(temp_db: None) -> None:
    when = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
    with session_scope() as session:
        # Bullish item with full impact + bearish item with zero impact.
        session.add(
            NewsItem(symbol="AAPL", published_at=when, source="x", title="up",
                     url="u1", sentiment=1.0, impact=1.0)
        )
        session.add(
            NewsItem(symbol="AAPL", published_at=when, source="x", title="down",
                     url="u2", sentiment=-1.0, impact=0.0)
        )

    series = get_daily_sentiment("AAPL", halflife_days=1.0)
    # (1.0*1.0 + -1.0*0.0) / (1.0 + 0.0) = 1.0
    assert abs(float(series.iloc[-1]) - 1.0) < 1e-9


def test_daily_sentiment_empty_without_news(temp_db: None) -> None:
    assert get_daily_sentiment("NOPE").empty
