from datetime import UTC, datetime

from equity_agent.decision.risk_off import apply_risk_off_gate
from equity_agent.storage import session_scope
from equity_agent.storage.models import NewsItem


def test_risk_off_gates_negative_sentiment(temp_db: None) -> None:
    when = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
    with session_scope() as s:
        s.add(NewsItem(symbol="BAD", published_at=when, source="x", title="crash",
                       url="u1", sentiment=-0.8, impact=0.9))
        s.add(NewsItem(symbol="OK", published_at=when, source="x", title="fine",
                       url="u2", sentiment=0.2, impact=0.5))

    gated_weights, gated = apply_risk_off_gate({"BAD": 0.5, "OK": 0.5}, threshold=-0.3)
    assert gated == ["BAD"]
    assert gated_weights["BAD"] == 0.0
    assert gated_weights["OK"] == 0.5  # untouched


def test_risk_off_noop_without_news(temp_db: None) -> None:
    w = {"A": 0.4, "B": 0.4}
    out, gated = apply_risk_off_gate(w)
    assert gated == []
    assert out == w
