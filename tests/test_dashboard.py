from equity_agent.dashboard.data import (
    factor_leaderboard,
    factor_performance,
    paper_overview,
    recent_news,
    strategy_curves,
)


def test_paper_overview_empty(temp_db: None) -> None:
    ov = paper_overview()
    assert ov["has_account"] is False
    assert ov["positions"].empty  # type: ignore[union-attr]
    assert ov["equity_curve"].empty  # type: ignore[union-attr]


def test_recent_news_empty(temp_db: None) -> None:
    assert recent_news().empty


def test_strategy_curves_no_data(temp_db: None) -> None:
    curves, metrics = strategy_curves()
    assert curves.empty
    assert metrics.empty


def test_factor_leaderboard_no_data(temp_db: None) -> None:
    assert factor_leaderboard() == {}


def test_factor_performance_no_data(temp_db: None) -> None:
    assert factor_performance().empty
