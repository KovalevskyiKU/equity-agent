import pytest

from equity_agent.alerts import check_alerts, create_alert, delete_alert, list_alerts


def test_above_alert_triggers_on_cross(temp_db: None) -> None:
    create_alert("AAA", "above", 100.0)
    assert check_alerts({"AAA": 95.0}, {}) == 0  # below level -> stays armed
    assert check_alerts({"AAA": 105.0}, {}) == 1  # >= level -> triggered
    assert list_alerts()[0]["status"] == "triggered"
    assert check_alerts({"AAA": 110.0}, {}) == 0  # already triggered -> no re-fire


def test_trend_alert_and_delete(temp_db: None) -> None:
    a = create_alert("BBB", "trend_up")
    assert check_alerts({}, {"BBB": "down"}) == 0
    assert check_alerts({}, {"BBB": "up"}) == 1
    delete_alert(int(a["id"]))  # type: ignore[arg-type]
    assert list_alerts() == []


def test_above_requires_level(temp_db: None) -> None:
    with pytest.raises(ValueError):
        create_alert("CCC", "above")
