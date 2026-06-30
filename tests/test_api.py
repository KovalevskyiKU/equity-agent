from fastapi.testclient import TestClient

from equity_agent.api import create_app


def test_health(temp_db: None) -> None:
    c = TestClient(create_app())
    assert c.get("/api/health").json() == {"status": "ok"}


def test_instruments_lists_equity_and_benchmark(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.get("/api/instruments")
    assert r.status_code == 200
    syms = {i["symbol"] for i in r.json()}
    assert "SPY" in syms  # benchmark present


def test_bars_404_without_data(temp_db: None) -> None:
    c = TestClient(create_app())
    assert c.get("/api/bars/AAPL").status_code == 404


def test_paper_order_then_portfolio(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.post("/api/orders", json={"symbol": "AAA", "side": "buy", "qty": 5, "price": 10.0})
    assert r.status_code == 200 and r.json()["filled"] == "ok"
    port = c.get("/api/portfolio").json()
    assert port["has_account"] is True
    assert any(p["symbol"] == "AAA" and p["qty"] == 5.0 for p in port["positions"])


def test_oversell_is_rejected(temp_db: None) -> None:
    c = TestClient(create_app())
    c.post("/api/orders", json={"symbol": "AAA", "side": "buy", "qty": 2, "price": 10.0})
    r = c.post("/api/orders", json={"symbol": "AAA", "side": "sell", "qty": 5, "price": 10.0})
    assert r.status_code == 400  # cannot short on paper


def test_live_venue_not_enabled(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.post(
        "/api/orders",
        json={"symbol": "AAA", "side": "buy", "qty": 1, "venue": "binance", "price": 10.0},
    )
    assert r.status_code == 501
