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


def test_live_order_requires_confirmation_first(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.post(
        "/api/orders",
        json={"symbol": "BTC-USD", "side": "buy", "qty": 1, "venue": "binance", "price": 50000.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["requires_confirmation"] is True and body["est_notional"] == 50000.0


def test_live_order_confirm_without_keys_errors(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.post(
        "/api/orders",
        json={
            "symbol": "BTC-USD", "side": "buy", "qty": 1, "venue": "binance",
            "price": 50000.0, "confirm": True,
        },
    )
    assert r.status_code == 502  # ccxt/keys not available -> not reachable


def test_unknown_venue_rejected(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.post(
        "/api/orders",
        json={"symbol": "AAA", "side": "buy", "qty": 1, "venue": "lol", "price": 10.0},
    )
    assert r.status_code == 400


def test_limit_order_lifecycle(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.post(
        "/api/orders",
        json={"symbol": "AAA", "side": "buy", "qty": 1, "order_type": "limit", "limit_price": 50.0},
    )
    assert r.status_code == 200 and r.json()["status"] == "open"
    oid = r.json()["order_id"]
    assert any(o["id"] == oid for o in c.get("/api/orders/open").json())
    assert c.delete(f"/api/orders/{oid}").status_code == 200
    assert c.get("/api/orders/open").json() == []


def test_limit_order_requires_price(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.post(
        "/api/orders", json={"symbol": "AAA", "side": "buy", "qty": 1, "order_type": "limit"}
    )
    assert r.status_code == 400


def test_alert_lifecycle(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.post("/api/alerts", json={"symbol": "AAA", "kind": "above", "level": 100.0})
    assert r.status_code == 200 and r.json()["status"] == "armed"
    aid = r.json()["id"]
    assert any(a["id"] == aid for a in c.get("/api/alerts").json())
    assert c.delete(f"/api/alerts/{aid}").status_code == 200
    assert c.get("/api/alerts").json() == []


def test_alert_above_requires_level(temp_db: None) -> None:
    c = TestClient(create_app())
    r = c.post("/api/alerts", json={"symbol": "AAA", "kind": "above"})
    assert r.status_code == 400


def test_ws_portfolio_pushes_snapshot(temp_db: None) -> None:
    c = TestClient(create_app())
    with c.websocket_connect("/ws/portfolio") as ws:
        msg = ws.receive_json()
        assert "positions" in msg and "cash" in msg and "monitor" in msg
