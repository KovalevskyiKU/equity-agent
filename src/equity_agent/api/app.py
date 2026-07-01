"""FastAPI application — exposes the equity-agent core to the trading frontend.

Read endpoints (instruments, bars, signals, portfolio, trades) + a manual order
endpoint. Orders default to the **paper** broker; live venues require an explicit
``venue`` and are gated for safety. All heavy logic is reused from the core packages.
"""

from __future__ import annotations

import asyncio
from typing import cast

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import PROJECT_ROOT, load_config
from ..storage.db import init_db

TRADING_DAYS_EQUITY = 252
TRADING_DAYS_CRYPTO = 365


def _crypto_symbols() -> set[str]:
    cfg = load_config()
    return {cfg.crypto_benchmark, *cfg.crypto_universe}


def _is_crypto(symbol: str) -> bool:
    return symbol in _crypto_symbols()


class OrderRequest(BaseModel):
    symbol: str
    side: str  # BUY | SELL (validated downstream)
    qty: float = Field(gt=0)
    venue: str = "paper"  # paper | ibkr | binance
    order_type: str = "market"  # market | limit (limit = paper resting order)
    limit_price: float | None = None  # required for order_type=limit
    price: float | None = None  # optional market-price override; default = latest close
    confirm: bool = False  # live venues require an explicit confirm (two-step)


class AlertRequest(BaseModel):
    symbol: str
    kind: str  # above | below | trend_up | trend_down
    level: float | None = None  # required for above/below


def build_portfolio() -> dict[str, object]:
    """Paper account snapshot: cash, positions marked to last close, monitor, curve.

    Module-level so both GET /api/portfolio and the WebSocket pusher reuse it.
    """
    from ..backtest.panels import load_price_panels
    from ..dashboard.data import paper_overview
    from ..execution.paper_broker import check_pending_fills, get_open_orders
    from ..monitoring import monitor_summary
    from ..signals.feature_store import load_bars

    # Fill any resting limit orders crossed by the latest price, then snapshot.
    open_orders = get_open_orders()
    if open_orders:
        fill_prices: dict[str, float] = {}
        for o in open_orders:
            b = load_bars(str(o["symbol"]))
            if not b.empty:
                fill_prices[str(o["symbol"])] = float(b["close"].iloc[-1])
        if fill_prices and check_pending_fills(fill_prices):
            open_orders = get_open_orders()

    # Evaluate armed alerts against the latest price / trend.
    from ..alerts import check_alerts, list_alerts

    alerts = list_alerts()
    armed = [a for a in alerts if a["status"] == "armed"]
    if armed:
        a_prices: dict[str, float] = {}
        a_trends: dict[str, str] = {}
        for a in armed:
            sym = str(a["symbol"])
            b = load_bars(sym)
            if b.empty:
                continue
            close = b["close"]
            a_prices[sym] = float(close.iloc[-1])
            if len(close) >= 100:
                fast = float(close.rolling(20).mean().iloc[-1])
                slow = float(close.rolling(100).mean().iloc[-1])
                a_trends[sym] = "up" if fast > slow else "down"
        if check_alerts(a_prices, a_trends):
            alerts = list_alerts()

    ov = paper_overview()
    positions = []
    for _, p in cast(pd.DataFrame, ov["positions"]).iterrows():
        bars_df = load_bars(p["symbol"])
        last = float(bars_df["close"].iloc[-1]) if not bars_df.empty else float(p["avg_price"])
        positions.append({
            "symbol": p["symbol"], "qty": float(p["qty"]),
            "avg_price": float(p["avg_price"]), "last": last,
            "market_value": last * float(p["qty"]),
            "unrealized_pnl": (last - float(p["avg_price"])) * float(p["qty"]),
        })
    cfg = load_config()
    _, close_b = load_price_panels([cfg.benchmark])
    spy_close = close_b[cfg.benchmark] if not close_b.empty else None
    curve = cast(pd.DataFrame, ov["equity_curve"])
    return {
        "has_account": ov["has_account"],
        "cash": float(ov["cash"]),  # type: ignore[arg-type]
        "starting_cash": float(ov["starting_cash"]),  # type: ignore[arg-type]
        "positions": positions,
        "open_orders": open_orders,
        "alerts": alerts,
        "monitor": monitor_summary(curve, spy_close),
        "equity_curve": [
            {"time": str(r["ts"]), "equity": float(r["equity"])} for _, r in curve.iterrows()
        ],
    }


def create_app() -> FastAPI:
    app = FastAPI(title="equity-agent API", version="0.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    init_db()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/instruments")
    def instruments() -> list[dict[str, str]]:
        cfg = load_config()
        out: list[dict[str, str]] = []
        for s in cfg.universe:
            out.append({"symbol": s, "asset_class": "equity", "role": "traded"})
        out.append({"symbol": cfg.benchmark, "asset_class": "equity", "role": "benchmark"})
        for s in cfg.crypto_universe:
            out.append({"symbol": s, "asset_class": "crypto", "role": "traded"})
        return out

    @app.get("/api/bars/{symbol}")
    def bars(symbol: str, limit: int = 750) -> list[dict[str, float | str]]:
        from ..signals.feature_store import load_bars

        df = load_bars(symbol)
        if df.empty:
            raise HTTPException(404, f"no bars for {symbol}")
        df = df.tail(limit)
        return [
            {
                "time": str(ts),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r["volume"]),
            }
            for ts, r in df.iterrows()
        ]

    @app.get("/api/signals/{symbol}")
    def signals(symbol: str, fast: int = 20, slow: int = 100) -> dict[str, object]:
        from ..signals.feature_store import load_bars

        df = load_bars(symbol)
        if df.empty:
            raise HTTPException(404, f"no bars for {symbol}")
        close = df["close"]
        tdays = TRADING_DAYS_CRYPTO if _is_crypto(symbol) else TRADING_DAYS_EQUITY
        sma_fast = float(close.rolling(fast).mean().iloc[-1])
        sma_slow = float(close.rolling(slow).mean().iloc[-1])
        vol = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(tdays))

        def _safe(x: float) -> float | None:
            return float(x) if np.isfinite(x) else None

        mom_3m = close.iloc[-1] / close.shift(63).iloc[-1] - 1 if len(close) > 63 else float("nan")
        return {
            "symbol": symbol,
            "asset_class": "crypto" if _is_crypto(symbol) else "equity",
            "last_price": float(close.iloc[-1]),
            "sma_fast": _safe(sma_fast),
            "sma_slow": _safe(sma_slow),
            "trend": "up" if sma_fast > sma_slow else "down",
            "ann_vol": _safe(vol),
            "momentum_3m": _safe(mom_3m),
        }

    @app.get("/api/portfolio")
    def portfolio() -> dict[str, object]:
        return build_portfolio()

    @app.get("/api/trades")
    def trades(limit: int = 50) -> list[dict[str, object]]:
        from ..dashboard.data import paper_overview

        df = cast(pd.DataFrame, paper_overview()["trades"])
        out = []
        for _, t in df.iterrows():
            out.append({
                "time": str(t["time"]), "symbol": t["symbol"], "side": t["side"],
                "qty": float(t["qty"]), "price": float(t["price"]),
                "pnl": None if pd.isna(t["pnl"]) else float(t["pnl"]),
            })
        return out[:limit]

    @app.get("/api/orders/open")
    def open_orders() -> list[dict[str, object]]:
        from ..execution.paper_broker import get_open_orders

        return get_open_orders()

    @app.delete("/api/orders/{order_id}")
    def cancel(order_id: int) -> dict[str, object]:
        from ..execution.paper_broker import cancel_order

        try:
            return cancel_order(order_id)
        except ValueError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/orders")
    def create_order(req: OrderRequest) -> dict[str, object]:
        from ..signals.feature_store import load_bars

        # Paper resting limit/stop order (trigger price = limit_price).
        otype = req.order_type.lower()
        if otype in {"limit", "stop"}:
            if req.venue.lower() != "paper":
                raise HTTPException(400, f"{otype} orders are paper-only for now")
            if req.limit_price is None or req.limit_price <= 0:
                raise HTTPException(400, f"limit_price (trigger) required for a {otype} order")
            from ..execution.paper_broker import place_limit_order

            try:
                return place_limit_order(req.symbol, req.side, req.qty, req.limit_price, kind=otype)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e

        price = req.price
        if price is None:
            bars_df = load_bars(req.symbol)
            if bars_df.empty:
                raise HTTPException(404, f"no price for {req.symbol}; pass an explicit price")
            price = float(bars_df["close"].iloc[-1])

        venue = req.venue.lower()
        if venue == "paper":
            from ..execution.paper_broker import place_order

            try:
                return place_order(req.symbol, req.side, req.qty, price)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e

        if venue not in {"ibkr", "binance"}:
            raise HTTPException(400, f"unknown venue '{req.venue}'")

        # Live venues are two-step: first call returns a preview to confirm, second
        # call (confirm=true) actually transmits. Nothing is sent without confirm.
        if not req.confirm:
            return {
                "preview": True, "requires_confirmation": True, "venue": venue,
                "symbol": req.symbol, "side": req.side.upper(), "qty": req.qty,
                "est_price": price, "est_notional": req.qty * price,
            }

        from ..config import get_settings

        s = get_settings()
        try:
            if venue == "ibkr":
                from ..execution.ibkr_broker import IBKRBroker

                broker = IBKRBroker(s.ibkr_host, s.ibkr_port, s.ibkr_client_id)
                broker.connect()
                try:
                    o = broker.place_order(req.symbol, req.side, req.qty, execute=True)
                finally:
                    broker.disconnect()
            else:  # binance
                from ..execution.crypto_broker import BinanceBroker

                broker_c = BinanceBroker(s.binance_api_key, s.binance_secret)
                broker_c.connect()
                o = broker_c.place_order(req.symbol, req.side, req.qty, execute=True)
        except Exception as e:  # noqa: BLE001 - surface connect/auth failures to the UI
            raise HTTPException(502, f"{venue} not reachable: {e}") from e
        return {
            "filled": "transmitted", "venue": venue,
            "symbol": o.symbol, "side": o.side, "qty": o.qty,
        }

    @app.get("/api/alerts")
    def alerts_list() -> list[dict[str, object]]:
        from ..alerts import list_alerts

        return list_alerts()

    @app.post("/api/alerts")
    def alerts_create(req: AlertRequest) -> dict[str, object]:
        from ..alerts import create_alert

        try:
            return create_alert(req.symbol, req.kind, req.level)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.delete("/api/alerts/{alert_id}")
    def alerts_delete(alert_id: int) -> dict[str, object]:
        from ..alerts import delete_alert

        try:
            return delete_alert(alert_id)
        except ValueError as e:
            raise HTTPException(404, str(e)) from e

    @app.websocket("/ws/portfolio")
    async def ws_portfolio(ws: WebSocket) -> None:
        """Push a portfolio snapshot every few seconds (live positions/P&L)."""
        await ws.accept()
        try:
            while True:
                snapshot = await asyncio.to_thread(build_portfolio)
                await ws.send_json(snapshot)
                await asyncio.sleep(3)
        except WebSocketDisconnect:
            return

    # Serve the built React cockpit (if present) so `eqa serve` runs the whole app on
    # one port. Mounted last so /api and /ws routes take precedence. Build it with
    # `cd frontend && npm run build`.
    dist = PROJECT_ROOT / "frontend" / "dist"
    if dist.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")

    return app


app = create_app()
