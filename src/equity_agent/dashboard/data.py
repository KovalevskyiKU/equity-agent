"""Data accessors for the dashboard — kept out of the Streamlit script so the UI
stays thin and this layer stays typed and unit-testable."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import desc, select

from ..backtest.engine import BacktestConfig, run_backtest
from ..backtest.metrics import return_summary
from ..backtest.panels import load_price_panels
from ..backtest.strategy import buy_and_hold_equal, single_asset, vol_target_weights
from ..config import load_config
from ..storage.db import session_scope
from ..storage.models import Account, EquitySnapshot, NewsItem, Position, Trade


def paper_overview() -> dict[str, object]:
    """Account summary + positions + equity curve + recent trades (as DataFrames)."""
    with session_scope() as s:
        acc = s.get(Account, 1)
        positions = pd.DataFrame(
            [(p.symbol, p.qty, p.avg_price) for p in s.scalars(select(Position)).all()],
            columns=["symbol", "qty", "avg_price"],
        )
        snaps = pd.DataFrame(
            [
                (e.ts, e.equity, e.cash, e.positions_value)
                for e in s.scalars(select(EquitySnapshot).order_by(EquitySnapshot.ts)).all()
            ],
            columns=["ts", "equity", "cash", "positions_value"],
        )
        trades = pd.DataFrame(
            [
                (t.executed_at, t.symbol, t.side, t.qty, t.price, t.pnl)
                for t in s.scalars(select(Trade).order_by(desc(Trade.executed_at)).limit(50)).all()
            ],
            columns=["time", "symbol", "side", "qty", "price", "pnl"],
        )
        return {
            "has_account": acc is not None,
            "cash": acc.cash if acc else 0.0,
            "starting_cash": acc.starting_cash if acc else 0.0,
            "positions": positions,
            "equity_curve": snaps,
            "trades": trades,
        }


def strategy_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-history equity curves + metrics for core vs basket vs benchmark."""
    cfg = load_config()
    open_u, close_u = load_price_panels(cfg.universe)
    if close_u.empty:
        return pd.DataFrame(), pd.DataFrame()

    bcfg = BacktestConfig()
    core = run_backtest(open_u, close_u, vol_target_weights(close_u), bcfg)
    basket = run_backtest(open_u, close_u, buy_and_hold_equal(close_u), bcfg)
    ob, cb = load_price_panels([cfg.benchmark])
    spy = run_backtest(ob, cb, single_asset(cb, cfg.benchmark), bcfg)

    curves = pd.DataFrame(
        {"core": core.equity, "basket": basket.equity, cfg.benchmark: spy.equity}
    )
    rows = [
        {"strategy": name, **{k: round(v, 3) for k, v in return_summary(res.returns).items()}}
        for name, res in (("core", core), ("basket", basket), (cfg.benchmark, spy))
    ]
    return curves, pd.DataFrame(rows)


def recent_news(limit: int = 50) -> pd.DataFrame:
    with session_scope() as s:
        rows = [
            (n.published_at, n.symbol, n.sentiment, n.impact, n.title)
            for n in s.scalars(
                select(NewsItem).order_by(desc(NewsItem.published_at)).limit(limit)
            ).all()
        ]
    return pd.DataFrame(rows, columns=["published", "symbol", "sentiment", "impact", "title"])
