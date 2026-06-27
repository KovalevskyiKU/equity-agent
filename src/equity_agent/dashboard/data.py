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


def factor_leaderboard(top_n: int = 15) -> dict[str, pd.DataFrame]:
    """Current top-N names by each factor — what each factor *favors today*.

    Informational only: backtests show these factors don't beat SPY net of costs
    once survivorship is removed (see docs/PHASE1_FINDINGS.md). This is "the picks",
    not a buy list. No network / no point-in-time mask (uses today's universe).
    """
    from ..research.factor_eval import momentum_factor
    from ..research.fundamental_factors import build_fundamental_panels

    cfg = load_config()
    _, close = load_price_panels(cfg.universe)
    if close.empty:
        return {}

    def lead(series: pd.Series, label: str, lo: float | None, hi: float | None) -> pd.DataFrame:
        # Drop implausible values (corporate-action / extraction artifacts: a 3000%
        # momentum from a spinoff, a 50%+ earnings yield, a 150%+ ROE on tiny equity).
        s = series.dropna()
        if lo is not None:
            s = s[s > lo]
        if hi is not None:
            s = s[s < hi]
        s = s.nlargest(top_n)
        return pd.DataFrame({"symbol": list(s.index), label: s.to_numpy().round(4)})

    funds = build_fundamental_panels(list(close.columns), close)
    return {
        "Momentum (12-1)": lead(momentum_factor(close).iloc[-1], "momentum", None, 3.0),
        "Value (earnings yield)": lead(funds["earnings_yield"].iloc[-1], "earn_yield", 0.0, 0.5),
        "Quality (ROE)": lead(funds["roe"].iloc[-1], "roe", 0.0, 1.5),
    }


def factor_performance() -> pd.DataFrame:
    """Backtested returns of monthly top-quintile factor portfolios vs SPY.

    Over today's universe -> survivorship-biased (inflated); SPY (cap-weight) is the
    honest bar. The corrected, point-in-time read is `eqa factor-backtest-pit`.
    """
    from typing import cast

    from ..backtest.factor_portfolio import run_factor_portfolios

    cfg = load_config()
    res = run_factor_portfolios(cfg.universe, cfg.benchmark)
    if not res:
        return pd.DataFrame()

    def row(name: str, m: dict) -> dict[str, object]:
        return {
            "strategy": name,
            "total_x": round(m["total_return"], 2),
            "cagr_%": round(m["cagr"] * 100, 1),
            "sharpe": round(m["sharpe"], 2),
            "max_dd_%": round(m["max_drawdown"] * 100, 1),
        }

    rows = [row(name, fb.portfolio) for name, fb in cast(dict, res["factors"]).items()]
    rows.append(row(cfg.benchmark, cast(dict, res["spy"])))
    return pd.DataFrame(rows)


def crypto_overview() -> dict[str, pd.DataFrame]:
    """Crypto comparison (hold-BTC vs managed) + funding carry, for the dashboard.

    Funding is only fetched (network) when crypto price data exists, so the empty
    case is cheap and offline.
    """
    from ..backtest.crypto import run_crypto_comparison
    from ..data.funding import carry_summary, fetch_funding

    comparison = run_crypto_comparison()
    if comparison.empty:
        return {"comparison": comparison, "funding": pd.DataFrame()}

    rows = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        f = fetch_funding(sym)
        if f.empty:
            continue
        s = carry_summary(f)
        rows.append(
            {
                "perp": sym,
                "gross_%/yr": round(s["ann_carry_gross"] * 100, 1),
                "net_%/yr": round(s["ann_carry_net"] * 100, 1),
                "pos_%": round(s["pct_positive"] * 100, 0),
                "n": s["n"],
            }
        )
    return {"comparison": comparison, "funding": pd.DataFrame(rows)}


def recent_news(limit: int = 50) -> pd.DataFrame:
    with session_scope() as s:
        rows = [
            (n.published_at, n.symbol, n.sentiment, n.impact, n.title)
            for n in s.scalars(
                select(NewsItem).order_by(desc(NewsItem.published_at)).limit(limit)
            ).all()
        ]
    return pd.DataFrame(rows, columns=["published", "symbol", "sentiment", "impact", "title"])
