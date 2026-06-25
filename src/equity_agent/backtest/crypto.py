"""Crypto backtests — the 24/7, 365-day-year asset class.

Headline comparison against the honest bar (**hold BTC**, the crypto SPY):

* **trend-following BTC** (MA-cross, long/flat) — the validated crypto edge: it
  de-risks in downtrends (sidestepping the −80% bears) while staying long the
  high-vol bull runs, so it beats buy-and-hold BTC risk-adjusted.
* **vol-target BTC** — does NOT help: crypto's big rallies are also high-vol, so
  scaling by vol cuts the upside too (the opposite of the equity result).
* **cross-sectional alt-momentum** — no edge over hold-BTC net of (higher) crypto
  costs, on a survivorship-biased survivor universe.

Crypto cost assumption is higher than equities (taker fees + wider spreads).
"""

from __future__ import annotations

import pandas as pd

from ..config import load_config
from ..research.factor_eval import momentum_factor, month_end_dates
from .engine import BacktestConfig, run_backtest
from .factor_portfolio import quantile_long_weights
from .metrics import return_summary
from .overlay_backtest import vol_target_index_weights
from .strategy import single_asset

CRYPTO_PERIODS = 365  # 24/7 market
CRYPTO_FEE_BPS = 10.0  # taker fee
CRYPTO_SLIP_BPS = 20.0  # wider spreads than equities


def trend_weights(
    close: pd.DataFrame, symbol: str, *, fast: int = 20, slow: int = 100
) -> pd.DataFrame:
    """Long/flat MA-cross: hold the asset when fast SMA > slow SMA, else cash."""
    px = close[symbol]
    sig = (px.rolling(fast).mean() > px.rolling(slow).mean()).astype(float)
    return pd.DataFrame({symbol: sig})


def run_crypto_comparison(
    *,
    trend_fast: int = 20,
    trend_slow: int = 100,
    vol_target: float = 0.50,
    mom_lookback: int = 90,
    q: float = 0.25,
    fee_bps: float = CRYPTO_FEE_BPS,
    slippage_bps: float = CRYPTO_SLIP_BPS,
) -> pd.DataFrame:
    """Hold-BTC vs trend / vol-target / alt-momentum — one metrics table (365-day)."""
    from .panels import load_price_panels

    cfg = load_config()
    bench = cfg.crypto_benchmark
    open_b, close_b = load_price_panels([bench])
    if close_b.empty:
        return pd.DataFrame()
    cost = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)

    def row(name: str, res: object) -> dict[str, object]:
        m = return_summary(res.returns, periods_per_year=CRYPTO_PERIODS)  # type: ignore[attr-defined]
        return {
            "strategy": name,
            "total_x": round(float(m["total_return"]), 1),
            "cagr_%": round(float(m["cagr"]) * 100, 1),
            "sharpe": round(float(m["sharpe"]), 2),
            "max_dd_%": round(float(m["max_drawdown"]) * 100, 1),
            "calmar": round(float(m["calmar"]), 2),
            "turnover": round(float(res.turnover), 1),  # type: ignore[attr-defined]
        }

    tw = trend_weights(close_b, bench, fast=trend_fast, slow=trend_slow)
    vw = vol_target_index_weights(
        close_b, bench, target_vol=vol_target, band=0.08, trading_days=CRYPTO_PERIODS
    )
    rows = [
        row(f"hold-{bench}", run_backtest(open_b, close_b, single_asset(close_b, bench), cost)),
        row(f"trend {trend_fast}/{trend_slow}", run_backtest(open_b, close_b, tw, cost)),
        row(f"vol-target {int(vol_target * 100)}%", run_backtest(open_b, close_b, vw, cost)),
    ]

    open_u, close_u = load_price_panels(cfg.crypto_universe)
    if not close_u.empty and close_u.shape[1] >= 8:
        mom = momentum_factor(close_u, lookback=mom_lookback, skip=7)
        w = quantile_long_weights(mom, month_end_dates(close_u.index), q=q, min_names=8)
        rows.append(row("alt-momentum topQ", run_backtest(open_u, close_u, w, cost)))
    return pd.DataFrame(rows)
