"""Build and read the Parquet feature store from stored daily bars.

One Parquet file per symbol under ``data/features/``. Columnar reads keep the
backtester and (later) the live decision loop fast. Market-context columns
(benchmark return, VIX level/z) are joined here, where we have cross-symbol
access, rather than inside the per-symbol :func:`build_features`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from ..config import PROJECT_ROOT, load_config
from ..storage.db import session_scope
from ..storage.models import DailyBar
from .features import build_features

logger = logging.getLogger("equity_agent")


def _safe_name(symbol: str) -> str:
    return symbol.replace("^", "_")


def _features_dir() -> Path:
    cfg = load_config()
    d = PROJECT_ROOT / cfg.data_dir / "features"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_bars(symbol: str) -> pd.DataFrame:
    """Load all stored daily bars for a symbol, indexed by date (ascending)."""
    with session_scope() as session:
        rows = session.scalars(
            select(DailyBar).where(DailyBar.symbol == symbol).order_by(DailyBar.ts)
        ).all()
        data = [
            {
                "ts": r.ts,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "adj_close": r.adj_close,
                "volume": r.volume,
            }
            for r in rows
        ]
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).set_index("ts")


def build_market_context() -> pd.DataFrame:
    """Benchmark daily return + VIX level/z, indexed by date. Causal (no shift forward)."""
    cfg = load_config()
    ctx = pd.DataFrame()

    bench = load_bars(cfg.benchmark)
    if not bench.empty:
        ctx = pd.DataFrame({"mkt_ret_1": bench["close"].pct_change()})

    vix_sym = next((s for s in cfg.regime_symbols if "VIX" in s.upper()), None)
    if vix_sym:
        vix = load_bars(vix_sym)
        if not vix.empty:
            lvl = vix["close"]
            vix_df = pd.DataFrame(
                {
                    "vix_level": lvl,
                    "vix_z": (lvl - lvl.rolling(20).mean()) / lvl.rolling(20).std(),
                }
            )
            ctx = vix_df if ctx.empty else ctx.join(vix_df, how="outer")

    # Macro regime (yfinance proxies, no API key): 10y yield level + 20d change, USD 20d return.
    tnx = load_bars("^TNX")
    if not tnx.empty:
        y = tnx["close"]
        tnx_df = pd.DataFrame({"tnx_level": y, "tnx_chg_20": y - y.shift(20)})
        ctx = tnx_df if ctx.empty else ctx.join(tnx_df, how="outer")

    usd = load_bars("DX-Y.NYB")
    if not usd.empty:
        usd_df = pd.DataFrame({"usd_ret_20": usd["close"].pct_change(20)})
        ctx = usd_df if ctx.empty else ctx.join(usd_df, how="outer")

    # Cross-asset / credit regime (yfinance ETFs, no key):
    # credit risk appetite = high-yield minus investment-grade 20d return.
    hyg, lqd = load_bars("HYG"), load_bars("LQD")
    if not hyg.empty and not lqd.empty:
        spread = hyg["close"].pct_change(20) - lqd["close"].pct_change(20)
        cr_df = pd.DataFrame({"credit_ret_20": spread})
        ctx = cr_df if ctx.empty else ctx.join(cr_df, how="outer")

    tlt = load_bars("TLT")
    if not tlt.empty:
        tlt_df = pd.DataFrame({"tlt_ret_20": tlt["close"].pct_change(20)})
        ctx = tlt_df if ctx.empty else ctx.join(tlt_df, how="outer")

    gld = load_bars("GLD")
    if not gld.empty:
        gld_df = pd.DataFrame({"gld_ret_20": gld["close"].pct_change(20)})
        ctx = gld_df if ctx.empty else ctx.join(gld_df, how="outer")

    # FRED macro (if ingested): 2s10s yield-curve slope, 10y real yield, HY OAS spread.
    dgs2, dgs10 = load_bars("DGS2"), load_bars("DGS10")
    if not dgs2.empty and not dgs10.empty:
        slope = dgs10["close"] - dgs2["close"]
        slope_df = pd.DataFrame({"slope_2s10s": slope})
        ctx = slope_df if ctx.empty else ctx.join(slope_df, how="outer")

    rry = load_bars("DFII10")
    if not rry.empty:
        rry_df = pd.DataFrame({"real_yield_10y": rry["close"]})
        ctx = rry_df if ctx.empty else ctx.join(rry_df, how="outer")

    oas = load_bars("BAMLH0A0HYM2")
    if not oas.empty:
        o = oas["close"]
        oas_df = pd.DataFrame({"hy_oas": o, "hy_oas_chg_20": o - o.shift(20)})
        ctx = oas_df if ctx.empty else ctx.join(oas_df, how="outer")

    return ctx.sort_index().ffill()


def build_symbol_features(symbol: str, context: pd.DataFrame | None = None) -> pd.DataFrame:
    bars = load_bars(symbol)
    if bars.empty:
        return pd.DataFrame()
    feats = build_features(bars)
    if context is not None and not context.empty:
        feats = feats.join(context, how="left")
    return feats


def build_feature_store(symbols: list[str] | None = None) -> dict[str, int]:
    """Compute features for each symbol and write them to Parquet. Returns row counts."""
    cfg = load_config()
    syms = symbols or cfg.all_data_symbols
    context = build_market_context()
    out_dir = _features_dir()
    counts: dict[str, int] = {}

    for symbol in syms:
        feats = build_symbol_features(symbol, context)
        if feats.empty:
            logger.warning("[%s] no bars -> no features (run `eqa ingest` first)", symbol)
            counts[symbol] = 0
            continue
        path = out_dir / f"{_safe_name(symbol)}.parquet"
        feats.to_parquet(path)
        counts[symbol] = len(feats)
        logger.info("[%s] %d feature rows -> %s", symbol, len(feats), path.name)

    return counts


def load_features(symbol: str) -> pd.DataFrame:
    """Read a symbol's feature frame back from the store."""
    path = _features_dir() / f"{_safe_name(symbol)}.parquet"
    return pd.read_parquet(path)
