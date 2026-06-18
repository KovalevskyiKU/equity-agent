"""Backtest the lean LLM decision agent on a sampled, recent window.

Calls the agent on a fixed cadence (one call per symbol per rebalance date) with
point-in-time signals, builds a target-weight matrix, and runs it through the
engine vs the benchmark. Deliberately sampled/recent to stay inside Gemini
free-tier limits — an indicative read, not the final walk-forward verdict.
Historical sentiment is unavailable (Finnhub recent-only), so by default this
validates the features(+optional Kronos) combination, not sentiment.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from ..decision.agent import decide_portfolio
from ..signals.bundle import build_bundle
from .engine import BacktestConfig, run_backtest
from .metrics import return_summary
from .panels import load_price_panels
from .strategy import buy_and_hold_equal, single_asset, vol_target_weights

logger = logging.getLogger("equity_agent")


@dataclass
class LLMBacktestReport:
    strategy: dict[str, float]
    voltarget: dict[str, float]
    basket: dict[str, float]
    benchmark: dict[str, float]
    n_calls: int
    n_decision_dates: int
    strategy_equity: pd.Series
    benchmark_equity: pd.Series


def build_llm_weights(
    symbols: list[str],
    calendar: pd.DatetimeIndex,
    *,
    rebalance_days: int,
    max_weight: float,
    model: str,
    with_kronos: bool,
    with_sentiment: bool,
    delay: float,
) -> tuple[pd.DataFrame, int]:
    decision_dates = list(calendar[::rebalance_days])
    weights = pd.DataFrame(index=calendar, columns=symbols, dtype=float)
    calls = 0
    for i, dt in enumerate(decision_dates, start=1):
        asof = dt.date() if isinstance(dt, pd.Timestamp) else dt
        bundles: dict[str, dict[str, object]] = {}
        for sym in symbols:
            try:
                bundles[sym] = build_bundle(
                    sym, asof=asof, with_kronos=with_kronos, with_sentiment=with_sentiment
                )
            except Exception as e:  # noqa: BLE001 - skip a symbol with no data on this date
                logger.warning("bundle failed for %s @ %s: %s", sym, asof, e)
        if not bundles:
            continue
        try:
            # One batched LLM call for all symbols on this date (3x fewer calls).
            for sym, out in decide_portfolio(bundles, model=model, max_weight=max_weight).items():
                if sym in weights.columns:
                    weights.loc[dt, sym] = out.target_weight
        except Exception as e:  # noqa: BLE001 - a failed date is forward-filled, not fatal
            logger.warning("portfolio decide failed @ %s: %s", asof, e)
        calls += 1
        if delay:
            time.sleep(delay)
        logger.info("  decided %s (%d/%d dates)", asof, i, len(decision_dates))
    return weights.ffill().fillna(0.0), calls


def run_llm_backtest(
    symbols: list[str],
    benchmark: str,
    *,
    months: int = 6,
    end: str | None = None,
    rebalance_days: int = 5,
    max_weight: float = 0.34,
    model: str = "llama-3.3-70b-versatile",
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
    with_kronos: bool = False,
    with_sentiment: bool = False,
    delay: float = 4.0,
) -> LLMBacktestReport:
    open_px, close_px = load_price_panels(symbols)
    if close_px.empty:
        raise RuntimeError("No price data; run `eqa ingest` first.")

    idx_dt = pd.to_datetime(close_px.index)
    end_ts = pd.Timestamp(end) if end else idx_dt.max()
    start_ts = end_ts - pd.Timedelta(days=int(months * 31))
    cal = close_px.index[(idx_dt >= start_ts) & (idx_dt <= end_ts)]
    logger.info(
        "LLM backtest: %d symbols, %d trading days, rebalance every %d",
        len(symbols), len(cal), rebalance_days,
    )

    weights, calls = build_llm_weights(
        symbols, cal, rebalance_days=rebalance_days, max_weight=max_weight,
        model=model, with_kronos=with_kronos, with_sentiment=with_sentiment, delay=delay,
    )
    cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)
    strat_res = run_backtest(open_px.loc[cal], close_px.loc[cal], weights, cfg)
    basket_res = run_backtest(
        open_px.loc[cal], close_px.loc[cal], buy_and_hold_equal(close_px.loc[cal]), cfg
    )
    voltarget_res = run_backtest(
        open_px.loc[cal],
        close_px.loc[cal],
        vol_target_weights(close_px.loc[cal], max_weight=max_weight),
        cfg,
    )

    ob, cb = load_price_panels([benchmark])
    bcal = cal[cal.isin(cb.index)]
    bench_res = run_backtest(ob.loc[bcal], cb.loc[bcal], single_asset(cb.loc[bcal], benchmark), cfg)

    return LLMBacktestReport(
        strategy=return_summary(strat_res.returns),
        voltarget=return_summary(voltarget_res.returns),
        basket=return_summary(basket_res.returns),
        benchmark=return_summary(bench_res.returns),
        n_calls=calls,
        n_decision_dates=len(list(cal[::rebalance_days])),
        strategy_equity=strat_res.equity,
        benchmark_equity=bench_res.equity,
    )
