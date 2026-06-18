"""Backtest a mechanical long/flat rule on the Kronos signal (no LLM, local).

At each rebalance date, take Kronos' P(up) per symbol (point-in-time), tilt long
proportional to max(0, P(up) - 0.5), then apply the risk exposure caps. Tests
whether Kronos carries edge as a plain rule — independent of the LLM agent.
Slow: one CPU model rollout per symbol per rebalance date.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..risk.limits import cap_exposure
from ..signals.feature_store import load_bars
from .engine import BacktestConfig, run_backtest
from .metrics import return_summary
from .panels import load_price_panels
from .strategy import buy_and_hold_equal, single_asset, vol_target_weights

logger = logging.getLogger("equity_agent")


def kronos_rule_weights(
    symbols: list[str],
    calendar: pd.DatetimeIndex,
    *,
    rebalance_days: int,
    max_weight: float,
    samples: int,
    horizon: int,
    lookback: int,
) -> pd.DataFrame:
    from ..signals.kronos_signal import KronosForecaster

    forecaster = KronosForecaster()  # load once, reuse for every rollout
    bars_by: dict[str, pd.DataFrame] = {}
    for s in symbols:
        b = load_bars(s)
        if not b.empty:
            b = b.copy()
            b.index = pd.to_datetime(b.index)
            bars_by[s] = b

    decision_dates = list(calendar[::rebalance_days])
    weights = pd.DataFrame(index=calendar, columns=symbols, dtype=float)
    for i, dt in enumerate(decision_dates, start=1):
        asof = pd.Timestamp(dt)
        for s in symbols:
            b = bars_by.get(s)
            if b is None:
                continue
            window = b[b.index <= asof].tail(lookback)
            if len(window) < lookback:
                continue
            sig = forecaster.signal(window, horizon=horizon, sample_count=samples)
            weights.loc[dt, s] = max(0.0, sig["k_p_up"] - 0.5)  # long tilt for P(up) > 0.5
        logger.info("  kronos-rule %s (%d/%d dates)", asof.date(), i, len(decision_dates))

    weights = weights.ffill().fillna(0.0)
    return cap_exposure(weights, max_per_name=max_weight, max_gross=1.0)


def run_kronos_backtest(
    symbols: list[str],
    benchmark: str,
    *,
    months: int = 12,
    rebalance_days: int = 21,
    max_weight: float = 0.20,
    samples: int = 10,
    horizon: int = 10,
    lookback: int = 256,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> dict[str, dict[str, float]]:
    open_u, close_u = load_price_panels(symbols)
    if close_u.empty:
        raise RuntimeError("No price data; run `eqa ingest` first.")
    idx = pd.to_datetime(close_u.index)
    end_ts = idx.max()
    start_ts = end_ts - pd.Timedelta(days=int(months * 31))
    cal = close_u.index[(idx >= start_ts) & (idx <= end_ts)]

    cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)
    kw = kronos_rule_weights(
        symbols, cal, rebalance_days=rebalance_days, max_weight=max_weight,
        samples=samples, horizon=horizon, lookback=lookback,
    )
    kr = run_backtest(open_u.loc[cal], close_u.loc[cal], kw, cfg)
    bk = run_backtest(open_u.loc[cal], close_u.loc[cal], buy_and_hold_equal(close_u.loc[cal]), cfg)
    vt = run_backtest(open_u.loc[cal], close_u.loc[cal], vol_target_weights(close_u.loc[cal]), cfg)
    ob, cb = load_price_panels([benchmark])
    bcal = cal[cal.isin(cb.index)]
    sp = run_backtest(ob.loc[bcal], cb.loc[bcal], single_asset(cb.loc[bcal], benchmark), cfg)

    return {
        "kronos": return_summary(kr.returns),
        "voltgt": return_summary(vt.returns),
        "basket": return_summary(bk.returns),
        "spy": return_summary(sp.returns),
    }
