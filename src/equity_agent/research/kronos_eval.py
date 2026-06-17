"""Does Kronos add edge? Score its signal with the same IC harness as the features.

Generates the Kronos signal at many historical as-of dates (point-in-time: each
uses only bars up to that date) and correlates it with the realised forward
return. Compute-heavy — one model rollout per point — so this is an *indicative*
read on a single symbol/sample, not the final verdict (that's the Phase 3
walk-forward). The same numbers let us compare Kronos against the technical
features fairly, on the same horizon and metric.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..signals.feature_store import load_bars
from ..signals.kronos_signal import KronosForecaster
from .signal_eval import information_coefficient

logger = logging.getLogger("equity_agent")

_SIGNAL_COLS = ["k_p_up", "k_exp_ret", "k_ret_std"]


def evaluate_kronos(
    symbol: str,
    horizon: int = 10,
    points: int = 60,
    lookback: int = 256,
    sample_count: int = 12,
    step: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> pd.DataFrame:
    """Return a per-point frame: as-of date, forward return, and Kronos signals."""
    bars = load_bars(symbol)
    if bars.empty:
        return pd.DataFrame()

    closes = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    last_asof = n - horizon - 1
    first_asof = lookback - 1
    positions = list(range(last_asof, first_asof, -step))[:points]
    if not positions:
        logger.warning("[%s] not enough history for points (have %d bars)", symbol, n)
        return pd.DataFrame()

    forecaster = KronosForecaster()
    rows: list[dict[str, object]] = []
    for k, i in enumerate(positions, start=1):
        window = bars.iloc[i - lookback + 1 : i + 1]
        sig = forecaster.signal(
            window,
            horizon=horizon,
            temperature=temperature,
            top_p=top_p,
            sample_count=sample_count,
        )
        rows.append(
            {
                "asof": bars.index[i],
                "fwd_ret": float(closes[i + horizon] / closes[i] - 1.0),
                **sig,
            }
        )
        if k % 10 == 0:
            logger.info("  [%s] %d/%d points", symbol, k, len(positions))

    return pd.DataFrame(rows).sort_values("asof").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """IC of each Kronos signal vs the forward return."""
    out: list[dict[str, object]] = []
    for col in _SIGNAL_COLS:
        r, t, n = information_coefficient(df[col], df["fwd_ret"])
        out.append({"signal": col, "ic": r, "t_stat": t, "n": n})
    return pd.DataFrame(out)
