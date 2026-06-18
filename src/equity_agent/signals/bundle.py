"""Assemble a point-in-time signal bundle for the decision agent.

Pulls the latest causal feature row, the most recent daily sentiment (if any),
and optionally a fresh Kronos signal — everything the LLM needs to decide, with
no future data.
"""

from __future__ import annotations

import pandas as pd

from .feature_store import load_bars, load_features
from .sentiment import get_daily_sentiment

_FEATURE_KEYS = [
    "ret_1",
    "roc_5",
    "roc_20",
    "rsi_14",
    "macd",
    "px_sma20",
    "atr_14",
    "vol_20",
    "vix_level",
]


def build_bundle(
    symbol: str,
    *,
    with_kronos: bool = True,
    kronos_samples: int = 20,
    kronos_horizon: int = 10,
    lookback: int = 256,
) -> dict[str, object]:
    feats = load_features(symbol)
    if feats.empty:
        raise RuntimeError(f"No features for {symbol}; run `eqa features` first.")

    last = feats.iloc[-1]
    bundle: dict[str, object] = {"symbol": symbol, "asof": str(feats.index[-1])}
    bundle["features"] = {
        k: round(float(last[k]), 4)
        for k in _FEATURE_KEYS
        if k in feats.columns and pd.notna(last[k])
    }

    sentiment = get_daily_sentiment(symbol)
    if not sentiment.empty:
        bundle["sentiment_recent"] = round(float(sentiment.iloc[-1]), 3)

    if with_kronos:
        bars = load_bars(symbol)
        if len(bars) >= lookback:
            from .kronos_signal import KronosForecaster

            sig = KronosForecaster().signal(
                bars.tail(lookback), horizon=kronos_horizon, sample_count=kronos_samples
            )
            bundle["kronos"] = {k: round(float(v), 4) for k, v in sig.items()}

    return bundle
