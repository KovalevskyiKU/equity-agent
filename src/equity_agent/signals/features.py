"""Causal feature engineering.

Every feature value at date *t* is computed from bars up to and including *t*
only — never from future bars. All operations are backward-looking (rolling
windows, recursive EWMs, shifts, cumulative sums), which is what makes the
feature store safe to use in a point-in-time backtest. This invariant is
enforced by ``tests/test_features.py::test_features_are_causal``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Feature columns produced by :func:`build_features` (market-context columns are
# added separately by the feature store, which has cross-symbol access).
FEATURE_COLUMNS = [
    "ret_1",
    "log_ret_1",
    "roc_5",
    "roc_10",
    "roc_20",
    "rsi_14",
    "macd",
    "macd_hist",
    "px_sma20",
    "px_sma50",
    "sma20_sma50",
    "bb_pos",
    "bb_width",
    "vol_20",
    "atr_14",
    "vol_z",
    "obv_z",
    "dow",
    "month",
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol causal features from an OHLCV DataFrame indexed by date."""
    df = bars.sort_index()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    log_ret = np.log(close / close.shift(1))

    f = pd.DataFrame(index=df.index)

    # Returns / momentum
    f["ret_1"] = close.pct_change()
    f["log_ret_1"] = log_ret
    f["roc_5"] = close / close.shift(5) - 1.0
    f["roc_10"] = close / close.shift(10) - 1.0
    f["roc_20"] = close / close.shift(20) - 1.0
    f["rsi_14"] = _rsi(close, 14)

    # MACD (price-normalised so it's comparable across symbols/price levels)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = macd / close
    f["macd_hist"] = (macd - macd_signal) / close

    # Trend
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    f["px_sma20"] = close / sma20 - 1.0
    f["px_sma50"] = close / sma50 - 1.0
    f["sma20_sma50"] = sma20 / sma50 - 1.0

    # Volatility
    std20 = close.rolling(20).std()
    f["bb_pos"] = (close - sma20) / (2.0 * std20)
    f["bb_width"] = (4.0 * std20) / sma20
    f["vol_20"] = log_ret.rolling(20).std() * np.sqrt(TRADING_DAYS)
    f["atr_14"] = _atr(high, low, close, 14) / close

    # Volume
    f["vol_z"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    f["obv_z"] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std()

    # Calendar (known at the bar's date — causal)
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    f["dow"] = idx.dayofweek
    f["month"] = idx.month

    return f[FEATURE_COLUMNS]
