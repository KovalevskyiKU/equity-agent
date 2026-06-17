from datetime import date, timedelta

import numpy as np
import pandas as pd

from equity_agent.signals.features import FEATURE_COLUMNS, build_features


def _make_bars(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0.0, 0.01, n))
    low = close * (1.0 - rng.uniform(0.0, 0.01, n))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    volume = rng.uniform(1e6, 2e6, n)
    idx = [date(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_build_features_columns_and_warmup() -> None:
    feats = build_features(_make_bars(150))
    assert list(feats.columns) == FEATURE_COLUMNS
    # After the longest warmup window (SMA-50), core features must be populated.
    core = ["ret_1", "rsi_14", "macd", "px_sma50", "bb_pos", "vol_20", "atr_14", "vol_z"]
    assert feats.iloc[60:][core].notna().all().all()


def test_features_are_causal() -> None:
    """Appending future bars must not change any earlier feature value."""
    bars = _make_bars(150)
    full = build_features(bars)
    truncated = build_features(bars.iloc[:80])

    common = full.index[40:80]
    pd.testing.assert_frame_equal(full.loc[common], truncated.loc[common])
