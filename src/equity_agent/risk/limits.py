"""Exposure limits — pure target-weight transforms (long-only, no leverage)."""

from __future__ import annotations

import pandas as pd


def cap_exposure(
    weights: pd.DataFrame, *, max_per_name: float = 0.34, max_gross: float = 1.0
) -> pd.DataFrame:
    """Clamp each name to ``max_per_name`` and scale rows down to ``max_gross`` gross.

    Long-only (negatives floored to 0). Rows under the gross cap are unchanged.
    """
    w = weights.clip(lower=0.0, upper=max_per_name)
    gross = w.sum(axis=1)
    scale = (max_gross / gross.replace(0.0, max_gross)).clip(upper=1.0)
    return w.mul(scale, axis=0)
