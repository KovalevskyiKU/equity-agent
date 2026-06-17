"""Validation utilities that fix the pitfalls the Kronos eval exposed.

The overlapping-window inflation (IC 0.45 collapsing to 0.17 once windows no
longer overlapped) is the motivating example. These tools make the honest
version routine:

* ``non_overlapping_ic`` — subsample every ``horizon`` rows so forward windows
  don't overlap, then measure IC.
* ``block_ic`` / ``ic_stability`` — IC per contiguous time block, so we can see
  whether a signal is steady or driven by one regime (IC information ratio +
  sign-consistency).
* ``PurgedWalkForwardSplit`` — expanding-window train/test folds with an embargo
  gap, for fitting a composite/meta-model out-of-sample later (Phase 3).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from .signal_eval import information_coefficient


def non_overlapping_ic(
    feature: pd.Series, target: pd.Series, horizon: int
) -> tuple[float, float, int]:
    """IC on a subsample taken every ``horizon`` rows (removes window overlap)."""
    return information_coefficient(feature.iloc[::horizon], target.iloc[::horizon])


def block_ic(feature: pd.Series, target: pd.Series, n_blocks: int = 10) -> pd.DataFrame:
    """IC within each of ``n_blocks`` contiguous time blocks."""
    pair = pd.concat([feature, target], axis=1).dropna()
    bounds = np.linspace(0, len(pair), n_blocks + 1).astype(int)
    rows: list[dict[str, float]] = []
    for i in range(n_blocks):
        block = pair.iloc[bounds[i] : bounds[i + 1]]
        r, t, n = information_coefficient(block.iloc[:, 0], block.iloc[:, 1])
        rows.append({"block": float(i), "ic": r, "t_stat": t, "n": float(n)})
    return pd.DataFrame(rows)


def ic_stability(feature: pd.Series, target: pd.Series, n_blocks: int = 10) -> dict[str, float]:
    """Summarise IC steadiness across blocks: mean, information ratio, sign consistency."""
    blocks = block_ic(feature, target, n_blocks)
    ic = blocks["ic"].dropna()
    if ic.empty:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "sign_consistency": float("nan")}
    mean_ic = float(ic.mean())
    sd = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    dominant_sign = np.sign(mean_ic) if mean_ic != 0 else 1.0
    return {
        "mean_ic": mean_ic,
        "ic_ir": mean_ic / sd if sd and not np.isnan(sd) else float("nan"),
        "sign_consistency": float((np.sign(ic) == dominant_sign).mean()),
    }


class PurgedWalkForwardSplit:
    """Expanding-window walk-forward splitter with an embargo gap.

    Train is everything up to ``embargo`` rows before each test fold, so a
    forward-looking target of length ``embargo`` cannot leak training labels
    into the test window.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 10) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        fold = n_samples // (self.n_splits + 1)
        if fold == 0:
            raise ValueError("not enough samples for the requested number of splits")
        for k in range(1, self.n_splits + 1):
            test_start = k * fold
            test_end = (k + 1) * fold if k < self.n_splits else n_samples
            train_end = max(0, test_start - self.embargo)
            if train_end == 0 or test_start >= test_end:
                continue
            yield np.arange(0, train_end), np.arange(test_start, test_end)
