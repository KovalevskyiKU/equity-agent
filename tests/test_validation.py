import numpy as np
import pandas as pd

from equity_agent.research.validation import (
    PurgedWalkForwardSplit,
    block_ic,
    non_overlapping_ic,
)


def test_non_overlapping_ic_subsamples() -> None:
    rng = np.random.default_rng(0)
    target = pd.Series(rng.normal(0, 1, 400))
    feature = target + pd.Series(rng.normal(0, 0.3, 400))
    r, _, n = non_overlapping_ic(feature, target, horizon=5)
    assert n == 80  # 400 / 5
    assert r > 0.5


def test_block_ic_returns_one_row_per_block() -> None:
    rng = np.random.default_rng(1)
    target = pd.Series(rng.normal(0, 1, 1000))
    feature = target + pd.Series(rng.normal(0, 0.5, 1000))
    out = block_ic(feature, target, n_blocks=8)
    assert len(out) == 8


def test_walkforward_no_leakage() -> None:
    splitter = PurgedWalkForwardSplit(n_splits=4, embargo=5)
    seen_test: list[int] = []
    for train_idx, test_idx in splitter.split(100):
        # embargo gap respected: training ends well before the test fold starts.
        assert test_idx.min() - train_idx.max() > 5
        # expanding train always starts at 0.
        assert train_idx.min() == 0
        seen_test.extend(test_idx.tolist())
    # test folds are disjoint and ordered.
    assert seen_test == sorted(seen_test)
    assert len(seen_test) == len(set(seen_test))
