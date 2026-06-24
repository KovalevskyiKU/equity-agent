"""Information-coefficient + quantile analysis of features vs forward returns.

Edge is summarised per feature by:

* **IC** — Spearman rank correlation between the feature at *t* and the forward
  return over *t -> t+h*, pooled across symbols and dates. Sign = direction,
  magnitude = strength. ``t_stat`` flags whether it's distinguishable from zero.
* **Q5-Q1 spread** — mean forward return of the top feature quintile minus the
  bottom. A monotonic, wide spread is a tradable signature.

``forward_return`` is the prediction *target*. It is the only forward-looking
quantity here and is never used as an input feature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, load_config
from ..signals.feature_store import load_bars, load_features
from ..signals.features import FEATURE_COLUMNS

logger = logging.getLogger("equity_agent")

# Calendar features are categorical — not meaningful under rank correlation.
_NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in ("dow", "month")]
_CONTEXT_FEATURES = [
    "mkt_ret_1", "vix_level", "vix_z", "tnx_level", "tnx_chg_20", "usd_ret_20",
    "credit_ret_20", "tlt_ret_20", "gld_ret_20",
    "slope_2s10s", "real_yield_10y", "hy_oas", "hy_oas_chg_20",
]
_MIN_OBS = 30


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return over the next ``horizon`` bars: close[t+h] / close[t] - 1 (the target)."""
    return close.shift(-horizon) / close - 1.0


def information_coefficient(feature: pd.Series, target: pd.Series) -> tuple[float, float, int]:
    """Spearman IC, its t-stat, and the sample size, on the aligned non-null pairs."""
    pair = pd.concat([feature, target], axis=1).dropna()
    n = len(pair)
    if n < _MIN_OBS:
        return float("nan"), float("nan"), n
    # Spearman == Pearson correlation of the ranks (numpy only — avoids a scipy dep).
    ranks = pair.rank()
    r = float(ranks.iloc[:, 0].corr(ranks.iloc[:, 1]))
    if not np.isfinite(r) or abs(r) >= 1.0:
        return r, float("nan"), n
    t = r * np.sqrt((n - 2) / (1.0 - r * r))
    return r, float(t), n


def build_panel(symbols: list[str], horizon: int) -> pd.DataFrame:
    """Stack each symbol's features + forward-return target into one pooled panel."""
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        feats = load_features(sym)
        bars = load_bars(sym)
        if feats.empty or bars.empty:
            continue
        df = feats.copy()
        df["fwd_ret"] = forward_return(bars["close"], horizon).reindex(df.index)
        df["symbol"] = sym
        frames.append(df)
    return pd.concat(frames) if frames else pd.DataFrame()


def evaluate(symbols: list[str], horizon: int) -> pd.DataFrame:
    """Per-feature IC + quantile spread over the pooled panel, ranked by |IC|."""
    panel = build_panel(symbols, horizon)
    if panel.empty:
        return pd.DataFrame()

    target = panel["fwd_ret"]
    # Non-overlapping view: every `horizon`-th row per symbol, so forward windows
    # don't overlap and the t-stat isn't inflated by autocorrelation.
    noov = panel.groupby("symbol", group_keys=False).apply(lambda g: g.iloc[::horizon])
    noov_target = noov["fwd_ret"]

    feature_cols = [c for c in (_NUMERIC_FEATURES + _CONTEXT_FEATURES) if c in panel.columns]
    rows: list[dict[str, float | str | int]] = []

    for col in feature_cols:
        r, t, n = information_coefficient(panel[col], target)
        nr, nt, nn = information_coefficient(noov[col], noov_target)

        q_spread = float("nan")
        sub = panel[[col, "fwd_ret"]].dropna()
        if len(sub) >= 100 and sub[col].nunique() > 5:
            try:
                buckets = pd.qcut(sub[col], 5, labels=False, duplicates="drop")
                means = sub["fwd_ret"].groupby(buckets).mean()
                q_spread = float(means.iloc[-1] - means.iloc[0])
            except ValueError:
                pass

        rows.append(
            {
                "feature": col,
                "ic": r,
                "t_stat": t,
                "noov_ic": nr,
                "noov_t": nt,
                "q5_q1_spread": q_spread,
                "n": n,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("noov_ic", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def run(
    symbols: list[str] | None = None,
    horizons: tuple[int, ...] = (1, 5, 10),
) -> dict[int, pd.DataFrame]:
    """Evaluate the traded universe across horizons; write CSV reports; return tables."""
    cfg = load_config()
    syms = symbols or cfg.universe  # traded names; benchmark/VIX enter as context columns
    reports_dir = PROJECT_ROOT / cfg.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, pd.DataFrame] = {}
    for h in horizons:
        table = evaluate(syms, h)
        results[h] = table
        if not table.empty:
            table.to_csv(reports_dir / f"signal_eval_h{h}.csv", index=False)
            logger.info("[h=%d] evaluated %d features -> signal_eval_h%d.csv", h, len(table), h)
    return results
