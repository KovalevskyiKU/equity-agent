"""Walk-forward test: does a model on the feature/regime cluster beat the basket OOS?

Fits a small ridge model on past data, predicts forward returns on the next
(embargoed) fold, turns the out-of-sample predictions into long-only weights, and
backtests them vs the equal-weight basket and SPY over the OOS period. The honest
payoff test — strictly out-of-sample (embargo = horizon, no look-ahead) and
regularised to limit overfit. No LLM / no API keys.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestConfig, run_backtest
from ..backtest.metrics import return_summary
from ..backtest.panels import load_price_panels
from ..backtest.strategy import buy_and_hold_equal, single_asset
from ..config import load_config
from ..risk.limits import cap_exposure
from ..signals.feature_store import load_bars, load_features
from .signal_eval import forward_return, information_coefficient
from .validation import PurgedWalkForwardSplit

logger = logging.getLogger("equity_agent")

_NON_FEATURES = {"dow", "month", "fwd_ret", "symbol", "date"}


def _build_panel(symbols: list[str], horizon: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        feats, bars = load_features(sym), load_bars(sym)
        if feats.empty or bars.empty:
            continue
        feats = feats.copy()
        feats.index = pd.to_datetime(feats.index)
        bars = bars.copy()
        bars.index = pd.to_datetime(bars.index)
        feats["fwd_ret"] = forward_return(bars["close"], horizon).reindex(feats.index)
        feats["symbol"] = sym
        feats["date"] = feats.index
        frames.append(feats)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge on standardised inputs (X'X normalised by n)."""
    n, f = x.shape
    a = x.T @ x / n + alpha * np.eye(f)
    b = x.T @ y / n
    return np.linalg.solve(a, b)


def run_walkforward_strategy(
    symbols: list[str] | None = None,
    *,
    horizon: int = 10,
    n_splits: int = 6,
    alpha: float = 0.1,
    max_weight: float = 0.34,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> dict[str, object]:
    cfg = load_config()
    syms = symbols or cfg.universe
    panel = _build_panel(syms, horizon)
    if panel.empty:
        return {}

    feat_cols = [c for c in panel.columns if c not in _NON_FEATURES]
    panel = panel.dropna(subset=[*feat_cols, "fwd_ret"])
    dates = np.array(sorted(panel["date"].unique()))

    splitter = PurgedWalkForwardSplit(n_splits=n_splits, embargo=horizon)
    preds: list[pd.DataFrame] = []
    for tr_idx, te_idx in splitter.split(len(dates)):
        tr = panel[panel["date"].isin(set(dates[tr_idx]))]
        te = panel[panel["date"].isin(set(dates[te_idx]))]
        if tr.empty or te.empty:
            continue
        x_tr = tr[feat_cols].to_numpy(dtype=float)
        y_tr = tr["fwd_ret"].to_numpy(dtype=float)
        mu, sd, ym = x_tr.mean(0), x_tr.std(0) + 1e-9, y_tr.mean()
        coef = _ridge((x_tr - mu) / sd, y_tr - ym, alpha)
        pred = ((te[feat_cols].to_numpy(dtype=float) - mu) / sd) @ coef + ym
        block = te[["date", "symbol", "fwd_ret"]].copy()
        block["pred"] = pred
        preds.append(block)

    if not preds:
        return {}
    pred_df = pd.concat(preds)
    ic, t_stat, n = information_coefficient(pred_df["pred"], pred_df["fwd_ret"])

    # Long-only weights: fully invest among names with positive predicted return
    # (cash when none), then apply per-name / gross caps.
    raw = pred_df.pivot_table(index="date", columns="symbol", values="pred").clip(lower=0.0)
    norm = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    weights = cap_exposure(norm, max_per_name=max_weight, max_gross=1.0)
    weights.index = [d.date() if hasattr(d, "date") else d for d in weights.index]

    open_u, close_u = load_price_panels(syms)
    cal = close_u.index[close_u.index.isin(weights.index)]
    bcfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)
    strat = run_backtest(open_u.loc[cal], close_u.loc[cal], weights.reindex(cal), bcfg)
    basket_w = buy_and_hold_equal(close_u.loc[cal])
    basket = run_backtest(open_u.loc[cal], close_u.loc[cal], basket_w, bcfg)
    ob, cb = load_price_panels([cfg.benchmark])
    bcal = cal[cal.isin(cb.index)]
    spy = run_backtest(ob.loc[bcal], cb.loc[bcal], single_asset(cb.loc[bcal], cfg.benchmark), bcfg)

    return {
        "oos_ic": ic,
        "oos_t": t_stat,
        "n_oos": n,
        "n_dates": len(cal),
        "strategy": return_summary(strat.returns),
        "basket": return_summary(basket.returns),
        "spy": return_summary(spy.returns),
    }
