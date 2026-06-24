"""Daily paper-trading run: core-strategy weights -> paper broker rebalance."""

from __future__ import annotations

import logging

import pandas as pd

from ..backtest.panels import load_price_panels
from ..backtest.strategy import buy_and_hold_equal, vol_target_weights
from ..config import load_config
from .paper_broker import rebalance

logger = logging.getLogger("equity_agent")


def compute_core_target(
    core_strategy: str,
    close_u: pd.DataFrame,
    close_b: pd.DataFrame,
    benchmark: str,
    exposure: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Latest core target weights + prices for the chosen core.

    * ``spy`` — hold the cap-weight benchmark (the honest core: research found that
      nothing beats it risk-adjusted once survivorship is removed).
    * ``vol_target`` / ``equal_weight`` — run over the (broad) ``universe`` panel.

    ``exposure`` (<= 1) scales all weights down toward cash — the hook for the
    vol-target risk overlay; 1.0 leaves the core fully invested.
    """
    if core_strategy == "spy":
        px = float(close_b[benchmark].iloc[-1])
        target = {benchmark: 1.0}
        prices = {benchmark: px}
    else:
        latest = close_u.iloc[-1]
        prices = {k: float(v) for k, v in latest.items() if v == v}  # drop NaN
        if core_strategy == "vol_target":
            row = vol_target_weights(close_u).iloc[-1]
        else:  # equal_weight
            row = buy_and_hold_equal(close_u).iloc[-1]
        target = {k: float(w) for k, w in row.items() if w > 0 and k in prices}

    if exposure < 1.0:
        target = {k: w * exposure for k, w in target.items()}
    return target, prices


def run_paper(
    *,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
    starting_cash: float = 100_000.0,
    risk_off: bool = False,
    news_days: int = 3,
    score_limit: int = 8,
    sentiment_model: str = "llama-3.1-8b-instant",
) -> dict[str, float]:
    """Compute the core target weights from latest data and rebalance the paper book.

    The core is selected by ``config.core_strategy`` (default ``spy``). With
    ``risk_off``, refresh recent news sentiment (cheap 8b model) and zero the weight
    of any held name with strongly negative sentiment before rebalancing.
    """
    cfg = load_config()
    _, close_b = load_price_panels([cfg.benchmark])
    close_u = pd.DataFrame()
    if cfg.core_strategy != "spy":
        _, close_u = load_price_panels(cfg.universe)

    need_universe = cfg.core_strategy != "spy"
    if (need_universe and close_u.empty) or close_b.empty:
        raise RuntimeError("No price data; run `eqa ingest` first.")

    exposure = 1.0
    if cfg.risk_overlay == "vol_target":
        from ..risk.overlay import vol_target_exposure

        spy_returns = close_b[cfg.benchmark].pct_change().dropna()
        exposure = vol_target_exposure(spy_returns, target_vol=cfg.risk_target_vol)
        logger.info("risk overlay (vol_target): exposure=%.2f", exposure)

    target, prices = compute_core_target(
        cfg.core_strategy, close_u, close_b, cfg.benchmark, exposure=exposure
    )

    if risk_off:
        from datetime import UTC, datetime, timedelta

        from ..decision.risk_off import apply_risk_off_gate
        from ..signals.sentiment import fetch_score_store

        end = datetime.now(UTC).date()
        start = end - timedelta(days=news_days)
        for sym in list(target):
            try:
                fetch_score_store(sym, start, end, model=sentiment_model, limit=score_limit)
            except Exception as e:  # noqa: BLE001 - news scoring is best-effort
                logger.warning("news scoring failed for %s: %s", sym, e)
        target, gated = apply_risk_off_gate(target)
        if gated:
            logger.info("risk-off gated (negative news): %s", gated)

    logger.info("Paper rebalance to %d names (core = %s)", len(target), cfg.core_strategy)
    return rebalance(
        target, prices, fee_bps=fee_bps, slippage_bps=slippage_bps, starting_cash=starting_cash
    )
