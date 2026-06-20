"""Daily paper-trading run: core-strategy weights -> paper broker rebalance."""

from __future__ import annotations

import logging

from ..backtest.panels import load_price_panels
from ..backtest.strategy import vol_target_weights
from ..config import load_config
from .paper_broker import rebalance

logger = logging.getLogger("equity_agent")


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

    With ``risk_off``, refresh recent news sentiment (cheap 8b model) and zero the
    weight of any name with strongly negative sentiment before rebalancing.
    """
    cfg = load_config()
    _, close_u = load_price_panels(cfg.universe)
    if close_u.empty:
        raise RuntimeError("No price data; run `eqa ingest` first.")

    latest_close = close_u.iloc[-1]
    prices = {k: float(v) for k, v in latest_close.items() if v == v}  # drop NaN
    weights_row = vol_target_weights(close_u).iloc[-1]
    target = {k: float(w) for k, w in weights_row.items() if w > 0 and k in prices}

    if risk_off:
        from datetime import UTC, datetime, timedelta

        from ..decision.risk_off import apply_risk_off_gate
        from ..signals.sentiment import fetch_score_store

        end = datetime.now(UTC).date()
        start = end - timedelta(days=news_days)
        for sym in cfg.universe:
            try:
                fetch_score_store(sym, start, end, model=sentiment_model, limit=score_limit)
            except Exception as e:  # noqa: BLE001 - news scoring is best-effort
                logger.warning("news scoring failed for %s: %s", sym, e)
        target, gated = apply_risk_off_gate(target)
        if gated:
            logger.info("risk-off gated (negative news): %s", gated)

    logger.info("Paper rebalance to %d names (core = vol-target)", len(target))
    return rebalance(
        target, prices, fee_bps=fee_bps, slippage_bps=slippage_bps, starting_cash=starting_cash
    )
