"""Daily paper-trading run: core-strategy weights -> paper broker rebalance."""

from __future__ import annotations

import logging

from ..backtest.panels import load_price_panels
from ..backtest.strategy import vol_target_weights
from ..config import load_config
from .paper_broker import rebalance

logger = logging.getLogger("equity_agent")


def run_paper(
    *, fee_bps: float = 1.0, slippage_bps: float = 5.0, starting_cash: float = 100_000.0
) -> dict[str, float]:
    """Compute the core target weights from latest data and rebalance the paper book."""
    cfg = load_config()
    _, close_u = load_price_panels(cfg.universe)
    if close_u.empty:
        raise RuntimeError("No price data; run `eqa ingest` first.")

    latest_close = close_u.iloc[-1]
    prices = {k: float(v) for k, v in latest_close.items() if v == v}  # drop NaN
    weights_row = vol_target_weights(close_u).iloc[-1]
    target = {k: float(w) for k, w in weights_row.items() if w > 0 and k in prices}

    logger.info("Paper rebalance to %d names (core = vol-target)", len(target))
    return rebalance(
        target, prices, fee_bps=fee_bps, slippage_bps=slippage_bps, starting_cash=starting_cash
    )
