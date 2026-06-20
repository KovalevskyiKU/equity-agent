"""LLM news risk-off gate — the narrow role for the LLM after the pivot.

Cut a name's target weight to zero (move to cash) when its recent news sentiment
is strongly negative. This is risk management, not allocation: the gate only ever
*reduces* exposure, it never picks or sizes positions.
"""

from __future__ import annotations

import logging

from ..signals.sentiment import get_daily_sentiment

logger = logging.getLogger("equity_agent")


def apply_risk_off_gate(
    weights: dict[str, float], *, threshold: float = -0.3
) -> tuple[dict[str, float], list[str]]:
    """Zero the weight of any name whose latest daily sentiment is below ``threshold``.

    Returns (gated_weights, gated_symbols). Cut weight goes to cash (not redistributed).
    """
    gated: list[str] = []
    out = dict(weights)
    for sym in list(out):
        series = get_daily_sentiment(sym)
        if not series.empty and float(series.iloc[-1]) < threshold:
            out[sym] = 0.0
            gated.append(sym)
    return out, gated
