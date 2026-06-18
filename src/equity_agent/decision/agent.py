"""Lean LLM decision agent (Phase 2).

One structured LLM call weighs the available signals for a symbol into a
long-only swing decision: a target portfolio weight in [0, max_weight] plus a
rationale. This is the chosen LLM-as-combiner, kept lean (one call per symbol
per decision) so it stays inside free-tier limits and can be backtested on a
sampled basis. The full TradingAgents multi-agent debate is a later swap behind
this same interface.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from ..llm import DEFAULT_MODEL, generate_structured

_VALID_ACTIONS = {"BUY", "HOLD", "SELL"}

_PROMPT = (
    "You are a disciplined LONG-ONLY daily-swing trader deciding a ~10-trading-day "
    "position in {symbol}. Use ONLY the signals provided (they are point-in-time, no "
    "future data).\n"
    "Research priors: realised-volatility signals (atr_14, vol_20, vix_level) carry a "
    "small but real edge (higher vol has preceded higher forward returns); "
    "momentum/MACD/RSI were noise on the full sample; the Kronos directional "
    "probability (k_p_up) is weak; news sentiment is short-lived. Individual signals "
    "are weak — be conservative and size modestly.\n"
    "Return: action (BUY = open/hold long, HOLD = keep, SELL = go flat), target_weight "
    "in [0, {max_weight}], confidence in [0, 1], and a one-sentence rationale.\n\n"
    "SIGNALS (JSON):\n{signals}"
)


class DecisionOut(BaseModel):
    action: str
    target_weight: float
    confidence: float
    rationale: str


def _normalize(action: str, target_weight: float, max_weight: float) -> tuple[str, float]:
    act = action.strip().upper()
    if act not in _VALID_ACTIONS:
        act = "HOLD"
    weight = max(0.0, min(float(max_weight), float(target_weight)))
    if act == "SELL":
        weight = 0.0
    return act, weight


def decide(
    bundle: dict[str, object], *, model: str = DEFAULT_MODEL, max_weight: float = 0.34
) -> DecisionOut:
    """Turn a point-in-time signal bundle into a long-only target-weight decision."""
    prompt = _PROMPT.format(
        symbol=bundle.get("symbol", "?"),
        max_weight=max_weight,
        signals=json.dumps(bundle, default=str),
    )
    out = generate_structured(prompt, DecisionOut, model=model)
    out.action, out.target_weight = _normalize(out.action, out.target_weight, max_weight)
    out.confidence = max(0.0, min(1.0, float(out.confidence)))
    return out
