"""Backtest & evaluation (Phase 3).

Two complementary engines:

* vectorbt — fast vectorized signal-level backtest (no LLM) over full history,
  to check whether features + Kronos carry edge before spending LLM tokens.
* event-driven simulator — replays day-by-day with point-in-time data and the
  full agent-in-the-loop, modelling fees, slippage and next-open execution.

Validation: walk-forward OOS, metrics vs SPY buy-and-hold, and edge-vs-luck
tests (Monte-Carlo permutation, deflated Sharpe).
"""
