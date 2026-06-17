"""Risk layer (Phase 4).

Sits between decision and execution. Position sizing (volatility targeting /
fractional Kelly), portfolio constraints (per-name, sector, gross exposure),
stops/trailing, earnings/event blackout, and a portfolio-level drawdown
circuit breaker (kill switch). Can veto or resize any decision.
"""
