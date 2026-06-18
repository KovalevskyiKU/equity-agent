"""Risk layer (Phase 4).

Sits between decision and execution. Position sizing (volatility targeting /
fractional Kelly), portfolio constraints (per-name, sector, gross exposure),
stops/trailing, earnings/event blackout, and a portfolio-level drawdown
circuit breaker (kill switch). Can veto or resize any decision.

Implemented so far: exposure limits (``cap_exposure``) and the drawdown circuit
breaker (in the backtest engine via ``BacktestConfig.max_drawdown_stop``).
"""

from .limits import cap_exposure

__all__ = ["cap_exposure"]
