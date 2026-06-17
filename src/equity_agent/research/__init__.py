"""Signal research (Phase 1 / cross-cutting).

Measures whether a signal carries predictive edge *before* it is wired into the
decision engine — the cheap gate in our layered validation. This is research,
not a trading backtest: no fees, sizing or execution (those belong to the
Phase 3 backtester). The same harness will score the Kronos and LLM signals as
they come online.
"""

from .signal_eval import evaluate, forward_return, information_coefficient, run

__all__ = ["evaluate", "forward_return", "information_coefficient", "run"]
