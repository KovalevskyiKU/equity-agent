"""Decision engine (Phase 2).

Wraps TauricResearch/TradingAgents: analyst team -> bull/bear debate -> trader
-> risk -> portfolio manager. The Kronos signal and engineered features are
injected as a quant/technical analyst input. Emits a persisted Decision row
(action, conviction, full rationale) per symbol per day.
"""
