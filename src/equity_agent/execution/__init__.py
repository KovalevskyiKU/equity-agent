"""Execution layer (Phase 4).

Broker abstraction over Alpaca (paper -> live): order placement, fill
reconciliation, retries, idempotency. A clean interface so the broker can be
swapped (Alpaca -> IBKR) without touching the decision or risk layers.
"""
