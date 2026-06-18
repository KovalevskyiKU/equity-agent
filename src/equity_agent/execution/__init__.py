"""Execution layer (Phase 4).

Broker abstraction over IBKR (paper -> live): order placement, fill
reconciliation, retries, idempotency. A clean interface so the broker can be
swapped without touching the decision or risk layers. (Alpaca was dropped — not
available in Ukraine.)
"""
