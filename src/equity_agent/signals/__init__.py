"""Signal layer (Phase 1).

Turns stored data into point-in-time features and per-source signals:

* feature engineering — causal technical / volatility / regime / calendar features
* Kronos adapter — probabilistic direction + confidence from sample dispersion
* LLM sentiment scorer — per-headline sentiment/impact, gated by ``published_at``

Output: a feature matrix written to the Parquet feature-store, consumed by both
the decision engine (live) and the backtester (research). No look-ahead allowed.
"""
