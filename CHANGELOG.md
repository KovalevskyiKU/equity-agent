# Changelog

Notable changes, newest first. Dates are when the work landed on `main`.

## 2026-06 — Phase 2/3: cross-sectional research, point-in-time, honest core

- **Universe → S&P 500** (503 names, config-driven) with batched yfinance ingest.
- **Cross-sectional factor research**: per-date IC, price factors (12-1 momentum,
  low-vol) and value/quality factors (earnings yield, ROE, margins) from
  **point-in-time** index membership and **point-in-time** fundamentals (Finnhub
  as-reported, lagged to filing date).
- **Verdict (honest null):** once survivorship bias is removed and returns are
  **dividend-adjusted (total-return)**, no factor — nor a sector-neutral
  multi-factor composite — beats the cap-weighted index (SPY) on a risk-adjusted
  basis. Survivorship bias was the dominant driver of the apparent edge.
- **Core now tracks SPY** (`config.core_strategy=spy`); the broad universe is for
  research only. Methodology locked in `docs/METHODOLOGY.md`; survivorship-biased
  commands print a NOTE.
- **Vol-target risk overlay** validated on SPY: better Sharpe/Calmar and ~half the
  drawdown (crash insurance), at a return cost — opt-in via `config.risk_overlay`.
- **Execution / ops:** IBKR adapter (`eqa live-run`, dry-run by default),
  monitoring (`eqa monitor`), lean daily cycle, Streamlit dashboard
  (`python app.py`) with predictions / %-profit / risk-overlay views, and a
  reproducible `eqa research-report`.

## Phase 0–1 — foundation & signals

- Repo, config, SQLAlchemy storage, idempotent daily-bar ingestion, CI.
- Causal feature store; IC/quantile research harness with non-overlapping
  correction; Kronos probabilistic signal + eval; purged walk-forward validation;
  performance metrics; LLM news risk-off gate; paper-trading loop.
