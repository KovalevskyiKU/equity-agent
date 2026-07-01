# Changelog

Notable changes, newest first. Dates are when the work landed on `main`.

## 2026-07 — Trading web app (FastAPI + React cockpit)

- **Backend API** (`src/equity_agent/api/`, `eqa serve`): instruments, OHLC bars,
  per-symbol signals, portfolio/P&L, trades, manual orders, and a `/ws/portfolio`
  WebSocket. Reuses the research/execution core.
- **Manual paper orders** (`paper_broker.place_order`): single market fill with
  fees/slippage, no shorting.
- **Live orders** routed to IBKR/Binance behind a two-step confirm (preview →
  `confirm:true` transmits); nothing live is sent without confirmation.
- **React/Vite cockpit** (`frontend/`): watchlist (equities+crypto), TradingView
  candlestick chart with SMA overlays, signal bar, live portfolio + equity curve,
  order ticket with venue selector, open-orders panel, recent trades.
- **Limit + stop orders**: paper resting orders (`PendingOrder`, kind limit|stop)
  that fill when the last price crosses the trigger (limit = favourable, stop =
  stop-loss/breakout); open-orders list + cancel.
- **Alerts**: price (above/below) and trend (SMA cross) alerts that fire on refresh.
- Live adapters gained `open_orders` / `cancel_order` (IBKR/Binance; keys required).
- **Single-command deploy**: FastAPI serves the built cockpit at one port
  (`eqa serve`); Dockerfile + docker-compose for a one-container run.

## 2026-06 — Crypto (separate asset class)

- 24/7, 365-day calendar; 20-coin universe via yfinance; bar = **hold BTC**.
- **Trend-following BTC** beats buy-and-hold risk-adjusted in-sample (Sharpe 1.14
  vs 1.02, robust across MA params); out-of-sample (walk-forward) the edge is
  marginal (1.07) — really a **drawdown-control** tool (−72% vs −83%) at a return
  cost. Vol-targeting, long/short trend, alt-momentum and alt-trend do **not** beat
  hold-BTC.
- **Funding carry** (Binance free API): a structural ~10%/yr delta-neutral yield,
  positive every year but decaying (2021 ~31% → 2026 ~1%).
- Tooling: `eqa ingest-crypto`, `backtest-crypto`, `backtest-overlay --crypto`,
  `crypto-funding`; dashboard Crypto tab; `crypto-live-run` (Binance/ccxt, dry-run
  by default). See `docs/CRYPTO_FINDINGS.md`.

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
