# equity-agent

[![CI](https://github.com/KovalevskyiKU/equity-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/KovalevskyiKU/equity-agent/actions/workflows/ci.yml)

Multi-signal, agentic **US-equities daily-swing** trading system.

The decision core is a multi-agent LLM (analysts → bull/bear debate → trader →
risk → portfolio manager, via [TradingAgents](https://github.com/TauricResearch/TradingAgents)).
The [Kronos](https://github.com/shiyu-coder/Kronos) foundation model contributes
**one** technical signal — a probabilistic direction estimate — not the verdict.
Everything is validated against a SPY buy-and-hold benchmark before any real capital.

> Daily swing (not intraday) by design: it sidesteps the PDT rule, uses free clean
> historical data, and matches the cadence at which news/LLM analysis is useful.

## Architecture

```
data ─▶ signals ─▶ decision ─▶ risk ─▶ execution
        (cross-cutting: storage · backtest/eval · orchestration & monitoring)
```

| Layer | Package | Phase |
|-------|---------|-------|
| Data (market/fundamental/news/macro) | `equity_agent.data` | 0–1 |
| Signals (features, Kronos, LLM sentiment) | `equity_agent.signals` | 1 |
| Decision (TradingAgents) | `equity_agent.decision` | 2 |
| Backtest & evaluation | `equity_agent.backtest` | 3 |
| Risk | `equity_agent.risk` | 4 |
| Execution (IBKR) | `equity_agent.execution` | 4 |
| Storage (SQLAlchemy: SQLite→Postgres) | `equity_agent.storage` | 0 |

## Setup

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env        :: then fill in keys as phases need them
```

## Usage

```cmd
eqa initdb       :: create tables + register the configured universe
eqa ingest       :: fetch & store daily bars (yfinance, no key needed)
eqa features     :: build the causal Parquet feature store
eqa research     :: rank features by predictive edge (IC + quantile spread)
eqa status       :: show stored bar counts per symbol
```

### Optional: Kronos signal

The Kronos model is vendored (cloned) and pulls in `torch`. One-time setup:

```cmd
git clone https://github.com/shiyu-coder/Kronos third_party/Kronos
pip install -e ".[kronos]"
eqa kronos-signal AAPL --horizon 10 --samples 20
```

Weights download from HuggingFace on first run (no token needed).

Paper trading and the dashboard:

```cmd
eqa paper-reset --cash 100000   :: start a paper account
eqa paper-run --risk-off        :: rebalance to the core (default: hold SPY) + LLM news gate
eqa paper-status                :: cash / equity / positions
eqa monitor                     :: equity, P&L, drawdown, Sharpe, tracking vs SPY
```

### Open the dashboard

```cmd
pip install -e ".[ui]"
python app.py                   :: <- run this (or click Run on app.py in PyCharm)
```

Opens at http://localhost:8501 with four tabs: **Paper portfolio** (equity, P&L,
drawdown, tracking vs SPY), **Predictions** (top names each factor favors now),
**Backtest (% profit)** (factor portfolios vs SPY), **Signals & news**. Equivalent:
`eqa dashboard` or `streamlit run dashboard_app.py`.

Backtesting: `eqa backtest --strategy vol-target`, `eqa backtest-sweep` (rolling,
no LLM), `eqa backtest-llm` (LLM agent), `eqa backtest-kronos` (Kronos rule).

Cross-sectional factors (point-in-time, the honest read):

```cmd
eqa ingest-fundamentals          :: point-in-time annual fundamentals (Finnhub)
eqa factor-ic                    :: per-date cross-sectional IC of price factors
eqa factor-backtest              :: monthly top-quintile factor portfolios (biased)
eqa factor-backtest-pit --fundamentals  :: survivorship-corrected, value/quality incl.
```

> See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): SPY (cap-weight) is the bar,
> membership and fundamentals are point-in-time. `factor-backtest`/`backtest-sweep`
> use today's universe and are survivorship-biased (they print a NOTE).

Crypto (separate asset class, 365-day year, bar = hold BTC):

```cmd
eqa ingest-crypto                :: daily bars for the crypto universe (yfinance)
eqa backtest-crypto              :: hold-BTC vs trend / vol-target / alt-momentum
eqa backtest-overlay --crypto    :: vol-target overlay on BTC across target vols
```

> Crypto verdict ([`docs/CRYPTO_FINDINGS.md`](docs/CRYPTO_FINDINGS.md)): unlike
> equities, **trend-following beats buy-and-hold BTC** risk-adjusted (Sharpe 1.14 vs
> 1.02, −68% vs −83% drawdown), robust across MA parameters; vol-targeting and
> alt-momentum do not. Survivorship caveats apply.

### Run it daily (orchestration)

`eqa daily` does the whole cycle: ingest latest bars → rebuild features → paper
rebalance (with the risk-off gate). Schedule it on Windows after the US close:

```
Task Scheduler → Create Task → Daily ~22:35 (your TZ, after 16:00 ET close)
  Program:   D:\Portative workhub\equity-agent\.venv\Scripts\eqa.exe
  Arguments: daily
  Start in:  D:\Portative workhub\equity-agent
```

### Live execution (IBKR, Phase 4)

```cmd
pip install -e ".[ibkr]"        :: ib_insync
:: start TWS or IB Gateway (paper port 7497) with the API enabled
eqa live-run                    :: DRY-RUN: print the orders it would place, send nothing
eqa live-run --execute          :: actually transmit market orders to reach the core target
```

> Safety: `live-run` is a **dry run by default** and transmits nothing without
> `--execute`. Order sizing (`plan_orders`) is pure and unit-tested; only the thin
> TWS glue needs a running gateway. Connection settings are in `.env`
> (`IBKR_HOST/PORT/CLIENT_ID`).

Universe and other non-secret params live in [`config.yaml`](config.yaml);
secrets (API keys, `DATABASE_URL`) live in `.env` (gitignored).

## Status

- **Phase 0 — foundation** ✅: repo, config, storage, daily-bar ingestion, tests, CI.
- **Phase 1 — signals** ✅: causal feature store, IC/quantile research harness with
  non-overlapping correction, Kronos probabilistic signal + eval, validation harness
  (non-overlapping IC, block stability, purged walk-forward), performance metrics.
- **Phase 2 — cross-sectional factors** ✅: S&P 500 universe, per-date cross-sectional
  IC, price (momentum/low-vol) + value/quality factors, **point-in-time index
  membership + fundamentals** (survivorship/look-ahead correct).
- **Phases 3–4** (risk/execution): paper-trading loop + risk-off gate live; IBKR pending.

### Research verdict (the honest result)

Across momentum, low-vol, value, quality and a sector-neutral composite, **no factor
beats the cap-weighted index (SPY) with statistical confidence once survivorship is
removed.** Survivorship bias was the dominant driver of the apparent edge: it inflated
the point-in-time equal-weight basket from 1.78x to 3.59x. The edge is **cap-weight
market exposure + diversification + discipline** — so the core defaults to tracking
SPY (`config.core_strategy`). Full write-up in
[`docs/PHASE1_FINDINGS.md`](docs/PHASE1_FINDINGS.md); rules in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Dev

```cmd
ruff check .
mypy
pytest
```
