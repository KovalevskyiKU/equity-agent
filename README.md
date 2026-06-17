# equity-agent

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
| Execution (Alpaca) | `equity_agent.execution` | 4 |
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
eqa status       :: show stored bar counts per symbol
```

Universe and other non-secret params live in [`config.yaml`](config.yaml);
secrets (API keys, `DATABASE_URL`) live in `.env` (gitignored).

## Status

**Phase 0 — foundation** (current): repo, config, storage, daily-bar ingestion,
tests, CI, monitoring hooks. Later phases are stubbed packages with documented
interfaces. See the roadmap discussion / project memory for the full plan.

## Dev

```cmd
ruff check .
mypy
pytest
```
