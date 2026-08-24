# equity-agent

[![CI](https://github.com/KovalevskyiKU/equity-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/KovalevskyiKU/equity-agent/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A quantitative research lab and trading application for US equities and crypto —
built to find out, honestly, whether a retail-scale system can beat the market.**

It has two halves that share one core:

1. **A research engine** that tests trading hypotheses the way a real quant fund
   does — point-in-time data, out-of-sample validation, net of trading costs — and is
   deliberately built to *disprove* its own ideas.
2. **A trading application** — a web cockpit with charts, live signals, a paper
   broker, and gated live order routing to IBKR / Binance.

> **The honest headline:** after testing ~15 hypotheses across 1,700+ instruments and
> 11 years of data, **almost nothing beats simply holding the index.** The one signal
> that survives every check is fading. That null result — and the machinery that
> proves it — is the point of this repository.

---

## Table of contents

- [What the research found](#what-the-research-found)
- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [Repository map](#repository-map)
- [Command reference](#command-reference)
- [Documentation](#documentation)
- [Methodology — the rules that keep it honest](#methodology--the-rules-that-keep-it-honest)
- [Status and limitations](#status-and-limitations)
- [Development](#development)

---

## What the research found

Everything below is **point-in-time** (only data that existed at the time),
**total-return** (dividends included) and **net of trading costs**. The bar to beat is
**SPY** for equities and **holding BTC** for crypto.

### Equities: no factor beats the index

| strategy (point-in-time, 2015-2026) | Sharpe | ann. alpha | alpha t-stat |
|-------------------------------------|-------:|-----------:|-------------:|
| **SPY (the benchmark)** | **0.81** | — | — |
| composite: value + quality + net issuance | 0.90 | +2.67% | 1.30 |
| gross margin | 0.80 | +0.91% | 0.38 |
| 12-1 momentum | 0.75 | +0.86% | 0.31 |
| low volatility | 0.71 | +1.10% | 0.42 |
| equal-weight basket | 0.70 | **−1.27%** | −0.69 |

Nothing reaches statistical significance (|t| > 2). Tested and rejected: momentum,
low-vol, value, quality, margins, asset growth, accruals, gross-profits-to-assets,
short-term reversal, a walk-forward ML model, the Kronos foundation model, and an LLM
decision agent.

### The most important finding: survivorship bias

The first results looked *excellent* — a momentum portfolio returning **6.4x** vs
SPY's 3.3x. Then historical index membership was reconstructed and everything re-run:

| | with today's members (biased) | point-in-time (honest) |
|---|---:|---:|
| equal-weight basket | 3.59x | **1.78x** |
| momentum top-quintile | 6.44x | 2.48x |

**Survivorship bias — silently testing only on the companies that survived — was
inflating results by roughly 2x.** Every earlier conclusion had to be rewritten. This
is the single most valuable lesson in the repository.

### The one signal that works (and is fading)

**Net share issuance** — companies buying back stock outperform those issuing it:

- Cross-sectional IC **t = 2.45** (statistically significant)
- Market-neutral long-short: **+1.64%/yr at Sharpe 0.44**, turnover only 15, so
  trading costs do *not* eat it
- Stable across halves of the sample (Sharpe 0.45 vs 0.43) and across quantile
  choices (0.32-0.49)

**But:** its alpha would need ~27 years of data to prove at t = 2, and it has gone
flat since 2023 — confirmed on two independent data sources (annual and quarterly
filings). A real but decaying effect.

### Crypto: risk control, not alpha

| strategy (365-day year, net of costs) | total | Sharpe | max drawdown |
|---------------------------------------|------:|-------:|-------------:|
| hold BTC | 189x | 1.02 | −83% |
| trend-following BTC (walk-forward) | 95x | 1.07 | −72% |
| vol-targeting BTC | 64x | 0.97 | −70% |
| alt-coin momentum | 6x | 0.61 | −92% |

Trend-following is a **drawdown-control tool**, not free outperformance: chosen
honestly out-of-sample it roughly ties holding BTC while giving up half the return.
**Funding carry** (a delta-neutral yield from perpetual futures) is structurally real
at ~10%/yr historically — but has decayed from ~31%/yr (2021) to ~1% (2026).

Full write-ups: [`docs/PHASE1_FINDINGS.md`](docs/PHASE1_FINDINGS.md) (equities),
[`docs/CRYPTO_FINDINGS.md`](docs/CRYPTO_FINDINGS.md) (crypto).

---

## What you get

### 1. Trading cockpit (web app)

A real interactive trading interface — FastAPI backend + React frontend:

- **Watchlist** across equities and crypto
- **Candlestick charts** (TradingView lightweight-charts) with moving-average overlays
- **Live signal panel** — price, trend, volatility, momentum per instrument
- **Order ticket** — market / limit / stop orders, venue selector (paper / IBKR / Binance)
- **Open orders** with cancel, **live portfolio** and equity curve over WebSocket
- **Price and trend alerts**

Live venues require a **two-step confirmation** — nothing is transmitted without it.

### 2. Research dashboard (Streamlit)

Six tabs: paper portfolio, factor "predictions" (what each factor favours today),
factor backtests vs SPY, the risk overlay, crypto, and news sentiment.

### 3. Research CLI

30+ commands for ingesting data, building features, measuring signals
(cross-sectional IC, alpha/beta, long-short spreads) and backtesting — see the
[command reference](#command-reference).

---

## Quick start

```bash
# 1. Install (Python 3.12)
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev,api,ui]"

# 2. Get data (free — no API key needed for prices)
eqa initdb
eqa ingest
eqa ingest-crypto

# 3a. Open the trading cockpit
cd frontend && npm install && npm run build && cd ..
eqa serve

# 3b. ...or the research dashboard
python app.py

# 4. Reproduce the headline research numbers
eqa research-report
```

The cockpit serves at `http://localhost:8000`, the dashboard at
`http://localhost:8501`. For one container instead:

```bash
docker compose up --build
```

**API keys are optional.** Prices, crypto bars and funding data are free. A free
[Finnhub](https://finnhub.io) key unlocks fundamentals,
[FRED](https://fred.stlouisfed.org/docs/api/api_key.html) unlocks macro series, and
Groq/Cerebras unlock the news-sentiment gate. Copy `.env.example` to `.env` and fill
in what you have.

---

## Repository map

```text
equity-agent/
├── src/equity_agent/          # the Python core
│   ├── data/                  # DATA INGESTION
│   │   ├── yfinance_provider.py   # free daily bars (equities + crypto), batched
│   │   ├── fundamentals.py        # point-in-time fundamentals (Finnhub, as-filed)
│   │   ├── sp500_history.py       # * historical index membership (survivorship fix)
│   │   ├── universe.py            # index constituents + sector map
│   │   ├── funding.py             # crypto perp funding rates (Binance, free)
│   │   ├── fred.py, news.py       # macro series / news articles
│   │   └── ingest.py              # idempotent upsert into the DB
│   ├── signals/               # FEATURE ENGINEERING
│   │   ├── features.py            # causal technical features (never look ahead)
│   │   ├── feature_store.py       # per-symbol Parquet store + market context
│   │   ├── kronos_signal.py       # Kronos foundation-model signal (optional)
│   │   └── sentiment.py           # LLM news scoring
│   ├── research/              # SIGNAL EVALUATION
│   │   ├── factor_eval.py         # * per-date cross-sectional IC, price factors
│   │   ├── fundamental_factors.py # value/quality/issuance factors, sector-neutral
│   │   ├── signal_eval.py         # pooled IC + quantile spreads
│   │   ├── validation.py          # purged walk-forward, non-overlapping IC
│   │   ├── wf_strategy.py         # walk-forward ML payoff test
│   │   └── report.py              # the `eqa research-report` generator
│   ├── backtest/              # SIMULATION
│   │   ├── engine.py              # event-driven, next-open fills, fees + slippage
│   │   ├── metrics.py             # * Sharpe/Sortino/Calmar + CAPM alpha & beta
│   │   ├── long_short.py          # * dollar-neutral long-short (high-power test)
│   │   ├── factor_portfolio.py    # quantile portfolios, point-in-time runner
│   │   ├── overlay_backtest.py    # vol-target risk overlay
│   │   ├── crypto.py              # crypto strategies (trend / vol-target / alts)
│   │   └── sweep.py, panels.py, strategy.py
│   ├── execution/             # ORDERS
│   │   ├── paper_broker.py        # stateful paper account (market/limit/stop)
│   │   ├── orders.py              # shared, pure order planner
│   │   ├── ibkr_broker.py         # IBKR adapter (dry-run by default)
│   │   ├── crypto_broker.py       # Binance adapter (dry-run by default)
│   │   └── runner.py              # daily rebalance to the core strategy
│   ├── api/app.py             # FastAPI: instruments, bars, signals, orders, WS
│   ├── dashboard/data.py      # Streamlit data accessors
│   ├── risk/                  # exposure limits + vol-target overlay
│   ├── storage/               # SQLAlchemy models + engine (SQLite -> Postgres)
│   ├── alerts.py              # price / trend alerts
│   ├── monitoring.py          # equity, P&L, drawdown, tracking vs benchmark
│   └── cli.py                 # the `eqa` command-line interface
├── frontend/                  # React + Vite + TypeScript trading cockpit
│   └── src/components/        # Chart, OrderTicket, OpenOrders, Alerts, Portfolio...
├── docs/                      # * research findings and methodology (start here)
├── tests/                     # 123 tests
├── app.py                     # launches the Streamlit dashboard
├── dashboard_app.py           # the Streamlit app itself
├── config.yaml                # universe, benchmark, core strategy (non-secret)
└── Dockerfile, docker-compose.yml
```

`*` marks the files carrying the most important ideas.

---

## Command reference

### Data

| command | what it does |
|---------|--------------|
| `eqa initdb` | create tables, register the configured universe |
| `eqa ingest` | daily bars for equities + benchmark + macro regime symbols |
| `eqa ingest-crypto` | daily bars for the crypto universe |
| `eqa ingest-fundamentals` | point-in-time annual fundamentals (Finnhub) |
| `eqa features` | build the causal Parquet feature store |
| `eqa status` | how many bars are stored per symbol |

### Research

| command | what it does |
|---------|--------------|
| `eqa research` | rank features by predictive edge (IC + quantile spread) |
| `eqa factor-ic` | per-date cross-sectional IC of price factors |
| `eqa factor-backtest` | monthly top-quintile factor portfolios — *survivorship-biased, prints a warning* |
| `eqa factor-backtest-pit` | **the honest version**: point-in-time membership; `--fundamentals` adds value/quality |
| `eqa walkforward` | does a walk-forward ML model beat the basket out-of-sample? |
| `eqa research-report` | regenerate the headline findings into `data/reports/` |

### Backtesting

| command | what it does |
|---------|--------------|
| `eqa backtest` | a baseline strategy vs basket vs SPY |
| `eqa backtest-sweep` | rolling-window comparison of mechanical strategies |
| `eqa backtest-overlay` | SPY vs the vol-target risk overlay (`--crypto` for BTC) |
| `eqa backtest-crypto` | hold-BTC vs trend / vol-target / alt-momentum |
| `eqa crypto-funding` | delta-neutral funding-carry yield from Binance perps |
| `eqa backtest-llm`, `eqa backtest-kronos` | the LLM agent / Kronos rule (both underperform) |

### Trading and operations

| command | what it does |
|---------|--------------|
| `eqa serve` | run the API + trading cockpit on one port |
| `python app.py` or `eqa dashboard` | the Streamlit research dashboard |
| `eqa paper-reset`, `paper-run`, `paper-status` | the paper-trading account |
| `eqa monitor` | equity, P&L, drawdown, Sharpe, tracking vs SPY |
| `eqa daily` | full daily cycle: ingest, rebalance, monitor |
| `eqa live-run` | **IBKR** orders — dry-run by default, `--execute` transmits |
| `eqa crypto-live-run` | **Binance** orders — dry-run by default, `--execute` transmits |
| `eqa news`, `eqa decide` | news sentiment / an LLM decision for one symbol |

Every command supports `--help`.

---

## Documentation

| document | what's inside |
|----------|---------------|
| [**STRATEGY.md**](docs/STRATEGY.md) | **Start here** — what to actually run, and what is *not* proven |
| [**PHASE1_FINDINGS.md**](docs/PHASE1_FINDINGS.md) | The full equity research log: every hypothesis, result and retraction |
| [**CRYPTO_FINDINGS.md**](docs/CRYPTO_FINDINGS.md) | Crypto: trend, vol-target, alt-momentum, funding carry |
| [**METHODOLOGY.md**](docs/METHODOLOGY.md) | The rules that keep results honest — read before adding a backtest |
| [**GLOSSARY.md**](docs/GLOSSARY.md) | Every term in plain language, plus a paid-data buyer's guide |
| [CHANGELOG.md](CHANGELOG.md) | What landed when |
| [docs/](docs/README.md) | Index of all documentation, with a phase-by-phase map of the research |

---

## Methodology — the rules that keep it honest

Each of these exists because breaking it produced a large fake edge at some point in
this project's history:

1. **Point-in-time index membership.** Rank only companies that were actually in the
   index that day. Ignoring this inflated results ~2x.
2. **Point-in-time fundamentals.** Lag every figure to its *filing date*, never the
   fiscal period end — a December quarter is not public until February.
3. **Benchmark = the cap-weighted index.** Beating an equal-weight basket proves
   nothing; equal-weight itself has negative alpha vs SPY.
4. **Measure alpha, not just Sharpe.** In a bull decade a low-beta strategy looks bad
   even with real alpha, so run the CAPM regression.
5. **Long-short, not just long-only.** A long-only quantile portfolio is mostly market
   beta plus a small tilt — a low-power test.
6. **Per-date cross-sectional IC**, with non-overlapping windows for the t-stat.
7. **Always net of costs**, with realistic slippage per market segment.
8. **Respect statistical power.** Proving an edge at t = 2 needs roughly
   `(2 / information-ratio)^2` years — 16 years at IR 0.5, 64 at IR 0.25. Eleven years
   can only ever prove *large* edges.

---

## Status and limitations

**Works and is tested:** data ingestion, feature store, research harness, backtest
engine, paper trading (market/limit/stop), alerts, monitoring, the API, the cockpit,
the dashboard and Docker deployment. 123 tests, ruff + mypy clean, CI green.

**Not verified end-to-end:** live order transmission to IBKR and Binance. The order
*maths* is pure and unit-tested, but the broker glue has only been exercised against
fakes — it needs real credentials and a running gateway.

**Known limitations of the research:**

- 11.4 years of history — enough to disprove large edges, not to prove small ones.
- ~113 delisted US names and ~20-24% of small caps have no free price history, so a
  residual survivorship bias remains, and it flatters the results.
- Fundamentals come from as-filed annual reports; share-count coverage is ~76%.
- Crypto rests on one asset (BTC) and roughly three market cycles.
- Short legs assume costless borrow.

**This is research software, not investment advice.** Nothing here is a recommendation
to buy or sell anything.

---

## Development

```bash
pytest                          # 123 tests
ruff check .                    # lint
mypy                            # type check
cd frontend && npm run build    # typecheck + bundle the cockpit
```

Contributions of the form "here is a hypothesis and the point-in-time test that kills
or confirms it" are the most welcome kind.
