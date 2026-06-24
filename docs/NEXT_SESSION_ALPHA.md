# Next session — universe expansion & cross-sectional alpha hunt

Kickoff brief for a fresh session. Goal: **expand the traded universe to a large,
liquid set and hunt CROSS-SECTIONAL alpha (equity factors)** — because the prior
research arc proved time-series signals on a small set do not beat the basket.

## Read first — where we are

A complete system is already built and runs end-to-end on **free infra**
(yfinance + Groq/Cerebras + local Kronos), ~44 commits, ~42 tests green:

- data ingest (`eqa ingest`, incl. FRED macro) → causal feature store (`eqa features`)
- IC / validation harness (`eqa research`; `research/validation.py`: PurgedWalkForwardSplit,
  non_overlapping_ic, ic_stability)
- event-driven backtester + metrics + rolling sweep (`eqa backtest`, `backtest-sweep`)
- walk-forward model test (`eqa walkforward`)
- risk layer (`risk/limits.py`, drawdown circuit breaker in the engine)
- paper-trading loop (`eqa paper-reset/run/status`, `eqa daily`) + Streamlit dashboard (`eqa dashboard`)
- multi-provider LLM with failover (`LLM_PROVIDER=groq,cerebras`; Ollama later) + LLM news risk-off gate

**VERDICT of the research so far (docs/PHASE1_FINDINGS.md):** no signal tried —
Kronos, momentum, volatility, rates, credit, or their walk-forward combination —
produces tradable out-of-sample alpha over a diversified equal-weight / vol-target
basket (OOS IC ~0, WF-model Sharpe 0.48 vs basket 1.23). The only edge so far is
**diversification + discipline**. Significant in-sample ICs were inflated by
overlapping windows, cross-sectional macro duplication, and multiple testing.

## Why pivot the hunt to cross-sectional factors

We only tested **time-series** technicals + macro on **8 correlated names**.
Cross-sectional alpha (which name out/under-performs which) is the documented
source of equity premia (value / quality / momentum / low-vol) and **needs many
names** — with 8 it is untestable. So the two goals are linked: **expanding the
universe is what makes cross-sectional factor research possible.** This is the
highest-legitimacy avenue we have not tried.

## Part A — expand the universe

1. Choose ~50–100 liquid US large/mid-caps across sectors (or the S&P 100 list).
   Keep it **config-driven** (`config.yaml: universe`). yfinance fetches them all
   (daily, no key); ingest is idempotent.
2. **Survivorship-bias warning:** using *today's* index membership over 10y of
   history is biased (dead/delisted names dropped). Either get point-in-time
   constituents (hard on free tiers) or **document the bias explicitly** and
   prefer names with full-period history.

## Part B — hunt cross-sectional alpha (factors)

Rank names cross-sectionally each rebalance; build long-only top-quantile (or
long-short) portfolios per factor + a combined multi-factor model.

Candidate factors:
- **Momentum** (12-1 month return) — classic, no fundamentals needed (price only).
- **Low-volatility** (trailing realised vol) — the low-vol anomaly; price only.
- **Value** (earnings yield, book-to-price) — needs fundamentals (FMP/Finnhub).
- **Quality** (ROE, gross margin, low accruals) — fundamentals.
- **Size** (smaller > larger) — weak at large-cap; include only if universe has mid-caps.

Start with the **price-only factors (momentum, low-vol)** — no point-in-time data
risk, testable immediately. Add fundamental factors after the connector exists.

Method:
- Build a factor panel; compute **per-date cross-sectional IC** (rank-corr of the
  factor vs forward return *across names on each date*, averaged over time). NOTE:
  the current `research/signal_eval.py` IC **pools** across names — add a
  cross-sectional (per-date) IC mode for factor work.
- Backtest top-quantile / long-short portfolios **walk-forward** vs the
  equal-weight basket **on the same expanded universe** (that is the bar to beat,
  risk-adjusted). Consider **monthly** rebalance — factors work better monthly
  than daily and cut turnover/costs.

## Methodology rules — do not fool ourselves

- **Point-in-time fundamentals (critical):** yfinance/Finnhub return *current*
  fundamentals, not as-reported-at-the-time → look-ahead. Lag fundamentals by the
  reporting delay (use reported/filing dates), or you will get fake alpha.
- **Cross-sectional IC per date**, not pooled. Correct for overlapping windows.
- **Walk-forward OOS only**; regularise; expect modest results — these factors are
  widely known and arbitraged.
- **Net of costs** (factor strategies have real turnover).
- Honest bar: beat the equal-weight / cap-weight basket on the expanded universe,
  risk-adjusted. If it doesn't, say so (as we did before).

## Constraints / context

- Horizon: daily-swing project, but factor work likely **monthly**. Long-only by
  default (no PDT issues). Objective = risk-adjusted (Sharpe/Calmar/drawdown).
- LLM (Groq+Cerebras now, Ollama on new hardware ~1 month) — useful later for
  qualitative fundamentals/filings at scale, not needed for the factor MVP.
- Keys live in `.env` (never `.env.example`).

## First concrete steps

1. Pick the expanded universe (~50–100 names) → `config.yaml`; `eqa ingest` + `eqa features`.
2. Add a **per-date cross-sectional IC** mode to the research harness.
3. Build **price-only factors** (12-1 momentum, low-vol); measure cross-sectional IC.
4. Walk-forward backtest a top-quantile (or long-short) momentum/low-vol portfolio
   vs the expanded equal-weight basket (monthly rebalance).
5. If promising: add a **point-in-time fundamentals connector** (FMP/Finnhub with
   reported dates) → value/quality factors → combined multi-factor model.
