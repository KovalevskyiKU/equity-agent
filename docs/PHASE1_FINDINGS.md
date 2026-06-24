# Phase 1 findings — signal research

Data: daily bars 2015-01-02 → 2026-06-17, AAPL / NVDA / JPM (+ SPY benchmark,
^VIX regime). Source: yfinance. Horizon studied: 10 trading days (the swing
horizon the edge lives at — 1-day is noise).

All correlations are Spearman rank IC. "non-overlap" subsamples every `horizon`
rows per symbol so forward windows don't overlap; its t-stat is the honest one
(the pooled t-stat is inflated by autocorrelation).

## Technical features (pooled AAPL/NVDA/JPM, h=10)

| feature    | IC     | non-overlap IC | non-overlap t |
|------------|--------|----------------|---------------|
| atr_14     | 0.082  | **0.095**      | **2.79**      |
| bb_width   | 0.060  | 0.075          | 2.19          |
| vol_20     | 0.071  | 0.073          | 2.15          |
| vix_level  | 0.044  | 0.059          | 1.73          |
| momentum / MACD / RSI / OBV | ~0 | < 0.05 | < 1.7 |

**Only the volatility cluster survives the non-overlap correction**, led by
`atr_14`. Higher recent realised volatility → higher 10-day forward return
(a volatility-risk-premium / rebound effect). Momentum looked strong on a 2-year
sample but collapsed to noise on the full 10 years — a regime artifact.

## Kronos probabilistic signal (non-overlapping, 150 points/symbol, h=10)

| symbol | k_p_up         | k_exp_ret     | k_ret_std     |
|--------|----------------|---------------|---------------|
| AAPL   | 0.214 (t=2.67) | 0.158 (t=1.95)| 0.006         |
| NVDA   | 0.060 (t=0.73) | 0.034 (t=0.41)| -0.078        |
| JPM    | 0.124 (t=1.53) | 0.127 (t=1.56)| 0.063         |
| pooled | **0.100 (t=2.13)** | 0.059 (t=1.24) | 0.036    |

Kronos' directional probability `k_p_up` has a **small but significant pooled
edge (IC 0.10, t=2.13)**, driven mostly by AAPL, negligible on NVDA. Magnitude
is comparable to the best technical feature (`atr_14`). It is **not** a strong
standalone oracle — consistent with using it as one signal, not the verdict.

The earlier hint that `k_ret_std` (model uncertainty) is a risk signal did **not**
hold on the full sample (pooled t=0.75, inconsistent signs) — it was a small-sample
artifact (n=33). Retracted.

## Caveats

- Pooled across only 3 liquid, tech-heavy, correlated names.
- Single horizon (10d). In-sample feature selection.
- No walk-forward yet — significance is suggestive, not a verdict. The Phase 3
  walk-forward + Monte-Carlo permutation is what confirms or kills these.

## Open decisions (awaiting alignment)

- (resolved) Combination = lean LLM agent; engine = custom event-driven; LLM =
  Groq free tier (Gemini free was ~20 req/day); broker target = IBKR (Phase 4).

## Phase 2 — LLM-agent backtest across regimes (2026-06-18)

Lean agent (Groq llama-3.3-70b), AAPL/NVDA/JPM, weekly rebalance, features-only
(no sentiment/Kronos), vs an equal-weight basket and SPY. The first prompt was
too timid (sat in cash, Sharpe 0.10); reframed for long-only (default invested,
trim only on adverse signals). Sharpe by 6-month window:

| window (6mo to) | regime              | LLM   | basket | SPY   |
|-----------------|---------------------|-------|--------|-------|
| 2026-06 latest  | calm bull           | 0.66  | 1.10   | 1.31  |
| 2020-06         | COVID crash+rebound | 0.88  | 0.72   | 0.01  |
| 2022-10         | 2022 bear           | -0.41 | -0.42  | -0.58 |

The agent behaves like a **risk-managed** strategy: in every window it ran lower
volatility and shallower max-drawdown than the basket (COVID DD -25% vs -35%;
2022 -19% vs -27%). In the volatile/adverse windows it beat the basket on return
AND risk (COVID +14.6% vs +13.5%, SPY -4.6%; 2022 -7.4% vs -11.2%); in the calm
bull its caution was a drag. Driven by the one validated signal (volatility).

Caveats: 3 hand-picked windows, 3 correlated names, features-only, not full
walk-forward — directional evidence, not a verdict. Next: add Kronos, rolling
multi-window / walk-forward, decide objective (risk-adjusted vs absolute).

## Mechanical strategy sweep — no LLM, 66 rolling 6mo windows (2026-06-18)

Median over 66 rolling windows (step 2mo), full history, AAPL/NVDA/JPM:

| strategy              | median Sharpe | median maxDD | median return |
|-----------------------|---------------|--------------|---------------|
| basket (equal-weight) | 1.66          | -13.4%       | +20.2%        |
| vol-target            | 1.49          | -11.3%       | +16.2%        |
| SPY                   | 1.07          | -9.0%        | +6.5%         |

vol-target beats the basket on Sharpe in only 27% of windows and has a shallower
drawdown in 41%. **Sobering:** over this tech-heavy decade the equal-weight
basket dominates risk-adjusted; mechanical risk management (vol targeting) mostly
gave up return for little average risk benefit — it helps only in the minority
of crash windows (COVID/2022). Implications: the bar for the LLM agent is high
(beat basket Sharpe ~1.66), and diversifying the universe likely matters more
than the allocator on 3 correlated names. Objective chosen: risk-adjusted.

### Diversified universe — 8 names across sectors (same 66-window sweep)

Universe expanded to AAPL/MSFT/NVDA/JPM/UNH/XOM/PG/HD. (Also fixed a leverage bug:
vol_target_weights sized each name to max_weight independently, ~2.7x gross on 8
names; now capped at 100% gross.)

| strategy   | median Sharpe | median maxDD | median return |
|------------|---------------|--------------|---------------|
| basket     | 1.56          | -8.2%        | +12.4%        |
| vol-target | 1.51          | -7.5%        | +10.7%        |
| SPY        | 1.07          | -9.0%        | +6.5%         |

Diversification did the heavy lifting: the basket's median drawdown fell from
-13.4% (3 tech names) to -8.2%. vol-target now genuinely protects — shallower
drawdown than the basket in 67% of windows (vs 41% on 3 names) — at a small return
cost and ~neutral Sharpe. Cleaner, more realistic baseline; LLM bar ~Sharpe 1.56.

### Kronos as a mechanical rule — 12mo, diversified universe (no LLM)

Long-tilt by Kronos P(up)-0.5 per name, risk-capped:

| metric | kronos | voltgt | basket | SPY  |
|--------|--------|--------|--------|------|
| return | 7.9%   | 13.4%  | 18.5%  | 24.4%|
| Sharpe | 0.80   | 1.19   | 1.55   | 1.78 |
| max DD | -9.7%  | -10.0% | -10.0% | -9.1%|

**Kronos-as-rule is the worst of all four** — it underperforms vol-target, the
basket and SPY on both return and Sharpe. Consistent with the weak directional IC
(~0.1, mostly AAPL): Kronos' P(up) does not carry tradable edge as a standalone
rule on this universe. Treat Kronos as at most a minor input, not a driver.

### LLM agent on diversified universe — clean recent 6mo run (2026-06-19)

| metric  | LLM    | voltgt | basket | SPY    |
|---------|--------|--------|--------|--------|
| Sharpe  | 0.77   | 0.94   | 1.14   | 1.41   |
| ann vol | 24.9%  | 11.3%  | 11.7%  | 13.8%  |
| max DD  | -17.6% | -8.6%  | -8.5%  | -9.1%  |

On 8 names the LLM took concentrated, high-vol positions (worst Sharpe) — the
opposite of its timid/defensive behaviour on 3 names. Prompt-sensitive and
inconsistent; has not beaten the basket risk-adjusted in any clean window. (The
2022 window this session was invalid — hit the ~100k tokens/day cap, 10/26 dates
failed.) **Accumulating picture: no "smart" layer — Kronos, momentum, or the LLM
agent — has beaten the diversified basket + mild vol-management on a risk-adjusted
basis in clean tests.** Basket + vol-management is the baseline to beat; nothing has yet.

## Locked core strategy (pivot, 2026-06-19)

Core = **vol-target weights over the diversified 8-name universe** (per-name cap
0.34, gross <= 1), **NO circuit breaker**. Full history (2015-2026):

| metric       | core (vol-target) | basket | SPY   |
|--------------|-------------------|--------|-------|
| total return | 8.65x             | 8.52x  | 2.66x |
| Sharpe       | 1.13              | 1.11   | 0.73  |
| max DD       | -34%              | -34%   | -34%  |

vol-target ≈ equal-weight basket full-history (Sharpe 1.13 vs 1.11); both beat SPY
~3x on return at similar Sharpe. The drawdown circuit breaker HURTS even at 25%
(return 8.65x -> 1.31x, Sharpe -> 0.67): it liquidates into V-shaped crashes
(2020/2022) and misses the recovery — keep it OFF (only a very deep tail backstop).
Honest core: **diversification is the edge**; vol-target is the principled
allocator; risk overlays add little to nothing over this period. The job now is
disciplined execution (paper loop) + monitoring + a narrow LLM news risk-off gate.

## Macro regime features — rates/USD via yfinance, no key (2026-06-24)

Added 10y yield (^TNX) and USD index (DX-Y.NYB) as regime context. IC vs 10d
forward return (pooled 8 names; non-overlap t is the honest column, though macro
is shared across names so even it is somewhat inflated by cross-sectional dup):

| feature     | IC     | non-overlap t |
|-------------|--------|---------------|
| atr_14      | 0.094  | 4.8           |
| vix_level   | 0.061  | 3.0           |
| tnx_level   | -0.053 | -2.6          |
| tnx_chg_20  | -0.043 | -2.2          |
| usd_ret_20  | ~0     | 1.4           |

**New, economically-sensible signal: interest rates.** Higher / rising 10y yields
precede lower forward equity returns. Going wide on data found real edge here —
unlike Kronos/momentum. Confirm with a rates-aware overlay backtest + proper
time-series stats (effective N ≈ #dates, not #rows). LLM provider is now
switchable (Groq now → local Ollama later via `LLM_PROVIDER`).

Cross-asset / credit (yfinance ETFs, no key) added next: **credit_ret_20 (HYG−LQD
20d) is significant (noov t −3.58)** — extended credit risk-on precedes lower
forward returns; tlt_ret_20 borderline (1.97); gold/USD ~0. So a coherent regime
cluster has emerged: **volatility (atr_14, vix) + rates (tnx) + credit
(HYG−LQD)** — all economically sensible and non-overlap-robust. (Same caveat:
macro features are shared across names, so even noov t is somewhat inflated.)
