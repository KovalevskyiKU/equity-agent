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

FRED (with key) added: **hy_oas (HY credit-spread level) is the single strongest
feature (IC 0.11, noov t 3.3)** — wide/stressed spreads precede higher forward
returns (risk-premium / mean-reversion). **Yield-curve slope (2s10s) and 10y real
yield show NO short-horizon edge** (expected — they are months-to-years macro
signals, not 10-day). So the validated regime cluster is now: volatility + rates
+ credit (both spread level and HY−IG momentum). Next: a walk-forward
regime-overlay test — does the cluster actually improve the strategy's P&L?

## Walk-forward payoff test (2026-06-24) — THE VERDICT

Ridge model on the full feature/regime cluster, fit walk-forward (6 folds,
embargo = 10d), OOS predictions → long-only weights → backtested over 619 OOS days:

| metric       | WF-model | basket | SPY  |
|--------------|----------|--------|------|
| Sharpe       | 0.48     | 1.23   | 1.23 |
| total return | 19%      | 51%    | 56%  |
| max DD       | -22%     | -17%   | -19% |

**OOS IC = 0.004 (t = 0.27) — essentially zero.** The combined model does NOT beat
the diversified basket out-of-sample; it underperforms. The individual features
had significant *in-sample* IC (0.05–0.11), but that was inflated (overlapping
windows, cross-sectional macro duplication, multiple testing); the honest OOS
combined IC is ~0 and the strategy is worse than holding the basket.

**Conclusion of the research arc:** no signal tried — Kronos, momentum,
volatility, rates, credit, or their walk-forward combination — produces tradable
out-of-sample alpha over the diversified basket. The edge is diversification +
discipline. Confirms the pivot: core = diversified basket; LLM only for the
narrow news risk-off gate.

## Phase 2 — cross-sectional factors on the S&P 500 (2026-06-24)

Pivot per `docs/NEXT_SESSION_ALPHA.md`: the prior arc only tested **time-series**
signals on 8 correlated names. Cross-sectional alpha (which name out/under-performs
which) needs many names, so we expanded the universe and built the right tooling.

**What was built**
- Universe expanded to the **S&P 500** (503 names, current membership as of
  2026-06-24, snapshot `data/sp500_constituents_2026-06-24.csv`), config-driven in
  `config.yaml`; helper `data/universe.py`. Ingested 2015→2026 (~1.43M bars) via a
  new batched yfinance bulk download (`get_daily_bars_batch`, ~100/chunk).
- **Per-date cross-sectional IC** (`research/factor_eval.py`) — the right measure
  for factors: rank-corr of factor vs forward return *across names on each date*,
  averaged over time, with effective-N = #rebalance dates (monthly ⇒ non-overlapping
  ⇒ honest t). Contrast with the old pooled IC in `signal_eval.py`.
- Price-only factors: **12-1 momentum** (skip recent month) and **low-vol**
  (−trailing realised vol). Monthly top-quintile portfolios + idealized long-short
  spread, backtested net of costs through the shared engine
  (`backtest/factor_portfolio.py`). CLI: `eqa factor-ic`, `eqa factor-backtest`.

**Cross-sectional IC vs 21d forward return (503 names; monthly = honest)**

| factor        | mean IC (monthly) | t   | mean IC (daily) | t (daily, inflated) |
|---------------|-------------------|-----|-----------------|---------------------|
| momentum_12_1 | 0.001             | 0.06| 0.007           | 1.84                |
| low_vol       | −0.045            | −2.07 | −0.043        | −9.34               |

Momentum has **no** monotonic cross-sectional IC at the monthly horizon. low_vol's
IC is *negative* — but that is vs **raw** forward return (the low-vol anomaly is a
*risk-adjusted* claim), so the backtest Sharpe is the real test.

**Monthly top-quintile portfolios, net of costs (1bps fee + 5bps slippage)**

| metric        | momentum top-Q | low_vol top-Q | basket (monthly EW) | SPY   |
|---------------|----------------|---------------|---------------------|-------|
| total return  | **6.44x**      | 1.36x         | 3.59x               | 2.59x |
| CAGR          | **19.2%**      | 7.8%          | 14.2%               | 11.8% |
| ann vol       | 20.3%          | 14.2%         | 17.4%               | 17.7% |
| Sharpe        | **0.96**       | 0.60          | 0.85                | 0.72  |
| max drawdown  | −36.8%         | −35.1%        | −38.3%              | −34.1%|
| Calmar        | **0.52**       | 0.22          | 0.37                | 0.35  |
| long-short (idealized, gross) Sharpe | **0.28** | −0.93 | — | — |

**Momentum (12-1, top quintile) is the FIRST signal in the whole arc to beat the
diversified basket *and* SPY risk-adjusted, net of costs** (Sharpe 0.96 vs 0.85 vs
0.72; +5pp CAGR; shallower drawdown). low-vol does **not** beat the basket on this
bull decade (Sharpe 0.60 < 0.85), consistent with its negative raw-return IC.

**Robustness of the momentum edge (the honesty check)**
- Wins the basket on **return in 8 of 11** active calendar years (warm-up = 2015).
- But on **Sharpe** the basket is often higher (2016/2017/2021) — momentum buys
  return with extra vol (20.3% vs 17.4%, a beta tilt).
- Rolling-1y: momentum's Sharpe beats the basket in only **55%** of windows
  (≈ coin flip); rolling-1y **excess return** is positive in **65%** (median +3.0%).
- 2024 (+39% vs +17.5%) and 2026-YTD (+31.5% vs +8.7%) amplify the edge, but it is
  broad, not solely recent.

**VERDICT (price factors).** Cross-sectional **momentum shows a real, tradable
edge over the basket** — the first thing in this project that does — but it is
**not yet confirmed alpha**, for three reasons: (1) **survivorship bias** — today's
S&P 500 over 2015-2026 inflates *momentum specifically* (winners that stayed in the
index; delisted losers absent); (2) the risk-adjusted edge is **marginal/
inconsistent** (beats basket Sharpe in only 55% of rolling windows) and the
market-neutral **long-short spread is weak (Sharpe 0.28)** — most of the gain is the
long-leg beta tilt, not a clean factor; (3) low-vol does not work here. The
decisive next steps: **point-in-time constituents** to kill survivorship bias, then
test momentum **as a tilt on the basket** (not standalone) and at **decile / true
long-short**, plus add **value/quality** once a point-in-time fundamentals connector
exists. Until survivorship is removed, treat the momentum result as a promising
upper bound, not a green light.

## Phase 2b — point-in-time membership: the survivorship correction (2026-06-24)

Built the survivorship fix (`data/sp500_history.py`): reconstruct historical S&P 500
membership from the Wikipedia *changes* log by undoing every change after the query
date (262 changes since 2015). On each rebalance we now rank **only names that were
actually in the index that day** (`eqa factor-backtest-pit`). Union universe = current
503 + 235 names removed since 2015; **122 of the 235 dropped names have yfinance
history, 113 do not** (delisted/acquired — the residual *deletion* bias we can't fix
on free data). Point-in-time universe with data = 625 names, ~463 rankable members per
rebalance. The *additions* bias is removed completely.

**Survivorship-biased (today's members over all history) → point-in-time**

| metric (full history)     | mom biased | mom **PIT** | basket biased | basket **PIT** | SPY  |
|---------------------------|-----------:|------------:|--------------:|---------------:|-----:|
| total return              | 6.44x      | **2.48x**   | 3.59x         | **1.78x**      | 2.59x|
| CAGR                      | 19.2%      | 11.5%       | 14.2%         | 9.3%           | 11.8%|
| Sharpe                    | 0.96       | 0.67        | 0.85          | 0.58           | 0.72 |
| max drawdown              | −36.8%     | −34.9%      | −38.3%        | −39.7%         | −34.1%|
| long-short (gross) Sharpe | 0.28       | **0.15**    | —             | —              | —    |

**Survivorship bias was the dominant driver of everything.** Correcting it cuts the
equal-weight basket from **3.59x to 1.78x** and momentum from 6.44x to 2.48x. Two
hard conclusions:

1. **The cap-weighted index (SPY) is the honest benchmark, and it wins.** Point-in-
   time, the equal-weight member basket returns 1.78x vs SPY's 2.59x (Sharpe 0.58 vs
   0.72). Rolling-1y, the basket beats SPY in only **23%** of windows and momentum in
   only **26%**. SPY beats both ~3/4 of the time. (Sanity check: the PIT basket ≈ the
   real S&P 500 *equal-weight* index RSP, which did underperform SPY over 2015-2026 —
   so the pipeline is realistic; the biased version was not.)
2. **No tradable cross-sectional alpha over the proper benchmark.** Momentum keeps a
   small edge over *equal-weight* (Sharpe 0.67 vs 0.58; beats it in 53% of rolling
   windows) but the market-neutral long-short spread is ~0 (Sharpe 0.15) and it does
   **not** beat SPY. low-vol still loses to the basket (Sharpe 0.56). Per-year, no
   consistent winner among mom/basket; SPY takes the strong recent years (2023-25).

**This also corrects the project's own history.** Every prior backtest used *current*
membership for its baskets (the 8-name "diversified basket", the sweeps), so their
"basket beats SPY ~3x" was **survivorship-inflated**. The honest finding is the
classic one: **equal-weight does not beat cap-weight once survivorship is removed, and
price-only factors do not beat the cap-weighted index.** Reinforces the core thesis
with a correction: the benchmark to beat is **SPY (cap-weight)**, not a
current-membership basket, and it is hard to beat.

**Methodology now locked for any future backtest:** point-in-time membership mask +
SPY (cap-weight) as the benchmark. **VERDICT (price factors, corrected): no edge over
SPY.** Next legitimate avenue: **value/quality**, which need a point-in-time
*fundamentals* connector (FMP/Finnhub with as-reported/filing dates) — without that
they will look-ahead and fake alpha exactly as momentum did under survivorship bias.
Residual caveat: 113 dead names are unrecoverable on free data, so even the PIT
numbers are a mild upper bound on the downside (a little *more* of the basket/momentum
return is still survivorship).

## Phase 2c — value & quality factors, point-in-time (2026-06-24)

Built a point-in-time fundamentals connector (`data/fundamentals.py`): Finnhub
``financials-reported`` **annual** as-filed figures, indexed by **filing date** (the
10-K isn't public until filed → no look-ahead). Chosen after two dead ends: FMP's
free tier caps history to 5 quarters, and Finnhub's *quarterly* feed reports
year-to-date figures (a naive TTM triple-counts). Value uses EPS/price (the
shares-outstanding tag is missing for many filers; diluted EPS is reliable).
Covered 712/738 union names. Factors (higher = better): **earnings_yield** (value),
**roe / net_margin / gross_margin** (quality). Tested with the same point-in-time
membership mask + SPY bar (`eqa ingest-fundamentals`, `eqa factor-backtest-pit
--fundamentals`).

**Point-in-time top-quintile, net of costs** (member basket Sharpe 0.58 / 1.78x; SPY 0.72 / 2.59x)

| factor (PIT)   | Sharpe | total | CAGR  | x-sec IC (monthly) | long-short Sharpe |
|----------------|-------:|------:|------:|-------------------:|------------------:|
| earnings_yield | 0.58   | 2.23x | 10.8% | +0.021 (t 2.24)    | 0.12              |
| roe            | 0.66   | 2.23x | 10.8% | +0.016 (t 1.73)    | 0.15              |
| net_margin     | 0.58   | 1.80x | 9.4%  | +0.005 (t 0.54)    | −0.11             |
| gross_margin   | **0.75** | **3.62x** | **14.3%** | +0.028 (t 2.16) | **0.41**      |

There is a **faint but real cross-sectional signal** in value (earnings_yield, IC
t 2.24) and quality (gross_margin t 2.16, roe t 1.73) — the first non-zero ICs in the
project. But **only gross_margin nominally beats SPY**, and it does not survive
scrutiny:
- **Coverage/sector tilt:** only ~152 names report a gross-profit line (banks,
  energy, insurers, REITs don't), so its top quintile is ~30 names that are **70%
  Information Technology + Health Care** — a concentrated tech/pharma bet, not a
  clean cross-sectional factor.
- **Regime-dependent, decaying:** it beats SPY in 2015-2021 but **lags SPY in 2024,
  2025 and 2026-YTD** (−4.2% vs +7.6%) and had a deeper 2022 drawdown (−26% vs −20%).
  The full-period 0.03 Sharpe edge is front-loaded and fading.

The genuinely-positive ICs of **earnings_yield and roe do not monetize long-only**:
their portfolios match the basket and lose to SPY (the cap-weight-vs-equal-weight gap
plus higher vol eats the thin signal). Value's flat decade here matches its
well-documented 2015-2020 drought.

**VERDICT (value/quality, point-in-time): no fundamental factor robustly beats SPY.**
The one that nominally does (gross_margin) is a tech/health sector concentration that
has decayed since 2023. Combined with Phase 2/2b, the **whole cross-sectional search is
now complete and the answer is consistent: momentum, low-vol, value and quality
produce at most a faint cross-sectional IC and none delivers tradable alpha over the
cap-weighted index once survivorship is removed.** The edge remains **cap-weighted
market exposure + diversification + discipline**; SPY is the bar and simple factors
don't clear it on this universe/period.

**Capstone — sector-neutral value+quality composite.** The obvious refinement: strip
the gross_margin sector tilt (z-score each factor *within sector* per date) and combine
the factors that have a real IC (earnings_yield + roe + gross_margin) into one
composite (`fundamental_factors.sector_neutralize` / `value_quality_composite`).
Point-in-time top-quintile, net of costs:

| metric        | composite | basket | SPY   |
|---------------|----------:|-------:|------:|
| total return  | 2.80x     | 1.78x  | 2.59x |
| CAGR          | 12.4%     | 9.3%   | 11.8% |
| Sharpe        | 0.733     | 0.583  | 0.719 |
| max drawdown  | −36.3%    | −39.7% | −34.1%|

The composite is the **only construction that doesn't lose to SPY** — it edges it
(Sharpe 0.733 vs 0.719, 2.80x vs 2.59x, beats SPY in 7/12 calendar years, shallower
drawdown than the basket). **But the edge is not statistically significant**: the
composite's own cross-sectional IC is +0.007 (t = 1.21) — neutralizing weakened the
single-factor ICs (t ≈ 2.2) rather than reinforcing them, and 0.014 Sharpe over SPY is
within noise. Honest reading: **the best legitimate factor construction reaches ~SPY
parity, not a confident beat.**

So the legitimate factor avenues are now **exhausted**: momentum, low-vol, value,
quality, and their sector-neutral composite — at best they *match* the cap-weighted
index point-in-time; none beats it with statistical confidence. Effort is better spent
on the **execution/risk of the cap-weight-tracking core** than on hunting further
price/fundamental alpha on this universe.

## Phase 3 — honest core + cost-aware tilt (2026-06-24)

Acted on the verdict. **Core now defaults to holding the cap-weight benchmark**
(`config.core_strategy = spy`); the paper loop tracks SPY rather than an
(underperforming, and previously survivorship-conflated) equal-weight basket over the
broad universe. Methodology locked in `docs/METHODOLOGY.md`; the survivorship-biased
commands now print a NOTE.

Last factor question — does a **cost-aware tilt** add anything over just holding SPY?
(Point-in-time, net of costs.)

| strategy                       | Sharpe | total | turnover |
|--------------------------------|-------:|------:|---------:|
| composite, monthly rebalance   | 0.733  | 2.80x | 27.8     |
| composite, annual rebalance    | 0.742  | 2.54x | 11.5     |
| blend 50/50 basket + composite | 0.661  | 2.25x | 18.2     |
| **SPY (hold)**                 | 0.719  | 2.59x | 0.0      |

Cutting turnover (annual rebalance) recovers a hair of Sharpe but still doesn't beat
SPY on return, and the blend is *worse* than SPY (the equal-weight core drags). **A
cost-aware factor tilt adds nothing over holding SPY.** Final decision: the core is
SPY-tracking; factor research is closed as an honest null. Remaining work is execution
(IBKR), risk overlays, and monitoring — not alpha hunting on this universe.
