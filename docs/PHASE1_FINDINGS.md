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

- Signal combination method: weights / ensemble / meta-model.
- Backtest engine: custom event-driven vs backtrader.
- API keys for: LLM sentiment (Anthropic), fundamentals/news (Finnhub/FMP),
  macro (FRED), execution (Alpaca).
