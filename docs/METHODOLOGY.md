# Methodology — locked rules for honest backtests

These rules are the distilled lessons of the research arc (see
`docs/PHASE1_FINDINGS.md`). Follow them for any new backtest or factor study; they
exist because breaking them produced large, fake edges that vanished under
correction.

## 1. Benchmark = SPY (cap-weight), not an equal-weight basket

The bar to beat is the **cap-weighted index (SPY)**, not a current-membership
equal-weight basket. Over 2015-2026, point-in-time equal-weight returned ~1.78x vs
SPY's ~2.59x — equal-weight **loses** to cap-weight. A strategy that beats an
equal-weight basket but not SPY has no edge worth trading. The live/paper core
default (`config.core_strategy = spy`) reflects this.

## 2. Point-in-time index membership (survivorship)

Using **today's** index members over historical data is survivorship-biased and
inflates everything (it inflated the equal-weight basket from 1.78x to **3.59x**).
On each rebalance, rank only names that were **actually in the index that day**
(`data/sp500_history.py` reconstructs membership from the Wikipedia changes log).

- The survivorship-biased commands (`backtest`, `backtest-sweep`, `factor-backtest`)
  print a NOTE and exist for convenience/sanity only.
- The honest, point-in-time command is **`factor-backtest-pit`**.
- Residual gap: ~113 delisted names have no free price history, so even the
  point-in-time numbers are a mild upper bound.

## 3. Point-in-time fundamentals (look-ahead)

Fundamentals must be lagged to the **filing date** (when the 10-K/10-Q became
public), never the fiscal period end. `data/fundamentals.py` uses Finnhub
as-reported **annual** figures indexed by `filed_date`. (Avoid the quarterly
as-reported feed: it carries year-to-date figures that break a naive TTM.)

## 4. Cross-sectional IC is per-date, not pooled

For factors, measure the **per-date cross-sectional IC** (rank-correlate factor vs
forward return *across names on each date*, then average), with effective N = number
of non-overlapping rebalance dates — not the pooled IC in `signal_eval.py`, whose
t-stat is inflated by overlapping windows and cross-sectional duplication.
`research/factor_eval.py` implements this.

## 5. Always net of costs; expect modest results

Every backtest applies fees + slippage and reports turnover. Factors are widely
known and arbitraged: expect at best parity with SPY. Where a factor *looks* like it
wins, check for a hidden sector/regime tilt and sub-period decay (gross_margin's
nominal edge was a ~70% tech/health concentration that faded post-2023).

## Bottom line

Across momentum, low-vol, value, quality and their sector-neutral composite, **no
construction beats SPY with statistical confidence** once survivorship is removed.
The edge is **cap-weight market exposure + diversification + discipline**; effort is
better spent on execution/cost/risk of a cap-weight-tracking core than on hunting
price/fundamental alpha on this universe.
