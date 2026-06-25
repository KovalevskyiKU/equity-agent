# Crypto findings

Same discipline as the equity arc (`docs/PHASE1_FINDINGS.md`): the bar is **hold
BTC** (the crypto SPY), everything is net of costs, and we trust out-of-sample /
broad-parameter robustness, not a single tuned number.

## Setup

- 20 liquid coins via yfinance (`-USD`), daily, 365-day year (24/7). BTC/LTC from
  2015; most from 2017-11; SOL/DOT/AVAX/AAVE from 2020.
- Costs assumed **higher than equities**: 10 bps fee + 20 bps slippage (taker +
  wider spreads).
- **Survivorship bias is brutal** (worse than equities): these are *survivors* —
  many past top-20 coins are dead (LUNA, FTT, …) and absent. UNI already shows it
  (yfinance history ends 2025-04). Treat cross-sectional alt results as an upper
  bound; BTC itself is the all-time winner, so "hold BTC" is a survivor benchmark.

## Headline (hold-BTC vs managed, 365-day, net of costs)

| strategy            | total | CAGR  | Sharpe | max DD | Calmar | turnover |
|---------------------|------:|------:|-------:|-------:|-------:|---------:|
| hold-BTC            | 189x  | 57.9% | 1.02   | −83.4% | 0.69   | 0        |
| **trend 20/100**    | 192x  | 58.1% | **1.14** | **−68.1%** | **0.85** | 48 |
| vol-target 50%      | 64x   | 43.8% | 0.97   | −70.3% | 0.62   | 38       |
| alt-momentum top-Q  | 6x    | 18.9% | 0.61   | −91.7% | 0.21   | 105      |

## What works, what doesn't

**Trend-following BTC beats buy-and-hold — robustly.** A long/flat SMA cross lifts
Sharpe (1.14 vs 1.02), Calmar (0.85 vs 0.69) and cuts max drawdown (−68% vs −83%)
with the *same* total return. It is **not overfit to 20/100**: across a fast∈{10,20,30,50}
× slow∈{50,100,150,200} grid, **every** combination beats hold-BTC on Sharpe (1.05–1.24).
Economic logic: crypto has strong persistent trends and brutal sustained bears
(−80%); a trend filter sits out the worst of the bears while staying long the bull
runs. **This is the opposite of equities**, where nothing beat buy-and-hold SPY.

**Vol-targeting does NOT help (opposite of equities).** Scaling BTC exposure by
realized vol lowers return and Sharpe (0.97 vs 1.02) and only reduces drawdown —
because in crypto the **big rallies are also high-vol**, so cutting exposure in high
vol throws away the upside (2017 +520% vs +1369%, 2020 +166% vs +303%). Vol is
symmetric here; trend (which keys off *direction*, not magnitude) is the right tool.

**Cross-sectional alt-momentum: no edge over hold-BTC.** Top-quantile alt momentum
returns 6x (Sharpe 0.61) vs hold-BTC 189x (1.02), with −92% drawdowns and a
cross-sectional IC ≈ 0 — net of crypto costs on a survivor universe, the documented
crypto momentum anomaly does not survive here.

## Verdict & caveats

**Crypto is a genuinely different result from equities: trend-following BTC is a
real, robust improvement over buy-and-hold.** But hold the caveats before trusting it
with capital:
- One asset (BTC), one historical path, ~3 cycles — strategy-level overfitting risk
  remains even though the MA *parameter* grid is robust.
- Survivorship: BTC is the all-time winner; trend-on-the-winner is favourable.
- Costs/whipsaw: choppy ranges (e.g. 2024-25) whipsaw trend; the full-period result
  holds but recent sub-periods are weaker.

**Reproduce:** `eqa ingest-crypto` then `eqa backtest-crypto`.

**Next (when the evaluation skill is connected):** walk-forward / out-of-sample test
of the trend rule; with **perps**, short the downtrend instead of going flat (and
test funding-carry); add a small alt-trend sleeve; on-chain signals (paid data) as
the frontier. Decide execution venue (Binance/Coinbase) for a crypto broker adapter.
