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

## Phase 2 — out-of-sample reality check (perps, carry, alts)

**Walk-forward tempers the trend result.** Phase 1's "trend beats hold" used a
hindsight-good 20/100. Picked honestly out-of-sample — anchored walk-forward, each
year trading the MA params that were best on prior data only:

| BTC strategy             | Sharpe | total | max DD | Calmar |
|--------------------------|-------:|------:|-------:|-------:|
| hold-BTC                 | 1.02   | 189x  | −83%   | 0.69   |
| trend 20/100 (in-sample) | 1.14   | 192x  | −68%   | 0.85   |
| **trend walk-forward (OOS)** | 1.07 | 95x | −72%   | 0.67   |

OOS, trend's Sharpe edge shrinks to marginal (1.07 vs 1.02), Calmar ties (0.67 vs
0.69), and it gives up **half the total return** (95x vs 189x) while cutting the
drawdown (−72% vs −83%). So honestly it is a **drawdown-reduction tool** (like the
equity vol-overlay), not free outperformance — the in-sample edge was partly
param-hindsight.

**Shorting the downtrend (perps) does NOT help.** Long/short MA-cross underperforms
both hold and long/flat (Sharpe 0.70–0.83 vs 1.02): crypto bears are choppy with
violent counter-trend rallies that squeeze the shorts. Going flat beats going short.

**Alt-trend does NOT beat hold-BTC.** Equal-weight trend across the alts: Sharpe 0.98
vs 1.02, with a deeper −93% drawdown. (An "alt equal-weight buy-hold" prints absurd
returns — a pure survivorship artifact of daily-rebalancing the surviving alts; not
real.) Trend doesn't generalize off BTC on this universe.

**Funding carry is the most robust edge — structural, not directional.** A
delta-neutral short-perp / long-spot collects perp funding (Binance free API,
`eqa crypto-funding`):

| perp     | gross carry | net (~2%/yr costs) | % positive | period |
|----------|------------:|-------------------:|-----------:|--------|
| BTCUSDT  | 11.7%/yr    | ~9.7%/yr           | 85%        | 2019-09→2026 |
| ETHUSDT  | 14.1%/yr    | ~12.1%/yr          | 86%        | 2019-11→2026 |

Positive **every single year** (longs persistently pay to be levered long). But it is
**decaying** as the market matures and arbs it away: BTC ~31%/yr (2021) → ~5% (2025)
→ ~1% (2026). It is a ~10%/yr cash-and-carry *yield*, not upside, and carries real
operational/basis/liquidation risk.

## Crypto verdict (Phase 1 + 2)

Two real edges survive honest testing, both modest:
1. **Trend-following BTC (long/flat)** — a drawdown-control tool: ~ties hold-BTC
   risk-adjusted OOS while cutting the worst drawdown, at the cost of ~half the
   return. Use it if a −83% drawdown is unacceptable.
2. **Funding carry** — a structural ~10%/yr delta-neutral yield, positive every year,
   but shrinking toward low single digits as it gets arbitraged.

Everything directional — vol-targeting, long/short trend, cross-sectional
alt-momentum, alt-trend — fails to beat buy-and-hold BTC. As in equities, beating the
benchmark on *return* is hard; the wins are in risk control and structural carry.

**Next (frontier, needs the evaluation skill / paid data):** on-chain signals
(Glassnode/Coin Metrics); a delta-neutral carry book across multiple perps with
funding-timing; a crypto execution adapter (Binance/Coinbase). Reproduce Phase 2 with
`eqa backtest-crypto` and `eqa crypto-funding`.
