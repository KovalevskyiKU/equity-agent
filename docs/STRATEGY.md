# Strategy — what to actually run

The actionable synthesis of the research. Full evidence: `docs/PHASE1_FINDINGS.md`
(equities), `docs/CRYPTO_FINDINGS.md` (crypto), `docs/METHODOLOGY.md` (rules).

## Philosophy (the bar)

Beat the **cap-weighted benchmark**, net of costs, **survivorship-corrected**, out
of sample. Anything that only beats a survivorship-biased basket, or only in-sample,
doesn't count. Most "edges" die against this bar — and that's the point.

## Equities — core = hold SPY

No price or fundamental factor (momentum, low-vol, value, quality, or a
sector-neutral composite) beats SPY on a risk-adjusted basis once survivorship and
dividends are handled honestly. So:

- **Core: hold SPY** (`config.core_strategy = spy`). The honest, hard-to-beat default.
- **Optional risk overlay** (`risk_overlay: vol_target`, ~15%): for a risk-averse
  mandate — lifts Sharpe 0.82→0.89, Calmar 0.41→0.57, ~halves the max drawdown, at a
  ~2.5pp/yr return cost. Crash insurance, not extra return. Off by default.
- Run: `eqa daily` (ingest → rebalance to core → monitor); `eqa monitor`;
  `eqa backtest-overlay`. Live: `eqa live-run` (IBKR, dry-run by default).

## Crypto — core = trend-managed BTC, optional carry

Bar = **hold BTC**. As in equities, beating it on *return* is hard; the wins are
risk control and structural carry.

- **Core: trend-managed BTC** (long/flat SMA cross). Out-of-sample it ~ties hold-BTC
  risk-adjusted (Sharpe 1.07 vs 1.02) while cutting the worst drawdown (−72% vs
  −83%) — a **drawdown-control** tool, at the cost of ~half the return. Choose it
  over hold-BTC only if an −83% drawdown is unacceptable.
- **Funding carry** (optional, perps): a delta-neutral short-perp/long-spot yield,
  structurally positive (~10%/yr historically, positive every year) but **decaying**
  (2021 ~31% → 2026 ~1%). A cash-like sleeve, not upside; real operational/basis risk.
- **Avoid:** vol-targeting BTC, long/short trend, cross-sectional alt-momentum,
  alt-trend — none beat hold-BTC.
- Run: `eqa ingest-crypto`, `eqa backtest-crypto`, `eqa crypto-funding`. Live:
  `eqa crypto-live-run` (Binance, dry-run by default).

## What is NOT proven (be honest)

- No alpha over either benchmark on return. The edges are drawdown control (equity
  vol-overlay, crypto trend) and crypto funding carry — all modest, some decaying.
- Crypto results sit on one asset / few cycles / a survivor universe; treat as
  suggestive, validate further before sizing.
- Execution adapters (IBKR, Binance) are dry-run-tested only — live needs keys/gateway.
- On-chain signals and a multi-perp carry book are unexplored (need paid data).

## Bottom line

The honest, runnable system: **hold the cap-weighted index** (SPY) for equities and
**trend-managed BTC** for crypto, each with an **optional risk/carry sleeve**, all
benchmarked and monitored. Effort beyond this should go to execution, cost, and risk
discipline — not to hunting more factor alpha on these universes.
