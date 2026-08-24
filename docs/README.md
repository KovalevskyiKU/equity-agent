# Documentation

Where to start depends on what you want:

| I want to… | read |
|------------|------|
| know what to actually run | [STRATEGY.md](STRATEGY.md) |
| understand a term I don't know | [GLOSSARY.md](GLOSSARY.md) |
| add a backtest without fooling myself | [METHODOLOGY.md](METHODOLOGY.md) |
| see every equity hypothesis and its result | [PHASE1_FINDINGS.md](PHASE1_FINDINGS.md) |
| see the crypto work | [CRYPTO_FINDINGS.md](CRYPTO_FINDINGS.md) |

---

## The documents

### [STRATEGY.md](STRATEGY.md) — what to run
The actionable synthesis: hold SPY for equities (optionally with a volatility
overlay), trend-managed BTC for crypto (optionally with funding carry), and an
explicit list of what is **not** proven.

### [PHASE1_FINDINGS.md](PHASE1_FINDINGS.md) — the equity research log
The complete chronological record, including the retractions. Roughly:

| phase | what happened |
|-------|---------------|
| Phase 1 | Technical features, Kronos, macro/rates/credit signals — only volatility looked real, and it died out-of-sample |
| Phase 2 | Expanded to the S&P 500; cross-sectional factors (momentum, low-vol, value, quality) |
| Phase 2b | **Point-in-time index membership** — survivorship bias was inflating results ~2x; earlier conclusions rewritten |
| Phase 2c | Point-in-time fundamentals; value/quality still lose to SPY |
| Phase 3 | Core switched to tracking SPY; vol-target overlay validated as drawdown control |
| Phase 4 | **The measurement audit**: alpha/beta regression and dollar-neutral long-short added — three method gaps closed |
| Phase 4b | Four untested classics; **net share issuance** is the one real signal |
| Phase 4c | Quarterly issuance data tested — weaker than annual; the decay is genuine |
| Phase 4d | Mid- and small-caps (S&P 400/600), point-in-time — no size premium, no factor premium |

### [CRYPTO_FINDINGS.md](CRYPTO_FINDINGS.md) — the crypto research log
Trend-following, volatility targeting, cross-sectional alt momentum and funding
carry, all against the honest bar of simply holding BTC.

### [METHODOLOGY.md](METHODOLOGY.md) — the rules
Short and load-bearing. Every rule exists because breaking it produced a fake edge
here at some point.

### [GLOSSARY.md](GLOSSARY.md) — terms and data costs
Plain-language definitions of every term used in this project (Sharpe, alpha, IC,
survivorship bias, long-short, funding carry…), plus a buyer's guide to paid data
with prices and an honest assessment of what each would change.

---

## Historical

[NEXT_SESSION_ALPHA.md](NEXT_SESSION_ALPHA.md) is the kickoff brief that started the
cross-sectional factor work. Kept for provenance — its plan has been executed and its
conclusions superseded by the phase write-ups above.

---

## Reproducing the numbers

Every headline figure can be regenerated:

```bash
eqa research-report          # factor verdict + risk overlay -> data/reports/
eqa factor-backtest-pit --fundamentals   # the point-in-time equity test
eqa backtest-crypto          # hold-BTC vs trend / vol-target / alt-momentum
eqa crypto-funding           # delta-neutral funding carry
```

Note that the index-membership data is cached to `data/*_changes.csv` on first fetch,
because the upstream Wikipedia page has already been restructured once — which
silently broke reproducibility until it was caught.
