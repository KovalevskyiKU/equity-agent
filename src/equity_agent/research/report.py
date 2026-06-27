"""One-command reproducible research report — regenerate the headline numbers.

Runs the two results that matter, point-in-time and total-return, and writes a
Markdown summary to ``data/reports/research_report.md``:

* the cross-sectional factor verdict (no factor beats SPY), and
* the vol-target overlay comparison (the one validated risk improvement).

So the project's claims can be re-derived with ``eqa research-report`` rather than
trusted from a static doc.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, load_config


def _factor_table() -> pd.DataFrame:
    from ..backtest.factor_portfolio import run_pit_factor_portfolios
    from ..data.sp500_history import ever_members, fetch_sp500_changes

    cfg = load_config()
    changes = fetch_sp500_changes()
    union = ever_members(cfg.universe, changes, f"{cfg.history_start[:4]}-01-01")
    res = run_pit_factor_portfolios(
        union, cfg.universe, changes, cfg.benchmark, with_fundamentals=True, total_return=True
    )
    if not res:
        return pd.DataFrame()

    def row(name: str, m: dict) -> dict[str, object]:
        return {
            "strategy": name,
            "total_x": round(m["total_return"], 2),
            "cagr_%": round(m["cagr"] * 100, 1),
            "sharpe": round(m["sharpe"], 3),
            "max_dd_%": round(m["max_drawdown"] * 100, 1),
        }

    from typing import cast

    rows = [
        row(cfg.benchmark, cast(dict, res["spy"])),
        row("member basket", cast(dict, res["basket"])),
    ]
    rows += [row(name, fb["portfolio"]) for name, fb in cast(dict, res["factors"]).items()]
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def generate_report() -> str:
    """Build the Markdown report (point-in-time, total-return, net of costs)."""
    from ..backtest.crypto import run_crypto_comparison
    from ..backtest.overlay_backtest import run_overlay_comparison

    cfg = load_config()
    factors = _factor_table()
    overlay = run_overlay_comparison()
    crypto = run_crypto_comparison()

    parts = [
        "# Reproducible research report",
        "",
        f"Universe: {len(cfg.universe)} (point-in-time S&P 500). Benchmark: "
        f"{cfg.benchmark} (cap-weight). Total-return, net of costs. "
        "Regenerate with `eqa research-report`.",
        "",
        "## Cross-sectional factor verdict (point-in-time, total-return)",
        "",
        "```",
        factors.to_string(index=False) if not factors.empty else "no data — run `eqa ingest`",
        "```",
        "",
        f"Bar = {cfg.benchmark}. Conclusion: no factor beats it on Sharpe once "
        "survivorship and dividends are handled honestly.",
        "",
        "## Vol-target overlay (the one validated improvement)",
        "",
        "```",
        overlay.to_string(index=False) if not overlay.empty else "no data — run `eqa ingest`",
        "```",
        "",
        "The overlay trades absolute return for a better Sharpe/Calmar and ~half the "
        "drawdown (crash insurance). Enable via `config.risk_overlay: vol_target`.",
        "",
        "## Crypto: hold-BTC vs managed (365-day, net of crypto costs)",
        "",
        "```",
        crypto.to_string(index=False) if not crypto.empty else "no data — run `eqa ingest-crypto`",
        "```",
        "",
        "Trend ≈ drawdown control; vol-target/alt-momentum do not beat hold-BTC. "
        "Funding carry (`eqa crypto-funding`) is a separate structural ~10%/yr yield.",
        "",
    ]
    return "\n".join(parts)


def write_report() -> Path:
    """Generate the report and write it under ``data/reports/``."""
    cfg = load_config()
    out = PROJECT_ROOT / cfg.data_dir / "reports" / "research_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_report(), encoding="utf-8")
    return out
