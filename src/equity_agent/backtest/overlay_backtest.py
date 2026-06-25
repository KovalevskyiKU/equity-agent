"""SPY vs vol-target-SPY — the one validated risk improvement, as a reusable backtest.

Shared by the CLI (`eqa backtest-overlay`) and the dashboard so the comparison is
computed one way. Total-return, net of costs by default (the honest setup).
"""

from __future__ import annotations

import pandas as pd

from ..config import load_config
from ..risk.overlay import vol_target_exposure_series
from .engine import BacktestConfig, run_backtest
from .metrics import return_summary
from .panels import load_price_panels
from .strategy import single_asset

_METRICS = ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar")


def vol_target_index_weights(
    close_b: pd.DataFrame,
    benchmark: str,
    *,
    target_vol: float = 0.15,
    lookback: int = 20,
    band: float = 0.05,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Daily benchmark-exposure weights for a vol-target overlay (rest in cash)."""
    rets = close_b[benchmark].pct_change().fillna(0.0)
    exposure = vol_target_exposure_series(
        rets, target_vol=target_vol, lookback=lookback, band=band, trading_days=trading_days
    )
    return pd.DataFrame({benchmark: exposure})


def run_overlay_comparison(
    target_vols: tuple[float, ...] = (0.10, 0.15, 0.20),
    *,
    benchmark: str | None = None,
    band: float = 0.05,
    lookback: int = 20,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
    total_return: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Benchmark buy-hold vs vol-target overlay across target vols. One metrics table.

    ``benchmark`` defaults to the equity benchmark; pass a crypto symbol with
    ``trading_days=365`` for the 24/7 crypto calendar.
    """
    cfg = load_config()
    bench = benchmark or cfg.benchmark
    open_b, close_b = load_price_panels([bench], total_return=total_return)
    if close_b.empty:
        return pd.DataFrame()
    btc = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)

    def row(name: str, res: object) -> dict[str, object]:
        m = return_summary(res.returns, periods_per_year=trading_days)  # type: ignore[attr-defined]
        out: dict[str, object] = {"strategy": name}
        out.update({k: round(float(m[k]), 3) for k in _METRICS})
        out["turnover"] = round(float(res.turnover), 1)  # type: ignore[attr-defined]
        return out

    bh = run_backtest(open_b, close_b, single_asset(close_b, bench), btc)
    rows = [row(f"{bench} buy-hold", bh)]
    for tv in target_vols:
        w = vol_target_index_weights(
            close_b, bench, target_vol=tv, lookback=lookback, band=band, trading_days=trading_days
        )
        rows.append(row(f"vol-target {int(tv * 100)}%", run_backtest(open_b, close_b, w, btc)))
    return pd.DataFrame(rows)
