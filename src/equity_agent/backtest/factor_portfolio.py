"""Monthly-rebalanced cross-sectional factor portfolios, backtested vs the basket.

Builds causal target-weight panels from a factor (date x symbol), runs them
through the shared event-driven :mod:`engine` (next-open fills, fees, slippage),
and compares against the **equal-weight basket on the same selectable universe**
(the honest bar to beat, per the research brief) and SPY.

Two readouts:
* **Top-quantile long-only** portfolio — tradable, routed through the cash engine
  net of costs. This is the real test.
* **Long-short top-minus-bottom spread** — the academic factor return. Computed
  analytically (idealized: no borrow/margin/financing modeled), so treat it as a
  signal-quality upper bound, not a P&L claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..research.factor_eval import PRICE_FACTORS, month_end_dates
from .engine import BacktestConfig, run_backtest
from .metrics import return_summary
from .strategy import single_asset


def quantile_long_weights(
    factor: pd.DataFrame, rebal_dates: pd.Index, q: float = 0.2, min_names: int = 30
) -> pd.DataFrame:
    """Equal-weight the top-``q`` fraction of names by factor on each rebalance date.

    Rows on non-rebalance dates are left NaN so the engine holds (no trade) between
    rebalances; on a rebalance date, selected names get 1/k and the rest 0 (sold).
    Rebalance dates with fewer than ``min_names`` rankable names are skipped (held).
    """
    w = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    for d in rebal_dates:
        if d not in factor.index:
            continue
        row = factor.loc[d].dropna()
        if len(row) < min_names:
            continue
        k = max(1, int(round(len(row) * q)))
        top = row.nlargest(k).index
        w.loc[d] = 0.0
        w.loc[d, top] = 1.0 / k
    return w


def equal_weight_monthly(
    factor: pd.DataFrame, rebal_dates: pd.Index, min_names: int = 30
) -> pd.DataFrame:
    """Equal weight across *all rankable names* on each rebalance date (the fair basket).

    Same selectable set as the factor portfolio (names with a valid factor value),
    so a head-to-head isolates selection skill rather than rebalance frequency.
    """
    w = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    for d in rebal_dates:
        if d not in factor.index:
            continue
        row = factor.loc[d].dropna()
        if len(row) < min_names:
            continue
        w.loc[d] = 0.0
        w.loc[d, row.index] = 1.0 / len(row)
    return w


def long_short_spread(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    rebal_dates: pd.Index,
    q: float = 0.2,
    min_names: int = 30,
) -> pd.DataFrame:
    """Realized top-minus-bottom quantile return over each rebalance period (gross).

    Idealized factor return: equal-weight long the top-q, short the bottom-q on
    each rebalance date, hold to the next. No financing/borrow/turnover cost — an
    upper bound on the spread's quality.
    """
    rd = [d for d in rebal_dates if d in factor.index and d in close.index]
    rows: list[dict[str, float]] = []
    for i in range(len(rd) - 1):
        d, nxt = rd[i], rd[i + 1]
        row = factor.loc[d].dropna()
        if len(row) < min_names:
            continue
        k = max(1, int(round(len(row) * q)))
        top, bot = row.nlargest(k).index, row.nsmallest(k).index
        period = close.loc[nxt] / close.loc[d] - 1.0
        long_r, short_r = float(period[top].mean()), float(period[bot].mean())
        rows.append({"date": nxt, "ls": long_r - short_r, "long": long_r, "short": short_r})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def spread_summary(ls: pd.DataFrame, periods_per_year: int = 12) -> dict[str, float]:
    """Annualized mean / vol / Sharpe / hit-rate of a long-short spread return series."""
    if ls.empty:
        return {"ann_return": float("nan"), "ann_vol": float("nan"),
                "sharpe": float("nan"), "hit_rate": float("nan"), "n_periods": 0}
    r = ls["ls"]
    mean, std = float(r.mean()), float(r.std(ddof=1))
    return {
        "ann_return": mean * periods_per_year,
        "ann_vol": std * np.sqrt(periods_per_year),
        "sharpe": (mean / std * np.sqrt(periods_per_year)) if std > 0 else float("nan"),
        "hit_rate": float((r > 0).mean()),
        "n_periods": len(r),
    }


def membership_basket_weights(
    tradable: pd.DataFrame, rebal_dates: pd.Index, min_names: int = 30
) -> pd.DataFrame:
    """Equal weight across all point-in-time members with a price on each rebalance date.

    ``tradable`` is a boolean (date x symbol) panel = "was an index member AND has a
    price that day". This is the honest point-in-time basket (the bar to beat),
    shared across factors so the comparison is apples-to-apples.
    """
    w = pd.DataFrame(np.nan, index=tradable.index, columns=tradable.columns)
    for d in rebal_dates:
        if d not in tradable.index:
            continue
        row = tradable.loc[d]
        cols = row[row].index
        if len(cols) < min_names:
            continue
        w.loc[d] = 0.0
        w.loc[d, cols] = 1.0 / len(cols)
    return w


@dataclass
class FactorBacktest:
    name: str
    portfolio: dict[str, float]  # top-quantile long-only metrics (net of costs)
    basket: dict[str, float]  # monthly equal-weight basket on the same set
    long_short: dict[str, float]  # idealized top-minus-bottom spread
    turnover: float
    n_trades: int


def backtest_factor(
    name: str,
    factor: pd.DataFrame,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    *,
    q: float = 0.2,
    min_names: int = 30,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> FactorBacktest:
    """Backtest one factor's top-quantile portfolio vs the fair basket + long-short spread."""
    rebal = month_end_dates(close_px.index)
    cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)

    port_w = quantile_long_weights(factor, rebal, q=q, min_names=min_names)
    basket_w = equal_weight_monthly(factor, rebal, min_names=min_names)
    port = run_backtest(open_px, close_px, port_w, cfg)
    basket = run_backtest(open_px, close_px, basket_w, cfg)
    ls = long_short_spread(factor, close_px, rebal, q=q, min_names=min_names)

    return FactorBacktest(
        name=name,
        portfolio=return_summary(port.returns),
        basket=return_summary(basket.returns),
        long_short=spread_summary(ls),
        turnover=port.turnover,
        n_trades=port.n_trades,
    )


def run_factor_portfolios(
    universe: list[str],
    benchmark: str,
    *,
    q: float = 0.2,
    min_names: int = 30,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> dict[str, object]:
    """Load panels, backtest every price-only factor, and add the SPY benchmark.

    Returns ``{"factors": {name: FactorBacktest}, "spy": metrics, "n_symbols": int}``.
    """
    from .panels import load_price_panels

    open_px, close_px = load_price_panels(universe)
    if close_px.empty:
        return {}

    factors = {
        name: fn(close_px) for name, fn in PRICE_FACTORS.items()
    }
    results = {
        name: backtest_factor(
            name, fac, open_px, close_px,
            q=q, min_names=min_names, fee_bps=fee_bps, slippage_bps=slippage_bps,
        )
        for name, fac in factors.items()
    }

    open_b, close_b = load_price_panels([benchmark])
    spy_cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)
    spy = run_backtest(open_b, close_b, single_asset(close_b, benchmark), spy_cfg)

    return {
        "factors": results,
        "spy": return_summary(spy.returns),
        "n_symbols": int(close_px.shape[1]),
    }


def run_pit_factor_portfolios(
    union: list[str],
    current_members: list[str],
    changes: pd.DataFrame,
    benchmark: str,
    *,
    q: float = 0.2,
    min_names: int = 30,
    fee_bps: float = 1.0,
    slippage_bps: float = 5.0,
    with_fundamentals: bool = False,
) -> dict[str, object]:
    """Point-in-time factor backtest: rank only names that were index members that day.

    Masks the factor panel with reconstructed historical membership (kills the
    additions/survivorship bias), backtests each factor's top-quantile portfolio vs
    a **shared** point-in-time member basket + SPY. Returns the same metrics plus
    ``mean_members`` (average rankable members per rebalance) for transparency.
    """
    from ..data.sp500_history import membership_mask
    from .panels import load_price_panels

    open_px, close_px = load_price_panels(union)
    if close_px.empty:
        return {}

    mask = membership_mask(close_px.index, current_members, changes)
    mask = mask.reindex(index=close_px.index, columns=close_px.columns, fill_value=False)
    tradable = mask & close_px.notna()
    rebal = month_end_dates(close_px.index)
    cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)

    basket = run_backtest(
        open_px, close_px, membership_basket_weights(tradable, rebal, min_names), cfg
    )

    factor_panels: dict[str, pd.DataFrame] = {
        name: fn(close_px) for name, fn in PRICE_FACTORS.items()
    }
    if with_fundamentals:
        from ..research.fundamental_factors import build_fundamental_panels

        factor_panels.update(build_fundamental_panels(list(close_px.columns), close_px))

    results: dict[str, dict[str, object]] = {}
    for name, factor in factor_panels.items():
        factor = factor.where(mask)
        port = run_backtest(
            open_px, close_px, quantile_long_weights(factor, rebal, q=q, min_names=min_names), cfg
        )
        ls = long_short_spread(factor, close_px, rebal, q=q, min_names=min_names)
        results[name] = {
            "portfolio": return_summary(port.returns),
            "long_short": spread_summary(ls),
            "turnover": port.turnover,
            "n_trades": port.n_trades,
        }

    open_b, close_b = load_price_panels([benchmark])
    spy = run_backtest(open_b, close_b, single_asset(close_b, benchmark), cfg)

    rebal_in = [d for d in rebal if d in tradable.index]
    mean_members = float(tradable.loc[rebal_in].sum(axis=1).mean())
    return {
        "factors": results,
        "basket": return_summary(basket.returns),
        "spy": return_summary(spy.returns),
        "n_symbols": int(close_px.shape[1]),
        "mean_members": mean_members,
    }
