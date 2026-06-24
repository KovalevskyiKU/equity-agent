"""Cross-sectional factor research — per-date IC, the right tool for equity factors.

The existing :mod:`signal_eval` **pools** observations across names and dates: it
answers "does a higher feature value predict a higher forward return overall?".
That is the wrong question for *cross-sectional* alpha, which is about **which
name out/under-performs which** on a given day. The honest measure there is the
**per-date cross-sectional IC**: on each rebalance date, rank-correlate the
factor across names with the forward return across names, then average those
daily ICs over time.

Everything here is causal: factors at date *t* use only bars up to *t*; the
forward-return target is the only forward-looking quantity and is never an input.

Factor sign convention: every factor is oriented so that **higher = expected
higher forward return** (e.g. the low-volatility factor is the *negative* of
trailing realised vol), so a positive IC always means "the factor works".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Targets and price-only factors (date x symbol panels)
# --------------------------------------------------------------------------- #
def forward_return_panel(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward return over the next ``horizon`` bars: close[t+h]/close[t]-1 (the target)."""
    return close.shift(-horizon) / close - 1.0


def momentum_factor(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Classic 12-1 month momentum: return from t-lookback to t-skip (skip recent month).

    Skipping the most recent ~month avoids the well-known short-term reversal that
    contaminates raw 12-month momentum. Price-only — no fundamentals, no look-ahead.
    """
    return close.shift(skip) / close.shift(lookback) - 1.0


def low_vol_factor(close: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """Low-volatility factor: NEGATIVE trailing realised vol (low-vol anomaly).

    Negated so the sign convention holds (higher factor = lower vol = expected
    higher risk-adjusted return). Price-only.
    """
    daily = close.pct_change()
    vol = daily.rolling(lookback).std() * np.sqrt(TRADING_DAYS)
    return -vol


PRICE_FACTORS = {
    "momentum_12_1": momentum_factor,
    "low_vol": low_vol_factor,
}


# --------------------------------------------------------------------------- #
# Rebalance calendar
# --------------------------------------------------------------------------- #
def month_end_dates(index: pd.Index) -> pd.Index:
    """Last available trading day of each calendar month in ``index`` (monthly rebalance).

    Works whether the index holds python ``date`` objects (as stored) or datetimes.
    """
    dt = pd.DatetimeIndex(pd.to_datetime(index))
    periods = dt.to_period("M")
    keep = ~periods.duplicated(keep="last")
    return index[keep]


# --------------------------------------------------------------------------- #
# Per-date cross-sectional IC
# --------------------------------------------------------------------------- #
def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation of two aligned series (numpy only, no scipy)."""
    ra, rb = a.rank(), b.rank()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cross_sectional_ic(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    dates: pd.Index,
    min_names: int = 30,
) -> pd.DataFrame:
    """Per-date cross-sectional Spearman IC of ``factor`` vs ``fwd`` over ``dates``.

    Returns a frame indexed by date with columns ``ic`` and ``n_names`` (the
    number of names with both a factor and a forward return that day). Dates with
    fewer than ``min_names`` names are skipped.
    """
    rows: list[dict[str, float]] = []
    for d in dates:
        if d not in factor.index or d not in fwd.index:
            continue
        pair = pd.concat([factor.loc[d], fwd.loc[d]], axis=1).dropna()
        if len(pair) < min_names:
            continue
        ic = _spearman(pair.iloc[:, 0], pair.iloc[:, 1])
        if np.isfinite(ic):
            rows.append({"date": d, "ic": ic, "n_names": len(pair)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


@dataclass
class ICSummary:
    mean_ic: float
    std_ic: float
    ic_ir: float  # mean / std of the per-date IC series (information ratio of the IC)
    t_stat: float  # ic_ir * sqrt(n_dates) — honest only if dates are non-overlapping
    hit_rate: float  # fraction of dates with IC > 0
    n_dates: int
    mean_n_names: float


def summarize_ic(ic_table: pd.DataFrame) -> ICSummary:
    """Collapse a per-date IC table into a single honest summary.

    The t-stat treats each date as one independent observation, which is valid
    only when the rebalance dates are non-overlapping (step >= horizon). For
    monthly rebalance with a ~monthly horizon that holds.
    """
    if ic_table.empty:
        return ICSummary(*( [float("nan")] * 5 + [0, float("nan")]))  # type: ignore[arg-type]
    ic = ic_table["ic"]
    n = len(ic)
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    ir = mean / std if std > 0 else float("nan")
    t = ir * np.sqrt(n) if np.isfinite(ir) else float("nan")
    return ICSummary(
        mean_ic=mean,
        std_ic=std,
        ic_ir=ir,
        t_stat=t,
        hit_rate=float((ic > 0).mean()),
        n_dates=n,
        mean_n_names=float(ic_table["n_names"].mean()),
    )


def evaluate_factor(
    close: pd.DataFrame,
    factor: pd.DataFrame,
    horizon: int,
    rebalance: str = "monthly",
    min_names: int = 30,
) -> tuple[pd.DataFrame, ICSummary]:
    """Per-date IC table + summary for one factor over the given rebalance calendar.

    ``rebalance='monthly'`` uses month-end dates (non-overlapping when horizon ~21,
    so the t-stat is honest). ``rebalance='daily'`` uses every date (overlapping
    windows -> inflated t-stat; for contrast only).
    """
    fwd = forward_return_panel(close, horizon)
    dates = month_end_dates(close.index) if rebalance == "monthly" else close.index
    table = cross_sectional_ic(factor, fwd, dates, min_names=min_names)
    return table, summarize_ic(table)


def run_cross_sectional_ic(
    universe: list[str], horizon: int = 21, min_names: int = 30
) -> tuple[dict[str, dict[str, ICSummary]], int]:
    """Cross-sectional IC for every price-only factor, monthly (honest) + daily (contrast).

    Returns ``({factor: {"monthly": ICSummary, "daily": ICSummary}}, n_symbols)``.
    """
    from ..backtest.panels import load_price_panels

    _, close = load_price_panels(universe)
    if close.empty:
        return {}, 0
    out: dict[str, dict[str, ICSummary]] = {}
    for name, fn in PRICE_FACTORS.items():
        factor = fn(close)
        _, monthly = evaluate_factor(close, factor, horizon, "monthly", min_names)
        _, daily = evaluate_factor(close, factor, horizon, "daily", min_names)
        out[name] = {"monthly": monthly, "daily": daily}
    return out, int(close.shape[1])
