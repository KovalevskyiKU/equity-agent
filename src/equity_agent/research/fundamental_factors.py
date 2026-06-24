"""Value/quality cross-sectional factors from point-in-time annual fundamentals.

Each fundamental series is indexed by **filing date** (when the 10-K became public)
and forward-filled onto the daily trading calendar — so the factor value on any
date uses only what was known then (no look-ahead). Sign convention: higher =
expected higher return.

* **earnings_yield** (value) = annual diluted EPS / price. Higher = cheaper.
* **roe** (quality) = net income / shareholders' equity (equity > 0 only).
* **net_margin** (quality) = net income / revenue.
* **gross_margin** (quality) = gross profit / revenue (NaN for banks/insurers that
  don't report a gross-profit line — they simply drop out of that factor's ranking).
"""

from __future__ import annotations

import pandas as pd

from ..data.fundamentals import load_fundamentals


def _to_daily(values: pd.Series, filed: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Step series keyed by filing date, forward-filled onto the daily calendar."""
    ser = pd.Series(values.to_numpy(), index=pd.to_datetime(filed.to_numpy()))
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    return ser.reindex(daily_index, method="ffill")


def build_fundamental_panels(
    symbols: list[str], close_px: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Build value/quality factor panels (date x symbol) aligned to ``close_px``.

    Returns ``{factor_name: panel}``; symbols without stored fundamentals are absent
    (NaN after reindex) and simply not ranked on those dates.
    """
    daily = pd.to_datetime(close_px.index)
    eps_d: dict[str, pd.Series] = {}
    roe_d: dict[str, pd.Series] = {}
    nm_d: dict[str, pd.Series] = {}
    gm_d: dict[str, pd.Series] = {}

    for s in symbols:
        df = load_fundamentals(s)
        if df.empty:
            continue
        df = df.sort_values("filed_date")
        filed = df["filed_date"]
        ni, rev, gp, eq, eps = (
            df["net_income"], df["revenue"], df["gross_profit"], df["equity"], df["eps"]
        )
        roe = (ni / eq).where(eq > 0)  # ROE undefined for non-positive book equity
        nm = (ni / rev).where(rev > 0)
        gm = (gp / rev).where(rev > 0)
        eps_d[s] = _to_daily(eps, filed, daily)
        roe_d[s] = _to_daily(roe, filed, daily)
        nm_d[s] = _to_daily(nm, filed, daily)
        gm_d[s] = _to_daily(gm, filed, daily)

    def panel(d: dict[str, pd.Series]) -> pd.DataFrame:
        p = pd.DataFrame(d)
        p.index = close_px.index
        return p.reindex(columns=close_px.columns)

    eps_panel = panel(eps_d)
    return {
        "earnings_yield": eps_panel / close_px,  # value: EPS / current price
        "roe": panel(roe_d),
        "net_margin": panel(nm_d),
        "gross_margin": panel(gm_d),
    }
