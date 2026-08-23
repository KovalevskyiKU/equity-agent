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

import numpy as np
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
    gpa_d: dict[str, pd.Series] = {}
    ag_d: dict[str, pd.Series] = {}
    iss_d: dict[str, pd.Series] = {}
    acc_d: dict[str, pd.Series] = {}

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

        # Second research pass. Sign convention (higher = expected higher return):
        #   gp_to_assets  Novy-Marx profitability — the formulation that works
        #                 (gross profit over ASSETS, not over revenue).
        #   asset_growth  conservative (low-investment) firms outperform -> negated.
        #   net_issuance  share issuers underperform, buybacks outperform -> negated.
        #   accruals      high accruals (earnings not backed by cash) underperform
        #                 -> negated.
        assets = df["assets"] if "assets" in df else pd.Series(float("nan"), index=df.index)
        shares = df["shares"] if "shares" in df else pd.Series(float("nan"), index=df.index)
        cfo = (
            df["cash_flow_ops"]
            if "cash_flow_ops" in df
            else pd.Series(float("nan"), index=df.index)
        )
        pos_assets = assets > 0
        gpa_d[s] = _to_daily((gp / assets).where(pos_assets), filed, daily)
        ag_d[s] = _to_daily(-(assets.pct_change()).where(pos_assets), filed, daily)
        iss_d[s] = _to_daily(-(shares.pct_change()).where(shares > 0), filed, daily)
        acc_d[s] = _to_daily(-((ni - cfo) / assets).where(pos_assets), filed, daily)

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
        "gp_to_assets": panel(gpa_d),
        "asset_growth": panel(ag_d),
        "net_issuance": panel(iss_d),
        "accruals": panel(acc_d),
    }


def sector_neutralize(factor: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """Cross-sectional z-score within each sector, per date.

    Strips sector tilts so a factor reflects *within-sector* ranking rather than a
    bet on whichever sector happens to score high (e.g. gross_margin -> tech/health).
    """
    sec = pd.Series(sectors).reindex(factor.columns).fillna("Unknown")
    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    for s in sec.unique():
        cols = sec.index[sec == s]
        sub = factor[cols]
        mu = sub.mean(axis=1)
        sd = sub.std(axis=1).replace(0.0, np.nan)
        out[cols] = sub.sub(mu, axis=0).div(sd, axis=0)
    return out


def value_quality_composite(
    panels: dict[str, pd.DataFrame],
    sectors: dict[str, str],
    keys: tuple[str, ...] = ("earnings_yield", "roe", "gross_margin"),
) -> pd.DataFrame:
    """Equal-weight mean of sector-neutral z-scores of ``keys`` (NaN-aware).

    The best legitimate construction found: combine the factors with a real
    cross-sectional IC after removing sector tilts.
    """
    base = panels[keys[0]]
    arr = np.stack(
        [
            sector_neutralize(panels[k], sectors)
            .reindex(index=base.index, columns=base.columns)
            .to_numpy()
            for k in keys
        ]
    )
    counts = (~np.isnan(arr)).sum(axis=0)
    totals = np.nansum(arr, axis=0)
    comp = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return pd.DataFrame(comp, index=base.index, columns=base.columns)
