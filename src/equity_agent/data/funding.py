"""Perp funding rates from Binance (free, no key) — for the funding-carry avenue.

A delta-neutral position (short perp + long spot) collects the funding rate when it
is positive (longs pay shorts), which historically it usually is — a structural
*carry* yield, not directional alpha. This fetches the 8-hourly funding history so we
can size that carry honestly (net of a cost assumption).
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger("equity_agent")

_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FUNDINGS_PER_DAY = 3  # every 8h


def fetch_funding(
    symbol: str = "BTCUSDT", start: str = "2019-09-01", timeout: float = 20.0
) -> pd.DataFrame:
    """Full 8-hourly funding-rate history for a Binance USDT-perp, indexed by time.

    Paginates forward from ``start``. Returns columns: ``funding_rate`` (per 8h),
    ``mark_price``. Empty frame on failure.
    """
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    rows: list[dict] = []
    for _ in range(50):  # hard page cap (50k points » full history)
        params = {"symbol": symbol, "startTime": str(start_ms), "limit": "1000"}
        try:
            r = requests.get(_URL, params=params, timeout=timeout)
        except requests.RequestException as e:
            logger.warning("funding fetch failed: %s", e)
            break
        if r.status_code != 200:
            logger.warning("funding status %s", r.status_code)
            break
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start_ms = int(batch[-1]["fundingTime"]) + 1
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df.dropna(subset=["funding_rate"]).drop_duplicates("time").set_index("time")
    return df[["funding_rate", "mark_price"]].sort_index()


def carry_summary(funding: pd.DataFrame, cost_bps_per_year: float = 200.0) -> dict[str, float]:
    """Annualized delta-neutral carry from a funding series (gross and net of costs).

    Gross carry ≈ mean 8h funding × 3 × 365. ``cost_bps_per_year`` is a rough all-in
    drag (rebalancing the two legs + basis slippage) subtracted for the net figure.
    """
    if funding.empty:
        return {"ann_carry_gross": float("nan"), "ann_carry_net": float("nan"),
                "pct_positive": float("nan"), "n": 0}
    fr = funding["funding_rate"]
    gross = float(fr.mean() * FUNDINGS_PER_DAY * 365)
    return {
        "ann_carry_gross": gross,
        "ann_carry_net": gross - cost_bps_per_year / 1e4,
        "pct_positive": float((fr > 0).mean()),
        "n": int(len(fr)),
    }
