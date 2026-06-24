"""FRED macro series fetcher + ingest (free API key).

FRED series are single-value daily time series; we store them as pseudo-symbols
in ``daily_bars`` (open=high=low=close=value, source="fred") so the rest of the
pipeline (load_bars, feature store) reuses one code path.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests
from sqlalchemy import select

from ..config import get_settings
from ..storage.db import session_scope
from ..storage.models import DailyBar

logger = logging.getLogger("equity_agent")

_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, start: date, end: date) -> pd.Series:
    """Fetch one FRED series as a date-indexed Series (missing '.' values dropped)."""
    key = get_settings().fred_api_key
    if not key:
        raise RuntimeError("FRED_API_KEY not set in .env")

    resp = requests.get(
        _FRED_URL,
        params={
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        },
        timeout=30,
    )
    resp.raise_for_status()

    data: dict[date, float] = {}
    for obs in resp.json().get("observations", []):
        value = obs.get("value")
        if value in (None, "."):
            continue
        try:
            data[date.fromisoformat(obs["date"])] = float(value)
        except (ValueError, KeyError):
            continue
    return pd.Series(data, name=series_id).sort_index()


def ingest_fred(series_ids: list[str], start: date, end: date) -> dict[str, int]:
    """Fetch and store FRED series as pseudo-symbols in daily_bars. Idempotent."""
    inserted: dict[str, int] = {}
    for sid in series_ids:
        series = fetch_series(sid, start, end)
        if series.empty:
            inserted[sid] = 0
            continue
        with session_scope() as session:
            existing = set(
                session.scalars(select(DailyBar.ts).where(DailyBar.symbol == sid)).all()
            )
            new_rows = 0
            for ts, value in series.items():
                if ts in existing:
                    continue
                val = float(value)
                session.add(
                    DailyBar(
                        symbol=sid, ts=ts, open=val, high=val, low=val, close=val,
                        adj_close=val, volume=0.0, source="fred",
                    )
                )
                new_rows += 1
        inserted[sid] = new_rows
        logger.info("[fred:%s] ingested %d obs", sid, new_rows)
    return inserted
