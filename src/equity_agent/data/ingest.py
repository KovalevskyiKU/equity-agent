"""Ingest daily bars into the store. Idempotent upsert keyed on (symbol, date)."""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import select

from ..storage.db import session_scope
from ..storage.models import DailyBar
from .base import MarketDataProvider

logger = logging.getLogger("equity_agent")


def ingest_daily_bars(
    symbols: list[str],
    start: date,
    end: date,
    provider: MarketDataProvider,
) -> dict[str, int]:
    """Fetch and store daily bars for each symbol. Returns {symbol: rows_inserted}.

    Re-running is safe: bars already present (same symbol + date) are skipped,
    never duplicated or silently mutated.
    """
    inserted: dict[str, int] = {}
    bars_by_symbol = provider.get_daily_bars_batch(symbols, start, end)

    for symbol in symbols:
        df = bars_by_symbol.get(symbol)
        if df is None or df.empty:
            logger.warning("[%s] no bars returned by %s", symbol, provider.name)
            inserted[symbol] = 0
            continue

        with session_scope() as session:
            existing = set(
                session.scalars(
                    select(DailyBar.ts).where(DailyBar.symbol == symbol)
                ).all()
            )
            rows = []
            for ts, row in df.iterrows():
                if ts in existing:
                    continue
                adj = row.get("adj_close")
                rows.append(
                    {
                        "symbol": symbol,
                        "ts": ts,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "adj_close": None if adj is None or adj != adj else float(adj),
                        "volume": float(row["volume"]),
                        "source": provider.name,
                    }
                )
            if rows:
                session.bulk_insert_mappings(DailyBar, rows)

        inserted[symbol] = len(rows)
        logger.info("[%s] ingested %d new bars (%s)", symbol, len(rows), provider.name)

    return inserted
