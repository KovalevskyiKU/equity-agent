"""yfinance-backed daily bars. No API key required — good default for Phase 0/1."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from .base import OHLCV_COLUMNS, MarketDataProvider

logger = logging.getLogger("equity_agent")


def _clean(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize a raw yfinance frame (single ticker) to the canonical OHLCV schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    # Single-ticker downloads can come back with a MultiIndex column header.
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = [c[0] for c in raw.columns]

    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    df = pd.DataFrame(index=raw.index)
    df["open"] = raw.get("open")
    df["high"] = raw.get("high")
    df["low"] = raw.get("low")
    df["close"] = raw.get("close")
    df["adj_close"] = raw.get("adj_close")
    df["volume"] = raw.get("volume")

    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df.index = pd.to_datetime(df.index).date
    df.index.name = "ts"
    return df[OHLCV_COLUMNS]


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        # yfinance `end` is exclusive — add a day so the requested end date is included.
        raw = yf.download(
            tickers=symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return _clean(raw)

    def get_daily_bars_batch(
        self, symbols: list[str], start: date, end: date, chunk_size: int = 100
    ) -> dict[str, pd.DataFrame]:
        """Bulk-download many tickers via one ``yf.download`` per chunk (much faster).

        Chunking keeps each request a sane size and limits the blast radius of a
        rate-limit hiccup. Missing/delisted tickers come back empty rather than
        failing the whole batch.
        """
        import yfinance as yf

        results: dict[str, pd.DataFrame] = {}
        start_s = start.isoformat()
        end_s = (end + timedelta(days=1)).isoformat()

        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            raw = yf.download(
                tickers=chunk,
                start=start_s,
                end=end_s,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            multi = raw is not None and not raw.empty and isinstance(raw.columns, pd.MultiIndex)
            for s in chunk:
                if raw is None or raw.empty:
                    results[s] = pd.DataFrame(columns=OHLCV_COLUMNS)
                elif multi:
                    sub = raw[s] if s in raw.columns.get_level_values(0) else None
                    results[s] = _clean(sub)
                else:
                    # Single-ticker chunk: columns are already field-level.
                    results[s] = _clean(raw)
            logger.info(
                "yfinance batch %d-%d/%d fetched", i + 1, i + len(chunk), len(symbols)
            )
        return results
