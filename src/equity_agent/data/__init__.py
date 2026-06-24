"""Data layer: market data, fundamentals, news, macro.

All providers implement a common interface so the rest of the system never
depends on a specific vendor (yfinance today, Finnhub/IBKR later).
"""

from .base import OHLCV_COLUMNS, MarketDataProvider
from .ingest import ingest_daily_bars
from .yfinance_provider import YFinanceProvider

__all__ = [
    "MarketDataProvider",
    "OHLCV_COLUMNS",
    "YFinanceProvider",
    "ingest_daily_bars",
]
