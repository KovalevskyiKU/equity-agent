"""Provider interfaces. Vendor-agnostic by design."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

# Canonical OHLCV schema every market-data provider must return.
# Index = tz-naive calendar date (the bar's trading day). adj_close may be NaN.
OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class MarketDataProvider(ABC):
    """Source of historical daily OHLCV bars."""

    name: str = "base"

    @abstractmethod
    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return a DataFrame indexed by date with :data:`OHLCV_COLUMNS`.

        Must be point-in-time correct: no rows dated after ``end``. Empty
        DataFrame (with the right columns) if the symbol has no data.
        """
        raise NotImplementedError

    def get_daily_bars_batch(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        """Fetch many symbols at once. Returns {symbol: DataFrame} (empty if no data).

        Default implementation loops :meth:`get_daily_bars`; providers that can
        bulk-fetch (e.g. yfinance) override this for speed and fewer round-trips.
        """
        return {s: self.get_daily_bars(s, start, end) for s in symbols}
