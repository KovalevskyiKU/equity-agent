from datetime import date

import pandas as pd

from equity_agent.data.base import OHLCV_COLUMNS, MarketDataProvider
from equity_agent.data.ingest import ingest_daily_bars


class FakeProvider(MarketDataProvider):
    name = "fake"

    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        idx = [date(2024, 1, 2), date(2024, 1, 3)]
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [2.0, 3.0],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "adj_close": [1.5, 2.5],
                "volume": [100.0, 200.0],
            },
            index=idx,
        )
        df.index.name = "ts"
        return df[OHLCV_COLUMNS]


def test_ingest_is_idempotent(temp_db: None) -> None:
    provider = FakeProvider()
    start, end = date(2024, 1, 1), date(2024, 1, 31)

    first = ingest_daily_bars(["TEST"], start, end, provider)
    assert first["TEST"] == 2

    # Re-running must not duplicate rows.
    second = ingest_daily_bars(["TEST"], start, end, provider)
    assert second["TEST"] == 0
