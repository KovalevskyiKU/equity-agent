from datetime import date

import pandas as pd

from equity_agent.backtest.panels import load_price_panels
from equity_agent.data.base import OHLCV_COLUMNS, MarketDataProvider
from equity_agent.data.ingest import ingest_daily_bars


class _DivProvider(MarketDataProvider):
    """Two bars where adj_close < close on day 1 (a past dividend), == on day 2."""

    name = "fake"

    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        idx = [date(2024, 1, 2), date(2024, 1, 3)]
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "adj_close": [98.0, 100.0],
                "volume": [1.0, 1.0],
            },
            index=idx,
        )
        df.index.name = "ts"
        return df[OHLCV_COLUMNS]


def test_panels_raw_uses_close(temp_db: None) -> None:
    ingest_daily_bars(["X"], date(2024, 1, 1), date(2024, 1, 31), _DivProvider())
    open_px, close_px = load_price_panels(["X"])
    assert close_px["X"].iloc[0] == 100.0
    assert open_px["X"].iloc[0] == 100.0


def test_panels_total_return_adjusts_open_and_close(temp_db: None) -> None:
    ingest_daily_bars(["X"], date(2024, 1, 1), date(2024, 1, 31), _DivProvider())
    open_px, close_px = load_price_panels(["X"], total_return=True)
    # Both open and close scaled by adj_close/close (= 0.98 on day 1, 1.0 on day 2).
    assert abs(close_px["X"].iloc[0] - 98.0) < 1e-9
    assert abs(open_px["X"].iloc[0] - 98.0) < 1e-9
    assert abs(close_px["X"].iloc[1] - 100.0) < 1e-9
    # Total return day1->day2 is higher than the flat raw return (dividend included).
    assert close_px["X"].iloc[1] / close_px["X"].iloc[0] - 1 > 0
