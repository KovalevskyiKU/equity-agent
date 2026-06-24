import numpy as np
import pandas as pd

from equity_agent.data.fundamentals import parse_filings
from equity_agent.research.fundamental_factors import _to_daily

_REV_FALLBACK = "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax"
_EQUITY_NCI = "us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"


def test_parse_filings_extracts_with_fallbacks() -> None:
    payload = [
        {
            "filedDate": "2021-02-15",
            "endDate": "2020-12-31",
            "report": {
                "ic": [
                    {"concept": "us-gaap_Revenues", "value": 1000},
                    {"concept": "us-gaap_NetIncomeLoss", "value": 100},
                    {"concept": "us-gaap_GrossProfit", "value": 400},
                    {"concept": "us-gaap_EarningsPerShareDiluted", "value": 2.5},
                ],
                "bs": [{"concept": "us-gaap_StockholdersEquity", "value": 500}],
            },
        },
        {
            "filedDate": "2020-02-15",
            "endDate": "2019-12-31",
            "report": {
                "ic": [
                    # revenue, net income and eps only available via fallback tags
                    {"concept": _REV_FALLBACK, "value": 900},
                    {"concept": "us-gaap_ProfitLoss", "value": 90},
                    {"concept": "us-gaap_EarningsPerShareBasic", "value": 2.0},
                ],
                "bs": [{"concept": _EQUITY_NCI, "value": 480}],
            },
        },
    ]
    df = parse_filings(payload).set_index("end_date")
    df.index = pd.to_datetime(df.index)

    y20 = df.loc[pd.Timestamp("2020-12-31")]
    assert y20["revenue"] == 1000 and y20["net_income"] == 100
    assert y20["gross_profit"] == 400 and y20["eps"] == 2.5 and y20["equity"] == 500

    y19 = df.loc[pd.Timestamp("2019-12-31")]
    assert y19["revenue"] == 900  # fallback revenue tag
    assert y19["net_income"] == 90  # ProfitLoss fallback
    assert y19["eps"] == 2.0  # basic-EPS fallback
    assert y19["equity"] == 480  # equity-incl-NCI fallback
    assert np.isnan(y19["gross_profit"])  # not reported -> NaN


def test_to_daily_is_point_in_time() -> None:
    filed = pd.Series(pd.to_datetime(["2020-02-15", "2021-02-15"]))
    vals = pd.Series([10.0, 20.0])
    daily = pd.to_datetime(
        ["2020-01-01", "2020-02-14", "2020-02-15", "2020-06-01", "2021-02-15", "2021-03-01"]
    )
    out = _to_daily(vals, filed, daily)
    # Before the first filing date -> unknown (NaN): no look-ahead.
    assert np.isnan(out.loc["2020-01-01"]) and np.isnan(out.loc["2020-02-14"])
    # From the filing date forward -> the filed value, held until the next filing.
    assert out.loc["2020-02-15"] == 10.0 and out.loc["2020-06-01"] == 10.0
    assert out.loc["2021-02-15"] == 20.0 and out.loc["2021-03-01"] == 20.0
