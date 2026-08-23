import numpy as np
import pandas as pd

from equity_agent.data.fundamentals import parse_filings
from equity_agent.research.fundamental_factors import (
    _to_daily,
    sector_neutralize,
    value_quality_composite,
)

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


def test_sector_neutralize_zeros_within_sector_mean() -> None:
    dates = pd.to_datetime(["2020-01-31", "2020-02-28"])
    cols = ["A", "B", "C", "D"]
    factor = pd.DataFrame(
        [[1.0, 3.0, 10.0, 20.0], [2.0, 4.0, 30.0, 50.0]], index=dates, columns=cols
    )
    sectors = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}
    z = sector_neutralize(factor, sectors)
    # Each sector's z-scores have mean ~0 per date; the high-value sector no longer
    # dominates (that's the point of neutralizing).
    for d in dates:
        assert abs(z.loc[d, ["A", "B"]].mean()) < 1e-9
        assert abs(z.loc[d, ["C", "D"]].mean()) < 1e-9
    # Within Tech, B > A on both dates -> positive z for B, negative for A.
    assert (z["B"] > 0).all() and (z["A"] < 0).all()


def test_value_quality_composite_averages_neutral_zscores() -> None:
    dates = pd.to_datetime(["2020-01-31"])
    cols = ["A", "B", "C", "D"]
    sectors = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}
    ey = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=dates, columns=cols)
    roe = pd.DataFrame([[4.0, 3.0, 2.0, 1.0]], index=dates, columns=cols)
    comp = value_quality_composite(
        {"earnings_yield": ey, "roe": roe}, sectors, keys=("earnings_yield", "roe")
    )
    assert comp.shape == (1, 4)
    # ey and roe are opposite rankings -> their neutral z-scores cancel to ~0.
    assert np.allclose(comp.loc[dates[0]].to_numpy(), 0.0, atol=1e-9)


def test_quarterly_net_issuance_sign_and_lag(tmp_path, monkeypatch) -> None:
    """Buybacks (shrinking share count) score HIGH; issuance scores low; YoY lag=4."""
    import equity_agent.data.fundamentals as fund
    from equity_agent.research.fundamental_factors import quarterly_net_issuance

    monkeypatch.setattr(fund, "_quarterly_dir", lambda: tmp_path)
    # 8 quarters: BUYER shrinks its share count, ISSUER grows it.
    ends = pd.date_range("2021-03-31", periods=8, freq="QE")
    filed = ends + pd.Timedelta(days=30)
    for sym, shares in (("BUYER", np.linspace(100, 86, 8)), ("ISSUER", np.linspace(100, 114, 8))):
        pd.DataFrame(
            {"filed_date": filed, "end_date": ends, "year": ends.year,
             "quarter": ends.quarter, "shares": shares}
        ).to_parquet(tmp_path / f"{sym}.parquet")

    close = pd.DataFrame(
        1.0,
        index=pd.date_range("2021-01-01", periods=900, freq="D"),
        columns=["BUYER", "ISSUER"],
    )
    f = quarterly_net_issuance(["BUYER", "ISSUER"], close)
    last = f.dropna(how="all").iloc[-1]
    assert last["BUYER"] > 0  # bought back -> positive score
    assert last["ISSUER"] < 0  # issued -> negative score
    assert last["BUYER"] > last["ISSUER"]


def test_quarterly_net_issuance_missing_symbol_is_nan(tmp_path, monkeypatch) -> None:
    import equity_agent.data.fundamentals as fund
    from equity_agent.research.fundamental_factors import quarterly_net_issuance

    monkeypatch.setattr(fund, "_quarterly_dir", lambda: tmp_path)
    close = pd.DataFrame(
        1.0, index=pd.date_range("2021-01-01", periods=10, freq="D"), columns=["NONE"]
    )
    assert quarterly_net_issuance(["NONE"], close)["NONE"].isna().all()
