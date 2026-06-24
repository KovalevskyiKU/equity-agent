import pandas as pd

from equity_agent.monitoring import monitor_summary


def test_empty_curve() -> None:
    assert monitor_summary(pd.DataFrame(columns=["ts", "equity"])) == {"snapshots": 0}


def test_single_snapshot_has_no_returns() -> None:
    df = pd.DataFrame({"ts": [pd.Timestamp("2026-01-02")], "equity": [100_000.0]})
    s = monitor_summary(df)
    assert s["snapshots"] == 1
    assert s["equity"] == 100_000.0
    assert "total_return" not in s  # need >= 2 snapshots


def test_metrics_and_spy_tracking() -> None:
    ts = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    df = pd.DataFrame({"ts": ts, "equity": [100_000.0, 101_000.0, 100_500.0]})
    spy = pd.Series(
        [400.0, 404.0, 402.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]).date,
    )
    s = monitor_summary(df, spy)
    assert s["snapshots"] == 3
    assert abs(s["total_return"] - (100_500 / 100_000 - 1)) < 1e-9
    assert abs(s["last_pnl"] - (-500.0)) < 1e-6
    assert "sharpe" in s and "max_drawdown" in s
    assert abs(s["spy_return"] - (402 / 400 - 1)) < 1e-9
    assert abs(s["excess_vs_spy"] - (s["total_return"] - s["spy_return"])) < 1e-9
