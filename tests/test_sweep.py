from equity_agent.backtest.sweep import _windows


def test_windows_are_rolling_and_bounded() -> None:
    w = _windows(100, 30, 10)
    assert w[0] == (0, 30)
    assert all(b - a == 30 for a, b in w)
    assert all(0 <= a < b <= 100 for a, b in w)
    # last window must fit within n
    assert w[-1][1] <= 100


def test_windows_empty_when_too_short() -> None:
    assert _windows(20, 30, 10) == []
