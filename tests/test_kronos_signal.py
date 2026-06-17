import numpy as np

from equity_agent.signals.kronos_signal import directional_signal


def test_directional_signal() -> None:
    # Terminal sampled closes vs a last close of 100.
    terminal = np.array([101.0, 102.0, 103.0, 99.0, 100.0])
    sig = directional_signal(terminal, last_close=100.0)

    # 3 of 5 strictly above 100 -> p_up = 0.6 (100.0 is not > 100.0).
    assert abs(sig["k_p_up"] - 0.6) < 1e-9
    # mean of [0.01, 0.02, 0.03, -0.01, 0.0] = 0.01
    assert abs(sig["k_exp_ret"] - 0.01) < 1e-9
    assert sig["k_ret_std"] >= 0.0
