from equity_agent.decision.agent import _normalize


def test_normalize_clamps_weight_to_max() -> None:
    assert _normalize("buy", 0.9, 0.34) == ("BUY", 0.34)


def test_normalize_sell_forces_flat() -> None:
    assert _normalize("SELL", 0.5, 0.34) == ("SELL", 0.0)


def test_normalize_unknown_action_becomes_hold() -> None:
    assert _normalize("weird", 0.1, 0.34) == ("HOLD", 0.1)


def test_normalize_floors_negative_weight() -> None:
    assert _normalize("HOLD", -0.2, 0.34) == ("HOLD", 0.0)
