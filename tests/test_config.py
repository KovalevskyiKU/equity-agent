from equity_agent.config import load_config


def test_default_config_loads() -> None:
    cfg = load_config()
    assert cfg.universe, "universe must not be empty"
    assert cfg.timeframe == "1d"


def test_all_data_symbols_includes_benchmark_and_regime_without_dupes() -> None:
    cfg = load_config()
    syms = cfg.all_data_symbols
    assert cfg.benchmark in syms
    for r in cfg.regime_symbols:
        assert r in syms
    assert len(syms) == len(set(syms)), "all_data_symbols must be de-duplicated"
