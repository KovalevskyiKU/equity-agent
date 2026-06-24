import pytest
from pydantic import BaseModel

from equity_agent import config
from equity_agent.llm import _compact_shape, generate_structured


class _Shape(BaseModel):
    a: int
    b: str
    c: list[float]


def test_compact_shape() -> None:
    shape = _compact_shape(_Shape)
    assert shape == {"a": "number", "b": "string", "c": ["number"]}


def test_unknown_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    config.get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            generate_structured("hi", _Shape)
    finally:
        config.get_settings.cache_clear()
