"""Single entry point for LLM calls — multi-provider with failover.

`LLM_PROVIDER` is a comma-separated priority list, e.g. "groq,deepseek". Each call
tries providers in order and falls through to the next on any error (e.g. a rate
limit), so several free tiers stack into more total throughput. Providers:
"groq" (free cloud), "deepseek" (5M free tokens for new accounts), "ollama"
(local, for when there's GPU hardware). Structured output = JSON mode + pydantic.
"""

from __future__ import annotations

import json
import logging
import typing

from pydantic import BaseModel

from .config import Settings, get_settings

logger = logging.getLogger("equity_agent")

DEFAULT_MODEL = "llama-3.3-70b-versatile"  # Groq default


def _annotation_shape(ann: object) -> object:
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return _compact_shape(ann)
    origin = typing.get_origin(ann)
    if origin in (list, tuple):
        args = typing.get_args(ann)
        return [_annotation_shape(args[0] if args else str)]
    if ann in (int, float):
        return "number"
    if ann is bool:
        return "boolean"
    return "string"


def _compact_shape(model: type[BaseModel]) -> dict[str, object]:
    """A tiny JSON shape (field -> type) — far cheaper in tokens than the full schema."""
    return {name: _annotation_shape(f.annotation) for name, f in model.model_fields.items()}


def _groq_structured[T: BaseModel](
    prompt: str, schema: type[T], model: str, temperature: float
) -> T:
    from groq import Groq

    client = Groq(api_key=get_settings().groq_api_key, max_retries=8)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("Groq returned empty content")
    return schema.model_validate_json(content)


def _openai_compatible_structured[T: BaseModel](
    prompt: str, schema: type[T], *, base_url: str, api_key: str, model: str, temperature: float
) -> T:
    """DeepSeek / Ollama / any OpenAI-compatible /chat/completions endpoint."""
    import requests

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "stream": False,
        },
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return schema.model_validate_json(content)


def generate_structured[T: BaseModel](
    prompt: str,
    schema: type[T],
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> T:
    """Call the configured LLM providers (in priority order, with failover)."""
    settings = get_settings()
    hint = json.dumps(_compact_shape(schema))
    full_prompt = f"{prompt}\n\nReturn ONLY a JSON object with exactly this shape:\n{hint}"

    providers = [p.strip().lower() for p in settings.llm_provider.split(",") if p.strip()]
    if not providers:
        raise ValueError("LLM_PROVIDER is empty")

    last_err: Exception | None = None
    for provider in providers:
        try:
            return _call_provider(provider, full_prompt, schema, model, temperature, settings)
        except Exception as e:  # noqa: BLE001 - fall through to the next provider on any error
            last_err = e
            logger.warning("LLM provider %s failed: %s", provider, str(e)[:140])
    assert last_err is not None
    raise last_err


def _call_provider[T: BaseModel](
    provider: str,
    prompt: str,
    schema: type[T],
    model: str | None,
    temperature: float,
    settings: Settings,
) -> T:
    if provider == "groq":
        return _groq_structured(prompt, schema, model or DEFAULT_MODEL, temperature)
    if provider == "deepseek":
        return _openai_compatible_structured(
            prompt, schema, base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key or "", model=settings.deepseek_model,
            temperature=temperature,
        )
    if provider == "gemini":
        return _openai_compatible_structured(
            prompt, schema,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key=settings.google_api_key or "", model=settings.gemini_model,
            temperature=temperature,
        )
    if provider == "cerebras":
        return _openai_compatible_structured(
            prompt, schema, base_url=settings.cerebras_base_url,
            api_key=settings.cerebras_api_key or "", model=settings.cerebras_model,
            temperature=temperature,
        )
    if provider == "ollama":
        return _openai_compatible_structured(
            prompt, schema, base_url=f"{settings.ollama_base_url}/v1",
            api_key="", model=settings.ollama_model, temperature=temperature,
        )
    raise ValueError(
        f"Unknown LLM provider {provider!r} (use groq / deepseek / gemini / cerebras / ollama)"
    )
