"""Single entry point for LLM calls — provider-switchable (Groq now, Ollama later).

`LLM_PROVIDER` selects the backend: "groq" (free cloud, used now) or "ollama"
(local, for when there's GPU hardware). Switching is a one-line env change; no
caller changes. Structured output = JSON mode + pydantic validation.
"""

from __future__ import annotations

import json
import typing

from pydantic import BaseModel

from .config import get_settings

DEFAULT_MODEL = "llama-3.3-70b-versatile"  # Groq default; Ollama uses settings.ollama_model


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

    # max_retries lets the SDK honour Retry-After on 429/503 — do NOT add a manual loop.
    client = Groq(api_key=get_settings().groq_api_key, max_retries=8)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned empty content")
    return schema.model_validate_json(content)


def _ollama_structured[T: BaseModel](
    prompt: str, schema: type[T], model: str, temperature: float, base_url: str
) -> T:
    import requests

    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return schema.model_validate_json(content)


def generate_structured[T: BaseModel](
    prompt: str,
    schema: type[T],
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> T:
    """Call the active LLM provider and return a validated instance of ``schema``."""
    settings = get_settings()
    hint = json.dumps(_compact_shape(schema))
    full_prompt = f"{prompt}\n\nReturn ONLY a JSON object with exactly this shape:\n{hint}"

    provider = settings.llm_provider.lower()
    if provider == "groq":
        return _groq_structured(full_prompt, schema, model or DEFAULT_MODEL, temperature)
    if provider == "ollama":
        return _ollama_structured(
            full_prompt, schema, settings.ollama_model, temperature, settings.ollama_base_url
        )
    raise ValueError(f"Unknown LLM_PROVIDER {settings.llm_provider!r} (use 'groq' or 'ollama')")
