"""Single entry point for LLM calls — Groq today, swappable.

Every LLM call goes through this function, so switching provider touches this
file only. Groq free tier is rate-limited (~30 req/min, ~6k tokens/min, ~1k/day);
we let the SDK honour Retry-After via max_retries so transient 429/503 pace
themselves instead of failing. Structured output = JSON mode + pydantic
validation.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from .config import get_settings

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def generate_structured[T: BaseModel](
    prompt: str,
    schema: type[T],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> T:
    """Call the LLM and return a validated instance of ``schema`` (a pydantic model)."""
    from groq import Groq

    # max_retries lets the SDK wait out 429/503 (honouring Retry-After) — do NOT add a
    # manual retry loop on top, it just multiplies requests and burns the daily quota.
    client = Groq(api_key=get_settings().groq_api_key, max_retries=8)
    hint = json.dumps(schema.model_json_schema())
    full_prompt = f"{prompt}\n\nReturn ONLY a JSON object matching this schema:\n{hint}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned empty content")
    return schema.model_validate_json(content)
