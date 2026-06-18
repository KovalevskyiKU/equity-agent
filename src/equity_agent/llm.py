"""Single entry point for LLM calls — Groq today, swappable.

Every LLM call goes through this function, so switching provider touches this
file only. Groq's free tier (llama-3.3-70b ~1000 req/day) handles our backtest
and live volume. Structured output is enforced via JSON mode + pydantic
validation, with retry/backoff on transient 429/503 (Groq is occasionally
"high demand") so a backtest doesn't silently lose decisions.
"""

from __future__ import annotations

import json
import time

from pydantic import BaseModel

from .config import get_settings

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def generate_structured[T: BaseModel](
    prompt: str,
    schema: type[T],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_attempts: int = 4,
) -> T:
    """Call the LLM and return a validated instance of ``schema`` (a pydantic model)."""
    from groq import Groq

    client = Groq(api_key=get_settings().groq_api_key, max_retries=2)
    schema_hint = json.dumps(schema.model_json_schema())
    full_prompt = (
        f"{prompt}\n\nReturn ONLY a JSON object conforming to this JSON Schema "
        f"(no markdown, no commentary):\n{schema_hint}"
    )

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
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
        except Exception as e:  # noqa: BLE001 - retry transient errors (429/503/parse)
            last_err = e
            time.sleep(min(2**attempt, 12))

    assert last_err is not None
    raise last_err
