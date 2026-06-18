"""Single entry point for LLM calls — Gemini today, swappable later.

Keeping every LLM call behind one function means switching provider (e.g. to
Claude when we buy the API) touches this file only, not the sentiment scorer or
the decision agent.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from .config import get_settings

DEFAULT_MODEL = "gemini-2.5-flash"


def generate_structured[T: BaseModel](
    prompt: str,
    schema: type[T],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> T:
    """Call the LLM and return a validated instance of ``schema`` (a pydantic model)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_settings().google_api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        ),
    )
    parsed = resp.parsed
    if isinstance(parsed, schema):
        return parsed
    if resp.text is None:
        raise RuntimeError("LLM returned neither a parsed object nor text")
    return schema(**json.loads(resp.text))
