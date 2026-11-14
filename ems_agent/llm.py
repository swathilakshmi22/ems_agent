import json
import os
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

from .config import LLM_FALLBACK_MODEL, LLM_MODEL, OPENROUTER_BASE_URL


@lru_cache(maxsize=1)
def _client_for(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=8.0,
        max_retries=0,
    )


def initialize_llm_client() -> None:
    """Create the reusable client before starting the voice event loop."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        _client_for(api_key)


async def chat_json(system_prompt: str, user_text: str) -> dict[str, Any]:
    """Call OpenRouter safely; return an empty object if every attempt fails."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {}

    client = _client_for(api_key)

    for model, attempts in ((LLM_MODEL, 2), (LLM_FALLBACK_MODEL, 1)):
        for _ in range(attempts):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    stream=False,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                )
                content = response.choices[0].message.content or "{}"
                result = json.loads(content)
                if isinstance(result, dict):
                    return result
            except Exception:
                continue

    return {}