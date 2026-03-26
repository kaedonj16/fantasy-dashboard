from __future__ import annotations

import os
from openai import OpenAI

_AI_CLIENT = None


def get_ai_client() -> OpenAI:
    global _AI_CLIENT
    if _AI_CLIENT is None:
        _AI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _AI_CLIENT


def ai_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("AI_ENABLED", "true").lower() == "true"


def generate_text(system_prompt: str, user_prompt: str, model: str = "gpt-5-mini") -> str:
    client = get_ai_client()

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.output_text.strip()
