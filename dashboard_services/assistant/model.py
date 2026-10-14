"""Single model-call helper for the assistant.

Wraps the pattern used by generate_trade_ai_result in
dashboard_services.ai.prompts: responses.create with a json_schema format,
clean_ai_text on the raw text, then json.loads. Rate-limit and API errors are
mapped onto the shared AI error types from dashboard_services.ai.client.
"""
from __future__ import annotations

import json
import logging
import time

from dashboard_services.ai.client import (
    AIRateLimitError,
    AIUnavailableError,
    clean_ai_text,
    get_ai_client,
)
from dashboard_services.ai.prompts import OPENAI_MODEL

logger = logging.getLogger(__name__)

# Rough gpt-5.4-mini list prices (USD per 1M tokens); only used for the log's
# est_cost_usd estimate, never for billing.
_INPUT_PER_M = 0.25
_OUTPUT_PER_M = 2.0
_CACHED_PER_M = 0.025

_last_usage: dict = {}


def last_usage() -> dict:
    return dict(_last_usage)


def _extract_usage(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "cached_tokens": None, "est_cost_usd": None}
    def _get(obj, *names):
        for n in names:
            v = getattr(obj, n, None) if not isinstance(obj, dict) else obj.get(n)
            if v is not None:
                return v
        return None
    inp = _get(usage, "input_tokens", "prompt_tokens")
    out = _get(usage, "output_tokens", "completion_tokens")
    details = _get(usage, "input_tokens_details", "prompt_tokens_details")
    cached = _get(details, "cached_tokens") if details is not None else None
    est = None
    if inp is not None or out is not None:
        cached_n = cached or 0
        est = round(
            (((inp or 0) - cached_n) * _INPUT_PER_M + cached_n * _CACHED_PER_M + (out or 0) * _OUTPUT_PER_M) / 1_000_000,
            6,
        )
    return {
        "input_tokens": int(inp) if inp is not None else None,
        "output_tokens": int(out) if out is not None else None,
        "cached_tokens": int(cached) if cached is not None else None,
        "est_cost_usd": est,
    }


def call_model_json(system: str, user: str, schema: dict, name: str, *, timeout_note: str = "") -> dict:
    """Call the model once and return the parsed JSON object.

    Raises AIRateLimitError / AIUnavailableError on API failure.
    """
    global _last_usage
    started = time.time()
    try:
        from openai import APIConnectionError, APIStatusError, RateLimitError
    except Exception:  # pragma: no cover - openai always present with client
        RateLimitError = ()  # type: ignore
    try:
        client = get_ai_client()
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text={"format": {"type": "json_schema", "name": name, "schema": schema}},
        )
    except Exception as exc:
        if RateLimitError and isinstance(exc, RateLimitError):
            raise AIRateLimitError(str(exc)) from exc
        # RateLimitError subclasses APIStatusError in the SDK, so check the
        # status code as a belt-and-braces mapping for wrapped errors.
        status = getattr(exc, "status_code", None)
        if status == 429:
            raise AIRateLimitError(str(exc)) from exc
        raise AIUnavailableError(str(exc)) from exc
    _last_usage = _extract_usage(resp)
    _last_usage["latency_ms"] = int((time.time() - started) * 1000)
    _last_usage["model"] = OPENAI_MODEL
    raw = clean_ai_text((getattr(resp, "output_text", "") or "").strip())
    if not raw:
        raise AIUnavailableError(f"empty model response for {name}")
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise AIUnavailableError(f"model response for {name} was not JSON") from exc
    if not isinstance(data, dict):
        raise AIUnavailableError(f"model response for {name} was not an object")
    return data
