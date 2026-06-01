from __future__ import annotations

import os
import time
import logging

from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

logger = logging.getLogger(__name__)

_AI_CLIENT = None

# Seconds to wait before each retry attempt (exponential backoff)
_RETRY_DELAYS = [1, 2, 4]
_REQUEST_TIMEOUT = 30  # seconds


def get_ai_client() -> OpenAI:
    global _AI_CLIENT
    if _AI_CLIENT is None:
        _AI_CLIENT = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=_REQUEST_TIMEOUT,
        )
    return _AI_CLIENT


def ai_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("AI_ENABLED", "true").lower() == "true"


class AIRateLimitError(Exception):
    """Raised when the OpenAI API returns a rate-limit response."""


class AIUnavailableError(Exception):
    """Raised when the OpenAI API is unreachable or returns a server error."""


def clean_ai_text(text: str) -> str:
    """Replace em dashes with hyphens so they don't appear in rendered output."""
    return text.replace("—", " - ")


def generate_text(system_prompt: str, user_prompt: str, model: str = "gpt-5-mini") -> str:
    client = get_ai_client()
    last_exc: Exception | None = None

    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return clean_ai_text(resp.output_text.strip())

        except RateLimitError as exc:
            last_exc = exc
            logger.warning("[ai-client] rate limited (attempt %d/%d)", attempt + 1, len(_RETRY_DELAYS) + 1)
            # Rate limits warrant waiting longer; use the retry-after header if present
            retry_after = getattr(exc, "response", None)
            if retry_after:
                try:
                    wait = float(retry_after.headers.get("retry-after", delay * 2))
                    time.sleep(min(wait, 16))
                except Exception:
                    pass

        except APIConnectionError as exc:
            last_exc = exc
            logger.warning("[ai-client] connection error (attempt %d/%d): %s", attempt + 1, len(_RETRY_DELAYS) + 1, exc)

        except APIStatusError as exc:
            # 5xx errors are transient; 4xx (except 429) are not worth retrying
            if exc.status_code and exc.status_code < 500:
                logger.error("[ai-client] non-retryable API error %s: %s", exc.status_code, exc)
                raise AIUnavailableError(str(exc)) from exc
            last_exc = exc
            logger.warning("[ai-client] server error %s (attempt %d/%d)", exc.status_code, attempt + 1, len(_RETRY_DELAYS) + 1)

    if isinstance(last_exc, RateLimitError):
        raise AIRateLimitError("OpenAI rate limit reached after retries") from last_exc
    raise AIUnavailableError("OpenAI API unavailable after retries") from last_exc
