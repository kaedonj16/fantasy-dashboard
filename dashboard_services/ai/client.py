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
    import re
    return re.sub(r'\s*—\s*', ', ', text)

