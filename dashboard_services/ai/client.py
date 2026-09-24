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
    # Strip em dashes (a common AI tell) from sentence prose, replacing them
    # with a comma so model output reads in a plain, human voice. Regular
    # hyphens are NEVER touched: hyphenated words (follow-up, all-play) and
    # records (2-0) must come through intact, as must negative JSON numbers.
    # En dashes only count as sentence punctuation when padded with whitespace
    # ("a – b"); ranges like "161.46–76.26" or "2–0" are left alone.
    # Runs on the raw JSON string, which only affects the string values
    # (structure uses no em/en dashes).
    # Only clean if the text looks like valid JSON (starts with { or [)
    if not text or not text.lstrip().startswith(('{', '[')):
        return text
    text = re.sub(r'\s*[—―]\s*', ', ', text)      # em dash / horizontal bar
    text = re.sub(r'\s+–\s*|\s*–\s+', ', ', text)  # en dash as punctuation only
    return text

