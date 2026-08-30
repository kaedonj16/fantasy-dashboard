"""Scrub leaked JSON field names from AI prose.

Kept separate from ``client.py`` so unit tests do not need the OpenAI SDK.
"""
from __future__ import annotations

import re

# JSON keys the model sometimes copies into user-facing prose (e.g. "playoff_pct
# sits at 78.5%"). Map to natural language. Order matters for multi-word phrases.
_AI_FIELD_NAME_REPLACEMENTS = (
    (r"\bplayoff_pct\s+sits\s+at\b", "playoff odds sit at"),
    (r"\bplayoff_pct\s+sit\s+at\b", "playoff odds sit at"),
    (r"\bplayoff_pct\b", "playoff odds"),
    (r"\bplayoff_status\b", "playoff standing"),
    (r"\bseason_phase\b", "season phase"),
    (r"\bdraft_grade\b", "draft grade"),
    (r"\bweakest_positions\b", "weakest positions"),
    (r"\bposition_strength\b", "position strength"),
    (r"\btop_assets\b", "top assets"),
    (r"\bwins?\s*_?\s*window\b", "win window"),
    (r"\bwin_window\b", "win window"),
)


def scrub_ai_prose_field_names(text: str) -> str:
    """Replace leaked JSON field names in model prose with human wording."""
    if text is None:
        return ""
    out = str(text)
    for pattern, repl in _AI_FIELD_NAME_REPLACEMENTS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def scrub_ai_result_strings(data: dict) -> dict:
    """Scrub field-name leaks from every string value in a structured AI result."""
    if not isinstance(data, dict):
        return data
    return {
        k: scrub_ai_prose_field_names(v) if isinstance(v, str) else v
        for k, v in data.items()
    }
