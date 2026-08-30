"""Scrub leaked JSON field names and broken status lead-ins from AI prose.

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

# Prompt says "state contend / bubble / out"; models often start the outlook
# with the bare label ("out, with playoff odds not provided here…").
_BARE_STATUS_LEADIN = re.compile(
    r"^\s*(out|bubble|contend)\s*,\s*",
    re.IGNORECASE,
)
_BARE_STATUS_EXPANSION = {
    "out": "This team looks out of playoff contention",
    "bubble": "This team is on the playoff bubble",
    "contend": "This team is built to contend",
}

# Narrating absent odds despite honesty rules.
_MISSING_ODDS_CLAUSES = (
    re.compile(
        r",?\s*with\s+playoff\s+odds\s+not\s+(?:provided|available|included)"
        r"(?:\s+here)?(?:\s+in\s+the\s+(?:json|data|context))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bplayoff\s+odds\s+(?:are\s+)?not\s+(?:provided|available|included)"
        r"(?:\s+here)?(?:\s+in\s+the\s+(?:json|data|context))?[^.]*\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|without)\s+playoff\s+odds\s+(?:provided|available|in\s+the\s+(?:json|data))[^.]*\.?",
        re.IGNORECASE,
    ),
)


def scrub_ai_prose_field_names(text: str) -> str:
    """Replace leaked JSON field names and broken status lead-ins in model prose."""
    if text is None:
        return ""
    out = str(text)
    for pattern, repl in _AI_FIELD_NAME_REPLACEMENTS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    m = _BARE_STATUS_LEADIN.match(out)
    if m:
        label = m.group(1).lower()
        out = _BARE_STATUS_EXPANSION[label] + ", " + out[m.end():]

    for pat in _MISSING_ODDS_CLAUSES:
        out = pat.sub("", out)

    # Clean leftover " , " / leading commas after clause removal.
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s*,+", ",", out)
    out = re.sub(r"^\s*,\s*", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    # Capitalize if we stripped a leading clause and left a lowercase start.
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


def scrub_ai_result_strings(data: dict) -> dict:
    """Scrub field-name leaks from every string value in a structured AI result."""
    if not isinstance(data, dict):
        return data
    return {
        k: scrub_ai_prose_field_names(v) if isinstance(v, str) else v
        for k, v in data.items()
    }
