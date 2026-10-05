"""Consolidated utils module: core.

tiny pure helpers (coercion, validation, formatting, urls)

Merged from: utils/coerce.py, utils/validation.py, utils/format.py, utils/relative_time.py, utils/json_sanitize.py, utils/safe_url.py, utils/api_params.py, utils/html_sanitize.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# --- imports carried over from utils/utils.py ---
import glob
import json
import os
import re
import threading as _threading
import requests
import time
import traceback
import uuid
from contextlib import contextmanager as _contextmanager
from bs4 import BeautifulSoup
from collections import OrderedDict as _OrderedDict, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# ======================================================================
# From utils/coerce.py
# ======================================================================

"""Canonical numeric coercion helpers.

These replace a family of near-identical private ``_safe_float`` / ``_safe_int``
helpers that had been copy-pasted across the codebase. Only the call sites whose
local helper was *proven* behaviorally identical (tested across a full input
battery and both call patterns) import from here; variants with intentionally
different semantics — pandas ``pd.isna`` handling, ``default=None``, NaN
stripping, or ``int(float(s))`` string parsing — were deliberately left in place.

Behavior: ``None`` and blank/whitespace-only strings return ``default``;
everything else is coerced via ``float()`` / ``int()``; any ``TypeError`` or
``ValueError`` returns ``default``. NaN is intentionally NOT stripped
(``float('nan')`` round-trips), matching the consolidated call sites.
"""


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


# ======================================================================
# From utils/validation.py
# ======================================================================

"""Pure input-validation / coercion helpers.

Extracted from app.py so they can be unit-tested without the pandas/DB stack.
"""


def safe_int_or_none(value, default=None):
    """Coerce ``value`` to int, returning ``default`` on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate_league_id(platform: str, league_id: str) -> "tuple[bool, Optional[str]]":
    """Validate a league id for a given platform.

    Returns ``(ok, error_message)``. ``error_message`` is None when valid.
    """
    if not league_id:
        return False, "League ID is required."

    platform = (platform or "").lower().strip()

    if platform == "sleeper":
        if not league_id.isdigit():
            return False, "Invalid Sleeper league ID. Please check it and try again."
        return True, None

    if platform == "espn":
        if not league_id.isdigit():
            return False, "Invalid ESPN league ID. It should be a number."
        return True, None

    if platform == "yahoo":
        if not league_id.isdigit():
            return False, "Invalid Yahoo league ID. It should be a number."
        return True, None

    if platform == "mfl":
        if not league_id.isdigit():
            return False, "Invalid MFL league ID. It should be a number."
        return True, None

    if platform == "fleaflicker":
        if not league_id.isdigit():
            return False, "Invalid Fleaflicker league ID. It should be a number."
        return True, None

    return False, f"Unsupported platform: {platform}"

# ======================================================================
# From utils/format.py
# ======================================================================

"""Small pure formatting helpers.

Extracted from app.py, where the same ordinal logic was reimplemented in at
least three places (_ord_str, _dash_ord, _ordinal) plus several inline
`{1:"st",2:"nd",3:"rd"}.get(...)` snippets. Centralized and unit-tested so the
"11th/12th/13th" special case can't be gotten wrong in one copy and right in
another.
"""


def ord_suffix(n) -> str:
    """Ordinal suffix for an integer: 1->'st', 2->'nd', 3->'rd', 4->'th',
    correctly returning 'th' for the 11-13 teens (11th, 12th, 13th)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "th"
    if 10 <= (n % 100) <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def ordinal(n) -> str:
    """Full ordinal string: 1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    return f"{n}{ord_suffix(n)}"


# ======================================================================
# From utils/relative_time.py
# ======================================================================

"""Human-relative timestamp formatting ("Just now", "2d ago", "May 12").

Extracted from app.py; ``now`` is injectable so the boundaries are testable.
"""
from datetime import timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def rel_time(dt, now: datetime = None) -> str:
    """Human-relative timestamp: 'Just now', '5m ago', 'Today 3:42 PM',
    'Yesterday', '3d ago', '2w ago', then 'May 12' beyond a month."""
    now = (now or datetime.now(EASTERN)).astimezone(EASTERN)
    dt_et = dt.astimezone(EASTERN)
    diff = now - dt_et
    secs = diff.total_seconds()
    if secs < 60:
        return "Just now"
    if secs < 3600:
        mins = int(secs // 60)
        return f"{mins}m ago"
    today = now.date()
    if dt_et.date() == today:
        hour = dt_et.strftime("%I").lstrip("0") or "12"
        return f"Today {hour}:{dt_et.strftime('%M %p')}"
    if dt_et.date() == (now - timedelta(days=1)).date():
        return "Yesterday"
    days = (today - dt_et.date()).days
    if days < 7:
        return f"{days}d ago"
    if days < 30:
        return f"{days // 7}w ago"
    return dt_et.strftime("%b %d")


# ======================================================================
# From utils/json_sanitize.py
# ======================================================================

"""Make payloads JSON-safe by replacing non-finite floats with None.

json.dumps happily emits NaN/Infinity literals, which are invalid JSON and
make the browser's fetch().json() throw. Consolidates app.py's two former
near-duplicates (_sanitize_for_json handled NaN and infinities;
clean_nan_for_json handled only NaN) into one function that handles both.
"""
import math


def sanitize_for_json(obj):
    """Recursively replace NaN/inf/-inf floats with None in dicts/lists."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ======================================================================
# From utils/safe_url.py
# ======================================================================

"""Same-origin redirect helpers.

Used by OAuth callbacks, Stripe return URLs, and other places that accept a
``next`` / ``return_to`` query param. Rejects protocol-relative and absolute
off-site URLs so open redirects cannot be chained through auth flows.
"""


def safe_local_url(value: str | None, fallback: str = "/", *, host_url: str | None = None) -> str:
    """Return ``value`` when it is a same-site path (or absolute same-host URL).

    Args:
        value: Candidate redirect target from user/query input.
        fallback: Used when ``value`` is empty or unsafe.
        host_url: Optional ``request.host_url`` (trailing slash ok). When set,
            absolute URLs matching that host are accepted; otherwise only
            root-relative paths are allowed.
    """
    value = str(value or "").strip()
    if not value:
        return fallback
    # Root-relative only — never protocol-relative ("//evil.com").
    if value.startswith("/") and not value.startswith("//"):
        return value
    if host_url:
        base = host_url.rstrip("/")
        if value.startswith(base + "/") or value == base:
            return value
    return fallback


# ======================================================================
# From utils/api_params.py
# ======================================================================

"""Typed query-parameter parsing for /api/* routes.

Bad values (``?season=abc``) must surface as 400 with a JSON body, not an
unhandled ValueError that becomes a 500. The app's 400 error handler renders
the JSON envelope for /api/* paths and JSON-Accept callers.
"""

from flask import abort, request

_MISSING = object()


def api_int(name: str, default=_MISSING, *, minimum: int | None = None,
            maximum: int | None = None) -> int | None:
    """Read an integer query parameter.

    - Missing or blank -> ``default`` (or 400 when no default is given).
    - Present but not an integer, or outside [minimum, maximum] -> 400.
    """
    raw = request.args.get(name)
    if raw is None or not str(raw).strip():
        if default is _MISSING:
            abort(400, description=f"Missing required query parameter: {name}.")
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        abort(400, description=f"Invalid '{name}' parameter: expected an integer.")
    if minimum is not None and value < minimum:
        abort(400, description=f"Invalid '{name}' parameter: must be at least {minimum}.")
    if maximum is not None and value > maximum:
        abort(400, description=f"Invalid '{name}' parameter: must be at most {maximum}.")
    return value


def api_str(name: str, default: str = "", *, max_length: int = 200) -> str:
    """Read a string query parameter, stripped and length-capped."""
    raw = request.args.get(name, default)
    text = str(raw or "").strip()
    return text[:max_length]


# ======================================================================
# From utils/html_sanitize.py
# ======================================================================

"""Strip developer HTML comments from served pages.

AdSense reviewers (and crawlers) see raw page text, so internal notes left in
HTML comments read as sloppy and can leak implementation detail. Source files
keep their comments (useful while developing); this removes them from the
responses users and reviewers actually receive.
"""

# Blocks whose raw text must be left alone: a `<!--` inside JS/CSS/pre/textarea
# is content, not markup (e.g. JS strings, JSON-LD).
_LITERAL_BLOCK_RE = re.compile(
    r"(?is)(<(?:script|style|pre|textarea)\b.*?</(?:script|style|pre|textarea)>)"
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def strip_html_comments(html: str) -> str:
    """Remove ``<!-- ... -->`` comments outside literal (script/style/pre/textarea) blocks."""
    parts = _LITERAL_BLOCK_RE.split(html)
    for i in range(0, len(parts), 2):
        parts[i] = _COMMENT_RE.sub("", parts[i])
    return "".join(parts)


# ======================================================================
# From utils/utils.py (split per consolidation map)
# ======================================================================

# --- utils/utils.py L81 ---
BETTER_OUTWARD_METRICS = ["PF", "PA", "MAX", "MIN", "AVG", "STD"]

# --- utils/utils.py L84 ---
BETTER_OUTWARD_SIGNS = [1.0, -1.0, 1.0, 1.0, 1.0, -1.0]

# --- utils/utils.py L247 ---
def z_better_outward(team_stats: "pd.DataFrame",
                     metrics=BETTER_OUTWARD_METRICS,
                     signs=BETTER_OUTWARD_SIGNS) -> "pd.DataFrame":
    import numpy as np
    signs = np.asarray(signs, dtype=float)
    Z = (team_stats[metrics] - team_stats[metrics].mean()) / team_stats[metrics].std(ddof=0)
    return Z * signs
