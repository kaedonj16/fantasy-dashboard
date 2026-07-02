"""Pure input-validation / coercion helpers.

Extracted from app.py so they can be unit-tested without the pandas/DB stack.
"""
from __future__ import annotations

from typing import Optional


def safe_int(value, default=None):
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

    return False, f"Unsupported platform: {platform}"
