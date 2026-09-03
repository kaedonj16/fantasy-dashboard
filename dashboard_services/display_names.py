"""Public team / owner labels that never leak provider privacy placeholders.

Yahoo Fantasy returns ``--hidden--`` as a manager nickname when the user has
hidden their username. Downstream APIs that prefer ``display_name`` over the
actual team name then render that placeholder in the trade calculator.
"""
from __future__ import annotations

from typing import Any

_HIDDEN_OWNER_LABELS = frozenset({
    "--hidden--",
    "-hidden-",
    "hidden",
})


def public_owner_label(*candidates: Any, fallback: str = "") -> str:
    """First non-empty candidate that is not a privacy placeholder."""
    for cand in candidates:
        text = str(cand or "").strip()
        if text and text.lower() not in _HIDDEN_OWNER_LABELS:
            return text
    return fallback


def team_label_from_user(
    user: dict | None,
    roster: dict | None = None,
    *,
    fallback: str = "",
) -> str:
    """Prefer roster/user team_name, then a public display_name / username."""
    user = user or {}
    roster = roster or {}
    umeta = user.get("metadata") or {}
    rmeta = roster.get("metadata") or {}
    return public_owner_label(
        rmeta.get("team_name"),
        umeta.get("team_name"),
        user.get("team_name"),
        user.get("display_name"),
        user.get("username"),
        fallback=fallback,
    )


def username_from_user(user: dict | None, *, fallback: str = "") -> str:
    """Owner handle that never surfaces ``--hidden--``."""
    user = user or {}
    umeta = user.get("metadata") or {}
    return public_owner_label(
        user.get("username"),
        user.get("display_name"),
        umeta.get("team_name"),
        user.get("team_name"),
        fallback=fallback,
    )
