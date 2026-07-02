"""Pure viewer-resolution helpers.

Extracted from app.py so the username -> roster resolution can be unit-tested
without the pandas/DB/session stack. No IO, no Flask session — just dict
matching over the league's users/rosters payloads.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union


def normalize_sleeper_username(value: str) -> str:
    return (value or "").strip().lower()


def resolve_viewer_for_league(users: List[Dict], rosters: List[Dict], username: str,
                              user_id: Optional[str] = None) -> Union[Dict, None]:
    """
    Resolve a Sleeper username (or ESPN team/owner name) to:
      - user_id
      - roster_id
      - display_name / team name

    For ESPN leagues the `username` field holds the owner's display_name or team_name
    because ESPN doesn't use Sleeper-style usernames.

    Prefer matching by user_id (unambiguous) when provided; fall back to name matching.
    """
    matched_user = None

    # Primary: match by user_id - avoids false matches on team_name collisions
    if user_id:
        for u in users or []:
            if str(u.get("user_id") or "") == str(user_id):
                matched_user = u
                break

    # Fallback: match by normalized username / display_name / team_name
    if not matched_user:
        wanted = normalize_sleeper_username(username)
        if not wanted:
            return None
        for u in users or []:
            meta = u.get("metadata") or {}
            candidates = [
                normalize_sleeper_username(u.get("display_name") or ""),
                normalize_sleeper_username(u.get("username") or ""),
                normalize_sleeper_username(meta.get("team_name") or ""),
            ]
            if wanted in candidates:
                matched_user = u
                break

    if not matched_user:
        return None

    user_id = str(matched_user.get("user_id") or "")
    if not user_id:
        return None

    matched_roster = None
    for r in rosters or []:
        owner_id = str(r.get("owner_id") or "")
        if owner_id == user_id:
            matched_roster = r
            break

    meta_u = matched_user.get("metadata") or {}
    if not matched_roster:
        return {
            "viewer_username": username,
            "viewer_user_id": user_id,
            "viewer_roster_id": None,
            "viewer_team_name": (
                    meta_u.get("team_name")
                    or matched_user.get("display_name")
                    or matched_user.get("username")
                    or "Unknown Team"
            ),
        }

    metadata = matched_roster.get("metadata") or {}
    team_name = (
            metadata.get("team_name")
            or meta_u.get("team_name")
            or matched_user.get("display_name")
            or matched_user.get("username")
            or f"Roster {matched_roster.get('roster_id')}"
    )

    return {
        "viewer_username": username,
        "viewer_user_id": user_id,
        "viewer_roster_id": str(matched_roster.get("roster_id")),
        "viewer_team_name": team_name,
    }
