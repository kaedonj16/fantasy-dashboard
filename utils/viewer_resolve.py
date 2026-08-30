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
    Callers that pass a league-scoped team/roster id as ``user_id`` (common for
    ESPN/Yahoo/MFL pickers) are also resolved: ESPN owner ids are SWIDs, so the
    roster_id path is required for scout and other personalized tabs.
    """
    matched_user = None
    matched_roster = None
    wanted_id = str(user_id or "").strip()

    # Primary: match by user_id - avoids false matches on team_name collisions
    if wanted_id:
        for u in users or []:
            if str(u.get("user_id") or "") == wanted_id:
                matched_user = u
                break
        # ESPN (and some link flows) pass the roster/team id, not the owner SWID.
        if not matched_user:
            for r in rosters or []:
                if str(r.get("roster_id") or "") == wanted_id:
                    matched_roster = r
                    owner_id = str(r.get("owner_id") or "")
                    if owner_id:
                        for u in users or []:
                            if str(u.get("user_id") or "") == owner_id:
                                matched_user = u
                                break
                    break

    # Fallback: match by username, then display_name, then team_name — but only
    # when the match is unique. Two "Dream Team" owners must not resolve to
    # whichever user happens to appear first in the payload.
    if not matched_user and not matched_roster:
        wanted = normalize_sleeper_username(username)
        if not wanted:
            return None

        def _unique_match(getter):
            hits = [u for u in (users or [])
                    if normalize_sleeper_username(getter(u)) == wanted]
            return hits[0] if len(hits) == 1 else None

        matched_user = (
            _unique_match(lambda u: u.get("username") or "")
            or _unique_match(lambda u: u.get("display_name") or "")
            or _unique_match(lambda u: (u.get("metadata") or {}).get("team_name") or "")
        )

    if not matched_user and not matched_roster:
        return None

    # Roster-only hit (team id known, owner missing from users payload): still
    # unlock scout / personalized tabs with the roster identity.
    if not matched_user and matched_roster:
        rid = str(matched_roster.get("roster_id") or "")
        meta_r = matched_roster.get("metadata") or {}
        team_name = (
            meta_r.get("team_name")
            or (username or "").strip()
            or f"Roster {rid}"
        )
        return {
            "viewer_username": username or team_name,
            "viewer_user_id": str(matched_roster.get("owner_id") or "") or None,
            "viewer_roster_id": rid or None,
            "viewer_team_name": team_name,
        }

    resolved_user_id = str(matched_user.get("user_id") or "")
    if not resolved_user_id:
        return None

    if not matched_roster:
        for r in rosters or []:
            owner_id = str(r.get("owner_id") or "")
            if owner_id == resolved_user_id:
                matched_roster = r
                break

    meta_u = matched_user.get("metadata") or {}
    if not matched_roster:
        return {
            "viewer_username": username,
            "viewer_user_id": resolved_user_id,
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
        "viewer_username": username or team_name,
        "viewer_user_id": resolved_user_id,
        "viewer_roster_id": str(matched_roster.get("roster_id")),
        "viewer_team_name": team_name,
    }
