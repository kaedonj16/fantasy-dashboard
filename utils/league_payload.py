"""Pure helpers over Sleeper/ESPN league payload dicts.

Extracted from app.py so these transforms can be unit-tested without the
pandas/DB stack. All pure — dict in, dict/value out.
"""
from __future__ import annotations

from typing import Optional

from utils.validation import safe_int


def format_sleeper_league_option(league: dict) -> dict:
    """Shape a raw Sleeper league dict into the option payload the picker uses."""
    settings = league.get("settings") or {}

    return {
        "league_id": str(league.get("league_id", "")),
        "name": league.get("name") or "Unnamed League",
        "season": str(league.get("season") or ""),
        "total_rosters": league.get("total_rosters") or settings.get("num_teams") or "",
        "avatar": league.get("avatar") or "",
        "label": (
            f"{league.get('name') or 'Unnamed League'} "
            f"({league.get('season') or ''}) • "
            f"{league.get('total_rosters') or settings.get('num_teams') or '?'} teams"
        ),
    }


def get_most_recent_valid_draft_for_season(drafts: list, season: int) -> Optional[dict]:
    """
    Pick the most recent draft from the provided list, using the best available
    timestamp field. Return it only if it belongs to the viewed season.

    If the newest draft is from an older season, return None so the caller
    can keep TBD logic.
    """
    if not isinstance(drafts, list) or not drafts:
        return None

    def draft_sort_ts(d: dict) -> int:
        if not isinstance(d, dict):
            return -1
        return max(
            safe_int(d.get("start_time"), -1),
            safe_int(d.get("created"), -1),
            safe_int(d.get("last_picked"), -1),
            safe_int(d.get("last_message_time"), -1),
        )

    valid_drafts = [d for d in drafts if isinstance(d, dict)]
    if not valid_drafts:
        return None

    most_recent = max(valid_drafts, key=draft_sort_ts)
    most_recent_season = safe_int(most_recent.get("season"))

    if most_recent_season != int(season):
        return None

    return most_recent


def build_roster_map(users: list, rosters: list) -> dict:
    """Map roster_id -> display name, using metadata.team_name with user fallback."""
    user_fallback = {
        u["user_id"]: (
                (u.get("metadata") or {}).get("team_name")
                or u.get("display_name")
                or u.get("username")
                or str(u["user_id"])
        )
        for u in users
    }
    roster_map = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        roster_map[rid] = (r.get("metadata") or {}).get("team_name") or user_fallback.get(
            owner_id, f"Roster {rid}"
        )
    return roster_map
