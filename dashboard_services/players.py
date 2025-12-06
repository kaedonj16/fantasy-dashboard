from __future__ import annotations

from typing import Dict, List, Optional

from .api import get_rosters, get_users, avatar_from_users


def _first(seq, default=None):
    return seq[0] if isinstance(seq, (list, tuple)) and len(seq) else default


def get_players_map(data: dict | None = None) -> dict[str, dict[str, str]]:
    """
    Build a simple player map:
      { player_id: {name, team, pos} }
    """
    mp: dict[str, dict[str, str]] = {}
    if not data:
        return mp

    for pid, p in data.items():
        get = p.get  # local binding for speed
        full_name = get("full_name")
        search_name = get("search_full_name")
        first_name = get("first_name")
        last_name = get("last_name")

        if full_name:
            name = full_name
        elif search_name:
            name = search_name
        elif first_name or last_name:
            name = " ".join(x for x in (first_name, last_name) if x)
        else:
            name = str(pid)

        team = get("team") or "FA"
        pos = get("position") or _first(get("fantasy_positions"), "")

        pid_s = str(pid)
        mp[pid_s] = {
            "name": str(name),
            "team": str(team),
            "pos": str(pos),
        }
    return mp


def build_roster_map(
    league_id: str,
    users: Optional[list[dict]] = None,
    rosters: Optional[list[dict]] = None,
) -> Dict[str, str]:
    """
    roster_id -> display team name
    """
    if users is None:
        users = get_users(league_id)
    if rosters is None:
        rosters = get_rosters(league_id)

    # Precompute fallback names for all users
    user_fallback: Dict[str, str] = {}
    for u in users:
        uid = u["user_id"]
        meta = u.get("metadata") or {}
        name = (
            meta.get("team_name")
            or u.get("display_name")
            or u.get("username")
            or str(uid)
        )
        user_fallback[uid] = name

    roster_map: Dict[str, str] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        meta = r.get("metadata") or {}
        owner_id = r.get("owner_id")
        display = meta.get("team_name") or user_fallback.get(owner_id, f"Roster {rid}")
        roster_map[rid] = display

    return roster_map


def get_league_rostered_player_ids(league_id: str) -> Dict[str, List[str]]:
    """
    roster_id -> [player_id, ...] including reserve.
    """
    rosters = get_rosters(league_id) or []
    by_roster: Dict[str, List[str]] = {}

    for r in rosters:
        rid = str(r.get("roster_id"))
        main = r.get("players") or ()
        reserve = r.get("reserve") or ()
        # keep same behavior: convert to str, drop falsy
        by_roster[rid] = [str(p) for p in (*main, *reserve) if p]

    return by_roster


def build_roster_display_maps(league_id: str):
    """
    Returns:
      roster_name:   roster_id -> display name
      roster_avatar: roster_id -> avatar url or None
    """
    users = get_users(league_id)
    rosters = get_rosters(league_id)

    user_fallback: Dict[str, str] = {}
    for u in users:
        uid = u["user_id"]
        meta = u.get("metadata") or {}
        name = (
            meta.get("team_name")
            or u.get("display_name")
            or u.get("username")
            or str(uid)
        )
        user_fallback[uid] = name

    roster_name: dict[str, str] = {}
    roster_avatar: dict[str, Optional[str]] = {}

    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        meta = r.get("metadata") or {}

        name = meta.get("team_name") or user_fallback.get(owner_id, f"Roster {rid}")
        roster_name[rid] = name
        roster_avatar[rid] = avatar_from_users(users, owner_id)

    return roster_name, roster_avatar
