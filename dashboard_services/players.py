from __future__ import annotations

from typing import Dict, List

from .api import get_rosters, get_users, _avatar_from_users


def _first(seq, default=None):
    return seq[0] if isinstance(seq, (list, tuple)) and len(seq) else default


def get_players_map(data: dict = None) -> dict[str, dict[str, str]]:
    mp: dict[str, dict[str, str]] = {}
    for pid, p in data.items():
        name = (
                p.get("full_name")
                or p.get("search_full_name")
                or " ".join([x for x in (p.get("first_name"), p.get("last_name")) if x])
                or str(pid)
        )
        team = p.get("team") or "FA"
        pos = p.get("position") or _first(p.get("fantasy_positions"), "")
        mp[str(pid)] = {"name": str(name), "team": str(team), "pos": str(pos)}
    return mp


def build_roster_map(league_id: str, users=None, rosters=None) -> Dict[str, str]:
    if users is None:
        users = get_users(league_id)
    if rosters is None:
        rosters = get_rosters(league_id)
    user_fallback = {
        u["user_id"]: (
                (u.get("metadata") or {}).get("team_name")
                or u.get("display_name")
                or u.get("username")
                or str(u["user_id"])
        )
        for u in users
    }
    roster_map: Dict[str, str] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        meta = r.get("metadata") or {}
        owner_id = r.get("owner_id")
        display = meta.get("team_name") or user_fallback.get(owner_id, f"Roster {rid}")
        roster_map[rid] = display
    return roster_map


def get_league_rostered_player_ids(league_id: str) -> Dict[str, List[str]]:
    rosters = get_rosters(league_id) or []
    by_roster: Dict[str, List[str]] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        main = r.get("players") or []
        reserve = r.get("reserve") or []
        by_roster[rid] = [str(p) for p in (list(main) + list(reserve)) if p]
    return by_roster


def build_roster_display_maps(league_id: str):
    users = get_users(league_id)
    rosters = get_rosters(league_id)

    # same display-name logic as build_tables
    user_fallback = {
        u["user_id"]: (
                (u.get("metadata") or {}).get("team_name")
                or u.get("display_name")
                or u.get("username")
                or str(u["user_id"])
        ) for u in users
    }

    roster_name: dict[str, str] = {}
    roster_avatar: dict[str, str | None] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        name = (r.get("metadata") or {}).get("team_name") or user_fallback.get(owner_id, f"Roster {rid}")
        roster_name[rid] = name
        roster_avatar[rid] = _avatar_from_users(users, owner_id)
    return roster_name, roster_avatar
