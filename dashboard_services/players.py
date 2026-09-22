from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .api import avatar_from_users, team_avatar
from .display_names import public_owner_label
from .platform_api import get_users, get_rosters


def _first(seq, default=None):
    return seq[0] if isinstance(seq, (list, tuple)) and len(seq) else default


# ``get_players_map`` turns the full NFL player universe (~11k entries) into a
# fresh {pid: {name, team, pos}} dict. ``build_league_context`` calls it once
# per league, so a member of many leagues would otherwise rebuild and retain
# one full-universe copy per cached context (a per-league memory cost that
# scaled with league count and drove OOM on the portfolio). The source is a
# process-shared global that changes rarely and every caller treats the result
# as read-only, so memoize a single instance keyed on the source's identity and
# share it across all contexts. A strong reference to the source is retained so
# its ``id`` cannot be reused by a later object while the cached map is live.
_PLAYERS_MAP_CACHE: dict[str, object] = {"src_id": None, "src": None, "map": None}
_PLAYERS_MAP_LOCK = threading.Lock()


def get_players_map(data: dict | None = None) -> dict[str, dict[str, str]]:
    """
    Build a simple player map:
      { player_id: {name, team, pos} }

    The returned dict is shared across callers and must be treated as
    read-only; mutating it would corrupt every league context that reused it.
    """
    if not data:
        return {}

    src_id = id(data)
    cache = _PLAYERS_MAP_CACHE
    if cache["src"] is data and cache["src_id"] == src_id and cache["map"] is not None:
        return cache["map"]  # type: ignore[return-value]

    mp = _build_players_map(data)

    with _PLAYERS_MAP_LOCK:
        cache["src_id"] = src_id
        cache["src"] = data
        cache["map"] = mp
    return mp


def _build_players_map(data: dict) -> dict[str, dict[str, str]]:
    mp: dict[str, dict[str, str]] = {}
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
        platform: str,
        season: str,
        users: Optional[list[dict]] = None,
        rosters: Optional[list[dict]] = None,
) -> Dict[str, str]:
    """
    roster_id -> display team name
    """
    if users is None:
        users = get_users(platform, league_id, season)
    if rosters is None:
        rosters = get_rosters(platform, league_id, season)

    user_fallback: Dict[str, str] = {}
    for u in users:
        uid = str(u.get("user_id") or "")
        if not uid:
            continue
        meta = u.get("metadata") or {}
        name = public_owner_label(
            meta.get("team_name"),
            u.get("display_name"),
            u.get("username"),
            fallback=uid,
        )
        user_fallback[uid] = name

    roster_map: Dict[str, str] = {}
    for r in rosters:
        rid = str(r["roster_id"])
        meta = r.get("metadata") or {}
        owner_id = str(r.get("owner_id") or "")
        display = public_owner_label(
            meta.get("team_name"),
            user_fallback.get(owner_id),
            fallback=f"Roster {rid}",
        )
        roster_map[rid] = display

    return roster_map


def get_league_rostered_player_ids(league_id: str, rosters) -> Dict[str, List[str]]:
    """
    roster_id -> [player_id, ...] including reserve.
    """
    by_roster: Dict[str, List[str]] = {}

    for r in rosters:
        rid = str(r.get("roster_id"))
        main = r.get("players") or ()
        reserve = r.get("reserve") or ()
        # keep same behavior: convert to str, drop falsy
        by_roster[rid] = [str(p) for p in (*main, *reserve) if p]

    return by_roster


def build_roster_display_maps(
        league_id: str,
        platform,
        season,
        users: Optional[list[dict]] = None,
        rosters: Optional[list[dict]] = None,
):
    """
    Returns:
      roster_name:   roster_id -> display name
      roster_avatar: roster_id -> avatar url or None
    """
    if users is None:
        users = get_users(platform, league_id, season)
    if rosters is None:
        rosters = get_rosters(platform, league_id, season)

    user_fallback: Dict[str, str] = {}
    for u in users:
        uid = u["user_id"]
        meta = u.get("metadata") or {}
        name = public_owner_label(
            meta.get("team_name"),
            u.get("display_name"),
            u.get("username"),
            fallback=str(uid),
        )
        user_fallback[uid] = name

    roster_name: dict[str, str] = {}
    roster_avatar: dict[str, Optional[str]] = {}

    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        meta = r.get("metadata") or {}

        name = public_owner_label(
            meta.get("team_name"),
            user_fallback.get(owner_id),
            fallback=f"Roster {rid}",
        )
        roster_name[rid] = name
        roster_avatar[rid] = team_avatar(platform, r, users)

    return roster_name, roster_avatar
