# dashboard_services/espn_api.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from espn_api.football import League


class ESPNError(Exception):
    pass


def _require_env(name: str) -> str:
    val = (os.getenv(name, "") or "").strip()
    if not val:
        raise ESPNError(f"Missing required env var: {name}")
    return val


def _normalize_swid(swid: str) -> str:
    swid = swid.strip()
    if swid and not (swid.startswith("{") and swid.endswith("}")):
        swid = "{" + swid.strip("{}") + "}"
    return swid


def _league(season: int, league_id: str) -> League:
    """
    Uses cwendt94/espn-api (pip: espn-api).
    Requires ESPN_S2 and ESPN_SWID.
    """
    espn_s2 = _require_env("ESPN_S2")
    swid = _normalize_swid(_require_env("ESPN_SWID"))

    try:
        return League(
            league_id=int(league_id),
            year=int(season),
            espn_s2=espn_s2,
            swid=swid,
        )
    except Exception as e:
        raise ESPNError(f"Failed to init ESPN League(season={season}, league_id={league_id}): {e}")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return default if x is None else float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return default if x is None else int(x)
    except Exception:
        return default


def _split_points(val: Any) -> tuple[int, int]:
    """
    Matches your existing fpts + fpts_decimal split.
    """
    f = _safe_float(val, 0.0)
    whole = int(f)
    dec = int(round((f - whole) * 100))
    return whole, dec


# -------------------------
# Public API (same exports)
# -------------------------

def get_league(season: int, league_id: str) -> Dict[str, Any]:
    lg = _league(season, league_id)

    # espn-api exposes some league fields; keep it resilient.
    name = (
        getattr(lg, "settings", None).name
        if getattr(lg, "settings", None) is not None and getattr(getattr(lg, "settings"), "name", None)
        else getattr(lg, "league_name", None)
        or getattr(lg, "name", None)
        or "ESPN League"
    )

    return {
        "league_id": str(league_id),
        "name": name,
        "season": int(season),
        # you can add "raw" here if you want, but you asked to remove raw for users;
        # keeping league small by default:
    }


def get_users(season: int, league_id: str) -> List[Dict[str, Any]]:
    lg = _league(season, league_id)
    swid = _normalize_swid(os.getenv("ESPN_SWID", "") or "")

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for t in getattr(lg, "teams", []) or []:
        owners = getattr(t, "owners", None) or []
        team_name = (
            getattr(t, "team_name", None)
            or getattr(t, "teamName", None)
            or getattr(t, "name", None)
            or ""
        )

        # Optional team logo (ESPN URL). If you don’t want this at all, set avatar=None.
        logo = (
            getattr(t, "logo_url", None)
            or getattr(t, "logoUrl", None)
            or getattr(t, "logo", None)
        )

        for o in owners:
            # owners are dict-ish in espn-api
            oid = str(o.get("id") or o.get("ownerId") or "").strip()
            if not oid or oid in seen:
                continue
            seen.add(oid)

            display = (
                o.get("displayName")
                or o.get("firstName")
                or (f'{o.get("firstName","")} {o.get("lastName","")}'.strip())
                or f"User {oid}"
            )

            out.append({
                "avatar": logo if logo else None,
                "display_name": display,
                "is_bot": False,
                "is_owner": (swid == oid) if swid else None,
                "league_id": str(league_id),
                "metadata": {"team_name": team_name} if team_name else {},
                "settings": None,
                "user_id": oid,
            })

    return out


def get_rosters(season: int, league_id: str) -> List[Dict[str, Any]]:
    """
    Returns your unified Sleeper-shaped rosters (same shape as your current ESPN version):
      [
        {
          'co_owners': None,
          'keepers': None,
          'league_id': str,
          'metadata': {'record': '', 'streak': '2W'},
          'owner_id': str | None,
          'player_map': None,
          'players': [...],
          'reserve': [...],
          'roster_id': int,
          'settings': {...},
          'starters': [...],
          'taxi': None
        }
      ]
    """
    lg = _league(season, league_id)

    rosters: List[Dict[str, Any]] = []

    for t in getattr(lg, "teams", []) or []:
        roster_id = _safe_int(getattr(t, "team_id", None) or getattr(t, "teamId", None) or getattr(t, "id", None), 0)
        owners = getattr(t, "owners", None) or []
        owner_id = str(owners[0].get("id")) if owners else None

        # record + points (espn-api Team usually has wins/losses/ties and points_for/points_against-ish fields,
        # but names can vary; we guard everything).
        wins = _safe_int(getattr(t, "wins", None), 0)
        losses = _safe_int(getattr(t, "losses", None), 0)
        ties = _safe_int(getattr(t, "ties", None), 0)

        points_for = (
            getattr(t, "points_for", None)
            or getattr(t, "pointsFor", None)
            or getattr(t, "points", None)
            or 0.0
        )
        points_against = (
            getattr(t, "points_against", None)
            or getattr(t, "pointsAgainst", None)
            or 0.0
        )

        fpts_whole, fpts_dec = _split_points(points_for)
        fpa_whole, fpa_dec = _split_points(points_against)

        # streak (espn-api often exposes streak as e.g. "W3" or similar; if not, blank it)
        streak = getattr(t, "streak", None) or ""
        # normalize to "3W" / "2L" style if it’s "W3"
        if isinstance(streak, str) and len(streak) >= 2 and streak[0] in ("W", "L", "T") and streak[1:].isdigit():
            streak = f"{streak[1:]}{streak[0]}"

        # roster players
        roster = getattr(t, "roster", None) or []
        players: List[str] = []
        starters: List[str] = []
        reserve: List[str] = []

        for p in roster:
            pid = getattr(p, "playerId", None) or getattr(p, "player_id", None)
            if pid is None:
                continue
            pid_s = str(pid)
            players.append(pid_s)

            # slot_position is common for lineup contexts; if missing, default to starters unknown.
            slot = getattr(p, "slot_position", None) or getattr(p, "slotPosition", None) or getattr(p, "lineupSlot", None)

            # common ESPN slot labels in espn-api box/roster contexts
            if slot in ("IR", "RES"):
                reserve.append(pid_s)
            elif slot in ("BE", "Bench", "Inactive"):
                # bench -> neither starter nor reserve
                pass
            else:
                starters.append(pid_s)

        rosters.append({
            "co_owners": None,
            "keepers": None,
            "league_id": str(league_id),
            "metadata": {
                "record": "",       # if you want W/L string by week, compute from schedule later
                "streak": streak or "",
            },
            "owner_id": owner_id,
            "player_map": None,
            "players": players,
            "reserve": reserve,
            "roster_id": roster_id,
            "settings": {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "fpts": fpts_whole,
                "fpts_decimal": fpts_dec,
                "fpts_against": fpa_whole,
                "fpts_against_decimal": fpa_dec,
                "ppts": 0,
                "ppts_decimal": 0,
                "total_moves": 0,          # espn-api doesn’t reliably expose this across leagues
                "waiver_budget_used": 0,
                "waiver_position": 0,
            },
            "starters": starters,
            "taxi": None,
        })

    return rosters


def get_matchups(season: int, league_id: str, week: int) -> List[Dict[str, Any]]:
    """
    Produces the exact shape you want (players, starters, points, players_points),
    using espn-api box_scores() so it won't be blank.
    """
    lg = _league(season, league_id)

    try:
        box_scores = lg.box_scores(week)
    except Exception as e:
        raise ESPNError(f"Failed to fetch box_scores for week={week}: {e}")

    out: List[Dict[str, Any]] = []
    matchup_id = 0

    for bs in box_scores:
        matchup_id += 1

        home_team = getattr(bs, "home_team", None)
        away_team = getattr(bs, "away_team", None)

        home_id = _safe_int(getattr(home_team, "team_id", None) or getattr(home_team, "id", None), 0)
        away_id = _safe_int(getattr(away_team, "team_id", None) or getattr(away_team, "id", None), 0)

        home_score = _safe_float(getattr(bs, "home_score", 0.0), 0.0)
        away_score = _safe_float(getattr(bs, "away_score", 0.0), 0.0)

        home_lineup = getattr(bs, "home_lineup", None) or []
        away_lineup = getattr(bs, "away_lineup", None) or []

        def build_side(roster_id: int, points: float, lineup: list) -> Dict[str, Any]:
            players: List[str] = []
            starters: List[str] = []
            starters_points: List[float] = []
            players_points: Dict[str, float] = {}

            for bp in lineup:
                pid = getattr(bp, "playerId", None) or getattr(bp, "player_id", None)
                if pid is None:
                    continue
                pid_s = str(pid)

                pts = _safe_float(getattr(bp, "points", None), 0.0)
                slot = getattr(bp, "slot_position", None) or getattr(bp, "slotPosition", None)

                players.append(pid_s)
                players_points[pid_s] = pts

                if slot not in ("BE", "Bench", "IR", "RES", "Inactive"):
                    starters.append(pid_s)
                    starters_points.append(pts)

            return {
                "points": points,
                "players": players,
                "roster_id": roster_id,
                "custom_points": None,
                "matchup_id": matchup_id,
                "starters": starters,
                "starters_points": starters_points,
                "players_points": players_points,
            }

        out.append(build_side(home_id, home_score, home_lineup))
        out.append(build_side(away_id, away_score, away_lineup))

    return out


def get_transactions(season: int, league_id: str, week: int) -> List[Dict[str, Any]]:
    """
    Keep your interface, but espn-api’s transaction support varies a lot by league/settings.
    If you later want this, we can wire it to:
      - lg.recent_activity(msgs=...) (if present)
      - lg.transactions(week) (if present)
    For now: safe empty list.
    """
    return []
