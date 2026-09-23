"""Provider-authoritative, team-scoped season schedules for the team modal.

This module deliberately does not synthesize pairings, scores, or lineups.  A
missing provider week remains unavailable; it is not interpreted as a bye.
"""
from __future__ import annotations

from typing import Any, Callable


def _number(value: Any):
    # bool is not a score, and missing is materially different from zero.
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _status(row: dict, opponent: dict | None, *, week: int, current_week: int,
            viewed_season: int, current_season: int) -> str:
    raw = str(row.get("status") or (opponent or {}).get("status") or "").lower()
    if row.get("finalized") is True or raw in {"final", "complete", "completed", "post"}:
        return "final"
    if raw in {"live", "in_progress", "in progress", "in"}:
        return "live"
    if raw in {"scheduled", "upcoming", "pre"}:
        return "scheduled"
    if viewed_season < current_season or (viewed_season == current_season and week < current_week):
        # Historical provider totals are authoritative, but only call the game
        # final when both sides actually published totals.
        return "final" if opponent and _number(row.get("points")) is not None and _number(opponent.get("points")) is not None else "unavailable"
    # A numeric provider total (including 0) is not evidence that games have
    # kicked off: ESPN publishes zero totals for future/current matchups.
    # Live must be explicit provider/shared-state information.
    return "scheduled"


def _result(status: str, mine: Any, theirs: Any) -> str | None:
    a, b = _number(mine), _number(theirs)
    if status != "final" or a is None or b is None:
        return None
    return "T" if a == b else ("W" if a > b else "L")


def _season_end(league: dict) -> int:
    settings = (league or {}).get("settings") or {}
    for key in ("total_weeks", "schedule_weeks", "last_scoring_period", "matchup_period_count"):
        try:
            value = int(settings.get(key) or 0)
            if value > 0:
                return min(value, 25)
        except (TypeError, ValueError):
            pass
    try:
        start = int(settings.get("playoff_week_start") or 0)
        teams = int(settings.get("playoff_teams") or 0)
        rounds = max(1, (teams - 1).bit_length()) if teams else 3
        if start > 0:
            return min(25, start + rounds - 1)
    except (TypeError, ValueError):
        pass
    # NFL fantasy providers may not expose a schedule bound.  Eighteen is only
    # an upper request bound; absent weeks remain unpublished, never fabricated.
    return 18


def build_team_schedule(*, platform: str, league_id: str, season: int, roster_id: str,
                        league: dict, rosters: list[dict], users: list[dict],
                        current_season: int, current_week: int,
                        get_week: Callable[[int], list[dict]], details: bool = False,
                        only_week: int | None = None) -> dict:
    from dashboard_services.api import team_avatar
    from utils.utils import load_players_index

    roster_by_id = {str(r.get("roster_id")): r for r in rosters}
    if str(roster_id) not in roster_by_id:
        raise KeyError("roster not found")
    user_by_owner = {str(u.get("user_id")): u for u in users}
    players = load_players_index() or {} if details else {}

    def team_meta(rid: str) -> dict:
        r = roster_by_id.get(str(rid)) or {}
        u = user_by_owner.get(str(r.get("owner_id"))) or {}
        metadata = u.get("metadata") or {}
        return {"roster_id": str(rid), "team_name": metadata.get("team_name") or u.get("display_name") or f"Team {rid}",
                "avatar": team_avatar(platform, r, users) or ""}

    start, end = (only_week, only_week) if only_week else (1, _season_end(league))
    weeks = []
    for week in range(int(start), int(end) + 1):
        rows = get_week(week) or []
        mine = next((r for r in rows if str(r.get("roster_id")) == str(roster_id)), None)
        if mine is None:
            weeks.append({"week": week, "state": "unpublished", "published": False})
            continue
        mid = mine.get("matchup_id")
        opponent = next((r for r in rows if r is not mine and str(r.get("matchup_id")) == str(mid)), None)
        state = _status(mine, opponent, week=week, current_week=current_week,
                        viewed_season=season, current_season=current_season)
        item = {"week": week, "matchup_id": mid, "published": True, "state": state,
                "is_bye": bool(mine.get("is_bye")),
                "team": team_meta(str(roster_id)),
                "opponent": team_meta(str(opponent.get("roster_id"))) if opponent else None,
                "team_points": _number(mine.get("points")),
                "opponent_points": _number(opponent.get("points")) if opponent else None,
                "team_projection": _number(mine.get("projected_points")),
                "opponent_projection": _number(opponent.get("projected_points")) if opponent else None}
        item["result"] = _result(state, item["team_points"], item["opponent_points"])
        raw_adjustment = mine.get("custom_points")
        if raw_adjustment is None:
            raw_adjustment = mine.get("adjustment")
        adjustment = _number(raw_adjustment)
        if adjustment is not None:
            item["team_adjustment"] = adjustment
        if details:
            def lineup(row):
                if not row:
                    return None
                ppts = {str(k): _number(v) for k, v in (row.get("players_points") or {}).items()}
                slots = row.get("starter_slots") or []
                result = []
                for index, pid in enumerate(row.get("starters") or []):
                    pid = str(pid)
                    meta = players.get(pid) or {}
                    result.append({"player_id": pid, "name": meta.get("name") or "Unknown player",
                                   "position": meta.get("pos") or "", "slot": slots[index] if index < len(slots) else None,
                                   "points": ppts.get(pid)})
                return result
            item["team_lineup"] = lineup(mine)
            item["opponent_lineup"] = lineup(opponent)
            item["lineup_available"] = bool(mine.get("starters"))
        weeks.append(item)
    return {"platform": platform, "league_id": str(league_id), "season": int(season),
            "roster_id": str(roster_id), "weeks": weeks}
