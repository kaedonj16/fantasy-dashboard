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
    from utils.data_cache import load_players_index

    roster_by_id = {str(r.get("roster_id")): r for r in rosters}
    if str(roster_id) not in roster_by_id:
        raise KeyError("roster not found")
    user_by_owner = {str(u.get("user_id")): u for u in users}
    players = load_players_index() or {} if details else {}

    def _settings_of(rid: str) -> dict:
        return (roster_by_id.get(str(rid)) or {}).get("settings") or {}

    def _record_of(rid: str) -> dict:
        s = _settings_of(rid)
        def _int(key):
            try:
                return int(s.get(key) or 0)
            except (TypeError, ValueError):
                return 0
        return {"wins": _int("wins"), "losses": _int("losses"), "ties": _int("ties")}

    def _points_total(rid: str, base_key: str, dec_key: str):
        s = _settings_of(rid)
        total = _number(s.get(base_key))
        if total is None:
            return None
        dec = _number(s.get(dec_key)) or 0.0
        return round(total + dec / 100.0, 1)

    def team_meta(rid: str) -> dict:
        r = roster_by_id.get(str(rid)) or {}
        u = user_by_owner.get(str(r.get("owner_id"))) or {}
        metadata = u.get("metadata") or {}
        return {"roster_id": str(rid), "team_name": metadata.get("team_name") or u.get("display_name") or f"Team {rid}",
                "avatar": team_avatar(platform, r, users) or "",
                "record": _record_of(rid),
                "points_for": _points_total(rid, "fpts", "fpts_decimal"),
                "points_against": _points_total(rid, "fpts_against", "fpts_against_decimal")}

    start, end = (only_week, only_week) if only_week else (1, _season_end(league))
    weeks = []
    all_rows_by_week: dict[int, list[dict]] = {}
    for week in range(int(start), int(end) + 1):
        rows = get_week(week) or []
        if only_week is None:
            all_rows_by_week[week] = rows
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
    payload = {"platform": platform, "league_id": str(league_id), "season": int(season),
               "roster_id": str(roster_id), "weeks": weeks}
    if only_week is None:
        payload["context"] = _schedule_context(
            league=league, weeks=weeks, all_rows_by_week=all_rows_by_week,
            roster_by_id=roster_by_id, roster_id=str(roster_id),
            record_of=_record_of, points_total=_points_total, team_meta=team_meta)
    return payload


def _win_pct(record: dict) -> float | None:
    games = (record.get("wins") or 0) + (record.get("losses") or 0) + (record.get("ties") or 0)
    if games <= 0:
        return None
    return ((record.get("wins") or 0) + 0.5 * (record.get("ties") or 0)) / games


def _schedule_context(*, league: dict, weeks: list[dict], all_rows_by_week: dict,
                      roster_by_id: dict, roster_id: str, record_of: Callable,
                      points_total: Callable, team_meta: Callable) -> dict:
    """League-context aggregates for the schedule tab, derived only from the
    provider rows and roster settings already in hand: current standings
    seeds, recent form, remaining strength of schedule, and a spotlight on
    the next unplayed matchup. Nothing here invents a score or a result;
    every number traces to a published provider row or a roster total."""
    settings = (league or {}).get("settings") or {}

    def _setting_int(key):
        try:
            value = int(settings.get(key) or 0)
            return value if value > 0 else None
        except (TypeError, ValueError):
            return None

    # Current standings seeds: wins first, then points for (the ordering the
    # standings surfaces use). Injected into each week's team metas so rows
    # can show where the opponent currently sits.
    def _pf(rid: str) -> float:
        return points_total(rid, "fpts", "fpts_decimal") or 0.0

    seeded = sorted(roster_by_id.keys(),
                    key=lambda rid: (-(record_of(rid).get("wins") or 0), -_pf(rid)))
    seeds = {rid: i + 1 for i, rid in enumerate(seeded)}
    for item in weeks:
        if not item.get("published"):
            continue
        for side in ("team", "opponent"):
            meta = item.get(side)
            if meta and str(meta.get("roster_id")) in seeds:
                meta["seed"] = seeds[str(meta["roster_id"])]

    published = [w for w in weeks if w.get("published")]
    finals = [w for w in published if w.get("state") == "final" and w.get("result")]
    final_weeks = {w["week"] for w in finals}

    # Per-team results across finalized weeks (for form lines). Pairing comes
    # from the same provider rows the schedule itself was built from.
    results_by_team: dict[str, list[dict]] = {}
    for week in sorted(final_weeks):
        rows = all_rows_by_week.get(week) or []
        by_mid: dict[str, list[dict]] = {}
        for row in rows:
            mid = row.get("matchup_id")
            if mid is None:
                continue
            by_mid.setdefault(str(mid), []).append(row)
        for pair in by_mid.values():
            if len(pair) != 2:
                continue
            a, b = pair
            pa, pb = _number(a.get("points")), _number(b.get("points"))
            if pa is None or pb is None:
                continue
            for mine, theirs, pts, opp_pts in ((a, b, pa, pb), (b, a, pb, pa)):
                rid = str(mine.get("roster_id"))
                results_by_team.setdefault(rid, []).append(
                    {"week": week, "points": pts, "opp_points": opp_pts,
                     "result": "T" if pts == opp_pts else ("W" if pts > opp_pts else "L")})

    def _form(rid: str, limit: int = 5) -> list[str]:
        rows = results_by_team.get(str(rid)) or []
        return [r["result"] for r in rows[-limit:]]

    form: dict[str, Any] = {"results": [], "avg_margin": None}
    if finals:
        recent = finals[-5:]
        form["results"] = [w["result"] for w in recent]
        margins = [(w["team_points"] or 0) - (w["opponent_points"] or 0)
                   for w in recent if w.get("team_points") is not None and w.get("opponent_points") is not None]
        if margins:
            form["avg_margin"] = round(sum(margins) / len(margins), 1)
    best = worst = None
    scored = [w for w in finals if w.get("team_points") is not None]
    if scored:
        b = max(scored, key=lambda w: w["team_points"])
        w_ = min(scored, key=lambda w: w["team_points"])
        best = {"week": b["week"], "points": b["team_points"]}
        worst = {"week": w_["week"], "points": w_["team_points"]}

    # Remaining strength of schedule: for every team, the average current
    # win% of its opponents in unplayed weeks; ranked hardest first.
    remaining_weeks = [w["week"] for w in published
                       if w.get("state") in {"scheduled", "live"} and w.get("opponent")]
    sos_rank = None
    sos_pct = None
    if remaining_weeks:
        def _remaining_opps(rid: str) -> list[str]:
            opps = []
            for week in remaining_weeks:
                rows = all_rows_by_week.get(week) or []
                mine = next((r for r in rows if str(r.get("roster_id")) == str(rid)), None)
                if not mine or mine.get("matchup_id") is None:
                    continue
                opp = next((r for r in rows if r is not mine
                            and str(r.get("matchup_id")) == str(mine.get("matchup_id"))), None)
                if opp is not None:
                    opps.append(str(opp.get("roster_id")))
            return opps

        def _sos(rid: str):
            pcts = [p for p in (_win_pct(record_of(o)) for o in _remaining_opps(rid)) if p is not None]
            return sum(pcts) / len(pcts) if pcts else None

        mine_sos = _sos(roster_id)
        if mine_sos is not None:
            sos_pct = round(mine_sos, 3)
            sos_rank = 1 + sum(1 for rid in roster_by_id
                               if rid != roster_id and (_sos(rid) or 0) > mine_sos)

    # Spotlight: the next unplayed matchup, both teams profiled from the
    # same aggregates (record, scoring rates over finalized games, form).
    spotlight = None
    upcoming = [w for w in published if w.get("state") in {"scheduled", "live"} and w.get("opponent")]
    if upcoming:
        nxt = upcoming[0]

        def _profile(meta: dict) -> dict:
            rid = str(meta.get("roster_id"))
            played = len(results_by_team.get(rid) or [])
            pf = points_total(rid, "fpts", "fpts_decimal")
            pa = points_total(rid, "fpts_against", "fpts_against_decimal")
            return {**meta,
                    "pf_per_game": round(pf / played, 1) if pf is not None and played else None,
                    "pa_per_game": round(pa / played, 1) if pa is not None and played else None,
                    "form": _form(rid)}

        spotlight = {"week": nxt["week"], "state": nxt["state"],
                     "team": _profile(nxt["team"]), "opponent": _profile(nxt["opponent"])}

    return {"playoff_week_start": _setting_int("playoff_week_start"),
            "playoff_teams": _setting_int("playoff_teams"),
            "seeds": seeds, "form": form, "best_week": best, "worst_week": worst,
            "remaining_sos": {"rank": sos_rank, "of": len(roster_by_id), "opp_win_pct": sos_pct},
            "spotlight": spotlight}
