"""Remaining-schedule race path (Frontier 2).

The GM brief's race line (Wave 3) is a snapshot: seed, record, games
back, PF delta. It cannot answer the forward questions a manager
actually asks in November: who do I still play, who does the team on
the cut line still play, how many wins make me safe? This module is
that arithmetic, on the TRUE field (the league-rules record's
playoff size, never a provider default).

Inputs are plain data (the brief's standings rows and the ctx
matchup slate), so the math is testable without a league context:

* ``remaining_schedules`` reads ``matchups_by_week`` in the shape
  the brief already consumes (``{week: [{left: {roster_id},
  right: {roster_id}}]}``).
* Clinch / elimination status reuses the settled safe conditions in
  ``utils.standings.compute_playoff_picture`` (wins-based, PF as
  seeding tiebreak; a team is never wrongly told it is in or out).
* The magic number uses the same conservative rule as the picture's
  clinch test: wins needed so that fewer than ``playoff_teams``
  rivals can even REACH the viewer's win floor. PF tiebreaks can
  still move a spot either way, and the payload says so.

Fail closed: no playoff size, no standings, or a season already
past the regular-season end yields None, and a missing future
slate yields a path with ``schedule_known`` False (records math
only) rather than invented opponents.
"""
from __future__ import annotations

from typing import Any, Optional


def remaining_schedules(matchups_by_week: Any, from_week: int) -> dict:
    """{roster_id: [{week, opp_id}]} for weeks >= ``from_week``."""
    out: dict = {}
    if not isinstance(matchups_by_week, dict):
        return out
    for raw_week, slate in matchups_by_week.items():
        try:
            week = int(raw_week)
        except (TypeError, ValueError):
            continue
        if week < from_week:
            continue
        for matchup in slate or []:
            if not isinstance(matchup, dict):
                continue
            sides = [matchup.get("left") or {}, matchup.get("right") or {}]
            rids = [str(s.get("roster_id")) for s in sides
                    if isinstance(s, dict)
                    and s.get("roster_id") is not None]
            if len(rids) != 2 or rids[0] == rids[1]:
                continue
            for rid, opp in ((rids[0], rids[1]), (rids[1], rids[0])):
                out.setdefault(rid, []).append(
                    {"week": week, "opp_id": opp})
    for games in out.values():
        games.sort(key=lambda g: g["week"])
    return out


def _win_value(row: dict) -> float:
    return float(row.get("wins") or 0) + 0.5 * float(row.get("ties") or 0)


def magic_number(viewer: dict, rows: list, playoff_teams: int,
                 games_left: int) -> Optional[int]:
    """Wins the viewer still needs so fewer than ``playoff_teams``
    rivals can reach his win floor (the picture's safe clinch rule).

    None when even winning out does not get there yet. 0 when the
    berth is already safe on wins alone.
    """
    vid = str(viewer.get("roster_id"))
    rivals = [r for r in rows if str(r.get("roster_id")) != vid]
    for k in range(0, max(0, games_left) + 1):
        floor = float(viewer.get("wins") or 0) + k
        threats = sum(
            1 for r in rivals
            if _win_value(r) + _games_left(r, games_left) >= floor)
        if threats < playoff_teams:
            return k
    return None


def _games_left(row: dict, default: int) -> int:
    """Games left for a row: the explicit per-team count the caller
    stamped from the real slate (``row['_games_left']``), else the
    viewer's count (every team normally has the same number left).
    """
    explicit = row.get("_games_left")
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    return default


def _remaining_block(row: dict, games: list, by_id: dict) -> dict:
    opps = []
    win_pcts = []
    for game in games:
        opp = by_id.get(game["opp_id"])
        if opp is None:
            continue
        played = (float(opp.get("wins") or 0) + float(opp.get("losses") or 0)
                  + float(opp.get("ties") or 0))
        pct = (_win_value(opp) / played) if played else None
        if pct is not None:
            win_pcts.append(pct)
        opps.append({
            "week": game["week"],
            "opp_id": game["opp_id"],
            "opp_name": opp.get("team_name"),
            "opp_wins": opp.get("wins"),
            "opp_losses": opp.get("losses"),
        })
    return {
        "remaining": opps,
        "remaining_sos": (round(sum(win_pcts) / len(win_pcts), 3)
                          if win_pcts else None),
    }


def build_race_path(rows: list, matchups_by_week: Any, *,
                    current_week: int, playoff_teams: int,
                    regular_season_end: Optional[int],
                    viewer_roster_id) -> Optional[dict]:
    """The forward-looking race payload, or None (fail closed)."""
    if not playoff_teams or not rows:
        return None
    by_id = {str(r.get("roster_id")): r for r in rows}
    viewer = by_id.get(str(viewer_roster_id))
    if viewer is None:
        return None
    played = (float(viewer.get("wins") or 0)
              + float(viewer.get("losses") or 0)
              + float(viewer.get("ties") or 0))
    schedules = remaining_schedules(matchups_by_week, int(current_week))
    schedule_known = bool(schedules)
    if regular_season_end:
        games_left = max(0, int(regular_season_end) - int(played))
    elif schedule_known:
        games_left = len(schedules.get(str(viewer_roster_id)) or [])
    else:
        return None
    if games_left <= 0:
        return None
    # Games left is record math (season end minus games played) for
    # every team: the slate names the opponents but may only be
    # partially loaded, so its length never shrinks a team's ceiling.
    work_rows = []
    for row in rows:
        clone = dict(row)
        if regular_season_end:
            row_played = (float(row.get("wins") or 0)
                          + float(row.get("losses") or 0)
                          + float(row.get("ties") or 0))
            clone["_games_left"] = max(
                0, int(regular_season_end) - int(row_played))
        else:
            clone["_games_left"] = len(
                schedules.get(str(row.get("roster_id"))) or [])
        work_rows.append(clone)
    by_id_work = {str(r.get("roster_id")): r for r in work_rows}

    picture = None
    try:
        from utils.standings import compute_playoff_picture

        teams = [{
            "id": r.get("roster_id"), "name": r.get("team_name"),
            "wins": int(float(r.get("wins") or 0)),
            "losses": int(float(r.get("losses") or 0)),
            "ties": int(float(r.get("ties") or 0)),
            "pf": float(r.get("pf") or 0),
        } for r in rows]
        pic = compute_playoff_picture(
            teams, int(playoff_teams),
            int(regular_season_end) if regular_season_end
            else int(played) + games_left)
        picture = {str(t.get("id")): t for t in pic}
    except Exception:
        picture = None

    def _block(row):
        games = schedules.get(str(row.get("roster_id"))) or []
        base = _remaining_block(row, games, by_id) if schedule_known \
            else {"remaining": [], "remaining_sos": None}
        pic = (picture or {}).get(str(row.get("roster_id"))) or {}
        return {
            "roster_id": str(row.get("roster_id")),
            "team_name": row.get("team_name"),
            "seed": row.get("seed"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "games_left": by_id_work[str(row.get("roster_id"))][
                "_games_left"],
            "max_wins": (float(row.get("wins") or 0)
                         + by_id_work[str(row.get("roster_id"))][
                             "_games_left"]),
            "status": pic.get("status"),
            **base,
        }

    viewer_pic = (picture or {}).get(str(viewer_roster_id)) or {}
    viewer_block = _block(viewer)
    viewer_block.update({
        "controls_own_fate": viewer_pic.get("controls_own_fate"),
        "scenario": viewer_pic.get("scenario"),
        "magic_number": magic_number(
            viewer, work_rows, int(playoff_teams), games_left),
    })
    # The cut-line window: two seeds either side of the line.
    line = int(playoff_teams)
    window = [r for r in rows
              if isinstance(r.get("seed"), (int, float))
              and line - 2 <= r["seed"] <= line + 2
              and str(r.get("roster_id")) != str(viewer_roster_id)]
    return {
        "viewer": viewer_block,
        "cut_line": [_block(r) for r in window],
        "playoff_teams": int(playoff_teams),
        "games_left": games_left,
        "schedule_known": schedule_known,
        "tiebreak_note": ("Wins-only math: points-for tiebreaks can "
                          "still move a spot either way."),
    }
