"""Shared D/ST + K streaming rankers.

Single home for the matchup-based streaming logic behind /api/streaming-options
and the K / D/ST tabs of /api/waiver-candidates. Rankings:

  * defenses: free-agent D/STs sorted by how weak the offense they face is
    (opponent Vegas implied total, ascending);
  * kickers: free-agent Ks on teams playing this week, one per team, sorted by
    their own team's Vegas implied total (descending).

Everything is pure with respect to the passed league context; schedule and
Vegas data flow through the same cached loaders the old app.py endpoint used.
Never raises: missing schedule/Vegas data degrades to empty lists so callers
fall back gracefully (e.g. offseason, or a league that starts no K/DST).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# FAAB-scale composite attached to each row (``stream_score``), so waiver-candidate
# K/DST rows can size FAAB bids on the same absolute scale as skill-position rows
# (FAAB_SCORE_LOW=45 .. FAAB_SCORE_HIGH=190 in utils.waiver_score). A genuinely
# elite streamer (bottom-3 opposing offense / top-3 team total) lands well above
# the waiver-floor score; a no-Vegas-data row sits mid-pack.
_STREAM_SCORE_FLOOR = 50.0
_STREAM_SCORE_CAP = 155.0
_STREAM_SCORE_NODATA = 75.0


def stream_score(implied, *, lower_is_better: bool) -> float:
    """Map a Vegas implied total onto the waiver composite scale.

    ``lower_is_better`` for defenses (a low *opponent* total is good), False for
    kickers (a high *own* total is good). Missing data returns a mid-pack score
    rather than zero so an unpriced game doesn't bury the row.
    """
    try:
        v = float(implied)
    except (TypeError, ValueError):
        return _STREAM_SCORE_NODATA
    if lower_is_better:
        # Opp implied 26 -> replacement-level streamer, 14 -> elite.
        s = _STREAM_SCORE_FLOOR + (26.0 - v) * 8.0
    else:
        # Own implied 17 -> replacement-level streamer, 30 -> elite.
        s = _STREAM_SCORE_FLOOR + (v - 17.0) * 8.0
    return round(max(_STREAM_SCORE_FLOOR - 10.0, min(_STREAM_SCORE_CAP, s)), 1)


def streaming_targets(ctx, season, current_week=None, players_index=None, limit=8):
    """Ranked free-agent defenses and kickers for the current week.

    ``ctx`` is the league context (rosters, roster_positions, ...),
    ``players_index`` maps player ids to {pos, team, name} (falls back to the
    ctx's own index). Returns ``{"defense": [...], "kicker": [...],
    "in_season": bool, "uses_def": bool, "uses_k": bool}``. Defense rows carry
    ``opp_implied``; kicker rows carry ``own_implied``; every row carries
    ``stream_score``.
    """
    try:
        return _streaming_targets(ctx, season, current_week, players_index, limit)
    except Exception:
        logger.debug("streaming_targets failed", exc_info=True)
        return {"defense": [], "kicker": [], "in_season": False,
                "uses_def": False, "uses_k": False}


def _streaming_targets(ctx, season, current_week, players_index, limit):
    ctx = ctx or {}
    if current_week is None:
        current_week = int(ctx.get("current_week") or 0)
    if current_week < 1 or ctx.get("offseason_mode"):
        return {"defense": [], "kicker": [], "in_season": False,
                "uses_def": False, "uses_k": False}

    # League's started positions gate which streamers are relevant at all.
    rpos = [str(s).upper() for s in (ctx.get("roster_positions") or [])]
    uses_def = any(s in ("DEF", "DST", "D/ST") for s in rpos)
    uses_k = "K" in rpos
    if not (uses_def or uses_k):
        return {"defense": [], "kicker": [], "in_season": True,
                "uses_def": False, "uses_k": False}

    # ── Schedule → opponent map + games for the Vegas lookup ──────────────────
    opponent_map: dict = {}
    week_games: list = []
    teams: set = set()
    try:
        from utils.utils import load_week_sched
        for g in (load_week_sched(season, current_week) or []):
            home = str(g.get("home") or "").upper()
            away = str(g.get("away") or "").upper()
            if home and away:
                opponent_map[home] = away
                opponent_map[away] = home
                week_games.append((home, away, str(g.get("gameDate") or "")))
                teams.add(home)
                teams.add(away)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    if not teams:
        return {"defense": [], "kicker": [], "in_season": True,
                "uses_def": uses_def, "uses_k": uses_k}

    conditions: dict = {}
    try:
        from utils.game_conditions import build_week_conditions
        conditions = build_week_conditions(season, current_week, week_games) or {}
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    def _implied(team):
        v = (conditions.get(team) or {}).get("implied_total")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _matchup(team):
        opp = opponent_map.get(team)
        if not opp:
            return "", None
        # home_team_of not tracked here; label as "vs OPP" (venue is secondary).
        return f"vs {opp}", opp

    rostered = {
        str(pid)
        for r in (ctx.get("rosters") or [])
        for pid in (r.get("players") or [])
    }
    players_index = players_index if players_index is not None else (ctx.get("players_index") or {})

    # ── Defenses: one per team, pid == team abbr in Sleeper. Best matchup is the
    # weakest opposing offense (lowest opponent implied total). ────────────────
    defense = []
    if uses_def:
        rows = []
        for t in teams:
            if t in rostered:
                continue
            label, opp = _matchup(t)
            opp_imp = _implied(opp)
            rows.append({
                "player_id": t, "name": f"{t} D/ST", "position": "DEF", "team": t,
                "opponent": opp, "matchup": label, "opp_implied": opp_imp,
                "stream_score": stream_score(opp_imp, lower_is_better=True),
            })
        rows.sort(key=lambda d: (d["opp_implied"] is None,
                                 d["opp_implied"] if d["opp_implied"] is not None else 99.0))
        defense = rows[:limit]

    # ── Kickers: free-agent Ks on teams playing this week, ranked by their own
    # implied total (more team scoring → more FGs/XPs). One per team. ──────────
    kicker = []
    if uses_k:
        cand = []
        for pid, meta in (players_index or {}).items():
            if str((meta or {}).get("pos") or "").upper() != "K":
                continue
            pid = str(pid)
            t = str((meta or {}).get("team") or "").upper()
            if not t or t not in teams or pid in rostered:
                continue
            cand.append((pid, (meta or {}).get("name") or f"Player {pid}", t, _implied(t)))
        cand.sort(key=lambda x: (x[3] is None, -(x[3] if x[3] is not None else 0.0)))
        seen_team = set()
        for pid, name, t, imp in cand:
            if t in seen_team:
                continue
            seen_team.add(t)
            label, opp = _matchup(t)
            kicker.append({
                "player_id": pid, "name": name, "position": "K", "team": t,
                "opponent": opp, "matchup": label, "own_implied": imp,
                "stream_score": stream_score(imp, lower_is_better=False),
            })
            if len(kicker) >= limit:
                break

    return {"defense": defense, "kicker": kicker, "in_season": True,
            "uses_def": uses_def, "uses_k": uses_k}
