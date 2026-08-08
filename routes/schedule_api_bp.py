"""Extracted from app.py — schedule_api_bp (see route list below)."""
from __future__ import annotations
import logging
import os
import json
from datetime import datetime
from flask import Blueprint, jsonify, request
from utils.utils import CACHE_DIR
logger = logging.getLogger(__name__)

schedule_api_bp = Blueprint("schedule_api_bp", __name__)

_SCHED_POS_COLORS = {"QB": "#3b82f6", "RB": "#22c55e", "WR": "#f59e0b",
                     "TE": "#a855f7", "K": "#6b7280", "DEF": "#6b7280"}


# ── Lazy shims to app.py internals (resolved at request time) ──
def _api_err(*a, **k):
    from app import _api_err as _fn
    return _fn(*a, **k)

def _build_roster_map(*a, **k):
    from app import _build_roster_map as _fn
    return _fn(*a, **k)

def _compute_schedule_grid(*a, **k):
    from app import _compute_schedule_grid as _fn
    return _fn(*a, **k)

def _matchup_cell_ease(*a, **k):
    from app import _matchup_cell_ease as _fn
    return _fn(*a, **k)

def _matchup_rank_table(*a, **k):
    from app import _matchup_rank_table as _fn
    return _fn(*a, **k)

def _norm_sched_team(*a, **k):
    from app import _norm_sched_team as _fn
    return _fn(*a, **k)

def _sched_rank_color(*a, **k):
    from app import _sched_rank_color as _fn
    return _fn(*a, **k)

def get_league_ctx_from_cache(*a, **k):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*a, **k)

def get_model_value_table_cached(*a, **k):
    from app import get_model_value_table_cached as _fn
    return _fn(*a, **k)

def get_nfl_state(*a, **k):
    from app import get_nfl_state as _fn
    return _fn(*a, **k)

def get_players_index_global(*a, **k):
    from app import get_players_index_global as _fn
    return _fn(*a, **k)

def get_rosters(*a, **k):
    from app import get_rosters as _fn
    return _fn(*a, **k)

def get_users(*a, **k):
    from app import get_users as _fn
    return _fn(*a, **k)

def load_pick_value_table(*a, **k):
    from app import load_pick_value_table as _fn
    return _fn(*a, **k)


@schedule_api_bp.route("/api/schedule")
def api_schedule():
    try:
        season = int(request.args.get("season") or 0)
        pids = [p for p in (request.args.get("pids") or "").split(",") if p]
        ws = int(request.args.get("week_start") or 1)
        we = int(request.args.get("week_end") or ws)
        if we < ws:
            ws, we = we, ws
        ws = max(1, min(ws, 18))
        we = max(1, min(we, 18))
        weeks = list(range(ws, we + 1))
        if not pids:
            return jsonify({"weeks": weeks, "players": []})
        players = _compute_schedule_grid(season, pids, weeks)
        return jsonify({"weeks": weeks, "players": players})
    except Exception as e:
        return _api_err("Schedule unavailable", e)


@schedule_api_bp.route("/api/schedule-rankings")
def api_schedule_rankings():
    try:
        import glob as _glob
        season    = int(request.args.get("season") or datetime.now().year)
        ws        = int(request.args.get("week_start") or 1)
        we        = int(request.args.get("week_end") or ws)
        position  = (request.args.get("position") or "RB").upper().strip()
        league_id = (request.args.get("league_id") or "").strip()
        platform  = (request.args.get("platform") or "sleeper").strip()

        ws = max(1, min(ws, 18))
        we = max(ws, min(we, 18))
        weeks = list(range(ws, we + 1))
        if position not in {"QB", "RB", "WR", "TE", "K"}:
            position = "RB"

        players_idx  = get_players_index_global() or {}

        # Build schedule lookup for requested weeks
        schedules = {}
        for w in weeks:
            try:
                from utils.utils import load_week_schedule as _lws
                games = _lws(season, w) or []
            except Exception:
                games = []
            lookup = {}
            for g in games:
                if not isinstance(g, dict): continue
                home = _norm_sched_team(g.get("home"))
                away = _norm_sched_team(g.get("away"))
                if home:
                    lookup[home] = {"opp": away, "is_home": True}
                if away:
                    lookup[away] = {"opp": home, "is_home": False}
            schedules[w] = lookup

        # Rank teams by strength-of-schedule-adjusted z-score (rank 1 = easiest).
        # Falls back to raw fpts-allowed until the cron builds the ratings table.
        rank_map, total_teams, rating_info, _is_z = _matchup_rank_table(season, position)

        # Get roster pids (+ owning team name) for the on-roster badge
        roster_pids: set = set()
        owner_by_pid: dict = {}
        if league_id:
            try:
                ctx = get_league_ctx_from_cache(platform, league_id, season)
                users_by_id = {str(u.get("user_id")): u for u in (ctx.get("users") or [])}
                for r in (ctx.get("rosters") or []):
                    owner = users_by_id.get(str(r.get("owner_id"))) or {}
                    meta = owner.get("metadata") or {}
                    tname = meta.get("team_name") or owner.get("display_name") or owner.get("username")
                    for pid in (r.get("players") or []):
                        roster_pids.add(str(pid))
                        if tname:
                            owner_by_pid[str(pid)] = tname
            except Exception:
                logger.debug("suppressed exception", exc_info=True)

        # Build value lookup once - used for depth-chart cap and final sort
        value_by_pid: dict = {}
        for _p in (get_model_value_table_cached() or []):
            _pid = str(_p.get("id") or "")
            if _pid:
                value_by_pid[_pid] = max(
                    float(_p.get("value") or 0),
                    float(_p.get("value_1qb") or 0),
                    float(_p.get("value_sf") or 0),
                )

        # Depth-chart cap: keep only the top N at this position per NFL team
        # (ranked by model value), plus any rostered players. Removes the long
        # tail of backups that share an identical schedule row.
        _POS_CAP = {"QB": 2, "RB": 3, "WR": 3, "TE": 2}
        cap = _POS_CAP.get(position)
        keep_pids: set = set(roster_pids)
        if cap:
            from collections import defaultdict as _dd_team
            by_team: dict = _dd_team(list)
            for _pid, _info in players_idx.items():
                if (_info.get("pos") or "").upper() != position:
                    continue
                _team = _norm_sched_team(_info.get("team"))
                if not _team or _team == "FA":
                    continue
                by_team[_team].append((str(_pid), value_by_pid.get(str(_pid), 0.0)))
            for _team, _lst in by_team.items():
                _lst.sort(key=lambda x: -x[1])
                for _pid, _ in _lst[:cap]:
                    keep_pids.add(_pid)

        # Load per-player actual points (completed weeks) and projections (future weeks)
        player_pts_actual: dict = {}
        weeks_with_stats: set  = set()
        for w in weeks:
            sfiles = _glob.glob(os.path.join(CACHE_DIR, "sleeper_stats", f"sleeper_stats_s{season}_w{w}*.json"))
            if not sfiles:
                continue
            weeks_with_stats.add(w)
            try:
                sd = json.load(open(sfiles[0]))
                if isinstance(sd, dict):
                    for _pid, _stats in sd.items():
                        _pts = float(_stats.get("pts_ppr") or 0)
                        if _pts > 0:
                            player_pts_actual.setdefault(str(_pid), {})[w] = round(_pts, 1)
            except Exception:
                logger.debug("suppressed exception", exc_info=True)

        player_pts_proj: dict = {}
        try:
            from utils.utils import load_week_projection as _lp
            for w in weeks:
                if w in weeks_with_stats:
                    continue
                _proj = _lp(season, w) or {}
                for _pid, _val in _proj.items():
                    _v = float(_val or 0)
                    if _v > 0:
                        player_pts_proj.setdefault(str(_pid), {})[w] = round(_v, 1)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)

        results = []
        for pid, info in players_idx.items():
            pos  = (info.get("pos") or "").upper()
            if pos != position:
                continue
            team = (info.get("team") or "").upper()
            if not team or team == "FA":
                continue
            if cap and str(pid) not in keep_pids:
                continue

            cells = []
            rank_sum  = 0
            ease_sum  = 0.0
            valid_wks = 0
            for w in weeks:
                game = schedules.get(w, {}).get(team)
                if not game:
                    cells.append({"week": w, "bye": True})
                    continue
                opp      = game["opp"]
                rank     = rank_map.get(opp)
                rinfo    = rating_info.get(opp, {})
                fpts_val = rinfo.get("fpts", 0)
                txt, bg  = _sched_rank_color(rank, total_teams) if rank else ("#94a3b8", "transparent")
                actual   = player_pts_actual.get(str(pid), {}).get(w)
                proj     = player_pts_proj.get(str(pid), {}).get(w)
                p_pts    = actual if actual is not None else proj
                p_type   = "actual" if actual is not None else ("proj" if proj is not None else None)
                cells.append({
                    "week": w, "bye": False,
                    "opp": opp,
                    "at": "" if game["is_home"] else "@",
                    "rank": rank, "total": total_teams,
                    "fpts": round(fpts_val, 1) if fpts_val else 0,
                    "txt": txt, "bg": bg,
                    "pts": p_pts, "pts_type": p_type,
                })
                if rank:
                    rank_sum  += rank
                    ease_sum  += _matchup_cell_ease(rank, total_teams, rinfo)
                    valid_wks += 1

            avg_rank   = round(rank_sum / valid_wks, 1) if valid_wks else 999
            # Ease from the z-score scale (avg over scheduled weeks); higher = easier
            ease_score = round(ease_sum / valid_wks, 1) if valid_wks else 0

            results.append({
                "pid":        str(pid),
                "name":       info.get("name") or str(pid),
                "pos":        pos,
                "color":      _SCHED_POS_COLORS.get(pos, "#6b7280"),
                "team":       team,
                "value":      value_by_pid.get(str(pid), 0.0),
                "on_roster":  str(pid) in roster_pids,
                "owner":      owner_by_pid.get(str(pid)),
                "cells":      cells,
                "avg_rank":   avg_rank,
                "ease_score": ease_score,
                "valid_weeks": valid_wks,
            })

        # Sort by ease first, then group teammates together under the best player
        # on that team.  Rank #1 = the most valuable player with the easiest
        # schedule; teammates follow immediately after.
        _team_max_ease: dict = {}
        _team_max_val:  dict = {}
        for r in results:
            t = r["team"]
            if r["ease_score"] > _team_max_ease.get(t, -1):
                _team_max_ease[t] = r["ease_score"]
            if r["value"] > _team_max_val.get(t, -1):
                _team_max_val[t] = r["value"]
        results.sort(key=lambda x: (
            -_team_max_ease.get(x["team"], 0.0),  # teams with easiest schedule first
            x["team"],                             # tie-break: keep same team together
            -x["value"],                           # within team: most valuable first
        ))
        return jsonify({
            "weeks":       weeks,
            "position":    position,
            "total_teams": total_teams,
            "rankings":    results,
        })
    except Exception as e:
        return _api_err("Schedule rankings unavailable", e)


@schedule_api_bp.route("/api/schedule-strength")
def api_schedule_strength():
    """
    Compute schedule strength remaining for each team in a league.

    For each team's future matchups (weeks > current_week), look up their
    opponent's average points scored this season. The team with the hardest
    remaining schedule faces the highest-scoring opponents on average.

    Query params: platform, league_id, season
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"error": "league_id required"}), 400

    nfl_state = get_nfl_state() or {}
    try:
        season = int(request.args.get("season") or nfl_state.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year

    current_week = int(nfl_state.get("leg") or nfl_state.get("week") or 0)
    season_type = str(nfl_state.get("season_type") or "").lower()
    FULL_SEASON_WEEKS = 17  # safe scan cap; loop breaks on empty weeks

    try:
        from dashboard_services.platform_api import get_matchups as pf_get_matchups

        rosters = get_rosters(platform, league_id, season) or []
        users = get_users(platform, league_id, season) or []
        roster_map = _build_roster_map(users, rosters)

        # Build per-roster average points from completed weeks
        avg_pts_by_rid: dict[str, float] = {}
        weekly_pts: dict[str, list] = {str(r.get("roster_id")): [] for r in rosters}

        for w in range(1, current_week + 1):
            try:
                week_data = pf_get_matchups(platform, league_id, w, season) or []
            except Exception:
                continue
            for m in week_data:
                rid = str(m.get("roster_id", ""))
                pts = float(m.get("points") or 0.0)
                if rid in weekly_pts:
                    weekly_pts[rid].append(pts)

        for rid, pts_list in weekly_pts.items():
            avg_pts_by_rid[rid] = round(sum(pts_list) / len(pts_list), 2) if pts_list else 0.0

        # When no games have been played, fall back to power rankings (roster value) as proxy
        games_played = sum(1 for pts in avg_pts_by_rid.values() if pts > 0)
        if games_played == 0:
            try:
                ctx = get_league_ctx_from_cache(platform, league_id, season)
                model_vals = ctx.get("model_value_table") or []
                picks_by_roster = ctx.get("picks_by_roster") or {}
                values_by_id = {str(p["id"]): float(p.get("value") or 0) for p in model_vals if p.get("id")}
                pick_values = load_pick_value_table() or {}
                standings_map = ctx.get("standings_map") or {}
                for r in rosters:
                    rid = str(r.get("roster_id", ""))
                    player_ids = [str(pid) for pid in (r.get("players") or [])]
                    roster_val = sum(values_by_id.get(pid, 0.0) for pid in player_ids)
                    # Normalize to a "projected points" scale (~100-160 range) for display consistency
                    avg_pts_by_rid[rid] = round(100.0 + roster_val / 50.0, 2)
            except Exception:
                logger.debug("suppressed exception", exc_info=True)

        # Build future matchups map: rid -> list of opponent roster_ids
        future_opponents: dict[str, list] = {str(r.get("roster_id")): [] for r in rosters}

        for w in range(current_week + 1, FULL_SEASON_WEEKS + 1):
            try:
                week_data = pf_get_matchups(platform, league_id, w, season) or []
            except Exception:
                continue
            if not week_data:
                break
            # Group by matchup_id
            by_mid: dict = {}
            for m in week_data:
                mid = m.get("matchup_id")
                if mid is None:
                    continue
                by_mid.setdefault(mid, []).append(str(m.get("roster_id", "")))
            for mid, rids in by_mid.items():
                if len(rids) == 2:
                    future_opponents[rids[0]].append(rids[1])
                    future_opponents[rids[1]].append(rids[0])

        results = []
        for r in rosters:
            rid = str(r.get("roster_id", ""))
            opp_rids = future_opponents.get(rid, [])
            opp_avgs = [avg_pts_by_rid.get(o, 0.0) for o in opp_rids]
            avg_opp = round(sum(opp_avgs) / len(opp_avgs), 2) if opp_avgs else 0.0
            results.append({
                "roster_id": rid,
                "team_name": roster_map.get(rid, f"Roster {rid}"),
                "games_remaining": len(opp_rids),
                "avg_opp_points": avg_opp,
                "my_avg_points": avg_pts_by_rid.get(rid, 0.0),
            })

        results.sort(key=lambda x: x["avg_opp_points"], reverse=True)

        return jsonify({
            "current_week": current_week,
            "weeks_remaining": max(len(v) for v in future_opponents.values()) if future_opponents else 0,
            "teams": results,
            "using_power_rankings": games_played == 0,
        })

    except Exception:
        logger.exception("[schedule-strength] Unexpected error")
        return jsonify({"error": "Internal error"}), 500
