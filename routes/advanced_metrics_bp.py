"""Advanced-metrics API (leaderboard, weekly-bulk, config, top role players,
breakout candidates). Extracted from app.py to shrink the monolith.

App.py internals are reached via the lazy shims below so importing this blueprint
at start-up stays free of a circular import."""
from __future__ import annotations

import logging
import time

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

advanced_metrics_bp = Blueprint("advanced_metrics_bp", __name__)

# Module-local caches (moved verbatim from app.py — used only by these handlers).
_ADVANCED_METRICS_TTL = 600
_ROLE_PLAYERS_CACHE: dict = {}
_ROLE_PLAYERS_CACHE_TS: dict = {}
_BREAKOUT_CACHE = None
_BREAKOUT_CACHE_TS = 0.0


# ── Lazy shims to app.py internals (resolved at request time) ─────────────────

def get_league_ctx_from_cache(*a, **k):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*a, **k)


def get_players_global(*a, **k):
    from app import get_players_global as _fn
    return _fn(*a, **k)


@advanced_metrics_bp.route("/api/advanced-metrics/leaderboard")
def api_advanced_metrics_leaderboard():
    """Players ranked by a single advanced metric, for the Advanced Metrics page.

    Premium-gated. Query params: metric (required, whitelisted), position (optional),
    league_id/platform (for the premium check).
    """
    from data_building.advanced_metrics import (
        get_metric_leaderboard, get_weekly_range_leaderboard,
        get_adv_weekly_range_leaderboard, adv_weekly_metric_supported,
        get_value_leaderboard, VALUE_METRICS,
        LEADERBOARD_METRICS, _WEEKLY_METRICS,
        PREMIUM_METRICS, premium_metrics_exposed,
    )

    metric = (request.args.get("metric") or "role_score").strip()
    if metric not in LEADERBOARD_METRICS:
        return jsonify({"error": "unknown metric"}), 400
    # Premium (PFF) metrics are not displayable publicly.
    if metric in PREMIUM_METRICS and not premium_metrics_exposed():
        return jsonify({"error": "metric not available"}), 403
    position = (request.args.get("position") or "").strip().upper() or None
    season_str = (request.args.get("season") or "").strip()
    season = int(season_str) if season_str.isdigit() else None
    min_vol_str = (request.args.get("min_vol") or "").strip()
    min_vol = int(min_vol_str) if min_vol_str.isdigit() else None
    week_start_str = (request.args.get("week_start") or "").strip()
    week_end_str   = (request.args.get("week_end") or "").strip()
    week_start = int(week_start_str) if week_start_str.isdigit() else None
    week_end   = int(week_end_str)   if week_end_str.isdigit()   else None

    # A metric is week-filterable if it has a usage-table aggregation
    # (_WEEKLY_METRICS) or an NGS/FTN/EPA weekly aggregation.
    adv_weekly       = adv_weekly_metric_supported(metric)
    weekly_capable   = (metric in _WEEKLY_METRICS) or adv_weekly
    is_week_filtered = bool(week_start or week_end) and weekly_capable

    try:
        if metric in VALUE_METRICS:
            # Value metrics (VORP/WAR) are league-size aware; derive num_teams
            # from the league context when available, else standard 12-team.
            _num_teams = 12
            try:
                _league_id = (request.args.get("league_id") or "").strip()
                _platform = (request.args.get("platform") or "sleeper").strip()
                if _league_id:
                    _ctx = get_league_ctx_from_cache(_platform, _league_id, season) or {}
                    _num_teams = int(_ctx.get("total_rosters") or 0) or 12
            except Exception:
                _num_teams = 12
            players = get_value_leaderboard(
                metric, position=position, season=season, num_teams=_num_teams,
            )
        elif is_week_filtered and adv_weekly and metric not in _WEEKLY_METRICS:
            players = get_adv_weekly_range_leaderboard(
                metric, position=position, season=season,
                week_start=week_start, week_end=week_end, min_vol=min_vol,
            )
        elif is_week_filtered:
            players = get_weekly_range_leaderboard(
                metric, position=position, season=season,
                week_start=week_start, week_end=week_end, min_vol=min_vol,
            )
        else:
            players = get_metric_leaderboard(metric, position=position, season=season, min_vol=min_vol)
    except Exception as e:
        logger.exception(f"[api/advanced-metrics/leaderboard] error for metric={metric}: {e}")
        players = []

    # Attach experience so the client can offer a rookie/years-exp filter; the
    # reduced players index has no years_exp, so read the full feed (cached).
    try:
        _full_players_lb = get_players_global() or {}
        for _p in players or []:
            if "years_exp" not in _p:
                _p["years_exp"] = (
                    _full_players_lb.get(str(_p.get("player_id"))) or {}
                ).get("years_exp")
    except Exception:
        logger.debug("leaderboard years_exp enrich failed", exc_info=True)

    # Season-accurate team(s): each row must reflect the team the player was on
    # DURING this season, not his current team - otherwise a traded player (e.g.
    # Mike Evans, Tampa Bay in 2025 -> San Francisco now) drops out when you
    # filter by his old team, even though he played there that whole season.
    # teams_in_season returns every team he appeared for that season with the
    # weeks on each, so the team filter can match any of them and the UI can flag
    # a mid-season move. Only runs when a season is specified (always, from the
    # page); skipped otherwise so behavior is unchanged.
    if season:
        try:
            from data_building.external_data.player_team_history import teams_in_season
            for _p in players or []:
                stints = teams_in_season(str(_p.get("player_id")), season)
                if not stints:
                    continue
                _p["teams"] = [s["team"] for s in stints]
                _p["team_weeks"] = {s["team"]: s["weeks"] for s in stints}
                # Display team = the one with the most weeks (ties -> first stint).
                _primary = max(stints, key=lambda s: len(s.get("weeks") or []))
                _p["team"] = _primary["team"]
                _p["multi_team"] = len(stints) > 1
        except Exception:
            logger.debug("leaderboard season-team enrich failed", exc_info=True)

    spec = LEADERBOARD_METRICS[metric]
    vol_col = (spec.get("min_vol") or {}).get("col") or "games"
    resp = jsonify({
        "metric": metric,
        "label": spec["label"],
        "positions": spec["positions"],
        "lower_better": bool(spec.get("lower_better")),
        "vol_col": vol_col,
        "weekly_capable": weekly_capable,
        "is_week_filtered": is_week_filtered,
        "players": players,
    })
    # Leaderboard data is rebuilt at most daily, so let the browser reuse the
    # response for a few minutes — makes graph reopens / metric toggles instant
    # even across page reloads within a session.
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp


@advanced_metrics_bp.route("/api/advanced-metrics/weekly-bulk")
def api_advanced_metrics_weekly_bulk():
    """All _WEEKLY_METRICS aggregated per player for a week range, in one response."""
    from data_building.advanced_metrics import get_all_weekly_metrics_bulk
    season_str    = (request.args.get("season")     or "").strip()
    wstart_str    = (request.args.get("week_start") or "").strip()
    wend_str      = (request.args.get("week_end")   or "").strip()
    position      = (request.args.get("position")   or "").strip().upper() or None
    season        = int(season_str)  if season_str.isdigit()  else None
    week_start    = int(wstart_str)  if wstart_str.isdigit()  else None
    week_end      = int(wend_str)    if wend_str.isdigit()    else None
    try:
        data = get_all_weekly_metrics_bulk(season, week_start, week_end, position)
        return jsonify(data)
    except Exception as e:
        logger.exception(f"[api/advanced-metrics/weekly-bulk] {e}")
        return jsonify({"byId": {}, "keys": []}), 200


@advanced_metrics_bp.route("/api/advanced-metrics/config")
def api_advanced_metrics_config():
    """Return a lightweight version of LEADERBOARD_METRICS for frontend rendering.

    Strips large internal-only keys (min_vol, computed_sql, computed_null, etc.)
    and returns only what the frontend needs: label, category, positions,
    lower_better, pct, pct_frac, efficiency, integer, weeklyCapable.
    Cached with a long TTL since the config only changes with deploys.
    """
    from data_building.advanced_metrics import LEADERBOARD_METRICS, ADV_WEEKLY_METRIC_KEYS
    from data_building.advanced_metrics import _WEEKLY_METRICS  # noqa: F401 — weekly_capable check
    weekly_keys = {*_WEEKLY_METRICS.keys(), *ADV_WEEKLY_METRIC_KEYS}
    out = {}
    for key, spec in LEADERBOARD_METRICS.items():
        if spec.get("hidden"):
            continue
        out[key] = {
            "label":       spec.get("label", key),
            "category":    spec.get("category", ""),
            "positions":   spec.get("positions", []),
            "lower_better": bool(spec.get("lower_better")),
            "pct":         bool(spec.get("pct")),
            "pct_frac":    bool(spec.get("pct_frac")),
            "efficiency":  bool(spec.get("efficiency")),
            "integer":     bool(spec.get("integer")),
            "weeklyCapable": key in weekly_keys,
            "desc":        spec.get("desc", ""),
        }
    resp = jsonify({"metrics": out})
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@advanced_metrics_bp.route("/api/advanced-metrics/top-role-players")
def api_top_role_players():
    """
    Get players with highest role scores (usage + efficiency composite).

    Query params:
        position: Filter by position (QB/RB/WR/TE) or omit for all
        limit: Max number of players (default 50)

    Returns:
        [
            {
                "player_id": "123",
                "position": "RB",
                "role_score": 78.5,
                "snap_share": 0.82,
                "opportunity_share": 18.3,
                ...
            },
            ...
        ]
    """
    try:
        global _ROLE_PLAYERS_CACHE, _ROLE_PLAYERS_CACHE_TS
        from data_building.advanced_metrics import get_top_role_players

        position = request.args.get("position")
        if position:
            position = position.upper().strip()
            if position not in ("QB", "RB", "WR", "TE"):
                position = None

        try:
            limit = int(request.args.get("limit", 50))
            limit = max(1, min(limit, 200))  # Clamp between 1-200
        except (TypeError, ValueError):
            limit = 50

        cache_key = (position, limit)
        now = time.time()
        if cache_key in _ROLE_PLAYERS_CACHE and now - _ROLE_PLAYERS_CACHE_TS.get(cache_key, 0) < _ADVANCED_METRICS_TTL:
            return jsonify(_ROLE_PLAYERS_CACHE[cache_key])

        players = get_top_role_players(position=position, limit=limit)

        # Clean up internal fields
        for player in players:
            player.pop("id", None)
            # Convert decimals to floats
            for k, v in player.items():
                if v is not None and k not in ("player_id", "position", "as_of_date"):
                    player[k] = float(v)

        _ROLE_PLAYERS_CACHE[cache_key] = players
        _ROLE_PLAYERS_CACHE_TS[cache_key] = now
        return jsonify(players)

    except Exception as e:
        logger.exception("[top-role-players] Error")
        import traceback
        traceback.print_exc()
        return jsonify([])


@advanced_metrics_bp.route("/api/advanced-metrics/breakout-candidates")
def api_advanced_metrics_breakout_candidates():
    """
    Get breakout candidates using multi-factor analysis.

    Query params:
        lookback_days: Days to analyze trends (default 14)
        min_games: Minimum games played (default 2)

    Returns:
        [
            {
                "player_id": "456",
                "name": "Player Name",
                "position": "WR",
                "age": 23.5,
                "breakout_score": 67.3,
                "score_components": {
                    "snap_increase": 15.2,
                    "opportunity_increase": 22.5,
                    "efficiency_gains": 12.1,
                    ...
                },
                "value_delta": 125.0
            },
            ...
        ]
    """
    try:
        global _BREAKOUT_CACHE, _BREAKOUT_CACHE_TS
        from data_building.advanced_metrics import detect_breakout_candidates

        try:
            lookback_days = int(request.args.get("lookback_days", 14))
            lookback_days = max(7, min(lookback_days, 90))  # Clamp 7-90 days
        except (TypeError, ValueError):
            lookback_days = 14

        try:
            min_games = int(request.args.get("min_games", 2))
            min_games = max(1, min(min_games, 10))
        except (TypeError, ValueError):
            min_games = 2

        now = time.time()
        if _BREAKOUT_CACHE is not None and now - _BREAKOUT_CACHE_TS < _ADVANCED_METRICS_TTL:
            return jsonify(_BREAKOUT_CACHE)

        candidates = detect_breakout_candidates(
            lookback_days=lookback_days,
            min_games=min_games,
        )

        _BREAKOUT_CACHE = candidates
        _BREAKOUT_CACHE_TS = now
        return jsonify(candidates)

    except Exception as e:
        logger.exception("[breakout-candidates] Error")
        import traceback
        traceback.print_exc()
        return jsonify([])
