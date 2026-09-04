"""Extracted from app.py — breakout_api_bp2 (see route list below)."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request, session
logger = logging.getLogger(__name__)

breakout_api_bp2 = Blueprint("breakout_api_bp2", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ──
def _compute_compare_baselines(*a, **k):
    from app import _compute_compare_baselines as _fn
    return _fn(*a, **k)

def get_model_value_table_cached(*a, **k):
    from app import get_model_value_table_cached as _fn
    return _fn(*a, **k)

def get_nfl_state(*a, **k):
    from app import get_nfl_state as _fn
    return _fn(*a, **k)

def get_players_global(*a, **k):
    from app import get_players_global as _fn
    return _fn(*a, **k)

def has_premium_for_viewer(*a, **k):
    from app import has_premium_for_viewer as _fn
    return _fn(*a, **k)


def _map_in_season_breakouts(breakout_ids, players_index, values_by_id):
    """Shape engine rows into the public candidate payload."""
    candidates = []
    for b in breakout_ids or []:
        player_id = str(b.get("player_id", ""))
        player_meta = players_index.get(player_id, {})
        player_value = values_by_id.get(player_id, {})
        candidates.append({
            "player_id": player_id,
            "name": player_meta.get("name", "Unknown"),
            "team": player_meta.get("team"),
            "position": player_meta.get("pos"),
            "age": player_value.get("age"),
            "value": player_value.get("value", 0),
            "sf_value": player_value.get("sf_value", player_value.get("value", 0)),
            "pos_rank": player_value.get("pos_rank"),
            "pos_rank_label": player_value.get("pos_rank_label"),
            "breakout_score": b.get("score", 0),
        })
    return candidates


def in_season_breakout_candidates(detect_fn, players_index, values_by_id):
    """In-season breakouts from the usage/engine detector only.

    If detection fails, return [] — never substitute 7-day value risers.
    A price move is not a breakout.
    """
    try:
        rows = detect_fn(lookback_days=14)
    except Exception:
        logger.exception("[breakout-candidates] In-season detection error")
        return []
    return _map_in_season_breakouts(rows, players_index or {}, values_by_id or {})


@breakout_api_bp2.route("/api/breakout-candidates")
def api_breakout_candidates():
    """Alias of ``/api/breakout/candidates`` — same envelope, same 3-preview."""
    from dashboard_services.breakout_api import candidates as canonical
    return canonical()


@breakout_api_bp2.route("/api/offseason-breakout-candidates")
def api_offseason_breakout_candidates():
    """
    Get offseason breakout candidates based on roster changes and vacated opportunity.

    PREMIUM FEATURE - Requires active subscription.

    Identifies players who will benefit from departed teammates (FA, trades, retirements).
    Examples:
    - Mike Evans leaves TB -> Egbuka gets targets
    - Second-year WR moves up depth chart
    - Backup RB becomes lead back

    Query params:
        season: Season year (default: current year)
        min_score: Minimum breakout score (default: 40)
        position: Filter by position (QB/RB/WR/TE)
        max_per_team_position: Max candidates per team+position (default: 2, range: 1-5)

    Returns:
        [
            {
                "player_id": "789",
                "name": "Emeka Egbuka",
                "team": "TB",
                "position": "WR",
                "age": 23,
                "years_exp": 1,
                "breakout_score": 65.5,
                "projection_factors": {
                    "absolute_opportunity_increase": 25.0,
                    "relative_opportunity_increase": 18.5,
                    "team_vacancy_size": 14.0,
                    "youth_experience_bonus": 15.0
                },
                "previous_season": {
                    "targets": 45,
                    "carries": 0,
                    "snap_share": 0.42
                },
                "projected": {
                    "targets": 120,
                    "carries": 0,
                    "snap_share": 0.75
                },
                "increases": {
                    "targets": 75,
                    "carries": 0,
                    "snap_share": 0.33
                },
                "departed_players": ["Mike Evans"],
                "context": "Benefits from Mike Evans departure"
            },
            ...
        ]
    """
    try:
        from data_building.breakout_opportunity_guard import UNAVAILABLE_BREAKOUT_REASON
        from dashboard_services.breakout_api import get_breakout_candidates, _resolve_bo_season

        # Breakout candidates are a PRO feature (3-candidate preview stays on
        # /api/breakout/candidates). League-plan users need league context so
        # membership can be verified.
        league_id = request.args.get("league_id")
        platform = request.args.get("platform") or "sleeper"
        if not has_premium_for_viewer(
            session.get("viewer_username"), session.get("viewer_user_id"),
            league_id, platform, request.args.get("season"),
        ):
            return jsonify({"paywall": True, "error": "Premium required"}), 403

        season = _resolve_bo_season(request.args.get("season", type=int))
        try:
            min_score = float(request.args.get("min_score", 40))
            min_score = max(0, min(min_score, 100))
        except (TypeError, ValueError):
            min_score = 40

        position = request.args.get("position")
        if position:
            position = position.upper().strip()
            if position not in ("QB", "RB", "WR", "TE"):
                position = None

        result = get_breakout_candidates(season, min_score, limit=None)
        if not result.get("data_available", True):
            return jsonify({
                "candidates": [],
                "count": 0,
                "data_available": False,
                "reason": result.get("reason") or UNAVAILABLE_BREAKOUT_REASON,
                "season": result.get("season") or season,
            })
        candidates = list(result.get("candidates") or [])
        if position:
            candidates = [c for c in candidates if c.get("position") == position]
        return jsonify({
            "candidates": candidates,
            "count": len(candidates),
            "data_available": True,
            "season": result.get("season") or season,
        })

    except Exception as e:
        logger.info(f"[offseason-breakout-candidates] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"candidates": [], "count": 0, "data_available": False})


@breakout_api_bp2.route("/api/calculate-breakout-scores")
def api_calculate_breakout_scores():
    """
    Calculate and save breakout scores for all players.

    This is an admin endpoint that runs the unified breakout engine
    and saves results to the database.

    Query params:
        season: Season year (default: current year)
        min_score: Minimum score to save (default: 30)

    Returns:
        {
            "success": true,
            "candidates_calculated": 150,
            "candidates_saved": 150,
            "phase": "post_free_agency",
            "season": 2026
        }
    """
    try:
        from datetime import datetime
        from data_building.breakout_engine import BreakoutEngine
        from utils.utils import load_players_index, load_usage_table

        # Get season
        nfl_state = get_nfl_state() or {}
        default_season = int(nfl_state.get("season") or datetime.now().year)

        try:
            season = int(request.args.get("season", default_season))
        except (TypeError, ValueError):
            season = default_season

        # Get min score
        try:
            min_score = float(request.args.get("min_score", 30))
        except (TypeError, ValueError):
            min_score = 30

        # Initialize engine. Pass the live Sleeper players feed so an injured
        # starter sitting ahead of a candidate on the depth chart boosts their
        # breakout (the same "starter in front got hurt" opening used for waiver
        # targets). Best-effort — a missing feed just leaves the score as-is.
        try:
            _full_players = get_players_global() or {}
        except Exception:
            _full_players = {}
        engine = BreakoutEngine(season=season, full_players=_full_players)

        # Get all players from usage table or players_index
        # This is a simplified version - in production you'd filter to relevant players
        usage_table = load_usage_table() or []
        players_index = load_players_index() or {}

        # Build player list (top 600 by value/relevance)
        player_list = []
        for player in usage_table[:600]:  # Limit to top 600
            player_id = player.get('player_id') or player.get('id')
            if not player_id:
                continue

            # Get additional metadata from players_index
            player_meta = players_index.get(player_id, {})

            player_list.append({
                'player_id': player_id,
                'player_name': player.get('name') or player_meta.get('full_name'),
                'team': player.get('team') or player_meta.get('team'),
                'position': player.get('position') or player_meta.get('pos'),
                'age': player_meta.get('age'),
                'years_exp': player_meta.get('years_exp', 0)
            })

        # Calculate scores
        candidates = engine.calculate_breakout_scores(player_list, min_score=min_score)

        # Save to database
        saved_count = engine.save_scores(candidates)

        return jsonify({
            "success": True,
            "candidates_calculated": len(candidates),
            "candidates_saved": saved_count,
            "phase": engine.phase,
            "season": season
        })

    except Exception:
        logger.exception("[calculate-breakout-scores] Error")
        return jsonify({"error": "Internal error", "success": False}), 500


@breakout_api_bp2.route("/api/compare-baselines")
def api_compare_baselines():
    """Selectable tier-average opponents for the compare page (Avg WR1, RB2, ...)."""
    try:
        return jsonify({"baselines": _compute_compare_baselines()})
    except Exception:
        logger.exception("[api_compare_baselines] error")
        return jsonify({"baselines": []}), 200
