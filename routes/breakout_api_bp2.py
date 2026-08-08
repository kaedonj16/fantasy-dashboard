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

def get_top_movers(*a, **k):
    from app import get_top_movers as _fn
    return _fn(*a, **k)

def has_premium_for_viewer(*a, **k):
    from app import has_premium_for_viewer as _fn
    return _fn(*a, **k)


@breakout_api_bp2.route("/api/breakout-candidates")
def api_breakout_candidates():
    """
    Get breakout candidates - automatically switches between offseason and in-season detection.
    Returns full candidate objects with stats, not just IDs.
    """
    league_id = request.args.get("league_id")
    platform = request.args.get("platform", "sleeper")
    if not has_premium_for_viewer(session.get("viewer_username"), session.get("viewer_user_id"),
                                  league_id, platform, request.args.get("season")):
        return jsonify({"paywall": True, "error": "Premium required"}), 403

    try:
        from datetime import datetime
        from utils.utils import load_players_index, load_model_value_table

        min_score = float(request.args.get("min_score", 40))  # Selective threshold
        limit = int(request.args.get("limit", 20))

        # Get current NFL state
        nfl_state = get_nfl_state() or {}
        current_season = int(nfl_state.get("season") or datetime.now().year)
        season_type = str(nfl_state.get("season_type", "")).lower().strip()
        is_offseason = season_type == "off"

        candidates = []

        if is_offseason:
            # Use offseason opportunity-based detection (FAST - uses database)
            try:
                from data_building.offseason_opportunity import get_offseason_breakout_candidates
                candidates = get_offseason_breakout_candidates(
                    current_season,
                    min_score=min_score,
                    limit=limit * 5,  # Get more initially for filtering
                    max_per_team_position=2
                )
                logger.info(f"[breakout-candidates] Offseason mode: {len(candidates)} candidates")
            except Exception as e:
                logger.info(f"[breakout-candidates] Offseason detection error: {e}")
        else:
            # Use in-season breakout detection with enrichment
            try:
                from data_building.advanced_metrics import detect_breakout_candidates

                breakout_ids = detect_breakout_candidates(lookback_days=14)

                # Enrich with full player data
                players_index = load_players_index() or {}
                value_table = get_model_value_table_cached() or []
                values_by_id = {str(p.get("id")): p for p in value_table}

                for b in breakout_ids:
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

                logger.info(f"[breakout-candidates] In-season mode: {len(candidates)} candidates")
            except Exception as e:
                logger.info(f"[breakout-candidates] In-season detection error: {e}")
                # Fallback to value movers
                movers_data = get_top_movers(days=7, limit=100) or {}
                players_index = load_players_index() or {}
                value_table = list(get_model_value_table_cached() or [])
                values_by_id = {str(p.get("id")): p for p in value_table}

                for player in movers_data.get("risers", []):
                    delta = player.get("delta", 0)
                    position = player.get("position", "")
                    threshold = 100 if position == "TE" else 75

                    if delta >= threshold:
                        player_id = str(player.get("player_id", ""))
                        player_meta = players_index.get(player_id, {})
                        player_value = values_by_id.get(player_id, {})

                        candidates.append({
                            "player_id": player_id,
                            "name": player.get("name", "Unknown"),
                            "team": player_meta.get("team"),
                            "position": position,
                            "age": player_value.get("age"),
                            "value": player.get("value", 0),
                            "sf_value": player.get("sf_value", player.get("value", 0)),
                            "pos_rank": player_value.get("pos_rank"),
                            "pos_rank_label": player_value.get("pos_rank_label"),
                            "breakout_score": delta,
                        })

        # Sort by breakout score and limit
        candidates.sort(key=lambda x: x.get("breakout_score", 0), reverse=True)
        return jsonify(candidates[:limit])

    except Exception as e:
        logger.exception("[breakout-candidates] Error")
        import traceback
        traceback.print_exc()
        return jsonify([])


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
        from datetime import datetime
        from data_building.offseason_opportunity import get_offseason_breakout_candidates

        # Breakout candidates are now available to all users (no premium check)

        # Get season (default to current year)
        nfl_state = get_nfl_state() or {}
        default_season = int(nfl_state.get("season") or datetime.now().year)

        try:
            season = int(request.args.get("season", default_season))
        except (TypeError, ValueError):
            season = default_season

        # Get min score threshold (default 40 for selectivity)
        try:
            min_score = float(request.args.get("min_score", 40))
            min_score = max(0, min(min_score, 100))
        except (TypeError, ValueError):
            min_score = 40

        # Get max per team/position (default 2, range 1-5)
        try:
            max_per_team_position = int(request.args.get("max_per_team_position", 2))
            max_per_team_position = max(1, min(max_per_team_position, 5))
        except (TypeError, ValueError):
            max_per_team_position = 2

        # Get position filter
        position = request.args.get("position")
        if position:
            position = position.upper().strip()
            if position not in ("QB", "RB", "WR", "TE"):
                position = None

        # Get candidates (FAST - uses database queries, no artificial filtering)
        candidates = get_offseason_breakout_candidates(
            season,
            min_score=min_score,
            max_per_team_position=max_per_team_position
        )

        # Filter by position if requested
        if position:
            candidates = [c for c in candidates if c.get("position") == position]

        # Filter out elite players (they shouldn't be breakout candidates)
        # Load model values to check elite thresholds
        model_values = get_model_value_table_cached() or []
        values_by_id = {str(p["id"]): p for p in model_values if isinstance(p, dict) and p.get("id")}

        # Position-specific elite thresholds
        elite_thresholds = {
            'RB': 650, 'WR': 650, 'TE': 550, 'QB': 400, 'K': 9999, 'DEF': 9999
        }

        filtered_candidates = []
        for candidate in candidates:
            player_id = str(candidate.get("player_id", ""))
            pos = candidate.get("position", "")
            threshold = elite_thresholds.get(pos, 750)

            # Get player value
            player_value = values_by_id.get(player_id, {})
            value = float(player_value.get("value", 0)) if player_value.get("value") else 0

            # Only include if not elite
            if value < threshold:
                filtered_candidates.append(candidate)

        candidates = filtered_candidates

        return jsonify(candidates)

    except Exception as e:
        logger.info(f"[offseason-breakout-candidates] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])


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

        # Initialize engine
        engine = BreakoutEngine(season=season)

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

    except Exception as e:
        logger.info(f"[calculate-breakout-scores] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@breakout_api_bp2.route("/api/compare-baselines")
def api_compare_baselines():
    """Selectable tier-average opponents for the compare page (Avg WR1, RB2, ...)."""
    try:
        return jsonify({"baselines": _compute_compare_baselines()})
    except Exception:
        logger.exception("[api_compare_baselines] error")
        return jsonify({"baselines": []}), 200
