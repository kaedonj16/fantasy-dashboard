"""
Player data API endpoints.

Routes:
    /api/player-advanced-metrics/<player_id>
    /api/player-news/<player_id>
    /api/nfl-news
    /api/player-value-history/<player_id>

Extracted from app.py to reduce monolith size.
Dependencies: dashboard_services.*, data_building.*, utils.* only - no app.py internals.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

from flask import Blueprint, jsonify, request, session

from dashboard_services.api import get_nfl_state
from dashboard_services.subscriptions import has_premium_access
from data_building.player_value_history import get_player_value_history

logger = logging.getLogger(__name__)

players_bp = Blueprint("players", __name__)


# ── /api/player-advanced-metrics/<player_id> ──────────────────────────────────

@players_bp.route("/api/player-advanced-metrics/<player_id>")
def api_player_advanced_metrics(player_id: str):
    """
    Get advanced efficiency metrics for a specific player.

    Query params:
        season: NFL season year (e.g. 2025). Omit to use the default season
                (current season during regular season, most recent with data
                during offseason).

    Returns:
        {
            "player_id": "123",
            "position": "WR",
            "season": 2025,
            "available_seasons": [2025, 2024],
            "metrics": {
                "yards_per_target": 8.5,
                "catch_rate": 0.72,
                ...
            },
            "as_of_date": "2025-01-15"
        }
    """
    try:
        from data_building.advanced_metrics import (
            get_player_metrics,
            get_player_metrics_by_season,
            get_player_career_metrics,
            get_available_seasons_for_player,
        )

        # Determine default season from NFL state
        nfl_state = get_nfl_state() or {}
        nfl_season = int(nfl_state.get("season") or datetime.now().year)
        is_offseason = (nfl_state.get("season_type") or "").lower() == "off"

        # Parse requested season from query param
        requested_season = request.args.get("season")
        is_career_request = requested_season == "career" or requested_season is None

        if requested_season and requested_season != "career":
            try:
                requested_season = int(requested_season)
            except (ValueError, TypeError):
                requested_season = None

        # Get all seasons with data for this player
        available_seasons = get_available_seasons_for_player(str(player_id))

        # Choose target season: explicit request → current (if in-season) → most recent
        if requested_season:
            target_season = requested_season
        elif not is_offseason and nfl_season in available_seasons:
            target_season = nfl_season
        elif available_seasons:
            target_season = available_seasons[0]  # most recent season with data
        else:
            target_season = nfl_season

        # Fetch metrics
        if is_career_request:
            # Career mode - aggregate across all seasons
            metrics = get_player_career_metrics(str(player_id))
            target_season = None
        elif requested_season:
            # Specific season requested
            metrics = get_player_metrics_by_season(str(player_id), requested_season)
            target_season = requested_season
        elif not is_offseason and nfl_season in available_seasons:
            # Current season (in-season)
            metrics = get_player_metrics_by_season(str(player_id), nfl_season)
            target_season = nfl_season
        elif available_seasons:
            # Most recent season with data
            target_season = available_seasons[0]
            metrics = get_player_metrics_by_season(str(player_id), target_season)
        else:
            # Fallback to latest
            metrics = get_player_metrics(str(player_id))
            target_season = None

        if not metrics:
            return jsonify({
                "player_id": str(player_id),
                "error": "No metrics available for this player"
            }), 404

        # Extract and clean metadata fields
        as_of_date = str(metrics.pop("as_of_date", None))
        season_val = metrics.pop("season", target_season)
        metrics.pop("id", None)

        metrics_payload = {
            k: (float(v) if v is not None and not isinstance(v, (datetime, date)) else (str(v) if isinstance(v, (datetime, date)) else None))
            for k, v in metrics.items()
            if k not in ("player_id", "position")
        }

        # Blend usage-based role score with PFF quality grades for a single
        # evaluation signal used by the modal.
        role = metrics_payload.get("role_score")
        off = metrics_payload.get("grades_offense")
        rush = metrics_payload.get("pff_rushing_grade")
        ppass = metrics_payload.get("pff_passing_grade")

        quality = ppass if metrics.get("position") == "QB" else (rush or off)
        if role is not None and quality is not None:
            metrics_payload["player_evaluation_score"] = round((float(role) * 0.65) + (float(quality) * 0.35), 1)
        elif role is not None:
            metrics_payload["player_evaluation_score"] = round(float(role), 1)
        elif quality is not None:
            metrics_payload["player_evaluation_score"] = round(float(quality), 1)

        return jsonify({
            "player_id": str(player_id),
            "position": metrics.get("position"),
            "season": season_val,
            "available_seasons": available_seasons,
            "metrics": metrics_payload,
            "as_of_date": as_of_date,
        })

    except Exception as e:
        logger.exception("[player-advanced-metrics] Error for %s", player_id)
        import traceback
        traceback.print_exc()
        return jsonify({
            "player_id": str(player_id),
            "error": "Failed to retrieve metrics"
        }), 500


# ── /api/player-value-history/<player_id> ─────────────────────────────────────

@players_bp.route("/api/player-value-history/<player_id>")
def api_player_value_history(player_id: str):
    try:
        days = int(request.args.get("days", 30))
    except (TypeError, ValueError):
        days = 30

    league_type = str(request.args.get("league_type", "1qb")).strip().lower()
    try:
        league_size = int(request.args.get("league_size", 10))
        if league_size not in (8, 10, 12, 14):
            league_size = 10
    except (TypeError, ValueError):
        league_size = 10

    history = get_player_value_history(
        player_id, days=max(days, 1),
        league_type=league_type, league_size=league_size,
    )
    return jsonify(
        {
            "player_id": str(player_id),
            "days": max(days, 1),
            "history": history,
        }
    )


# ── /api/player-news/<player_id> ──────────────────────────────────────────────

@players_bp.route("/api/player-news/<player_id>")
def api_player_news(player_id: str):
    """Recent news headlines for a single player (ESPN free API, 30-min cache)."""
    try:
        from utils.utils import load_players_index
        from dashboard_services.news import get_player_news

        players_index = load_players_index() or {}
        meta = players_index.get(str(player_id)) or {}
        name = meta.get("name") or meta.get("full_name") or ""
        espn_id = str(meta.get("espnID") or "").strip()
        # Build headshot URL from espnID (players_index has espnID, not espnHeadshot)
        headshot = (
            meta.get("espnHeadshot")
            or (f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png" if espn_id else "")
        )

        items = get_player_news(player_name=name, espn_headshot=headshot, limit=4)
        return jsonify({"player_id": player_id, "name": name, "news": items})
    except Exception:
        logger.exception("[player-news] error")
        return jsonify({"player_id": player_id, "news": []}), 200


# ── /api/nfl-news ─────────────────────────────────────────────────────────────

@players_bp.route("/api/nfl-news")
def api_nfl_news():
    """Latest NFL headlines for the activity feed sidebar (ESPN free API, 15-min cache)."""
    try:
        from dashboard_services.news import get_nfl_news
        limit = min(int(request.args.get("limit") or 15), 30)
        items = get_nfl_news(limit=limit)
        return jsonify({"news": items})
    except Exception:
        logger.exception("[nfl-news] error")
        return jsonify({"news": []}), 200
