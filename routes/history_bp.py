"""History API endpoints (AI recap, summary, standings, season-trend chart).

Routes:
    /api/history/ai-recap
    /api/history/<platform>/<int:season>/<league_id>/summary
    /api/history/<platform>/<int:season>/<league_id>/standings
    /api/history/<platform>/<int:season>/<league_id>/chart

Extracted from app.py to reduce monolith size.

Dependencies:
    - extensions.limiter for the rate-limit decorator
    - dashboard_services.* for resolve_league_id_for_season, the history-page
      renderers, and get_history_ai_recap
    - app.py internals (get_league_ctx_from_cache, get_available_history_seasons,
      _api_err) are imported lazily inside the handlers to avoid a circular
      import at module load - the same pattern the other blueprints use.
"""
from __future__ import annotations

import logging

import pandas as pd
from flask import Blueprint, jsonify, request

from dashboard_services.ai.history_recap import get_history_ai_recap
from dashboard_services.api import resolve_league_id_for_season
from extensions import limiter

logger = logging.getLogger(__name__)

history_bp = Blueprint("history", __name__)


@history_bp.route("/api/history/ai-recap")
@limiter.limit("10 per minute")
def history_ai_recap():
    """Generate AI-powered season recap for a specific team."""
    from app import get_league_ctx_from_cache

    league_id = request.args.get("league_id")
    season = request.args.get("season")
    roster_id = request.args.get("roster_id")

    if not all([league_id, season, roster_id]):
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        # Get the same context that the history page uses
        platform = (request.args.get("platform") or "sleeper").strip().lower()
        base_league_id = request.args.get("base_season", season)

        # Resolve the correct league ID for the historical season
        resolved_history_league_id = resolve_league_id_for_season(
            platform=platform,
            league_id=league_id,
            current_season=int(base_league_id),
            target_season=int(season),
        )

        # Get the exact same context the history page uses
        ctx = get_league_ctx_from_cache(platform, resolved_history_league_id, int(season))
        if not ctx:
            return jsonify({"error": "League context not found"}), 404

        # Generate recap
        recap_html = get_history_ai_recap(ctx, roster_id)

        return jsonify({"html": recap_html})

    except Exception as e:
        return jsonify({"error": "Failed to generate recap"}), 500


@history_bp.route("/api/history/<platform>/<int:season>/<league_id>/summary")
def api_history_summary(platform: str, season: int, league_id: str):
    """Get season awards/summary data."""
    from app import _api_err, get_available_history_seasons, get_league_ctx_from_cache

    try:
        from dashboard_services.pages.history_page import get_history_summary_html

        history_season = int(request.args.get("history_season", season))

        # Check if this is a valid history season
        available_seasons = get_available_history_seasons(platform, league_id, season)
        if not available_seasons:
            return jsonify({
                "html": "<div class='history-empty'>This is your first season. Historical data will be available after the season completes.</div>"
            })

        if history_season not in available_seasons:
            return jsonify({
                "html": "<div class='history-empty'>No data available for this season.</div>"
            })

        resolved_history_league_id = resolve_league_id_for_season(
            platform=platform,
            league_id=league_id,
            current_season=season,
            target_season=history_season,
        )

        history_ctx = get_league_ctx_from_cache(platform, resolved_history_league_id, history_season)
        if not history_ctx:
            return jsonify({"error": "League context not found"}), 404

        html = get_history_summary_html(history_ctx)
        return jsonify({"html": html})

    except Exception as e:
        logger.exception("[api_history_summary] Error")
        return _api_err("Request failed", e)


@history_bp.route("/api/history/<platform>/<int:season>/<league_id>/standings")
def api_history_standings(platform: str, season: int, league_id: str):
    """Get regular season standings."""
    from app import _api_err, get_available_history_seasons, get_league_ctx_from_cache

    try:
        from dashboard_services.pages.history_page import get_history_standings_html

        history_season = int(request.args.get("history_season", season))

        # Check if this is a valid history season
        available_seasons = get_available_history_seasons(platform, league_id, season)
        if not available_seasons:
            return jsonify({
                "html": "<div class='history-empty'>This is your first season. Historical standings will be available after the season completes.</div>"
            })

        if history_season not in available_seasons:
            return jsonify({
                "html": "<div class='history-empty'>No standings data available for this season.</div>"
            })

        resolved_history_league_id = resolve_league_id_for_season(
            platform=platform,
            league_id=league_id,
            current_season=season,
            target_season=history_season,
        )

        history_ctx = get_league_ctx_from_cache(platform, resolved_history_league_id, history_season)
        if not history_ctx:
            return jsonify({"error": "League context not found"}), 404

        html = get_history_standings_html(history_ctx)
        return jsonify({"html": html})

    except Exception as e:
        logger.exception("[api_history_standings] Error")
        return _api_err("Request failed", e)


@history_bp.route("/api/history/<platform>/<int:season>/<league_id>/chart")
def api_history_chart(platform: str, season: int, league_id: str):
    """Get season trend chart data."""
    from app import _api_err, get_available_history_seasons, get_league_ctx_from_cache

    try:
        from dashboard_services.pages.history_page import _filtered_season_df

        history_season = int(request.args.get("history_season", season))

        # Check if this is a valid history season
        available_seasons = get_available_history_seasons(platform, league_id, season)
        if not available_seasons:
            return jsonify({
                "error": "No data",
                "html": "<div class='history-empty'>This is your first season. Week-by-week trends will be available after the season completes.</div>"
            })

        if history_season not in available_seasons:
            return jsonify({
                "error": "No data",
                "html": "<div class='history-empty'>No weekly data available for this season.</div>"
            })

        resolved_history_league_id = resolve_league_id_for_season(
            platform=platform,
            league_id=league_id,
            current_season=season,
            target_season=history_season,
        )

        history_ctx = get_league_ctx_from_cache(platform, resolved_history_league_id, history_season)
        if not history_ctx:
            return jsonify({"error": "League context not found"}), 404

        df_weekly = history_ctx.get("df_weekly", pd.DataFrame())
        chart_df = _filtered_season_df(df_weekly)

        if chart_df.empty or not {"week", "owner", "points"}.issubset(chart_df.columns):
            return jsonify({"error": "No data",
                            "html": "<div class='history-empty'>No weekly scoring data available for this season.</div>"})

        # Build chart data for each team
        chart_data = []
        for owner, grp in chart_df.groupby("owner"):
            grp = grp.sort_values("week")
            chart_data.append({
                "name": str(owner),
                "x": grp["week"].tolist(),
                "y": grp["points"].tolist(),
            })

        return jsonify({"data": chart_data})

    except Exception as e:
        logger.exception("[api_history_chart] Error")
        return _api_err("Request failed", e)
