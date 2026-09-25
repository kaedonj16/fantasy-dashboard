"""Weekly API endpoints (Weekly Wrapped overlay).

Routes:
    /api/weekly/<platform>/<int:season>/<league_id>/<int:week>/wrapped

Extracted pattern follows routes/history_bp.py: resolve the league ctx from the
app cache (lazy import to avoid a circular import), build the slides, and
return {"html": ...}. The overlay is fetched lazily on first click of the
Weekly Wrapped launcher, so the per-week boxscore aggregation never blocks the
hub render.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

weekly_bp = Blueprint("weekly", __name__)


def _weekly_wrapped_request_has_premium(platform, season, league_id) -> bool:
    """Per-user PRO check for the weekly wrapped overlay endpoint. Fail closed."""
    from flask import session
    from dashboard_services.subscriptions import has_premium_for_viewer
    try:
        return bool(has_premium_for_viewer(
            session.get("viewer_username"), session.get("viewer_user_id"),
            league_id, platform or "sleeper", season,
        ))
    except Exception:
        logger.debug("weekly wrapped premium check failed", exc_info=True)
        return False


@weekly_bp.route("/api/weekly/<platform>/<int:season>/<league_id>/<int:week>/wrapped")
def api_weekly_wrapped(platform: str, season: int, league_id: str, week: int):
    """Build the full Weekly Wrapped overlay for one completed week (including
    the boxscore-backed top-player / position-leader / dud slides). Returns
    {"html": ""} when the week has no completed games."""
    from app import _api_err, get_league_ctx_from_cache

    try:
        from dashboard_services.pages.history_page import render_weekly_wrapped_overlay

        if week < 1 or week > 25:
            return jsonify({"html": ""})

        ctx = get_league_ctx_from_cache(platform, league_id, season)
        if not ctx:
            return jsonify({"html": ""})

        show_pro_cta = not _weekly_wrapped_request_has_premium(
            platform, season, league_id)
        html = render_weekly_wrapped_overlay(ctx, week, show_pro_cta=show_pro_cta)
        return jsonify({"html": html})

    except Exception as e:
        logger.exception("[api_weekly_wrapped] Error")
        return _api_err("Request failed", e)
