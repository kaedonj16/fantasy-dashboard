"""
League info pages: standings, waivers, and the scout redirect.

Routes:
    /<platform>/<season>/<league_id>/standings   (page_standings)
    /api/standings-week                           (api_standings_week)
    /<platform>/<season>/<league_id>/waivers      (page_waivers)
    /<platform>/<season>/<league_id>/scout        (redirect to the Matchups scout tab)

Extracted from app.py to reduce monolith size. App.py internals are reached via
the lazy shims below so importing the blueprint at start-up stays free of a
circular import — the real functions are only fetched when a request is served.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, redirect, request, url_for

logger = logging.getLogger(__name__)

league_pages_bp = Blueprint("league_pages", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ─────────────────

def render_page(*args, **kwargs):
    from app import render_page as _fn
    return _fn(*args, **kwargs)

def get_league_ctx_from_cache(*args, **kwargs):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*args, **kwargs)

def _build_offseason_standings_body(*args, **kwargs):
    from app import _build_offseason_standings_body as _fn
    return _fn(*args, **kwargs)

def build_standings_body(*args, **kwargs):
    from app import build_standings_body as _fn
    return _fn(*args, **kwargs)

def build_standings_as_of_week(*args, **kwargs):
    from app import build_standings_as_of_week as _fn
    return _fn(*args, **kwargs)

def _standings_panels(*args, **kwargs):
    from app import _standings_panels as _fn
    return _fn(*args, **kwargs)

def build_waivers_body(*args, **kwargs):
    from app import build_waivers_body as _fn
    return _fn(*args, **kwargs)


@league_pages_bp.route("/<platform>/<int:season>/<league_id>/standings")
def page_standings(platform: str, season: int, league_id: str):
    ctx = get_league_ctx_from_cache(platform, league_id, season)

    if ctx.get("offseason_mode"):
        body = _build_offseason_standings_body(ctx)
    else:
        body = build_standings_body(ctx)

    return render_page("BR Fantasy Standings", league_id, "standings", body, platform, season)


@league_pages_bp.route("/api/standings-week")
def api_standings_week():
    """Re-render the standings/power/sidebar panels as they stood through a
    chosen finalized week, for the standings page week-selector."""
    platform = request.args.get("platform", "sleeper")
    league_id = request.args.get("league_id", "")
    try:
        season = int(request.args.get("season", 0))
        week = int(request.args.get("week", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad params"}), 400
    if not league_id or week <= 0:
        return jsonify({"ok": False, "error": "missing params"}), 400
    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        capped = build_standings_as_of_week(ctx, week)
        panels = _standings_panels(capped, power_rankings=None)
        return jsonify({
            "ok": True,
            "week": week,
            "standings_html": panels["standings"],
            "details_html": panels["details"],
            "power_html": panels["power"],
            "sidebar_html": panels["sidebar"],
        })
    except Exception as e:
        logger.warning("[standings-week] render failed: %s", e, exc_info=True)
        return jsonify({"ok": False, "error": "render failed"}), 500


@league_pages_bp.route("/<platform>/<int:season>/<league_id>/waivers")
def page_waivers(platform: str, season: int, league_id: str):
    ctx = get_league_ctx_from_cache(platform, league_id, season)
    body = build_waivers_body(platform, season, league_id, ctx)
    return render_page("BR Fantasy Waivers", league_id, "waivers", body, platform, season)


@league_pages_bp.route("/<platform>/<int:season>/<league_id>/scout")
def page_scout(platform: str, season: int, league_id: str):
    # Scout Report lives as a tab on the Matchups page, not its own page.
    # Redirect (keeps old links/bookmarks working) to that tab.
    return redirect(
        url_for("page_weekly", platform=platform, season=season, league_id=league_id)
        + "?tab=scout"
    )
