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
from werkzeug.exceptions import HTTPException

from dashboard_services.ai.history_recap import get_history_ai_recap
from dashboard_services.api import resolve_league_id_for_season
from extensions import limiter
from utils.api_params import api_int

from dashboard_services.historical.aggregates_store import aggregates_version, load_profile_aggregates
from dashboard_services.historical.board import build_deep_panel, build_historical_trends
from dashboard_services.historical.cohorts import evaluate_cohort
from dashboard_services.historical.filters import scout_matching_players

logger = logging.getLogger(__name__)

history_bp = Blueprint("history", __name__)


def _wrapped_request_has_premium(platform, season, league_id) -> bool:
    """Per-user PRO check for the wrapped overlay endpoints. Fail closed."""
    from flask import session
    from dashboard_services.subscriptions import has_premium_for_viewer
    try:
        return bool(has_premium_for_viewer(
            session.get("viewer_username"), session.get("viewer_user_id"),
            league_id, platform or "sleeper", season,
        ))
    except Exception:
        logger.debug("wrapped premium check failed", exc_info=True)
        return False


@history_bp.route("/api/history/ai-recap")
@limiter.limit("10 per minute")
def history_ai_recap():
    """Generate AI-powered season recap for a specific team."""
    from app import get_league_ctx_from_cache

    league_id = request.args.get("league_id")
    season = request.args.get("season")
    roster_id = request.args.get("roster_id")
    owner_id = request.args.get("owner_id")

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

        # Roster ids are season-local.  Resolve the requested stable provider
        # owner in the selected season and never use a team name as identity.
        if owner_id:
            from dashboard_services.historical_identity import roster_id_for_owner
            resolved_roster_id = roster_id_for_owner(ctx, owner_id)
            if resolved_roster_id is None:
                return jsonify({"error": "Owner is not a member of this season"}), 404
            roster_id = resolved_roster_id
        else:
            valid_ids = {str(r.get("roster_id")) for r in (ctx.get("rosters") or [])}
            if str(roster_id) not in valid_ids:
                return jsonify({"error": "Team is not a member of this season"}), 404

        # Generate recap
        recap_html = get_history_ai_recap(ctx, str(roster_id))

        return jsonify({"html": recap_html})

    except Exception as e:
        return jsonify({"error": "Failed to generate recap"}), 500


@history_bp.route("/api/history/<platform>/<int:season>/<league_id>/summary")
def api_history_summary(platform: str, season: int, league_id: str):
    """Get season awards/summary data."""
    from app import _api_err, get_available_history_seasons, get_league_ctx_from_cache

    try:
        from dashboard_services.pages.history_page import get_history_summary_html

        history_season = api_int("history_season", season)

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

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[api_history_summary] Error")
        return _api_err("Request failed", e)


@history_bp.route("/api/history/<platform>/<int:season>/<league_id>/wrapped")
def api_history_wrapped(platform: str, season: int, league_id: str):
    """Build the full Season Wrapped overlay (including the boxscore-backed MVP /
    position slides). Lazy-loaded on first click so the page render never blocks
    on the per-week fetches."""
    from app import _api_err, get_available_history_seasons, get_league_ctx_from_cache

    try:
        from dashboard_services.pages.history_page import (
            _build_summary,
            render_history_wrapped_overlay,
        )

        history_season = api_int("history_season", season)

        available_seasons = get_available_history_seasons(platform, league_id, season)
        if not available_seasons or history_season not in available_seasons:
            return jsonify({"html": ""})

        resolved_history_league_id = resolve_league_id_for_season(
            platform=platform,
            league_id=league_id,
            current_season=season,
            target_season=history_season,
        )

        history_ctx = get_league_ctx_from_cache(platform, resolved_history_league_id, history_season)
        if not history_ctx:
            return jsonify({"html": ""})

        # Summary is cheap and the overlay builder needs it; cache it on the ctx.
        history_ctx.setdefault("summary", _build_summary(history_ctx))
        show_pro_cta = not _wrapped_request_has_premium(
            platform, season, resolved_history_league_id)
        html = render_history_wrapped_overlay(history_ctx, history_season,
                                              show_pro_cta=show_pro_cta)
        return jsonify({"html": html})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[api_history_wrapped] Error")
        return _api_err("Request failed", e)


@history_bp.route("/api/history/<platform>/<int:season>/<league_id>/standings")
def api_history_standings(platform: str, season: int, league_id: str):
    """Get regular season standings."""
    from app import _api_err, get_available_history_seasons, get_league_ctx_from_cache

    try:
        from dashboard_services.pages.history_page import get_history_standings_html

        history_season = api_int("history_season", season)

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

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[api_history_standings] Error")
        return _api_err("Request failed", e)


@history_bp.route("/api/history/<platform>/<int:season>/<league_id>/chart")
def api_history_chart(platform: str, season: int, league_id: str):
    """Get season trend chart data."""
    from app import _api_err, get_available_history_seasons, get_league_ctx_from_cache

    try:
        from dashboard_services.pages.history_page import _filtered_season_df

        history_season = api_int("history_season", season)

        # Check if this is a valid history season
        available_seasons = get_available_history_seasons(platform, league_id, season)
        if not available_seasons:
            return jsonify({
                "html": "<div class='history-empty'>This is your first season. Week-by-week trends will be available after the season completes.</div>"
            })

        if history_season not in available_seasons:
            return jsonify({
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
            return jsonify({
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

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[api_history_chart] Error")
        return _api_err("Request failed", e)


# ────────────────────────────────────────────────────────────────────────────
# Merged from routes/weekly_bp.py: Weekly API endpoints (Weekly Wrapped overlay).
# ────────────────────────────────────────────────────────────────────────────

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


@history_bp.route("/api/weekly/<platform>/<int:season>/<league_id>/<int:week>/wrapped")
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


# ────────────────────────────────────────────────────────────────────────────
# Merged from routes/wrapped_share_bp.py: Shareable public links for Wrapped decks.
# ────────────────────────────────────────────────────────────────────────────

#: Hard cap on the stored overlay HTML (a typical deck is ~10-30KB).
MAX_OVERLAY_BYTES = 200 * 1024
#: Hard cap on the share payload JSON.
MAX_SHARE_DATA_BYTES = 50 * 1024

_VALID_KINDS = ("season", "weekly")


def _share_label(kind: str, share_data: dict) -> str:
    league = str(share_data.get("league") or "League")
    if kind == "weekly" and share_data.get("week"):
        return f"{league}: Week {share_data['week']} Wrapped"
    season = str(share_data.get("season") or "").strip()
    return f"{league}: {season} Season Wrapped".replace("  ", " ").strip(" :")


@history_bp.route("/api/wrapped/share", methods=["POST"])
@limiter.limit("20 per hour")
def api_wrapped_share_create():
    """Mint a public share link for a Wrapped deck.

    Body: {kind: "season"|"weekly", ns, overlay_html, share_data}.
    Returns {"url": "<public url>"}.
    """
    from dashboard_services.wrapped_shares import create_wrapped_share, sanitize_overlay_html

    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    kind = str(body.get("kind") or "").strip().lower()
    ns = str(body.get("ns") or "wrapped").strip() or "wrapped"
    overlay_html = body.get("overlay_html") or ""
    share_data = body.get("share_data") or {}

    if kind not in _VALID_KINDS:
        return jsonify({"error": "kind must be 'season' or 'weekly'"}), 400
    if not isinstance(overlay_html, str) or "wrapped-slide" not in overlay_html:
        return jsonify({"error": "overlay_html missing or invalid"}), 400
    # The overlay is untrusted client HTML rendered verbatim on a public page:
    # strip scripts, event handlers, and unsafe URLs before storing.
    overlay_html = sanitize_overlay_html(overlay_html)
    if "wrapped-slide" not in overlay_html:
        return jsonify({"error": "overlay_html missing or invalid"}), 400
    if len(overlay_html.encode("utf-8")) > MAX_OVERLAY_BYTES:
        return jsonify({"error": "Deck too large"}), 413
    if not isinstance(share_data, dict):
        return jsonify({"error": "share_data must be an object"}), 400
    import json as _json
    if len(_json.dumps(share_data).encode("utf-8")) > MAX_SHARE_DATA_BYTES:
        return jsonify({"error": "Share data too large"}), 413
    if len(ns) > 64 or not ns.replace("-", "").replace("_", "").isalnum():
        return jsonify({"error": "Invalid ns"}), 400

    label = _share_label(kind, share_data)
    try:
        token = create_wrapped_share(
            kind=kind, ns=ns, overlay_html=overlay_html,
            share_data=share_data, label=label,
        )
    except Exception:
        logger.exception("[api_wrapped_share_create] store failed")
        return jsonify({"error": "Could not create link"}), 500

    url = f"{request.host_url.rstrip('/')}/wrapped/{token}"
    return jsonify({"url": url})


@history_bp.route("/wrapped/<token>")
def wrapped_share_view(token: str):
    """Public no-auth viewer for a shared Wrapped deck. 404 when unknown/expired."""
    from dashboard_services.pages.history_page import render_wrapped_share_page
    from dashboard_services.wrapped_shares import get_wrapped_share

    share = get_wrapped_share(token)
    if not share:
        return (
            "<!DOCTYPE html><html><head><title>Link expired</title>"
            '<meta name="robots" content="noindex, nofollow"></head>'
            "<body style='background:#0b0b16;color:#fff;font-family:sans-serif;"
            "display:flex;align-items:center;justify-content:center;height:100vh;"
            "margin:0'><p>This Wrapped link has expired or doesn't exist.</p>"
            "</body></html>",
            404,
        )

    share_data = share.get("share_data") or {}
    if isinstance(share_data, str):
        import json as _json
        try:
            share_data = _json.loads(share_data)
        except Exception:
            share_data = {}

    html = render_wrapped_share_page(
        overlay_html=share.get("overlay_html") or "",
        share_data=share_data,
        label=share.get("label") or "Fantasy Wrapped",
        ns=share.get("ns") or "wrapped",
        css_url="/static/dashboard.css",
        logo_url=f"{request.host_url.rstrip('/')}/static/BR_Logo_dark.png",
    )
    from flask import Response
    return Response(html, mimetype="text/html")


# ────────────────────────────────────────────────────────────────────────────
# Merged from routes/historical_api_bp.py: Historical deep-panel API (JSON lookup, no parquet).
# ────────────────────────────────────────────────────────────────────────────

@history_bp.route("/api/historical-player/<player_id>")
def api_historical_player(player_id: str):
    """Lazy deep panel: named comps and rates from precomputed JSON leaves."""
    aggs = load_profile_aggregates()
    if not aggs:
        return jsonify({"available": False, "player_id": str(player_id or "")})
    extra = {}
    adp = request.args.get("adp")
    if adp not in (None, ""):
        extra["adp"] = adp
    redraft = request.args.get("redraft_avg_pick")
    if redraft not in (None, ""):
        extra["redraft_avg_pick"] = redraft
    pos = request.args.get("position")
    if pos:
        extra["position"] = pos
    roster = request.args.get("roster_spot")
    if roster not in (None, ""):
        extra["roster_spot"] = roster
    proj = request.args.get("proj_ppg")
    if proj not in (None, ""):
        extra["proj_ppg"] = proj
        extra["projected_ppg"] = proj
    proj_rk = request.args.get("proj_rk")
    if proj_rk not in (None, ""):
        extra["projected_positional_rank"] = proj_rk
        extra["proj_rk"] = proj_rk
    adp_rk = request.args.get("adp_rk")
    if adp_rk not in (None, ""):
        extra["adp_positional_rank"] = adp_rk
        extra["adp_rk"] = adp_rk
    try:
        payload = build_deep_panel(player_id, aggs, extra=extra or None)
        return jsonify(payload)
    except Exception:
        logger.exception("[historical-player] %s failed", player_id)
        return jsonify({"available": False, "player_id": str(player_id or "")})


@history_bp.route("/api/historical-trends")
def api_historical_trends():
    """Position-level historical trend tables for the cheat-sheet Trends tab."""
    aggs = load_profile_aggregates()
    if not aggs:
        return jsonify({"available": False, "descriptive_only": True, "not_in_ranking": True})
    try:
        payload = build_historical_trends(aggs)
        return jsonify(payload)
    except Exception:
        logger.exception("[historical-trends] failed")
        return jsonify({"available": False, "descriptive_only": True, "not_in_ranking": True})


@history_bp.route("/api/historical-cohort", methods=["POST"])
def api_historical_cohort():
    """Combined historical hit rate for selected Trends buckets. JSON index only."""
    aggs = load_profile_aggregates()
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        body = {}
    pos = body.get("position")
    filters = body.get("filters") or []
    if not isinstance(filters, list):
        filters = []
    tier = body.get("tier") or "top_12"
    if not aggs:
        return jsonify({
            "available": False,
            "descriptive_only": True,
            "not_in_ranking": True,
            "not_in_pick_score": True,
            "position": pos,
            "unknown_reason": "aggregates_missing",
        })
    try:
        payload = evaluate_cohort(
            aggs,
            position=pos,
            filters=filters,
            tier=tier,
            data_version=aggregates_version(),
        )
        # Scout matches use the same Python predicates as the cohort index.
        # board_features is request-scoped and must not enter the cohort cache.
        board_features = body.get("board_features")
        if isinstance(board_features, dict) and filters:
            payload = dict(payload)
            payload["scout_matches"] = scout_matching_players(board_features, filters)
        return jsonify(payload)
    except Exception:
        logger.exception("[historical-cohort] failed")
        return jsonify({
            "available": False,
            "descriptive_only": True,
            "not_in_ranking": True,
            "not_in_pick_score": True,
            "position": pos,
            "unknown_reason": "error",
        })

