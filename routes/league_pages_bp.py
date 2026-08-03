"""
League info pages: standings, waivers, weekly recap, and the scout redirect.

Routes:
    /<platform>/<season>/<league_id>/standings   (page_standings)
    /api/standings-week                           (api_standings_week)
    /<platform>/<season>/<league_id>/waivers      (page_waivers)
    /<platform>/<season>/<league_id>/recap        (page_recap)
    /<platform>/<season>/<league_id>/scout        (redirect to the Matchups scout tab)

Extracted from app.py to reduce monolith size. App.py internals are reached via
the lazy shims below so importing the blueprint at start-up stays free of a
circular import — the real functions are only fetched when a request is served.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime

from flask import Blueprint, jsonify, redirect, request, session, url_for

logger = logging.getLogger(__name__)

league_pages_bp = Blueprint("league_pages", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ─────────────────

def render_page(*args, **kwargs):
    from app import render_page as _fn
    return _fn(*args, **kwargs)

def get_league_ctx_from_cache(*args, **kwargs):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*args, **kwargs)

def build_recap_body(*args, **kwargs):
    from app import build_recap_body as _fn
    return _fn(*args, **kwargs)

def build_commissioner_body(*args, **kwargs):
    from app import build_commissioner_body as _fn
    return _fn(*args, **kwargs)

def get_available_history_seasons(*args, **kwargs):
    from app import get_available_history_seasons as _fn
    return _fn(*args, **kwargs)

def get_model_value_table_cached(*args, **kwargs):
    from app import get_model_value_table_cached as _fn
    return _fn(*args, **kwargs)

def _serve_cached_or_background(*args, **kwargs):
    from app import _serve_cached_or_background as _fn
    return _fn(*args, **kwargs)

def _build_tour_mock_df_weekly(*args, **kwargs):
    from app import _build_tour_mock_df_weekly as _fn
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


# ── Weekly recap ──────────────────────────────────────────────────────────────

@league_pages_bp.route("/<platform>/<int:season>/<league_id>/recap")
def page_recap(platform: str, season: int, league_id: str):
    ctx = get_league_ctx_from_cache(platform, league_id, season)
    try:
        week = int(request.args.get("week") or 0) or None
    except (ValueError, TypeError):
        week = None
    body = build_recap_body(ctx, selected_week=week)
    league_name = html.escape((ctx.get("league") or {}).get("name") or "Fantasy League")
    week_label = f"Week {week} Recap" if week else "Weekly Recap"
    page_url = request.url
    base_url = request.host_url.rstrip("/")
    week_param = f"?week={week}" if week else ""
    og_image = f"{base_url}/{platform}/{season}/{league_id}/recap/og.png{week_param}"
    og_tags = (
        f"<meta property='og:title' content='{week_label} - {league_name} | BR Fantasy'>"
        f"<meta property='og:description' content='Weekly fantasy football recap: scoreboard, highlights, and AI analysis.'>"
        f"<meta property='og:image' content='{og_image}'>"
        f"<meta property='og:image:width' content='1200'>"
        f"<meta property='og:image:height' content='630'>"
        f"<meta property='og:type' content='website'>"
        f"<meta property='og:url' content='{html.escape(page_url)}'>"
        f"<meta name='twitter:card' content='summary_large_image'>"
        f"<meta name='twitter:title' content='{week_label} - {league_name} | BR Fantasy'>"
        f"<meta name='twitter:description' content='Weekly fantasy football recap: scoreboard, highlights, and AI analysis.'>"
        f"<meta name='twitter:image' content='{og_image}'>"
    )
    return render_page("Weekly Recap", league_id, "recap", body, platform, season, og_tags=og_tags)


# ── Advanced metrics leaderboard ──────────────────────────────────────────────

@league_pages_bp.route("/metrics")
@league_pages_bp.route("/<platform>/<int:season>/<league_id>/metrics")
def page_advanced_metrics(platform: str = None, season: int = None, league_id: str = None):
    """Premium Advanced Metrics leaderboard page."""
    from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
    from data_building.advanced_metrics import LEADERBOARD_METRICS
    user_id = session.get("viewer_username")
    body = build_advanced_metrics_body(
        True, LEADERBOARD_METRICS, league_id, season, platform
    )
    # When a shared graph link is opened (?graph=1&gx=&gy=...), give it a rich
    # social preview whose image is a server-rendered screenshot of that graph.
    og_tags = ""
    if request.args.get("graph") == "1":
        gx = request.args.get("gx") or ""
        gy = request.args.get("gy") or ""
        def _mlabel(k):
            m = LEADERBOARD_METRICS.get(k) or {}
            return m.get("label") or k
        if gx and gy:
            origin = request.host_url.rstrip("/")
            from urllib.parse import urlencode as _ue
            _og_qs = {k: request.args.get(k) for k in ("gx", "gy", "gz", "gn", "season", "metric", "pos", "minvol")
                      if request.args.get(k)}
            og_img = f"{origin}/{platform}/{season}/{league_id}/metrics/og.png?{_ue(_og_qs)}"
            og_title = f"{_mlabel(gy)} vs {_mlabel(gx)} | BR Fantasy"
            og_desc = "Advanced metrics scatter: compare efficiency and opportunity across the league."
            t = html.escape(og_title, quote=True)
            d = html.escape(og_desc, quote=True)
            img = html.escape(og_img, quote=True)
            url = html.escape(request.url, quote=True)
            og_tags = (
                f"<meta property=\"og:site_name\" content=\"BR Fantasy\">"
                f"<meta property=\"og:type\" content=\"website\">"
                f"<meta property=\"og:title\" content=\"{t}\">"
                f"<meta property=\"og:description\" content=\"{d}\">"
                f"<meta property=\"og:url\" content=\"{url}\">"
                f"<meta property=\"og:image\" content=\"{img}\">"
                f"<meta property=\"og:image:width\" content=\"1200\">"
                f"<meta property=\"og:image:height\" content=\"630\">"
                f"<meta name=\"twitter:card\" content=\"summary_large_image\">"
                f"<meta name=\"twitter:title\" content=\"{t}\">"
                f"<meta name=\"twitter:description\" content=\"{d}\">"
                f"<meta name=\"twitter:image\" content=\"{img}\">"
            )
    return render_page("Advanced Metrics", league_id, "advanced-metrics", body, platform, season, og_tags=og_tags)

# ── Optimal lineup (redirect to the Matchups tab) ─────────────────────────────

@league_pages_bp.route("/<platform>/<int:season>/<league_id>/optimal")
def page_optimal(platform: str, season: int, league_id: str):
    # Optimal Lineup lives as a tab on the Matchups page, not its own page.
    # Redirect (keeps old links/bookmarks working) to that tab.
    return redirect(
        url_for("page_weekly", platform=platform, season=season, league_id=league_id)
        + "?tab=optimal"
    )

# ── League health / commissioner ──────────────────────────────────────────────

@league_pages_bp.route("/<platform>/<int:season>/<league_id>/league_health")
@league_pages_bp.route("/<platform>/<int:season>/<league_id>/commissioner")  # legacy redirect
def page_commissioner(platform: str, season: int, league_id: str):
    ctx = get_league_ctx_from_cache(platform, league_id, season)
    body = build_commissioner_body(ctx)
    return render_page("League Health", league_id, "league_health", body, platform, season)


# ── Graphs (league value/performance charts) ──────────────────────────────────

@league_pages_bp.route("/<platform>/<int:season>/<league_id>/graphs")
def page_graphs(platform: str, season: int, league_id: str):
    from dashboard_services.pages.graphs_page import (
        build_graphs_body, build_tour_mock_graphs_ctx, render_graphs_html)

    # Tour preview: render with mock data, bypass real league fetch
    if request.args.get("tour"):
        try:
            mock_ctx = build_tour_mock_graphs_ctx(_build_tour_mock_df_weekly())
            body_html = build_graphs_body(mock_ctx)
        except Exception as exc:
            body_html = (
                f"<div class='card central'><div class='card-body'>"
                f"<p>Graphs preview unavailable: {exc}</p></div></div>"
            )
        return render_page("BR Fantasy Graphs", league_id, "graphs", body_html, platform, season)

    # Determining the default view needs the (shared) league context; the heavy,
    # graphs-specific work is the career aggregation, deferred below.
    ctx = get_league_ctx_from_cache(platform, league_id, season)
    default_view = "career" if bool(ctx.get("offseason_mode")) else str(season)
    view = request.args.get("view", default_view)
    members = "all" if str(request.args.get("members", "current")).lower() == "all" else "current"

    # Resolve the /graphs URL here, in the request context, so the career
    # background build never has to call url_for off the request thread. The page
    # body is built by dashboard_services.pages.graphs_page, which takes every
    # app-level accessor as a parameter (see render_graphs_html).
    graphs_base_url = url_for("league_pages.page_graphs", platform=platform, season=season, league_id=league_id)

    def _render(v: str, m: str) -> str:
        return render_graphs_html(
            platform, season, league_id, v, m,
            ctx=get_league_ctx_from_cache(platform, league_id, season),
            available_seasons=get_available_history_seasons(platform, league_id, season),
            get_ctx=get_league_ctx_from_cache,
            model_value_table=get_model_value_table_cached() or [],
            graphs_base_url=graphs_base_url,
        )

    # Career view aggregates every past season (slow on a cold cache) -> build in
    # the background and show a skeleton. Season views are a single, light league
    # context, so render them inline.
    if view == "career":
        return _serve_cached_or_background(
            platform, season, league_id, f"graphs:career:{members}",
            "BR Fantasy Graphs", "graphs",
            lambda: _render("career", members),
            "Building career graphs",
        )

    body_html = _render(view, members)
    return render_page("BR Fantasy Graphs", league_id, "graphs", body_html, platform, season)


# ── League history ────────────────────────────────────────────────────────────

@league_pages_bp.route("/<platform>/<int:season>/<league_id>/history")
def page_history(platform: str, season: int, league_id: str):
    from dashboard_services.pages.history_page import (
        build_history_body, build_tour_mock_history_ctx)
    from dashboard_services.api import resolve_league_id_for_season
    from utils.history_seasons import get_default_history_season
    # Tour preview: render with mock data, bypass real league fetch
    if request.args.get("tour"):
        try:
            mock_ctx = build_tour_mock_history_ctx(_build_tour_mock_df_weekly())
            body_html = build_history_body(
                history_ctx=mock_ctx,
                available_seasons=[datetime.now().year - 1],
                base_platform=platform,
                base_season=season,
                base_league_id=league_id,
                selected_history_season=datetime.now().year - 1,
                resolved_history_league_id="tour_mock",
            )
        except Exception as exc:
            body_html = f"<div class='card central'><div class='card-body'><p>History preview unavailable: {exc}</p></div></div>"
        return render_page("League History", league_id, "history", body_html, platform, season)

    # Captured here (request thread); the background build must not touch request.
    selected_history_season_param = request.args.get("history_season")
    page_cache_key = f"history:{selected_history_season_param}" if selected_history_season_param else "history"

    def _build() -> str:
        available_seasons = get_available_history_seasons(platform, league_id, season)

        # First-year league case
        if not available_seasons:
            return """
        <div class="card central">
          <div class="card-body">
            <div class="bract-empty-state">
              <div class="bract-empty-title">Welcome to Your First Season!</div>
              <div class="bract-empty-copy">This is the first year of your league. Historical season data, AI-powered recaps, and year-over-year comparisons will appear here after your current season completes. Check back after championship week!</div>
            </div>
          </div>
        </div>
        """

        default_history_season = get_default_history_season(available_seasons, season)
        selected_history_season = int(selected_history_season_param or default_history_season)
        if selected_history_season not in available_seasons:
            selected_history_season = default_history_season

        resolved_history_league_id = resolve_league_id_for_season(
            platform=platform,
            league_id=league_id,
            current_season=season,
            target_season=selected_history_season,
        )
        history_ctx = get_league_ctx_from_cache(
            platform, resolved_history_league_id, selected_history_season,
        )
        return build_history_body(
            history_ctx=history_ctx,
            available_seasons=available_seasons,
            base_platform=platform,
            base_season=season,
            base_league_id=league_id,
            selected_history_season=selected_history_season,
            resolved_history_league_id=resolved_history_league_id,
        )

    return _serve_cached_or_background(
        platform, season, league_id, page_cache_key,
        "League History", "history", _build, "Building league history",
    )
