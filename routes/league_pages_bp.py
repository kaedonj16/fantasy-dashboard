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
circular import -- the real functions are only fetched when a request is served.
Page builders that have already been extracted into service modules are imported
directly.
"""
from __future__ import annotations

import html
import logging
import time
from datetime import datetime

from flask import Blueprint, abort, jsonify, redirect, request, session, url_for

from dashboard_services.pages.autopilot_page import build_autopilot_body
from dashboard_services.pages.waivers_page import build_waivers_body

logger = logging.getLogger(__name__)

league_pages_bp = Blueprint("league_pages", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ─────────────────

def render_page(*args, **kwargs):
    from app import render_page as _fn
    return _fn(*args, **kwargs)

def get_league_ctx_from_cache(*args, **kwargs):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*args, **kwargs)

def get_league(*args, **kwargs):
    from app import get_league as _fn
    return _fn(*args, **kwargs)

def build_recap_body(*args, **kwargs):
    from app import build_recap_body as _fn
    return _fn(*args, **kwargs)

def _fill_activity_section_async(*args, **kwargs):
    from app import _fill_activity_section_async as _fn
    return _fn(*args, **kwargs)


def ensure_weekly_bits(*args, **kwargs):
    from app import ensure_weekly_bits as _fn
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

def get_page_html_from_cache(*args, **kwargs):
    from app import get_page_html_from_cache as _fn
    return _fn(*args, **kwargs)

def store_page_html(*args, **kwargs):
    from app import store_page_html as _fn
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

def _standings_available_weeks(*args, **kwargs):
    from app import _standings_available_weeks as _fn
    return _fn(*args, **kwargs)

def _standings_panels(*args, **kwargs):
    from app import _standings_panels as _fn
    return _fn(*args, **kwargs)

def _use_offseason_standings(ctx) -> bool:
    from app import _use_offseason_standings as _fn
    return _fn(ctx)


@league_pages_bp.route("/<platform>/<int:season>/<league_id>/standings")
def page_standings(platform: str, season: int, league_id: str):
    # The standings body carries no viewer-valued data (the one viewer-gated
    # piece, the playoff-rules confirm note, is rendered hidden and revealed
    # client-side from window._viewerRid), so it is safe to cache per league
    # and serve through the Teams/Activity cached-or-background pattern: a
    # warm body serves instantly, a cold one builds in the background while
    # the client polls a skeleton.
    def _build() -> str:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        # Before any games are on this week's slate the in-season standings
        # are all 0-0, so show the populated value-based layout. It switches
        # to in-season automatically once a regular-season game is scheduled
        # this NFL week, not only after the first week finalizes.
        if _use_offseason_standings(ctx):
            return _build_offseason_standings_body(ctx)
        return build_standings_body(ctx)

    # Warm tier (the Teams pattern the cached-or-background helper lacks):
    # when a ctx is already available without a build (local cache, Redis,
    # or the stale window), the body build is cheap, so serve it inline and
    # populate the page cache instead of making the reader poll a skeleton.
    # The best-effort probe returns {} on a true cold league (and kicks the
    # background warm), which falls through to the background path below.
    if get_league_ctx_from_cache(platform, league_id, season, allow_build=False):
        body = _build()
        store_page_html(platform, season, league_id, "standings", body)
        return render_page("BR Fantasy Standings", league_id, "standings", body, platform, season)

    return _serve_cached_or_background(
        platform, season, league_id, "standings",
        "BR Fantasy Standings", "standings", _build, "Loading standings")


@league_pages_bp.route("/api/standings-week")
def api_standings_week():
    """Re-render the standings/power panels as they stood through a
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
        # Same split the weekly recap follows, so the same week reads the
        # same everywhere: the latest finalized week is the standings
        # page's default load (live blended list, with last week's blended
        # list as the movement baseline); older weeks rank by the
        # performance PowerScore in the capped team_stats.
        _weeks = _standings_available_weeks(ctx)
        _power_rankings = None
        _prev_power_ranks = None
        if _weeks and week >= _weeks[-1]:
            from dashboard_services.ai.context_builders import (
                build_power_rankings_context,
            )
            _power_rankings = (
                build_power_rankings_context(capped) or {}
            ).get("teams") or []
            if len(_weeks) >= 2:
                _prev_capped = build_standings_as_of_week(ctx, _weeks[-2])
                _prev_teams = (
                    build_power_rankings_context(_prev_capped) or {}
                ).get("teams") or []
                _prev_power_ranks = {
                    str(t.get("roster_id")): int(t.get("rank") or (i + 1))
                    for i, t in enumerate(_prev_teams)
                    if t.get("roster_id") is not None
                }
        panels = _standings_panels(
            capped, power_rankings=_power_rankings,
            prev_power_ranks=_prev_power_ranks)
        return jsonify({
            "ok": True,
            "week": week,
            "standings_html": panels["standings"],
            "power_html": panels["power"],
            "tiles_html": panels.get("tiles", ""),
            "shares_html": panels["shares"],
        })
    except Exception as e:
        logger.warning("[standings-week] render failed: %s", e, exc_info=True)
        return jsonify({"ok": False, "error": "render failed"}), 500


@league_pages_bp.route("/<platform>/<int:season>/<league_id>/waivers")
def page_waivers(platform: str, season: int, league_id: str):
    # The waivers body is a league-generic shell (all data loads client-side
    # per viewer); the only ctx read in the builder is the is_best_ball
    # boolean. Serve the cached shell when present, and on a miss use a
    # best-effort ctx so a cold league never blocks this route on a full
    # context build: allow_build=False serves a warm/stale ctx (or {} while
    # a background warm runs). With {} the Start/Sit tab shows until the
    # client learns the format, the accepted cold best-ball flash.
    cached = get_page_html_from_cache(platform, season, league_id, "waivers")
    if cached:
        return render_page("BR Fantasy Waivers", league_id, "waivers", cached, platform, season)
    ctx = get_league_ctx_from_cache(platform, league_id, season, allow_build=False)
    if not ctx:
        # No usable ctx and no cached shell: this league has never loaded
        # here, so serving a blind shell would also mask a league that
        # cannot load at all. One light league fetch (the context build's
        # own first call) separates cold-but-loadable from provider-down:
        # an outage raises ProviderUnavailableError, which the app
        # errorhandler maps to the branded 503 this page has always
        # returned on provider failure. A healthy fetch also supplies the
        # league dict the shell reads for is_best_ball, so the format is
        # known before the shell is cached below.
        _league = get_league(platform, league_id, season)
        if _league:
            ctx = {"league": _league}
    body = build_waivers_body(platform, season, league_id, ctx)
    if ctx:
        # Only cache once the format is known (a warm ctx), so a best-ball
        # league's shell is never cached with the Start/Sit tab visible.
        store_page_html(platform, season, league_id, "waivers", body)
    return render_page("BR Fantasy Waivers", league_id, "waivers", body, platform, season)


@league_pages_bp.route("/<platform>/<int:season>/<league_id>/autopilot")
def page_autopilot(platform: str, season: int, league_id: str):
    # Real Phase 1 page: stored settings, the live plan, the real
    # roster, and the audit log for the acting account's league. Gated
    # by autopilot_access (pilot + PRO + usable ESPN league); everyone
    # and everywhere else gets a plain 404 so the surface does not
    # exist publicly.
    from dashboard_services import autopilot_store as store
    from dashboard_services.autopilot_preview import autopilot_access

    account_id = session.get("account_id")
    if not account_id or not autopilot_access(account_id, platform, league_id, season):
        abort(404)
    ctx = get_league_ctx_from_cache(platform, league_id, season, allow_build=False) or {}
    data: dict = {
        "settings": store.get_settings(account_id, platform, league_id, season),
        "kill_switch": store.get_kill_switch(account_id),
        "audit": store.list_audit(account_id, platform, league_id, season, limit=25),
        "roster": [],
        "plan": None,
        "plan_status": "",
        "plan_error": None,
    }
    settings = data["settings"]
    # Fresh roster read (best effort): feeds the protected-players
    # editor and the team identity even when no plan is built.
    state = None
    try:
        from dashboard_services.providers import espn_writes

        state = espn_writes.read_lineup_state(account_id, league_id, season)
        data["roster"] = state["entries"]
        data["team_name"] = state["team_name"]
        data["scoring_period"] = state["scoring_period_id"]
    except Exception as exc:
        data["plan_error"] = str(exc)
    # Plan: reuse the open plan, or build one on load when the league
    # is on and runnable (plan Step 3 trigger). The builder persists
    # and audits; a failed build surfaces honestly on the page.
    if (
        settings["mode"] != "off"
        and not settings["paused"]
        and not data["kill_switch"]
        and data["plan_error"] is None
    ):
        open_plan = store.get_open_plan(account_id, platform, league_id, season)
        if open_plan is not None:
            data["plan"] = open_plan
            data["plan_status"] = "planned" if open_plan.get("swaps") else "no_moves"
            data["scoring_period"] = open_plan["scoring_period"]
        else:
            try:
                from dashboard_services import autopilot_plan

                built = autopilot_plan.build_plan(
                    account_id, platform, league_id, season
                )
                data["plan_status"] = built.get("status") or ""
                if built.get("plan"):
                    data["plan"] = built["plan"]
                    data["scoring_period"] = built["plan"]["scoring_period"]
            except Exception as exc:
                logger.warning("autopilot page plan build failed", exc_info=True)
                data["plan_error"] = str(exc)
    body = build_autopilot_body(platform, season, league_id, ctx, data)
    return render_page("BR Fantasy Autopilot", league_id, "autopilot", body, platform, season)


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
    try:
        week = int(request.args.get("week") or 0) or None
    except (ValueError, TypeError):
        week = None

    def _build() -> str:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        # Reuse the same normalized historical lineups and authoritative
        # scoring as Matchups. This cached league-level hydrate avoids
        # per-team/player calls.
        ensure_weekly_bits(ctx)
        # Generic body: no session reads, no viewer data, no premium AI
        # column, no inline activity sweep (see build_recap_body). The
        # per-viewer pieces are filled client-side from the section APIs
        # below, so this body is safe to cache per league.
        return build_recap_body(ctx, selected_week=week, generic=True)

    # League name for the OG tags only: best-effort, never a blocking build.
    ctx = get_league_ctx_from_cache(platform, league_id, season, allow_build=False)
    league_name = html.escape((ctx.get("league") or {}).get("name") or "Fantasy League")
    week_label = f"Week {week} Recap" if week else "Weekly Recap"
    page_url = request.url
    base_url = request.host_url.rstrip("/")
    week_param = f"?week={week}" if week else ""
    og_image = f"{base_url}/{platform}/{season}/{league_id}/recap/og.png{week_param}"
    if week_param:
        og_image += f"&v={int(time.time()) // 3600}"
    else:
        og_image += f"?v={int(time.time()) // 3600}"
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
    return _serve_cached_or_background(
        platform, season, league_id, f"recap:{week or 'latest'}",
        "Weekly Recap", "recap", _build, "Loading weekly recap",
        og_tags=og_tags)


def _recap_snapshot() -> dict:
    """Session values the recap builder needs, captured on the request
    thread (the builder itself never reads the session)."""
    return {
        "viewer_roster_id": session.get("viewer_roster_id"),
        "viewer_username": session.get("viewer_username"),
        "viewer_user_id": session.get("viewer_user_id"),
    }


def _recap_section_params():
    platform = request.args.get("platform", "sleeper")
    league_id = request.args.get("league_id", "")
    try:
        season = int(request.args.get("season", 0))
        week = int(request.args.get("week") or 0) or None
    except (TypeError, ValueError):
        return None
    if not league_id or not season:
        return None
    return platform, season, league_id, week


def _extract_marked(body: str, name: str) -> str:
    start = f"<!--recap-{name}-start-->"
    end = f"<!--recap-{name}-end-->"
    if start not in body or end not in body:
        return ""
    return body.split(start, 1)[1].split(end, 1)[0]


def _private_json(payload, status=200):
    resp = jsonify(payload)
    resp.status_code = status
    # Per-viewer section HTML: never store it in a shared or league cache.
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


@league_pages_bp.route("/api/recap-ai-column")
def api_recap_ai_column():
    """Per-viewer Weekly Story + Up Next sections for the cached recap body.

    Cache-only in spirit: a cold league ctx answers pending (the getter's
    best-effort path kicks a background warm) instead of blocking on a
    build. The response is built for this request's viewer (premium gate
    from the session snapshot) and is never cached league-keyed.
    """
    params = _recap_section_params()
    if params is None:
        return _private_json({"ready": False, "error": "bad params"}, 400)
    platform, season, league_id, week = params
    ctx = get_league_ctx_from_cache(platform, league_id, season, allow_build=False)
    if not ctx:
        return _private_json({"ready": False, "pending": True})
    try:
        ensure_weekly_bits(ctx)
        # defer_activity: this endpoint only extracts the story sections, so
        # the builder must not pay the transactions sweep for them.
        body = build_recap_body(ctx, selected_week=week,
                                viewer_snapshot=_recap_snapshot(),
                                defer_activity=True)
    except Exception as e:
        logger.warning("[recap-ai-column] render failed: %s", e, exc_info=True)
        return _private_json({"ready": False, "error": "render failed"}, 500)
    return _private_json({
        "ready": True,
        "story_html": _extract_marked(body, "ai"),
        "upnext_html": _extract_marked(body, "upnext"),
    })


@league_pages_bp.route("/api/recap-activity")
def api_recap_activity():
    """League Activity section for the cached recap body.

    While the ctx's deferred activity sweep is still pending, kick the
    bounded background fill (the same helper since-last-visit uses) and
    answer pending; once it lands, extract the rendered section from a
    viewer-built body. The section itself is league-generic, but it is
    served per request and never stored in a league-keyed cache.
    """
    params = _recap_section_params()
    if params is None:
        return _private_json({"ready": False, "error": "bad params"}, 400)
    platform, season, league_id, week = params
    ctx = get_league_ctx_from_cache(platform, league_id, season, allow_build=False)
    if not ctx:
        return _private_json({"ready": False, "pending": True})
    if ctx.get("activity_df") is None:
        _fill_activity_section_async(platform, league_id, season)
        return _private_json({"ready": False, "pending": True})
    try:
        ensure_weekly_bits(ctx)
        body = build_recap_body(ctx, selected_week=week,
                                viewer_snapshot=_recap_snapshot())
    except Exception as e:
        logger.warning("[recap-activity] render failed: %s", e, exc_info=True)
        return _private_json({"ready": False, "error": "render failed"}, 500)
    return _private_json({
        "ready": True,
        "html": _extract_marked(body, "activity"),
    })


# ── Advanced metrics leaderboard ──────────────────────────────────────────────

@league_pages_bp.route("/metrics")
@league_pages_bp.route("/<platform>/<int:season>/<league_id>/metrics")
def page_advanced_metrics(platform: str = None, season: int = None, league_id: str = None):
    """Premium Advanced Metrics leaderboard page."""
    from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
    from data_building.advanced_metrics import LEADERBOARD_METRICS
    from dashboard_services.subscriptions import has_premium_for_viewer
    has_premium = has_premium_for_viewer(
        session.get("viewer_username"), session.get("viewer_user_id"),
        league_id, platform or "sleeper", season,
    )
    body = build_advanced_metrics_body(
        has_premium, LEADERBOARD_METRICS, league_id, season, platform,
        is_guest=not session.get("viewer_username"),
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
            # League-free shared links (the /metrics route) get a league-free
            # preview image; the league-scoped path keeps its own so the
            # screenshot renders with that league's context.
            if platform and season and league_id:
                og_img = f"{origin}/{platform}/{season}/{league_id}/metrics/og.png?{_ue(_og_qs)}"
            else:
                og_img = f"{origin}/metrics/og.png?{_ue(_og_qs)}"
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
    # Guests get the lite shell: the body is server-rendered with its own
    # inline styles and a self-contained inline script (every bundle global
    # it touches is typeof-guarded, and the player modal force-loads the
    # features bundle on demand), so it paints from seo_lite.css instead of
    # splash-gating on the full dashboard bundle. Signed-in viewers are
    # unaffected (render_page only goes lite for logged-out sessions).
    return render_page("Advanced Metrics", league_id, "advanced-metrics", body, platform, season, og_tags=og_tags, lite_js=True)

# ── NFL Teams (public research page; league variant keeps nav context) ────────

@league_pages_bp.route("/nfl-teams")
@league_pages_bp.route("/<platform>/<int:season>/<league_id>/nfl-teams")
def page_nfl_teams(platform: str = None, season: int = None, league_id: str = None):
    """Public NFL team rankings page. No league connection required."""
    from dashboard_services.pages.nfl_teams_page import build_nfl_teams_body
    from app import _list_team_tab_seasons, get_nfl_state

    try:
        cur = int((get_nfl_state() or {}).get("season") or 0)
    except Exception:
        cur = 0
    import datetime as _dt
    if not cur:
        cur = _dt.datetime.now().year
    available = _list_team_tab_seasons(cur)
    try:
        want = int(request.args.get("season") or (season or cur))
    except (TypeError, ValueError):
        want = season or cur
    if want not in available:
        want = available[0] if available else cur
    team = "".join(ch for ch in str(request.args.get("team") or "").upper() if ch.isalpha())[:3]
    view = str(request.args.get("view") or "overview")
    body = build_nfl_teams_body(want, team=team, view=view, available_seasons=available)
    return render_page(
        "NFL Teams", league_id, "nfl-teams", body, platform, season,
        description="NFL team rankings: real points per game, per-game offense splits, O-line ratings, depth charts, and schedules.",
        canonical="/nfl-teams",
        # Guests get the lite shell (see page_advanced_metrics): the body is
        # server-rendered, inline-styled, and its inline script only touches
        # bundle globals behind typeof guards.
        lite_js=True,
    )

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

    # Demo/mock preview only (?tour=1). Live site-tour resume uses ?tour_step=N
    # and must hit the real league path below -- do not conflate the two.
    if request.args.get("tour") and not request.args.get("tour_step"):
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
    tab = str(request.args.get("tab", "perf")).lower()
    if tab not in ("perf", "value", "trends"):
        tab = "perf"

    # Resolve the /graphs URL here, in the request context, so the career
    # background build never has to call url_for off the request thread. The page
    # body is built by dashboard_services.pages.graphs_page, which takes every
    # app-level accessor as a parameter (see render_graphs_html).
    graphs_base_url = url_for("league_pages.page_graphs", platform=platform, season=season, league_id=league_id)

    # The career body is viewer-personalized (the viewer's team is
    # highlighted in every chart and the insights read "You rank ..."), but
    # it is built in a background thread with no session and cached under a
    # league-only key (Class A audit) -- so the cached copy baked an empty
    # viewer and served it to everyone. Capture this request's viewer here,
    # re-apply it inside _render, and scope the cache key by it. v3
    # invalidates the v2 entries baked without a viewer.
    viewer_snapshot = dict(ctx.get("viewer") or {})
    viewer_rid = str(viewer_snapshot.get("viewer_roster_id") or "")

    def _render(v: str, m: str, t: str) -> str:
        render_ctx = get_league_ctx_from_cache(platform, league_id, season)
        if viewer_snapshot:
            # Shallow copy: never mutate the shared cached ctx from a
            # background thread with one request's viewer.
            render_ctx = {**render_ctx, "viewer": viewer_snapshot}
        return render_graphs_html(
            platform, season, league_id, v, m,
            ctx=render_ctx,
            available_seasons=get_available_history_seasons(platform, league_id, season),
            get_ctx=get_league_ctx_from_cache,
            model_value_table=get_model_value_table_cached() or [],
            graphs_base_url=graphs_base_url,
            tab=t,
        )

    # Career view aggregates every past season (slow on a cold cache) -> build in
    # the background and show a skeleton.
    if view == "career":
        return _serve_cached_or_background(
            # v2 invalidates HTML built before career rows were keyed by stable
            # provider owner ids rather than mutable team display names.
            platform, season, league_id, f"graphs:career:v3:{members}:vwr{viewer_rid or 'anon'}",
            "BR Fantasy Graphs", "graphs",
            lambda: _render("career", members, tab),
            "Building career graphs",
        )

    # Season views build all three tab panes (18 card builders) on every
    # load, so they take the same shared page-HTML cache as career: same
    # TTL and league-bust invalidation, keyed on the viewed season + tab +
    # viewer (the body highlights the viewer's team). ``members`` has no
    # effect on season HTML (career-only filter), so it stays out of the
    # key; no finer data stamp exists at this layer, so freshness rides
    # the shared page-cache TTL/bust exactly like the career view.
    cache_view = view if view.isdigit() else str(season)
    return _serve_cached_or_background(
        platform, season, league_id,
        f"graphs:season:v1:{cache_view}:{tab}:vwr{viewer_rid or 'anon'}",
        "BR Fantasy Graphs", "graphs",
        lambda: _render(view, members, tab),
        "Building season graphs",
    )


# ── League history ────────────────────────────────────────────────────────────

@league_pages_bp.route("/<platform>/<int:season>/<league_id>/history")
def page_history(platform: str, season: int, league_id: str):
    from dashboard_services.pages.history_page import (
        build_history_body, build_tour_mock_history_ctx)
    from dashboard_services.api import resolve_league_id_for_season
    from utils.history_seasons import get_default_history_season
    # Demo/mock preview only (?tour=1). Live site-tour resume uses ?tour_step=N
    # and must hit the real league path below -- do not conflate the two.
    if request.args.get("tour") and not request.args.get("tour_step"):
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
