import hashlib
import json
import math
import numpy as np
import os
import pandas as pd
import threading
import time
from collections import defaultdict
from datetime import date, datetime
from flask import Flask, request, render_template_string, redirect, url_for, jsonify
from pathlib import Path
from plotly.offline import plot as plotly_plot, get_plotlyjs
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

from dashboard_services.api import get_nfl_players, get_nfl_state, avatar_from_users, \
    get_nfl_scores_for_date, build_team_game_lookup, \
    get_effective_scoring_settings, get_roster_positions, get_league_settings, get_total_rosters, \
    get_sleeper_user_by_username, get_sleeper_user_leagues
from dashboard_services.awards import compute_awards_season, render_awards_section
from dashboard_services.injuries import build_injury_report, render_injury_accordion
from dashboard_services.matchups import render_matchup_slide, render_matchup_carousel_weeks, \
    compute_team_projections_for_weeks
from dashboard_services.pages.graphs_page import build_graphs_body
from dashboard_services.pages.trade_calculator_page import build_trade_calculator_body
from dashboard_services.picks import load_pick_value_table
from dashboard_services.platform_api import (
    get_league,
    get_users,
    get_rosters,
    get_traded_picks,
    get_bracket,
    get_drafts
)
from dashboard_services.player_value_history import (
    init_value_history_db,
    get_top_movers
)
from dashboard_services.players import get_players_map
from dashboard_services.service import build_tables, playoff_bracket, matchup_cards_last_week, render_top_three, \
    build_matchups_by_week, build_picks_by_roster, render_teams_sidebar, build_week_activity, pill, \
    seed_top6_from_team_stats, build_standings_map
from dashboard_services.utils import load_teams_index, streak_class, build_teams_overview, load_model_value_table, \
    load_players_index, \
    load_week_projection, bucket_for_slot, clear_activity_cache_for_league, clear_weekly_cache_for_league, \
    build_status_for_week, clear_teams_cache_for_league, get_week_projections_cached, \
    fetch_week_from_tank01, count_roster_positions, load_idp_index, get_live_game_ids_for_today, \
    build_and_save_week_stats_for_league, load_week_schedule
from data_building.build_daily_value_table import build_daily_data

daily_lock = threading.Lock()
daily_completed = None
EASTERN = ZoneInfo("America/New_York")

DASHBOARD_CACHE = {}

# How long a league context is considered fresh
CACHE_TTL = 60 * 60 * 6  # 6 hours

# How long value-table cache entries live
VALUE_CACHE_TTL = 60 * 60 * 3  # 3 hours

# How long to cache rendered page HTML (Teams, Activity, Graphs) per league
PAGE_HTML_TTL = 60  # seconds; bump if you want

daily_init_done = False
os.environ["TZ"] = "America/New_York"
time.tzset()

# directory to hold value-table files
VALUE_TABLE_DIR = Path(__file__).resolve().parents[0] / "data"
VALUE_TABLE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder="static",  # points to site/static
    static_url_path="/static"  # URL base for static files
)

app.secret_key = os.urandom(32)
plotly_js = get_plotlyjs()
try:
    init_value_history_db()
except Exception as e:
    print(f"[value-history] init skipped: {e}")

FORM_BODY = """
<div class="home-page">
  <section class="home-hero">
    <div class="home-hero-left">
      <h1 class="home-title">BR Fantasy Dashboard</h1>
      <p class="home-subtitle">
        Turn your Sleeper league into a real front office:
        live projections, matchup previews, power rankings, and more—all in one place.
      </p>

      <ul class="home-bullets">
        <li>League-wide dashboard with weekly storylines and stats</li>
        <li>Matchup hub with projections and live scoring context</li>
        <li>Trade calculator powered by custom value models</li>
      </ul>
    </div>

    <div class="home-hero-right">
      <div class="home-card">
        <h2 class="home-card-title">Get started</h2>

        <div class="row">
          <label for="username">Sleeper Username</label>
          <input type="text" id="username" name="username" value="{{ username or '' }}">
        </div>

        <div class="row">
          <button type="button" id="lookupBtn">Find My Leagues</button>
        </div>

        <form method="post" id="leagueSelectForm">
          <input type="hidden" name="platform" value="sleeper">
          <input type="hidden" name="season" value="{{ viewed_season }}">

          <div class="row" id="leagueSelectWrap" style="display:none;">
            <label for="league">Choose League</label>
            <select id="league" name="league" required>
              <option value="">Select a league</option>
            </select>
          </div>

          <div class="row" id="generateWrap" style="display:none;">
            <button type="submit">Generate Dashboard</button>
          </div>

          <div id="lookupError" class="error-message" style="display:none;"></div>

          {% if error %}
          <div class="error-message">{{ error }}</div>
          {% endif %}
        </form>

        <p class="hint">
          Enter your Sleeper username, choose one of your leagues, and generate the dashboard.
        </p>
      </div>
    </div>
  </section>
   <section class="home-feature-grid">
        <div class="home-feature-card">
            <h3>Weekly Hub</h3>
            <p>
                See every starter, projection, and live score in one view.
                Perfect for Sunday trash talk and recap videos.
            </p>
        </div>
        <div class="home-feature-card">
            <h3>Trade Calculator</h3>
            <p>
                Evaluate trades with BR’s custom value engine so you don’t get
                fleeced by the league shark.
            </p>
        </div>
        <div class="home-feature-card">
            <h3>Graphs & Insights</h3>
            <p>
                Visualize PF, PA, luck, and schedule strength so you can prove
                who’s actually good.
            </p>
        </div>
    </section>
</div>
"""

BASE_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="google-adsense-account" content="ca-pub-9164153092633845">
    <title>{title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
    <!-- Google AdSense -->
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-9164153092633845"
            crossorigin="anonymous"></script>
    
    <link rel="icon" href="/static/BR_Logo.png" type="image/x-icon">

    <link rel="stylesheet" href="/static/dashboard.css">
    <script>
      {plotly_js}
    </script>
  </head>
  <body>
    <div id="app-scale">
      {nav}
      <main id="page-root" class="overview-layout">
        {body}
      </main>
    </div>

    <footer class="site-footer">
      <div class="site-footer-inner">
        <div class="site-footer-left">
          <span class="footer-brand">BR Fantasy</span>
          <span class="footer-tagline">Tools for obsessive commissioners.</span>
        </div>
        <div class="site-footer-links">
          <a href="{privacy_url}">Privacy</a>
          <a href="{faq_url}">FAQ</a>
          <a href="{support_url}">Support the site</a>
          <a href="{yt_url}" target="_blank" rel="noopener">YouTube</a>
          <a href="{contact_url}">Contact</a>
        </div>
        <div class="site-footer-note">
          © 2025 BR Fantasy. All rights reserved.
        </div>
      </div>
    </footer>

    <script src="/static/app.js"></script>
  </body>
</html>
"""


def format_sleeper_league_option(league: dict) -> dict:
    settings = league.get("settings") or {}

    return {
        "league_id": str(league.get("league_id", "")),
        "name": league.get("name") or "Unnamed League",
        "season": str(league.get("season") or ""),
        "total_rosters": league.get("total_rosters") or settings.get("num_teams") or "",
        "avatar": league.get("avatar") or "",
        "label": (
            f"{league.get('name') or 'Unnamed League'} "
            f"({league.get('season') or ''}) • "
            f"{league.get('total_rosters') or settings.get('num_teams') or '?'} teams"
        ),
    }


def _cache_key(platform: str, season: int, league_id: str):
    return str(platform).lower().strip(), int(season), str(league_id).strip()


def get_page_html_from_cache(platform: str, season: int, league_id: str, page: str) -> Optional[str]:
    entry = DASHBOARD_CACHE.get(_cache_key(platform, season, league_id))
    if not entry:
        return None
    pages = entry.get("page_html", {})
    rec = pages.get(page)
    if not rec:
        return None
    ts, html = rec
    if time.time() - ts > PAGE_HTML_TTL:
        return None
    return html


def store_page_html(platform: str, season: int, league_id: str, page: str, html: str) -> None:
    entry = DASHBOARD_CACHE.setdefault(_cache_key(platform, season, league_id), {})
    pages = entry.setdefault("page_html", {})
    pages[page] = (time.time(), html)


# -------- global NFL data caches (shared across leagues) --------
_PLAYERS_GLOBAL = None
_PLAYERS_INDEX_GLOBAL = None
_TEAMS_INDEX_GLOBAL = None


def get_players_global():
    global _PLAYERS_GLOBAL
    if _PLAYERS_GLOBAL is None:
        _PLAYERS_GLOBAL = get_nfl_players()
    return _PLAYERS_GLOBAL


def get_players_index_global():
    global _PLAYERS_INDEX_GLOBAL
    if _PLAYERS_INDEX_GLOBAL is None:
        _PLAYERS_INDEX_GLOBAL = load_players_index()
    return _PLAYERS_INDEX_GLOBAL


def get_teams_index_global():
    global _TEAMS_INDEX_GLOBAL
    if _TEAMS_INDEX_GLOBAL is None:
        _TEAMS_INDEX_GLOBAL = load_teams_index()
    return _TEAMS_INDEX_GLOBAL


def run_daily_data_async(season: int, week: int) -> None:
    """Start daily data build in a background thread."""
    thread = threading.Thread(
        target=build_daily_data,
        args=(season, week),
        daemon=True,
    )
    thread.start()


def _weeks_hash(weeks):
    raw = ",".join(str(w) for w in weeks)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def store_value_table(
        league_id: str,
        season: int,
        weeks: List[int],
        value_table: Dict[str, float],
) -> None:
    """
    Store value table in:
      1) in-memory DASHBOARD_CACHE
      2) disk as JSON (for reuse / training), named value_table_{date}.json
    """
    key = f"values_{season}_{_weeks_hash(weeks)}"

    # --- in-memory cache ---
    entry = DASHBOARD_CACHE.setdefault(league_id, {})
    bundle = entry.setdefault("value_tables", {})
    bundle[key] = (time.time(), value_table)

    # --- disk cache with date-stamped filename ---
    today_str = date.today().isoformat()  # e.g. "2025-11-22"
    value_dir = Path(VALUE_TABLE_DIR)
    value_dir.mkdir(parents=True, exist_ok=True)

    filename = f"usage_table_{today_str}.json"
    value_path = value_dir / filename

    with value_path.open("w", encoding="utf-8") as f:
        json.dump(value_table, f, ensure_ascii=False)


def store_model_values(
        league_id: str,
        season: int,
        weeks: List[int],
        value_table: Dict[str, float],
) -> None:
    """
    Store value table in:
      1) in-memory DASHBOARD_CACHE
      2) disk as JSON (for reuse / training), named value_table_{date}.json
    """
    key = f"model_values_{season}_{_weeks_hash(weeks)}"

    # --- in-memory cache ---
    entry = DASHBOARD_CACHE.setdefault(league_id, {})
    bundle = entry.setdefault("value_tables", {})
    bundle[key] = (time.time(), value_table)

    # --- disk cache with date-stamped filename ---
    today_str = date.today().isoformat()  # e.g. "2025-11-22"
    value_dir = Path(VALUE_TABLE_DIR)
    value_dir.mkdir(parents=True, exist_ok=True)

    filename = f"model_values_{today_str}.json"
    value_path = value_dir / filename

    with value_path.open("w", encoding="utf-8") as f:
        json.dump(value_table, f, ensure_ascii=False)


def build_nav(league_id: Optional[str], active: str, platform: str, season: int) -> str:
    """
    active (league pages): 'dashboard','standings','power','weekly','teams','activity','injuries','trade','graphs'
    active (global pages): 'home','privacy','faq','contact','support'
    """
    nfl_state = get_nfl_state() or {}
    offseason_mode = ((nfl_state.get("season_type") or "").lower() == "off") and (
            int(nfl_state.get("season") or datetime.now().year) == int(season or 0)
    )

    if not league_id:
        def simple_pill(label: str, href: str, key: str) -> str:
            cls = "nav-pill active" if key == active else "nav-pill"
            return f"<a class='{cls}' href='{href}'>{label}</a>"

        pills = [
            simple_pill("Home", "/", "home"),
            simple_pill("Trade Calc", "/trade", "trade"),
            simple_pill("FAQ", "/faq", "faq"),
            simple_pill("Privacy", "/privacy", "privacy"),
            simple_pill("Support the site", "/support", "support"),
            simple_pill("Contact", "/contact", "contact"),
        ]

        return (
            "<nav class='top-nav'>"
            "  <div><img src='/static/Website_Logo.png' alt='League Logo' class='site-logo'/></div>"
            "  <div class='top-nav-links'>"
            f"    {''.join(pills)}"
            "  </div>"
            "</nav>"
        )

    # -------- League nav (with league_id) --------

    def nav_pill(label: str, endpoint: str, key: str) -> str:
        cls = "nav-pill active" if key == active else "nav-pill"
        href = url_for(endpoint, platform=platform, season=season, league_id=league_id)
        return f"<a class='{cls}' href='{href}'>{label}</a>"

    # Only show refresh for pages that still have meaningful refresh behavior
    refreshable_pages = {"dashboard", "teams", "activity", "trade"}
    if not offseason_mode:
        refreshable_pages.update({"weekly", "standings", "graphs"})

    refresh_label_map = {
        "dashboard": "↻",
        "weekly": "↻",
        "teams": "↻",
        "activity": "↻",
        "standings": "↻",
        "graphs": "↻",
        "trade": "↻",
    }
    refresh_label = refresh_label_map.get(active, "↻")

    refresh_btn = ""
    if active in refreshable_pages:
        refresh_btn = (
            f"<button type='button'"
            f"        id='refreshBtn'"
            f"        class='refresh-icon'"
            f"        data-page='{active}'"
            f"        data-league='{league_id}'"
            f"        data-platform='{platform}'"
            f"        data-season='{season}'"
            f"        style='display:inline-flex;gap:6px;color: #122d4b;font-size: x-large;"
            f"               background: white;border: white;transform: rotate(90deg);'>"
            f"{refresh_label}"
            f"</button>"
        )

    pills = []
    if refresh_btn:
        pills.append(refresh_btn)

    # Always available
    pills.append(nav_pill("Dashboard", "page_dashboard", "dashboard"))
    pills.append(nav_pill("Trade Calc", "page_trade", "trade"))
    pills.append(nav_pill("Teams", "page_teams", "teams"))
    pills.append(nav_pill("Activity", "page_activity", "activity"))

    # In-season only
    if not offseason_mode:
        pills.append(nav_pill("Weekly Hub", "page_weekly", "weekly"))
        pills.append(nav_pill("Standings", "page_standings", "standings"))
        pills.append(nav_pill("Graphs", "page_graphs", "graphs"))

    pills.append("<a class='nav-pill logout-pill' href='/logout'>Logout</a>")

    return (
        "<nav class='top-nav'>"
        "  <div style='display:flex;align-items:center;gap:10px;'>"
        "    <img src='/static/Website_Logo.png' alt='League Logo' class='site-logo'/>"
        "  </div>"
        "  <div>"
        f"    {''.join(pills)}"
        "  </div>"
        "</nav>"
    )


def render_page(
        title: str,
        league_id: Optional[str],
        active: str,
        body_html: str,
        platform: Optional[str] = None,
        season: Optional[int] = None,
        *args,
        **kwargs,
) -> str:
    nav_html = build_nav(league_id, active, platform, season)

    wrapped_body = f"<div class='page-shell' data-page='{active}'>{body_html}</div>"

    return BASE_HTML.format(
        title=title,
        nav=nav_html,
        body=wrapped_body,
        plotly_js=plotly_js,
        privacy_url=league_url("privacy", league_id),
        faq_url=league_url("faq", league_id),
        support_url=league_url("support", league_id),
        contact_url=league_url("contact", league_id),
        yt_url="https://youtube.com/@hoodiekj",
    )


def validate_league_id(platform: str, league_id: str) -> tuple[bool, Optional[str]]:
    if not league_id:
        return False, "League ID is required."

    platform = (platform or "").lower().strip()

    if platform == "sleeper":
        if not league_id.isdigit():
            return False, "Invalid Sleeper league ID. Please check it and try again."
        return True, None

    if platform == "espn":
        if not league_id.isdigit():
            return False, "Invalid ESPN league ID. It should be a number."
        return True, None

    return False, f"Unsupported platform: {platform}"


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_most_recent_valid_draft_for_season(drafts: list, season: int) -> Optional[dict]:
    """
    Pick the most recent draft from the provided list, using the best available
    timestamp field. Return it only if it belongs to the viewed season.

    If the newest draft is from an older season, return None so the caller
    can keep TBD logic.
    """
    if not isinstance(drafts, list) or not drafts:
        return None

    def draft_sort_ts(d: dict) -> int:
        if not isinstance(d, dict):
            return -1
        return max(
            _safe_int(d.get("start_time"), -1),
            _safe_int(d.get("created"), -1),
            _safe_int(d.get("last_picked"), -1),
            _safe_int(d.get("last_message_time"), -1),
        )

    valid_drafts = [d for d in drafts if isinstance(d, dict)]
    if not valid_drafts:
        return None

    most_recent = max(valid_drafts, key=draft_sort_ts)
    most_recent_season = _safe_int(most_recent.get("season"))

    if most_recent_season != int(season):
        return None

    return most_recent


def build_league_context(platform: str, league_id: str, season: int) -> dict:
    """
    Fetch all core data for a league once and reuse across pages.
    Platform-agnostic. Cached at the route level.

    Offseason-safe behavior:
    - Offseason mode only applies to the CURRENT season while the NFL is in offseason.
    - Past seasons still behave like completed historical seasons.
    - In-season behavior is unchanged.
    """

    resolved_league_id = league_id
    print(
        f"[build_league_context] requested_league_id={league_id} "
        f"resolved_league_id={resolved_league_id} platform={platform} season={season}"
    )

    # Core league data
    league = get_league(platform, resolved_league_id, season)
    users = get_users(platform, resolved_league_id, season)
    rosters = get_rosters(platform, resolved_league_id, season)

    traded = None
    if platform == "sleeper":
        traded = get_traded_picks(platform, resolved_league_id, season)

    try:
        drafts = get_drafts(platform, resolved_league_id, season) or []
        latest_draft = get_most_recent_valid_draft_for_season(drafts, season)
    except Exception as e:
        print(f"[build_league_context] failed to load drafts for league {resolved_league_id}: {e}")
        drafts = []
        latest_draft = None

    # Global NFL state
    current = get_nfl_state() or {}
    season_type = (current.get("season_type") or "").lower()
    current_season = int(current.get("season") or datetime.now().year)
    current_week = int(current.get("week") or 0)
    current_leg = int(current.get("leg") or current_week or 0)

    FULL_SEASON_WEEKS = 18

    # IMPORTANT:
    # Offseason mode should only apply to the current upcoming season,
    # not to old historical seasons.
    offseason_mode = (season == current_season and season_type == "off")
    mode = "offseason" if offseason_mode else "in_season"

    if offseason_mode:
        max_week = 0
        season_complete = False
    elif season < current_season:
        max_week = FULL_SEASON_WEEKS
        season_complete = True
    else:
        max_week = max(1, current_week or current_leg or 1)
        season_complete = False

    weeks = max_week

    # Global reference data
    players = get_players_global()
    players_index = load_players_index()
    teams_index = load_teams_index()
    players_map = get_players_map(players)

    # League settings
    scoring_settings = get_effective_scoring_settings()
    raw_scoring_settings = get_scoring_settings() if "get_scoring_settings" in globals() else None
    roster_positions = get_roster_positions()
    league_settings = get_league_settings()
    total_rosters = get_total_rosters()

    # Core computed tables
    if offseason_mode:
        df_weekly = pd.DataFrame()
        team_stats = pd.DataFrame()
    else:
        df_weekly, team_stats, _ = build_tables(
            league_id=resolved_league_id,
            max_week=max_week,
            players=players,
            users=users,
            rosters=rosters,
            season=season,
            platform=platform,
        )

    # Always build roster_map from current rosters/users so offseason pages work
    user_fallback = {
        u["user_id"]: (
                (u.get("metadata") or {}).get("team_name")
                or u.get("display_name")
                or u.get("username")
                or str(u["user_id"])
        )
        for u in users
    }

    roster_map = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        roster_map[rid] = (r.get("metadata") or {}).get("team_name") or user_fallback.get(
            owner_id, f"Roster {rid}"
        )

    if df_weekly.empty and not offseason_mode:
        print(
            f"[build_league_context] no weekly data for requested_league_id={league_id}, "
            f"resolved_league_id={resolved_league_id}, season={season}"
        )

    activity_df = build_week_activity(resolved_league_id, platform, season, players_map)
    injury_df = build_injury_report(
        resolved_league_id,
        players,
        roster_map,
        rosters,
        "America/New_York",
        False,
    )

    if team_stats is not None and not team_stats.empty and {"Wins", "PF"}.issubset(team_stats.columns):
        standings_map = build_standings_map(team_stats, roster_map)
    else:
        standings_map = {}

    picks_by_roster = {}
    if platform == "sleeper":
        picks_by_roster = build_picks_by_roster(
            num_future_seasons=3,
            league=league,
            rosters=rosters,
            traded=traded,
        )

    scores_body = get_nfl_scores_for_date(date.today().strftime("%Y%m%d"))
    team_game_lookup = build_team_game_lookup(scores_body)

    model_value_table = load_model_value_table() or []

    return {
        "platform": platform,
        "league": league,
        "league_id": league_id,
        "resolved_league_id": resolved_league_id,
        "season": season,
        "rosters": rosters,
        "users": users,
        "traded": traded,
        "current_season": current_season,
        "current_week": current_week,
        "current_leg": current_leg,
        "season_type": season_type,
        "season_complete": season_complete,
        "weeks": weeks,
        "players": players,
        "players_map": players_map,
        "players_index": players_index,
        "teams_index": teams_index,
        "df_weekly": df_weekly,
        "team_stats": team_stats,
        "roster_map": roster_map,
        "injury_df": injury_df,
        "activity_df": activity_df,
        "standings_map": standings_map,
        "picks_by_roster": picks_by_roster,
        "team_game_lookup": team_game_lookup,
        "model_value_table": model_value_table,
        "scoring_settings": scoring_settings,
        "raw_scoring_settings": raw_scoring_settings,
        "roster_positions": roster_positions,
        "league_settings": league_settings,
        "total_rosters": total_rosters,
        "mode": mode,
        "offseason_mode": offseason_mode,
        "drafts": drafts,
        "latest_draft": latest_draft,
    }


def ensure_weekly_bits(ctx: dict) -> None:
    """
    Lazily populate projections, statuses, matchups, and df_weekly['proj']
    into the ctx. Only used by Dashboard + Weekly Hub + related APIs.
    """

    if ctx.get("offseason_mode"):
        ctx["proj_by_week"] = {}
        ctx["statuses"] = {}
        ctx["matchups_by_week"] = {}
        ctx["proj_by_roster"] = {}
        return

    def _apply_proj_column() -> None:
        df = ctx.get("df_weekly")
        proj_by_roster = ctx.get("proj_by_roster", {})

        if df is None or df.empty:
            return

        if "week" not in df.columns or "roster_id" not in df.columns:
            return

        key_series = list(zip(df["week"].astype(int), df["roster_id"].astype(str)))
        df = df.copy()
        df["proj"] = [proj_by_roster.get(k, float("nan")) for k in key_series]
        ctx["df_weekly"] = df

    # If already populated, just ensure df_weekly has proj
    if all(k in ctx for k in ("proj_by_week", "statuses", "matchups_by_week", "proj_by_roster")):
        if "proj" not in ctx.get("df_weekly", pd.DataFrame()).columns:
            _apply_proj_column()
        return

    viewed_season = int(ctx["season"])
    weeks = int(ctx["weeks"])
    platform = ctx["platform"]
    league_id = ctx.get("resolved_league_id", ctx["league_id"])
    players = ctx["players"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    roster_map = ctx["roster_map"]

    roster_counts = count_roster_positions(get_roster_positions())
    has_idp = any(k in roster_counts for k in ["DL", "LB", "DB", "IDP_FLEX"])

    proj_by_week = build_projections_by_week(viewed_season, weeks)

    if has_idp:
        statuses = build_status_by_week(
            viewed_season,
            weeks,
            players_index,
            teams_index,
            load_idp_index(),
        )
    else:
        statuses = build_status_by_week(
            viewed_season,
            weeks,
            players_index,
            teams_index,
        )

    matchups_by_week = build_matchups_by_week(
        league_id,
        range(1, weeks + 1),
        roster_map,
        players,
        viewed_season,
        platform,
    )

    proj_by_roster = compute_team_projections_for_weeks(
        matchups_by_week,
        statuses,
        proj_by_week,
        roster_map,
    )

    ctx["proj_by_week"] = proj_by_week
    ctx["statuses"] = statuses
    ctx["matchups_by_week"] = matchups_by_week
    ctx["proj_by_roster"] = proj_by_roster

    _apply_proj_column()


def refresh_league_ctx_section(platform: str, league_id: str, page: str, season: int) -> dict:
    key = _cache_key(platform, season, league_id)
    entry = DASHBOARD_CACHE.get(key)

    if not entry:
        ctx = build_league_context(platform, league_id, season)
        DASHBOARD_CACHE[key] = {"ctx": ctx, "ts": time.time(), "page_html": {}}
        return ctx

    ctx = entry["ctx"]

    viewed_season = int(ctx["season"])
    current_season = int(ctx.get("current_season") or viewed_season)
    current_week = int(ctx.get("current_week") or 0)
    weeks = int(ctx["weeks"])
    season_type = (ctx.get("season_type") or "").lower()
    season_complete = bool(ctx.get("season_complete", False))
    offseason_mode = bool(ctx.get("offseason_mode", False))
    resolved_league_id = ctx.get("resolved_league_id", league_id)

    rosters = get_rosters(platform, resolved_league_id, viewed_season)
    users = get_users(platform, resolved_league_id, viewed_season)

    ctx["rosters"] = rosters
    ctx["users"] = users

    players = ctx["players"]
    players_map = ctx["players_map"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]

    roster_counts = count_roster_positions(get_roster_positions())
    has_idp = any(k in roster_counts for k in ["DL", "LB", "DB", "IDP_FLEX"])

    if page in ("standings", "dashboard", "weekly"):
        if offseason_mode:
            ctx["df_weekly"] = pd.DataFrame()
            ctx["team_stats"] = pd.DataFrame()
            ctx["standings_map"] = {}
        else:
            df_weekly, team_stats, roster_map = build_tables(
                league_id=resolved_league_id,
                max_week=weeks,
                players=players,
                users=users,
                rosters=rosters,
                season=viewed_season,
                platform=platform,
            )

            ctx["df_weekly"] = df_weekly
            ctx["team_stats"] = team_stats
            ctx["roster_map"] = roster_map
            if team_stats is not None and not team_stats.empty and {"Wins", "PF"}.issubset(team_stats.columns):
                ctx["standings_map"] = build_standings_map(team_stats, roster_map)
            else:
                ctx["standings_map"] = {}

    if page in ("activity", "dashboard"):
        clear_activity_cache_for_league(league_id)

        ctx["activity_df"] = build_week_activity(
            resolved_league_id,
            platform,
            viewed_season,
            players_map,
        )

        roster_map = ctx["roster_map"]

        if has_idp:
            statuses = build_status_by_week(
                viewed_season,
                weeks,
                players_index,
                teams_index,
                load_idp_index(),
            )
        else:
            statuses = build_status_by_week(
                viewed_season,
                weeks,
                players_index,
                teams_index,
            )

        ctx["statuses"] = statuses

        ctx["injury_df"] = build_injury_report(
            resolved_league_id,
            players,
            roster_map,
            rosters,
            "America/New_York",
            False,
        )

    if page in ("weekly", "dashboard"):
        clear_weekly_cache_for_league(league_id)

        if offseason_mode:
            ctx["proj_by_week"] = {}
            ctx["statuses"] = {}
            ctx["matchups_by_week"] = {}
            ctx["proj_by_roster"] = {}
        else:
            should_refresh_live_week = (
                    viewed_season == current_season
                    and not season_complete
                    and not offseason_mode
                    and current_week > 0
                    and season_type != "off"
            )

            if should_refresh_live_week:
                live_game_ids = get_live_game_ids_for_today(
                    load_week_schedule(current_season, current_week)
                )
                build_and_save_week_stats_for_league(
                    load_teams_index(),
                    current_season,
                    current_week,
                    live_game_ids,
                )

                get_week_projections_cached(
                    current_season,
                    current_week,
                    fetch_week_from_tank01,
                    True,
                )

            roster_map = ctx["roster_map"]

            ctx["proj_by_week"] = build_projections_by_week(viewed_season, weeks)

            if has_idp:
                ctx["statuses"] = build_status_by_week(
                    viewed_season,
                    weeks,
                    players_index,
                    teams_index,
                    load_idp_index(),
                )
            else:
                ctx["statuses"] = build_status_by_week(
                    viewed_season,
                    weeks,
                    players_index,
                    teams_index,
                )

            ctx["matchups_by_week"] = build_matchups_by_week(
                resolved_league_id,
                range(1, weeks + 1),
                roster_map,
                players,
                viewed_season,
                platform,
            )

            proj_by_roster = compute_team_projections_for_weeks(
                ctx["matchups_by_week"],
                ctx["statuses"],
                ctx["proj_by_week"],
                roster_map,
            )
            ctx["proj_by_roster"] = proj_by_roster

            df = ctx.get("df_weekly")
            if df is not None and not df.empty:
                key_series = list(zip(df["week"].astype(int), df["roster_id"].astype(str)))
                df = df.copy()
                df["proj"] = [proj_by_roster.get(k, float("nan")) for k in key_series]
                ctx["df_weekly"] = df

    if page == "teams":
        clear_teams_cache_for_league(league_id)
        ctx["rosters"] = rosters
        ctx["users"] = users

    entry["ts"] = time.time()
    return ctx


def render_standings(team_stats, length) -> str:
    if team_stats is None or team_stats.empty:
        return """
        <div class="card-body">
          <p>No standings data available for this season yet.</p>
        </div>
        """

    rows = []

    df = team_stats.copy()
    df["WinPct"] = df["Win%"].astype(float)
    df = (
        df.sort_values(
            by=["Wins", "PF", "PA"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    df["Rank"] = df.index + 1

    for _, row in df.iterrows():
        record = f"{int(row['Wins'])}-{int(row['Losses'])}"
        if int(row.get("Ties", 0)):
            record += f"-{int(row['Ties'])}"

        streak = row.get("Streak", "")
        avatar = row.get("avatar", "")

        img = (
            f"<img class='avatar sm' src='{avatar}' "
            "onerror=\"this.style.display='none'\">"
            if avatar else ""
        )

        rows.append(f"""
            <tr>
              <td class="num">{int(row['Rank'])}</td>
              <td class="team">{img} {row['owner']}</td>
              <td>{record}</td>
              <td>{row['PF']:.1f}</td>
              <td>{row['PA']:.1f}</td>
              <td>{streak}</td>
              <td>{row.get('past_sos', 0.0):.1f}</td>
              <td>{row.get('ros_sos', 0.0):.1f}</td>
            </tr>
        """)

    total_rows = rows[:length] if len(rows) != length else rows

    return f"""
        <table class="standings-table" data-page="standings">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Team</th>
              <th>Record</th>
              <th>PF</th>
              <th>PA</th>
              <th>Streak</th>
              <th>SOS Past</th>
              <th>SOS Future</th>
            </tr>
          </thead>
          <tbody>
            {''.join(total_rows)}
          </tbody>
        </table>
    """


def build_dashboard_body(ctx: dict) -> str:
    league_id = ctx["league_id"]
    platform = ctx["platform"]
    season = ctx["season"]  # viewed season, not live NFL season
    rosters = ctx["rosters"]
    users = ctx["users"]
    current_week = int(ctx.get("current_week") or 0)
    weeks = int(ctx.get("weeks") or 1)
    players_map = ctx["players_map"]
    df_weekly = ctx["df_weekly"]
    team_stats = ctx["team_stats"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    statuses = ctx["statuses"]
    proj_by_week = ctx["proj_by_week"]
    matchups_by_week = ctx["matchups_by_week"]
    picks_by_roster = ctx["picks_by_roster"]
    team_game_lookup = ctx["team_game_lookup"]
    season_complete = bool(ctx.get("season_complete", False))
    offseason_mode = bool(ctx.get("offseason_mode", False))

    # --- Standings snapshot ---
    standings_html = render_standings(team_stats, 5)

    # --- Finalized games + last_final_week ---
    if (
            df_weekly is not None
            and not df_weekly.empty
            and "finalized" in df_weekly.columns
            and "week" in df_weekly.columns
    ):
        finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    else:
        finalized_df = pd.DataFrame()

    if not finalized_df.empty and "week" in finalized_df.columns:
        last_final_week = int(finalized_df["week"].max())
    else:
        last_final_week = max(1, min(current_week or 1, weeks))

    # Offseason / completed season should default to the last finalized week
    if season_complete or offseason_mode:
        display_week = last_final_week
    else:
        display_week = max(1, min(current_week or 1, weeks))

    week_statuses = (statuses.get(display_week) or {}).get("statuses", {}) or {}
    week_matchups = matchups_by_week.get(display_week, []) or []

    slides = [
        render_matchup_slide(
            season,
            m,
            display_week,
            last_final_week,
            status_by_pid=week_statuses,
            projections=proj_by_week,
            players=players_index,
            teams=teams_index,
            team_game_lookup=team_game_lookup,
        )
        for m in week_matchups
    ]

    slides_by_week = {
        display_week: "".join(slides) if slides else "<div class='m-empty'>No matchups</div>"
    }

    matchup_html = render_matchup_carousel_weeks(
        slides_by_week,
        dashboard=True,
        active_week=display_week,
    )

    awards = compute_awards_season(
        finalized_df,
        players_map,
        league_id,
        platform,
        season,
        users,
        rosters,
    )
    awards_html = render_awards_section(awards)

    teams_ctx = build_teams_overview(
        rosters=rosters,
        users_list=users,
        picks_by_roster=picks_by_roster,
        players=players_map,
        players_index=players_index,
        teams_index=teams_index,
        platform=platform
    )

    teams_sidebar_html = render_teams_sidebar(teams_ctx)

    season_note = ""
    if offseason_mode:
        season_note = (
            f"<div class='mini-label' style='margin-bottom:10px;'>"
            f"Viewing {season} season data during the offseason."
            f"</div>"
        )

    body = f"""
    <aside class="overview-sidebar-left">
      {awards_html}
    </aside>
    <div class="overview-main">
      <div class="card central">
        <h2>Standings</h2>
        {season_note}
        {standings_html}
      </div>
      {matchup_html}
    </div>
    <aside class="overview-sidebar">
      {teams_sidebar_html}
    </aside>
    """

    return body


def render_power_and_playoffs(team_stats, roster_map: dict[str, str], league_id: str, platform, season) -> str:
    """
    Single card that shows:
      - Power Rankings (by PowerScore if present)
      - Playoff Picture (using bracket)
    """
    if team_stats is None or team_stats.empty:
        return ""

    # ---- Sort by PowerScore, with PF as tiebreaker if available ----
    has_power = "PowerScore" in team_stats.columns
    has_pf = "PF" in team_stats.columns

    if has_power:
        if has_pf:
            pr_sorted = team_stats.sort_values(
                ["PowerScore", "PF"],
                ascending=[False, False],
            ).reset_index(drop=True)
        else:
            pr_sorted = team_stats.sort_values(
                "PowerScore",
                ascending=False,
            ).reset_index(drop=True)
    else:
        if has_pf:
            pr_sorted = team_stats.sort_values("PF", ascending=False).reset_index(drop=True)
        else:
            pr_sorted = team_stats.reset_index(drop=True)

    top3 = pr_sorted.head(3)

    # width scaling based on PowerScore range
    if has_power:
        p = pr_sorted["PowerScore"].astype(float)
        pmin, pmax = float(p.min()), float(p.max())
    else:
        p = None
        pmin = pmax = 0.0

    def pct_width(v: float) -> float:
        """Convert a PowerScore into a 2–100% bar width."""
        if p is None:
            return 100.0
        if pmax == pmin:
            return 100.0
        return max(2.0, (float(v) - pmin) / (pmax - pmin) * 100.0)

    def safe_int(val, default=0):
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def podium_slot(rank: int, row) -> str:
        name = row.get("owner", "Unknown")

        # record
        wins = safe_int(row.get("Wins"), 0)
        games = safe_int(row.get("G"), 0)
        losses = max(games - wins, 0)
        ties_val = safe_int(row.get("Ties"), 0)
        rec = f"{wins}-{losses}" + (f"-{ties_val}" if ties_val else "")

        size = {"1": "38px", "2": "32px", "3": "32px"}[str(rank)]
        base_cls = {1: "first", 2: "second", 3: "third"}[rank]

        power_val = safe_float(row.get("PowerScore"), 0.0)
        w = pct_width(power_val)

        # streak bits
        streak_chip = row.get("Streak", "")  # e.g., "W3", "L2"
        streak_frame_cls = streak_class(row)  # assumes you already have this helper
        avatar_url = row.get("avatar")
        avatar_html = (
            f"<img class='avatar' src='{avatar_url}' "
            "onerror=\"this.style.display='none'\">"
            if avatar_url else ""
        )

        # PF/G, PA/G, diff
        pf = safe_float(row.get("PF"), 0.0)
        pa = safe_float(row.get("PA"), 0.0)
        g = games if games > 0 else 1
        pfpg_v = pf / g
        papg_v = pa / g
        diff_v = pfpg_v - papg_v
        diff_class = "diff-pos" if diff_v > 0 else "diff-neg" if diff_v < 0 else ""

        chips_html = "<div class='chips'>"
        chips_html += f"<span class='chip'>PF/G {pfpg_v:.1f}</span>"
        chips_html += f"<span class='chip'>PA/G {papg_v:.1f}</span>"
        chips_html += f"<span class='chip {diff_class}'>{diff_v:+.1f}</span>"
        if streak_chip and streak_frame_cls == "streak-hot":
            chips_html += f"<span class='chip chip-streak'>🔥{streak_chip}</span>"
        elif streak_chip and streak_frame_cls == "streak-cold":
            chips_html += f"<span class='chip chip-streak'>❄️{streak_chip}</span>"
        chips_html += "</div>"

        return f"""
          <div class="slot {base_cls} {streak_frame_cls}">
            <div class="wrap">
              <div class='podium-header'>
                <h3 style="font-size:{size}">#{rank}</h3>
                {avatar_html}
              </div>
              <div class="name">{name}</div>
              <div class="rec">{rec}</div>
              <div class="bar"><div style="width:{w:.1f}%"></div></div>
              {chips_html}
            </div>
          </div>
        """

    # ---- Top 3 podium ----
    podium_html = """
      <div class="podium">
        {slot1}
        {slot2}
        {slot3}
      </div>
    """.format(
        slot1=podium_slot(1, top3.iloc[0]) if len(top3) > 0 else "",
        slot2=podium_slot(2, top3.iloc[1]) if len(top3) > 1 else "",
        slot3=podium_slot(3, top3.iloc[2]) if len(top3) > 2 else "",
    )

    # ---- Remaining ranks list ----
    others = pr_sorted.iloc[3:].reset_index(drop=True)
    rank_cards = []
    for i, row in others.iterrows():
        pos = i + 4
        team = row.get("owner", "Unknown")

        wins = safe_int(row.get("Wins"), 0)
        games = safe_int(row.get("G"), 0)
        losses = max(games - wins, 0)
        ties_val = safe_int(row.get("Ties"), 0)
        record = f"{wins}-{losses}" + (f"-{ties_val}" if ties_val else "")

        power_val = safe_float(row.get("PowerScore"), 0.0)
        bar_w = pct_width(power_val)

        # per-row PF/G, PA/G, diff
        pf = safe_float(row.get("PF"), 0.0)
        pa = safe_float(row.get("PA"), 0.0)
        g = games if games > 0 else 1
        pfpg_v = pf / g
        papg_v = pa / g
        diff_v = pfpg_v - papg_v
        diff_class = "diff-pos" if diff_v > 0 else "diff-neg" if diff_v < 0 else ""

        streak_chip = row.get("Streak", "")
        chips_html = (
            f"<span class='chip'>PF/G {pfpg_v:.1f}</span>"
            f"<span class='chip'>PA/G {papg_v:.1f}</span>"
        )
        css_cls = streak_class(row)
        if streak_chip and css_cls == "streak-hot":
            chips_html += f"<span class='chip chip-streak'>🔥{streak_chip}</span>"
        elif streak_chip and css_cls == "streak-cold":
            chips_html += f"<span class='chip chip-streak'>❄️{streak_chip}</span>"
        chips_html += f"<span class='chip {diff_class}'>{diff_v:+.1f}</span>"

        avatar_url = row.get("avatar")
        img = (
            f"<img class='avatar sm' src='{avatar_url}' "
            "onerror=\"this.style.display='none'\">"
            if avatar_url else ""
        )

        rank_cards.append(
            f"<div class='rank-item {css_cls} '>"
            f"<span class='pos'>#{pos}</span>"
            f"<span class='name'>{img}&nbsp;{team}</span>"
            f"<span class='rec'>{record}</span>"
            f"<div class='power-row'>"
            f"<div class='bar'><div style='width:{bar_w:.1f}%'></div></div>"
            f"<div class='chips'>{chips_html}</div>"
            f"</div>"
            f"</div>"
        )

    rankings_html = "<div class='rank-grid'>" + "".join(rank_cards) + "</div>"

    # ---- Playoff bracket ----
    wb = get_bracket(platform, league_id, "winners", season)
    roster_avatar_map = {
        str(owner): av
        for owner, av in zip(team_stats["owner"], team_stats["avatar"])
        if pd.notna(owner)
    }

    seed_map = seed_top6_from_team_stats(team_stats, roster_map)

    bracket_html = playoff_bracket(
        wb,
        roster_name_map=roster_map,
        roster_avatar_map=roster_avatar_map,
        seed_map=seed_map,
    )

    podium_card = f"""
          <div class="card power" data-section="overview">
            <div class="card-tabs" data-card="power">
              <div class="tab-strip">
                <button class="tab-btn active" data-tab="power">Power Rankings</button>
                <button class="tab-btn" data-tab="playoff">Playoff Picture</button>
              </div>
              <div class="tab-panels">
                <div class="tab-panel active" data-tab="power">
                  {podium_html}
                  {rankings_html}
                </div>
                <div class="tab-panel" data-tab="playoff">
                  {bracket_html}
                </div>
              </div>
            </div>
          </div>
        """

    return podium_card


def render_standings_sidebar(team_stats) -> str:
    if team_stats is None or team_stats.empty:
        return ""

    ts = team_stats.copy()

    # --------------------
    # Best Offense / Defense
    # --------------------
    best_off = ts.loc[ts["PF"].idxmax()] if "PF" in ts.columns else None
    best_def = ts.loc[ts["PA"].idxmin()] if "PA" in ts.columns else None

    # --------------------
    # Hottest / Coldest Streaks
    # --------------------
    hottest = None
    coldest = None
    if "StreakLen" in ts.columns and "StreakType" in ts.columns:
        hot_df = ts[ts["StreakType"] == "W"]
        cold_df = ts[ts["StreakType"] == "L"]

        if not hot_df.empty:
            hottest = hot_df.loc[hot_df["StreakLen"].idxmax()]
        if not cold_df.empty:
            coldest = cold_df.loc[cold_df["StreakLen"].idxmax()]

    cards = []

    # --------------------
    # Best Offense Card
    # --------------------
    if best_off is not None:
        cards.append(f"""
        <div class="card small">
          <div class="card-header">
            <h3>Best Offense</h3>
            <h3>{best_off['PF']:.1f} PF</h3>
          </div>
          <div class="card-body">
            <div class="highlight-game-card">
              <div class="hg-row">
                <div class="hg-team">
                  <span class="hg-name">{best_off['owner']}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        """)

    # --------------------
    # Best Defense Card
    # --------------------
    if best_def is not None:
        cards.append(f"""
        <div class="card small">
          <div class="card-header">
            <h3>Best Defense</h3>
            <h3>{best_def['PA']:.1f} PA</h3>
          </div>
          <div class="card-body">
            <div class="highlight-game-card">
              <div class="hg-row">
                <div class="hg-team">
                  <span class="hg-name">{best_def['owner']}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        """)

    # --------------------
    # Hottest Team Card
    # --------------------
    if hottest is not None:
        cards.append(f"""
        <div class="card small" style="background: linear-gradient(180deg, #fff8e7, #ffe5b4);border:1px solid #f97316;">
          <div class="card-header">
            <h3 style="color:#dc2626;">Hottest Team</h3>
            <h3 style="color:#dc2626;">🔥 {hottest['Streak']}</h3>
          </div>
          <div class="card-body">
            <div class="highlight-game-card">
              <div class="hg-row">
                <div class="hg-team">
                  <span class="hg-name">{hottest['owner']}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        """)

    # --------------------
    # Coldest Team Card
    # --------------------
    if coldest is not None:
        cards.append(f"""
        <div class="card small" style="border: 1px solid #163b82f6;background: rgb(44 166 173 / 12%);color: #163b82f6;">
          <div class="card-header">
            <h3>Coldest Team</h3>
            <h3>❄️ {coldest['Streak']}</h3>
          </div>
          <div class="card-body">
            <div class="highlight-game-card">
              <div class="hg-row">
                <div class="hg-team">
                  <span class="hg-name">{coldest['owner']}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        """)

    return "".join(cards)


def render_team_stats(team_stats, df_weekly) -> str:
    if team_stats is None or team_stats.empty or df_weekly is None or df_weekly.empty:
        return """
        <div class="card-body">
          <p>No detailed stats available for this season yet.</p>
        </div>
        """

    best = df_weekly.groupby("owner")["points"].max().rename("Best Week")
    worst = df_weekly.groupby("owner")["points"].min().rename("Worst Week")

    stats_tbl = (
        team_stats.rename(columns={"owner": "Team", "AVG": "Average", "STD": "Std Dev", "Win%": "Win %"})
        .merge(best, left_on="Team", right_index=True, how="left")
        .merge(worst, left_on="Team", right_index=True, how="left")
    )

    cols = ["Team", "Win %", "PF", "PA", "Average", "Std Dev", "Best Week", "Worst Week"]
    stats_tbl = stats_tbl[cols].copy()

    for c in ["Win %", "PF", "PA", "Average", "Std Dev", "Best Week", "Worst Week"]:
        stats_tbl[c] = stats_tbl[c].astype(float).round(3 if c == "Win %" else 2)

    body_rows = []
    for _, r in stats_tbl[cols].iterrows():
        avatar = r.get("avatar", "")
        img = (
            f"<img class='avatar sm' src='{avatar}' "
            "onerror=\"this.style.display='none'\">"
            if avatar else ""
        )
        body_rows.append("<tr>" + "".join([
            f"<td class='team'>{img} {r['Team']}</td>",
            f"<td class='num'>{r['Win %']:.3f}</td>",
            f"<td class='num'>{float(r['PF']):.2f}</td>",
            f"<td class='num'>{float(r['PA']):.2f}</td>",
            f"<td class='num'>{float(r['Average']):.2f}</td>",
            f"<td class='num'>{float(r['Std Dev']):.2f}</td>",
            f"<td class='num'>{float(r['Best Week']):.2f}</td>",
            f"<td class='num'>{float(r['Worst Week']):.2f}</td>",
        ]) + "</tr>")

    table_html = f"""
        <table id="stats" class="standings-table">
          <thead><tr>{"".join([f"<th data-col='{i}'>{c}</th>" for i, c in enumerate(cols)])}</tr></thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
    """
    return table_html


def build_standings_body(ctx: dict) -> str:
    team_stats = ctx["team_stats"]
    roster_map = ctx["roster_map"]
    df_weekly = ctx["df_weekly"]
    rosters = ctx["rosters"]
    num_teams = len({str(r.get("roster_id")) for r in rosters})

    standings_html = render_standings(team_stats, num_teams)

    if (
            df_weekly is not None
            and not df_weekly.empty
            and "finalized" in df_weekly.columns
    ):
        detailed_df = df_weekly[df_weekly["finalized"] == True].copy()
    else:
        detailed_df = pd.DataFrame()

    table_html = render_team_stats(team_stats, detailed_df)
    power_playoffs_html = render_power_and_playoffs(
        team_stats,
        roster_map,
        ctx.get("resolved_league_id", ctx["league_id"]),
        ctx["platform"],
        ctx["season"],
    )
    sidebar_html = render_standings_sidebar(team_stats)

    body = f"""
    <div class="standings-main two-col-standings">
      <div class="standings-col">
        <div class="card">
          <div class="card-tabs">
            <div class="tab-strip">
              <button class="tab-btn active" data-tab="standings">Standings</button>
              <button class="tab-btn" data-tab="details">Detailed Stats</button>
              <div class="tab-panels">
                <div class="tab-panel active" data-tab="standings">
                  {standings_html}
                </div>
                <div class="tab-panel" data-tab="details">
                  {table_html}
                  <div class="footer">
                    Default sort: Win% ↓ then PF ↓. Click headers to sort.
                  </div>
                </div>
              </div>
            </div>
          </div
        </div>
      </div>
    </div>
    <div class="standings-col">
      {power_playoffs_html}
    </div>
    </div>
    <aside class="overview-sidebar">
      {sidebar_html}
    </aside>
    """

    return body


def build_offseason_dashboard_body(ctx: dict) -> str:
    league = ctx["league"]
    platform = ctx["platform"]
    season = ctx["season"]
    rosters = ctx["rosters"]
    users = ctx["users"]
    roster_map = ctx["roster_map"]
    picks_by_roster = ctx.get("picks_by_roster", {})
    players_index = ctx["players_index"]
    players_map = ctx["players_map"]
    model_value_table = ctx.get("model_value_table") or []

    latest_draft = ctx.get("latest_draft")
    draft_text = "Draft date not set"
    countdown_text = "TBD"
    draft_subtext = "Set once your league schedules the draft."

    draft_ts_ms = None

    if isinstance(latest_draft, dict):
        draft_ts_ms = _safe_int(latest_draft.get("start_time"))

    if draft_ts_ms is None:
        draft_ts_ms = _safe_int(league.get("draft_day"))

    if draft_ts_ms:
        try:
            draft_dt = datetime.fromtimestamp(draft_ts_ms / 1000, tz=EASTERN)
            now_dt = datetime.now(EASTERN)
            delta_days = (draft_dt.date() - now_dt.date()).days

            draft_text = draft_dt.strftime("%b %d, %Y at %I:%M %p %Z")
            countdown_text = f"{delta_days} days" if delta_days >= 0 else "Draft passed"
            draft_subtext = "Countdown to your next league draft."
        except Exception:
            pass

    teams_ctx = build_teams_overview(
        rosters=rosters,
        users_list=users,
        picks_by_roster=picks_by_roster,
        players=players_map,
        players_index=players_index,
        teams_index=ctx["teams_index"],
        platform=platform,
    )
    teams_sidebar_html = render_teams_sidebar(teams_ctx)

    values_by_id = {}
    for row in model_value_table:
        if isinstance(row, dict) and row.get("id") is not None:
            try:
                values_by_id[str(row["id"])] = float(row.get("value") or 0.0)
            except Exception:
                values_by_id[str(row["id"])] = 0.0

    roster_cards = []
    total_future_picks = 0

    for r in rosters:
        rid = str(r.get("roster_id"))
        team_name = roster_map.get(rid, f"Roster {rid}")
        player_ids = [str(pid) for pid in (r.get("players") or [])]
        roster_value = sum(values_by_id.get(pid, 0.0) for pid in player_ids)
        team_picks = picks_by_roster.get(rid, []) if isinstance(picks_by_roster, dict) else []
        pick_count = len(team_picks)
        total_future_picks += pick_count

        first_round_count = 0
        for pk in team_picks:
            try:
                if int(pk.get("round") or 0) == 1:
                    first_round_count += 1
            except Exception:
                pass

        badge_bits = []
        if first_round_count > 0:
            badge_bits.append(f"{first_round_count} first{'s' if first_round_count != 1 else ''}")
        if pick_count > 0:
            badge_bits.append(f"{pick_count} future picks")

        badge_html = ""
        if badge_bits:
            badge_html = "".join(
                f"<span class='os-snapshot-chip'>{bit}</span>" for bit in badge_bits[:2]
            )

        roster_cards.append({
            "team_name": team_name,
            "roster_value": roster_value,
            "pick_count": pick_count,
            "html": f"""
            <div class="os-snapshot-card">
              <div class="os-snapshot-top">
                <div class="os-snapshot-rank-block">
                  <div class="os-snapshot-team">{team_name}</div>
                  <div class="os-snapshot-meta">Roster value</div>
                </div>
                <div class="os-snapshot-value">{roster_value:.0f}</div>
              </div>
              <div class="os-snapshot-bottom">
                <div class="os-snapshot-chip-row">
                  {badge_html}
                </div>
              </div>
            </div>
            """
        })

    roster_cards.sort(key=lambda x: x["roster_value"], reverse=True)

    ranked_snapshot_html = []
    for idx, card in enumerate(roster_cards, start=1):
        ranked_snapshot_html.append(
            f"""
            <div class="os-snapshot-rank-wrap">
              <div class="os-snapshot-rank">#{idx}</div>
              <div class="os-snapshot-rank-card">
                {card["html"]}
              </div>
            </div>
            """
        )

    roster_cards_html = "".join(ranked_snapshot_html)

    roster_leader = roster_cards[0]["team_name"] if roster_cards else "N/A"
    highest_roster_value = f"{roster_cards[0]['roster_value']:.0f}" if roster_cards else "0"

    rostered_ids = {
        str(pid)
        for r in rosters
        for pid in (r.get("players") or [])
    }

    top_waiver_assets = []
    for row in model_value_table:
        if not isinstance(row, dict):
            continue

        pid = str(row.get("id") or "")
        pos = str(row.get("position") or row.get("pos") or "").upper()

        if not pid or pid in rostered_ids:
            continue
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue

        try:
            val = float(row.get("value") or 0.0)
        except Exception:
            val = 0.0

        if val <= 0:
            continue

        top_waiver_assets.append({
            "name": row.get("name", "Unknown"),
            "position": pos,
            "team": row.get("team") or "",
            "value": val,
            "age": row.get("age"),
            "pos_rank_label": row.get("pos_rank_label") or "",
        })

    top_waiver_assets.sort(key=lambda x: x["value"], reverse=True)

    waiver_html = []
    for p in top_waiver_assets[:10]:
        sub_bits = [p["position"]]
        if p["team"]:
            sub_bits.append(p["team"])
        if p["pos_rank_label"]:
            sub_bits.append(p["pos_rank_label"])

        subline = " • ".join(sub_bits)

        waiver_html.append(
            f"""
            <div class="os-waiver-row">
              <div class="os-waiver-main">
                <div class="os-waiver-name">{p['name']}</div>
                <div class="os-waiver-sub">{subline}</div>
              </div>
              <div class="os-waiver-value">{p['value']:.0f}</div>
            </div>
            """
        )

    top_waiver_assets_html = "".join(waiver_html)

    body = f"""
    <div class="os-layout">
      <aside class="os-left-col">
        <section class="os-card os-card-soft">
          <div class="os-section-head">
            <h2 class="os-section-title">Offseason Team Snapshot</h2>
            <div class="os-section-subtitle">Roster value and future capital across the league</div>
          </div>
          <div class="os-snapshot-list">
            {roster_cards_html or "<p>No offseason roster data available yet.</p>"}
          </div>
        </section>
      </aside>

      <main class="os-main-col">
        <section class="os-hero-card">
          <div class="os-hero-top">
            <div>
              <div class="os-hero-kicker">Viewing {season} offseason league data</div>
              <h1 class="os-hero-title">Offseason Hub</h1>
              <p class="os-hero-copy">
                Focus on roster building, draft prep, waiver value, and trade opportunities.
              </p>
            </div>
            <div class="os-hero-badge">Offseason</div>
          </div>

          <div class="os-hero-stats">
            <div class="os-stat-card">
              <div class="os-stat-label">Draft countdown</div>
              <div class="os-stat-value">{countdown_text}</div>
              <div class="os-stat-sub">{draft_text}</div>
            </div>
            <div class="os-stat-card">
              <div class="os-stat-label">League leader</div>
              <div class="os-stat-value">{highest_roster_value}</div>
              <div class="os-stat-sub">{roster_leader}</div>
            </div>
            <div class="os-stat-card">
              <div class="os-stat-label">Future picks tracked</div>
              <div class="os-stat-value">{total_future_picks}</div>
              <div class="os-stat-sub">{len(rosters)} teams in league</div>
            </div>
          </div>

          <div class="os-hero-footer">
            {draft_subtext}
          </div>
        </section>

        <section class="os-card">
          <div class="os-section-head">
            <h2 class="os-section-title">Top Waiver Assets</h2>
            <div class="os-section-subtitle">Best currently unrostered players by BR value</div>
          </div>
          <div class="os-waiver-list">
            {top_waiver_assets_html or "<p>No waiver values available yet.</p>"}
          </div>
        </section>
      </main>

      <aside class="os-right-col">
        <div class="os-sidebar-shell">
          {teams_sidebar_html}
        </div>
      </aside>
    </div>
    """
    return body


def apply_multi_for_one_adjustment(side_a: dict, side_b: dict) -> None:
    """
    Multi-for-one adjustment:

    - Only uses *player* values (ignores picks entirely).
    - Gives a bonus to the side getting FEWER players, scaled by:
        * gap in player value
        * how much of that side is tied up in its best player ("stud")
        * how many extra pieces the other side is sending
    - Adjustment is added on top of raw_total (which can still include picks).
    """

    vals_a = side_a.get("player_values", []) or []
    vals_b = side_b.get("player_values", []) or []
    n_a, n_b = len(vals_a), len(vals_b)

    # No players, or same number of players → no adjustment.
    if n_a == 0 or n_b == 0 or n_a == n_b:
        side_a["effective_total"] = side_a["raw_total"]
        side_b["effective_total"] = side_b["raw_total"]
        side_a["adjustment"] = 0.0
        side_b["adjustment"] = 0.0
        return

    # Decide which side is consolidating (fewer players)
    if n_a < n_b:
        fewer = side_a
        more = side_b
        fewer_is_a = True
    else:
        fewer = side_b
        more = side_a
        fewer_is_a = False

    fewer_vals = fewer.get("player_values", []) or []
    more_vals = more.get("player_values", []) or []

    # Player-only totals (picks are ignored here on purpose)
    fewer_players_total = float(fewer.get("raw_players_total", 0.0) or 0.0)
    more_players_total = float(more.get("raw_players_total", 0.0) or 0.0)

    # Safety guard
    if not fewer_vals or fewer_players_total <= 0:
        side_a["effective_total"] = side_a["raw_total"]
        side_b["effective_total"] = side_b["raw_total"]
        side_a["adjustment"] = 0.0
        side_b["adjustment"] = 0.0
        return

    extra_pieces = max(len(more_vals) - len(fewer_vals), 0)
    if extra_pieces <= 0:
        # Shouldn't happen given earlier check, but be safe
        side_a["effective_total"] = side_a["raw_total"]
        side_b["effective_total"] = side_b["raw_total"]
        side_a["adjustment"] = 0.0
        side_b["adjustment"] = 0.0
        return

    # How big is the stud relative to the consolidating side?
    stud_val = max(fewer_vals)
    stud_share = stud_val / max(fewer_players_total, 1.0)  # 0–1
    stud_share = max(0.0, min(stud_share, 1.0))

    # Gap in *player* value between sides
    player_gap = abs(more_players_total - fewer_players_total)

    # --- Adjustment recipe ---
    # 1. Base from player_gap, scaled heavier when stud dominates the side.
    #    (about 30–70% of the player gap)
    base_from_gap = player_gap * (0.30 + 0.40 * stud_share)

    # 2. Extra multiplier for more pieces; 1 extra piece ~0.4, 2 ~0.6, 3+ ~0.8
    piece_factor = 0.4 + 0.2 * min(extra_pieces, 3)

    raw_adj = base_from_gap * piece_factor

    # 3. Caps so it never blows up:
    #    - at most 60% of the stud
    #    - at most 35% of the consolidating side's total *player* value
    cap_stud = 0.60 * stud_val
    cap_side = 0.35 * fewer_players_total
    adj_cap = max(0.0, min(cap_stud, cap_side))

    adj = min(raw_adj, adj_cap)

    # Apply to fewer-players side only; picks stay baked into raw_total
    if fewer_is_a:
        side_a["adjustment"] = adj
        side_b["adjustment"] = 0.0
    else:
        side_a["adjustment"] = 0.0
        side_b["adjustment"] = adj

    side_a["effective_total"] = side_a["raw_total"] + side_a["adjustment"]
    side_b["effective_total"] = side_b["raw_total"] + side_b["adjustment"]


def render_weekly_top_scorers_for_week(
        league_id: str,
        df_weekly: pd.DataFrame,
        roster_map: dict,
        players_map: dict,
        projections: dict,  # <–– pass ALL projections in once
        rosters: dict,
        w: int,
        users: list,
        platform: str,
        season: str
) -> str:
    # 1. Filter to ONLY this week
    week_df = df_weekly[df_weekly["week"] == w].copy()
    # --------------------------------------------
    # CASE 1: Week finalized → use real scores
    # --------------------------------------------
    if not week_df.empty and week_df["points"].any():
        _, _, top_by_pos = matchup_cards_last_week(
            league_id,
            week_df,
            roster_map,
            players_map,
            rosters,
            users,
            platform,
            season
        )
        return render_top_three(top_by_pos, rosters, roster_map)

    # --------------------------------------------
    # CASE 2: Week not finalized → use projections for this week
    # --------------------------------------------
    if projections is None:
        empty = {pos: [] for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]}
        return render_top_three(empty, rosters, roster_map)

    # Build projected rows
    proj_rows = []
    week_projection_bundle = projections.get(w, {}) or {}

    for _, proj in week_projection_bundle.items():
        if not isinstance(proj, dict):
            continue
        for pid, val in proj.items():
            p = players_map.get(str(pid))
            if not p:
                continue

            pos = p.get("position") or p.get("pos")
            if pos not in ["QB", "RB", "WR", "TE", "K", "DEF"]:
                continue

            proj_rows.append({
                "pid": pid,
                "name": p.get("name", "Unknown"),
                "pos": pos,
                "team": p.get("team", ""),
                "points": float(val),
            })

    top_by_pos = {pos: [] for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]}

    for pos in top_by_pos:
        f = [r for r in proj_rows if r["pos"] == pos]
        f.sort(key=lambda r: r["points"], reverse=True)
        top_by_pos[pos] = f[:3]

    return render_top_three(top_by_pos, rosters, roster_map)


def _render_weekly_matchups(df_weekly: pd.DataFrame, week: int) -> str:
    wdf = df_weekly[df_weekly["week"] == week].copy()
    if wdf.empty:
        return ""

    rows = []
    for (wk, mid), grp in wdf.groupby(["week", "matchup_id"]):
        if len(grp) != 2:
            continue
        g = grp.sort_values("points", ascending=False)
        win = g.iloc[0]
        lose = g.iloc[1]
        margin = float(win["points"] - lose["points"])

        rows.append(
            f"<div class='matchup-row'>"
            f"  <div class='m-team-col'>"
            f"    <div class='m-team-name winner'>{win['owner']}</div>"
            f"    <div class='m-score'>{float(win['points']):.1f}</div>"
            f"  </div>"
            f"  <div class='m-vs-col'>def</div>"
            f"  <div class='m-team-col'>"
            f"    <div class='m-team-name loser'>{lose['owner']}</div>"
            f"    <div class='m-score'>{float(lose['points']):.1f}</div>"
            f"  </div>"
            f"  <div class='m-margin'>+{margin:.1f}</div>"
            f"</div>"
        )

    return f"""
    <div class="card">
      <div class="card-header">
        <h2>Week {week} Matchups</h2>
      </div>
      <div class="card-body matchup-list">
        {''.join(rows)}
      </div>
    </div>
    """


def _render_weekly_highlights(df_weekly: pd.DataFrame, week: int) -> str:
    wdf = df_weekly[df_weekly["week"] == week].copy()
    if wdf.empty:
        return f"""
        <div class='card small'>
          <div class='card-header'><h3>Week {week} Highlights</h3></div>
          <div class='card-body'>
            <p>No highlights for this week yet.</p>
          </div>
        </div>
        """

    # ------------------------------------------------------------
    # Use projections for weeks that are NOT finalized
    # ------------------------------------------------------------
    if not wdf["finalized"].any():
        wdf["use_score"] = wdf["proj"]
    else:
        wdf["use_score"] = wdf["points"]

    # ------------------------------------------------------------
    # Highest / Lowest Score Cards
    # ------------------------------------------------------------
    top = wdf.sort_values("use_score", ascending=False).iloc[0]
    low = wdf.sort_values("use_score", ascending=True).iloc[0]

    highest_card = f"""
    <div class="card small">
      <div class="card-header"><h3>Highest Score</h3></div>
      <div class="card-body">
        <div class="highlight-game-card white">
          <div class="hg-row">
            <div class="hg-team">
              <span class="hg-name">{top['owner']}</span>
            </div>
            <div class="hg-score">{top['use_score']:.1f}</div>
          </div>
        </div>
      </div>
    </div>
    """

    lowest_card = f"""
    <div class="card small">
      <div class="card-header"><h3>Lowest Score</h3></div>
      <div class="card-body">
        <div class="highlight-game-card white">
          <div class="hg-row">
            <div class="hg-team">
              <span class="hg-name">{low['owner']}</span>
            </div>
            <div class="hg-score">{low['use_score']:.1f}</div>
          </div>
        </div>
      </div>
    </div>
    """

    # ------------------------------------------------------------
    # Closest Game / Blowout Game
    # ------------------------------------------------------------
    matchups = []
    for (_, _), grp in wdf.groupby(["week", "matchup_id"]):
        if len(grp) != 2:
            continue
        g = grp.sort_values("use_score", ascending=False)
        win = g.iloc[0]
        lose = g.iloc[1]
        margin = float(win["use_score"] - lose["use_score"])

        matchups.append({
            "winner": win["owner"],
            "winnerPts": float(win["use_score"]),
            "loser": lose["owner"],
            "loserPts": float(lose["use_score"]),
            "margin": margin,
        })

    closest_card = ""
    blowout_card = ""

    if matchups:
        closest = min(matchups, key=lambda m: abs(m["margin"]))
        blowout = max(matchups, key=lambda m: abs(m["margin"]))

        closest_card = f"""
        <div class="card small">
          <div class="card-header">
            <h3>Closest Game</h3><h3>{closest['margin']:.1f} Points</h3>
          </div>
          <div class="card-body">
            <div class="highlight-game-card white">
              <div class="hg-row">
                <span class="hg-name">{closest['winner']}</span>
                <span class="hg-score">{closest['winnerPts']:.1f}</span>
              </div>
              <div class="hg-row">
                <span class="hg-name">{closest['loser']}</span>
                <span class="hg-score">{closest['loserPts']:.1f}</span>
              </div>
            </div>
          </div>
        </div>
        """

        blowout_card = f"""
        <div class="card small">
          <div class="card-header">
            <h3>Biggest Blowout</h3><h3>{blowout['margin']:.1f} Points</h3>
          </div>
          <div class="card-body">
            <div class="highlight-game-card white">
              <div class="hg-row">
                <span class="hg-name">{blowout['winner']}</span>
                <span class="hg-score">{blowout['winnerPts']:.1f}</span>
              </div>
              <div class="hg-row">
                <span class="hg-name">{blowout['loser']}</span>
                <span class="hg-score">{blowout['loserPts']:.1f}</span>
              </div>
            </div>
          </div>
        </div>
        """

    return highest_card + lowest_card + closest_card + blowout_card


def build_weekly_hub_body(ctx: dict) -> str:
    import json

    league_id = ctx["league_id"]
    platform = ctx["platform"]
    season = ctx["season"]  # viewed season
    rosters = ctx["rosters"]
    users = ctx["users"]
    df_weekly = ctx["df_weekly"]
    roster_map = ctx["roster_map"]
    players_map = ctx["players_map"]
    current_week = int(ctx.get("current_week") or 0)
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    proj_by_week = ctx["proj_by_week"]
    weeks = int(ctx["weeks"])
    statuses = ctx["statuses"]
    team_game_lookup = ctx["team_game_lookup"]
    matchups_by_week = ctx["matchups_by_week"]
    season_complete = bool(ctx.get("season_complete", False))
    offseason_mode = bool(ctx.get("offseason_mode", False))

    if (
            df_weekly is not None
            and not df_weekly.empty
            and "finalized" in df_weekly.columns
            and "week" in df_weekly.columns
    ):
        finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    else:
        finalized_df = pd.DataFrame()

    if not finalized_df.empty and "week" in finalized_df.columns:
        last_final_week = int(finalized_df["week"].max())
    else:
        last_final_week = max(1, min(current_week or 1, weeks))

    max_week = max(1, weeks)

    def clamp_week(w: int) -> int:
        return max(1, min(max_week, int(w)))

    if season_complete or offseason_mode:
        default_week = clamp_week(last_final_week)
    else:
        default_week = clamp_week(current_week or 1)

    default_matchups = matchups_by_week.get(default_week, []) or []
    slides = [
        render_matchup_slide(
            season,
            m,
            default_week,
            last_final_week,
            status_by_pid=(statuses.get(default_week) or {}).get("statuses", {}) or {},
            projections=proj_by_week,
            players=players_index,
            teams=teams_index,
            team_game_lookup=team_game_lookup,
        )
        for m in default_matchups
    ]
    slides_html = "".join(slides) if slides else "<div class='m-empty'>No matchups</div>"
    slides_by_week = {default_week: slides_html}

    matchup_html = render_matchup_carousel_weeks(
        slides_by_week,
        dashboard=False,
        active_week=default_week,
    )

    options = []
    for w in range(1, max_week + 1):
        sel = " selected" if w == default_week else ""
        options.append(f"<option value='{w}'{sel}>Week {w}</option>")
    week_select_html = "".join(options)

    top_scorers_html = render_weekly_top_scorers_for_week(
        league_id,
        df_weekly,
        roster_map,
        players_map,
        proj_by_week,
        rosters,
        default_week,
        users,
        platform,
        season
    )
    highlights_html = _render_weekly_highlights(df_weekly, default_week)

    main_panel_html = f"""
          <div class="week-main-panel active" data-week="{default_week}">
            {top_scorers_html}
          </div>
    """
    side_panel_html = f"""
          <div class="week-side-panel active" data-week="{default_week}">
            {highlights_html}
          </div>
    """

    platform_js = json.dumps(platform)
    season_js = json.dumps(season)
    league_js = json.dumps(league_id)

    return f"""
    <div class="page-layout weekly-hub">
      <main class="page-main">
        <div class="card">
          <div class="card-header-row">
            <h2>Weekly Hub</h2>
            <div class="week-selector">
              <select id="hubWeek" class="search">
                {week_select_html}
              </select>
            </div>
          </div>
        </div>

        <div class="standings-main two-col-standings">
          <div class="standings-col">
            <div class="week-main-panels">
              {main_panel_html}
            </div>
          </div>
          <div class="standings-col">
            <div class="matchups-shell">
              <div id="weeklyMatchupsContainer">
                {matchup_html}
              </div>
              <div id="weeklyMatchupsLoading" class="matchups-loading hidden">
                <div class="matchups-loading-inner">
                  <div class="matchups-spinner"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <aside class="page-sidebar">
        <div class="week-side-panels">
          {side_panel_html}
        </div>
      </aside>
    </div>

<script>
(function() {{
  var leagueId  = {league_js};
  var platform  = {platform_js};
  var season    = {season_js};

  var sel = document.getElementById('hubWeek');
  if (!sel) return;
  if (sel.__hubWeekBound) return;
  sel.__hubWeekBound = true;

  var matchupsContainer = document.getElementById('weeklyMatchupsContainer');
  var loadingOverlay    = document.getElementById('weeklyMatchupsLoading');
  var mainContainer = document.querySelector('.week-main-panels');
  var sideContainer = document.querySelector('.week-side-panels');

  function showLoading() {{
    if (loadingOverlay) loadingOverlay.classList.remove('hidden');
    sel.disabled = true;
  }}

  function hideLoading() {{
    if (loadingOverlay) loadingOverlay.classList.add('hidden');
    sel.disabled = false;
  }}

  var controller = null;
  var requestSeq = 0;

  sel.addEventListener('change', function() {{
    var w = String(this.value || '');
    if (!w) return;

    if (controller) {{
      try {{ controller.abort(); }} catch (e) {{}}
    }}
    controller = (window.AbortController ? new AbortController() : null);

    var mySeq = ++requestSeq;
    showLoading();

    var url =
      '/api/weekly-week?platform=' + encodeURIComponent(platform) +
      '&season=' + encodeURIComponent(season) +
      '&league_id=' + encodeURIComponent(leagueId) +
      '&week=' + encodeURIComponent(w);

    fetch(url, {{
      signal: controller ? controller.signal : undefined
    }})
      .then(function(res) {{
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      }})
      .then(function(data) {{
        if (mySeq !== requestSeq) return;
        if (!data || !data.ok) {{
          console.error('Failed to load week', w, data && data.error);
          return;
        }}

        if (mainContainer && typeof data.top_html === 'string') {{
          mainContainer.innerHTML =
            '<div class="week-main-panel active" data-week="' + w + '">' +
              data.top_html +
            '</div>';
        }}

        if (sideContainer && typeof data.highlights_html === 'string') {{
          sideContainer.innerHTML =
            '<div class="week-side-panel active" data-week="' + w + '">' +
              data.highlights_html +
            '</div>';
        }}

        if (matchupsContainer && typeof data.matchups_html === 'string') {{
          matchupsContainer.innerHTML = data.matchups_html;

          if (typeof window.resetMatchupCarousels === 'function') {{
            window.resetMatchupCarousels(matchupsContainer);
          }}
          if (typeof window.initPageRoot === 'function') {{
            window.initPageRoot(matchupsContainer);
          }}
        }}
      }})
      .catch(function(err) {{
        if (err && err.name === 'AbortError') return;
        console.error('Error fetching week', w, err);
      }})
      .finally(function() {{
        if (mySeq === requestSeq) hideLoading();
      }});
  }});
}})();
</script>
"""


def build_projections_by_week(season: int, weeks: int):
    bundles = {}
    for w in range(1, weeks + 1):
        try:
            projections = load_week_projection(season, w)
            bundles[w] = {"projections": projections}
        except Exception as e:
            print(f"Error loading week {w} projections: {e}")
            bundles[w] = {"projections": {}}
    return bundles


def build_status_by_week(season: int, weeks: int, players_index, teams_index, idp_player_index: dict[str, dict] = None):
    bundles = {}
    for w in range(1, weeks + 1):
        try:
            statuses = build_status_for_week(season, w, players_index, teams_index, idp_player_index)
            bundles[w] = {"statuses": statuses}
        except Exception as e:
            print(f"Error loading week {w} schedule: {e}")
            bundles[w] = {"statuses": {}}
    return bundles


def build_activity_body(ctx: dict) -> str:
    league_id = ctx["league_id"]
    resolved_league_id = ctx.get("resolved_league_id", league_id)
    activity_df = ctx["activity_df"]
    injury_df = ctx["injury_df"]
    standings_map = ctx["standings_map"]
    platform = ctx["platform"]
    season = ctx["season"]

    players_values_raw = ctx.get("model_value_table") or []
    player_val_by_key: dict[tuple[str, str, str], float] = {}
    player_val_by_key_np: dict[tuple[str, str], float] = {}
    rank_label_by_name: dict[str, str] = {}

    if isinstance(players_values_raw, list):
        for row in players_values_raw:
            if not isinstance(row, dict):
                continue
            raw_name = str(row.get("search_name") or "").strip()
            if not raw_name:
                continue
            name_lower = raw_name.lower()
            pos = str(row.get("position") or row.get("pos") or "").strip().upper()
            team = str(row.get("team") or "").strip().upper()
            if not pos:
                continue
            try:
                val = float(row.get("value") or 0.0)
            except Exception:
                val = 0.0

            player_val_by_key[(name_lower, pos, team)] = val
            player_val_by_key_np[(name_lower, pos)] = val

            lbl = row.get("pos_rank_label") or pos
            rank_label_by_name[name_lower] = str(lbl)

    def player_value(p: dict) -> tuple[float, str]:
        name = str(p.get("name") or "").strip()
        name_lower = name.lower()
        pos = str(p.get("pos") or p.get("position") or "").strip().upper()
        team = str(p.get("team") or "").strip().upper()
        if not name or not pos:
            return 0.0, ""

        val = float(
            player_val_by_key.get((name_lower, pos, team))
            or player_val_by_key_np.get((name_lower, pos), 0.0)
        )

        rank_label = rank_label_by_name.get(name_lower, pos)
        return val, rank_label

    pick_values = load_pick_value_table() or {}

    def pick_bucket_from_seed(seed: Optional[int], num_teams: int = 10) -> Optional[str]:
        if seed is None:
            return None
        if 1 <= seed <= 3:
            return "early"
        if 4 <= seed <= 7:
            return "mid"
        if 8 <= seed <= num_teams:
            return "late"
        return None

    def pick_value(pick: dict, standings_map: dict[int, int], num_teams: int = 10) -> float:
        try:
            year = int(pick.get("season") or 0)
            rnd = int(pick.get("round") or 0)
        except Exception:
            return 0.0
        if not year or not rnd:
            return 0.0

        prev_owner = pick.get("previous_owner_id")
        seed = None
        try:
            if prev_owner is not None:
                seed = standings_map.get(int(prev_owner))
        except Exception:
            seed = None

        bucket = pick_bucket_from_seed(seed, num_teams=num_teams)

        if bucket:
            key_bucket = f"{year}_{rnd}_{bucket}"
            if key_bucket in pick_values:
                return float(pick_values[key_bucket])
            key_generic = f"{year}_{rnd}"
            if key_generic in pick_values:
                return float(pick_values[key_generic])

        for b in ("mid", "early", "late"):
            key = f"{year}_{rnd}_{b}"
            if key in pick_values:
                return float(pick_values[key])

        key_generic = f"{year}_{rnd}"
        if key_generic in pick_values:
            return float(pick_values[key_generic])

        return 0.0

    def pick_subline(pick: dict, rid_to_name: dict, users: list, num_teams: int = 10) -> str:
        prev_owner = pick.get("previous_owner_id")
        seed = None
        try:
            if prev_owner is not None:
                seed = standings_map.get(int(prev_owner))
        except Exception:
            seed = None

        bucket = pick_bucket_from_seed(seed, num_teams=num_teams)
        bucket_label = bucket.capitalize() if bucket else None

        orig_rid = pick.get("roster_id")
        orig_team = rid_to_name.get(orig_rid, f"User {orig_rid}") if orig_rid is not None else "Unknown"
        orig_name = next(
            (
                u.get("display_name")
                for u in users
                if u.get("metadata", {}).get("team_name") == orig_team
            ),
            None
        )

        owner_txt = f"from {orig_name}" if orig_name else "Traded Pick"
        return f"{bucket_label} • {owner_txt}" if bucket_label else owner_txt

    def verdict_from_net(net_total: float) -> tuple[str, str]:
        if net_total >= 20:
            return "bract-verdict-win", "Strong win"
        if net_total >= 8:
            return "bract-verdict-win", "Slight win"
        if net_total <= -20:
            return "bract-verdict-loss", "Strong loss"
        if net_total <= -8:
            return "bract-verdict-loss", "Slight loss"
        return "bract-verdict-even", "Fair"

    trade_count = 0
    waiver_count = 0
    most_active_counts: dict[str, int] = {}
    traded_asset_counts: dict[str, int] = {}
    biggest_trade_label = "No trade data"
    biggest_trade_delta = 0.0

    activity_html = ""
    if activity_df is not None and not activity_df.empty:

        def html_trade(txrow):
            nonlocal trade_count, biggest_trade_label, biggest_trade_delta

            data = txrow["data"]
            teams = data["teams"]
            users = get_users(platform, resolved_league_id, season)

            rid_to_name = {}
            for tm in teams:
                rid = tm.get("roster_id")
                if rid is not None:
                    rid_to_name[rid] = tm.get("name") or f"Team {rid}"
                team_name = tm.get("name") or f"Team {rid}"
                most_active_counts[team_name] = most_active_counts.get(team_name, 0) + 1

            trade_count += 1

            def render_player_row(p, io_class):
                name = str(p.get("name") or "").strip()
                if name:
                    traded_asset_counts[name] = traded_asset_counts.get(name, 0) + 1

                val, pos_rank_label = player_value(p)
                val_txt = f"{val:.1f}" if val > 0 else ""
                val_html = f'<div class="player-trade-value">{val_txt}</div>' if val_txt else ""
                return (
                    "<div class='player-activity'>"
                    "<div style='gap: 10px;display: flex;align-items: center;'>"
                    f"<span class='io {io_class}'>"
                    f"{'+' if io_class == 'add' else '−'}</span>"
                    "<div>"
                    f"  <div style='font-weight:600'>{p['name']}</div>"
                    f"  <div style='color:#64748b;font-size:12px'>{pos_rank_label} • {p['team']}</div>"
                    "</div></div>"
                    f"{val_html}</div>"
                )

            def render_pick_row(pick, io_class):
                traded_asset_counts["Draft Pick"] = traded_asset_counts.get("Draft Pick", 0) + 1

                rnd_suffix = {1: "st", 2: "nd", 3: "rd"}.get(pick.get("round"), "th")
                round_label = f"{pick.get('round')}" + rnd_suffix
                pick_season = str(pick.get("season") or "")
                subline = pick_subline(pick, rid_to_name, users)
                val = pick_value(pick, standings_map)
                val_txt = f"{val:.1f}" if val > 0 else ""
                val_html = f'<div class="player-trade-value">{val_txt}</div>' if val_txt else ""
                return (
                    "<div class='player-activity'>"
                    "<div style='gap: 10px;display: flex;align-items: center;'>"
                    f"<span class='io {io_class}'>"
                    f"{'+' if io_class == 'add' else '−'}</span>"
                    "<div>"
                    f"  <div style='font-weight:600'>{pick_season} {round_label}</div>"
                    f"  <div style='color:#64748b;font-size:12px'>{subline}</div>"
                    "</div></div>"
                    f"{val_html}</div>"
                )

            draft_picks = data.get("draft_picks", []) or []
            picks_by_receiver = {}
            picks_by_sender = {}
            for dp in draft_picks:
                recv = dp.get("owner_id")
                send = dp.get("previous_owner_id")
                if recv is not None:
                    picks_by_receiver.setdefault(recv, []).append(dp)
                if send is not None:
                    picks_by_sender.setdefault(send, []).append(dp)

            side_map: dict[int, dict] = {}
            for tm in teams:
                rid = tm.get("roster_id")
                if rid is None:
                    continue
                in_players = tm.get("gets") or []
                in_picks = picks_by_receiver.get(rid, []) or []

                in_player_pairs = [player_value(p) for p in in_players]
                in_player_vals = [v for (v, _label) in in_player_pairs]
                in_pick_vals = [pick_value(pk, standings_map) for pk in in_picks]

                raw_players_total = sum(in_player_vals)
                raw_picks_total = sum(in_pick_vals)
                raw_total = raw_players_total + raw_picks_total

                side_map[rid] = {
                    "raw_total": raw_total,
                    "raw_players_total": raw_players_total,
                    "raw_picks_total": raw_picks_total,
                    "player_values": in_player_vals,
                    "breakdown": [],
                    "adjustment": 0.0,
                    "effective_total": raw_total,
                }

            if len(side_map) == 2:
                rid_list = list(side_map.keys())
                side_a = side_map[rid_list[0]]
                side_b = side_map[rid_list[1]]
                apply_multi_for_one_adjustment(side_a, side_b)

            net_values = []

            cols = []
            for tm in teams:
                roster_id = tm.get("roster_id")

                gets_parts = []
                for p in (tm.get("gets") or []):
                    gets_parts.append(render_player_row(p, "add"))
                gets_players = "".join(gets_parts)

                gets_pick_parts = []
                if roster_id is not None:
                    for pick in picks_by_receiver.get(roster_id, []):
                        gets_pick_parts.append(render_pick_row(pick, "add"))
                gets_picks = "".join(gets_pick_parts)
                gets = gets_players + gets_picks
                if not gets:
                    gets = "<div class='bract-empty-mini'>No incoming assets</div>"

                sends_parts = []
                for p in (tm.get("sends") or []):
                    sends_parts.append(render_player_row(p, "drop"))
                sends_players = "".join(sends_parts)

                sends_pick_parts = []
                if roster_id is not None:
                    for pick in picks_by_sender.get(roster_id, []):
                        sends_pick_parts.append(render_pick_row(pick, "drop"))
                sends_picks = "".join(sends_pick_parts)
                sends = sends_players + sends_picks

                side_info = side_map.get(roster_id)
                eff_in = side_info["effective_total"] if side_info else 0.0

                out_total = 0.0
                for p in (tm.get("sends") or []):
                    out_total += player_value(p)[0]
                if roster_id is not None:
                    for pick in picks_by_sender.get(roster_id, []):
                        out_total += pick_value(pick, standings_map)

                net_total = eff_in - out_total
                net_values.append((tm.get("name", ""), net_total))

                verdict_cls, verdict_txt = verdict_from_net(net_total)
                net_num_cls = (
                    "bract-net-pos" if net_total > 0 else
                    "bract-net-neg" if net_total < 0 else
                    "bract-net-even"
                )

                total_html = (
                    "<div class='trade-total-row bract-total-row'>"
                    "<hr style='margin-top:8px;margin-bottom:8px;border:none;border-top:1px solid #e2e8f0;'>"
                    "<div class='bract-total-head'>"
                    "<span>Total Value</span>"
                    f"<span class='{net_num_cls}'>{net_total:.0f}</span>"
                    "</div>"
                    f"<div class='bract-verdict {verdict_cls}'>{verdict_txt}</div>"
                    "</div>"
                )

                avatar = tm.get("avatar") or ""
                img = (
                    f"<img class='avatar' src='{avatar}' "
                    "onerror=\"this.style.display='none'\">"
                    if avatar else ""
                )
                cols.append(
                    "<div class='team-col'>"
                    f"  <header>{img}<div class='team-name'>{tm.get('name', '')}</div></header>"
                    f"  <div class='plist'>{gets}{sends}{total_html}</div>"
                    "</div>"
                )

            if len(net_values) == 2:
                delta = abs(net_values[0][1] - net_values[1][1])
                if delta > biggest_trade_delta:
                    biggest_trade_delta = delta
                    biggest_trade_label = f"{net_values[0][0]} vs {net_values[1][0]}"

            when = (
                txrow["ts"].astimezone(ZoneInfo("America/New_York")).strftime("%b %d, %I:%M %p")
                if pd.notna(txrow["ts"])
                else ""
            )
            return (
                "<div class='tx trade-card activity-item' data-kind='trade'>"
                f"  <div class='meta'>{pill('Trade completed')} • {when}</div>"
                f"  <div class='teams'>{''.join(cols)}</div>"
                "</div>"
            )

        def html_waiver(txrow):
            nonlocal waiver_count

            d = txrow["data"]
            team_name = d.get("name") or "Unknown Team"
            most_active_counts[team_name] = most_active_counts.get(team_name, 0) + 1
            waiver_count += 1

            avatar = d.get("avatar") or ""
            img = (
                f"<img class='avatar' src='{avatar}' "
                "onerror=\"this.style.display='none'\">"
                if avatar else ""
            )
            adds_parts = []
            for p in d.get("adds", []):
                name = str(p.get("name") or "").strip()
                if name:
                    traded_asset_counts[name] = traded_asset_counts.get(name, 0) + 1

                val, pos_rank_label = player_value(p)
                val_txt = f"{val:.1f}" if val > 0 else ""
                val_html = f'<div class="player-trade-value">{val_txt}</div>' if val_txt else ""
                adds_parts.append(
                    "<div class='player-activity'>"
                    "<div style='gap: 10px;display: flex;align-items: center;'>"
                    "<span class='io add'>+</span>"
                    "<div>"
                    f"  <div style='font-weight:600'>{p['name']}</div>"
                    f"  <div style='color:#64748b;font-size:12px'>{pos_rank_label} • {p['team']}</div>"
                    "</div></div>"
                    f"{val_html}</div>"
                )
            adds = "".join(adds_parts) or "<div class='bract-empty-mini'>No adds recorded</div>"

            when = (
                txrow["ts"].astimezone(ZoneInfo("America/New_York")).strftime("%b %d, %I:%M %p")
                if pd.notna(txrow["ts"])
                else ""
            )
            return (
                "<div class='tx activity-item' data-kind='waiver'>"
                f"  <div class='meta'>{pill('Waiver')} • {when}</div>"
                "  <div class='team-col'>"
                f"    <header>{img}<div class='team-name'>{team_name}</div></header>"
                f"    <div class='plist'>{adds}</div>"
                "  </div>"
                "</div>"
            )

        cards = []
        for _, row in activity_df.iterrows():
            cards.append(html_trade(row) if row["kind"] == "trade" else html_waiver(row))

        most_active_team = max(most_active_counts.items(), key=lambda x: x[1])[0] if most_active_counts else "None"
        most_moved_asset = max(traded_asset_counts.items(), key=lambda x: x[1])[0] if traded_asset_counts else "None"

        summary_html = (
            "<div class='bract-summary-grid'>"
            f"  <div class='bract-summary-card'><div class='bract-summary-label'>Trades</div><div class='bract-summary-value'>{trade_count}</div></div>"
            f"  <div class='bract-summary-card'><div class='bract-summary-label'>Waivers</div><div class='bract-summary-value'>{waiver_count}</div></div>"
            f"  <div class='bract-summary-card'><div class='bract-summary-label'>Most Active</div><div class='bract-summary-value bract-summary-text'>{most_active_team}</div></div>"
            f"  <div class='bract-summary-card'><div class='bract-summary-label'>Most Moved Asset</div><div class='bract-summary-value bract-summary-text'>{most_moved_asset}</div></div>"
            "</div>"
        )

        trade_spotlight = (
            "<div class='bract-spotlight'>"
            "  <div class='bract-spotlight-title'>Recent activity snapshot</div>"
            f"  <div class='bract-spotlight-copy'>Biggest recent trade: <strong>{biggest_trade_label}</strong>"
            f"  {'(' + str(round(biggest_trade_delta, 1)) + ' value swing)' if biggest_trade_delta > 0 else ''}</div>"
            "</div>"
        )

        activity_html = (
            "<div class='card activity-card' data-section='activity'>"
            "  <div class='card-header-row'>"
            "    <h2>Trades & Waiver Claims</h2>"
            "  </div>"
            f"  {summary_html}"
            f"  {trade_spotlight}"
            "  <div class='scroll-box'>"
            "    <div class='feed'>"
            f"      {''.join(cards)}"
            "    </div>"
            "  </div>"
            "</div>"
        )

    injury_html = ""
    if injury_df is not None and not injury_df.empty:
        injury_html = render_injury_accordion(injury_df)
    else:
        injury_html = (
            "<div class='card'>"
            "  <div class='card-body'>"
            "    <div class='bract-empty-state'>"
            "      <div class='bract-empty-title'>No injury data right now</div>"
            "      <div class='bract-empty-copy'>Either the feed is quiet or there are no currently tracked injury updates for this view.</div>"
            "    </div>"
            "  </div>"
            "</div>"
        )

    if not activity_html:
        activity_html = (
            "<div class='card'>"
            "  <div class='card-body'>"
            "    <div class='bract-empty-state'>"
            "      <div class='bract-empty-title'>No recent activity yet</div>"
            "      <div class='bract-empty-copy'>When trades and waiver claims come through, they’ll show up here with value context and team-by-team breakdowns.</div>"
            "    </div>"
            "  </div>"
            "</div>"
        )

    return f"""
    <div class="page-layout activity-page">
      <main class="page-main activity-main">
        <div class="activity-col">
          {activity_html}
        </div>
        <div class="injury-col">
          {injury_html}
        </div>
      </main>

      <aside class="page-sidebar">
        <div class="card small">
          <div class="card-header">
            <h3>Filters</h3>
          </div>
          <div class="card-body">
            <label class="mini-label">Activity Types</label>
            <div class="pill-row">
              <button class="pill-toggle act-toggle active" data-kind="waiver">Waivers</button>
              <button class="pill-toggle act-toggle active" data-kind="trade">Trades</button>
            </div>

            <label class="mini-label" style="margin-top:12px;">Injury Status</label>
            <div class="pill-row">
              <button class="pill-toggle inj-toggle active" data-status="all">All</button>
              <button class="pill-toggle inj-toggle" data-status="IR">IR</button>
              <button class="pill-toggle inj-toggle" data-status="OUT">Out</button>
              <button class="pill-toggle inj-toggle" data-status="QUESTIONABLE">Q</button>
            </div>
          </div>
        </div>
      </aside>
    </div>

    <style>
      .bract-summary-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin: 0 0 12px 0;
      }}

      .bract-summary-card {{
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        border-radius: 12px;
        padding: 12px 14px;
      }}

      .bract-summary-label {{
        font-size: 11px;
        line-height: 1.2;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 6px;
        font-weight: 700;
      }}

      .bract-summary-value {{
        font-size: 24px;
        line-height: 1.1;
        font-weight: 800;
        color: #0f172a;
      }}

      .bract-summary-text {{
        font-size: 16px;
        line-height: 1.3;
      }}

      .bract-spotlight {{
        border: 1px solid #dbeafe;
        background: #eff6ff;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 14px;
      }}

      .bract-spotlight-title {{
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #1d4ed8;
        margin-bottom: 4px;
      }}

      .bract-spotlight-copy {{
        font-size: 14px;
        color: #1e293b;
      }}

      .bract-total-row {{
        padding-bottom: 2px;
      }}

      .bract-total-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        font-size: 14px;
        font-weight: 700;
        color: #0f172a;
      }}

      .bract-net-pos {{
        color: #15803d;
      }}

      .bract-net-neg {{
        color: #b91c1c;
      }}

      .bract-net-even {{
        color: #475569;
      }}

      .bract-verdict {{
        display: inline-flex;
        align-items: center;
        margin-top: 8px;
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
      }}

      .bract-verdict-win {{
        background: #dcfce7;
        color: #166534;
      }}

      .bract-verdict-loss {{
        background: #fee2e2;
        color: #991b1b;
      }}

      .bract-verdict-even {{
        background: #e2e8f0;
        color: #334155;
      }}

      .bract-empty-state {{
        padding: 10px 4px;
      }}

      .bract-empty-title {{
        font-size: 18px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 6px;
      }}

      .bract-empty-copy {{
        font-size: 14px;
        line-height: 1.5;
        color: #64748b;
      }}

      .bract-empty-mini {{
        color: #64748b;
        font-size: 13px;
      }}

      @media (max-width: 900px) {{
        .bract-summary-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>

    <script>
    (function() {{
      document.querySelectorAll('.act-toggle').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          this.classList.toggle('active');
          const activeKinds = Array.from(document.querySelectorAll('.act-toggle.active'))
            .map(b => b.getAttribute('data-kind'));

          document.querySelectorAll('.activity-item').forEach(function(item) {{
            const k = item.getAttribute('data-kind');
            item.style.display = activeKinds.length === 0 || activeKinds.includes(k)
              ? ''
              : 'none';
          }});
        }});
      }});

      document.querySelectorAll('.inj-toggle').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          document.querySelectorAll('.inj-toggle').forEach(b => b.classList.remove('active'));
          this.classList.add('active');

          const status = this.getAttribute('data-status');
          const rows = document.querySelectorAll('.inj-row');

          rows.forEach(function(row) {{
            if (status === 'all') {{
              row.style.display = '';
              return;
            }}
            const chips = row.querySelectorAll('.chip');
            let matched = false;
            chips.forEach(function(c) {{
              if (c.textContent.trim().toUpperCase() === status) {{
                matched = true;
              }}
            }});
            row.style.display = matched ? '' : 'none';
          }});
        }});
      }});
    }})();
    </script>
    """


def render_pos_section(rid: int, pos_label: str, pos_code: str) -> str:
    plist = roster_pos_players.get(rid, {}).get(pos_code, [])
    if not plist:
        return ""  # no block if they have no players at that position

    rows_html = []
    for p in plist:
        val = float(p.get("value", 0.0))
        val_txt = f"{val:.1f}" if val > 0 else ""
        rows_html.append(
            # reuse your same flex layout style as activity tab
            f"<div class='player-activity'>"
            f"  <div style='display:flex;align-items:center;justify-content:space-between;width:100%'>"
            f"    <div>"
            f"      <div style='font-weight:600'>{p.get('name', '')}</div>"
            f"      <div style='color:#64748b;font-size:12px'>"
            f"        {p.get('position', '')} • {p.get('team', '')}"
            f"      </div>"
            f"    </div>"
            f"    <div class='player-trade-value'>{val_txt}</div>"
            f"  </div>"
            f"</div>"
        )

    return (
        f"<div class='pos-group'>"
        f"  <div class='pos-header'>{pos_label}</div>"
        f"  <div class='pos-list'>{''.join(rows_html)}</div>"
        f"</div>"
    )


def build_teams_body(ctx: dict) -> str:
    """
    Teams page:
      - One card per team
      - Within each card:
          * positional strength table (value + z-score + bar)
          * each position row can expand to show that position's players + values
      - Positional Index summary per team in header
    """
    rosters = ctx["rosters"]  # Sleeper /rosters
    roster_map = ctx["roster_map"]  # mapping roster_id -> team name
    users = ctx["users"]
    platform = ctx["platform"]

    # ----------------- Load value table -----------------
    # Expected rows like {id, name, position, team, value, search_name}
    model_vals = ctx.get("model_value_table") or []

    name_to_rank_label: dict[str, str] = {}
    name_to_age: dict[str, float | None] = {}

    for obj in model_vals:
        if not isinstance(obj, dict):
            continue
        safe_name = str(obj.get("search_name") or "").strip().lower()
        if not safe_name:
            continue
        pos_lbl = obj.get("pos_rank_label") or obj.get("position") or obj.get("pos") or ""
        name_to_rank_label[safe_name] = str(pos_lbl)
        age_val = obj.get("age")
        if age_val is not None:
            try:
                name_to_age[safe_name] = float(age_val)
            except Exception:
                name_to_age[safe_name] = None

    # map sleeper_id -> row
    by_id: dict[str, dict] = {
        str(p["id"]): p
        for p in model_vals
        if isinstance(p, dict) and p.get("id") is not None
    }

    CORE_POS = {"QB", "RB", "WR", "TE"}
    POS_ORDER = ["QB", "RB", "WR", "TE"]

    # ----------------- Roster → position → players (for dropdowns) -----------------
    roster_pos_players: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for r in rosters:
        rid = r.get("roster_id")
        if rid is None:
            continue
        try:
            rid_int = int(rid)
        except Exception:
            continue

        for pid in (r.get("players") or []):
            p = by_id.get(str(pid))
            if not p:
                continue
            pos = str(p.get("position") or p.get("pos") or "").upper()
            if pos == "PICK":
                continue
            if pos not in CORE_POS:
                continue  # only core positions in dropdown

            roster_pos_players[rid_int][pos].append(p)

    # sort each position bucket by value (high → low)
    for rid, pos_map in roster_pos_players.items():
        for pos, plist in pos_map.items():
            plist.sort(key=lambda x: float(x.get("value", 0.0)), reverse=True)

    # ----------------- Build per-team position value buckets (for strength table) -----------------
    team_meta: dict[int, dict] = {}  # name, avatar
    team_pos_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rosters:
        rid = r.get("roster_id")
        if rid is None:
            continue

        display_name = roster_map.get(str(rid)) if isinstance(roster_map, dict) else str(rid)
        avatar = avatar_from_users(platform, users, str(rid))
        team_meta[rid] = {
            "name": display_name,
            "avatar": avatar,
        }

        for pid in (r.get("players") or []):
            row = by_id.get(str(pid))
            if not row:
                continue
            pos = str(row.get("position") or row.get("pos") or "").upper()
            try:
                val = float(row.get("value") or 0.0)
            except Exception:
                val = 0.0
            if val <= 0:
                continue
            team_pos_values[rid][pos].append(val)

    # ensure every team has all core pos keys for the table
    for rid in team_meta.keys():
        for pos in POS_ORDER:
            team_pos_values[rid].setdefault(pos, [])

    # ----------------- Compute per-team averages + league baselines -----------------
    team_pos_avg: dict[int, dict[str, float]] = defaultdict(dict)

    for rid, pos_map in team_pos_values.items():
        for pos, vals in pos_map.items():
            if vals:
                team_pos_avg[rid][pos] = float(sum(vals) / len(vals))
            else:
                team_pos_avg[rid][pos] = 0.0

    league_pos_avg: dict[str, float] = {}
    league_pos_std: dict[str, float] = {}

    for pos in POS_ORDER:
        series = [team_pos_avg[rid][pos] for rid in team_meta.keys()]
        if not series:
            league_pos_avg[pos] = 0.0
            league_pos_std[pos] = 0.0
            continue
        mean = sum(series) / len(series)
        var = sum((x - mean) ** 2 for x in series) / len(series)
        std = math.sqrt(var)
        league_pos_avg[pos] = mean
        league_pos_std[pos] = std

    # ----------------- Z-scores & positional index -----------------
    team_pos_z: dict[int, dict[str, float]] = defaultdict(dict)
    team_pos_index: dict[int, float] = {}
    slot_counts = count_roster_positions(get_roster_positions())

    LINEUP_WEIGHTS = {
        "QB": slot_counts.get("QB") or 1,
        "RB": slot_counts.get("RB") or 2,
        "WR": slot_counts.get("WR") or 2,
        "TE": slot_counts.get("TE") or 1,
        "FLEX": slot_counts.get("FLEX") or 1,
    }
    weight_sum = sum(LINEUP_WEIGHTS[pos] for pos in POS_ORDER if LINEUP_WEIGHTS.get(pos, 0) > 0) or 1.0

    pos_z_min: dict[str, float] = {pos: float("inf") for pos in POS_ORDER}
    pos_z_max: dict[str, float] = {pos: float("-inf") for pos in POS_ORDER}

    for rid in team_meta.keys():
        idx_num = 0.0

        for pos in POS_ORDER:
            team_avg = team_pos_avg[rid][pos]
            mu = league_pos_avg[pos]
            sigma = league_pos_std[pos]
            if sigma > 0:
                z = (team_avg - mu) / sigma
            else:
                z = 0.0
            team_pos_z[rid][pos] = z

            pos_z_min[pos] = min(pos_z_min[pos], z)
            pos_z_max[pos] = max(pos_z_max[pos], z)

            w = LINEUP_WEIGHTS.get(pos, 0)
            idx_num += w * z

        team_pos_index[rid] = idx_num / weight_sum

    for pos in POS_ORDER:
        if pos_z_min[pos] == float("inf"):
            pos_z_min[pos] = 0.0
        if pos_z_max[pos] == float("-inf"):
            pos_z_max[pos] = 0.0

    # ----------------- Positional ranks (per position) -----------------
    # pos_rank[pos][rid] = rank (1 = best at that position)
    pos_rank: dict[str, dict[int, int]] = {pos: {} for pos in POS_ORDER}

    for pos in POS_ORDER:
        # rank by z-score (strongest to weakest)
        ranked = sorted(
            team_meta.keys(),
            key=lambda rid: team_pos_z[rid].get(pos, 0.0),
            reverse=True,
        )
        for i, rid in enumerate(ranked, start=1):
            pos_rank[pos][rid] = i

    # ----------------- Helper: players under a position row -----------------
    def render_pos_players(rid: int, pos_code: str) -> str:
        plist = roster_pos_players.get(rid, {}).get(pos_code, [])
        if not plist:
            return "<div style='color:#64748b;font-size:12px;'>No players at this position.</div>"

        rows_html = []
        for p in plist:
            name = p.get("name")
            name_raw = p.get('search_name', '')
            name_key = str(name_raw or "").strip().lower()

            rank_label = name_to_rank_label.get(
                name_key,
                p.get('position', '')
            )
            age = name_to_age.get(name_key)

            try:
                val = float(p.get("value") or 0.0)
            except Exception:
                val = 0.0
            val_txt = f"{val:.1f}" if val > 0 else ""

            rows_html.append(
                "<div class='player-activity'>"
                "  <div style='display:flex;align-items:center;justify-content:space-between;width:100%'>"
                "    <div style='display: inline-flex;gap: 5px;align-items: center;'>"
                f"      <div style='font-weight:600'>{name}</div>"
                f"      <div style='color:#64748b;font-size:12px'>"
                f"        {rank_label} • {p.get('team', '')} • {age} yrs"
                "      </div>"
                "    </div>"
                f"    <div class='player-trade-value'>{val_txt}</div>"
                "  </div>"
                "</div>"
            )

        return "".join(rows_html)

    # ----------------- Build HTML cards -----------------
    cards_html = []

    for rid, meta in team_meta.items():
        name = meta["name"]
        avatar = meta.get("avatar") or ""
        img_html = (
            f"<img class='avatar' src='{avatar}' onerror=\"this.style.display='none'\">"
            if avatar else ""
        )

        z_map = team_pos_z[rid]
        strongest_pos = max(POS_ORDER, key=lambda p: z_map.get(p, 0.0))
        weakest_pos = min(POS_ORDER, key=lambda p: z_map.get(p, 0.0))

        table_rows = []
        for pos in POS_ORDER:
            vals = team_pos_values[rid][pos]
            count = len(vals)
            total = sum(vals)
            avg = team_pos_avg[rid][pos]
            z = z_map[pos]

            # bar width scaled within this position across league
            z_min = pos_z_min[pos]
            z_max = pos_z_max[pos]
            if z_max > z_min:
                pct = 10 + 80 * (z - z_min) / (z_max - z_min)  # 10–90%
            else:
                pct = 50.0

            highlight_class = ""
            if pos == strongest_pos:
                highlight_class = " pos-strongest"
            elif pos == weakest_pos:
                highlight_class = " pos-weakest"

            rank = pos_rank[pos].get(rid, 0)

            # main row (clickable)
            main_row = (

                "<tr class='pos-row{cls}' data-pos='{pos}'>"
                "  <td class='pos-name'>"
                "    <span class='pos-row-toggle'>▾</span> {pos}"
                "  </td>"
                "  <td class='pos-count'>{count}</td>"
                "  <td class='pos-total'>{total:.1f}</td>"
                "  <td class='pos-avg'>{avg:.1f}</td>"
                "  <td class='pos-z'>{z:.2f}</td>"
                "  <td class='pos-bar-cell'>"
                "    <div class='pos-bar-outer'>"
                "      <div class='pos-bar-inner' style='width:{pct:.0f}%;'></div>"
                "    </div>"
                "  </td>"
                "<td class='pos-rank'>#{rank}</td>"
                "</tr>".format(
                    cls=highlight_class,
                    rank=rank,
                    pos=pos,
                    count=count,
                    total=total,
                    avg=avg,
                    z=z,
                    pct=pct,
                )
            )

            # detail row right under it (collapsed by default)
            detail_html = render_pos_players(rid, pos)
            detail_row = (
                "<tr class='pos-detail-row' data-pos='{pos}' style='display:none;'>"
                "  <td colspan='7'>"
                "    <div class='pos-detail-inner'>"
                f"      {detail_html}"
                "    </div>"
                "  </td>"
                "</tr>".format(pos=pos)
            )

            table_rows.append(main_row)
            table_rows.append(detail_row)

        card_html = (
            "<div class='card team-strength-card'>"
            "  <div class='card-header-row'>"
            f"    <div style='display:flex;align-items:center;gap:8px;'>{img_html}<h2>{name}</h2></div>"
            f"    <div class='mini-label'>Positional Index: "
            f"<span style='font-weight:600'>{team_pos_index[rid]:+.2f}</span></div>"
            "  </div>"
            "  <div class='card-body'>"
            "    <table class='pos-strength-table'>"
            "    <table class='pos-strength-table'>"
            "      <thead>"
            "        <tr>"
            "          <th>Pos</th>"
            "          <th>#</th>"
            "          <th>Value</th>"
            "          <th>Avg Value</th>"
            "          <th>Z-Score</th>"
            "          <th>Strength</th>"
            "          <th>Rank</th>"
            "        </tr>"
            "      </thead>"
            "      <tbody>"
            f"        {''.join(table_rows)}"
            "      </tbody>"
            "    </table>"
            "  </div>"
            "</div>"
        )

        cards_html.append(card_html)

    all_cards_html = "".join(
        cards_html) or "<div class='card'><div class='card-body'><p>No teams found.</p></div></div>"

    # ---------- Page shell ----------
    return f"""
    <div class="page-layout teams-page">
      <main class="page-main">
        <div class="teams-grid">
          {all_cards_html}
        </div>
      </main>

      <aside class="page-sidebar">
        <div class="card small">
          <div class="card-header">
            <h3>Legend</h3>
          </div>
          <div class="card-body">
            <p class="mini-label">Positional Index</p>
            <p style="font-size:13px;color:#64748b;">
              Weighted average of each position's Z-score using lineup slot counts.
              Positive = stronger than league at those positions; negative = weaker.
            </p>
            <ul class="ticker-list">
              <li><span class="mini-label">Green row</span> – strongest position for that team.</li>
              <li><span class="mini-label">Red row</span> – weakest position for that team.</li>
              <li><span class="mini-label">Strength bar</span> – how this team ranks vs others at that position.</li>
              <li>Click a position row to view all players for that position.</li>
            </ul>
          </div>
        </div>
      </aside>
    </div>

    <script>
    (function() {{
      // Click a position row to toggle its detail row
      document.addEventListener('click', function(e) {{
        const row = e.target.closest('.pos-row');
        if (!row) return;
        const detail = row.nextElementSibling;
        if (!detail || !detail.classList.contains('pos-detail-row')) return;

        const isOpen = detail.style.display === '' || detail.style.display === 'table-row';
        detail.style.display = isOpen ? 'none' : 'table-row';

        // rotate the little arrow
        const chevron = row.querySelector('.pos-row-toggle');
        if (chevron) {{
          chevron.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(180deg)';
        }}
      }});
    }})();
    </script>
    """


@app.route("/privacy")
@app.route("/<platform>/<int:season>/<league_id>/privacy")
def privacy_page(platform: Optional[str] = None, season: Optional[int] = None, league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">

            <h1 class="static-hero-title">Privacy Policy</h1>
            <div class="static-section">
              <div class="static-section-title">What We Collect</div>
              <p>
                We use your Sleeper league ID and public Sleeper data to build dashboards,
                projections, and tools. No passwords, payment info, or sensitive personal data
                is collected.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">What We Don't Collect</div>
              <p>
                We don’t store personal identifying information, sell data, or track you outside
                of this site.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Data Storage</div>
              <p>
                League data is cached temporarily on the server to improve performance.
                You may request removal at any time via the Contact page.
              </p>
            </div>

            <div class="highlight-box">
              Have questions or want your league data removed?  
              Reach out using the Contact page.
            </div>

          </div>
        </div>
        """
    return render_page("BR Fantasy Privacy", league_id if league_id else None, "privacy", body, platform, season)


@app.route("/support")
@app.route("/<platform>/<int:season>/<league_id>/support")
def support_page(platform: Optional[str] = None, season: Optional[int] = None, league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">
            <h1 class="static-hero-title">Support the Site</h1>

            <div class="static-section">
              <div class="static-section-title">1. Direct Support</div>
              <p>
                If you find the dashboard helpful for your league, you can support
                ongoing development and hosting costs.
              </p>
              <p style="margin-top:6px;">
                <a
                  class="link-pill"
                  href="https://buymeacoffee.com/brfantasy"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  💸 Make a donation
                </a>
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">2. Premium Ad-Free Mode (Coming Soon)</div>
              <p>
                The long-term plan is to offer a premium, ad-free experience with extra
                features (advanced graphs, additional projections, league history views,
                and more) while keeping a solid free version for everyone.
              </p>
              <p style="margin-top:8px;">
                Want early access or to give feedback on premium ideas? Reach out on the
                Contact page and include “Premium” in your message.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">3. Share With Your League</div>
              <p>
                Honestly one of the best ways to support this is just using it.
                Share the link with your league mates, show the dashboards on stream,
                or use the matchup previews in your weekly recaps.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">4. Follow & Subscribe</div>
              <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <a class="link-pill" href="https://youtube.com/@hoodiekj" target="_blank">▶️ YouTube</a>
                <a class="link-pill" href="https://twitch.tv/hoodiekj1" target="_blank">🎮 Twitch</a>
                <a class="link-pill" href="https://twitter.com/hoodiekj16" target="_blank">🐦 Twitter/X</a>
              </div>
            </div>

            <div class="highlight-box">
              Every bit of support helps keep the site online and evolving for future seasons.
              Thanks for using BR Fantasy.
            </div>
          </div>
        </div>
        """
    return render_page("BR Fantasy Support", league_id if league_id else None, "support", body, platform, season)


@app.route("/faq")
@app.route("/<platform>/<int:season>/<league_id>/faq")
def faq_page(platform: Optional[str] = None, season: Optional[int] = None, league_id: Optional[str] = None):
    body = """
        <div class="static-page">
          <div class="static-card-page">
            <h1 class="static-hero-title">FAQ</h1>

            <div class="static-section">
              <div class="static-section-title">General</div>

              <details class="faq-item" open>
                <summary>What is the BR Fantasy Dashboard?</summary>
                <p>
                  It’s a custom fantasy football dashboard that pulls in your Sleeper league
                  data and turns it into power rankings, weekly summaries, matchup previews,
                  graphs, and more—all in one place.
                </p>
              </details>

              <details class="faq-item">
                <summary>What do I need to use it?</summary>
                <p>
                  All you need is your Sleeper or ESPN league ID. Paste it into the home screen,
                  and the dashboard will fetch public data for that league.
                </p>
              </details>

              <details class="faq-item">
                <summary>Does this change anything in my Fantasy league?</summary>
                <p>
                  No. The dashboard is read-only. It just reads public data from your league’s
                  API and never modifies your league, rosters, or settings.
                </p>
              </details>
            </div>

            <div class="static-section">
              <div class="static-section-title">Data & Privacy</div>

              <details class="faq-item">
                <summary>What data do you store?</summary>
                <p>
                  Some league data may be cached temporarily so pages load quickly
                  (rosters, users, scores, projections, etc.). We do not store your
                  password or payment information. See the Privacy Policy for more details.
                </p>
              </details>

              <details class="faq-item">
                <summary>Can I have my league data removed?</summary>
                <p>
                  Yes. Use the Contact page to send your Sleeper league ID and request
                  removal. We’ll clear cached data for that league.
                </p>
              </details>
            </div>

            <div class="static-section">
              <div class="static-section-title">Premium / Ads / Support</div>

              <details class="faq-item">
                <summary>Is there a premium or ad-free mode?</summary>
                <p>
                  A premium, ad-free experience is planned. The idea is to keep a fully
                  functional free tier while offering extra features and an ad-free UI for
                  people who want to support the project.
                </p>
              </details>

              <details class="faq-item">
                <summary>How can I support the site?</summary>
                <p>
                  You can support the project through donations, using premium when it’s
                  available, or by sharing the site with your league mates.
                  Visit the Support page for options.
                </p>
              </details>
            </div>

            <div class="static-section">
              <div class="static-section-title">Issues & Feedback</div>

              <details class="faq-item">
                <summary>The numbers look wrong—what should I do?</summary>
                <p>
                  First, hit the refresh button on the nav to clear cached data for your
                  league. If something still looks off, send a message via the Contact
                  page with your league ID and a short description of the issue.
                </p>
              </details>

              <details class="faq-item">
                <summary>Can I request new features?</summary>
                <p>
                  Absolutely. This project is built for fantasy degenerates.
                  Drop your ideas on the Contact page and they might make it onto the roadmap.
                </p>
              </details>
            </div>
          </div>
        </div>
        """
    return render_page("BR Fantasy FAQ", league_id if league_id else None, "faq", body, platform, season)


@app.route("/contact", methods=["GET", "POST"])
@app.route("/<platform>/<int:season>/<league_id>/contact")
def contact_page(platform: Optional[str] = None, season: Optional[int] = None, league_id: Optional[str] = None):
    # super simple "email us" style page; you can later hook this to a form handler
    body = """
        <div class="static-page">
          <div class="static-card-page">

            <h1 class="static-hero-title">Contact</h1>

            <div class="static-section">
              <div class="static-section-title">Message</div>
              <p>You can message the creator directly via social platforms:</p>

              <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:10px;">
                <a class="link-pill" href="https://youtube.com/@hoodiekj" target="_blank">▶️ YouTube</a>
                <a class="link-pill" href="https://twitch.tv/hoodiekj1" target="_blank">🎮 Twitch</a>
                <a class="link-pill" href="https://twitter.com/hoodiekj16" target="_blank">🐦 Twitter/X</a>
              </div>
            </div>

            <div class="static-section">
              <div class="static-section-title">What to include</div>
              <ul style="margin-left:20px; color:#4b5563; font-size:14px;">
                <li>Your Sleeper league ID</li>
                <li>Which page you were on</li>
                <li>What wasn’t working or looked incorrect</li>
                <li>Screenshots if possible</li>
              </ul>
            </div>

            <div class="highlight-box">
              Feedback helps shape future features — thanks for helping improve BR Fantasy.
            </div>

          </div>
        </div>
        """
    return render_page("BR Fantasy Contact", league_id if league_id else None, "contact", body, platform, season)


def league_url(slug: str, league_id: Optional[str] = None) -> str:
    """
    Build a URL that keeps league context if we have one.
    slug examples: 'faq', 'privacy', 'support', 'contact'
    """
    if league_id:
        return f"/league/{league_id}/{slug}"
    return f"/{slug}"


def get_league_ctx_from_cache(platform: str, league_id: str, season: int) -> dict:
    key = _cache_key(platform, season, league_id)
    entry = DASHBOARD_CACHE.get(key)
    if not entry or (time.time() - entry.get("ts", 0) > CACHE_TTL):
        ctx = build_league_context(platform, league_id, season)
        DASHBOARD_CACHE[key] = {"ctx": ctx, "ts": time.time(), "page_html": {}}
        return ctx
    return entry["ctx"]


@app.route("/<platform>/<int:season>/<league_id>/dashboard")
def page_dashboard(platform: str, season: int, league_id: str):
    ctx = get_league_ctx_from_cache(platform, league_id, season)
    print("[page_dashboard] offseason_mode=", ctx.get("offseason_mode"), "season_type=", ctx.get("season_type"))

    if ctx.get("offseason_mode"):
        body = build_offseason_dashboard_body(ctx)
    else:
        ensure_weekly_bits(ctx)
        body = build_dashboard_body(ctx)

    return render_page("BR Fantasy Dashboard", league_id, "dashboard", body, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/standings")
def page_standings(platform: str, season: int, league_id: str):
    ctx = get_league_ctx_from_cache(platform, league_id, season)

    if ctx.get("offseason_mode"):
        body = """
        <div class="card central">
          <div class="card-header"><h2>Standings Unavailable</h2></div>
          <div class="card-body">
            <p>Standings will appear once the season begins.</p>
            <p>During the offseason, use Teams, Activity, and Trade Calc for roster planning.</p>
          </div>
        </div>
        """
    else:
        body = build_standings_body(ctx)

    return render_page("BR Fantasy Standings", league_id, "standings", body, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/weekly")
def page_weekly(platform: str, season: int, league_id: str):
    ctx = get_league_ctx_from_cache(platform, league_id, season)

    if ctx.get("offseason_mode"):
        body = """
        <div class="card central">
          <div class="card-header"><h2>Weekly Hub Unavailable</h2></div>
          <div class="card-body">
            <p>The Weekly Hub becomes active once the season begins.</p>
            <p>Use the dashboard, teams, activity, and trade tools for offseason planning.</p>
          </div>
        </div>
        """
    else:
        ensure_weekly_bits(ctx)
        body = build_weekly_hub_body(ctx)

    return render_page("BR Fantasy Weekly Hub", league_id, "weekly", body, platform, season)


@app.route("/trade")
@app.route("/<platform>/<int:season>/<league_id>/trade")
def page_trade(platform: Optional[str] = None, season: Optional[int] = None, league_id: Optional[str] = None):
    if league_id:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        body = build_trade_calculator_body(ctx["league_id"], ctx["season"])
    else:
        state = get_nfl_state() or {}
        current_season = int(state.get("season") or datetime.now().year)
        body = build_trade_calculator_body(None, current_season)
    return render_page("BR Fantasy Trade Calculator", league_id, "trade", body, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/activity")
def page_activity(platform: str, season: int, league_id: str):
    cached = get_page_html_from_cache(platform, season, league_id, "activity")
    if cached:
        return render_page("BR Fantasy Activity", league_id, "activity", cached, platform, season)

    ctx = get_league_ctx_from_cache(platform, league_id, season)
    body = build_activity_body(ctx)
    store_page_html(platform, season, league_id, "activity", body)
    return render_page("BR Fantasy Activity", league_id, "activity", body, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/graphs")
def page_graphs(platform: str, season: int, league_id: str):
    cached = get_page_html_from_cache(platform, season, league_id, "graphs")
    if cached:
        return render_page("BR Fantasy Graphs", league_id, "graphs", cached, platform, season)

    ctx = get_league_ctx_from_cache(platform, league_id, season)

    if ctx.get("offseason_mode"):
        body_html = """
        <div class="card central">
          <div class="card-header"><h2>Graphs Unavailable</h2></div>
          <div class="card-body">
            <p>Weekly scoring graphs will appear once the season begins.</p>
            <p>During the offseason, use Dashboard, Teams, Activity, and Trade Calc for roster planning.</p>
          </div>
        </div>
        """
    else:
        body_html = build_graphs_body(ctx)

    store_page_html(platform, season, league_id, "graphs", body_html)
    return render_page("BR Fantasy Graphs", league_id, "graphs", body_html, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/teams")
def page_teams(platform: str, season: int, league_id: str):
    cached = get_page_html_from_cache(platform, season, league_id, "teams")
    if cached:
        return render_page("BR Fantasy Teams", league_id, "teams", cached, platform, season)

    ctx = get_league_ctx_from_cache(platform, league_id, season)
    body_html = build_teams_body(ctx)
    store_page_html(platform, season, league_id, "teams", body_html)
    return render_page("BR Fantasy Teams", league_id, "teams", body_html, platform, season)


@app.before_request
def maybe_run_daily():
    global daily_completed

    today_et: date = datetime.now(EASTERN).date()

    if daily_completed == today_et:
        return

    if daily_lock.acquire(blocking=False):
        try:
            if daily_completed != today_et:
                print(f"[daily] Running daily data process for {today_et} (ET)...")

                state = get_nfl_state() or {}
                season = int(state.get("season") or datetime.now().year)
                week = int(state.get("week") or 0)

                run_daily_data_async(season, week)
                daily_completed = today_et
        finally:
            daily_lock.release()


@app.route("/", methods=["GET", "POST"])
def index():
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    viewed_season = current_season

    if request.method == "POST":
        platform = (request.form.get("platform") or "sleeper").strip().lower()
        league_id = (request.form.get("league") or "").strip()
        season = int(request.form.get("season") or viewed_season)

        ok, err = validate_league_id(platform, league_id)
        if not ok:
            body_html = render_template_string(
                FORM_BODY,
                username="",
                viewed_season=viewed_season,
                error=err,
            )
            return render_page("BR Fantasy Dashboard", None, "home", body_html)

        key = _cache_key(platform, season, league_id)
        entry = DASHBOARD_CACHE.get(key)
        if entry and (time.time() - entry["ts"] < CACHE_TTL):
            return redirect(url_for(
                "page_dashboard",
                platform=platform,
                season=season,
                league_id=league_id,
            ))

        ctx = build_league_context(
            platform=platform,
            league_id=league_id,
            season=season,
        )
        DASHBOARD_CACHE[key] = {"ctx": ctx, "ts": time.time(), "page_html": {}}

        return redirect(url_for(
            "page_dashboard",
            platform=platform,
            season=season,
            league_id=league_id,
        ))

    body_html = render_template_string(
        FORM_BODY,
        username="",
        viewed_season=viewed_season,
        error=None,
    )
    return render_page("BR Fantasy Dashboard", None, "home", body_html)


@app.route("/api/weekly-week")
def api_weekly_week():
    platform = (request.args.get("platform") or "").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    season = int(request.args.get("season") or 0)

    try:
        week = int(request.args.get("week") or 0)
    except ValueError:
        week = 0

    if not league_id or not week:
        return jsonify({"ok": False, "error": "Missing league_id or week"}), 400

    ctx = get_league_ctx_from_cache(platform, league_id, season)

    if ctx.get("offseason_mode"):
        return jsonify({
            "ok": False,
            "error": "Weekly Hub is unavailable during the offseason."
        }), 400

    ensure_weekly_bits(ctx)

    df_weekly = ctx["df_weekly"]
    roster_map = ctx["roster_map"]
    players_map = ctx["players_map"]
    proj_by_week = ctx["proj_by_week"]
    rosters = ctx["rosters"]
    users = ctx["users"]

    team_game_lookup = ctx["team_game_lookup"]
    matchups_by_week = ctx["matchups_by_week"]
    statuses = ctx["statuses"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    season = ctx["season"]
    current_week = ctx["current_week"]
    max_weeks = ctx["weeks"]

    if week < 1 or week > max_weeks:
        return jsonify({"ok": False, "error": f"Week {week} out of range"}), 400

    if (
            df_weekly is not None
            and not df_weekly.empty
            and "finalized" in df_weekly.columns
            and "week" in df_weekly.columns
    ):
        finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    else:
        finalized_df = pd.DataFrame()

    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())
    else:
        last_final_week = max(1, min(int(current_week or 1), int(max_weeks or 1)))

    resolved_league_id = ctx.get("resolved_league_id", league_id)

    top_html = render_weekly_top_scorers_for_week(
        resolved_league_id,
        df_weekly,
        roster_map,
        players_map,
        proj_by_week,
        rosters,
        week,
        users,
        platform,
        season
    )
    highlights_html = _render_weekly_highlights(df_weekly, week)

    matchups = matchups_by_week.get(week, []) or []
    status_by_pid = (statuses.get(week) or {}).get("statuses", {}) or {}

    slides = [
        render_matchup_slide(
            season,
            m,
            week,
            last_final_week,
            status_by_pid=status_by_pid,
            projections=proj_by_week,
            players=players_index,
            teams=teams_index,
            team_game_lookup=team_game_lookup,
        )
        for m in matchups
    ]

    slides_html = "".join(slides) if slides else "<div class='m-empty'>No matchups</div>"
    slides_by_week = {week: slides_html}

    matchups_html = render_matchup_carousel_weeks(
        slides_by_week,
        dashboard=False,
        active_week=week,
    )

    return jsonify({
        "ok": True,
        "top_html": top_html,
        "highlights_html": highlights_html,
        "matchups_html": matchups_html,
    })


@app.route("/api/refresh-page", methods=["POST"])
def api_refresh_page():
    payload = request.get_json(silent=True) or {}
    league_id = (payload.get("league_id") or "").strip()
    platform = (payload.get("platform") or "").strip().lower()
    season = int(payload.get("season") or 0)
    page = (payload.get("page") or "").strip().lower()

    if not league_id or not page:
        return jsonify({"ok": False, "error": "Missing league_id or page"}), 400

    valid_pages = {"activity", "standings", "teams", "weekly", "dashboard", "graphs", "trade"}
    if page not in valid_pages:
        return jsonify({"ok": False, "error": f"Unknown page '{page}'"}), 400

    try:
        ctx = refresh_league_ctx_section(platform, league_id, page, season)

        if page == "dashboard":
            if ctx.get("offseason_mode"):
                body_html = build_offseason_dashboard_body(ctx)
            else:
                ensure_weekly_bits(ctx)
                body_html = build_dashboard_body(ctx)

        elif page == "standings":
            if ctx.get("offseason_mode"):
                body_html = """
                <div class="card central">
                  <div class="card-header"><h2>Standings Unavailable</h2></div>
                  <div class="card-body">
                    <p>Standings will appear once the season begins.</p>
                    <p>During the offseason, use Teams, Activity, and Trade Calc for roster planning.</p>
                  </div>
                </div>
                """
            else:
                body_html = build_standings_body(ctx)

        elif page == "weekly":
            if ctx.get("offseason_mode"):
                body_html = """
                <div class="card central">
                  <div class="card-header"><h2>Weekly Hub Unavailable</h2></div>
                  <div class="card-body">
                    <p>The Weekly Hub becomes active once the season begins.</p>
                    <p>Use the dashboard, teams, activity, and trade tools for offseason planning.</p>
                  </div>
                </div>
                """
            else:
                ensure_weekly_bits(ctx)
                body_html = build_weekly_hub_body(ctx)

        elif page == "graphs":
            if ctx.get("offseason_mode"):
                body_html = """
                <div class="card central">
                  <div class="card-header"><h2>Graphs Unavailable</h2></div>
                  <div class="card-body">
                    <p>Weekly scoring graphs will appear once the season begins.</p>
                    <p>During the offseason, use Dashboard, Teams, Activity, and Trade Calc for roster planning.</p>
                  </div>
                </div>
                """
            else:
                body_html = build_graphs_body(ctx)
                store_page_html(platform, season, league_id, "graphs", body_html)

        elif page == "activity":
            body_html = build_activity_body(ctx)
            store_page_html(platform, season, league_id, "activity", body_html)

        elif page == "teams":
            body_html = build_teams_body(ctx)
            store_page_html(platform, season, league_id, "teams", body_html)

        elif page == "trade":
            body_html = build_trade_calculator_body(ctx["league_id"], ctx["season"])

        else:
            body_html = ""

        return jsonify({
            "ok": True,
            "refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "current_week": ctx.get("current_week"),
            "body_html": body_html,
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Refresh failed: {e}",
        }), 500


@app.route("/logout")
def logout():
    # Clear the session + cached league context
    from flask import session
    session.clear()
    return redirect(url_for("index"))


# ---------- global cache for model value table used by trade eval ----------
_MODEL_VALUE_CACHE = None
_MODEL_VALUE_CACHE_TS = 0
_MODEL_VALUE_TTL = 60 * 60  # 1 hour


def get_model_value_table_cached():
    global _MODEL_VALUE_CACHE, _MODEL_VALUE_CACHE_TS
    now = time.time()
    if _MODEL_VALUE_CACHE is not None and now - _MODEL_VALUE_CACHE_TS < _MODEL_VALUE_TTL:
        return _MODEL_VALUE_CACHE
    tbl = load_model_value_table() or []
    _MODEL_VALUE_CACHE = tbl
    _MODEL_VALUE_CACHE_TS = now
    return tbl


@app.route("/api/trade-eval", methods=["POST"])
def api_trade_eval():
    payload = request.get_json(force=True)

    side_a_players = [str(pid) for pid in payload.get("side_a_players", [])]
    side_b_players = [str(pid) for pid in payload.get("side_b_players", [])]
    side_a_picks = payload.get("side_a_picks", []) or []
    side_b_picks = payload.get("side_b_picks", []) or []

    # ---------- Load model player value table ----------
    # This SHOULD return your list[dict] of players
    value_table = get_model_value_table_cached()

    if not isinstance(value_table, list):
        raise ValueError("model_value_table must be a list of player objects")

    # Index players by id for quick lookup
    players_by_id = {str(p["id"]): p for p in value_table if isinstance(p, dict) and "id" in p}

    # ---------- Helpers ----------

    def value_pick(pk: str) -> float:
        """
        pk is like '2026_1_04' -> year, round, slot (within round).
        We bucket slot -> early/mid/late and look up a blended
        value from PICK_VALUES built from FantasyCalc + DynastyProcess.
        """
        try:
            yr_str, rnd_str, slot_str = pk.split("_")
            year = int(yr_str)
            rnd = int(rnd_str)
            slot = int(slot_str)
        except Exception:
            return 0.0

        # convert slot to early/mid/late based on league size
        bucket = bucket_for_slot(slot, num_teams=10)  # use 10 or 12 based on your league
        key = f"{year}_{rnd}_{bucket}"

        val = PICK_VALUES.get(key)
        if val is not None:
            return float(val)

        # Optional: generic fallback like any-year blended value if you ever add that
        generic_key = f"any_{rnd}_{bucket}"
        if generic_key in PICK_VALUES:
            return float(PICK_VALUES[generic_key])

        return 0.0

    def build_side(players_ids, picks_ids):
        """
        Build the basic info for a side using value_table payload:

          {
            "id": "9509",
            "name": "Bijan Robinson",
            "team": "ATL",
            "position": "RB",
            "age": 23.8,
            "value": 968.0
          }
        """

        raw_players_total = 0.0
        raw_picks_total = 0.0
        player_values: list[float] = []
        breakdown = []

        # Players
        for pid in players_ids:
            pid_str = str(pid)
            player = players_by_id.get(pid_str)

            if not player:
                breakdown.append({
                    "type": "player",
                    "id": pid_str,
                    "name": f"Player {pid_str}",
                    "value": 0.0,
                    "position": None,
                    "team": None,
                })
                continue

            val = float(player.get("value", 0.0) or 0.0)
            name = player.get("name")
            pos = player.get("position")
            team = player.get("team")

            breakdown.append({
                "type": "player",
                "id": pid_str,
                "name": name,
                "value": val,
                "position": pos,
                "team": team,
            })
            raw_players_total += val
            player_values.append(val)

        # Picks
        for pk in picks_ids:
            pk_str = str(pk)
            val = float(value_pick(pk_str))
            breakdown.append({
                "type": "pick",
                "id": pk_str,
                "value": val,
            })
            raw_picks_total += val

        raw_total = raw_players_total + raw_picks_total

        return {
            "raw_total": raw_total,
            "raw_players_total": raw_players_total,
            "raw_picks_total": raw_picks_total,
            "player_values": player_values,
            "breakdown": breakdown,
            "effective_total": raw_total,  # will be adjusted later
            "adjustment": 0.0,
        }

    side_a = build_side(side_a_players, side_a_picks)
    side_b = build_side(side_b_players, side_b_picks)

    apply_multi_for_one_adjustment(side_a, side_b)

    a_eff = side_a["effective_total"]
    b_eff = side_b["effective_total"]

    diff = a_eff - b_eff
    abs_diff = abs(diff)

    FAIR_PCT = 0.08  # 8% band; tweak as needed
    baseline = max(a_eff, b_eff, 1.0)
    fair_band = baseline * FAIR_PCT

    if abs_diff <= fair_band:
        pct = (abs_diff / baseline) * 100.0
        verdict = f"This trade looks very fair (about {pct:.1f}% apart)."
    elif diff > 0:
        verdict = f"Team 1 is favored by about {abs_diff:.1f} value."
    else:
        verdict = f"Team 2 is favored by about {abs_diff:.1f} value."

    return jsonify({
        "side_a": side_a,
        "side_b": side_b,
        "diff": diff,
        "abs_diff": abs_diff,
        "fair_threshold": fair_band,
        "fair_pct": FAIR_PCT,
        "verdict": verdict,
    })


def _sanitize_for_json(obj):
    """
    Recursively walk obj and replace NaN/inf/-inf floats with None
    so that json.dumps / jsonify produce valid JSON that
    fetch().json() can parse.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


@app.route("/api/league-players")
def api_league_players():
    model_value_table = load_model_value_table()
    if not isinstance(model_value_table, list):
        raise ValueError("model_value_table must be a list of player objects")

    cleaned = _sanitize_for_json(model_value_table)
    return jsonify(cleaned)


@app.route("/api/value-movers")
def api_value_movers():
    try:
        days = int(request.args.get("days", 7))
    except (TypeError, ValueError):
        days = 7

    try:
        limit = int(request.args.get("limit", 15))
    except (TypeError, ValueError):
        limit = 15

    payload = get_top_movers(days=max(days, 1), limit=max(limit, 1))
    return jsonify(payload)


@app.route("/api/player-value-history/<player_id>")
def api_player_value_history(player_id: str):
    try:
        days = int(request.args.get("days", 30))
    except (TypeError, ValueError):
        days = 30

    history = get_player_value_history(player_id, days=max(days, 1))
    return jsonify(
        {
            "player_id": str(player_id),
            "days": max(days, 1),
            "history": history,
        }
    )


@app.route("/api/sleeper-user-leagues")
def api_sleeper_user_leagues():
    username = (request.args.get("username") or "").strip()
    season = int(request.args.get("season") or get_nfl_state().get("season"))

    if not username:
        return jsonify({"ok": False, "error": "Missing username"}), 400

    try:
        user = get_sleeper_user_by_username(username)
        if not user:
            return jsonify({"ok": False, "error": "Sleeper username not found"}), 404

        leagues = get_sleeper_user_leagues(user["user_id"], season)

        # Optional: filter out leagues without usable ids
        leagues = [lg for lg in leagues if lg.get("league_id")]

        # Optional: sort nicer
        leagues.sort(
            key=lambda lg: (
                str(lg.get("status") or "") != "in_season",
                -(int(lg.get("total_rosters") or 0)),
                str(lg.get("name") or "").lower(),
            )
        )

        return jsonify({
            "ok": True,
            "user": {
                "user_id": user.get("user_id"),
                "username": user.get("username"),
                "display_name": user.get("display_name"),
                "avatar": user.get("avatar"),
            },
            "leagues": [format_sleeper_league_option(lg) for lg in leagues],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
