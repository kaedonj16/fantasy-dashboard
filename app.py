import hashlib
import html
import json
import os
import threading
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from zoneinfo import ZoneInfo

import math
import pandas as pd
from flask import (
    Flask,
    request,
    render_template_string,
    redirect,
    url_for,
    jsonify,
    session,
    send_file,
)
from plotly.offline import get_plotlyjs

from dashboard_services.ai.history_recap import get_history_ai_recap
from dashboard_services.ai.renderer import get_team_gm_memo, get_front_office_briefing
from dashboard_services.api import (
    avatar_from_users,
    build_league_history_map,
    build_team_game_lookup,
    get_effective_scoring_settings,
    get_league_settings,
    get_nfl_players,
    get_nfl_scores_for_date,
    get_nfl_state,
    get_roster_positions,
    get_sleeper_user_by_username,
    get_sleeper_user_leagues,
    get_total_rosters,
    resolve_league_id_for_season,
)
from dashboard_services.awards import compute_awards_season, render_awards_section
from dashboard_services.changelog import CHANGELOG
from dashboard_services.injuries import build_injury_report, render_injury_accordion
from dashboard_services.matchups import (
    compute_team_projections_for_weeks,
    render_matchup_carousel_weeks,
    render_matchup_slide,
)
from dashboard_services.pages.graphs_page import build_graphs_body
from dashboard_services.pages.history_page import (
    build_history_body,
    build_regular_season_team_stats,
    sort_team_stats,
)
from dashboard_services.pages.trade_calculator_page import build_trade_calculator_body
from dashboard_services.picks import load_pick_value_table
from dashboard_services.platform_api import (
    get_bracket,
    get_drafts,
    get_league,
    get_rosters,
    get_traded_picks,
    get_users,
    sync_league_globals,
)
from dashboard_services.players import get_players_map
from dashboard_services.providers.espn_api import safe_float
from dashboard_services.service import (
    build_matchups_by_week,
    build_picks_by_roster,
    build_standings_map,
    build_tables,
    build_week_activity,
    matchup_cards_last_week,
    pill,
    playoff_bracket,
    render_teams_sidebar,
    render_top_three,
    seed_top6_from_team_stats,
)
from data_building.build_daily_value_table import build_daily_data
from data_building.player_value_history import get_top_movers, init_value_history_db, get_player_value_history
from utils.utils import (
    build_and_save_week_stats_for_league,
    build_status_for_week,
    build_teams_overview,
    clear_activity_cache_for_league,
    clear_teams_cache_for_league,
    clear_weekly_cache_for_league,
    count_roster_positions,
    fetch_week_from_tank01,
    get_live_game_ids_for_today,
    get_week_projections_cached,
    load_idp_index,
    load_model_value_table,
    load_players_index,
    load_teams_index,
    load_week_projection,
    load_week_schedule,
    streak_class,
)

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

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
plotly_js = get_plotlyjs()
try:
    init_value_history_db()
except Exception as e:
    print(f"[value-history] init skipped: {e}")

# Register breakout detection API routes
try:
    from dashboard_services.breakout_api import register_breakout_routes
    register_breakout_routes(app)
    print("[breakout-api] Breakout API endpoints registered")
except Exception as e:
    print(f"[breakout-api] Registration skipped: {e}")

# Register rookie prospect API routes
try:
    from dashboard_services.rookie_api import register_rookie_routes
    register_rookie_routes(app)
    print("[rookie-api] Rookie API endpoints registered")
except Exception as e:
    print(f"[rookie-api] Registration skipped: {e}")


def generate_recent_updates_html(limit=5):
    """Generate HTML for recent changelog updates."""
    from dashboard_services.changelog import CHANGELOG

    recent = CHANGELOG[:limit]
    if not recent:
        return ""

    html_parts = []
    for entry in recent:
        tag_class = f"update-tag-{entry['tag']}"
        html_parts.append(f'''
        <div class="home-update-item">
          <div class="home-update-item-header">
            <span class="home-update-tag {tag_class}">{entry['tag']}</span>
            <span class="home-update-date">{entry['date']}</span>
          </div>
          <p class="home-update-text">{entry['text']}</p>
        </div>
        ''')

    return '\n'.join(html_parts)


def get_team_full_name(abbreviation: str) -> str:
    """Map team abbreviation to full team name."""
    team_names = {
        "ARI": "Arizona Cardinals",
        "ATL": "Atlanta Falcons",
        "BAL": "Baltimore Ravens",
        "BUF": "Buffalo Bills",
        "CAR": "Carolina Panthers",
        "CHI": "Chicago Bears",
        "CIN": "Cincinnati Bengals",
        "CLE": "Cleveland Browns",
        "DAL": "Dallas Cowboys",
        "DEN": "Denver Broncos",
        "DET": "Detroit Lions",
        "GB": "Green Bay Packers",
        "HOU": "Houston Texans",
        "IND": "Indianapolis Colts",
        "JAX": "Jacksonville Jaguars",
        "KC": "Kansas City Chiefs",
        "LV": "Las Vegas Raiders",
        "LAC": "Los Angeles Chargers",
        "LAR": "Los Angeles Rams",
        "MIA": "Miami Dolphins",
        "MIN": "Minnesota Vikings",
        "NE": "New England Patriots",
        "NO": "New Orleans Saints",
        "NYG": "New York Giants",
        "NYJ": "New York Jets",
        "PHI": "Philadelphia Eagles",
        "PIT": "Pittsburgh Steelers",
        "SF": "San Francisco 49ers",
        "SEA": "Seattle Seahawks",
        "TB": "Tampa Bay Buccaneers",
        "TEN": "Tennessee Titans",
        "WAS": "Washington Commanders",
        "WSH": "Washington Commanders"
    }
    return team_names.get(abbreviation.upper(), abbreviation)


FORM_BODY = """
<div class="home-page">
  <section class="home-hero">
    <div class="home-hero-left">
      <h1 class="home-title">BR Fantasy Dashboard</h1>
      <p class="home-subtitle">
        Your dynasty league, upgraded. Advanced analytics, AI-powered insights, and professional-grade tools for Sleeper and ESPN leagues.
      </p>

      <ul class="home-bullets">
        <li><strong>AI Trade Analyst</strong> — Personalized deal evaluation with counter suggestions</li>
        <li><strong>Dynasty Value Engine</strong> — Hybrid model combining consensus data and advanced metrics</li>
        <li><strong>Weekly Projections</strong> — Live scoring, matchup previews, and storyline tracking</li>
        <li><strong>Historical Analysis</strong> — Season recaps, power rankings, and trend visualization</li>
      </ul>
    </div>

    <div class="home-hero-right">
      <div class="home-card">
        <h2 class="home-card-title">Get started</h2>

        <div class="row">
          <label for="platformSelect">Platform</label>
          <div class="platform-selector">
            <button type="button" class="platform-btn active" data-platform="sleeper">Sleeper</button>
            <button type="button" class="platform-btn" data-platform="espn" disabled style="opacity: 0.6; cursor: not-allowed;">
              ESPN <span style="font-size: 0.75em; font-weight: 400;">(Coming Soon)</span>
            </button>
          </div>
        </div>

        <!-- Sleeper Flow -->
        <div id="sleeperFlow">
          <div class="row">
            <label for="username">Sleeper Username</label>
            <input type="text" id="username" name="username" value="{{ username or '' }}">
          </div>

          <div class="row">
            <button type="button" id="lookupBtn">Find My Leagues</button>
          </div>
        </div>

        <!-- ESPN Flow -->
        <!-- DISABLED
        <div id="espnFlow" style="display:none;">
          <div class="row">
            <label for="espnLeagueIdInput">ESPN League ID</label>
            <input type="text" id="espnLeagueIdInput" placeholder="e.g. 123456789" autocomplete="off">
          </div>
          <div class="row">
            <label for="espnTeamName">Your Team Name <span style="font-weight:400;font-size:0.85em;">(optional, to track your team)</span></label>
            <input type="text" id="espnTeamName" placeholder="e.g. Dynasty Monsters">
          </div>
          <div class="row">
            <button type="button" id="espnSubmitBtn">Go to Dashboard</button>
          </div>
          <p class="hint" style="margin-top:6px;">
            ESPN private leagues require <code>ESPN_S2</code> and <code>ESPN_SWID</code> environment variables set on the server.
          </p>
        </div>
        -->

        <form method="post" id="leagueSelectForm">
          <input type="hidden" name="platform" id="formPlatform" value="sleeper">
          <input type="hidden" name="season" value="{{ viewed_season }}">
          <input type="hidden" name="username" id="formUsername" value="">

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

        <p class="hint" id="sleeperHint">
          Enter your Sleeper username, choose one of your leagues, and unlock advanced analytics.
        </p>
      </div>
    </div>
  </section>

  <div class="home-content-wrapper">
    <section class="home-feature-grid">
    <div class="home-feature-card">
      <div class="home-feature-icon">📊</div>
      <h3>Trade Calculator</h3>
      <p>
        AI-powered trade analysis personalized to your roster. Get real-time value assessments,
        balance indicators, and specific counter suggestions — not generic advice.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon">📈</div>
      <h3>Dynasty Values</h3>
      <p>
        Hybrid valuation model blending market consensus with production metrics, age curves,
        and positional scarcity. Updated daily for all players and picks.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon">⚡</div>
      <h3>Weekly Hub</h3>
      <p>
        Live scoring context for every matchup. See projections, starters, and real-time updates
        in one clean view — perfect for Sunday trash talk.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon">🎯</div>
      <h3>Team Analytics</h3>
      <p>
        Position strength breakdowns, roster composition analysis, and competitive advantages
        mapped across your league. Know where you stand.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon">📉</div>
      <h3>Graphs & Trends</h3>
      <p>
        Visualize points for/against, strength of schedule, playoff odds, and luck metrics.
        Prove who's actually good and who just got lucky.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon">🏆</div>
      <h3>Historical Insights</h3>
      <p>
        AI-generated season recaps personalized to your team. Track multi-year trends,
        rivalry records, and championship runs across league history.
      </p>
    </div>
  </section>

    <aside class="home-updates-sidebar">
      <h3 class="home-updates-sidebar-title">Recent Updates</h3>
      <div class="home-updates-list">
        {{ recent_updates | safe }}
      </div>
    </aside>
  </div>
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

      <!-- Top Banner Ad -->
      <div class="ad-container ad-top-banner">
        <ins class="adsbygoogle"
             style="display:block"
             data-ad-client="ca-pub-9164153092633845"
             data-ad-slot="5233061286"
             data-ad-format="auto"
             data-full-width-responsive="true"></ins>
      </div>

      <main id="page-root" class="overview-layout">
        {body}
      </main>

      <!-- Bottom Content Ad -->
      <div class="ad-container ad-bottom-content">
        <ins class="adsbygoogle"
             style="display:block"
             data-ad-client="ca-pub-9164153092633845"
             data-ad-slot="5233061286"
             data-ad-format="auto"
             data-full-width-responsive="true"></ins>
      </div>
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

    <!-- Cookie Consent Banner -->
    <div id="cookieConsent" class="cookie-consent" style="display: none;">
      <div class="cookie-consent-content">
        <p>
          We use cookies to improve your experience and show relevant ads. By continuing to use this site, you consent to our use of cookies.
          <a href="{privacy_url}" target="_blank">Learn more</a>
        </p>
        <div class="cookie-consent-buttons">
          <button id="acceptCookies" class="cookie-btn cookie-btn-accept">Accept</button>
          <button id="declineCookies" class="cookie-btn cookie-btn-decline">Decline</button>
        </div>
      </div>
    </div>

    <script src="/static/app.js"></script>
    <script>
      // Initialize AdSense ads after page loads
      window.addEventListener('load', function() {{
        setTimeout(function() {{
          try {{
            (adsbygoogle = window.adsbygoogle || []).push({{}});
            (adsbygoogle = window.adsbygoogle || []).push({{}});
          }} catch (e) {{
            console.warn('AdSense initialization error:', e);
          }}
        }}, 100);
      }});

      // Cookie consent handling
      (function() {{
        const consentKey = 'brfantasy_cookie_consent';
        const consentBanner = document.getElementById('cookieConsent');
        const acceptBtn = document.getElementById('acceptCookies');
        const declineBtn = document.getElementById('declineCookies');

        // Check if user has already made a choice
        const consent = localStorage.getItem(consentKey);
        if (!consent) {{
          consentBanner.style.display = 'block';
        }}

        acceptBtn.addEventListener('click', function() {{
          localStorage.setItem(consentKey, 'accepted');
          consentBanner.style.display = 'none';
        }});

        declineBtn.addEventListener('click', function() {{
          localStorage.setItem(consentKey, 'declined');
          consentBanner.style.display = 'none';
          // Optionally disable ads for users who decline
        }});
      }})();
    </script>
  </body>
</html>
"""


def normalize_sleeper_username(value: str) -> str:
    return (value or "").strip().lower()


def resolve_viewer_for_league(users: List[Dict], rosters: List[Dict], username: str) -> Union[Dict, None]:
    """
    Resolve a Sleeper username (or ESPN team/owner name) to:
      - user_id
      - roster_id
      - display_name / team name

    For ESPN leagues the `username` field holds the owner's display_name or team_name
    because ESPN doesn't use Sleeper-style usernames.
    """
    wanted = normalize_sleeper_username(username)
    if not wanted:
        return None

    matched_user = None
    for u in users or []:
        # Check display_name, username, and ESPN team_name (in metadata)
        meta = u.get("metadata") or {}
        candidates = [
            normalize_sleeper_username(u.get("display_name") or ""),
            normalize_sleeper_username(u.get("username") or ""),
            normalize_sleeper_username(meta.get("team_name") or ""),
        ]
        if wanted in candidates:
            matched_user = u
            break

    if not matched_user:
        return None

    user_id = str(matched_user.get("user_id") or "")
    if not user_id:
        return None

    matched_roster = None
    for r in rosters or []:
        owner_id = str(r.get("owner_id") or "")
        if owner_id == user_id:
            matched_roster = r
            break

    meta_u = matched_user.get("metadata") or {}
    if not matched_roster:
        return {
            "viewer_username": username,
            "viewer_user_id": user_id,
            "viewer_roster_id": None,
            "viewer_team_name": (
                    meta_u.get("team_name")
                    or matched_user.get("display_name")
                    or matched_user.get("username")
                    or "Unknown Team"
            ),
        }

    metadata = matched_roster.get("metadata") or {}
    team_name = (
            metadata.get("team_name")
            or meta_u.get("team_name")
            or matched_user.get("display_name")
            or matched_user.get("username")
            or f"Roster {matched_roster.get('roster_id')}"
    )

    return {
        "viewer_username": username,
        "viewer_user_id": user_id,
        "viewer_roster_id": str(matched_roster.get("roster_id")),
        "viewer_team_name": team_name,
    }


def save_viewer_session(viewer: dict) -> None:
    session["viewer_username"] = viewer.get("viewer_username")
    session["viewer_user_id"] = viewer.get("viewer_user_id")
    session["viewer_roster_id"] = viewer.get("viewer_roster_id")
    session["viewer_team_name"] = viewer.get("viewer_team_name")


def get_viewer_session_for_league(users: List[Dict], rosters: List[Dict]) -> dict:
    """Get viewer session resolved for the current league instead of stale session data."""
    session_viewer = get_viewer_session()
    username = session_viewer.get("viewer_username")

    if not username:
        return session_viewer

    # Resolve the viewer for this specific league
    league_viewer = resolve_viewer_for_league(users, rosters, username)

    if league_viewer:
        save_viewer_session(league_viewer)
        return league_viewer
    else:
        print(f"[get_viewer_session_for_league] Could not resolve {username} in current league, returning session data")
        return session_viewer


def get_viewer_session() -> dict:
    viewer_data = {
        "viewer_username": session.get("viewer_username"),
        "viewer_user_id": session.get("viewer_user_id"),
        "viewer_roster_id": session.get("viewer_roster_id"),
        "viewer_team_name": session.get("viewer_team_name"),
    }
    return viewer_data


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


def get_available_history_seasons(platform: str, league_id: str, current_season: int) -> List[int]:
    """
    Returns completed seasons only (excludes current season).
    """
    seasons = sorted(
        build_league_history_map(platform, league_id, current_season).keys(),
        reverse=True,
    )

    # remove current season
    seasons = [s for s in seasons if int(s) < int(current_season)]

    return seasons


def get_default_history_season(available_seasons: List[int], current_season: int) -> int:
    """
    Default to the most recent completed season, not the current season.
    If there is no prior season, fall back to the newest available season.
    """
    available = sorted({int(s) for s in available_seasons if s}, reverse=True)
    if not available:
        return int(current_season)

    past = [s for s in available if s < int(current_season)]
    if past:
        return past[0]

    return available[0]


def build_nav(league_id: Optional[str], active: str, platform: str, season: int) -> str:
    """
    active (league pages): 'dashboard','standings','power','weekly','teams','activity','injuries','trade','graphs'
    active (global pages): 'home','privacy','faq','contact','support'
    """
    nfl_state = get_nfl_state() or {}
    offseason_mode = ((nfl_state.get("season_type") or "").lower() == "off") and (
            int(nfl_state.get("season") or datetime.now().year) == int(season or 0)
    )

    # Changelog bell (used in both home and league nav)
    changelog_bell = (
        "<div class='changelog-bell-wrapper'>"
        "  <button type='button' id='changelogBell' class='changelog-bell-btn' aria-label='Recent Updates'>"
        "    <img src='/static/bell.png' style='width: 16px; height: 16px;' alt='Recent Updates'>"
        "  </button>"
        "  <div class='changelog-dot changelog-dot-hidden'></div>"
        "  <div id='changelogDropdown' class='changelog-dropdown' style='display:none;'></div>"
        "</div>"
    )

    # Dark mode toggle button HTML (used in settings dropdown)
    dark_mode_toggle_html = (
        "<button type='button' class='settings-menu-item' id='settingsDarkModeBtn'>"
        "  <img src='/static/moon.png' class='settings-menu-icon theme-icon light-icon' alt='Toggle dark mode'>"
        "  <img src='/static/moon.png' class='settings-menu-icon theme-icon dark-icon' style='display:none;' alt='Toggle dark mode'>"
        "  <span class='settings-menu-label'>Dark Mode</span>"
        "</button>"
    )

    # Build settings dropdown content (minimal for logged-out users)
    settings_content = dark_mode_toggle_html

    # Settings gear dropdown (used in both home and league nav)
    settings_gear = (
        "<div class='settings-gear-wrapper'>"
        "  <button type='button' id='settingsGearBtn' class='utility-icon-btn' "
        "          aria-label='Settings' title='Settings'>"
        "    <img src='/static/gear.png' style='width: 16px; height: 16px;' alt='Settings'>"
        "  </button>"
        f"  <div id='settingsDropdown' class='settings-dropdown' style='display:none;'>"
        f"    {settings_content}"
        "  </div>"
        "</div>"
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

        # Build utility bar for home screen (just settings gear with dark mode)
        home_utility_bar = (
            "<div class='nav-utility-bar'>"
            f"  {changelog_bell}"
            f"  {settings_gear}"
            "</div>"
        )

        # Build pills container
        pills_html = ''.join(pills)
        home_pills_container = (
            "<div class='nav-pills-container'>"
            f"  {pills_html}"
            "</div>"
        )

        return (
            "<nav class='top-nav'>"
            "  <div class='nav-left'>"
            "    <a href='/'>"
            "      <img src='/static/Website_Logo.png' alt='League Logo' class='site-logo'/>"
            "    </a>"
            "  </div>"
            "  <div class='nav-center'>"
            "    <button class='nav-hamburger' id='navToggle'>☰</button>"
            f"    {home_pills_container}"
            "  </div>"
            "  <div class='nav-right'>"
            f"    {home_utility_bar}"
            "  </div>"
            "</nav>"
        )

    # -------- League nav (with league_id) --------

    def nav_pill(label: str, endpoint: str, key: str) -> str:
        cls = "nav-pill active" if key == active else "nav-pill"
        href = url_for(endpoint, platform=platform, season=season, league_id=league_id)
        return f"<a class='{cls}' href='{href}'>{label}</a>"

    def nav_pill_dropdown(label: str, items: list, active_keys: list) -> str:
        """Build a dropdown nav pill. items = list of (label, endpoint_or_none, key, disabled)."""
        is_active = active in active_keys
        btn_cls = "nav-pill active" if is_active else "nav-pill"
        item_html = ""
        for item_label, endpoint, item_key, disabled in items:
            if disabled:
                item_html += (
                    f"<span class='nav-pill-dropdown-item disabled'>"
                    f"{item_label} <span style='font-size:10px;margin-left:4px;'>Soon</span>"
                    f"</span>"
                )
            else:
                href = url_for(endpoint, platform=platform, season=season, league_id=league_id)
                item_cls = "nav-pill-dropdown-item active" if item_key == active else "nav-pill-dropdown-item"
                item_html += f"<a class='{item_cls}' href='{href}'>{item_label}</a>"
        return (
            f"<div class='nav-pill-dropdown-wrapper' id='playersNavDropdown'>"
            f"  <button type='button' class='{btn_cls}' id='playersNavBtn' onclick='togglePlayersNav(event)'>"
            f"    {label} <span class='nav-pill-chevron'>&#x25BE;</span>"
            f"  </button>"
            f"  <div class='nav-pill-dropdown-menu' id='playersNavMenu'>"
            f"    {item_html}"
            f"  </div>"
            f"</div>"
        )

    # Generate dashboard URL for logo link
    dashboard_url = url_for("page_dashboard", platform=platform, season=season, league_id=league_id)

    # Navigation pills (no utilities)
    nav_pills = []
    nav_pills.append(nav_pill("Dashboard", "page_dashboard", "dashboard"))
    nav_pills.append(nav_pill("Trade Calc", "page_trade", "trade"))
    # Show Weekly Hub if draft has ended (during offseason) OR if in-season
    draft_ended = has_draft_ended(league_id, platform, season)
    if draft_ended or not offseason_mode:
        nav_pills.append(nav_pill("Weekly Hub", "page_weekly", "weekly"))
    nav_pills.append(nav_pill("Teams", "page_teams", "teams"))
    nav_pills.append(nav_pill("Activity", "page_activity", "activity"))
    nav_pills.append(nav_pill_dropdown("Players", [
        ("Player Rankings", "page_players",  "players",  False),
        ("Breakouts",       "page_breakouts","breakouts", False),
        ("Rookies",         "page_rookies",  "rookies",   False),
    ], ["players", "breakouts", "rookies"]))
    nav_pills.append(nav_pill("History", "page_history", "history"))

    # Standings and Graphs remain in-season only
    if not offseason_mode:
        nav_pills.append(nav_pill("Standings", "page_standings", "standings"))
        nav_pills.append(nav_pill("Graphs", "page_graphs", "graphs"))

    # Changelog bell
    # League switcher dropdown (if user is logged in)
    # Single switcher for settings dropdown (works on both desktop and mobile)
    league_switcher_html = ""
    viewer_username = session.get("viewer_username")
    if viewer_username:
        league_switcher_html = (
            f"<div class='league-switcher-wrapper'>"
            f"  <select id='leagueSwitcher' class='league-switcher' "
            f"          data-current-league='{league_id}' "
            f"          data-current-platform='{platform}' "
            f"          data-current-season='{season}' "
            f"          data-current-username='{viewer_username}'>"
            f"    <option value=''>Loading leagues...</option>"
            f"  </select>"
            f"</div>"
        )

        # Update settings dropdown content for logged-in users with full menu
        settings_content = (
            f"<button type='button' id='refreshBtn' class='settings-menu-item' "
            f"        data-page='{active}' data-league='{league_id}' "
            f"        data-platform='{platform}' data-season='{season}'>"
            "  <img src='/static/refresh.png' class='settings-menu-icon' alt='Refresh'>"
            "  <span class='settings-menu-label'>Refresh Data</span>"
            "</button>"
            "<button type='button' class='settings-menu-item' id='settingsChangelogBtn'>"
            "  <img src='/static/bell.png' class='settings-menu-icon' alt='Changelog'>"
            "  <span class='settings-menu-label'>Changelog</span>"
            "</button>"
            f"{dark_mode_toggle_html}"
            f"{league_switcher_html}"
            "<a href='/logout' class='settings-menu-item settings-menu-logout'>"
            "  <img src='/static/logout.png' class='settings-menu-icon' alt='Logout'>"
            "  <span class='settings-menu-label'>Logout</span>"
            "</a>"
        )

        # Rebuild settings gear with updated content
        settings_gear = (
            "<div class='settings-gear-wrapper'>"
            "  <button type='button' id='settingsGearBtn' class='utility-icon-btn' "
            "          aria-label='Settings' title='Settings'>"
            "    <img src='/static/gear.png' style='width: 16px; height: 16px;' alt='Settings'>"
            "  </button>"
            f"  <div id='settingsDropdown' class='settings-dropdown' style='display:none;'>"
            f"    {settings_content}"
            "  </div>"
            "</div>"
        )

    # Build utility bar (desktop right side, mobile header)
    utility_bar = (
        "<div class='nav-utility-bar'>"
        f"  {changelog_bell}"
        f"  {settings_gear}"
        "</div>"
    )

    # Build pills container (includes league switcher and logout for mobile menu)
    pills_html = ''.join(nav_pills)
    pills_container = (
        "<div class='nav-pills-container'>"
        f"  {pills_html}"
        "</div>"
    )

    return (
        "<nav class='top-nav'>"
        "  <div class='nav-left'>"
        f"    <a href='{dashboard_url}'>"
        "      <img src='/static/Website_Logo.png' alt='League Logo' class='site-logo'/>"
        "    </a>"
        "  </div>"
        "  <div class='nav-center'>"
        "    <button class='nav-hamburger' id='navToggle'>☰</button>"
        f"    {pills_container}"
        "  </div>"
        "  <div class='nav-right'>"
        f"    {utility_bar}"
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


def has_draft_ended(league_id: str, platform: str, season: int) -> bool:
    """
    Check if the draft has ended for a given league/season.
    Returns True if draft ended, False if not yet started or in progress.
    """
    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        if not ctx:
            return False

        latest_draft = ctx.get("latest_draft")
        league = ctx.get("league")

        draft_ts_ms = None
        if isinstance(latest_draft, dict):
            draft_ts_ms = _safe_int(latest_draft.get("start_time"))
        if draft_ts_ms is None:
            draft_ts_ms = _safe_int(league.get("draft_day"))

        if not draft_ts_ms:
            return False

        draft_dt = datetime.fromtimestamp(draft_ts_ms / 1000, tz=EASTERN)
        now_dt = datetime.now(EASTERN)
        return now_dt > draft_dt
    except Exception:
        return False


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


def _build_roster_map(users: list, rosters: list) -> dict:
    """Map roster_id → display name, using metadata.team_name with user fallback."""
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
    return roster_map


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

    # Populate league-wide globals (scoring, roster positions, etc.)
    # For Sleeper these were already populated by get_league() above.
    # For ESPN we need an explicit sync step.
    sync_league_globals(platform, resolved_league_id, season)

    # League settings
    scoring_settings = get_effective_scoring_settings()
    raw_scoring_settings = get_effective_scoring_settings() if "get_scoring_settings" in globals() else None
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
    roster_map = _build_roster_map(users, rosters)

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
        "viewer": get_viewer_session_for_league(users, rosters),
    }


def build_team_gm_context(ctx: dict, viewer_roster_id: str) -> Optional[dict]:
    rosters = ctx.get("rosters") or []
    roster = next((r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)), None)
    if not roster:
        return None

    roster_map = ctx.get("roster_map") or {}
    team_name = roster_map.get(str(viewer_roster_id)) or f"Roster {viewer_roster_id}"

    players_index = ctx.get("players_index") or {}
    players_map = ctx.get("players_map") or {}
    standings_map = ctx.get("standings_map") or {}
    picks_by_roster = ctx.get("picks_by_roster") or {}
    model_value_table = ctx.get("model_value_table") or {}
    roster_positions = ctx.get("roster_positions")
    # Handle DataFrame case - convert to list if it's a DataFrame
    if roster_positions is not None and hasattr(roster_positions, 'tolist'):
        roster_positions = roster_positions.tolist()
    elif roster_positions is None:
        roster_positions = []
    total_rosters = safe_float(ctx.get("total_rosters"), 10)
    team_stats = ctx.get("team_stats")
    # Handle DataFrame case - convert to list of dicts if it's a DataFrame
    if team_stats is not None and hasattr(team_stats, 'to_dict'):
        team_stats = team_stats.to_dict('records')
    elif team_stats is None:
        team_stats = []

    values_by_id = {}
    for row in model_value_table:
        if isinstance(row, dict) and row.get("id") is not None:
            values_by_id[str(row["id"])] = row

    standings = standings_map.get(str(viewer_roster_id), {}) or {}

    def pick_player_meta(pid: str) -> dict:
        mv = values_by_id.get(pid) or {}
        pmeta = players_index.get(pid) or players_map.get(pid) or {}

        position = (
                mv.get("position")
                or mv.get("pos")
                or pmeta.get("position")
                or pmeta.get("pos")
                or "?"
        )
        position = str(position).upper()

        team = mv.get("team") or pmeta.get("team") or ""
        age = mv.get("age")
        if age in (None, ""):
            age = pmeta.get("age")

        value = safe_float(mv.get("value"))
        name = (
                mv.get("name")
                or pmeta.get("full_name")
                or pmeta.get("name")
                or f"Player {pid}"
        )

        return {
            "id": pid,
            "name": name,
            "position": position,
            "team": team,
            "age": age,
            "value": value,
            "pos_rank_label": mv.get("pos_rank_label") or "",
        }

    all_player_ids = [str(pid) for pid in (roster.get("players") or [])]
    starter_ids = [str(pid) for pid in (roster.get("starters") or []) if str(pid) not in {"0", "", "None"}]

    players = [pick_player_meta(pid) for pid in all_player_ids]
    players.sort(key=lambda x: x["value"], reverse=True)

    starter_set = set(starter_ids)
    starters = [pick_player_meta(pid) for pid in starter_ids if pid in all_player_ids]
    starters.sort(key=lambda x: x["value"], reverse=True)

    bench = [p for p in players if p["id"] not in starter_set]
    bench.sort(key=lambda x: x["value"], reverse=True)

    pos_groups: dict[str, list[dict]] = {}
    for p in players:
        pos = p["position"]
        pos_groups.setdefault(pos, []).append(p)

    starter_pos_groups: dict[str, list[dict]] = {}
    for p in starters:
        pos = p["position"]
        starter_pos_groups.setdefault(pos, []).append(p)

    bench_pos_groups: dict[str, list[dict]] = {}
    for p in bench:
        pos = p["position"]
        bench_pos_groups.setdefault(pos, []).append(p)

    for group in pos_groups.values():
        group.sort(key=lambda x: x["value"], reverse=True)
    for group in starter_pos_groups.values():
        group.sort(key=lambda x: x["value"], reverse=True)
    for group in bench_pos_groups.values():
        group.sort(key=lambda x: x["value"], reverse=True)

    pos_summary = {}
    for pos, vals in pos_groups.items():
        numbers = [safe_float(p["value"]) for p in vals]
        ages = [safe_float(p["age"]) for p in vals if p.get("age") not in (None, "")]
        starter_vals = [safe_float(p["value"]) for p in starter_pos_groups.get(pos, [])]
        bench_vals = [safe_float(p["value"]) for p in bench_pos_groups.get(pos, [])]

        pos_summary[pos] = {
            "count": len(vals),
            "starter_count": len(starter_pos_groups.get(pos, [])),
            "bench_count": len(bench_pos_groups.get(pos, [])),
            "total_value": round(sum(numbers), 1),
            "top_1": round(sum(numbers[:1]), 1),
            "top_2": round(sum(numbers[:2]), 1),
            "top_3_sum": round(sum(numbers[:3]), 1),
            "top_5_sum": round(sum(numbers[:5]), 1),
            "best": round(numbers[0], 1) if numbers else 0.0,
            "starter_value": round(sum(starter_vals), 1),
            "bench_value": round(sum(bench_vals), 1),
            "avg_age": round(sum(ages) / len(ages), 1) if ages else None,
            "top_players": vals[:3],
        }

    future_picks = picks_by_roster.get(str(viewer_roster_id), []) or []

    pick_summary = {
        "total": len(future_picks),
        "firsts": 0,
        "seconds": 0,
        "thirds_plus": 0,
        "by_year": {},
    }

    cleaned_picks = []
    for pk in future_picks:
        if not isinstance(pk, dict):
            continue

        year = str(pk.get("season") or pk.get("year") or "")
        rnd = int(pk.get("round") or 0)

        if rnd == 1:
            pick_summary["firsts"] += 1
        elif rnd == 2:
            pick_summary["seconds"] += 1
        elif rnd >= 3:
            pick_summary["thirds_plus"] += 1

        if year:
            if year not in pick_summary["by_year"]:
                pick_summary["by_year"][year] = {"firsts": 0, "seconds": 0, "thirds_plus": 0}
            if rnd == 1:
                pick_summary["by_year"][year]["firsts"] += 1
            elif rnd == 2:
                pick_summary["by_year"][year]["seconds"] += 1
            elif rnd >= 3:
                pick_summary["by_year"][year]["thirds_plus"] += 1

        cleaned_picks.append({
            "season": year,
            "round": rnd,
            "original_owner": pk.get("original_owner_id"),
            "owner_id": pk.get("owner_id"),
        })

    ages = [safe_float(p.get("age")) for p in players if p.get("age") not in (None, "")]
    avg_age = sum(ages) / len(ages) if ages else 0.0

    total_value = round(sum(safe_float(p["value"]) for p in players), 1)
    starter_value_total = round(sum(safe_float(p["value"]) for p in starters), 1)
    bench_value_total = round(sum(safe_float(p["value"]) for p in bench), 1)

    elite_assets = sum(1 for p in players if p["value"] >= 675)
    strong_assets = sum(1 for p in players if p["value"] >= 500)
    insulated_assets = sum(1 for p in players if safe_float(p["age"]) <= 25 and p["value"] >= 400)
    aging_assets = [
        p for p in players
        if p.get("age") not in (None, "") and safe_float(p["age"]) >= 28 and p["value"] >= 250
    ][:6]

    premium_assets = [p for p in players if p["value"] >= 550][:8]
    liquid_trade_chips = [
        p for p in players
        if 225 <= p["value"] <= 650
    ][:8]

    young_core = [
        p for p in players
        if p.get("age") not in (None, "") and safe_float(p["age"]) <= 25 and p["value"] >= 300
    ][:8]

    fragile_assets = [
        p for p in players
        if (
                p.get("age") not in (None, "")
                and safe_float(p["age"]) >= 28
                and p["value"] >= 350
        )
    ][:6]

    weak_positions = []
    strong_positions = []
    for pos, meta in pos_summary.items():
        top3 = safe_float(meta.get("top_3_sum"))
        bench_val = safe_float(meta.get("bench_value"))
        starter_val = safe_float(meta.get("starter_value"))

        if starter_val >= 900 or top3 >= 900:
            strong_positions.append(pos)
        if starter_val <= 350 or (meta.get("count", 0) <= 1 and top3 <= 250):
            weak_positions.append(pos)
        elif bench_val <= 80 and meta.get("count", 0) <= 2:
            weak_positions.append(pos)

    strong_positions = list(dict.fromkeys(strong_positions))
    weak_positions = list(dict.fromkeys(weak_positions))

    firsts = pick_summary["firsts"]

    if elite_assets >= 3 and avg_age and avg_age <= 27.5 and starter_value_total >= 2600:
        direction = "contender"
    elif firsts >= 3 and elite_assets < 2 and avg_age >= 25.5:
        direction = "rebuild"
    elif firsts >= 2 or (len(young_core) >= 4 and elite_assets < 3):
        direction = "retool"
    else:
        direction = "balanced"

    roster_health = "stable"
    if len(weak_positions) >= 2 and bench_value_total < 700:
        roster_health = "fragile"
    elif len(strong_positions) >= 2 and bench_value_total >= 850:
        roster_health = "deep"
    elif len(premium_assets) <= 2 and firsts >= 2:
        roster_health = "transitioning"

    record = standings.get("record") or standings.get("display_record") or ""
    wins = safe_float(standings.get("wins"))
    losses = safe_float(standings.get("losses"))
    ties = safe_float(standings.get("ties"))
    pf = round(safe_float(standings.get("PF")), 1)
    pa = round(safe_float(standings.get("PA")), 1)

    win_pct = 0.0
    games_played = wins + losses + ties
    if games_played > 0:
        win_pct = round((wins + (0.5 * ties)) / games_played, 3)

    place = None
    if team_stats:
        try:
            sorted_stats = sorted(
                team_stats,
                key=lambda x: (
                    -safe_float(x.get("win_pct")),
                    -safe_float(x.get("avg")),
                    -safe_float(x.get("pf")),
                ),
            )
            for idx, row in enumerate(sorted_stats, start=1):
                rid = str(row.get("roster_id") or "")
                if rid == str(viewer_roster_id):
                    place = idx
                    break
        except Exception:
            place = None

    lineup_requirements = {}
    if roster_positions and isinstance(roster_positions, (list, tuple)):
        for slot in roster_positions:
            slot_str = str(slot).upper()
            if slot_str in {"QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX", "SFLEX"}:
                lineup_requirements[slot_str] = lineup_requirements.get(slot_str, 0) + 1

    starter_profile = {
        "count": len(starters),
        "total_value": starter_value_total,
        "avg_value": round(starter_value_total / len(starters), 1) if starters else 0.0,
        "top_starters": starters[:6],
    }

    bench_profile = {
        "count": len(bench),
        "total_value": bench_value_total,
        "avg_value": round(bench_value_total / len(bench), 1) if bench else 0.0,
        "top_bench": bench[:6],
    }

    market_profile = {
        "premium_assets": premium_assets,
        "liquid_trade_chips": liquid_trade_chips,
        "young_core": young_core,
        "fragile_assets": fragile_assets,
        "aging_assets": aging_assets,
    }

    summary_flags = []
    if direction == "contender":
        summary_flags.append("win-now build")
    if direction == "rebuild":
        summary_flags.append("future-oriented")
    if firsts >= 2:
        summary_flags.append("has first-round capital")
    if len(weak_positions) >= 2:
        summary_flags.append("multiple roster holes")
    if len(strong_positions) >= 2:
        summary_flags.append("clear strength pockets")
    if roster_health == "fragile":
        summary_flags.append("thin depth")
    if roster_health == "deep":
        summary_flags.append("strong depth")

    return {
        "league_id": ctx.get("league_id"),
        "season": ctx.get("season") or ctx.get("current_season"),
        "viewer_roster_id": str(viewer_roster_id),
        "team_name": team_name,
        "record": record,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_pct": win_pct,
        "place": place,
        "league_size": int(total_rosters) if total_rosters else None,
        "points_for": pf,
        "points_against": pa,
        "avg_age": round(avg_age, 2) if avg_age else 0.0,
        "direction": direction,
        "roster_health": roster_health,
        "summary_flags": summary_flags,
        "total_roster_value": total_value,
        "starter_value_total": starter_value_total,
        "bench_value_total": bench_value_total,
        "elite_assets_count": elite_assets,
        "strong_assets_count": strong_assets,
        "insulated_assets_count": insulated_assets,
        "lineup_requirements": lineup_requirements,
        "top_assets": players[:10],
        "starters": starters[:10],
        "bench": bench[:10],
        "starter_profile": starter_profile,
        "bench_profile": bench_profile,
        "position_strength": pos_summary,
        "strong_positions": strong_positions,
        "weak_positions": weak_positions,
        "future_picks": cleaned_picks[:12],
        "pick_summary": pick_summary,
        "market_profile": market_profile,
    }


def get_trade_ai_analysis(
        ctx: dict,
        viewer_roster_id: str,
        viewer_side: str,
        side_a: dict,
        side_b: dict,
) -> str:
    """Get AI analysis for a trade using the new generator module"""
    from dashboard_services.ai.renderer import get_trade_ai_analysis as renderer_analysis
    return renderer_analysis(ctx, viewer_roster_id, viewer_side, side_a, side_b)


@app.route("/api/history/ai-recap")
def history_ai_recap():
    """Generate AI-powered season recap for a specific team."""
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


@app.route("/api/history/<platform>/<int:season>/<league_id>/summary")
def api_history_summary(platform: str, season: int, league_id: str):
    """Get season awards/summary data."""
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
        print(f"[api_history_summary] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/<platform>/<int:season>/<league_id>/standings")
def api_history_standings(platform: str, season: int, league_id: str):
    """Get regular season standings."""
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
        print(f"[api_history_standings] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/<platform>/<int:season>/<league_id>/chart")
def api_history_chart(platform: str, season: int, league_id: str):
    """Get season trend chart data."""
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
        print(f"[api_history_chart] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def render_simple_ai_copy(title: str, subtitle: str, text: str) -> str:
    return f"""
    <div class="ai-copy">
      <p><strong>{html.escape(title)}</strong></p>
      <p>{html.escape(subtitle)}</p>
      <div>{html.escape(text)}</div>
    </div>
    """


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

    viewed_season = int(ctx.get("season") or season)
    current_season = int(ctx.get("current_season") or viewed_season)
    current_week = int(ctx.get("current_week") or 0)
    weeks = int(ctx.get("weeks") or 0)
    season_type = (ctx.get("season_type") or "").lower()
    season_complete = bool(ctx.get("season_complete", False))
    offseason_mode = bool(ctx.get("offseason_mode", False))
    resolved_league_id = ctx.get("resolved_league_id", league_id)

    # ---------- Always refresh core league objects ----------
    league = get_league(platform, resolved_league_id, viewed_season)
    users = get_users(platform, resolved_league_id, viewed_season)
    rosters = get_rosters(platform, resolved_league_id, viewed_season)

    traded = None
    if platform == "sleeper":
        try:
            traded = get_traded_picks(platform, resolved_league_id, viewed_season)
        except Exception:
            traded = ctx.get("traded")

    try:
        drafts = get_drafts(platform, resolved_league_id, viewed_season) or []
        latest_draft = get_most_recent_valid_draft_for_season(drafts, viewed_season)
    except Exception:
        drafts = ctx.get("drafts") or []
        latest_draft = ctx.get("latest_draft")

    ctx["league"] = league
    ctx["users"] = users
    ctx["rosters"] = rosters
    ctx["traded"] = traded
    ctx["drafts"] = drafts
    ctx["latest_draft"] = latest_draft

    players = ctx["players"]
    players_map = ctx["players_map"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]

    # ---------- Rebuild roster_map every refresh ----------
    ctx["roster_map"] = _build_roster_map(users, rosters)

    # ---------- League settings / refs that can matter to multiple pages ----------
    sync_league_globals(platform, resolved_league_id, viewed_season)
    try:
        ctx["scoring_settings"] = get_effective_scoring_settings()
    except Exception as e:
        print(f"[ctx] scoring_settings failed: {e}")

    try:
        ctx["roster_positions"] = get_roster_positions()
    except Exception as e:
        print(f"[ctx] roster_positions failed: {e}")

    try:
        ctx["league_settings"] = get_league_settings()
    except Exception as e:
        print(f"[ctx] league_settings failed: {e}")

    try:
        ctx["total_rosters"] = get_total_rosters()
    except Exception as e:
        print(f"[ctx] total_rosters failed, falling back to len(rosters): {e}")
        ctx["total_rosters"] = len(rosters)

    roster_counts = count_roster_positions(get_roster_positions())
    has_idp = any(k in roster_counts for k in ["DL", "LB", "DB", "IDP_FLEX"])

    # ---------- Shared live / reference data ----------
    try:
        scores_body = get_nfl_scores_for_date(date.today().strftime("%Y%m%d"))
        ctx["team_game_lookup"] = build_team_game_lookup(scores_body)
    except Exception:
        pass

    # ---------- Standings / dashboard / weekly core tables ----------
    if page in ("standings", "dashboard", "weekly"):
        if offseason_mode:
            ctx["df_weekly"] = pd.DataFrame()
            ctx["team_stats"] = pd.DataFrame()
            ctx["standings_map"] = {}
        else:
            df_weekly, team_stats, fresh_roster_map = build_tables(
                league_id=resolved_league_id,
                max_week=weeks,
                players=players,
                users=users,
                rosters=rosters,
                season=viewed_season,
                platform=platform,
            )

            # Prefer fresh roster_map from build_tables if returned
            if isinstance(fresh_roster_map, dict) and fresh_roster_map:
                ctx["roster_map"] = fresh_roster_map
                roster_map = fresh_roster_map

            ctx["df_weekly"] = df_weekly
            ctx["team_stats"] = team_stats

            if team_stats is not None and not team_stats.empty and {"Wins", "PF"}.issubset(team_stats.columns):
                ctx["standings_map"] = build_standings_map(team_stats, roster_map)
            else:
                ctx["standings_map"] = {}

    # ---------- Activity / injuries ----------
    if page in ("activity", "dashboard"):
        clear_activity_cache_for_league(resolved_league_id)

        ctx["activity_df"] = build_week_activity(
            resolved_league_id,
            platform,
            viewed_season,
            players_map,
        )

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

    # ---------- Weekly / dashboard projections & matchups ----------
    if page in ("weekly", "dashboard"):
        clear_weekly_cache_for_league(resolved_league_id)

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
                try:
                    live_game_ids = get_live_game_ids_for_today(
                        load_week_schedule(current_season, current_week)
                    )
                    build_and_save_week_stats_for_league(
                        load_teams_index(),
                        current_season,
                        current_week,
                        live_game_ids,
                    )
                except Exception as e:
                    print(f"[refresh] live week stats refresh skipped: {e}")

                try:
                    get_week_projections_cached(
                        current_season,
                        current_week,
                        fetch_week_from_tank01,
                        True,
                    )
                except Exception as e:
                    print(f"[refresh] live projections refresh skipped: {e}")

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
            if df is not None and not df.empty and "week" in df.columns and "roster_id" in df.columns:
                key_series = list(zip(df["week"].astype(int), df["roster_id"].astype(str)))
                df = df.copy()
                df["proj"] = [proj_by_roster.get(k, float("nan")) for k in key_series]
                ctx["df_weekly"] = df

    # ---------- Teams page ----------
    if page == "teams":
        clear_teams_cache_for_league(resolved_league_id)
        ctx["model_value_table"] = ctx.get("model_value_table") or load_model_value_table() or []

    # ---------- Trade page ----------
    if page == "trade":
        # Keep the shared table fresh so the trade calc reflects newest values
        ctx["model_value_table"] = ctx.get("model_value_table") or load_model_value_table() or []

        # also refresh global model-value API cache used by /api/trade-eval
        global _MODEL_VALUE_CACHE, _MODEL_VALUE_CACHE_TS
        _MODEL_VALUE_CACHE = ctx["model_value_table"]
        _MODEL_VALUE_CACHE_TS = time.time()

    # ---------- Offseason dashboard refresh ----------
    if page == "dashboard" and offseason_mode:
        ctx["model_value_table"] = ctx.get("model_value_table") or load_model_value_table() or []

        if platform == "sleeper":
            try:
                ctx["picks_by_roster"] = build_picks_by_roster(
                    num_future_seasons=3,
                    league=league,
                    rosters=rosters,
                    traded=traded,
                )
            except Exception as e:
                print(f"[refresh] picks refresh skipped: {e}")

        # These aren't strictly required because build_offseason_dashboard_body can rebuild
        # from ctx, but keeping them fresh helps consistency if reused elsewhere.
        try:
            ctx["teams_overview"] = build_teams_overview(
                rosters=rosters,
                users_list=users,
                picks_by_roster=ctx.get("picks_by_roster", {}),
                players=players_map,
                players_index=players_index,
                teams_index=teams_index,
                platform=platform,
            )
        except Exception as e:
            print(f"[refresh] teams overview refresh skipped: {e}")

    # ---------- Invalidate rendered HTML cache for refreshable pages ----------
    page_html = entry.setdefault("page_html", {})
    if page == "dashboard":
        for p in ("dashboard", "activity", "teams", "graphs", "standings", "weekly"):
            page_html.pop(p, None)
    else:
        page_html.pop(page, None)

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
    season = ctx["current_season"]
    rosters = ctx["rosters"]
    users = ctx["users"]
    current_week = ctx["current_week"]
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

    viewer = ctx.get("viewer") or {}
    viewer_roster_id = viewer.get("viewer_roster_id")

    print(f"[dashboard] Building dashboard body")
    print(f"[dashboard] Viewer data: {viewer}")
    print(f"[dashboard] Viewer roster ID: {viewer_roster_id}")

    gm_memo_html = ""
    front_office_html = ""

    if viewer_roster_id:
        print(f"[dashboard] Attempting to get GM memo for roster {viewer_roster_id}")
        try:
            gm_memo_html = get_team_gm_memo(ctx, str(viewer_roster_id))
            print(f"[dashboard] GM memo result: {len(gm_memo_html)} chars")
        except Exception as e:
            print(f"[dashboard] gm memo exception: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[dashboard] No viewer_roster_id, skipping GM memo")

        try:
            front_office_html = get_front_office_briefing(ctx, str(viewer_roster_id))
        except Exception as e:
            print(f"[dashboard] front office briefing skipped: {e}")

    standings_html = render_standings(team_stats, 5)

    finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())
    else:
        last_final_week = current_week

    slides = [
        render_matchup_slide(
            season,
            m,
            current_week,
            last_final_week,
            status_by_pid=statuses[current_week].get("statuses", {}),
            projections=proj_by_week,
            players=players_index,
            teams=teams_index,
            team_game_lookup=team_game_lookup,
        )
        for m in matchups_by_week.get(current_week, [])
    ]
    slides_by_week = {current_week: "".join(slides)}
    matchup_html = render_matchup_carousel_weeks(
        slides_by_week,
        dashboard=True,
        active_week=current_week,
    )

    awards = compute_awards_season(finalized_df, players_map, league_id)
    awards_html = render_awards_section(awards)

    teams_ctx = build_teams_overview(
        rosters=rosters,
        users_list=users,
        picks_by_roster=picks_by_roster,
        players=players_map,
        players_index=players_index,
        teams_index=teams_index,
    )
    teams_sidebar_html = render_teams_sidebar(teams_ctx)

    gm_card_html = ""
    if gm_memo_html:
        gm_card_html = f"""
        <div class="card gm-card">
          <div class="card-header">
            <h2>Your GM Memo</h2>
            <div class="subtle-label">{viewer.get("viewer_team_name") or "Your Team"}</div>
          </div>
          <div class="card-body">
            {gm_memo_html}
          </div>
        </div>
        """

    front_office_card_html = ""
    if front_office_html:
        front_office_card_html = f"""
        <div class="card fo-brief-card">
          <div class="card-header">
            <h2>Front Office Briefing</h2>
            <div class="subtle-label">Daily plan</div>
          </div>
          <div class="card-body">
            {front_office_html}
          </div>
        </div>
        """

    body = f"""
    <aside class="overview-sidebar-left">
      {awards_html}
    </aside>
    <div class="overview-main">
      {gm_card_html}
      {front_office_card_html}
      <div class="card central">
        <h2>Standings</h2>
        {standings_html}
      </div>
      {matchup_html}
    </div>
    <aside class="overview-sidebar">
      {teams_sidebar_html}
    </aside>
    """

    return body


def render_power_and_playoffs(team_stats, roster_map: Dict[str, str], league_id: str, platform, season) -> str:
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

    viewer = ctx.get("viewer") or {}
    viewer_roster_id = viewer.get("viewer_roster_id")

    front_office_html = ""
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

            if delta_days >= 0:
                # Draft hasn't happened yet
                countdown_text = f"{delta_days} days"
                draft_subtext = "Countdown to your next league draft."
            else:
                # Draft has passed - show Week 1 countdown
                nfl_state = get_nfl_state() or {}
                season_start_date = nfl_state.get("season_start_date")

                if season_start_date:
                    try:
                        # Parse season start date (format: "YYYY-MM-DD")
                        from dateutil import parser
                        week1_dt = parser.parse(season_start_date).replace(tzinfo=EASTERN)
                        week1_delta = (week1_dt.date() - now_dt.date()).days

                        if week1_delta > 0:
                            countdown_text = f"{week1_delta} days"
                            draft_subtext = "Countdown to Week 1 kickoff."
                        elif week1_delta == 0:
                            countdown_text = "Today!"
                            draft_subtext = "Week 1 starts today!"
                        else:
                            countdown_text = "Season started"
                            draft_subtext = "Week 1 is underway!"
                    except Exception as e:
                        print(f"[offseason] Failed to parse season_start_date: {e}")
                        countdown_text = "Draft passed"
                        draft_subtext = "Season starting soon."
                else:
                    countdown_text = "Draft passed"
                    draft_subtext = "Awaiting season start date."
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

    # Calculate Draft Capital Index from pick value table
    pick_value_table = load_pick_value_table() or {}
    total_draft_capital = sum(pick_value_table.values())

    roster_cards = []

    for r in rosters:
        rid = str(r.get("roster_id"))
        team_name = roster_map.get(rid, f"Roster {rid}")
        player_ids = [str(pid) for pid in (r.get("players") or [])]
        roster_value = sum(values_by_id.get(pid, 0.0) for pid in player_ids)
        team_picks = picks_by_roster.get(rid, []) if isinstance(picks_by_roster, dict) else []
        pick_count = len(team_picks)

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
            "roster_id": rid,
            "html": f"""
            <div class="os-snapshot-card team-clickable" style="cursor:pointer;" data-roster-id="{rid}" data-team-name="{team_name}">
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
            "player_id": pid,
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
                <div class="os-waiver-name player-clickable" style="cursor:pointer;font-weight:600;" data-player-id='{p['player_id']}' data-player-name='{p['name']}'>{p['name']}</div>
                <div class="os-waiver-sub">{subline}</div>
              </div>
              <div class="os-waiver-value">{p['value']:.0f}</div>
            </div>
            """
        )

    top_waiver_assets_html = "".join(waiver_html)

    # Build matchup carousel (even if offseason/preseason - show with 0 projections)
    matchup_html = ""
    try:
        from dashboard_services.matchups import render_matchup_slide, render_matchup_carousel_weeks

        matchups_by_week = ctx.get("matchups_by_week", {})
        proj_by_week = ctx.get("proj_by_week", {})
        statuses = ctx.get("statuses", {})
        players_index = ctx.get("players_index", {})
        teams_index = ctx.get("teams_index", {})
        team_game_lookup = ctx.get("team_game_lookup", {})
        current_week = ctx.get("current_week", 1)

        # Generate slides for Week 1 (or current week)
        week_to_show = current_week if current_week > 0 else 1
        matchups_for_week = matchups_by_week.get(week_to_show, [])

        if matchups_for_week:
            slides = [
                render_matchup_slide(
                    season,
                    m,
                    week_to_show,
                    week_to_show,
                    status_by_pid=statuses.get(week_to_show, {}).get("statuses", {}),
                    projections=proj_by_week,
                    players=players_index,
                    teams=teams_index,
                    team_game_lookup=team_game_lookup,
                )
                for m in matchups_for_week
            ]
            slides_by_week = {week_to_show: "".join(slides)}
            matchup_html = render_matchup_carousel_weeks(
                slides_by_week,
                dashboard=True,
                active_week=week_to_show,
            )
    except Exception as e:
        print(f"[offseason] Failed to generate matchup carousel: {e}")
        matchup_html = ""

    gm_card_html = ""
    if viewer_roster_id:
        # Show button to generate GM memo instead of auto-generating
        gm_card_html = f"""
        <section class="os-card">
          <div class="os-section-head">
            <div class="os-section-head-content">
              <h2 class="os-section-title">BR Front Office Report</h2>
              <div class="os-section-subtitle">{viewer.get("viewer_team_name") or "Your Team"}</div>
            </div>
            <div class="os-section-head-actions">
              <button type="button" id="generateGmMemoBtn" class="recap-generate-btn" 
                      data-league-id="{ctx.get('league_id')}" 
                      data-season="{ctx.get('season')}" 
                      data-platform="{ctx.get('platform')}" 
                      data-viewer-roster-id="{viewer_roster_id}">
                Generate Report
              </button>
              <button type="button" class="card-collapse-toggle" data-target="gm-memo-body">▼</button>
            </div>
          </div>
          <div class="os-ai-copy card-collapsible-body" id="gm-memo-body">
            <div class="otc-ai-empty" id="gm-memo-empty">
              <div class="otc-ai-empty-sub">
                Get personalized analysis on your roster, trade targets, and offseason strategy.
              </div>
            </div>
            <div class="otc-ai-empty" id="gm-memo-loading" style="display:none;">
              <div class="otc-ai-empty-title">Analyzing Your Roster...</div>
              <div class="otc-ai-empty-sub">
                <div class="loading-spinner" style="margin: 10px auto; width: 30px; height: 30px; border: 3px solid #f3f4f6; border-radius: 50%; border-top-color: #3498db; animation: spin 1s linear infinite; border-right-color: transparent;"></div>
              </div>
            </div>
            <div id="gm-memo-result" style="display:none;"></div>
          </div>
        </section>
        """

    front_office_card_html = ""
    if front_office_html:
        front_office_card_html = f"""
        <section class="os-card">
          <div class="os-section-head">
            <div class="os-section-head-content">
              <h2 class="os-section-title">Front Office Briefing</h2>
              <div class="os-section-subtitle">Offseason priorities</div>
            </div>
            <button type="button" class="card-collapse-toggle" data-target="front-office-body">▼</button>
          </div>
          <div class="os-ai-copy card-collapsible-body" id="front-office-body">
            {front_office_html}
          </div>
        </section>
        """

    body = f"""
    <div class="os-layout">
      <aside class="os-left-col">
        <section class="os-card os-card-soft">
          <div class="os-section-head">
            <div class="os-section-head-content">
              <h2 class="os-section-title">Offseason Team Snapshot</h2>
              <div class="os-section-subtitle">Roster value and future capital across the league</div>
            </div>
            <button type="button" class="card-collapse-toggle" data-target="team-snapshot-body">▼</button>
          </div>
          <div class="os-snapshot-list card-collapsible-body" id="team-snapshot-body">
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
              <div class="os-stat-label">Draft Capital Index</div>
              <div class="os-stat-value">{total_draft_capital:.0f}</div>
              <div class="os-stat-sub">Based on modeled pick values</div>
            </div>
          </div>

          <div class="os-hero-footer">
            {draft_subtext}
          </div>
        </section>

        {gm_card_html}
        {front_office_card_html}

        {matchup_html}

        <section class="os-card">
          <div class="os-section-head">
            <div class="os-section-head-content">
              <h2 class="os-section-title">Top Waiver Assets</h2>
              <div class="os-section-subtitle">Best currently unrostered players by BR value</div>
            </div>
            <button type="button" class="card-collapse-toggle" data-target="waiver-assets-body">▼</button>
          </div>
          <div class="os-waiver-list card-collapsible-body" id="waiver-assets-body">
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

    # 3. Caps: at most 75% of the stud value, or 50% of the consolidating
    #    side's total player value — whichever is smaller.
    cap_stud = 0.75 * stud_val
    cap_side = 0.50 * fewer_players_total
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
              <span class="hg-name team-clickable" style="cursor:pointer;" data-roster-id="{top['roster_id']}" data-team-name="{top['owner']}">{top['owner']}</span>
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
              <span class="hg-name team-clickable" style="cursor:pointer;" data-roster-id="{low['roster_id']}" data-team-name="{low['owner']}">{low['owner']}</span>
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
            "winner_rid": win["roster_id"],
            "winnerPts": float(win["use_score"]),
            "loser": lose["owner"],
            "loser_rid": lose["roster_id"],
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
                <span class="hg-name team-clickable" style="cursor:pointer;" data-roster-id="{closest['winner_rid']}" data-team-name="{closest['winner']}">{closest['winner']}</span>
                <span class="hg-score">{closest['winnerPts']:.1f}</span>
              </div>
              <div class="hg-row">
                <span class="hg-name team-clickable" style="cursor:pointer;" data-roster-id="{closest['loser_rid']}" data-team-name="{closest['loser']}">{closest['loser']}</span>
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
                <span class="hg-name team-clickable" style="cursor:pointer;" data-roster-id="{blowout['winner_rid']}" data-team-name="{blowout['winner']}">{blowout['winner']}</span>
                <span class="hg-score">{blowout['winnerPts']:.1f}</span>
              </div>
              <div class="hg-row">
                <span class="hg-name team-clickable" style="cursor:pointer;" data-roster-id="{blowout['loser_rid']}" data-team-name="{blowout['loser']}">{blowout['loser']}</span>
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

    proj_warn_html = ""
    if not proj_by_week.get("_available"):
        proj_warn_html = (
            "<div class='card' style='margin-bottom:12px;background:#fffbeb;border:1px solid #f59e0b;'>"
            "  <div class='card-body' style='padding:10px 14px;font-size:13px;color:#92400e;'>"
            "    <strong>Projections unavailable</strong> — projected scores can't be loaded right now. "
            "    Actual scores will still appear once games are final."
            "  </div>"
            "</div>"
        )

    main_panel_html = f"""
          <div class="week-main-panel active" data-week="{default_week}">
            {proj_warn_html}
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
    any_projections = False
    for w in range(1, weeks + 1):
        projections = load_week_projection(season, w)
        bundles[w] = {"projections": projections or {}}
        if projections:
            any_projections = True
    if not any_projections:
        print(f"[projections] No projection data available for season {season}")
    bundles["_available"] = any_projections
    return bundles


def build_status_by_week(season: int, weeks: int, players_index, teams_index, idp_player_index: Dict[str, Dict] = None):
    bundles = {}
    for w in range(1, weeks + 1):
        try:
            statuses = build_status_for_week(season, w, players_index, teams_index, idp_player_index)
            bundles[w] = {"statuses": statuses}
        except Exception as e:
            print(f"Error loading week {w} schedule: {e}")
            bundles[w] = {"statuses": {}}
    return bundles


HISTORICAL_PICK_SLOT_CACHE: Dict[Tuple[str, str, int], Dict[int, int]] = {}

def build_historical_pick_slot_map(
        platform: str,
        root_league_id: str,
        current_season: int,
        source_season: int,
) -> Dict[int, int]:
    """
    For a given source season, returns:
      { roster_id: rookie_pick_slot }

    Example:
      source_season=2025 -> order used for 2026 rookie picks
    """
    cache_key = (str(platform).lower(), str(root_league_id), int(source_season))
    cached = HISTORICAL_PICK_SLOT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    resolved_league_id = resolve_league_id_for_season(
        platform=platform,
        league_id=root_league_id,
        current_season=current_season,
        target_season=source_season,
    )

    hist_ctx = get_league_ctx_from_cache(
        platform,
        resolved_league_id,
        source_season,
    )

    df_weekly = hist_ctx.get("df_weekly", pd.DataFrame())
    league = hist_ctx.get("league") or {}
    roster_map = hist_ctx.get("roster_map") or {}

    reg_team_stats = build_regular_season_team_stats(df_weekly, league)
    reg_team_stats = sort_team_stats(reg_team_stats)

    if reg_team_stats is None or reg_team_stats.empty:
        HISTORICAL_PICK_SLOT_CACHE[cache_key] = {}
        return {}

    # roster_map is expected to look like {roster_id: owner/team_name}
    name_to_roster_id: Dict[str, int] = {}
    for rid, team_name in roster_map.items():
        try:
            name_to_roster_id[str(team_name)] = int(rid)
        except Exception:
            continue

    total_teams = len(reg_team_stats)
    slot_map: Dict[int, int] = {}

    # Rank 1 = best team, so reverse for rookie draft slot:
    # worst team -> 1, next worst -> 2, etc.
    for _, row in reg_team_stats.iterrows():
        owner = str(row.get("owner") or "")
        rank = _safe_int(row.get("Rank"), 0)
        roster_id = name_to_roster_id.get(owner)

        if not owner or rank <= 0 or roster_id is None:
            continue

        slot = total_teams - rank + 1
        slot_map[int(roster_id)] = int(slot)

    HISTORICAL_PICK_SLOT_CACHE[cache_key] = slot_map
    return slot_map


def resolve_exact_pick_slot(
        platform: str,
        root_league_id: str,
        current_season: int,
        pick: dict,
) -> Union[int, None]:
    """
    For a 2026 pick, look at 2025 standings of the previous owner.
    """
    pick_year = _safe_int(pick.get("season"), 0)
    rnd = _safe_int(pick.get("round"), 0)

    if not pick_year or not rnd:
        return None

    source_season = pick_year - 1
    if source_season <= 0:
        return None

    prev_owner = pick.get("previous_owner_id")
    if prev_owner is None:
        prev_owner = pick.get("owner_id")

    try:
        prev_owner = int(prev_owner)
    except Exception:
        return None

    slot_map = build_historical_pick_slot_map(
        platform=platform,
        root_league_id=root_league_id,
        current_season=current_season,
        source_season=source_season,
    )

    return slot_map.get(prev_owner)


def format_pick_round_label(pick: dict) -> str:
    rnd = _safe_int(pick.get("round"), 0)
    slot = _safe_int(pick.get("slot"), 0)
    if rnd <= 0:
        return "Pick"
    if slot > 0:
        return f"{rnd}.{slot:02d}"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
    return f"{rnd}{suffix}"


def format_pick_display_label(
        platform: str,
        root_league_id: str,
        current_season: int,
        pick: dict,
) -> str:
    year = _safe_int(pick.get("season"), 0)
    rnd = _safe_int(pick.get("round"), 0)

    if not year or not rnd:
        return "Pick"

    exact_slot = resolve_exact_pick_slot(
        platform=platform,
        root_league_id=root_league_id,
        current_season=current_season,
        pick=pick,
    )

    if exact_slot is not None:
        return f"{year} {rnd}.{exact_slot:02d}"

    return f"{year} {format_pick_round_label(pick)}"


def build_activity_body(ctx: dict) -> str:
    league_id = ctx["league_id"]
    resolved_league_id = ctx.get("resolved_league_id", league_id)
    activity_df = ctx["activity_df"]
    injury_df = ctx["injury_df"]
    standings_map = ctx["standings_map"]
    platform = ctx["platform"]
    season = _safe_int(ctx["season"], 0)

    players_values_raw = ctx.get("model_value_table") or []
    player_val_by_key: Dict[Tuple[str, str, str], float] = {}
    player_val_by_key_np: Dict[Tuple[str, str], float] = {}
    rank_label_by_name: Dict[str, str] = {}

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

    def pick_value(pick: Dict, standings_map: Dict[int, int], num_teams: int = 10) -> float:
        """
        Prefer exact historical slot when available, then fall back to bucketed values.
        """
        year = _safe_int(pick.get("season"), 0)
        rnd = _safe_int(pick.get("round"), 0)
        if not year or not rnd:
            return 0.0

        exact_slot = resolve_exact_pick_slot(
            platform=platform,
            root_league_id=league_id,
            current_season=season,
            pick=pick,
        )

        if exact_slot is not None:
            exact_key = f"{year}_{rnd}_{exact_slot:02d}"
            if exact_key in pick_values:
                return float(pick_values[exact_key])

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

        exact_slot = resolve_exact_pick_slot(
            platform=platform,
            root_league_id=league_id,
            current_season=season,
            pick=pick,
        )

        bucket = pick_bucket_from_seed(seed, num_teams=num_teams)
        bucket_label = None

        if exact_slot is not None:
            bucket_label = f"Pick {pick.get('round')}.{int(exact_slot):02d}"
        elif bucket:
            bucket_label = bucket.capitalize()

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
    most_active_counts: Dict[str, int] = {}
    traded_asset_counts: Dict[str, int] = {}
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

                # Make player name clickable using the pid from the player dict
                pid = p.get("pid", "")
                clickable_attrs = f" class='player-clickable' style='cursor:pointer;font-weight:600;' data-player-id='{pid}' data-player-name='{name}'" if pid else " style='font-weight:600'"

                return (
                    "<div class='player-activity'>"
                    "<div style='gap: 10px;display: flex;align-items: center;'>"
                    f"<span class='io {io_class}'>"
                    f"{'+' if io_class == 'add' else '−'}</span>"
                    "<div>"
                    f"  <div{clickable_attrs}>{name}</div>"
                    f"  <div style='color:#64748b;font-size:12px'>{pos_rank_label} • {p['team']}</div>"
                    "</div></div>"
                    f"{val_html}</div>"
                )

            def render_pick_row(pick, io_class):
                traded_asset_counts["Draft Pick"] = traded_asset_counts.get("Draft Pick", 0) + 1

                pick_label = format_pick_display_label(
                    platform=platform,
                    root_league_id=league_id,
                    current_season=season,
                    pick=pick,
                )
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
                    f"  <div style='font-weight:600'>{pick_label}</div>"
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

            side_map: Dict[int, Dict] = {}
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
                team_name = tm.get('name', '')
                roster_id = tm.get('roster_id', '')
                cols.append(
                    "<div class='team-col'>"
                    f"  <header>{img}<div class='team-name team-clickable' style='cursor:pointer;' data-roster-id='{roster_id}' data-team-name='{team_name}'>{team_name}</div></header>"
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
            roster_id = d.get("roster_id", "")
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
                f"    <header>{img}<div class='team-name team-clickable' style='cursor:pointer;' data-roster-id='{roster_id}' data-team-name='{team_name}'>{team_name}</div></header>"
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
        if platform == "espn":
            _empty_title = "Activity not available for ESPN leagues"
            _empty_copy = "Transaction history requires Sleeper league data. ESPN leagues currently show scores and standings only."
        else:
            _empty_title = "No recent activity yet"
            _empty_copy = "When trades and waiver claims come through, they’ll show up here with value context and team-by-team breakdowns."
        activity_html = (
            "<div class=’card’>"
            "  <div class=’card-body’>"
            "    <div class=’bract-empty-state’>"
            f"      <div class=’bract-empty-title’>{_empty_title}</div>"
            f"      <div class=’bract-empty-copy’>{_empty_copy}</div>"
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
        border: 1px solid var(--border);
        background: var(--card-soft);
        border-radius: 12px;
        padding: 12px 14px;
      }}

      .bract-summary-label {{
        font-size: 11px;
        line-height: 1.2;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 6px;
        font-weight: 700;
      }}

      .bract-summary-value {{
        font-size: 24px;
        line-height: 1.1;
        font-weight: 800;
        color: var(--text);
      }}

      .bract-summary-text {{
        font-size: 16px;
        line-height: 1.3;
      }}

      .bract-spotlight {{
        border: 1px solid var(--border);
        background: var(--accent-soft);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 14px;
      }}

      .bract-spotlight-title {{
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 4px;
      }}

      .bract-spotlight-copy {{
        font-size: 14px;
        color: var(--text);
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
        color: var(--text);
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


def _weighted_pos_strength(vals: List[float], pos: str, slot_counts: Dict[str, int]) -> float:
    """
    Emphasize top-end talent over pure depth.

    Examples:
      - QB: mostly QB1, tiny credit for QB2
      - RB/WR: strong weight on top 2, smaller weight on next few
      - TE: mostly TE1, tiny credit for TE2

    This prevents 5 mid players from outscoring 2 elite starters.
    """
    if not vals:
        return 0.0

    vals = sorted((float(v or 0.0) for v in vals), reverse=True)

    flex_slots = int(slot_counts.get("FLEX") or 0)

    if pos == "QB":
        weights = [1.0, 0.20]

    elif pos == "RB":
        # RB1/RB2 matter most, then some flex/depth credit
        if flex_slots >= 2:
            weights = [1.0, 0.85, 0.35, 0.20, 0.10]
        elif flex_slots == 1:
            weights = [1.0, 0.85, 0.30, 0.15]
        else:
            weights = [1.0, 0.85, 0.15]

    elif pos == "WR":
        # Same idea as RB
        if flex_slots >= 2:
            weights = [1.0, 0.85, 0.35, 0.20, 0.10]
        elif flex_slots == 1:
            weights = [1.0, 0.85, 0.30, 0.15]
        else:
            weights = [1.0, 0.85, 0.15]

    elif pos == "TE":
        # TE premium on starter, little on TE2 unless you want more
        if flex_slots >= 1:
            weights = [1.0, 0.20, 0.08]
        else:
            weights = [1.0, 0.15]

    else:
        weights = [1.0]

    used = vals[:len(weights)]
    denom = sum(weights[:len(used)]) or 1.0
    return sum(v * w for v, w in zip(used, weights)) / denom


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

    name_to_rank_label: Dict[str, str] = {}
    name_to_age: Dict[str, Union[float, None]] = {}

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
    by_id: Dict[str, Dict] = {
        str(p["id"]): p
        for p in model_vals
        if isinstance(p, dict) and p.get("id") is not None
    }

    CORE_POS = {"QB", "RB", "WR", "TE"}
    POS_ORDER = ["QB", "RB", "WR", "TE"]

    # ----------------- Roster → position → players (for dropdowns) -----------------
    roster_pos_players: Dict[int, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

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
    team_meta: Dict[int, Dict] = {}  # name, avatar
    team_pos_values: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

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

    # ----------------- Compute per-team positional strength + league baselines -----------------
    team_pos_strength: Dict[int, Dict[str, float]] = defaultdict(dict)
    slot_counts = count_roster_positions(get_roster_positions())

    for rid, pos_map in team_pos_values.items():
        for pos, vals in pos_map.items():
            team_pos_strength[rid][pos] = _weighted_pos_strength(vals, pos, slot_counts)

    league_pos_avg: Dict[str, float] = {}
    league_pos_std: Dict[str, float] = {}

    for pos in POS_ORDER:
        series = [team_pos_strength[rid][pos] for rid in team_meta.keys()]
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
    team_pos_z: Dict[int, Dict[str, float]] = defaultdict(dict)
    team_pos_index: Dict[int, float] = {}

    LINEUP_WEIGHTS = {
        "QB": slot_counts.get("QB") or 1,
        "RB": slot_counts.get("RB") or 2,
        "WR": slot_counts.get("WR") or 2,
        "TE": slot_counts.get("TE") or 1,
        "FLEX": slot_counts.get("FLEX") or 1,
    }
    weight_sum = sum(LINEUP_WEIGHTS[pos] for pos in POS_ORDER if LINEUP_WEIGHTS.get(pos, 0) > 0) or 1.0

    pos_z_min: Dict[str, float] = {pos: float("inf") for pos in POS_ORDER}
    pos_z_max: Dict[str, float] = {pos: float("-inf") for pos in POS_ORDER}

    for rid in team_meta.keys():
        idx_num = 0.0

        for pos in POS_ORDER:
            team_strength = team_pos_strength[rid][pos]
            mu = league_pos_avg[pos]
            sigma = league_pos_std[pos]
            if sigma > 0:
                z = (team_strength - mu) / sigma
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
    pos_rank: Dict[str, Dict[int, int]] = {pos: {} for pos in POS_ORDER}

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
            age_txt = f"{age:.1f} yrs" if age is not None else ""

            try:
                val = float(p.get("value") or 0.0)
            except Exception:
                val = 0.0
            val_txt = f"{val:.1f}" if val > 0 else ""

            # Build meta parts (rank, team, age)
            meta_parts = [rank_label, p.get('team', '')]
            if age_txt:
                meta_parts.append(age_txt)
            meta_str = " • ".join(filter(None, meta_parts))

            player_id = p.get("id", "")
            position = p.get('position', '')
            years_exp = p.get('years_exp')
            rows_html.append(
                "<div class='player-activity'>"
                "  <div style='display:flex;align-items:center;justify-content:space-between;width:100%'>"
                "    <div style='display: inline-flex;gap: 5px;align-items: center;'>"
                f"      <div style='font-weight:600;cursor:pointer;' class='player-clickable' data-player-id='{player_id}' data-player-name='{name}' data-position='{position}' data-years-exp='{years_exp}' data-value='{val}' data-breakout-check='true'>{name}</div>"
                f"      <div style='color:#64748b;font-size:12px'>"
                f"        {meta_str}"
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
            strength_score = team_pos_strength[rid][pos]
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
                "  <td class='pos-avg'>{strength_score:.1f}</td>"
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
                    z=z,
                    pct=pct,
                    strength_score=strength_score,
                )
            )

            # detail row right under it (collapsed by default)
            detail_html = render_pos_players(rid, pos)
            detail_row = (
                f"<tr class='pos-detail-row' data-pos='{pos}' style='display:none;'>"
                "  <td colspan='7'>"
                "    <div class='pos-detail-inner'>"
                f"      {detail_html}"
                "    </div>"
                "  </td>"
                "</tr>"
            )

            table_rows.append(main_row)
            table_rows.append(detail_row)

        card_html = (
            "<div class='card team-strength-card'>"
            "  <div class='card-header-row'>"
            f"    <div style='display:flex;align-items:center;gap:8px;'>{img_html}<h2 class='team-clickable' style='cursor:pointer;' data-roster-id='{rid}' data-team-name='{name}'>{name}</h2></div>"
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
            "          <th>Starter Score</th>"
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

            <div class="static-section">
              <div class="static-section-title">Advertising</div>
              <p>
                This site displays advertisements through Google AdSense. Google uses cookies
                to serve ads based on your prior visits to this site or other websites.
                Google's use of advertising cookies enables it and its partners to serve ads
                based on your visit to this site and/or other sites on the Internet.
              </p>
              <p style="margin-top:8px;">
                You may opt out of personalized advertising by visiting
                <a href="https://www.google.com/settings/ads" target="_blank" rel="noopener">
                  Google's Ads Settings
                </a> or
                <a href="http://www.aboutads.info/choices/" target="_blank" rel="noopener">
                  www.aboutads.info
                </a>.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Cookies</div>
              <p>
                We use cookies to maintain your login session and improve your experience.
                Third-party vendors, including Google, also use cookies to serve ads based
                on your browsing activity. By using this site, you consent to the use of
                cookies as described in this policy.
              </p>
            </div>

            <div class="static-section">
              <div class="static-section-title">Third-Party Links</div>
              <p>
                Our site may contain links to external websites. We are not responsible
                for the privacy practices or content of these third-party sites.
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
    # Update viewer session with fresh data even for cached contexts
    ctx = entry["ctx"]
    ctx["viewer"] = get_viewer_session()
    return ctx


@app.route("/api/refresh-league", methods=["POST"])
def api_refresh_league():
    """Force-expire a league context so the next request rebuilds it from source."""
    payload = request.get_json(silent=True) or {}
    platform = (payload.get("platform") or "sleeper").strip().lower()
    league_id = (payload.get("league_id") or "").strip()
    season = int(payload.get("season") or datetime.now().year)
    if not league_id:
        return jsonify({"error": "league_id required"}), 400
    key = _cache_key(platform, season, league_id)
    if key in DASHBOARD_CACHE:
        DASHBOARD_CACHE[key]["ts"] = 0  # expire immediately
    return jsonify({"ok": True})


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
    draft_ended = has_draft_ended(league_id, platform, season)

    if not draft_ended:
        body = """
        <div class="card central">
          <div class="card-header"><h2>Weekly Hub Unavailable</h2></div>
          <div class="card-body">
            <p>The Weekly Hub becomes active once your league draft has completed.</p>
            <p>Use the dashboard, teams, activity, and trade tools for pre-draft planning.</p>
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
        league_id_safe = ctx.get("league_id") or league_id
        season_safe = int(ctx.get("season") or season or datetime.now().year)
        num_teams = ctx.get("total_rosters") or None
        rec = float((ctx.get("scoring_settings") or {}).get("rec") or 0)
        scoring_format = "ppr" if rec >= 1.0 else "half" if rec >= 0.5 else "std"
        body = build_trade_calculator_body(league_id_safe, season_safe, num_teams=num_teams,
                                           scoring_format=scoring_format)
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


@app.route("/<platform>/<int:season>/<league_id>/players")
def page_players(platform: str, season: int, league_id: str):
    """Player Rankings page — searchable, filterable, sortable list of all players."""
    body_html = """
    <div class="card central">
      <div class="card-header">
        <h2>Player Rankings</h2>
        <div style="font-size:14px;color:var(--text-muted);margin-top:4px;">
          All players ranked by dynasty value — search, filter, and sort to explore
        </div>
      </div>
      <div class="card-body" style="padding-top:0;">

        <!-- Controls -->
        <div class="filter-controls-container">
          <!-- Row 1: Primary filters -->
          <div class="filter-row filter-row-primary">
            <!-- Search -->
            <div class="filter-search">
              <input id="prSearch" type="text" placeholder="Search players…" autocomplete="off"
                style="width:100%;padding:8px 32px 8px 34px;border-radius:8px;
                       border:1px solid var(--border);background:var(--card-bg);
                       color:var(--text);font-size:13px;outline:none;box-sizing:border-box;">
              <span style="position:absolute;left:10px;top:50%;transform:translateY(-50%);
                           color:var(--text-muted);font-size:14px;pointer-events:none;">🔍</span>
              <button id="prSearchClear" onclick="prClearSearch()"
                style="display:none;position:absolute;right:8px;top:50%;transform:translateY(-50%);
                       background:none;border:none;cursor:pointer;color:var(--text-muted);
                       font-size:16px;line-height:1;padding:2px;">&#x2715;</button>
            </div>

            <!-- Position filters -->
            <div class="filter-positions">
              <button class="pos-pill active" data-pos="ALL" onclick="prTogglePos('ALL')">All</button>
              <button class="pos-pill" data-pos="QB" onclick="prTogglePos('QB')">QB</button>
              <button class="pos-pill" data-pos="RB" onclick="prTogglePos('RB')">RB</button>
              <button class="pos-pill" data-pos="WR" onclick="prTogglePos('WR')">WR</button>
              <button class="pos-pill" data-pos="TE" onclick="prTogglePos('TE')">TE</button>
              <button class="pos-pill" data-pos="PICK" onclick="prTogglePos('PICK')">Picks</button>
            </div>

            <!-- Settings button -->
            <div style="position:relative;">
              <button id="prSettingsBtn" class="filter-settings-btn" onclick="prToggleSettings()">
                ⚙️ Settings
              </button>

              <!-- Settings panel (hidden by default) -->
              <div id="prSettingsPanel" class="filter-settings-panel" style="display:none;">
                <div class="settings-section">
                  <span class="settings-section-label">League Format</span>
                  <div class="settings-toggle-group">
                    <button class="settings-toggle active" data-value="1qb" onclick="prSetLeagueType('1qb')">1QB</button>
                    <button class="settings-toggle" data-value="sf" onclick="prSetLeagueType('sf')">SF</button>
                  </div>
                </div>
                <div class="settings-section">
                  <span class="settings-section-label">League Size</span>
                  <div class="settings-toggle-group">
                    <button class="settings-toggle" data-value="8" onclick="prSetSize(8)">8</button>
                    <button class="settings-toggle active" data-value="10" onclick="prSetSize(10)">10</button>
                    <button class="settings-toggle" data-value="12" onclick="prSetSize(12)">12</button>
                    <button class="settings-toggle" data-value="14" onclick="prSetSize(14)">14</button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Row 2: Secondary filters -->
          <div class="filter-row filter-row-secondary">
            <!-- Sort dropdown -->
            <div class="filter-sort">
              <label class="filter-label">Sort by</label>
              <select id="prSort" onchange="prRender()"
                style="padding:7px 10px;border-radius:8px;border:1px solid var(--border);
                       background:var(--card-bg);color:var(--text);font-size:12px;cursor:pointer;outline:none;min-height:34px;">
                <option value="rank">Rank</option>
                <option value="value">Value</option>
                <option value="age">Age (youngest)</option>
                <option value="pos_rank">Pos Rank</option>
              </select>
            </div>

            <!-- Active settings indicator -->
            <div id="prActiveSettings" class="active-settings-indicator">
              <span class="active-setting-tag">10-Team</span>
              <span class="active-setting-tag">1QB</span>
            </div>
          </div>
        </div>

        <!-- Loading -->
        <div id="prLoading" style="text-align:center;padding:40px;color:var(--text-muted);">
          <div class="loading-spinner" style="margin:0 auto 12px;"></div>
          Loading players…
        </div>

        <!-- Player count -->
        <div id="prCount" style="font-size:12px;color:var(--text-muted);margin-bottom:8px;display:none;"></div>

        <!-- Table header -->
        <div id="prTableHeader" style="display:none;
             grid-template-columns:44px 1fr 64px 50px 50px 64px;
             gap:0;padding:6px 12px;border-radius:6px;
             background:var(--accent-soft);font-size:11px;
             font-weight:700;color:var(--accent);letter-spacing:0.04em;
             text-transform:uppercase;" class="pr-grid-row">
          <span>#</span>
          <span>Player</span>
          <span style="text-align:center;">Pos</span>
          <span style="text-align:center;">Age</span>
          <span style="text-align:right;">Team</span>
          <span style="text-align:right;">Value</span>
        </div>

        <!-- Player rows -->
        <div id="prList"></div>

        <!-- Empty state -->
        <div id="prEmpty" style="display:none;text-align:center;padding:40px;color:var(--text-muted);">
          <div style="font-size:24px;margin-bottom:8px;">🔍</div>
          No players match your filters
        </div>

      </div>
    </div>

    <style>
      .pr-grid-row {
        display: grid;
        grid-template-columns: 44px 1fr 64px 50px 50px 64px;
        align-items: center;
        gap: 0;
      }
      .pr-player-row {
        padding: 9px 12px;
        cursor: pointer;
        transition: background 0.12s ease;
      }
      .pr-player-row:hover { background: var(--accent-soft); }
      .pr-player-row + .pr-player-row { border-top: 1px solid var(--border); }
      .pr-rank {
        font-size: 12px;
        font-weight: 700;
        color: var(--text-muted);
      }
      .pr-name {
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 5px;
        flex-wrap: wrap;
        min-width: 0;
      }
      .pr-pos-cell {
        text-align: center;
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
      }
      .pr-age {
        text-align: center;
        font-size: 12px;
        color: var(--text-muted);
      }
      .pr-team {
        text-align: right;
        font-size: 11px;
        color: var(--text-muted);
      }
      .pr-value {
        text-align: right;
        font-size: 13px;
        font-weight: 700;
        color: var(--accent);
      }
      /* Filter Controls */
      .filter-controls-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 16px 0 14px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 12px;
      }
      .filter-row {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
      }
      .filter-row-primary {
        gap: 12px;
      }
      .filter-row-secondary {
        padding-top: 4px;
      }
      .filter-search {
        position: relative;
        flex: 1;
        min-width: 200px;
        max-width: 400px;
      }
      .filter-positions {
        display: flex;
        gap: 3px;
        flex-wrap: wrap;
      }
      .pos-pill {
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--card-bg);
        color: var(--text-muted);
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.12s;
        white-space: nowrap;
      }
      .pos-pill.active {
        background: var(--accent);
        color: #fff;
        border-color: var(--accent);
      }
      .filter-settings-btn {
        padding: 7px 14px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--card-bg);
        color: var(--text);
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
        transition: all 0.12s;
      }
      .filter-settings-btn:hover {
        background: var(--accent-soft);
        border-color: var(--accent);
        color: var(--accent);
      }
      .filter-settings-panel {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 8px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        padding: 16px;
        min-width: 260px;
        z-index: 1000;
      }
      .settings-section {
        margin-bottom: 16px;
      }
      .settings-section:last-of-type {
        margin-bottom: 0;
      }
      .settings-section-label {
        display: block;
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
      }
      .settings-toggle-group {
        display: flex;
        gap: 6px;
      }
      .settings-toggle {
        flex: 1;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--card-bg);
        color: var(--text-muted);
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.12s;
      }
      .settings-toggle.active {
        background: var(--accent);
        color: #fff;
        border-color: var(--accent);
      }
      .active-settings-indicator {
        display: flex;
        gap: 6px;
        align-items: center;
        flex-wrap: wrap;
      }
      .active-setting-tag {
        padding: 4px 10px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 11px;
        font-weight: 600;
      }
      .filter-sort {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .filter-label {
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }

      /* Mobile responsive */
      @media (max-width: 768px) {
        .filter-row-primary {
          flex-direction: column;
          align-items: stretch;
        }
        .filter-search {
          max-width: 100%;
        }
        .filter-positions {
          justify-content: center;
        }
        .active-settings-indicator {
          justify-content: center;
          order: -1;
          width: 100%;
        }
      }
    </style>

    <script>
      var prAllPlayers = [];
      var prIndicators = {};
      var prLeagueType = '1qb';
      var prLeagueSize = 10;
      var prPosFilters = new Set();   // empty = All
      var prSearchQuery = '';
      var prLoaded = false;

      // ---- Fuzzy search (mirrors trade calc logic) ----
      function prFuzzyScore(name, query) {
        if (!name || !query) return 0;
        const n = name.toLowerCase(), q = query.toLowerCase();
        if (n.includes(q)) return 100 + (100 - n.indexOf(q));
        const nw = n.split(/[\\s\\-]+/), qw = q.split(/\\s+/).filter(Boolean);
        if (qw.length > 1) {
          if (qw.every((qx, i) => nw.slice(i).some(w => w.startsWith(qx)))) return 70;
        }
        if (nw.some(w => w.startsWith(q))) return 60;
        if (q.length >= 4) {
          for (let i = 0; i < q.length; i++) {
            const del = q.slice(0, i) + q.slice(i + 1);
            if (n.includes(del)) return 40;
            for (const c of 'abcdefghijklmnopqrstuvwxyz') {
              const sub = q.slice(0, i) + c + q.slice(i + 1);
              if (n.includes(sub) && sub !== q) return 30;
            }
          }
        }
        return 0;
      }

      function prGetValue(p) {
        let base;
        if (prLeagueType === 'sf') {
          const key = prLeagueSize === 10 ? 'sf_value' : 'sf_value_' + prLeagueSize;
          base = Number(p[key] ?? p.sf_value ?? p.value ?? 0);
        } else {
          const key = prLeagueSize === 10 ? 'value' : 'value_' + prLeagueSize;
          base = Number(p[key] ?? p.value ?? 0);
        }
        return Math.round(base * 10) / 10;
      }

      function prFormatValue(v) {
        if (!v || v <= 0) return '-';
        return v.toFixed(1);
      }

      function prIsRookie(id) {
        return prIndicators.rookies && prIndicators.rookies.includes(String(id));
      }

      function prIsBreakout(id) {
        return prIndicators.breakouts && prIndicators.breakouts.includes(String(id));
      }

      // Settings panel toggle
      function prToggleSettings() {
        const panel = document.getElementById('prSettingsPanel');
        const btn = document.getElementById('prSettingsBtn');
        if (!panel || !btn) return;

        const isOpen = panel.style.display === 'block';
        panel.style.display = isOpen ? 'none' : 'block';
        btn.classList.toggle('active', !isOpen);
      }

      // Update active settings indicator tags
      function updateSettingsIndicator() {
        const indicator = document.getElementById('prActiveSettings');
        if (!indicator) return;

        const sizeTag = indicator.querySelector('.active-setting-tag:first-child');
        const formatTag = indicator.querySelector('.active-setting-tag:last-child');

        if (sizeTag) sizeTag.textContent = prLeagueSize + '-Team';
        if (formatTag) formatTag.textContent = prLeagueType.toUpperCase();
      }

      function prSetLeagueType(type) {
        prLeagueType = type;

        // Update settings panel toggles
        document.querySelectorAll('#prSettingsPanel .settings-toggle[data-value]').forEach(btn => {
          const section = btn.closest('.settings-section');
          if (section && section.querySelector('.settings-section-label').textContent.includes('Format')) {
            btn.classList.toggle('active', btn.getAttribute('data-value') === type);
          }
        });

        updateSettingsIndicator();
        prRender();
      }

      function prSetSize(size) {
        prLeagueSize = size;

        // Update settings panel toggles
        document.querySelectorAll('#prSettingsPanel .settings-toggle[data-value]').forEach(btn => {
          const section = btn.closest('.settings-section');
          if (section && section.querySelector('.settings-section-label').textContent.includes('Size')) {
            const btnSize = parseInt(btn.getAttribute('data-value'));
            btn.classList.toggle('active', btnSize === size);
          }
        });

        updateSettingsIndicator();
        prRender();
      }

      // Multi-select position toggle
      function prTogglePos(pos) {
        if (pos === 'ALL') {
          prPosFilters.clear();
        } else {
          if (prPosFilters.has(pos)) {
            prPosFilters.delete(pos);
          } else {
            prPosFilters.add(pos);
          }
        }
        // Sync button states
        document.querySelectorAll('.pos-pill').forEach(b => {
          const p = b.getAttribute('data-pos');
          if (p === 'ALL') {
            b.classList.toggle('active', prPosFilters.size === 0);
          } else {
            b.classList.toggle('active', prPosFilters.has(p));
          }
        });
        prRender();
      }

      function prClearSearch() {
        document.getElementById('prSearch').value = '';
        prSearchQuery = '';
        document.getElementById('prSearchClear').style.display = 'none';
        prRender();
      }

      // Build overall rank map keyed by player id (ranked by current value)
      function prBuildRankMap() {
        const ranked = prAllPlayers
          .filter(p => p.position !== 'PICK')
          .slice()
          .sort((a, b) => prGetValue(b) - prGetValue(a));
        return new Map(ranked.map((p, i) => [String(p.id), i + 1]));
      }

      // Sort and filter players, then render rows into the main table
      function prRender() {
        if (!prLoaded) return;
        const sortBy = document.getElementById('prSort').value;

        let players = prAllPlayers.slice();

        // Position filter (multi-select)
        if (prPosFilters.size > 0) {
          players = players.filter(p => prPosFilters.has(p.position));
        }

        // Search filter — fuzzy match, sort by score when query present
        if (prSearchQuery.length > 0) {
          const scored = players
            .map(p => ({
              p,
              score: Math.max(prFuzzyScore(p.name, prSearchQuery), prFuzzyScore(p.search_name, prSearchQuery))
            }))
            .filter(x => x.score > 0)
            .sort((a, b) => b.score !== a.score ? b.score - a.score : prGetValue(b.p) - prGetValue(a.p));
          players = scored.map(x => x.p);
        } else {
          // Normal sort when no search query
          players.sort((a, b) => {
            if (sortBy === 'value') {
              return prGetValue(b) - prGetValue(a);
            } else if (sortBy === 'age') {
              return (a.age != null ? a.age : 99) - (b.age != null ? b.age : 99);
            } else if (sortBy === 'pos_rank') {
              const rA = prLeagueType === 'sf' ? (a.sf_pos_rank || a.pos_rank || 9999) : (a.pos_rank || 9999);
              const rB = prLeagueType === 'sf' ? (b.sf_pos_rank || b.pos_rank || 9999) : (b.pos_rank || 9999);
              return rA - rB;
            } else {
              return prGetValue(b) - prGetValue(a);
            }
          });
        }

        const list   = document.getElementById('prList');
        const empty  = document.getElementById('prEmpty');
        const count  = document.getElementById('prCount');
        const header = document.getElementById('prTableHeader');

        if (players.length === 0) {
          list.innerHTML = '';
          empty.style.display = 'block';
          header.style.display = 'none';
          count.style.display = 'none';
          return;
        }

        empty.style.display = 'none';
        header.style.display = 'grid';
        count.style.display = 'block';
        count.textContent = players.length + ' player' + (players.length !== 1 ? 's' : '');

        const rankMap = prBuildRankMap();

        list.innerHTML = '';
        players.forEach((p, idx) => {
          const row = document.createElement('div');
          row.className = 'pr-player-row pr-grid-row player-clickable';
          row.setAttribute('data-player-id', p.id);
          row.setAttribute('data-player-name', p.name || '');

          const displayRank = p.position === 'PICK' ? '' : (rankMap.get(String(p.id)) || (idx + 1));
          const posRank = prLeagueType === 'sf'
            ? (p.sf_pos_rank_label || p.pos_rank_label || p.position)
            : (p.pos_rank_label || p.position);
          const age = p.age != null ? Number(p.age).toFixed(1) : '—';
          const val = prGetValue(p);

          let badges = '';
          if (prIsRookie(p.id))   badges += '<span class="player-badge player-badge-rookie">ROOKIE</span>';
          if (prIsBreakout(p.id)) badges += '<span class="player-badge player-badge-breakout">🔥 BREAKOUT</span>';

          row.innerHTML =
            '<span class="pr-rank">'  + (displayRank ? '#' + displayRank : '—') + '</span>' +
            '<span class="pr-name">'  + (p.name || 'Unknown') + badges + '</span>' +
            '<span class="pr-pos-cell">' + posRank + '</span>' +
            '<span class="pr-age">'   + (p.position === 'PICK' ? '—' : age) + '</span>' +
            '<span class="pr-team">'  + (p.team || '—') + '</span>' +
            '<span class="pr-value">' + prFormatValue(val) + '</span>';

          list.appendChild(row);
        });

        if (typeof initGlobalPlayerModals === 'function') initGlobalPlayerModals();
      }

      // Wire up search input
      (function() {
        const inp   = document.getElementById('prSearch');
        const clear = document.getElementById('prSearchClear');
        if (!inp) return;

        inp.addEventListener('input', function() {
          prSearchQuery = inp.value.trim();
          clear.style.display = prSearchQuery.length > 0 ? 'block' : 'none';
          prRender();
        });
      })();

      // Close settings panel when clicking outside
      document.addEventListener('click', function(e) {
        const panel = document.getElementById('prSettingsPanel');
        const btn = document.getElementById('prSettingsBtn');

        if (panel && btn && panel.style.display === 'block') {
          if (!panel.contains(e.target) && !btn.contains(e.target)) {
            panel.style.display = 'none';
            btn.classList.remove('active');
          }
        }
      });

      // Load data
      Promise.all([
        fetch('/api/league-players', { cache: 'no-store' }).then(r => r.json()),
        fetch('/api/player-indicators?league_type=1qb&league_size=10', { cache: 'no-store' })
          .then(r => r.json()).catch(() => ({}))
      ]).then(([players, indicators]) => {
        prIndicators = indicators || {};
        const rawPlayers = Array.isArray(players) ? players : [];

        prAllPlayers = rawPlayers
          .filter(p => p && p.id != null)
          .map(p => ({
            id:               String(p.id),
            name:             p.name || p.full_name || 'Unknown',
            team:             p.team || '',
            position:         String(p.position || '').toUpperCase(),
            age:              p.age != null ? Number(p.age) : null,
            value:            Number(p.value    || 0),
            value_8:          Number(p.value_8  || p.value    || 0),
            value_12:         Number(p.value_12 || p.value    || 0),
            value_14:         Number(p.value_14 || p.value    || 0),
            sf_value:         Number(p.sf_value    || p.value || 0),
            sf_value_8:       Number(p.sf_value_8  || p.sf_value || p.value || 0),
            sf_value_12:      Number(p.sf_value_12 || p.sf_value || p.value || 0),
            sf_value_14:      Number(p.sf_value_14 || p.sf_value || p.value || 0),
            pos_rank_label:   p.pos_rank_label    || '',
            sf_pos_rank_label:p.sf_pos_rank_label || '',
            pos_rank:         Number(p.pos_rank    || 9999),
            sf_pos_rank:      Number(p.sf_pos_rank || 9999),
            search_name:      p.search_name || '',
          }))
          .filter(p => ['QB','RB','WR','TE','PICK'].includes(p.position))
          .sort((a, b) => Number(b.value || 0) - Number(a.value || 0));

        document.getElementById('prLoading').style.display = 'none';
        prLoaded = true;
        prRender();
      }).catch(err => {
        console.error('Error loading player rankings:', err);
        document.getElementById('prLoading').innerHTML =
          '<div style="color:#ef4444;">Failed to load players. Please refresh.</div>';
      });
    </script>
    """
    return render_page("Player Rankings", league_id, "players", body_html, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/rookies")
def page_rookies(platform: str, season: int, league_id: str):
    """Rookie prospect rankings page — active class auto-detected."""
    from dashboard_services.pages.rookies_page import build_rookies_body
    body_html = build_rookies_body(platform, season, league_id)
    return render_page("Rookie Rankings", league_id, "rookies", body_html, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/breakouts")
def page_breakouts(platform: str, season: int, league_id: str):
    """Dedicated page for breakout candidates with detailed projections."""
    body_html = f"""
    <div class="card central">
      <div class="card-header">
        <h2>Breakout Candidates</h2>
        <div style="font-size: 14px; color: var(--text-muted); margin-top: 4px;">
          Players positioned for breakouts based on opportunity, efficiency, and roster changes
        </div>
      </div>
      <div class="card-body">
        <!-- Position Filter -->
        <div style="display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap;">
          <button class="breakout-filter-btn active" data-position="ALL" onclick="filterBreakouts('ALL')">All Positions</button>
          <button class="breakout-filter-btn" data-position="QB" onclick="filterBreakouts('QB')">QB</button>
          <button class="breakout-filter-btn" data-position="RB" onclick="filterBreakouts('RB')">RB</button>
          <button class="breakout-filter-btn" data-position="WR" onclick="filterBreakouts('WR')">WR</button>
          <button class="breakout-filter-btn" data-position="TE" onclick="filterBreakouts('TE')">TE</button>
        </div>

        <!-- Loading State -->
        <div id="breakoutsLoading" class="player-modal-loading" style="padding: 40px;">
          <div class="loading-spinner"></div>
          <div style="margin-top: 12px;">Loading breakout candidates...</div>
        </div>

        <!-- Breakouts Container -->
        <div id="breakoutsContainer" style="display: none;"></div>

        <!-- Empty State -->
        <div id="breakoutsEmpty" style="display: none; text-align: center; padding: 40px; color: var(--text-muted);">
          <div style="font-size: 24px; margin-bottom: 12px;">📊</div>
          <div>No breakout candidates found</div>
        </div>
      </div>
    </div>

    <script>
      let breakoutCandidates = [];
      let currentFilter = 'ALL';

      // Fetch breakout candidates on page load (using new BreakoutEngine API)
      fetch('/api/breakout/candidates?season={season}&min_score=25')
        .then(res => res.json())
        .then(data => {{
          breakoutCandidates = (data && data.candidates) || [];
          document.getElementById('breakoutsLoading').style.display = 'none';

          if (breakoutCandidates.length === 0) {{
            document.getElementById('breakoutsEmpty').style.display = 'block';
          }} else {{
            renderBreakouts();
          }}
        }})
        .catch(err => {{
          console.error('Error loading breakouts:', err);
          document.getElementById('breakoutsLoading').innerHTML = '<div style="color: #ef4444;">Failed to load breakout candidates</div>';
        }});

      function filterBreakouts(position) {{
        currentFilter = position;

        // Update active button
        document.querySelectorAll('.breakout-filter-btn').forEach(btn => {{
          btn.classList.toggle('active', btn.getAttribute('data-position') === position);
        }});

        renderBreakouts();
      }}

      function renderBreakouts() {{
        const container = document.getElementById('breakoutsContainer');
        const filtered = currentFilter === 'ALL'
          ? breakoutCandidates
          : breakoutCandidates.filter(c => c.position === currentFilter);

        if (filtered.length === 0) {{
          document.getElementById('breakoutsEmpty').style.display = 'block';
          container.style.display = 'none';
          return;
        }}

        document.getElementById('breakoutsEmpty').style.display = 'none';
        container.style.display = 'block';

        let html = '<div class="breakout-grid">';

        filtered.forEach(candidate => {{
          const name = candidate.player_name || 'Unknown';
          const team = candidate.team || '?';
          const pos = candidate.position || '?';
          const age = candidate.age ? parseFloat(candidate.age).toFixed(1) : '-';
          const score = candidate.breakout_opportunity_score ? parseFloat(candidate.breakout_opportunity_score).toFixed(1) : '0';
          const pid = candidate.player_id || '';

          // Breakout type classification
          const breakoutType = candidate.breakout_type || {{}};
          const emoji = breakoutType.emoji || '📊';
          const label = breakoutType.profile_label || 'Breakout Candidate';
          const driver = breakoutType.primary_driver || 'balanced';

          // Component scores
          const oppScore = parseFloat(candidate.opportunity_opened_score || 0).toFixed(1);
          const readyScore = parseFloat(candidate.player_readiness_score || 0).toFixed(1);
          const confScore = parseFloat(candidate.confidence_score || 0).toFixed(1);
          const teamEnv = parseFloat(candidate.team_environment_score || 0).toFixed(1);

          // Key reasons (from engine)
          const reasons = candidate.key_reasons || '';
          const reasonsList = reasons.split('\\n').filter(r => r.trim() && r.startsWith('•')).map(r => r.substring(1).trim());

          // Score badge color based on score ranges
          let scoreColor = '#10b981'; // green (50+)
          if (score < 50) scoreColor = '#3b82f6'; // blue (40-49)
          if (score < 40) scoreColor = '#f59e0b'; // amber (30-39)
          if (score < 30) scoreColor = '#6b7280'; // gray (<30)

          html += `
            <div class="breakout-card">
              <div class="breakout-card-header">
                <div>
                  <div class="breakout-player-name player-clickable" data-player-id='` + pid + `' data-player-name='` + name + `'>` + name + `</div>
                  <div class="breakout-player-meta">${{age}} • ${{team}} • ${{pos}}</div>
                </div>
                <div class="breakout-score-badge" style="background: ${{scoreColor}};">
                  ${{score}}
                </div>
              </div>

              <div class="breakout-card-body">
                <!-- Breakout Type Badge -->
                <div class="breakout-type-badge" style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: var(--card-bg); border-radius: 6px; margin-bottom: 12px; border: 1px solid var(--border-color);">
                  <span style="font-size: 20px;">${{emoji}}</span>
                  <span style="font-weight: 500; flex: 1;">${{label}}</span>
                  <span style="font-size: 12px; color: var(--text-muted); text-transform: uppercase;">${{driver}} driven</span>
                </div>

                <!-- Key Reasons -->
                ${{reasonsList.length > 0 ? `
                  <div class="breakout-section">
                    <div class="breakout-section-title">Why This Breakout?</div>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13px; line-height: 1.6;">
                      ${{reasonsList.map(r => `<li>${{r}}</li>`).join('')}}
                    </ul>
                  </div>
                ` : ''}}

                <!-- Component Scores -->
                <div class="breakout-section">
                  <div class="breakout-section-title">Component Breakdown</div>
                  <div class="breakout-components" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 12px;">
                    ${{driver === 'opportunity' || driver === 'balanced' ? `
                      <div style="background: var(--card-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color);">
                        <div style="color: var(--text-muted); margin-bottom: 2px;">Opportunity</div>
                        <div style="font-weight: 600; font-size: 14px; color: #10b981;">${{oppScore}}</div>
                      </div>
                    ` : ''}}
                    ${{driver === 'readiness' || driver === 'balanced' ? `
                      <div style="background: var(--card-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color);">
                        <div style="color: var(--text-muted); margin-bottom: 2px;">Talent/Readiness</div>
                        <div style="font-weight: 600; font-size: 14px; color: #3b82f6;">${{readyScore}}</div>
                      </div>
                    ` : ''}}
                    <div style="background: var(--card-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color);">
                      <div style="color: var(--text-muted); margin-bottom: 2px;">Team Environment</div>
                      <div style="font-weight: 600; font-size: 14px;">${{teamEnv}}</div>
                    </div>
                    <div style="background: var(--card-bg); padding: 8px; border-radius: 4px; border: 1px solid var(--border-color);">
                      <div style="color: var(--text-muted); margin-bottom: 2px;">Confidence</div>
                      <div style="font-weight: 600; font-size: 14px;">${{confScore}}%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          `;
        }});

        html += '</div>';
        container.innerHTML = html;
      }}
      
      // Initialize global player modals for clickable player names
      initGlobalPlayerModals();
    </script>
    """
    return render_page("Breakout Candidates", league_id, "breakouts", body_html, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/teams")
def page_teams(platform: str, season: int, league_id: str):
    cached = get_page_html_from_cache(platform, season, league_id, "teams")
    if cached:
        return render_page("BR Fantasy Teams", league_id, "teams", cached, platform, season)

    ctx = get_league_ctx_from_cache(platform, league_id, season)
    body_html = build_teams_body(ctx)
    store_page_html(platform, season, league_id, "teams", body_html)
    return render_page("BR Fantasy Teams", league_id, "teams", body_html, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/history")
def page_history(platform: str, season: int, league_id: str):
    available_seasons = get_available_history_seasons(platform, league_id, season)

    # Handle first-year league case
    if not available_seasons:
        body_html = """
        <div class="card central">
          <div class="card-body">
            <div class="bract-empty-state">
              <div class="bract-empty-title">Welcome to Your First Season!</div>
              <div class="bract-empty-copy">This is the first year of your league. Historical season data, AI-powered recaps, and year-over-year comparisons will appear here after your current season completes. Check back after championship week!</div>
            </div>
          </div>
        </div>
        """
        return render_page(
            "League History",
            league_id,
            "history",
            body_html,
            platform,
            season,
        )

    default_history_season = get_default_history_season(available_seasons, season)

    selected_history_season = int(
        request.args.get("history_season") or default_history_season
    )

    if selected_history_season not in available_seasons:
        selected_history_season = default_history_season

    resolved_history_league_id = resolve_league_id_for_season(
        platform=platform,
        league_id=league_id,
        current_season=season,
        target_season=selected_history_season,
    )

    history_ctx = get_league_ctx_from_cache(
        platform,
        resolved_history_league_id,
        selected_history_season,
    )

    body_html = build_history_body(
        history_ctx=history_ctx,
        available_seasons=available_seasons,
        base_platform=platform,
        base_season=season,
        base_league_id=league_id,
        selected_history_season=selected_history_season,
        resolved_history_league_id=resolved_history_league_id,
    )

    if platform == "espn":
        espn_notice = (
            "<div class='card' style='margin-bottom:16px;'>"
            "  <div class='card-body'>"
            "    <div class='bract-empty-state'>"
            "      <div class='bract-empty-title'>Limited history for ESPN leagues</div>"
            "      <div class='bract-empty-copy'>Full season recaps and AI-powered history analysis are optimized for Sleeper leagues. "
            "Some data may be incomplete for ESPN.</div>"
            "    </div>"
            "  </div>"
            "</div>"
        )
        body_html = espn_notice + body_html

    return render_page(
        "League History",
        league_id,
        "history",
        body_html,
        platform,
        season,
    )


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


@app.route("/ads.txt")
def ads_txt():
    """Serve ads.txt file for ad network authorization"""
    try:
        ads_file = Path(__file__).resolve().parent / "ads.txt"
        if ads_file.exists():
            return send_file(ads_file, mimetype="text/plain")
        else:
            # Return a placeholder if file doesn't exist
            return "# ads.txt - Add your ad network credentials here", 200, {"Content-Type": "text/plain"}
    except Exception as e:
        print(f"[ads.txt] Error serving file: {e}")
        return "# ads.txt unavailable", 500, {"Content-Type": "text/plain"}


@app.route("/", methods=["GET", "POST"])
def index():
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    viewed_season = current_season

    if request.method == "POST":
        platform = (request.form.get("platform") or "sleeper").strip().lower()
        league_id = (request.form.get("league") or "").strip()
        season = int(request.form.get("season") or viewed_season)
        username = (request.form.get("username") or "").strip()

        ok, err = validate_league_id(platform, league_id)
        if not ok:
            body_html = render_template_string(
                FORM_BODY,
                username=username,
                viewed_season=viewed_season,
                error=err,
                recent_updates=generate_recent_updates_html(),
            )
            return render_page("BR Fantasy Dashboard", None, "home", body_html)

        # If username/team-name provided, set viewer session
        # For ESPN leagues the "username" field holds the team owner's name or team name
        if username:
            ctx = get_league_ctx_from_cache(platform, league_id, season)
            viewer = resolve_viewer_for_league(ctx["users"], ctx["rosters"], username)

            if viewer:
                save_viewer_session(viewer)
            else:
                # For ESPN, skip the hard error — viewer matching is optional
                if platform != "espn":
                    body_html = render_template_string(
                        FORM_BODY,
                        username=username,
                        viewed_season=viewed_season,
                        error="Could not match that username to a team in this league.",
                        recent_updates=generate_recent_updates_html(),
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
        recent_updates=generate_recent_updates_html(),
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


@app.route("/set-viewer", methods=["POST"])
def set_viewer():
    league_id = (request.form.get("league_id") or "").strip()
    username = (request.form.get("username") or "").strip()
    platform = (request.form.get("platform") or "sleeper").strip().lower()
    season = int(request.form.get("season") or datetime.now().year)

    if not league_id or not username:
        return redirect(url_for("home"))

    ctx = get_league_ctx_from_cache(platform=platform, league_id=league_id, season=season)
    viewer = resolve_viewer_for_league(ctx["users"], ctx["rosters"], username)

    if not viewer:
        return render_template_string(
            FORM_BODY,
            league=league_id,
            error="Could not match that username to a team in this league.",
            recent_updates=generate_recent_updates_html(),
        )

    save_viewer_session(viewer)
    return redirect(url_for("page_dashboard", league_id=league_id))


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
            league_id_safe = ctx.get("league_id") or league_id
            season_safe = int(ctx.get("season") or season or datetime.now().year)
            num_teams = ctx.get("total_rosters") or None
            rec = float((ctx.get("scoring_settings") or {}).get("rec") or 0)
            scoring_format = "ppr" if rec >= 1.0 else "half" if rec >= 0.5 else "std"
            body_html = build_trade_calculator_body(league_id_safe, season_safe, num_teams=num_teams,
                                                    scoring_format=scoring_format)

        else:
            body_html = ""

        return jsonify({
            "ok": True,
            "refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "current_week": ctx.get("current_week"),
            "offseason_mode": bool(ctx.get("offseason_mode")),
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


@app.route("/api/gm-memo", methods=["POST"])
def api_gm_memo():
    payload = request.get_json(force=True)

    league_id = str(payload.get("league_id") or "").strip()
    season = int(payload.get("season") or datetime.now().year)
    platform = str(payload.get("platform") or "sleeper").strip()
    viewer_roster_id = str(payload.get("viewer_roster_id") or "").strip()

    if not league_id or not season or not viewer_roster_id:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        gm_memo_html = get_team_gm_memo(ctx, viewer_roster_id)

        return jsonify({
            "success": True,
            "gm_memo_html": gm_memo_html
        })
    except Exception as e:
        print(f"[api-gm-memo] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/trade-eval", methods=["POST"])
def api_trade_eval():
    payload = request.get_json(force=True)

    league_id = str(payload.get("league_id") or "").strip()
    season = int(payload.get("season") or datetime.now().year)
    platform = str(payload.get("platform") or "sleeper").strip().lower()
    league_type = str(payload.get("league_type") or "1qb").strip().lower()
    scoring_format = str(payload.get("scoring_format") or "ppr").strip().lower()
    viewer_side = (payload.get("viewer_side") or "a").strip().lower()

    # Position-based multipliers for non-PPR formats.
    # RBs gain value in standard (rush-heavy); WRs/TEs lose value (fewer receptions).
    _SCORING_MULTS = {
        "ppr": {"QB": 1.00, "RB": 1.00, "WR": 1.00, "TE": 1.00},
        "half": {"QB": 1.00, "RB": 1.06, "WR": 0.97, "TE": 0.94},
        "std": {"QB": 1.00, "RB": 1.13, "WR": 0.93, "TE": 0.87},
    }
    scoring_mults = _SCORING_MULTS.get(scoring_format, _SCORING_MULTS["ppr"])

    side_a_players = [str(pid) for pid in payload.get("side_a_players", [])]
    side_b_players = [str(pid) for pid in payload.get("side_b_players", [])]
    side_a_picks = payload.get("side_a_picks", []) or []
    side_b_picks = payload.get("side_b_picks", []) or []

    value_table = get_model_value_table_cached()

    if not isinstance(value_table, list):
        raise ValueError("model_value_table must be a list of player objects")

    players_by_id = {
        str(p["id"]): p
        for p in value_table
        if isinstance(p, dict) and "id" in p
    }

    pick_values = load_pick_value_table()

    def value_pick(pk: str) -> float:
        try:
            yr_str, rnd_str, slot_str = pk.split("_")
            year = int(yr_str)
            rnd = int(rnd_str)
        except Exception:
            return 0.0

        key = f"{year}_{rnd}_{slot_str}"

        val = pick_values.get(key)
        if val is not None:
            return float(val)

        generic_key = f"any_{rnd}_{slot_str}"
        if generic_key in pick_values:
            return float(pick_values[generic_key])

        return 0.0

    def build_side(players_ids, picks_ids):
        raw_players_total = 0.0
        raw_picks_total = 0.0
        player_values = []
        breakdown = []
        assets = []

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
                assets.append({
                    "id": pid_str,
                    "name": f"Player {pid_str}",
                    "value": 0.0,
                    "position": None,
                    "team": None,
                    "age": None,
                })
                continue

            # Use sf_value for Superflex leagues, otherwise use regular value
            if league_type == "sf":
                val = float(player.get("sf_value", player.get("value", 0.0)) or 0.0)
            else:
                val = float(player.get("value", 0.0) or 0.0)

            name = player.get("name")
            pos = player.get("position")

            # Apply scoring format multiplier
            val = round(val * scoring_mults.get((pos or "").upper(), 1.0), 1)
            team = player.get("team")
            age = player.get("age")

            breakdown.append({
                "type": "player",
                "id": pid_str,
                "name": name,
                "value": val,
                "position": pos,
                "team": team,
            })
            assets.append({
                "id": pid_str,
                "name": name,
                "value": val,
                "position": pos,
                "team": team,
                "age": age,
            })

            raw_players_total += val
            player_values.append(val)

        for pk in picks_ids:
            pk_str = str(pk)
            val = float(value_pick(pk_str))
            breakdown.append({
                "type": "pick",
                "id": pk_str,
                "value": val,
            })
            raw_picks_total += val
            # Also add pick to assets array
            assets.append({
                "id": pk_str,
                "name": pk_str,  # Will be cleaned up in renderer
                "position": "PICK",
                "team": "",
                "age": None,
                "value": val,
            })

        raw_total = raw_players_total + raw_picks_total

        return {
            "raw_total": raw_total,
            "raw_players_total": raw_players_total,
            "raw_picks_total": raw_picks_total,
            "player_values": player_values,
            "breakdown": breakdown,
            "assets": assets,
            "pick_ids": [str(pk) for pk in picks_ids],
            "effective_total": raw_total,
            "adjustment": 0.0,
        }

    side_a = build_side(side_a_players, side_a_picks)
    side_b = build_side(side_b_players, side_b_picks)

    apply_multi_for_one_adjustment(side_a, side_b)

    a_eff = side_a["effective_total"]
    b_eff = side_b["effective_total"]

    diff = a_eff - b_eff
    abs_diff = abs(diff)

    # Fair band: tighter % for bigger trades (large trades need less slack),
    # floored at 25 value points so tiny trades aren't hair-trigger.
    baseline = max(a_eff, b_eff, 1.0)
    if baseline >= 600:
        FAIR_PCT = 0.05
    elif baseline >= 300:
        FAIR_PCT = 0.07
    else:
        FAIR_PCT = 0.10
    fair_band = max(baseline * FAIR_PCT, 25.0)

    if abs_diff <= fair_band:
        pct = (abs_diff / baseline) * 100.0
        verdict = f"This trade looks very fair (about {pct:.1f}% apart)."
    elif diff > 0:
        verdict = f"Team 1 is favored by about {abs_diff:.1f} value."
    else:
        verdict = f"Team 2 is favored by about {abs_diff:.1f} value."

    analysis_html = ""
    viewer_roster_id = payload.get("viewer_roster_id")
    viewer_team_name = payload.get("viewer_team_name")

    if league_id and viewer_roster_id:
        try:
            ctx = get_league_ctx_from_cache(platform=platform, league_id=league_id, season=season)
            analysis_html = get_trade_ai_analysis(
                ctx=ctx,
                viewer_roster_id=str(viewer_roster_id),
                viewer_side=viewer_side,
                side_a=side_a,
                side_b=side_b,
            )
        except Exception as e:
            print(f"[trade-ai] skipped: {e}")
            analysis_html = ""

    return jsonify({
        "side_a": side_a,
        "side_b": side_b,
        "diff": diff,
        "abs_diff": abs_diff,
        "fair_threshold": fair_band,
        "fair_pct": FAIR_PCT,
        "verdict": verdict,
        "analysis_html": analysis_html,
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

    cleaned_players = _sanitize_for_json(model_value_table)

    return jsonify(cleaned_players)


@app.route("/api/teams")
def api_teams():
    league_id = (request.args.get("league_id") or "").strip()
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    season = int(request.args.get("season") or datetime.now().year)

    if not league_id:
        return jsonify([])

    try:
        ctx = get_league_ctx_from_cache(platform=platform, league_id=league_id, season=season)
        users = ctx.get("users", [])
        rosters = ctx.get("rosters", [])

        teams = []
        for roster in rosters:
            roster_id = str(roster.get("roster_id", ""))
            user_id = roster.get("owner_id")

            # Find the user for this roster
            user = next((u for u in users if u.get("user_id") == user_id), None)
            if user:
                team_name = user.get("team_name") or user.get("display_name") or f"Team {roster_id}"
                username = user.get("username") or user.get("display_name") or ""
            else:
                team_name = f"Team {roster_id}"
                username = ""

            teams.append({
                "roster_id": roster_id,
                "team_name": team_name,
                "username": username,
                "user_id": user_id
            })

        # Sort by team name for consistent ordering
        teams.sort(key=lambda x: x["team_name"])
        return jsonify(teams)

    except Exception as e:
        print(f"[api/teams] error: {e}")
        return jsonify([])


@app.route("/api/value-movers")
def api_value_movers():
    try:
        days = int(request.args.get("days", 7))
    except (TypeError, ValueError):
        days = 7

    try:
        limit = int(request.args.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5

    league_type = str(request.args.get("league_type", "1qb")).strip().lower()

    try:
        league_size = int(request.args.get("league_size", 10))
        # Validate league size is one of the supported sizes
        if league_size not in [8, 10, 12, 14]:
            league_size = 10
    except (TypeError, ValueError):
        league_size = 10

    payload = get_top_movers(days=max(days, 1), limit=max(limit, 1), league_type=league_type,
                             league_size=league_size) or {}

    if isinstance(payload, list):
        movers = payload
        risers = [m for m in movers if float(m.get("delta") or 0) > 0]
        fallers = [m for m in movers if float(m.get("delta") or 0) < 0]

        risers.sort(key=lambda x: float(x.get("delta") or 0), reverse=True)
        fallers.sort(key=lambda x: float(x.get("delta") or 0))

        payload = {
            "used_days": days,
            "risers": risers[:limit],
            "fallers": fallers[:limit],
        }
    else:
        payload = {
            "used_days": payload.get("used_days", days),
            "risers": payload.get("risers", [])[:limit],
            "fallers": payload.get("fallers", [])[:limit],
        }

    return jsonify(payload)


@app.route("/api/player-deltas")
def api_player_deltas():
    """
    Return recent 7-day deltas for all players in a compact format.
    Used for showing delta badges in the trade calculator.
    """
    try:
        days = int(request.args.get("days", 7))
    except (TypeError, ValueError):
        days = 7

    league_type = str(request.args.get("league_type", "1qb")).strip().lower()

    try:
        league_size = int(request.args.get("league_size", 10))
        if league_size not in [8, 10, 12, 14]:
            league_size = 10
    except (TypeError, ValueError):
        league_size = 10

    # Get top movers data which includes deltas
    movers_data = get_top_movers(days=days, limit=1000, league_type=league_type, league_size=league_size) or {}

    # Build a simple player_id -> delta map
    deltas = {}
    for player in movers_data.get("risers", []) + movers_data.get("fallers", []):
        pid = str(player.get("player_id", ""))
        delta = player.get("delta")
        if pid and delta is not None:
            deltas[pid] = float(delta)

    return jsonify(deltas)


@app.route("/api/player-indicators")
def api_player_indicators():
    """
    Return rookie and breakout indicators for players.
    Returns: {
      "rookies": ["player_id1", "player_id2", ...],
      "breakouts": ["player_id3", "player_id4", ...]
    }
    """
    try:
        from datetime import datetime

        league_type = str(request.args.get("league_type", "1qb")).strip().lower()

        try:
            league_size = int(request.args.get("league_size", 10))
            if league_size not in [8, 10, 12, 14]:
                league_size = 10
        except (TypeError, ValueError):
            league_size = 10

        # Get current NFL state
        nfl_state = get_nfl_state() or {}
        current_season = int(nfl_state.get("season") or datetime.now().year)

        # Load all players to check for rookies
        players_index = load_players_index() or {}
        rookies = []

        for player_id, player_data in players_index.items():
            # Check if rookie (years_exp == 0 or rookie_year == current_season)
            years_exp = player_data.get("years_exp")
            rookie_year = player_data.get("rookie_year")

            if years_exp == 0 or years_exp == "0":
                rookies.append(str(player_id))
            elif rookie_year and int(rookie_year) == current_season:
                rookies.append(str(player_id))

        # Get breakouts using multi-factor algorithm
        # Falls back to value-based detection if advanced metrics not available
        breakouts = []

        # Check if we're in offseason
        season_type = str(nfl_state.get("season_type", "")).lower().strip()
        is_offseason = season_type == "off"

        if is_offseason:
            # During offseason, use roster change-based breakout detection
            try:
                from data_building.offseason_opportunity import get_offseason_breakout_candidates

                offseason_candidates = get_offseason_breakout_candidates(current_season, min_score=30)
                breakouts = [str(c["player_id"]) for c in offseason_candidates]
                print(f"[player-indicators] Offseason: Found {len(breakouts)} breakout candidates from roster changes")

            except Exception as e:
                print(f"[player-indicators] Offseason breakout detection failed: {e}")

        else:
            # During season, use in-season breakout detection
            try:
                from data_building.advanced_metrics import detect_breakout_candidates

                breakout_candidates = detect_breakout_candidates(lookback_days=14)
                breakouts = [str(b["player_id"]) for b in breakout_candidates]
                print(f"[player-indicators] Found {len(breakouts)} breakout candidates using advanced metrics")

            except Exception as e:
                # Fallback to simple value-based detection (more restrictive threshold)
                print(f"[player-indicators] Advanced metrics unavailable, using fallback: {e}")
                movers_data = get_top_movers(days=7, limit=1000) or {}

                for player in movers_data.get("risers", []):
                    delta = player.get("delta", 0)
                    position = player.get("position", "")

                    # More restrictive thresholds to reduce false positives
                    # Higher threshold for TEs since they're more volatile
                    threshold = 100 if position == "TE" else 75

                    if delta >= threshold:
                        pid = str(player.get("player_id", ""))
                        if pid:
                            breakouts.append(pid)

        return jsonify({
            "rookies": rookies,
            "breakouts": breakouts
        })

    except Exception as e:
        print(f"[player-indicators] Error: {e}")
        return jsonify({"rookies": [], "breakouts": []})


@app.route("/api/breakout-candidates")
def api_breakout_candidates():
    """
    Get breakout candidates - automatically switches between offseason and in-season detection.
    Returns full candidate objects with stats, not just IDs.
    """
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
                print(f"[breakout-candidates] Offseason mode: {len(candidates)} candidates")
            except Exception as e:
                print(f"[breakout-candidates] Offseason detection error: {e}")
        else:
            # Use in-season breakout detection with enrichment
            try:
                from data_building.advanced_metrics import detect_breakout_candidates

                breakout_ids = detect_breakout_candidates(lookback_days=14)

                # Enrich with full player data
                players_index = load_players_index() or {}
                value_table = load_model_value_table() or []
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

                print(f"[breakout-candidates] In-season mode: {len(candidates)} candidates")
            except Exception as e:
                print(f"[breakout-candidates] In-season detection error: {e}")
                # Fallback to value movers
                movers_data = get_top_movers(days=7, limit=100) or {}
                players_index = load_players_index() or {}
                value_table = load_model_value_table() or []
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
        print(f"[breakout-candidates] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])


@app.route("/api/player-advanced-metrics/<player_id>")
def api_player_advanced_metrics(player_id: str):
    """
    Get advanced efficiency metrics for a specific player.

    PREMIUM FEATURE - Requires active subscription.

    Returns:
        {
            "player_id": "123",
            "position": "WR",
            "metrics": {
                "yards_per_target": 8.5,
                "catch_rate": 0.72,
                "yards_per_reception": 11.8,
                "yards_per_carry": 4.2,
                "snap_share": 0.78,
                "opportunity_share": 8.5,
                "red_zone_usage": 2.1,
                "role_score": 67.3,
                "usage_trend": 15.2,
                "efficiency_trend": 8.7
            },
            "as_of_date": "2025-01-15"
        }
    """
    try:
        from data_building.advanced_metrics import get_player_metrics

        # Advanced metrics are now available to all users (no premium check)
        metrics = get_player_metrics(str(player_id))

        if not metrics:
            return jsonify({
                "player_id": str(player_id),
                "error": "No metrics available for this player"
            }), 404

        # Extract date and clean up metrics
        as_of_date = str(metrics.pop("as_of_date", None))
        metrics.pop("id", None)  # Remove internal ID

        return jsonify({
            "player_id": str(player_id),
            "position": metrics.get("position"),
            "metrics": {
                k: float(v) if v is not None else None
                for k, v in metrics.items()
                if k != "player_id" and k != "position"
            },
            "as_of_date": as_of_date,
        })

    except Exception as e:
        print(f"[player-advanced-metrics] Error for {player_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "player_id": str(player_id),
            "error": "Failed to retrieve metrics"
        }), 500


@app.route("/api/advanced-metrics/top-role-players")
def api_top_role_players():
    """
    Get players with highest role scores (usage + efficiency composite).

    Query params:
        position: Filter by position (QB/RB/WR/TE) or omit for all
        limit: Max number of players (default 50)

    Returns:
        [
            {
                "player_id": "123",
                "position": "RB",
                "role_score": 78.5,
                "snap_share": 0.82,
                "opportunity_share": 18.3,
                ...
            },
            ...
        ]
    """
    try:
        from data_building.advanced_metrics import get_top_role_players

        position = request.args.get("position")
        if position:
            position = position.upper().strip()
            if position not in ("QB", "RB", "WR", "TE"):
                position = None

        try:
            limit = int(request.args.get("limit", 50))
            limit = max(1, min(limit, 200))  # Clamp between 1-200
        except (TypeError, ValueError):
            limit = 50

        players = get_top_role_players(position=position, limit=limit)

        # Clean up internal fields
        for player in players:
            player.pop("id", None)
            # Convert decimals to floats
            for k, v in player.items():
                if v is not None and k not in ("player_id", "position", "as_of_date"):
                    player[k] = float(v)

        return jsonify(players)

    except Exception as e:
        print(f"[top-role-players] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])


@app.route("/api/advanced-metrics/breakout-candidates")
def api_advanced_metrics_breakout_candidates():
    """
    Get breakout candidates using multi-factor analysis.

    Query params:
        lookback_days: Days to analyze trends (default 14)
        min_games: Minimum games played (default 2)

    Returns:
        [
            {
                "player_id": "456",
                "name": "Player Name",
                "position": "WR",
                "age": 23.5,
                "breakout_score": 67.3,
                "score_components": {
                    "snap_increase": 15.2,
                    "opportunity_increase": 22.5,
                    "efficiency_gains": 12.1,
                    ...
                },
                "value_delta": 125.0
            },
            ...
        ]
    """
    try:
        from data_building.advanced_metrics import detect_breakout_candidates

        try:
            lookback_days = int(request.args.get("lookback_days", 14))
            lookback_days = max(7, min(lookback_days, 90))  # Clamp 7-90 days
        except (TypeError, ValueError):
            lookback_days = 14

        try:
            min_games = int(request.args.get("min_games", 2))
            min_games = max(1, min(min_games, 10))
        except (TypeError, ValueError):
            min_games = 2

        candidates = detect_breakout_candidates(
            lookback_days=lookback_days,
            min_games=min_games,
        )

        return jsonify(candidates)

    except Exception as e:
        print(f"[breakout-candidates] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])


@app.route("/api/offseason-breakout-candidates")
def api_offseason_breakout_candidates():
    """
    Get offseason breakout candidates based on roster changes and vacated opportunity.

    PREMIUM FEATURE - Requires active subscription.

    Identifies players who will benefit from departed teammates (FA, trades, retirements).
    Examples:
    - Mike Evans leaves TB → Egbuka gets targets
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
        from utils.utils import load_model_value_table
        model_values = load_model_value_table() or []
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
        print(f"[offseason-breakout-candidates] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])


@app.route("/api/calculate-breakout-scores")
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
        print(f"[calculate-breakout-scores] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


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


@app.route("/api/player-details/<player_id>")
def api_player_details(player_id: str):
    """Get comprehensive player details for modal display."""
    try:
        from utils.utils import load_players_index, load_model_value_table
        from dashboard_services.api import get_effective_scoring_settings
        from dashboard_services.platform_api import sync_league_globals
        import json
        import os
        import glob
        import re

        # Get league context
        league_id = request.args.get("league_id")
        platform = request.args.get("platform", "sleeper")
        season = int(request.args.get("season", datetime.now().year))

        # Sync league globals if league_id provided
        if league_id:
            sync_league_globals(platform, league_id, season)
            scoring_settings = get_effective_scoring_settings()
        else:
            # Default scoring settings if no league context
            scoring_settings = {
                "passYards": 0.04,
                "passTD": 4.0,
                "passInterceptions": -2.0,
                "rushYards": 0.1,
                "rushTD": 6.0,
                "pointsPerReception": 1.0,
                "receivingYards": 0.1,
                "receivingTD": 6.0,
                "fumbles": -2.0
            }

        players_index = load_players_index() or {}
        player_meta = players_index.get(player_id, {})

        if not player_meta:
            return jsonify({"error": "Player not found"}), 404

        player_team = player_meta.get("team", "")

        # Get value data
        value_table = load_model_value_table() or []
        player_value = next((p for p in value_table if str(p.get("id")) == str(player_id)), {})

        # Get FULL value history from database (not just 90 days)
        value_history = get_player_value_history(player_id, days=365)

        # Load game logs from sleeper_stats for all available seasons
        game_logs_by_year = {}

        # Find all available season years (handle both old and new naming patterns)
        stats_files = glob.glob(os.path.join("cache", "sleeper_stats", "sleeper_stats_*.json"))
        available_years = set()
        for stats_file in stats_files:
            try:
                basename = os.path.basename(stats_file)
                # New pattern: sleeper_stats_s2025_w1_2025-12-16.json
                if basename.startswith("sleeper_stats_s"):
                    # Use regex to extract year from s{YEAR}_w{WEEK} pattern
                    match = re.match(r'sleeper_stats_s(\d+)_w(\d+)', basename)
                    if match:
                        year = int(match.group(1))
                        available_years.add(year)
            except:
                continue

        # Process each available year
        for season_year in sorted(available_years, reverse=True):  # Most recent first
            game_logs = []

            # Load schedule data for ALL weeks to show all games
            schedule_by_week = {}
            schedule_pattern = os.path.join("cache", "schedule", f"schedule_s{season_year}_w*_d*.json")
            for schedule_file in glob.glob(schedule_pattern):
                try:
                    # Extract week from filename: schedule_s2024_w1_d2024-09-05.json
                    filename = os.path.basename(schedule_file)
                    week_num = int(filename.split('_w')[1].split('_')[0])

                    with open(schedule_file, 'r') as f:
                        games = json.load(f)
                        # Ensure games is a list
                        if isinstance(games, list) and week_num not in schedule_by_week:
                            schedule_by_week[week_num] = games
                except Exception as e:
                    print(f"[api_player_details] Error loading schedule {schedule_file}: {e}")
                    continue

            # Load all stats for this season into memory
            stats_by_week = {}
            stats_pattern = os.path.join("cache", "sleeper_stats", f"sleeper_stats_s{season_year}_w*.json")
            week_files = glob.glob(stats_pattern)

            for week_file in week_files:
                try:
                    basename = os.path.basename(week_file)
                    # Extract week number from filename
                    match = re.match(r'sleeper_stats_s(\d+)_w(\d+)', basename)
                    if match:
                        week_num = int(match.group(2))
                    else:
                        continue

                    with open(week_file, 'r') as f:
                        week_stats = json.load(f)
                        stats_by_week[week_num] = week_stats
                except Exception as e:
                    continue

            # Check if player has ANY stats in this season
            # Skip the season if player has no stats at all (didn't exist yet or retired)
            player_has_stats_this_season = False
            for week_stats in stats_by_week.values():
                if player_id in week_stats:
                    player_has_stats_this_season = True
                    break

            if not player_has_stats_this_season:
                print(f"[api_player_details] Player {player_id} has no stats in {season_year}, skipping season")
                continue

            # Now iterate through schedule and create game logs for ALL games
            for week_num in sorted(schedule_by_week.keys()):
                games = schedule_by_week[week_num]

                # Ensure games is a list
                if not isinstance(games, list):
                    continue

                # Find player's team game this week
                opponent = ""
                is_away = False
                game_date = ""

                for game in games:
                    # Ensure game is a dict
                    if not isinstance(game, dict):
                        continue

                    home_team = game.get("home", "")
                    away_team = game.get("away", "")

                    if player_team == home_team:
                        opponent = away_team
                        is_away = False
                        game_date = game.get("gameDate", "")
                        break
                    elif player_team == away_team:
                        opponent = home_team
                        is_away = True
                        game_date = game.get("gameDate", "")
                        break

                # Skip if player's team didn't have a game this week
                if not opponent:
                    continue

                # Check if we have stats for this player this week
                week_stats = stats_by_week.get(week_num, {})
                stats = week_stats.get(player_id)

                if stats:
                    # Player has stats - calculate fantasy points using league scoring settings
                    pts = 0.0
                    
                    # Base scoring
                    pts += (stats.get("pass_yd") or 0) * scoring_settings.get("passYards", 0.04)
                    pts += (stats.get("pass_td") or 0) * scoring_settings.get("passTD", 4.0)
                    pts += (stats.get("pass_int") or 0) * scoring_settings.get("passInterceptions", -2.0)
                    pts += (stats.get("rush_yd") or 0) * scoring_settings.get("rushYards", 0.1)
                    pts += (stats.get("rush_td") or 0) * scoring_settings.get("rushTD", 6.0)
                    pts += (stats.get("rec") or 0) * scoring_settings.get("pointsPerReception", 1.0)
                    pts += (stats.get("rec_yd") or 0) * scoring_settings.get("receivingYards", 0.1)
                    pts += (stats.get("rec_td") or 0) * scoring_settings.get("receivingTD", 6.0)
                    pts += (stats.get("fum_lost") or 0) * scoring_settings.get("fumbles", -2.0)
                    
                    # Yardage bonuses
                    pass_yds = stats.get("pass_yd") or 0
                    rush_yds = stats.get("rush_yd") or 0
                    rec_yds = stats.get("rec_yd") or 0
                    rush_rec_yds = rush_yds + rec_yds
                    
                    # Pass yardage bonuses
                    if pass_yds >= 400:
                        pts += scoring_settings.get("bonus_pass_yd_400", 0)
                    elif pass_yds >= 300:
                        pts += scoring_settings.get("bonus_pass_yd_300", 0)
                    
                    # Rush yardage bonuses
                    if rush_yds >= 200:
                        pts += scoring_settings.get("bonus_rush_yd_200", 0)
                    elif rush_yds >= 100:
                        pts += scoring_settings.get("bonus_rush_yd_100", 0)
                    
                    # Receiving yardage bonuses
                    if rec_yds >= 200:
                        pts += scoring_settings.get("bonus_rec_yd_200", 0)
                    elif rec_yds >= 100:
                        pts += scoring_settings.get("bonus_rec_yd_100", 0)
                    
                    # Combined rush/rec yardage bonuses
                    if rush_rec_yds >= 200:
                        pts += scoring_settings.get("bonus_rush_rec_yd_200", 0)
                    elif rush_rec_yds >= 100:
                        pts += scoring_settings.get("bonus_rush_rec_yd_100", 0)

                    game_log = {
                        "week": week_num,
                        "date": game_date,
                        "opponent": f"@{opponent}" if is_away else opponent,
                        "fantasy_pts": round(pts, 1),
                        "stats": {
                            "pass_yd": stats.get("pass_yd"),
                            "pass_td": stats.get("pass_td"),
                            "pass_int": stats.get("pass_int"),
                            "rush_att": stats.get("rush_att"),
                            "rush_yd": stats.get("rush_yd"),
                            "rush_td": stats.get("rush_td"),
                            "rec": stats.get("rec"),
                            "rec_tgt": stats.get("rec_tgt"),
                            "rec_yd": stats.get("rec_yd"),
                            "rec_td": stats.get("rec_td"),
                            "fum_lost": stats.get("fum_lost"),
                        }
                    }
                else:
                    # No stats - show game but with 0 points
                    game_log = {
                        "week": week_num,
                        "date": game_date,
                        "opponent": f"@{opponent}" if is_away else opponent,
                        "fantasy_pts": 0.0,
                        "stats": {
                            "pass_yd": None,
                            "pass_td": None,
                            "pass_int": None,
                            "rush_att": None,
                            "rush_yd": None,
                            "rush_td": None,
                            "rec": None,
                            "rec_tgt": None,
                            "rec_yd": None,
                            "rec_td": None,
                            "fum_lost": None,
                        }
                    }

                game_logs.append(game_log)

            # Only add year if there are games
            if game_logs:
                # Sort game logs chronologically by date (earliest to latest)
                game_logs.sort(key=lambda g: g.get("date", "") or "")
                game_logs_by_year[season_year] = game_logs

        response = {
            "player_id": player_id,
            "name": player_meta.get("name", "Unknown"),
            "position": player_meta.get("pos"),
            "team": player_meta.get("team"),
            "age": player_value.get("age"),
            "pos_rank": player_value.get("pos_rank"),
            "pos_rank_label": player_value.get("pos_rank_label"),
            "stats": {
                "value": round(player_value.get("value", 0), 1) if player_value.get("value") else None,
                "sf_value": round(player_value.get("sf_value", 0), 1) if player_value.get("sf_value") else None,
                "pos_rank": player_value.get("pos_rank"),
                "pos_rank_label": player_value.get("pos_rank_label"),
                "years_exp": player_meta.get("years_exp"),
            },
            "value_history": value_history,
            "game_logs_by_year": game_logs_by_year,
        }

        return jsonify(response)

    except Exception as e:
        print(f"[api_player_details] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-details/<roster_id>")
def api_team_details(roster_id: str):
    """Get comprehensive team details for modal display."""
    try:
        from utils.utils import load_players_index, load_model_value_table
        from dashboard_services.api import get_nfl_state

        # Get league context
        league_id = request.args.get("league_id")
        platform = request.args.get("platform", "sleeper")
        season = request.args.get("season")

        if not league_id:
            return jsonify({"error": "league_id required"}), 400

        if not season:
            nfl_state = get_nfl_state()
            season = str(nfl_state.get("season") or datetime.now().year)

        season = int(season)

        # Get league data
        league = get_league(platform, league_id, season)
        rosters = get_rosters(platform, league_id, season) or []
        users = get_users(platform, league_id, season) or []

        # Find the specific roster
        roster = next((r for r in rosters if str(r.get("roster_id")) == str(roster_id)), None)
        if not roster:
            return jsonify({"error": "Roster not found"}), 404

        # Get owner info
        owner_id = roster.get("owner_id")
        user = next((u for u in users if u.get("user_id") == owner_id), None)

        username = user.get("display_name") if user else None
        team_name = user.get("metadata", {}).get("team_name") if user else username
        avatar = avatar_from_users(platform, users, owner_id)
        if team_name is None:
            team_name = username

        # Get record
        settings = roster.get("settings") or {}
        wins = settings.get("wins", 0)
        losses = settings.get("losses", 0)
        ties = settings.get("ties", 0)

        record_str = f"{wins}-{losses}"
        if ties:
            record_str += f"-{ties}"

        # Get players with values
        players_index = load_players_index() or {}
        value_table = load_model_value_table() or []
        values_by_id = {str(row["id"]): row for row in value_table if isinstance(row, dict) and row.get("id")}

        player_ids = roster.get("players") or []
        starters = roster.get("starters") or []

        # Build roster with values
        roster_players = []
        total_value = 0.0

        ages_found = 0
        ages_missing = 0
        for pid in player_ids:
            pid_str = str(pid)
            player_meta = players_index.get(pid_str, {})
            value_row = values_by_id.get(pid_str, {})

            value = value_row.get("value", 0) or 0
            total_value += float(value)

            position = player_meta.get("pos") or ""
            if position == "PK":
                position = "K"
            elif position in ["DST", "D/ST"]:
                position = "DEF"

            # Special handling for defense players (team abbreviations as IDs)
            player_name = player_meta.get("name", "Unknown")
            player_team = player_meta.get("team")

            # If player_id is a team abbreviation and no metadata found, treat as defense
            if len(pid_str) == 3 and pid_str.isupper() and player_name == "Unknown":
                # This is likely a defense player with team ID
                full_team_name = get_team_full_name(pid_str)
                player_name = f"{full_team_name} Defense"
                position = "DEF"
                player_team = pid_str

            # Get age from value table (has calculated age) or player meta
            # For kickers, prioritize players_index birthday info
            if position == "K":
                age = player_meta.get("age") or value_row.get("age")
            else:
                age = value_row.get("age") or player_meta.get("age")
            if age is not None:
                ages_found += 1
            else:
                ages_missing += 1

            roster_players.append({
                "player_id": pid_str,
                "name": player_name,
                "position": position,
                "team": player_team,
                "age": age,
                "years_exp": player_meta.get("years_exp"),
                "value": round(float(value), 1) if value else None,
                "pos_rank_label": value_row.get("pos_rank_label"),
                "is_starter": pid_str in starters
            })

        print(
            f"[api_team_details] Ages: {ages_found} found, {ages_missing} missing out of {len(roster_players)} total players")
        # Sort by position order (QB, RB, WR, TE, K, DEF), then by value within position
        pos_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4, "DEF": 5}
        roster_players.sort(key=lambda p: (pos_order.get(p["position"], 99), -(p["value"] or 0)))

        # Get draft picks
        traded_picks = get_traded_picks(platform, league_id, season)
        num_rounds = int((league.get("settings") or {}).get("draft_rounds", 4))
        current_season = int(league.get("season") or season)

        # Build picks
        all_picks = []
        for offset in range(3):  # Next 3 years
            year = current_season + offset
            for rnd in range(1, num_rounds + 1):

                # Collect all picks this team owns for this year/round
                owned_picks = []

                # First check: All picks this team acquired from other teams
                for tp in traded_picks:
                    try:
                        if (int(tp.get("season")) == year and
                                int(tp.get("round")) == rnd and
                                int(tp.get("owner_id")) == int(roster_id)):
                            # This team acquired this pick from another team
                            owned_picks.append({
                                "current_owner": int(tp.get("owner_id")),
                                "original_owner": int(tp.get("roster_id")),
                                "previous_owner": int(tp.get("previous_owner_id")),
                                "trade_data": tp
                            })
                    except:
                        pass

                # Second check: This team's own draft position pick (only if not already found as acquired)
                own_position_found = any(p["original_owner"] == int(roster_id) for p in owned_picks)
                if not own_position_found:
                    for tp in traded_picks:
                        try:
                            if (int(tp.get("season")) == year and
                                    int(tp.get("round")) == rnd and
                                    int(tp.get("roster_id")) == int(roster_id)):
                                # This pick belongs to this roster's draft position
                                owned_picks.append({
                                    "current_owner": int(tp.get("owner_id")),
                                    "original_owner": int(tp.get("roster_id")),
                                    "previous_owner": int(tp.get("previous_owner_id")),
                                    "trade_data": tp
                                })
                                break
                        except:
                            pass

                # If no traded picks found, check if this team owns their own pick by default
                # BUT also check if we should add the default pick in addition to acquired picks
                own_position_as_acquired = any(p["original_owner"] == int(roster_id) for p in owned_picks)

                if not own_position_as_acquired:
                    # This team owns their own pick unless it was traded away
                    pick_traded_away = False
                    for tp in traded_picks:
                        try:
                            if (int(tp.get("season")) == year and
                                    int(tp.get("round")) == rnd and
                                    int(tp.get("roster_id")) == int(roster_id) and
                                    int(tp.get("owner_id")) != int(roster_id)):
                                pick_traded_away = True
                                break
                        except:
                            pass

                    if not pick_traded_away:
                        owned_picks.append({
                            "current_owner": int(roster_id),
                            "original_owner": int(roster_id),
                            "previous_owner": None,
                            "trade_data": None
                        })

                # Add all owned picks to the list
                for pick_info in owned_picks:
                    if pick_info["current_owner"] == int(roster_id):
                        via = None
                        previous_owner = pick_info["previous_owner"]
                        original_owner = pick_info["original_owner"]

                        if previous_owner is not None and previous_owner != original_owner:
                            # Find who it came from (the previous owner who traded it away)
                            via_roster = next((r for r in rosters if r.get("roster_id") == previous_owner), None)
                            if via_roster:
                                via_owner_id = via_roster.get("owner_id")
                                via_user = next((u for u in users if u.get("user_id") == via_owner_id), None)
                                via = via_user.get("display_name") if via_user else f"Team {previous_owner}"
                        elif original_owner != int(roster_id):
                            # This team acquired the pick from its original owner
                            via_roster = next((r for r in rosters if r.get("roster_id") == original_owner), None)
                            if via_roster:
                                via_owner_id = via_roster.get("owner_id")
                                via_user = next((u for u in users if u.get("user_id") == via_owner_id), None)
                                via = via_user.get("display_name") if via_user else f"Team {original_owner}"

                        all_picks.append({
                            "year": year,
                            "round": rnd,
                            "via": via
                        })

        # Sort picks by year then round
        all_picks.sort(key=lambda p: (p["year"], p["round"]))

        # Get graph data for team modal
        graphs_data = {}
        try:
            from utils.utils import z_better_outward
            import pandas as pd
            from dashboard_services.api import get_nfl_state

            # Check if we're in offseason and should use previous season data
            nfl_state = get_nfl_state() or {}
            current_nfl_season = int(nfl_state.get("season", current_season))
            season_type = str(nfl_state.get("season_type", "")).lower().strip()

            print(
                f"[api_team_details] Graph logic - current_season: {current_season}, current_nfl_season: {current_nfl_season}, season_type: '{season_type}'")

            # Determine which season to use for graphs
            graph_season = current_season
            if current_nfl_season > int(season) and season_type in {"offseason", "pre"}:
                # We're in offseason before current season has started, use previous season data
                graph_season = int(season) - 1
            elif current_nfl_season == int(season) and season_type == "offseason":
                # Current season is over, use completed season data
                graph_season = int(season)
            elif season_type in {"offseason", "pre"}:
                # We're in some form of offseason, try previous season
                graph_season = int(season) - 1

            print(f"[api_team_details] Using graph_season: {graph_season}")

            # Get league context for graphs
            ctx = get_league_ctx_from_cache(platform, league_id, graph_season)
            team_stats = ctx.get("team_stats")
            df_weekly = ctx.get("df_weekly")

            print(
                f"[api_team_details] Graph context - team_stats exists: {team_stats is not None}, df_weekly exists: {df_weekly is not None and not df_weekly.empty}")
            if df_weekly is not None and not df_weekly.empty:
                print(f"[api_team_details] df_weekly shape: {df_weekly.shape}")

            # If we don't have weekly data for the chosen season, try previous season
            if (df_weekly is None or df_weekly.empty) and graph_season > 2025:
                fallback_season = graph_season - 1
                print(f"[api_team_details] No data for {graph_season}, trying fallback season: {fallback_season}")

                # Resolve correct league_id for fallback season
                from dashboard_services.api import resolve_league_id_for_season
                fallback_league_id = resolve_league_id_for_season(
                    platform=platform,
                    league_id=league_id,
                    current_season=current_season,
                    target_season=fallback_season
                )
                print(
                    f"[api_team_details] Using fallback league_id: {fallback_league_id} for season: {fallback_season}")

                ctx = get_league_ctx_from_cache(platform, fallback_league_id, fallback_season)
                team_stats = ctx.get("team_stats")
                df_weekly = ctx.get("df_weekly")
                graph_season = fallback_season
                print(
                    f"[api_team_details] Fallback context - team_stats exists: {team_stats is not None}, df_weekly exists: {df_weekly is not None and not df_weekly.empty}")
                if df_weekly is not None and not df_weekly.empty:
                    print(f"[api_team_details] Fallback df_weekly shape: {df_weekly.shape}")

            # Remove debug prints for cleaner logs
            if team_stats is not None and df_weekly is not None and not df_weekly.empty:
                # Filter to finalized weeks only (if finalized column exists)
                if "finalized" in df_weekly.columns:
                    df_weekly = df_weekly[df_weekly["finalized"] == True].copy()

                # Only build graphs if we have data after filtering
                if not df_weekly.empty:
                    print(f"[api_team_details] Building graphs with df_weekly shape: {df_weekly.shape}")
                    # Get weekly scores for this team
                    team_weekly = df_weekly[df_weekly["owner"] == team_name]
                    weekly_scores = []
                    if not team_weekly.empty:
                        for _, row in team_weekly.sort_values("week").iterrows():
                            weekly_scores.append({
                                "week": int(row["week"]),
                                "points": round(float(row["points"]), 1)
                            })

                    # Get league average weekly scores
                    league_avg = df_weekly.groupby("week")["points"].mean().reset_index()
                    league_avg_scores = []
                    for _, row in league_avg.iterrows():
                        league_avg_scores.append({
                            "week": int(row["week"]),
                            "points": round(float(row["points"]), 1)
                        })

                    # Get z-scores for radar chart
                    metrics = ["PF", "PA", "MAX", "MIN", "AVG", "STD"]
                    Z = z_better_outward(team_stats, metrics)

                    # Find this team's row in team_stats
                    if team_stats is not None:
                        available_teams = team_stats["owner"].tolist() if "owner" in team_stats.columns else []
                        print(f"[api_team_details] Available team names in stats: {available_teams}")
                        print(f"[api_team_details] Looking for team_name: '{team_name}'")

                    # Try exact match first
                    team_row = team_stats[team_stats["owner"] == team_name]

                    # If no exact match, try fuzzy matching for team name variations
                    if team_row.empty and available_teams:
                        # Try case-insensitive match
                        team_row = team_stats[team_stats["owner"].str.lower() == team_name.lower()]

                        # If still no match, try partial matching (handle team name changes)
                        if team_row.empty:
                            for available_team in available_teams:
                                # Remove special characters and convert to lowercase for comparison
                                clean_available = ''.join(c.lower() for c in available_team if c.isalnum())
                                clean_target = ''.join(c.lower() for c in team_name if c.isalnum())

                                if clean_available in clean_target or clean_target in clean_available:
                                    team_row = team_stats[team_stats["owner"] == available_team]
                                    print(f"[api_team_details] Fuzzy matched '{team_name}' to '{available_team}'")
                                    break

                    if not team_row.empty:
                        team_idx = team_row.index[0]
                        z_scores = Z.iloc[team_idx].values.astype(float).tolist()

                        # Get raw stats too
                        raw_stats = {}
                        for metric in metrics:
                            raw_stats[metric] = round(float(team_row[metric].iloc[0]), 1)

                        graphs_data = {
                            "weekly_scores": weekly_scores,
                            "league_avg_scores": league_avg_scores,
                            "radar": {
                                "metrics": metrics,
                                "z_scores": z_scores,
                                "raw_stats": raw_stats
                            },
                            "season_used": graph_season  # Add info about which season data was used
                        }
                        print(f"[api_team_details] Successfully generated graphs_data with {len(weekly_scores)} weeks")
                    else:
                        print(f"[api_team_details] No graphs generated - team_row is empty for team_name='{team_name}'")
                else:
                    print(f"[api_team_details] No graphs generated - df_weekly is empty after filtering")
            else:
                print(
                    f"[api_team_details] No graphs generated - team_stats: {team_stats is not None}, df_weekly: {df_weekly is not None and not df_weekly.empty}")
        except Exception as graph_err:
            print(f"[api_team_details] Error getting graph data: {graph_err}")
            import traceback
            traceback.print_exc()
            # Continue without graph data

        response = {
            "roster_id": roster_id,
            "team_name": team_name,
            "username": username,
            "avatar": avatar,
            "record": record_str,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "total_value": round(total_value, 1),
            "roster": roster_players,
            "picks": all_picks,
            "graphs": graphs_data
        }

        return jsonify(response)

    except Exception as e:
        print(f"[api_team_details] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/subscription-status")
def api_subscription_status():
    """Check if user has premium access for a league."""
    from dashboard_services.subscriptions import get_subscription_info

    user_id = request.args.get("user_id") or session.get("viewer_username")
    league_id = request.args.get("league_id")
    platform = request.args.get("platform", "sleeper")

    try:
        sub_info = get_subscription_info(user_id, league_id, platform)
        return jsonify(sub_info)
    except Exception as e:
        print(f"[api_subscription_status] Error: {e}")
        return jsonify({"has_premium": False, "subscription_type": None, "error": str(e)}), 500


@app.route("/api/sleeper-user-leagues")
def api_sleeper_user_leagues():
    username = (request.args.get("username") or "").strip()

    # If no username provided, try to get from session
    if not username:
        username = session.get("viewer_username")

    if not username:
        return jsonify({"ok": False, "error": "Missing username"}), 400

    season = int(request.args.get("season") or get_nfl_state().get("season"))

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


@app.route("/api/changelog")
def api_changelog():
    """Return the changelog entries."""
    return jsonify(CHANGELOG)


@app.route("/api/espn-validate-league")
def api_espn_validate_league():
    """
    Validate an ESPN league ID and return basic league info.
    Used by the landing page ESPN flow.
    """
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id or not league_id.isdigit():
        return jsonify({"ok": False, "error": "Invalid ESPN league ID. Must be a number."}), 400

    nfl_state = get_nfl_state() or {}
    season = int(request.args.get("season") or nfl_state.get("season") or datetime.now().year)

    try:
        from dashboard_services.providers.espn_api import get_league as espn_get_league
        info = espn_get_league(season, league_id)
        return jsonify({
            "ok": True,
            "league": {
                "league_id": info.get("league_id"),
                "name": info.get("name") or f"ESPN League {league_id}",
                "season": info.get("season"),
            },
        })
    except Exception as e:
        msg = str(e)
        if "Missing required env var" in msg:
            return jsonify({"ok": False, "error": "Server not configured for ESPN (missing ESPN_S2/ESPN_SWID)."}), 503
        return jsonify({"ok": False, "error": f"Could not load ESPN league: {msg}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
