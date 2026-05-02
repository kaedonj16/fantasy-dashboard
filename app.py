import hashlib
import html
import json
import logging
import os
import re
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
from dashboard_services.ai.renderer import (
    get_team_gm_memo,
    get_front_office_briefing,
    get_power_rankings_html,
    get_trade_suggestions_html,
    get_roster_grade,
    render_roster_grade_badge,
)
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
from dashboard_services.pages.graphs_page import build_graphs_body, build_career_graphs_body
from dashboard_services.pages.history_page import (
    build_history_body,
    build_regular_season_team_stats,
    get_champion_and_runner_up,
    sort_team_stats,
    _build_summary as _build_history_summary,
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
from data_building.trade_intel.league_discovery import seed_user as _seed_user_leagues
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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Sentry error tracking ─────────────────────────────────────────────────────
_sentry_dsn = os.environ.get("SENTRY_DSN", "")
if _sentry_dsn:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration
        sentry_sdk.init(
            dsn=_sentry_dsn,
            integrations=[FlaskIntegration()],
            traces_sample_rate=0.05,   # 5% of requests sampled for performance
            send_default_pii=False,
        )
        logger.info("[sentry] Error tracking enabled")
    except ImportError:
        logger.warning("[sentry] sentry-sdk not installed — error tracking disabled")
else:
    logger.info("[sentry] SENTRY_DSN not set — error tracking disabled")

DASHBOARD_CACHE = {}

# How long a league context is considered fresh
CACHE_TTL = 60 * 60 * 6  # 6 hours

# How long value-table cache entries live
VALUE_CACHE_TTL = 60 * 60 * 3  # 3 hours

# How long to cache rendered page HTML (Teams, Activity, Graphs) per league
PAGE_HTML_TTL = 60 * 10  # 10 minutes

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

_secret_key = os.environ.get('FLASK_SECRET_KEY', '')
if not _secret_key:
    logging.warning(
        "FLASK_SECRET_KEY is not set — using insecure default. "
        "Set this env var in production to protect session cookies."
    )
    _secret_key = 'dev-secret-key-change-in-production'
app.secret_key = _secret_key
del _secret_key

plotly_js = get_plotlyjs()

# ── Rate limiting ─────────────────────────────────────────────────────────────
_redis_url = os.environ.get("REDIS_URL", "")
_limiter_storage = f"redis://{_redis_url.split('://')[-1]}" if _redis_url else "memory://"
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],
        storage_uri=_limiter_storage,
    )
    backend = "redis" if _redis_url else "memory (set REDIS_URL for multi-worker)"
    logger.info("[limiter] Flask-Limiter enabled (%s backend)", backend)
except ImportError:
    logger.warning("[limiter] Flask-Limiter not installed — rate limiting disabled")
    class _NoopLimiter:
        def limit(self, *a, **kw):
            def decorator(f): return f
            return decorator
    limiter = _NoopLimiter()

try:
    init_value_history_db()
except Exception as e:
    logger.warning("[value-history] init skipped: %s", e)

# Register breakout detection API routes
try:
    from dashboard_services.breakout_api import register_breakout_routes
    register_breakout_routes(app)
    logger.info("[breakout-api] Breakout API endpoints registered")
except Exception as e:
    logger.warning("[breakout-api] Registration skipped: %s", e)

# Register rookie prospect API routes
try:
    from dashboard_services.rookie_api import register_rookie_routes
    register_rookie_routes(app)
    logger.info("[rookie-api] Rookie API endpoints registered")
except Exception as e:
    logger.warning("[rookie-api] Registration skipped: %s", e)


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
            <button type="button" class="platform-btn" data-platform="espn">ESPN</button>
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
        <div id="espnFlow" style="display:none;">
          <div class="row">
            <label for="espnLeagueIdInput">ESPN League ID</label>
            <input type="text" id="espnLeagueIdInput" placeholder="e.g. 336414" autocomplete="off">
          </div>
          <div class="row">
            <label for="espnTeamName">Your Team Name <span style="font-weight:400;font-size:0.85em;">(optional)</span></label>
            <input type="text" id="espnTeamName" placeholder="e.g. Dynasty Monsters">
          </div>
          <div class="row">
            <button type="button" id="espnSubmitBtn">Find My League</button>
          </div>
          <div id="espnError" class="error-message" style="display:none;"></div>
          <p class="hint" style="margin-top:6px;" id="espnHint">
            Private leagues also need <code>ESPN_S2</code> and <code>ESPN_SWID</code> cookies set on the server.
          </p>
        </div>

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
      <div class="home-feature-icon"><svg style="width:32px;height:32px;color:#3b82f6;" viewBox="0 0 24 24" fill="currentColor"><path d="M5 9.2h3V19H5V9.2zM10.6 5h2.8v14h-2.8V5zm5.6 8H19v6h-2.8v-6z"/></svg></div>
      <h3>Trade Calculator</h3>
      <p>
        AI-powered trade analysis personalized to your roster. Get real-time value assessments,
        balance indicators, and specific counter suggestions — not generic advice.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon"><svg style="width:32px;height:32px;color:#10b981;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg></div>
      <h3>Dynasty Values</h3>
      <p>
        Hybrid valuation model blending market consensus with production metrics, age curves,
        and positional scarcity. Updated daily for all players and picks.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon"><svg style="width:32px;height:32px;color:#f59e0b;" viewBox="0 0 24 24" fill="currentColor"><path d="M7 2v11h3v9l7-12h-4l4-8z"/></svg></div>
      <h3>Weekly Hub</h3>
      <p>
        Live scoring context for every matchup. See projections, starters, and real-time updates
        in one clean view — perfect for Sunday trash talk.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon"><svg style="width:32px;height:32px;color:#ef4444;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg></div>
      <h3>Team Analytics</h3>
      <p>
        Position strength breakdowns, roster composition analysis, and competitive advantages
        mapped across your league. Know where you stand.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon"><svg style="width:32px;height:32px;color:#f43f5e;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg></div>
      <h3>Graphs & Trends</h3>
      <p>
        Visualize points for/against, strength of schedule, playoff odds, and luck metrics.
        Prove who's actually good and who just got lucky.
      </p>
    </div>

    <div class="home-feature-card">
      <div class="home-feature-icon"><i class="fa-solid fa-trophy" aria-hidden="true"></i></div>
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
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#38bdf8">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="BR Fantasy">

    <link rel="stylesheet" href="/static/dashboard.css">
    <link rel="stylesheet" href="/static/icons.css">
    <link rel="stylesheet" href="/static/font-awesome.css">

    <script>
      {plotly_js}
    </script>
    <script>
      if ('serviceWorker' in navigator) {{
        navigator.serviceWorker.register('/sw.js').catch(() => {{}});
      }}
    </script>
  </head>
  <body>
    <div id="app-scale">
      {nav}

      <!-- Top Banner Ad -->
      <div class="ad-container ad-top-banner">
        <ins class="adsbygoogle"
             style="display:block;max-height:90px;overflow:hidden;"
             data-ad-client="ca-pub-9164153092633845"
             data-ad-slot="5233061286"
             data-ad-format="horizontal"></ins>
      </div>

      <main id="page-root" class="overview-layout">
        {body}
      </main>

      <!-- Bottom Content Ad -->
      <div class="ad-container ad-bottom-content">
        <ins class="adsbygoogle"
             style="display:block;max-height:90px;overflow:hidden;"
             data-ad-client="ca-pub-9164153092633845"
             data-ad-slot="5233061286"
             data-ad-format="horizontal"></ins>
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


def _background_seed_user(user_id: str, username: Optional[str]) -> None:
    """Fire-and-forget: seed dynasty leagues for a Sleeper user on first login."""
    def _run():
        try:
            _seed_user_leagues(user_id, username=username)
        except Exception:
            pass  # never crash the request thread
    threading.Thread(target=_run, daemon=True).start()


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

def get_awards_agg_from_cache(platform: str, season: int, league_id: str):
    entry = DASHBOARD_CACHE.get(_cache_key(platform, season, league_id))
    if not entry:
        return None
    rec = entry.get("awards_agg")
    if not rec:
        return None
    ts, payload = rec
    if time.time() - ts > PAGE_HTML_TTL:
        return None
    return payload


def store_awards_agg(platform: str, season: int, league_id: str, payload) -> None:
    entry = DASHBOARD_CACHE.setdefault(_cache_key(platform, season, league_id), {})
    entry["awards_agg"] = (time.time(), payload)


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
        "  <img src='/static/images/sun-solid.png' class='settings-menu-icon theme-icon dark-icon' style='display:none;' alt='Toggle light mode'>"
        "  <span class='settings-menu-label theme-text'>Dark Mode</span>"
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
        "  <span id='gearDot' class='nav-notif-dot' style='display:none'></span>"
        f"  <div id='settingsDropdown' class='settings-dropdown' style='display:none;'>"
        f"    {settings_content}"
        "  </div>"
        "</div>"
    )

    if not league_id:
        def simple_pill(label: str, href: str, key: str) -> str:
            cls = "nav-pill active" if key == active else "nav-pill"
            return f"<a class='{cls}' href='{href}'>{label}</a>"

        def simple_dropdown(label: str, items: list, active_keys: list, dropdown_id: str = "playersNavDropdown") -> str:
            is_active = active in active_keys
            btn_cls = "nav-pill active" if is_active else "nav-pill"
            item_html = ""
            for item_label, href, item_key in items:
                item_cls = "nav-pill-dropdown-item active" if item_key == active else "nav-pill-dropdown-item"
                item_html += f"<a class='{item_cls}' href='{href}'>{item_label}</a>"
            btn_id  = dropdown_id.replace("Dropdown", "Btn")
            menu_id = dropdown_id.replace("Dropdown", "Menu")
            return (
                f"<div class='nav-pill-dropdown-wrapper' id='{dropdown_id}'>"
                f"  <button type='button' class='{btn_cls}' id='{btn_id}' onclick='toggleNavDropdown(event,\"{dropdown_id}\")'>"
                f"    {label} <span class='nav-pill-chevron'>&#x25BE;</span>"
                f"  </button>"
                f"  <div class='nav-pill-dropdown-menu' id='{menu_id}'>"
                f"    {item_html}"
                f"  </div>"
                f"</div>"
            )

        pills = [
            simple_pill("Home", "/", "home"),
            simple_dropdown("Trades", [
                ("Trade Calculator", "/trade",          "trade"),
                ("Trade Database",   "/trade-database", "trade-database"),
                ("Trade Intel",      "/trade-intel",    "trade-intel"),
            ], ["trade", "trade-database", "trade-intel"], "tradesNavDropdown"),
            simple_dropdown("Players", [
                ("Player Rankings", "/players",   "players"),
                ("Breakouts",       "/breakouts", "breakouts"),
                ("Rookies",         "/rookies",   "rookies"),
            ], ["players", "breakouts", "rookies"], "playersNavDropdown"),
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

    def nav_pill_dropdown(label: str, items: list, active_keys: list, dropdown_id: str = "playersNavDropdown") -> str:
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
        btn_id  = dropdown_id.replace("Dropdown", "Btn")
        menu_id = dropdown_id.replace("Dropdown", "Menu")
        return (
            f"<div class='nav-pill-dropdown-wrapper' id='{dropdown_id}'>"
            f"  <button type='button' class='{btn_cls}' id='{btn_id}' onclick='toggleNavDropdown(event,\"{dropdown_id}\")'>"
            f"    {label} <span class='nav-pill-chevron'>&#x25BE;</span>"
            f"  </button>"
            f"  <div class='nav-pill-dropdown-menu' id='{menu_id}'>"
            f"    {item_html}"
            f"  </div>"
            f"</div>"
        )

    # Generate dashboard URL for logo link
    dashboard_url = url_for("page_dashboard", platform=platform, season=season, league_id=league_id)

    # Navigation pills (no utilities)
    nav_pills = []
    nav_pills.append(nav_pill("Dashboard", "page_dashboard", "dashboard"))
    nav_pills.append(nav_pill_dropdown("Trades", [
        ("Trade Calculator", "page_trade",          "trade",          False),
        ("Trade Database",   "page_trade_database", "trade-database", False),
        ("Trade Intel",      "page_trade_intel",    "trade-intel",    False),
    ], ["trade", "trade-database", "trade-intel"], "tradesNavDropdown"))
    # Show Weekly Hub if draft has ended (during offseason) OR if in-season
    draft_ended = has_draft_ended(league_id, platform, season)
    if draft_ended or not offseason_mode:
        nav_pills.append(nav_pill("Weekly Hub", "page_weekly", "weekly"))
    nav_pills.append(nav_pill("Teams", "page_teams", "teams"))
    nav_pills.append(nav_pill("Activity", "page_activity", "activity"))
    nav_pills.append(nav_pill_dropdown("Players", [
        ("Player Rankings", "page_players",  "players",  False),
        ("Prospect Rankings", "page_prospects",  "prospects",   False),
        ("Breakout Engine", "page_breakouts","breakouts", False),
    ], ["players", "breakouts", "prospects"], "playersNavDropdown"))
    nav_pills.append(nav_pill_dropdown("Stats", [
        ("Awards",  "page_awards",  "awards",  False),
        ("Graphs",  "page_graphs",  "graphs",  False),
        ("History", "page_history", "history", False),
    ], ["awards", "graphs", "history"], "statsNavDropdown"))
    if not offseason_mode:
        nav_pills.append(nav_pill("Standings", "page_standings", "standings"))

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
            "  <span class='settings-menu-label'>Notifications</span>"
            "  <span id='settingsNotifDot' class='settings-notif-dot' style='display:none'></span>"
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
            "  <span id='gearDot' class='nav-notif-dot' style='display:none'></span>"
            f"  <div id='settingsDropdown' class='settings-dropdown' style='display:none;'>"
            f"    {settings_content}"
            "  </div>"
            "</div>"
        )
    else:
        # Logged-out user on a league page — offer quick sign-in
        signin_item = (
            "<button type='button' class='settings-menu-item' "
            "        onclick='document.getElementById(\"signinModal\").style.display=\"flex\"'>"
            "  <img src='/static/logout.png' class='settings-menu-icon' alt='Sign In' "
            "       style='transform:scaleX(-1);'>"
            "  <span class='settings-menu-label'>Sign In</span>"
            "</button>"
        )
        settings_content = signin_item + dark_mode_toggle_html
        settings_gear = (
            "<div class='settings-gear-wrapper'>"
            "  <button type='button' id='settingsGearBtn' class='utility-icon-btn' "
            "          aria-label='Settings' title='Settings'>"
            "    <img src='/static/gear.png' style='width: 16px; height: 16px;' alt='Settings'>"
            "  </button>"
            "  <span id='gearDot' class='nav-notif-dot' style='display:none'></span>"
            f"  <div id='settingsDropdown' class='settings-dropdown' style='display:none;'>"
            f"    {settings_content}"
            "  </div>"
            "</div>"
        )

    watchlist_btn = ""  # disabled

    # Build utility bar (desktop right side, mobile header)
    utility_bar = (
        "<div class='nav-utility-bar'>"
        f"  {watchlist_btn}"
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

    signin_modal = (
        f"<div id='signinModal' style='display:none;position:fixed;inset:0;"
        f"background:rgba(0,0,0,0.55);z-index:9999;align-items:center;justify-content:center;'>"
        f"  <div style='background:var(--card-bg,#1e2432);border:1px solid var(--border-color,#2d3748);"
        f"border-radius:12px;padding:28px 24px;width:320px;max-width:90vw;box-shadow:0 8px 32px rgba(0,0,0,0.4);'>"
        f"    <h3 style='margin:0 0 4px;font-size:18px;'>Sign In to your team</h3>"
        f"    <p style='margin:0 0 16px;font-size:13px;color:var(--text-muted,#94a3b8);'>"
        f"      Enter your Sleeper username to restore personalized features.</p>"
        f"    <form method='POST' action='/set-viewer'>"
        f"      <input type='hidden' name='platform' value='{platform}'>"
        f"      <input type='hidden' name='season' value='{season}'>"
        f"      <input type='hidden' name='league_id' value='{league_id}'>"
        f"      <input type='text' name='username' placeholder='sleeper_username' autofocus"
        f"             style='width:100%;box-sizing:border-box;padding:9px 12px;border-radius:8px;"
        f"border:1px solid var(--border-color,#2d3748);background:var(--input-bg,#0f1623);"
        f"color:var(--text-primary,#e2e8f0);font-size:14px;margin-bottom:14px;'>"
        f"      <div style='display:flex;gap:8px;'>"
        f"        <button type='submit' style='flex:1;padding:9px;border-radius:8px;border:none;"
        f"background:#3b82f6;color:#fff;font-weight:600;cursor:pointer;font-size:14px;'>Sign In</button>"
        f"        <button type='button' style='flex:1;padding:9px;border-radius:8px;border:1px solid"
        f" var(--border-color,#2d3748);background:transparent;color:var(--text-primary,#e2e8f0);"
        f"cursor:pointer;font-size:14px;'"
        f"                onclick='document.getElementById(\"signinModal\").style.display=\"none\"'>Cancel</button>"
        f"      </div>"
        f"    </form>"
        f"  </div>"
        f"</div>"
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
        f"{signin_modal}"
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


def _load_rookie_rankings_for_ctx() -> list[dict]:
    """Load current draft class rookies sorted by overall_rank for pick projection."""
    try:
        from data_building.rookie_pipeline.pipeline import get_rookie_rankings_from_db, get_active_rookie_class, is_draft_complete
        draft_year = get_active_rookie_class()
        try:
            from dashboard_services.db import get_conn as _get_conn
            with _get_conn() as _dc:
                _draft_done = is_draft_complete(draft_year, _dc)
        except Exception:
            _draft_done = is_draft_complete(draft_year)
        rows = get_rookie_rankings_from_db(draft_year, filter_undrafted=_draft_done)
        return [
            {
                "player_id":    r.get("player_id", ""),
                "name":         r.get("name", ""),
                "position":     str(r.get("position") or "").upper(),
                "overall_rank": int(r.get("overall_rank") or 999),
                "value_1qb":    float(r.get("rookie_value") or 0),
                "value_sf":     float(r.get("rookie_sf_value") or r.get("rookie_value") or 0),
            }
            for r in rows
            if r.get("position") in ("QB", "RB", "WR", "TE")
        ]
    except Exception as e:
        print(f"[rookie_rankings] skipped: {e}")
        return []


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
        "rookie_rankings": _load_rookie_rankings_for_ctx(),
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


@app.errorhandler(500)
def handle_500(e):
    logger.exception("[500] Internal server error")
    return (
        "<!doctype html><html><head><title>Error — BR Fantasy</title>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<style>body{font-family:sans-serif;background:#0f1623;color:#e2e8f0;"
        "display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}"
        ".box{text-align:center;padding:40px 24px;max-width:400px;}"
        "h2{margin:0 0 8px;font-size:22px;}p{color:#94a3b8;margin:0 0 24px;font-size:14px;}"
        "a{display:inline-block;padding:10px 20px;background:#3b82f6;color:#fff;"
        "border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;}</style>"
        "</head><body><div class='box'>"
        "<h2>Something went wrong</h2>"
        "<p>The server hit an unexpected error. This usually fixes itself — please try again in a moment.</p>"
        "<a href='/'>&#8592; Back to home</a>"
        "</div></body></html>"
    ), 500


@app.route("/health")
def health():
    """Uptime / readiness probe used by Render and load balancers."""
    from dashboard_services.db import get_database_url
    db_ok = False
    try:
        import psycopg
        url = get_database_url()
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception as exc:
        logger.warning("[health] DB check failed: %s", exc)

    payload = {"status": "ok" if db_ok else "degraded", "db": db_ok}
    status_code = 200 if db_ok else 503
    return jsonify(payload), status_code


@app.route("/api/history/ai-recap")
@limiter.limit("10 per minute")
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
            chips_html += f"<span class='chip chip-streak'><i class='fa-solid fa-fire'></i>{streak_chip}</span>"
        elif streak_chip and streak_frame_cls == "streak-cold":
            chips_html += f"<span class='chip chip-streak'><i class='fa-solid fa-snowflake'></i>{streak_chip}</span>"
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
            chips_html += f"<span class='chip chip-streak'><i class='fa-solid fa-fire'></i>{streak_chip}</span>"
        elif streak_chip and css_cls == "streak-cold":
            chips_html += f"<span class='chip chip-streak'><i class='fa-solid fa-snowflake'></i>{streak_chip}</span>"
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
            <h3 style="color:#dc2626;"><i class="fa-solid fa-fire"></i> {hottest['Streak']}</h3>
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
            <h3><i class="fa-solid fa-snowflake"></i> {coldest['Streak']}</h3>
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

    # Build pick-value lookup from WLS-derived table (overlays FantasyCalc/DynastyProcess)
    pick_by_key: Dict[str, float] = load_pick_value_table() or {}

    roster_cards = []

    for r in rosters:
        rid = str(r.get("roster_id"))
        team_name = roster_map.get(rid, f"Roster {rid}")
        player_ids = [str(pid) for pid in (r.get("players") or [])]
        roster_value = sum(values_by_id.get(pid, 0.0) for pid in player_ids)
        team_picks = picks_by_roster.get(rid, []) if isinstance(picks_by_roster, dict) else []
        pick_count = len(team_picks)

        # Add pick values to total roster value
        league_id_str = str(ctx.get("league_id") or "")
        roster_value += _team_pick_value(
            team_picks, pick_by_key,
            platform=platform, league_id=league_id_str, season=_safe_int(season, 0),
        )

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
                  <div class="os-snapshot-meta">Total value</div>
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
    
    # Calculate total draft capital across all rosters
    total_draft_capital = 0.0
    for roster in rosters:
        roster_id = str(roster.get("roster_id"))
        player_ids = [str(pid) for pid in (roster.get("players") or [])]
        roster_value = sum(values_by_id.get(pid, 0.0) for pid in player_ids)
        team_picks = picks_by_roster.get(roster_id, []) if isinstance(picks_by_roster, dict) else []
        pick_count = len(team_picks)
        
        # Add pick values to total roster value
        roster_value += _team_pick_value(team_picks, pick_by_key)
        
        total_draft_capital += roster_value

    rostered_ids = {
        str(pid)
        for r in rosters
        for pid in (r.get("players") or [])
    }

    # --- Waiver Recommendations: gather candidates ---
    # Build a set of sleeper_ids for this year's rookie class from DB.
    _rookie_sids: set[str] = set()
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class as _arc
        _ry = _arc()
        from dashboard_services.db import get_conn as _gc_w
        with _gc_w() as _wc:
            _rr = _wc.execute(
                "SELECT sleeper_id FROM rookie_prospects WHERE draft_class_year = %s AND sleeper_id IS NOT NULL",
                (_ry,),
            ).fetchall()
        _rookie_sids = {str(r["sleeper_id"]) for r in _rr if r["sleeper_id"]}
    except Exception:
        pass

    # Rookies are only waiver-eligible after the fantasy rookie draft is complete.
    # Detect by checking if any rookie from this year's class is already rostered.
    _rookie_draft_done = bool(_rookie_sids and any(sid in rostered_ids for sid in _rookie_sids))

    waiver_candidates = []
    for row in model_value_table:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or "")
        pos = str(row.get("position") or row.get("pos") or "").upper()
        if not pid or pid in rostered_ids:
            continue
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue
        if pid in _rookie_sids and not _rookie_draft_done:
            continue
        try:
            val = float(row.get("value") or 0.0)
        except Exception:
            val = 0.0
        if val <= 0:
            continue

        try:
            age = float(row.get("age") or 0)
        except Exception:
            age = 0.0

        rank_change = row.get("rank_change_7d")

        # Prioritize name from the current value table row, then fallback to players_index
        player_name = (
            row.get("name") or 
            players_index.get(pid, {}).get("name") or 
            f"Player {pid}"  # More informative fallback than "Unknown"
        )
        
        waiver_candidates.append({
            "player_id": pid,
            "name": player_name,
            "position": pos,
            "team": row.get("team") or players_index.get(pid, {}).get("team") or "",
            "value": val,
            "age": age,
            "pos_rank_label": row.get("pos_rank_label") or "",
            "rank_change_7d": rank_change,
        })

    # Bulk-fetch breakout scores for waiver candidates from DB
    waiver_breakout: dict = {}
    try:
        _db_url = os.getenv("DATABASE_URL", "").strip()
        if _db_url and not any(t in _db_url for t in ("USER", "PASSWORD", "HOST")):
            from dashboard_services.db import get_conn as _gc
            _pids = [c["player_id"] for c in waiver_candidates[:100]]
            if _pids:
                with _gc() as _conn:
                    with _conn.cursor() as _cur:
                        _cur.execute(
                            """
                            SELECT DISTINCT ON (player_id)
                                player_id,
                                breakout_opportunity_score
                            FROM breakout_opportunity_scores
                            WHERE player_id = ANY(%s)
                            ORDER BY player_id, as_of_date DESC
                            """,
                            (_pids,),
                        )
                        for _r in _cur.fetchall():
                            _r = dict(_r)
                            if _r.get("breakout_opportunity_score") is not None:
                                waiver_breakout[_r["player_id"]] = float(_r["breakout_opportunity_score"])
    except Exception:
        pass

    # Age primes by position (peak dynasty window)
    _prime_max = {"QB": 33, "RB": 26, "WR": 28, "TE": 29}

    def _waiver_pickup_score(c: dict) -> float:
        val = c["value"]
        age = c["age"] or 0
        pos = c["position"]
        rank_chg = c["rank_change_7d"] or 0
        bscore = waiver_breakout.get(c["player_id"], 0)
        prime = _prime_max.get(pos, 28)

        # Trend bonus: up to +60 for strong 7d movement
        trend_bonus = min(rank_chg * 4, 60) if rank_chg and rank_chg > 0 else 0
        # Breakout bonus: up to +50
        breakout_bonus = min(bscore * 0.5, 50)
        # Age bonus: peak age = +30, every year past prime = -10
        age_bonus = 30 - max(0, (age - prime) * 10) if age else 0

        return val + trend_bonus + breakout_bonus + age_bonus

    def _waiver_signal(c: dict) -> tuple[str, str]:
        """Return (badge_class, label) for the pickup signal."""
        rank_chg = c["rank_change_7d"] or 0
        age = c["age"] or 0
        pos = c["position"]
        bscore = waiver_breakout.get(c["player_id"], 0)
        prime = _prime_max.get(pos, 28)

        if bscore >= 55:
            return ("signal-breakout", "Breakout")
        if rank_chg >= 8:
            return ("signal-rising", "Rising Fast")
        if rank_chg >= 3:
            return ("signal-rising", "Trending Up")
        if age < prime - 2 and c["value"] >= 300:
            return ("signal-value", "Value Play")
        if age > prime + 2:
            return ("signal-aging", "Sell Window")
        return ("signal-hold", "Available")

    waiver_candidates.sort(key=_waiver_pickup_score, reverse=True)

    waiver_html = []
    for p in waiver_candidates[:10]:
        sub_bits = [p["position"]]
        if p["team"]:
            sub_bits.append(p["team"])
        if p["pos_rank_label"]:
            sub_bits.append(p["pos_rank_label"])
        if p["age"]:
            sub_bits.append(f"Age {p['age']:.1f}")
        subline = " • ".join(sub_bits)

        sig_cls, sig_label = _waiver_signal(p)

        rank_arrow = ""
        chg = p["rank_change_7d"]
        if chg and chg != 0:
            arrow_cls = "waiver-arrow-up" if chg > 0 else "waiver-arrow-down"
            arrow_sym = "▲" if chg > 0 else "▼"
            rank_arrow = f'<span class="{arrow_cls}">{arrow_sym}{abs(chg)}</span>'

        waiver_html.append(
            f"""
            <div class="os-waiver-row">
              <div class="os-waiver-main">
                <div class="os-waiver-name-row">
                  <span class="os-waiver-name player-clickable" style="cursor:pointer;font-weight:600;" data-player-id='{p['player_id']}' data-player-name='{p['name']}'>{p['name']}</span>
                  {rank_arrow}
                </div>
                <div class="os-waiver-sub">{subline}</div>
              </div>
              <div class="os-waiver-right">
                <span class="waiver-signal {sig_cls}">{sig_label}</span>
                <span class="os-waiver-value">{p['value']:.0f}</span>
              </div>
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
        <section class="os-card os-card-soft os-col-fill">
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

        <section class="os-card os-col-fill">
          <div class="os-section-head">
            <div class="os-section-head-content">
              <h2 class="os-section-title">Waiver Wire Targets</h2>
              <div class="os-section-subtitle">Smart pickup recommendations — value + trend + breakout potential</div>
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

    - Counts total assets (players + picks) on each side.
    - Gives a bonus to the side with FEWER total assets, scaled by:
        * gap in player value between the stud and the opponent's best player
        * how much of that side is concentrated in its best player ("stud")
        * how many extra pieces the other side is sending (including picks)
    - Adjustment is added on top of raw_total.
    """

    vals_a = side_a.get("player_values", []) or []
    vals_b = side_b.get("player_values", []) or []
    picks_a = len(side_a.get("pick_ids", []) or [])
    picks_b = len(side_b.get("pick_ids", []) or [])

    # Total assets (players + picks) on each side
    assets_a = len(vals_a) + picks_a
    assets_b = len(vals_b) + picks_b

    # No assets on either side, or same total asset count → no adjustment.
    if assets_a == 0 or assets_b == 0 or assets_a == assets_b:
        side_a["effective_total"] = side_a["raw_total"]
        side_b["effective_total"] = side_b["raw_total"]
        side_a["adjustment"] = 0.0
        side_b["adjustment"] = 0.0
        return

    # Decide which side is consolidating (fewer total assets)
    if assets_a < assets_b:
        fewer = side_a
        more = side_b
        fewer_is_a = True
    else:
        fewer = side_b
        more = side_a
        fewer_is_a = False

    fewer_vals = fewer.get("player_values", []) or []
    more_vals = more.get("player_values", []) or []

    # Player-only totals (picks excluded from gap calc)
    fewer_players_total = float(fewer.get("raw_players_total", 0.0) or 0.0)
    more_players_total = float(more.get("raw_players_total", 0.0) or 0.0)

    # Safety guard — need at least one player on the consolidating side
    if not fewer_vals or fewer_players_total <= 0:
        side_a["effective_total"] = side_a["raw_total"]
        side_b["effective_total"] = side_b["raw_total"]
        side_a["adjustment"] = 0.0
        side_b["adjustment"] = 0.0
        return

    # Extra pieces = total asset count difference (players + picks)
    extra_pieces = abs(assets_a - assets_b)

    # How big is the stud relative to the consolidating side?
    stud_val = max(fewer_vals)
    stud_share = stud_val / max(fewer_players_total, 1.0)  # 0–1
    stud_share = max(0.0, min(stud_share, 1.0))

    # Gap in player value between the two sides
    player_gap = abs(more_players_total - fewer_players_total)

    # --- Adjustment recipe ---
    # 1. Base from player_gap, scaled heavier when stud dominates the side.
    base_from_gap = player_gap * (0.35 + 0.45 * stud_share)

    # 2. Extra multiplier per extra piece: 1 extra ~0.55, 2 ~0.75, 3+ ~0.90
    piece_factor = 0.55 + 0.20 * min(extra_pieces - 1, 2)

    raw_adj = base_from_gap * piece_factor

    # 3. Caps: at most 80% of the stud value, or 55% of the consolidating
    #    side's total player value — whichever is smaller.
    cap_stud = 0.80 * stud_val
    cap_side = 0.55 * fewer_players_total
    adj_cap = max(0.0, min(cap_stud, cap_side))

    adj = min(raw_adj, adj_cap)

    # Apply to the consolidating (fewer-asset) side only
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
    if df_weekly.empty or "week" not in df_weekly.columns:
        return ""
    week_df = df_weekly[df_weekly["week"] == w].copy()

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
    if df_weekly.empty or "week" not in df_weekly.columns:
        return ""
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
    if df_weekly.empty or "week" not in df_weekly.columns:
        return ""
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

    Draft order = reverse of final overall standings:
      - Non-playoff teams: ordered by regular-season record (worst → slot 1)
      - Playoff teams: ordered by playoff finish (earliest eliminated → next slot,
        champion → last slot)

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
    league    = hist_ctx.get("league") or {}
    roster_map = hist_ctx.get("roster_map") or {}

    reg_team_stats = build_regular_season_team_stats(df_weekly, league)
    reg_team_stats = sort_team_stats(reg_team_stats)

    if reg_team_stats is None or reg_team_stats.empty:
        HISTORICAL_PICK_SLOT_CACHE[cache_key] = {}
        return {}

    # roster_map: {roster_id_str: team_name}
    name_to_roster_id: Dict[str, int] = {}
    for rid, team_name in roster_map.items():
        try:
            name_to_roster_id[str(team_name)] = int(rid)
        except Exception:
            continue

    # Regular-season ranks: {roster_id: rank_int}  (rank 1 = best regular-season team)
    reg_ranks: Dict[int, int] = {}
    for _, row in reg_team_stats.iterrows():
        owner = str(row.get("owner") or "")
        rank  = _safe_int(row.get("Rank"), 0)
        rid   = name_to_roster_id.get(owner)
        if rid is not None and rank > 0:
            reg_ranks[rid] = rank

    total_teams = len(reg_ranks) or len(reg_team_stats)

    # ---- Try to get playoff bracket for accurate final standings ----
    slot_map: Dict[int, int] = {}
    try:
        winners_bracket = get_bracket(platform, resolved_league_id, "winners", source_season) or []

        if winners_bracket:
            # Collect every roster_id that appears in the bracket as a direct integer
            playoff_rids: set[int] = set()
            for m in winners_bracket:
                for key in ("t1", "t2", "w", "l"):
                    v = m.get(key)
                    if isinstance(v, int) and v > 0:
                        playoff_rids.add(v)

            # Determine final placement from matchups that have a "p" field.
            # Sleeper sets p on the decisive matchup for each placement:
            #   winner → placement p, loser → placement p+1
            playoff_placements: Dict[int, int] = {}
            for m in winners_bracket:
                p = m.get("p")
                if p is None:
                    continue
                p = int(p)
                w = m.get("w")
                l = m.get("l")
                if isinstance(w, int) and w > 0:
                    playoff_placements[w] = p
                if isinstance(l, int) and l > 0:
                    playoff_placements[l] = p + 1

            if playoff_placements:
                # Non-playoff teams: assign slots 1…N ordered worst→best regular season
                non_playoff = sorted(
                    [(rid, rank) for rid, rank in reg_ranks.items() if rid not in playoff_rids],
                    key=lambda x: x[1],   # highest rank number = worst record
                    reverse=True,
                )
                # Playoff teams: assign next slots ordered by worst→best playoff finish
                playoff_ordered = sorted(
                    [(rid, place) for rid, place in playoff_placements.items()],
                    key=lambda x: x[1],   # highest placement number = worst finish
                    reverse=True,
                )

                slot = 1
                for rid, _ in non_playoff:
                    slot_map[rid] = slot
                    slot += 1
                for rid, _ in playoff_ordered:
                    slot_map[rid] = slot
                    slot += 1

    except Exception:
        pass  # fall through to regular-season-only fallback

    # ---- Fallback: regular-season standings only ----
    if not slot_map:
        for rid, rank in reg_ranks.items():
            slot_map[rid] = total_teams - rank + 1

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
            nonlocal trade_count, biggest_trade_label, biggest_trade_delta, season

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
                import json as _json
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

                yr = _safe_int(pick.get("season"), 0)
                rnd = _safe_int(pick.get("round"), 0)
                _pv = pick_values
                tier_vals = {
                    "early": float(_pv.get(f"{yr}_{rnd}_early") or _pv.get(f"{yr}_{rnd}") or 0),
                    "mid":   float(_pv.get(f"{yr}_{rnd}_mid")   or _pv.get(f"{yr}_{rnd}") or 0),
                    "late":  float(_pv.get(f"{yr}_{rnd}_late")  or _pv.get(f"{yr}_{rnd}") or 0),
                }
                pick_data = _json.dumps({
                    "label": pick_label,
                    "season": yr,
                    "round": rnd,
                    "value": round(val, 1),
                    "tiers": tier_vals,
                }, separators=(",", ":"))
                pick_data_attr = pick_data.replace('"', '&quot;')

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
            # Build data payload for outcome check (sent/received per team)
            trade_date_str = ""
            if pd.notna(txrow["ts"]):
                trade_date_str = txrow["ts"].strftime("%Y-%m-%d")

            outcome_data = []
            for tm in teams:
                rid = tm.get("roster_id")

                # Include players
                gets_pids = [{"id": str(p.get("pid") or ""), "name": str(p.get("name") or "")} for p in (tm.get("gets") or []) if p.get("pid")]
                sends_pids = [{"id": str(p.get("pid") or ""), "name": str(p.get("name") or "")} for p in (tm.get("sends") or []) if p.get("pid")]

                # Include picks with asset_type and pick details
                gets_picks = []
                for pick in picks_by_receiver.get(rid, []):
                    season = pick.get('season', '')
                    round_num = pick.get('round', '')
                    roster_id = pick.get('roster_id', '')

                    # Try to resolve exact slot from roster_id
                    exact_slot = None
                    if roster_id:
                        try:
                            exact_slot = resolve_exact_pick_slot(platform, resolved_league_id, int(season), pick)
                        except Exception:
                            pass

                    # Use exact slot if available, otherwise use roster_id as fallback
                    _rd_sfx = {1: "st", 2: "nd", 3: "rd"}.get(int(round_num or 0), "th")
                    if exact_slot:
                        pick_id = f"{season} {round_num}.{exact_slot:02d}"
                        display_name = pick_id
                        slot_value = exact_slot
                    else:
                        pick_id = f"{season} {round_num}.{roster_id}" if roster_id else f"{season} {round_num}.XX"
                        display_name = f"{season} {round_num}{_rd_sfx} Rd"
                        slot_value = None

                    gets_picks.append({
                        "id": pick_id,
                        "name": display_name,
                        "asset_type": "pick",
                        "pick_season": season,
                        "pick_round": round_num,
                        "pick_order": pick.get("order"),
                        "pick_slot": slot_value,
                    })

                sends_picks = []
                for pick in picks_by_sender.get(rid, []):
                    season = pick.get('season', '')
                    round_num = pick.get('round', '')
                    roster_id = pick.get('roster_id', '')

                    # Try to resolve exact slot from roster_id
                    exact_slot = None
                    if roster_id:
                        try:
                            exact_slot = resolve_exact_pick_slot(platform, resolved_league_id, int(season), pick)
                        except Exception:
                            pass

                    # Use exact slot if available, otherwise use roster_id as fallback
                    _rd_sfx = {1: "st", 2: "nd", 3: "rd"}.get(int(round_num or 0), "th")
                    if exact_slot:
                        pick_id = f"{season} {round_num}.{exact_slot:02d}"
                        display_name = pick_id
                        slot_value = exact_slot
                    else:
                        pick_id = f"{season} {round_num}.{roster_id}" if roster_id else f"{season} {round_num}.XX"
                        display_name = f"{season} {round_num}{_rd_sfx} Rd"
                        slot_value = None

                    sends_picks.append({
                        "id": pick_id,
                        "name": display_name,
                        "asset_type": "pick",
                        "pick_season": season,
                        "pick_round": round_num,
                        "pick_order": pick.get("order"),
                        "pick_slot": slot_value,
                    })

                # Combine players and picks
                all_gets = gets_pids + gets_picks
                all_sends = sends_pids + sends_picks

                outcome_data.append({"roster_id": rid, "team_name": tm.get("name", ""), "gets": all_gets, "sends": all_sends})

            import json as _json
            outcome_json = _json.dumps(outcome_data).replace('"', '&quot;')
            outcome_btn = (
                f"<button class='outcome-check-btn' "
                f"data-trade-teams='{outcome_json}' "
                f"data-trade-date='{trade_date_str}' "
                f"onclick='checkTradeOutcome(this)'>Check Outcome</button>"
            )
            outcome_result_id = f"outcome_{trade_count}"
            return (
                "<div class='tx trade-card activity-item' data-kind='trade'>"
                f"  <div class='meta'>{pill('Trade completed')} • {when}{outcome_btn}</div>"
                f"  <div class='teams'>{''.join(cols)}</div>"
                f"  <div id='{outcome_result_id}' class='trade-outcome-result' style='display:none;'></div>"
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

        <div class="card small" id="nflNewsCard" style="margin-top:12px;">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;">
            <h3>NFL News</h3>
            <span style="font-size:10px;color:var(--text-muted);font-weight:500;">via ESPN</span>
          </div>
          <div id="nflNewsList" class="card-body" style="padding:0;max-height:340px;overflow-y:auto;">
            <div style="padding:16px 14px;display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text-muted);"><div class="loading-spinner" style="width:14px;height:14px;margin:0;flex-shrink:0;"></div>Loading…</div>
          </div>
        </div>
      </aside>
    </div>

    <script>
    (function() {{
      function loadNflNews() {{
        var list = document.getElementById('nflNewsList');
        if (!list) return;
        fetch('/api/nfl-news?limit=12')
          .then(function(r) {{ return r.json(); }})
          .then(function(data) {{
            var items = data.news || [];
            if (!items.length) {{
              list.innerHTML = '<div style="padding:12px 14px;font-size:13px;color:var(--text-muted);">No news available.</div>';
              return;
            }}
            list.innerHTML = items.map(function(n) {{
              var linkOpen = n.url ? '<a href="' + n.url + '" target="_blank" rel="noopener" class="act-news-link">' : '<span>';
              var linkClose = n.url ? '</a>' : '</span>';
              return '<div class="act-news-item">' +
                '<div class="act-news-headline">' + linkOpen + n.headline + linkClose + '</div>' +
                (n.description ? '<div class="act-news-desc">' + n.description + '</div>' : '') +
                '<div class="act-news-meta">' + [n.source, n.age].filter(Boolean).join(' · ') + '</div>' +
              '</div>';
            }}).join('');
          }})
          .catch(function() {{ /* fail silently */ }});
      }}

      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', loadNflNews);
      }} else {{
        loadNflNews();
      }}
    }})();
    </script>

    <style>
      .act-news-item {{
        padding: 10px 14px;
        border-bottom: 1px solid var(--border);
      }}
      .act-news-item:last-child {{ border-bottom: none; }}
      .act-news-headline {{ font-size: 12px; font-weight: 600; color: var(--text); line-height: 1.4; margin-bottom: 3px; }}
      .act-news-link {{ color: var(--text); text-decoration: none; }}
      .act-news-link:hover {{ text-decoration: underline; color: #3b82f6; }}
      .act-news-desc {{ font-size: 11px; color: var(--text-muted); line-height: 1.35; margin-bottom: 3px; }}
      .act-news-meta {{ font-size: 10px; color: var(--text-muted); opacity: .7; }}

      .bract-summary-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
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
        font-size: 12px;
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


def _avg_pick_value_for_round(by_id: dict, season: int, rnd: int) -> float:
    """Average model value of all picks matching season + round prefix."""
    prefix = f"{season}_{rnd}_"
    vals = [v for k, v in by_id.items() if k.startswith(prefix)]
    return (sum(vals) / len(vals)) if vals else 0.0


def _team_pick_value(
    picks: list,
    by_id: dict,
    platform: str = None,
    league_id: str = None,
    season: int = None,
) -> float:
    """
    Total model value of a team's draft picks.

    When platform/league_id/season are provided, resolves the exact draft
    slot for each pick by looking up the original owner's previous-season
    standings (via resolve_exact_pick_slot). Falls back to the round-average
    when the slot cannot be determined (e.g. picks 2+ years out).
    """
    total = 0.0
    for pk in picks:
        try:
            pk_season = int(pk.get("season") or 0)
            rnd = int(pk.get("round") or 0)
        except (TypeError, ValueError):
            continue

        exact_slot = None
        if platform and league_id and season and pk.get("original_owner"):
            pick_for_slot = {
                "season": pk_season,
                "round": rnd,
                "previous_owner_id": pk.get("original_owner"),
            }
            try:
                exact_slot = resolve_exact_pick_slot(platform, league_id, season, pick_for_slot)
            except Exception:
                pass

        if exact_slot:
            key = f"{pk_season}_{rnd}_{exact_slot:02d}"
            val = by_id.get(key)
            if val is not None:
                total += val
                continue

        # Fall back to round average when exact slot is unknown
        total += _avg_pick_value_for_round(by_id, pk_season, rnd)
    return total


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
    picks_by_roster = ctx.get("picks_by_roster") or {}
    league_id = str(ctx.get("league_id") or "")
    current_season = _safe_int((ctx.get("league") or {}).get("season"), datetime.now().year)
    
    viewer = ctx.get("viewer") or {}
    viewer_roster_id = viewer.get("viewer_roster_id")

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

    # ----------------- Compute per-team draft capital value -----------------
    pick_by_key: Dict[str, float] = load_pick_value_table() or {}
    team_pick_value: Dict[int, float] = {}
    for r in rosters:
        rid = r.get("roster_id")
        if rid is None:
            continue
        team_pick_value[int(rid)] = _team_pick_value(
            picks_by_roster.get(str(rid), []), pick_by_key,
            platform=platform, league_id=league_id, season=current_season,
        )

    pick_series = list(team_pick_value.values())
    _pick_mean = sum(pick_series) / len(pick_series) if pick_series else 0.0
    _pick_var = sum((v - _pick_mean) ** 2 for v in pick_series) / len(pick_series) if pick_series else 0.0
    _pick_std = math.sqrt(_pick_var)
    team_pick_z: Dict[int, float] = {
        rid: ((v - _pick_mean) / _pick_std if _pick_std > 0 else 0.0)
        for rid, v in team_pick_value.items()
    }
    pick_z_min = min(team_pick_z.values()) if team_pick_z else 0.0
    pick_z_max = max(team_pick_z.values()) if team_pick_z else 0.0

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

    # Pre-compute global chart Y-max so all team cards share the same Y-axis scale
    _chart_all_pos_vals = []
    for _rid in team_meta:
        _chart_all_pos_vals.extend([
            sum(team_pos_values[_rid].get("QB", [])),
            sum(team_pos_values[_rid].get("RB", [])),
            sum(team_pos_values[_rid].get("WR", [])),
            sum(team_pos_values[_rid].get("TE", [])),
            team_pick_value.get(_rid, 0.0),
        ])
    _chart_y_max = round(max(_chart_all_pos_vals) * 1.15, 1) if _chart_all_pos_vals else 100.0

    # Pre-compute roster grades for all teams
    from dashboard_services.ai.context_builders import calculate_roster_grade as _calc_grade

    _n_teams = len(team_meta)

    def _grade_for_roster(r_id: int) -> dict:
        roster_obj = next((r for r in rosters if r.get("roster_id") == r_id), {})
        flat_players = []
        for pid in roster_obj.get("players") or []:
            row = by_id.get(str(pid))
            if not row:
                continue
            pos = str(row.get("position") or row.get("pos") or "").upper()
            if pos not in CORE_POS:
                continue
            val = float(row.get("value") or 0.0)
            nm = str(row.get("name") or "").strip().lower()
            age = name_to_age.get(nm)
            flat_players.append({"position": pos, "value": val, "age": age})
        flat_players.sort(key=lambda x: x["value"], reverse=True)
        picks = picks_by_roster.get(str(r_id), [])
        p_ranks = {pos: pos_rank[pos].get(r_id, _n_teams) for pos in POS_ORDER}
        return _calc_grade(flat_players, picks, position_ranks=p_ranks, num_teams=_n_teams)

    team_grades = {rid: _grade_for_roster(rid) for rid in team_meta}

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

        # Draft Capital row
        pick_val = team_pick_value.get(rid, 0.0)
        pick_z = team_pick_z.get(rid, 0.0)
        if pick_z_max > pick_z_min:
            pick_pct = 10 + 80 * (pick_z - pick_z_min) / (pick_z_max - pick_z_min)
        else:
            pick_pct = 50.0
        pick_count = len(picks_by_roster.get(str(rid), []))
        table_rows.append(
            "<tr class='pos-row pos-picks-row'>"
            "  <td class='pos-name'>"
            "    <i class='fa-solid fa-clipboard-list' style='font-size:11px;opacity:0.7;'></i> PICKS"
            "  </td>"
            f"  <td class='pos-count'>{pick_count}</td>"
            f"  <td class='pos-total'>{pick_val:.1f}</td>"
            "  <td class='pos-avg'>—</td>"
            f"  <td class='pos-z'>{pick_z:+.2f}</td>"
            "  <td class='pos-bar-cell'>"
            "    <div class='pos-bar-outer'>"
            f"      <div class='pos-bar-inner' style='width:{pick_pct:.0f}%;background:var(--color-pick,#8b5cf6);'></div>"
            "    </div>"
            "  </td>"
            "  <td class='pos-rank'></td>"
            "</tr>"
        )

        # ── Position value bar chart ──────────────────────────────────────────
        _chart_labels  = ["QB", "RB", "WR", "TE", "Picks"]
        _chart_colors  = ["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#c92c68"]
        _chart_values  = [
            round(sum(team_pos_values[rid].get("QB", [])), 1),
            round(sum(team_pos_values[rid].get("RB", [])), 1),
            round(sum(team_pos_values[rid].get("WR", [])), 1),
            round(sum(team_pos_values[rid].get("TE", [])), 1),
            round(team_pick_value.get(rid, 0.0), 1),
        ]
        _chart_div_id  = f"teamValueChart_{rid}"
        _chart_data    = json.dumps([{
            "type":          "bar",
            "x":             _chart_labels,
            "y":             _chart_values,
            "marker":        {"color": _chart_colors},
            "hovertemplate": "%{x}: %{y:,.0f}<extra></extra>",
        }])
        _chart_layout  = json.dumps({
            "margin":       {"t": 8, "b": 28, "l": 44, "r": 8},
            "paper_bgcolor":"rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "height":       200,
            "yaxis": {
                "range":      [0, _chart_y_max],
                "tickformat": ".2s",
                "showgrid":   True,
                "gridcolor":  "rgba(100,116,139,0.2)",
                "zeroline":   False,
                "tickfont":   {"size": 11},
            },
            "xaxis": {"showgrid": False, "tickfont": {"size": 12}},
            "showlegend":   False,
            "bargap":       0.3,
        })
        _chart_html = (
            f"<div id='{_chart_div_id}' class='team-value-chart'></div>"
            f"<script>(function(){{"
            f"  var d={_chart_data},l={_chart_layout};"
            f"  function createChart(){{"
            f"    if(typeof Plotly!=='undefined'){{"
            f"      Plotly.newPlot('{_chart_div_id}',d,l,{{responsive:true,displayModeBar:false}});"
            f"    }} else {{"
            f"      setTimeout(createChart, 100);"
            f"    }}"
            f"  }}"
            f"  if(document.readyState==='loading'){{"
            f"    document.addEventListener('DOMContentLoaded', createChart);"
            f"  }} else {{"
            f"    createChart();"
            f"  }}"
            f"}})();</script>"
        )

        _gdata = team_grades.get(rid, {})
        _grade = _gdata.get("grade", "?")
        _win_window = _gdata.get("win_window", "")
        _grade_cls = "grade-a" if _grade.startswith("A") else "grade-b" if _grade.startswith("B") else "grade-c" if _grade.startswith("C") else "grade-d"
        _grade_badge = f"<span class='roster-grade-inline {_grade_cls}' title='{_win_window}'>{_grade}</span>"

        # Numeric sort keys for client-side sorting
        _grade_num = {"A+":10,"A":9,"B+":8,"B":7,"C+":6,"C":5,"D+":4,"D":3,"F":2}.get(_grade, 1)
        _archetype_num = {"Win-Now Window":1,"Contender Window":2,"Aging Contender":3,
                          "2-3 Year Window":4,"Rising Contender":5,"Building":6,
                          "Retooling":7,"Holding Pattern":8,"Full Rebuild":9}.get(_win_window, 5)
        _pos_idx = team_pos_index[rid]

        card_html = (
            f"<div class='card team-strength-card' data-sort-grade='{_grade_num}' data-sort-posindex='{_pos_idx:.4f}' data-sort-archetype='{_archetype_num}'>"
            "  <div class='card-header-row'>"
            f"    <div style='display:flex;align-items:center;gap:8px;'>{img_html}<h2 class='team-clickable' style='cursor:pointer;' data-roster-id='{rid}' data-team-name='{name}'>{name}</h2>{_grade_badge}</div>"
            f"    <div class='mini-label'><span class='grade-window-label'>{_win_window}</span> &bull; Positional Index: "
            f"<span style='font-weight:600'>{team_pos_index[rid]:+.2f}</span></div>"
            "  </div>"
            "  <div class='card-body'>"
            f"    {_chart_html}"
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

    # ---------- League analytics section (lazy-loaded) ----------
    platform_js = platform
    season_js = current_season

    # Detect league type (sf vs 1qb) from roster positions
    _rp = get_roster_positions()
    _rp_list = list(_rp) if _rp else []
    _is_sf = any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in _rp_list)
    _league_type_js = "sf" if _is_sf else "1qb"
    _league_size_js = int(len(rosters)) if rosters else 10

    _offseason_mode_js = bool(ctx.get("offseason_mode", False))
    _draft_ended_js = has_draft_ended(league_id, platform, current_season)

    analytics_html = f"""
    <div class="card teams-analytics-card" id="teamsAnalyticsCard">
      <div class="card-tabs">
        <div class="tab-strip" id="teamsAnalyticsTabs">
          <button class="tab-btn active" data-tab="btm">Value</button>
          <button class="tab-btn" data-tab="roster-intel">Roster Intel</button>
          <button class="tab-btn" data-tab="power-rankings">Power Rankings</button>
          <button class="tab-btn" data-tab="sos" id="sosTabBtn" style="display:none">Schedule</button>
          <button class="tab-btn" data-tab="draft" id="draftTabBtn" style="display:none">Draft</button>
          <div class="tab-panels">
            <div class="tab-panel active" data-tab="btm" id="btmPanel">
              <div class="analytics-loading">Loading…</div>
            </div>
            <div class="tab-panel" data-tab="roster-intel" id="rosterIntelPanel">
              <div class="analytics-loading">Loading…</div>
            </div>
            <div class="tab-panel" data-tab="power-rankings" id="powerRankingsPanel">
              <div class="analytics-loading">Loading…</div>
            </div>
            <div class="tab-panel" data-tab="sos" id="sosPanel">
              <div class="analytics-loading">Loading…</div>
            </div>
            <div class="tab-panel" data-tab="draft" id="draftPanel">
              <div class="analytics-loading">Loading…</div>
            </div>
        </div>
      </div>
    </div>
    <script>
    (function() {{
      var _platform        = {repr(platform_js)};
      var _leagueId        = {repr(league_id)};
      var _season          = {season_js};
      var _leagueType      = {repr(_league_type_js)};
      var _leagueSize      = {_league_size_js};
      var _viewerRosterId  = {repr(str(viewer_roster_id or ''))};
      var _offseasonMode   = {str(_offseason_mode_js).lower()};
      var _draftEnded      = {str(_draft_ended_js).lower()};
      var _loaded          = {{}};


      function loadBtm() {{
        if (_loaded.btm) return;
        _loaded.btm = true;
        var panel = document.getElementById('btmPanel');
        if (!panel) return;
        var slim = !!panel.closest('.teams-sidebar');

        function fmtDate(isoStr) {{
          if (!isoStr) return '';
          var d = new Date(isoStr + 'T00:00:00');
          return d.toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
        }}

        function renderBtm(data, days) {{
          if (data.error) {{
            panel.innerHTML = '<p class="analytics-empty">' + data.error + '</p>';
            return;
          }}
          var rows = data.rosters || [];
          var avgDelta = data.league_avg_delta || 0;
          var avgSign = avgDelta >= 0 ? '+' : '';
          var avgFmt = avgSign + Math.round(avgDelta).toLocaleString();
          var html = '';

          // Header: title + window pills (full mode only)
          if (!slim) {{
            html += '<div class="btm-header">' +
              '<div class="btm-header-text">' +
                '<span class="btm-title">Value Tracker</span>' +
                '<span class="btm-subtitle">Which rosters gained the most dynasty value?</span>' +
              '</div>' +
              '<div class="btm-window-pills">' +
                '<button class="btm-pill' + (days === 7  ? ' active' : '') + '" data-days="7">7d</button>' +
                '<button class="btm-pill' + (days === 14 ? ' active' : '') + '" data-days="14">14d</button>' +
                '<button class="btm-pill' + (days === 30 ? ' active' : '') + '" data-days="30">30d</button>' +
                '<button class="btm-pill' + (days === 60 ? ' active' : '') + '" data-days="60">60d</button>' +
              '</div>' +
            '</div>';
          }}

          // Meta: date range + league avg
          html += '<div class="btm-meta">' +
            '<span class="btm-date-range">' + fmtDate(data.baseline_date) + ' – ' + fmtDate(data.latest_date) + '</span>' +
            '<span class="btm-league-avg">League Avg: <strong>' + avgFmt + '</strong></span>' +
          '</div>';

          // Column header
          html += '<div class="btm-col-header' + (slim ? ' btm-slim' : '') + '">' +
            '<span></span>' +
            '<span>Team</span>' +
            '<span style="text-align:right;">' + days + 'd</span>' +
            '<span style="text-align:right;">vs Avg</span>' +
          '</div>';

          // Rows
          html += '<div class="btm-rows">';
          rows.forEach(function(r, idx) {{
            var pos    = r.vs_avg >= 0;
            var cls    = pos ? 'btm-pos' : 'btm-neg';
            var pdSign = r.total_delta >= 0 ? '+' : '';
            var vsSign = pos ? '+' : '';

            var rankHtml;
            if      (idx === 0) rankHtml = '<span class="btm-rank-badge rk-gold">1</span>';
            else if (idx === 1) rankHtml = '<span class="btm-rank-badge rk-silver">2</span>';
            else if (idx === 2) rankHtml = '<span class="btm-rank-badge rk-bronze">3</span>';
            else                rankHtml = '<span class="btm-rank-num">' + (idx + 1) + '</span>';

            var moversHtml = '';
            if (!slim && r.top_movers && r.top_movers.length) {{
              moversHtml = '<div class="btm-movers-row">';
              r.top_movers.slice(0, 4).forEach(function(m) {{
                var mc       = m.delta >= 0 ? 'btm-mover-pos' : 'btm-mover-neg';
                var arrow    = m.delta >= 0 ? '↑' : '↓';
                var lastName = m.name.split(' ').slice(-1)[0];
                var dFmt     = (m.delta >= 0 ? '+' : '') + Math.round(m.delta);
                moversHtml  += '<span class="btm-mover ' + mc + '" title="' + m.name + ' · ' + m.position + '">' +
                  arrow + ' <strong>' + lastName + '</strong>&nbsp;' + dFmt +
                '</span>';
              }});
              moversHtml += '</div>';
            }}

            html += '<div class="btm-row ' + cls + (slim ? ' btm-slim' : '') + '">' +
              '<div class="btm-rank-cell">' + rankHtml + '</div>' +
              '<div class="btm-team-cell">' +
                '<div class="btm-team-name">' + r.team_name + '</div>' +
                moversHtml +
              '</div>' +
              '<div class="btm-change-cell">' +
                '<div class="btm-change-num ' + cls + '">' + pdSign + Math.round(r.total_delta).toLocaleString() + '</div>' +
              '</div>' +
              '<div class="btm-vsavg-cell">' +
                '<span class="btm-vsavg-badge ' + cls + '">' + vsSign + Math.round(r.vs_avg).toLocaleString() + '</span>' +
              '</div>' +
            '</div>';
          }});
          html += '</div>';

          panel.innerHTML = html;

          panel.querySelectorAll('.btm-pill').forEach(function(btn) {{
            btn.addEventListener('click', function() {{
              fetchBtm(parseInt(this.getAttribute('data-days')));
            }});
          }});
        }}

        function fetchBtm(days) {{
          panel.innerHTML = '<div class="analytics-loading">Loading…</div>';
          fetch('/api/beat-the-market?platform=' + _platform +
                '&league_id=' + _leagueId + '&season=' + _season +
                '&league_type=' + _leagueType + '&league_size=' + _leagueSize + '&days=' + days)
            .then(function(r) {{ return r.json(); }})
            .then(function(data) {{ renderBtm(data, days); }})
            .catch(function() {{ panel.innerHTML = '<p class="analytics-empty">Could not load data.</p>'; }});
        }}

        fetchBtm(30);
      }}

      function loadSos() {{
        if (_loaded.sos) return;
        _loaded.sos = true;
        var panel = document.getElementById('sosPanel');
        if (!panel) return;
        fetch('/api/schedule-strength?platform=' + _platform +
              '&league_id=' + _leagueId + '&season=' + _season)
          .then(r => r.json())
          .then(data => {{
            if (data.error) {{ panel.innerHTML = '<p class="analytics-empty">' + data.error + '</p>'; return; }}
            var teams = data.teams || [];
            if (!teams.length) {{ panel.innerHTML = '<p class="analytics-empty">No schedule data available.</p>'; return; }}
            var usingPR = data.using_power_rankings;
            var maxOpp = Math.max(...teams.map(t => t.avg_opp_points), 1);
            var wr = data.weeks_remaining || 0;
            var sortLabel = usingPR ? 'Based on roster strength (no games played yet)' : 'Sorted by avg opponent score (hardest first)';
            var html = '<div class="analytics-btm-header"><span class="analytics-date-label">Weeks remaining: ' + wr +
              '</span><span class="analytics-avg-label">' + sortLabel + '</span></div>';
            if (usingPR) {{
              html += '<p class="analytics-empty" style="margin:4px 0 8px;font-size:12px;color:var(--text-muted)">No games played — opponent strength estimated from roster values.</p>';
            }}
            html += '<div class="analytics-bar-list">';
            teams.forEach(function(t) {{
              var pct = Math.min(100, Math.round(t.avg_opp_points / maxOpp * 100));
              var cls = t.avg_opp_points >= maxOpp * 0.75 ? 'analytics-bar-neg' :
                        t.avg_opp_points <= maxOpp * 0.5  ? 'analytics-bar-pos' : 'analytics-bar-mid';
              var valLabel = usingPR ? '' : t.avg_opp_points.toFixed(1);
              html += '<div class="analytics-bar-row">' +
                '<span class="analytics-bar-name">' + t.team_name + '</span>' +
                '<div class="analytics-bar-track">' +
                  '<div class="analytics-bar-fill ' + cls + '" style="width:' + pct + '%"></div>' +
                '</div>' +
                '<span class="analytics-bar-val">' + valLabel + '</span>' +
              '</div>';
            }});
            html += '</div>';
            panel.innerHTML = html;
          }})
          .catch(function() {{ panel.innerHTML = '<p class="analytics-empty">Could not load data.</p>'; }});
      }}

      function loadDraft() {{
        if (_loaded.draft) return;
        _loaded.draft = true;
        var panel = document.getElementById('draftPanel');
        if (!panel) return;
        panel.innerHTML = '<div class="analytics-loading">Loading…</div>';
        fetch('/api/draft-grades?platform=' + _platform +
              '&league_id=' + _leagueId + '&season=' + _season + '&league_type=' + _leagueType)
          .then(function(r) {{ return r.json(); }})
          .then(function(data) {{
            if (data.error) {{ panel.innerHTML = '<p class="analytics-empty">' + data.error + '</p>'; return; }}
            var teams = data.teams || [];
            if (!teams.length) {{ panel.innerHTML = '<p class="analytics-empty">No draft data available.</p>'; return; }}

            var numTeams    = data.num_teams || 10;
            var totalRounds = data.total_rounds || 1;

            // Build team name lookup: roster_id -> team_name
            var teamNames = {{}};
            teams.forEach(function(t) {{ teamNames[t.roster_id] = t.team_name; }});

            // Flatten all picks across all teams (for By Round view)
            var allPicks = [];
            teams.forEach(function(t) {{
              t.picks.forEach(function(p) {{
                allPicks.push(Object.assign({{}}, p, {{ _team_name: t.team_name }}));
              }});
            }});
            allPicks.sort(function(a,b) {{ return a.pick_no - b.pick_no; }});

            var chevronSvg = '<svg class="draft-acc-chevron" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6l4 4 4-4"/></svg>';

            // ── Shared pick row renderer ─────────────────────────────────────
            function renderPickRow(p, showTeamName) {{
              var pgcls = 'dg-' + p.grade.replace('+', 'plus');

              var adpLine = '';
              if (p.avg_pick != null) {{
                var diff = p.adp_diff;
                var diffHtml = diff > 1
                  ? '<span class="adp-value">+' + diff.toFixed(1) + ' picks ahead</span>'
                  : diff < -1
                    ? '<span class="adp-reach">' + diff.toFixed(1) + '</span>'
                    : '<span class="adp-neutral">on ADP</span>';
                var posTag = p.pos_rank != null ? ' · ' + p.position + p.pos_rank : '';
                var waitTag = p.could_wait ? ' <span class="adp-wait">Reach</span>' : '';
                adpLine = '<div class="analytics-pick-adp-line">ADP ' + p.avg_pick.toFixed(2) + posTag + ' ' + diffHtml + waitTag + '</div>';
              }}

              var bpaLine = '';
              if (p.bpa && p.bpa.length) {{
                var bpaNames = p.bpa.map(function(b) {{
                  var posRank = b.pos_rank != null ? b.pos_rank : '';
                  // Show first-initial + last name so "Isaiah Likely" renders as "I. Likely"
                  var parts = (b.name || '').split(' ');
                  var suffixRe = /^(jr\.?|sr\.?|ii|iii|iv|v)$/i;
                  var suffix = parts.length > 1 && suffixRe.test(parts[parts.length - 1]) ? ' ' + parts[parts.length - 1] : '';
                  var coreParts = suffix ? parts.slice(0, -1) : parts;
                  var displayName = coreParts.length > 1
                    ? coreParts[0][0] + '. ' + coreParts[coreParts.length - 1] + suffix
                    : b.name;
                  return '<span class="bpa-name pos-' + (b.position || '').toLowerCase() + '">' +
                    displayName + ' (' + (b.position || '') + posRank + ')</span>';
                }}).join(' ');
                bpaLine = '<div class="analytics-pick-bpa">Available: ' + bpaNames + '</div>';
              }}

              var needBadge = p.need ? ' <span class="draft-need-badge">Need</span>' : '';
              var teamTag = showTeamName
                ? '<span class="draft-pick-team-tag">' + (p._team_name || '') + '</span>'
                : '';

              return '<div class="analytics-pick-row">' +
                '<span class="analytics-pick-num">#' + p.pick_no + '</span>' +
                '<span class="analytics-pick-grade ' + pgcls + '">' + p.grade + '</span>' +
                '<div class="analytics-pick-info">' +
                  '<div class="analytics-pick-name">' + p.name +
                    ' <span class="analytics-pick-pos pos-' + (p.position || '').toLowerCase() + '">' + (p.position || '') + '</span>' +
                    needBadge + teamTag +
                  '</div>' +
                  adpLine +
                  bpaLine +
                '</div>' +
              '</div>';
            }}

            // ── Build "By Team" accordion HTML ───────────────────────────────
            function buildByTeamHtml() {{
              var html = '<div class="draft-accordion">';
              teams.forEach(function(t, idx) {{
                var gcls = 'dg-' + t.grade.replace('+', 'plus');
                html += '<div class="draft-acc-item' + (idx === 0 ? ' open' : '') + '">' +
                  '<button class="draft-acc-header" type="button">' +
                    '<span class="draft-acc-name">' + t.team_name + '</span>' +
                    '<div class="draft-acc-right">' +
                      '<span class="draft-acc-grade ' + gcls + '">' + t.grade + '</span>' +
                      chevronSvg +
                    '</div>' +
                  '</button>' +
                  '<div class="draft-acc-body"><div class="draft-acc-picks">';
                t.picks.forEach(function(p) {{ html += renderPickRow(p, false); }});
                html += '</div></div></div>';
              }});
              html += '</div>';
              return html;
            }}

            // ── By Round state & renderer ────────────────────────────────────
            var currentRound = 1;
            var roundContainerId = 'draftRoundView_' + Date.now();

            function buildByRoundHtml(round) {{
              var roundPicks = allPicks.filter(function(p) {{ return p.round === round; }});
              var ordinals = ['','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th',
                              '11th','12th','13th','14th','15th'];
              var label = (ordinals[round] || (round + 'th')) + ' Round';

              var prevDis = round <= 1 ? ' disabled' : '';
              var nextDis = round >= totalRounds ? ' disabled' : '';

              var html = '<div class="draft-round-nav">' +
                '<button class="draft-round-btn"' + prevDis + ' id="draftRoundPrev">&#8592; Prev</button>' +
                '<span class="draft-round-label">' + label + '</span>' +
                '<button class="draft-round-btn"' + nextDis + ' id="draftRoundNext">Next &#8594;</button>' +
              '</div>' +
              '<div class="draft-acc-picks">';

              if (!roundPicks.length) {{
                html += '<p class="analytics-empty" style="padding:12px;">No picks recorded for this round yet.</p>';
              }} else {{
                roundPicks.forEach(function(p) {{ html += renderPickRow(p, true); }});
              }}
              html += '</div>';
              return html;
            }}

            function renderRoundView(container, round) {{
              container.innerHTML = buildByRoundHtml(round);
              var prev = container.querySelector('#draftRoundPrev');
              var next = container.querySelector('#draftRoundNext');
              if (prev) prev.addEventListener('click', function() {{
                if (currentRound > 1) {{ currentRound--; renderRoundView(container, currentRound); }}
              }});
              if (next) next.addEventListener('click', function() {{
                if (currentRound < totalRounds) {{ currentRound++; renderRoundView(container, currentRound); }}
              }});
            }}

            // ── Wire up tabs and render ──────────────────────────────────────
            var tabsHtml =
              '<div class="draft-view-tabs">' +
                '<button class="draft-view-tab active" data-view="team">By Team</button>' +
                '<button class="draft-view-tab" data-view="round">By Round</button>' +
              '</div>' +
              '<div id="draftTeamView">' + buildByTeamHtml() + '</div>' +
              '<div id="draftRoundView" style="display:none;"></div>';

            panel.innerHTML = tabsHtml;

            // Accordion toggle for By Team view
            panel.querySelectorAll('.draft-acc-header').forEach(function(btn) {{
              btn.addEventListener('click', function() {{
                var item = this.closest('.draft-acc-item');
                var wasOpen = item.classList.contains('open');
                panel.querySelectorAll('.draft-acc-item').forEach(function(el) {{
                  el.classList.remove('open');
                }});
                if (!wasOpen) item.classList.add('open');
              }});
            }});

            // Tab switching
            var roundViewEl = panel.querySelector('#draftRoundView');
            var teamViewEl  = panel.querySelector('#draftTeamView');
            var roundRendered = false;
            panel.querySelectorAll('.draft-view-tab').forEach(function(tab) {{
              tab.addEventListener('click', function() {{
                panel.querySelectorAll('.draft-view-tab').forEach(function(t) {{ t.classList.remove('active'); }});
                this.classList.add('active');
                if (this.dataset.view === 'round') {{
                  teamViewEl.style.display = 'none';
                  roundViewEl.style.display = '';
                  if (!roundRendered) {{
                    roundRendered = true;
                    renderRoundView(roundViewEl, currentRound);
                  }}
                }} else {{
                  roundViewEl.style.display = 'none';
                  teamViewEl.style.display = '';
                }}
              }});
            }});
          }})
          .catch(function() {{ panel.innerHTML = '<p class="analytics-empty">Could not load data.</p>'; }});
      }}

      function loadRosterIntel() {{
        if (_loaded.rosterIntel) return;
        _loaded.rosterIntel = true;
        var panel = document.getElementById('rosterIntelPanel');
        if (!panel) return;
        fetch('/api/roster-intel?platform=' + _platform +
              '&league_id=' + _leagueId + '&season=' + _season +
              '&league_type=' + _leagueType)
          .then(r => r.json())
          .then(data => {{
            if (data.error) {{ panel.innerHTML = '<p class="analytics-empty">' + data.error + '</p>'; return; }}
            var teams = data.teams || [];
            if (!teams.length) {{ panel.innerHTML = '<p class="analytics-empty">No roster data available.</p>'; return; }}

            // Show only the logged-in user's team; fall back to all teams if not known
            if (_viewerRosterId) {{
              teams = teams.filter(function(t) {{ return String(t.roster_id) === String(_viewerRosterId); }});
            }}

            var sigColor = {{
              'Core':           '#22c55e',
              'Hold — Breakout':'#f59e0b',
              'Sell High':      '#ef4444',
              'Buy Window':     '#3b82f6',
              'Hold':           'var(--text-muted)',
              'Cut':            '#94a3b8',
            }};
            var sigBg = {{
              'Core':           '#dcfce7',
              'Hold — Breakout':'#fef3c7',
              'Sell High':      '#fee2e2',
              'Buy Window':     '#dbeafe',
              'Hold':           'var(--row)',
              'Cut':            'var(--row)',
            }};

            var sigOrder = {{'Sell High':0,'Cut':1,'Hold — Breakout':2,'Buy Window':3,'Core':4,'Hold':5}};
            var html = '';
            teams.forEach(function(t) {{
              // Show everything except plain Hold — sorted by urgency
              var actionPlayers = t.players
                .filter(function(p) {{ return p.signal !== 'Hold'; }})
                .sort(function(a, b) {{
                  return (sigOrder[a.signal] ?? 9) - (sigOrder[b.signal] ?? 9);
                }});
              if (!actionPlayers.length) return;
              html += '<div class="ri-team-block">' +
                '<div class="ri-team-name">' + t.team_name + '</div>';
              actionPlayers.slice(0, 8).forEach(function(p) {{
                var chgHtml = '';
                if (p.rank_change_7d && p.rank_change_7d !== 0) {{
                  var sym = p.rank_change_7d > 0 ? '▲' : '▼';
                  var col = p.rank_change_7d > 0 ? '#22c55e' : '#ef4444';
                  chgHtml = '<span style="font-size:10px;color:' + col + ';margin-left:4px;">' + sym + Math.abs(p.rank_change_7d) + '</span>';
                }}
                html += '<div class="ri-player-row">' +
                  '<div class="ri-player-info">' +
                    '<span class="ri-player-name">' + p.name + chgHtml + '</span>' +
                    '<span class="ri-player-meta">' + p.position + (p.pos_rank_label ? ' · ' + p.pos_rank_label : '') + '</span>' +
                  '</div>' +
                  '<span class="ri-signal" style="background:' + (sigBg[p.signal]||'var(--row)') + ';color:' + (sigColor[p.signal]||'var(--text-muted)') + '">' + p.signal + '</span>' +
                '</div>';
              }});
              html += '</div>';
            }});

            var emptyMsg = _viewerRosterId
              ? 'Your roster looks stable — no urgent actions flagged.'
              : 'All rosters look stable — no urgent actions flagged.';
            panel.innerHTML = html || '<p class="analytics-empty">' + emptyMsg + '</p>';
          }})
          .catch(function() {{ panel.innerHTML = '<p class="analytics-empty">Could not load data.</p>'; }});
      }}

      function loadPowerRankings() {{
        if (_loaded.powerRankings) return;
        _loaded.powerRankings = true;
        var panel = document.getElementById('powerRankingsPanel');
        if (!panel) return;
        fetch('/api/power-rankings', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{platform: _platform, league_id: _leagueId, season: _season}})
        }})
          .then(r => r.json())
          .then(data => {{
            if (!data.success) {{ panel.innerHTML = '<p class="analytics-empty">' + (data.error || 'Failed to load.') + '</p>'; return; }}
            panel.innerHTML = data.html || '<p class="analytics-empty">No rankings available.</p>';
          }})
          .catch(function() {{ panel.innerHTML = '<p class="analytics-empty">Could not load power rankings.</p>'; }});
      }}

      // Show Schedule/Draft tabs conditionally
      (function() {{
        var sosBtn = document.getElementById('sosTabBtn');
        var draftBtn = document.getElementById('draftTabBtn');
        // Schedule: visible when not in pure offseason (in-season or preseason)
        if (sosBtn && !_offseasonMode) sosBtn.style.display = '';
        // Draft: visible once the draft has occurred
        if (draftBtn && _draftEnded) draftBtn.style.display = '';
      }})();

      // Wire data-loading onto the tab buttons; visibility is handled by initCardTabs
      function wireAnalyticsTabs() {{
        var tabs = document.querySelectorAll('#teamsAnalyticsTabs > .tab-btn');
        tabs.forEach(function(btn) {{
          btn.addEventListener('click', function() {{
            var tab = btn.dataset.tab;
            if (tab === 'btm')             loadBtm();
            if (tab === 'roster-intel')    loadRosterIntel();
            if (tab === 'power-rankings')  loadPowerRankings();
            if (tab === 'sos')             loadSos();
            if (tab === 'draft')           loadDraft();
          }});
        }});
        loadBtm();  // load first tab immediately
      }}

      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', wireAnalyticsTabs);
      }} else {{
        wireAnalyticsTabs();
      }}
    }})();
    </script>
    """

    # ---------- Page shell ----------
    return f"""
    <div class="page-layout teams-page">
      <main class="page-main">
        <div class="teams-sort-bar">
          <span style="font-size:12px;color:var(--text-muted);margin-right:8px;">Sort by:</span>
          <button class="teams-sort-btn active" data-sort="posindex">Positional Index</button>
          <button class="teams-sort-btn" data-sort="grade">Team Grade</button>
          <button class="teams-sort-btn" data-sort="archetype">Archetype</button>
        </div>
        <div class="teams-grid" id="teamsGrid">
          {all_cards_html}
        </div>
      </main>

      <aside class="page-sidebar teams-sidebar">
        {analytics_html}
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

      // Teams sort bar
      var _sortKey = 'posindex';
      function sortTeams(key) {{
        _sortKey = key;
        var grid = document.getElementById('teamsGrid');
        if (!grid) return;
        var cards = Array.from(grid.querySelectorAll('.team-strength-card'));
        cards.sort(function(a, b) {{
          if (key === 'grade') {{
            return Number(b.dataset.sortGrade) - Number(a.dataset.sortGrade);
          }} else if (key === 'archetype') {{
            return Number(a.dataset.sortArchetype) - Number(b.dataset.sortArchetype);
          }} else {{
            // posindex: higher is better
            return Number(b.dataset.sortPosindex) - Number(a.dataset.sortPosindex);
          }}
        }});
        cards.forEach(function(c) {{ grid.appendChild(c); }});
        document.querySelectorAll('.teams-sort-btn').forEach(function(btn) {{
          btn.classList.toggle('active', btn.dataset.sort === key);
        }});
      }}
      document.querySelectorAll('.teams-sort-btn').forEach(function(btn) {{
        btn.addEventListener('click', function() {{ sortTeams(btn.dataset.sort); }});
      }});
      // Default sort on load
      sortTeams('posindex');
    }})();
    </script>
    """


@app.route("/sw.js")
def service_worker():
    """Serve service worker from root scope so it can control all pages."""
    return send_file("static/sw.js", mimetype="application/javascript")


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
              <div class="static-section-title">Trade Analytics</div>
              <p>
                When you enter your Sleeper username, your connected league IDs and Sleeper
                user ID may be used to improve trade value accuracy across the platform.
                This data is not sold or shared with third parties.
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
@app.route("/<platform>/<int:season>/<league_id>/contact", methods=["GET", "POST"])
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


def league_url(slug: str, league_id: Optional[str] = None, platform: Optional[str] = None, season: Optional[int] = None) -> str:
    """
    Build a URL that keeps league context if we have one.
    slug examples: 'faq', 'privacy', 'support', 'contact'
    """
    # Build base URL
    base_url = ""
    
    # Add platform if provided
    if platform:
        base_url += f"/{platform}"
    
    # Add season if provided
    if season:
        base_url += f"/{season}"
    
    # Add league_id if provided
    if league_id:
        base_url += f"/{league_id}"
    
    return f"{base_url}/{slug}"


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


@app.route("/api/trade-count")
def api_trade_count():
    """Get the count of trades from trade_intel_trades table."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trade_intel_trades")
            count = cursor.fetchone()[0]
        return jsonify({"count": count})
    except Exception as e:
        # Return fallback count if table doesn't exist or other error
        return jsonify({"count": 15000})


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
        viewer = get_viewer_session_for_league(ctx.get("users") or [], ctx.get("rosters") or [])
        viewer_roster_id = viewer.get("viewer_roster_id") or ""
        body = build_trade_calculator_body(league_id_safe, season_safe, num_teams=num_teams,
                                           scoring_format=scoring_format,
                                           viewer_roster_id=viewer_roster_id)
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


# ── Tour mock data helpers ─────────────────────────────────────────────────

_TOUR_MOCK_TEAMS = [
    "Dynasty Kings", "Gridiron Ghosts", "Blitz Brigade",
    "Redzone Rebels", "Endzone Elite", "Pocket Protectors",
]

def _build_tour_mock_df_weekly() -> pd.DataFrame:
    """Seeded, deterministic weekly scores for 6 mock teams over 13 weeks."""
    import random as _rand
    rng = _rand.Random(42)
    rows = []
    for week in range(1, 14):
        pairs = [(0, 1), (2, 3), (4, 5)] if week % 2 == 0 else [(0, 2), (1, 4), (3, 5)]
        for mid, (a, b) in enumerate(pairs, start=week * 10):
            rows += [
                {
                    "week": week, "matchup_id": mid,
                    "owner": _TOUR_MOCK_TEAMS[a],
                    "points": round(rng.gauss(98, 12), 2),
                    "finalized": True,
                },
                {
                    "week": week, "matchup_id": mid,
                    "owner": _TOUR_MOCK_TEAMS[b],
                    "points": round(rng.gauss(95, 11), 2),
                    "finalized": True,
                },
            ]
    return pd.DataFrame(rows)


def _build_tour_mock_graphs_ctx() -> dict:
    """Minimal ctx for build_graphs_body with pre-computed team_stats."""
    df = _build_tour_mock_df_weekly()
    mock_league: dict = {"settings": {"playoff_week_start": 14}}
    team_stats = build_regular_season_team_stats(df, mock_league)
    team_stats = sort_team_stats(team_stats)

    if not team_stats.empty and "PF" in team_stats.columns:
        pf_z = (team_stats["PF"] - team_stats["PF"].mean()) / max(float(team_stats["PF"].std()), 1.0)
        win_z = (team_stats["Win%"] - team_stats["Win%"].mean()) / max(float(team_stats["Win%"].std()), 1.0)
        avg_z = (team_stats["AVG"] - team_stats["AVG"].mean()) / max(float(team_stats["AVG"].std()), 1.0)
        team_stats["PowerScore"] = 0.30 * pf_z + 0.40 * win_z + 0.30 * avg_z
        # Z-score columns required by z_better_outward
        for col in ["PF", "PA", "MAX", "MIN", "AVG", "STD"]:
            zc = f"Z_{col}"
            if col in team_stats.columns:
                col_vals = team_stats[col]
                std_val = float(col_vals.std())
                team_stats[zc] = (col_vals - col_vals.mean()) / max(std_val, 1.0)

    return {"team_stats": team_stats, "df_weekly": df}


def _build_tour_mock_history_ctx() -> dict:
    """Minimal history context for build_history_body."""
    df = _build_tour_mock_df_weekly()
    return {
        "platform": "sleeper",
        "season": 2024,
        "league_id": "tour_mock",
        "resolved_league_id": "tour_mock",
        "league": {
            "name": "Demo League",
            "league_id": "tour_mock",
            "settings": {"playoff_week_start": 14},
        },
        "df_weekly": df,
        "roster_map": {},
        "users": {},
        "rosters": [],
        "offseason_mode": False,
    }


@app.route("/<platform>/<int:season>/<league_id>/graphs")
def page_graphs(platform: str, season: int, league_id: str):
    # Tour preview: render with mock data, bypass real league fetch
    if request.args.get("tour"):
        try:
            mock_ctx = _build_tour_mock_graphs_ctx()
            body_html = build_graphs_body(mock_ctx)
        except Exception as exc:
            body_html = (
                f"<div class='card central'><div class='card-body'>"
                f"<p>Graphs preview unavailable: {exc}</p></div></div>"
            )
        return render_page("BR Fantasy Graphs", league_id, "graphs", body_html, platform, season)

    ctx = get_league_ctx_from_cache(platform, league_id, season)
    offseason = bool(ctx.get("offseason_mode"))
    available_seasons = get_available_history_seasons(platform, league_id, season)

    # Default view: career during offseason, current season in-season
    default_view = "career" if offseason else str(season)
    view = request.args.get("view", default_view)

    # Build the season-selector dropdown (navigate via URL query param)
    graphs_base_url = url_for("page_graphs", platform=platform, season=season, league_id=league_id)
    selector_opts = []
    selector_opts.append(
        f"<option value='{graphs_base_url}?view=career' "
        f"{'selected' if view == 'career' else ''}>Career (all seasons)</option>"
    )
    if not offseason:
        selector_opts.append(
            f"<option value='{graphs_base_url}?view={season}' "
            f"{'selected' if view == str(season) else ''}>{season} (current)</option>"
        )
    for s in available_seasons:
        selector_opts.append(
            f"<option value='{graphs_base_url}?view={s}' "
            f"{'selected' if view == str(s) else ''}>{s}</option>"
        )

    season_selector_html = f"""
    <div class="graphs-season-selector">
      <label class="graphs-season-label">View:</label>
      <select class="graphs-season-select" onchange="window.location.href=this.value">
        {"".join(selector_opts)}
      </select>
    </div>"""

    # ── Render the appropriate graphs ──────────────────────────────────────
    if view == "career":
        if not available_seasons:
            charts_html = """
            <div class="card central">
              <div class="card-body">
                <p style="color:var(--text-muted);">
                  Career graphs appear after your first completed season.
                </p>
              </div>
            </div>"""
        else:
            try:
                # Build career ctx from all available seasons
                career_ctx = _build_career_graphs_ctx_live(platform, league_id, season, available_seasons)
                charts_html = build_career_graphs_body(career_ctx)
            except Exception as exc:
                import traceback; traceback.print_exc()
                charts_html = (
                    f"<div class='card central'><div class='card-body'>"
                    f"<p>Career graphs unavailable: {exc}</p></div></div>"
                )
    else:
        target_season = int(view) if view.isdigit() else season
        if target_season == season and not offseason:
            season_ctx = ctx
        else:
            rid = resolve_league_id_for_season(platform, league_id, season, target_season)
            season_ctx = get_league_ctx_from_cache(platform, rid, target_season)

        if season_ctx.get("offseason_mode") or season_ctx.get("df_weekly", pd.DataFrame()).empty:
            charts_html = f"""
            <div class="card central">
              <div class="card-body">
                <p style="color:var(--text-muted);">
                  No weekly data available for {target_season}.
                  Select another season or choose Career view.
                </p>
              </div>
            </div>"""
        else:
            charts_html = build_graphs_body(season_ctx)

    body_html = season_selector_html + charts_html
    return render_page("BR Fantasy Graphs", league_id, "graphs", body_html, platform, season)


def _build_career_graphs_ctx_live(
    platform: str, league_id: str, season: int, available_seasons: list
) -> dict:
    """Aggregate team_stats and df_weekly across all completed seasons for career graphs."""
    career: dict = {}  # owner -> {Wins, Losses, Ties, PF, PA, weekly_pts, season_pf}
    season_pf_rows: list = []  # rows for per-season bar chart: {season, owner, pf}

    for hist_s in available_seasons:
        rid = resolve_league_id_for_season(platform, league_id, season, hist_s)
        try:
            hctx = get_league_ctx_from_cache(platform, rid, hist_s)
        except Exception:
            continue

        df = hctx.get("df_weekly", pd.DataFrame())
        if df.empty or "owner" not in df.columns:
            continue

        mock_lg = hctx.get("league") or {}
        ts = build_regular_season_team_stats(df, mock_lg)

        for _, row in ts.iterrows():
            owner = str(row.get("owner", "?"))
            if owner not in career:
                career[owner] = {
                    "Wins": 0, "Losses": 0, "Ties": 0,
                    "PF": 0.0, "PA": 0.0, "weekly_pts": [],
                }
            career[owner]["Wins"]   += int(row.get("Wins", 0))
            career[owner]["Losses"] += int(row.get("Losses", 0))
            career[owner]["Ties"]   += int(row.get("Ties", 0))
            career[owner]["PF"]     += float(row.get("PF", 0))
            career[owner]["PA"]     += float(row.get("PA", 0))
            season_pf_rows.append({"season": hist_s, "owner": owner, "pf": float(row.get("PF", 0))})

        sub = df[df["finalized"] == True] if "finalized" in df.columns else df
        for owner, grp in sub.groupby("owner"):
            career.setdefault(str(owner), {
                "Wins": 0, "Losses": 0, "Ties": 0, "PF": 0.0, "PA": 0.0, "weekly_pts": [],
            })["weekly_pts"].extend(grp["points"].tolist() if "points" in grp else [])

    # Build career team_stats DataFrame
    stat_rows = []
    for owner, d in career.items():
        pts = d["weekly_pts"]
        games = d["Wins"] + d["Losses"] + d["Ties"]
        stat_rows.append({
            "owner": owner,
            "Wins": d["Wins"],
            "Losses": d["Losses"],
            "Ties": d["Ties"],
            "PF": d["PF"],
            "PA": d["PA"],
            "AVG": d["PF"] / games if games > 0 else 0.0,
            "Win%": d["Wins"] / games if games > 0 else 0.0,
            "MAX": max(pts) if pts else 0.0,
            "MIN": min(pts) if pts else 0.0,
            "STD": float(pd.Series(pts).std()) if len(pts) > 1 else 0.0,
        })

    team_stats = pd.DataFrame(stat_rows) if stat_rows else pd.DataFrame()

    if not team_stats.empty and "PF" in team_stats.columns:
        # Approximate PowerScore for career (win% + avg PF rank)
        pf_z  = (team_stats["PF"]  - team_stats["PF"].mean())  / max(float(team_stats["PF"].std()),  1.0)
        win_z = (team_stats["Win%"]- team_stats["Win%"].mean()) / max(float(team_stats["Win%"].std()), 1.0)
        avg_z = (team_stats["AVG"] - team_stats["AVG"].mean())  / max(float(team_stats["AVG"].std()),  1.0)
        team_stats["PowerScore"] = 0.30 * pf_z + 0.40 * win_z + 0.30 * avg_z
        for col in ["PF", "PA", "MAX", "MIN", "AVG", "STD"]:
            if col in team_stats.columns:
                cv = team_stats[col]
                sd = max(float(cv.std()), 1.0)
                team_stats[f"Z_{col}"] = (cv - cv.mean()) / sd

    # Combined df_weekly (with season column) for box/line charts
    combined_dfs = []
    for hist_s in available_seasons:
        rid = resolve_league_id_for_season(platform, league_id, season, hist_s)
        try:
            hctx = get_league_ctx_from_cache(platform, rid, hist_s)
            df = hctx.get("df_weekly", pd.DataFrame()).copy()
            if not df.empty and "owner" in df.columns:
                df["season"] = hist_s
                combined_dfs.append(df)
        except Exception:
            continue

    df_combined = pd.concat(combined_dfs, ignore_index=True) if combined_dfs else pd.DataFrame()
    season_pf_df = pd.DataFrame(season_pf_rows) if season_pf_rows else pd.DataFrame()

    return {
        "team_stats": team_stats,
        "df_weekly": df_combined,
        "season_pf_df": season_pf_df,
        "is_career": True,
    }


@app.route("/players")
@app.route("/<platform>/<int:season>/<league_id>/players")
def page_players(platform: str = None, season: int = None, league_id: str = None):
    """Player Rankings page — searchable, filterable, sortable list of all players."""
    body_html = """
    <div class="card central">
      <div class="card-header">
        <h2>Player Rankings</h2>
        <div style="font-size: 14px; color: var(--text-muted); margin-top: 4px;">
          All players ranked by dynasty value.
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
                           color:var(--text-muted);font-size:13px;pointer-events:none;"><i class="fa-solid fa-magnifying-glass"></i></span>
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
                League️ Settings
              </button>

              <!-- Settings panel (hidden by default) -->
              <div id="prSettingsPanel" class="filter-settings-panel" style="display:none;">
                <div class="settings-section">
                  <span class="settings-section-label">Scoring Type</span>
                  <div class="settings-toggle-group">
                    <button class="settings-toggle active" data-value="dynasty" onclick="prSetScoringType('dynasty')">Dynasty</button>
                    <button class="settings-toggle" data-value="redraft" disabled title="Coming soon">Redraft </button>
                  </div>
                </div>
                <div class="settings-section">
                  <span class="settings-section-label">League Format</span>
                  <div class="settings-toggle-group">
                    <button class="settings-toggle active" data-value="1qb" onclick="prSetLeagueType('1qb')">1QB</button>
                    <button class="settings-toggle" data-value="sf" onclick="prSetLeagueType('sf')">SF</button>
                  </div>
                </div>
                <div class="settings-section" id="prSizeSection">
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
            <div id="prActiveSettings" class="active-settings-indicator">
              <span class="active-setting-tag">10-Team</span>
              <span class="active-setting-tag">1QB</span>
              <span class="active-setting-tag">Dynasty</span>
            </div>
            <!-- Sort dropdown -->
            <div class="filter-sort">
              <label class="filter-label">Sort by</label>
              <select id="prSort" onchange="prRender()"
                style="padding:7px 10px;border-radius:8px;border:1px solid var(--border);
                       background:var(--card-bg);color:var(--text);font-size:12px;cursor:pointer;outline:none;min-height:34px;">
                <option value="rank">Overall Rank</option>
                <option value="value">Value</option>
                <option value="age">Age</option>
                <option value="pos_rank">Pos Rank</option>
              </select>
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
             grid-template-columns:34px 20px 1fr 64px 50px 50px 64px;
             gap:0;padding:6px 12px;border-radius:6px;
             background:var(--accent-soft);font-size:11px;
             font-weight:700;color:var(--accent);letter-spacing:0.04em;
             text-transform:uppercase;" class="pr-grid-row">
          <span>#</span>
          <span style="text-align:center;"></span>
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
          <div style="font-size:24px;margin-bottom:8px;opacity:0.4;"><i class="fa-solid fa-magnifying-glass"></i></div>
          No players match your filters
        </div>

      </div>
    </div>

    <style>
      .pr-grid-row {
        display: grid;
        grid-template-columns: 25px 25px 1fr 64px 50px 50px 64px;
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
        display: flex;
        align-items: center;
        gap: 3px;
      }
      .pr-rank-arrow {
        font-size: 16px;
        font-weight: 700;
        line-height: 1;
      }
      .pr-rank-arrow.up   { color: #22c55e; }
      .pr-rank-arrow.down { color: #ef4444; }
      .pr-arrows {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 10px;
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
        justify-content: space-between;
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
          justify-content: flex-start;
          gap: 5px;
        }
        .pos-pill {
          padding: 6px 10px;
          font-size: 11px;
        }
        .filter-label {
          white-space: nowrap;
        }
        .active-settings-indicator {
          justify-content: center;
          order: -1;
          width: 100%;
        }
        .filter-row-secondary {
          flex-wrap: wrap;
          gap: 8px;
        }
        .filter-sort,
        .filter-sort select {
          width: 100%;
        }
      }
    </style>

    <script>
      var prAllPlayers = [];
      var prIndicators = {};
      var prLeagueType   = '1qb';
      var prLeagueSize   = 10;
      var prScoringType  = 'dynasty';  // 'dynasty' | 'redraft'
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
        if (prScoringType === 'redraft') {
          const base = Number(prLeagueType === 'sf'
            ? (p.redraft_value_sf ?? p.redraft_value_1qb ?? 0)
            : (p.redraft_value_1qb ?? 0));
          return Math.round(base * 10) / 10;
        }
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

      function prIsElite(id) {
        return prIndicators.elites && prIndicators.elites.includes(String(id));
      }

      function prIsProspect(id) {
        return prIndicators.prospects && prIndicators.prospects.includes(String(id));
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

        const tags = indicator.querySelectorAll('.active-setting-tag');
        if (tags[0]) tags[0].textContent = prLeagueSize + '-Team';
        if (tags[1]) tags[1].textContent = prLeagueType.toUpperCase();
        if (tags[2]) tags[2].textContent = prScoringType === 'redraft' ? 'Redraft' : 'Dynasty';
      }

      function prSetScoringType(type) {
        prScoringType = type;
        // Update panel toggles
        document.querySelectorAll('#prSettingsPanel .settings-toggle[data-value]').forEach(btn => {
          const section = btn.closest('.settings-section');
          if (section && section.querySelector('.settings-section-label').textContent.includes('Scoring')) {
            btn.classList.toggle('active', btn.getAttribute('data-value') === type);
          }
        });
        // Hide league-size in redraft (size doesn't affect redraft values)
        const sizeSection = document.getElementById('prSizeSection');
        if (sizeSection) sizeSection.style.display = type === 'redraft' ? 'none' : '';
        // Hide PICK and ROOKIE filters in redraft
        document.querySelectorAll('.pos-pill[data-pos="PICK"], .pos-pill[data-pos="ROOKIE"]').forEach(btn => {
          btn.style.display = type === 'redraft' ? 'none' : '';
        });
        if (type === 'redraft' && (prPosFilters.has('PICK') || prPosFilters.has('ROOKIE'))) {
          prPosFilters.clear();
        }
        updateSettingsIndicator();
        prRender();
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

        // In redraft mode exclude picks and rookies (no redraft value), and
        // only show players who actually have a redraft value.
        if (prScoringType === 'redraft') {
          players = players.filter(p => {
            if (p.position === 'PICK' || p.is_rookie) return false;
            const v = prLeagueType === 'sf'
              ? (p.redraft_value_sf ?? p.redraft_value_1qb)
              : p.redraft_value_1qb;
            return v != null && Number(v) > 0;
          });
        }

        // Position filter (multi-select)
        const isDrafted = p => p.is_rookie && p.team && p.team !== 'FA';
        if (prPosFilters.has('ROOKIE')) {
          players = players.filter(p => p.is_rookie);
        } else if (prPosFilters.size > 0) {
          players = players.filter(p => prPosFilters.has(p.position) && (!p.is_rookie || isDrafted(p)));
        } else {
          players = players.filter(p => !p.is_rookie || isDrafted(p));
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
          row.className = 'pr-player-row pr-grid-row';
          row.style.cursor = 'pointer';
          row.onclick = function(e) {
            e.stopPropagation();
            const _drafted = p.is_rookie && p.team && p.team !== 'FA';
            if (p.is_rookie && !_drafted) {
              if (typeof rkOpenModal === 'function') {
                rkOpenModal(p);
              } else {
                openProspectModal(p.id, p.name || 'Unknown');
              }
            } else {
              openPlayerModal(p.id, p.name || 'Unknown');
            }
          };

          const _drafted = p.is_rookie && p.team && p.team !== 'FA';
          const displayRank = (p.position === 'PICK' || (p.is_rookie && !_drafted)) ? '' : (rankMap.get(String(p.id)) || (idx + 1));
          const posRank = prLeagueType === 'sf'
            ? (p.sf_pos_rank_label || p.pos_rank_label || p.position)
            : (p.pos_rank_label || p.position);
          const age = p.age != null ? Number(p.age).toFixed(1) : '—';
          const val = prGetValue(p);

          let badges = '';
          if (prIsRookie(p.id)) badges += '<span class="player-badge player-badge-prospect"><i class="fa-solid fa-seedling" aria-hidden="true"></i> PROSPECT</span>';
          else if (p.is_rookie) badges += '<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>';
          if (!p.is_rookie && prIsProspect(p.id)) badges += '<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>';
          if (prIsBreakout(p.id)) badges += '<span class="player-badge player-badge-breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i> BREAKOUT</span>';

          const rankChange = p.rank_change_7d;
                    let rankArrow = '';
          if (rankChange != null && rankChange !== 0) {
            const dir = rankChange > 0 ? 'up' : 'down';
            const icon = rankChange > 0 ? 'fa-chevron-up' : 'fa-chevron-down';
            rankArrow = `<span class="pr-rank-arrow ${dir}" title="${Math.abs(rankChange)} spot${Math.abs(rankChange)!==1?'s':''} in 7 days"><i class="fa-solid ${icon}" aria-hidden="true"></i></span>`;
            // Debug: Check if Font Awesome is loaded
            if (window.debugFA === undefined) {
              setTimeout(() => {
                const testIcon = document.querySelector('.fa-arrow-up, .fa-arrow-down');
                if (testIcon) {
                  const styles = window.getComputedStyle(testIcon);
                  console.log('Icon font-family:', styles.fontFamily);
                  console.log('Icon display:', styles.display);
                }
                window.debugFA = true;
              }, 1000);
            }
          }

          row.innerHTML =
            '<span class="pr-rank">'  + (displayRank ? '#' + displayRank : '—') + '</span>' +
            '<span class="pr-arrows">' + rankArrow + '</span>' +
            '<span class="pr-name player-clickable">'  + (p.name || 'Unknown') + badges + '</span>' +
            '<span class="pr-pos-cell">' + posRank + '</span>' +
            '<span class="pr-age">'   + (p.position === 'PICK' ? '—' : age) + '</span>' +
            '<span class="pr-team">'  + (p.team || '—') + '</span>' +
            '<span class="pr-value">' + prFormatValue(val) + '</span>';

          list.appendChild(row);
        });
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
            redraft_value_1qb: p.redraft_value_1qb != null ? Number(p.redraft_value_1qb) : null,
            redraft_value_sf:  p.redraft_value_sf  != null ? Number(p.redraft_value_sf)  : null,
            pos_rank_label:   p.pos_rank_label    || '',
            sf_pos_rank_label:p.sf_pos_rank_label || '',
            pos_rank:         Number(p.pos_rank    || 9999),
            sf_pos_rank:      Number(p.sf_pos_rank || 9999),
            search_name:      p.search_name || '',
            is_rookie:        p.is_rookie === true,
            rank_change_7d:   p.rank_change_7d != null ? Number(p.rank_change_7d) : null,
          }))
          .filter(p => ['QB','RB','WR','TE','PICK'].includes(p.position) || p.is_rookie)
          
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
    if platform:
        return render_page("Player Rankings", league_id, "players", body_html, platform, season)

    return render_page("Player Rankings", active="players", body_html=body_html)


@app.route("/<platform>/<int:season>/<league_id>/prospects")
def page_prospects(platform: str, season: int, league_id: str):
    """Rookie prospect rankings page — active class auto-detected."""
    from dashboard_services.pages.rookies_page import build_prospects_body
    body_html = build_prospects_body(platform, season, league_id)
    return render_page("Prospect Rankings", league_id, "prospects", body_html, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/breakouts")
def page_breakouts(platform: str, season: int, league_id: str):
    """Dedicated page for breakout candidates with detailed projections."""
    body_html = f"""
    <div class="card central">
      <div class="card-header">
        <h2>Breakout Engine</h2>
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
          <div style="font-size: 24px; margin-bottom: 12px; opacity:0.4;"><i class="fa-solid fa-chart-bar"></i></div>
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
          const iconClass = breakoutType.icon_class || 'fa-chart-bar';
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
            <div class="breakout-card" style="cursor:pointer;" onclick="openBreakoutModal('` + pid + `')">
              <div class="breakout-card-header">
                <div>
                  <div class="breakout-player-name">` + name + `</div>
                  <div class="breakout-player-meta">${{age}} • ${{team}} • ${{pos}}</div>
                </div>
                <div class="breakout-score-badge" style="background: ${{scoreColor}};">
                  ${{score}}
                </div>
              </div>

              <div class="breakout-card-body">
                <!-- Breakout Type Badge -->
                <div class="breakout-type-badge" style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: var(--card-bg); border-radius: 6px; margin-bottom: 12px; border: 1px solid var(--border-color);">
                  <i class="fa-solid ${{iconClass}}" style="font-size:18px;width:20px;text-align:center;"></i>
                  <span style="font-weight: 500; flex: 1;">${{label}}</span>
                  <span style="font-size: 12px; color: var(--text-muted); text-transform: uppercase;">${{driver}} driven</span>
                </div>

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
    </script>
    """
    return render_page("Breakout Engine", league_id, "breakouts", body_html, platform, season)


# Guest-accessible versions of content pages (no league required)
@app.route("/players")
def page_players_guest():
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_players(platform="sleeper", season=current_season, league_id=None)



@app.route("/breakouts")
def page_breakouts_guest():
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_breakouts(platform="sleeper", season=current_season, league_id=None)


@app.route("/<platform>/<int:season>/<league_id>/trade-intel")
def page_trade_intel(platform: str, season: int, league_id: str):
    body_html = f"""
    <div class="card central" style="max-width:960px;">
      <div class="card-header" style="border-bottom:1px solid var(--border-color);padding-bottom:16px;margin-bottom:0;">
        <h2 style="margin:0 0 4px;font-size:20px;">Trade Intelligence</h2>
        <div style="font-size:13px;color:var(--text-muted);">
          Actionable insights from thousands of real dynasty trades across multiple platforms
        </div>
      </div>
      <div class="card-body" style="padding-top:20px;">

        <div class="ti-controls">
          <div class="ti-tabs">
            <button class="ti-tab active" data-tab="trending" onclick="switchTITab('trending')"><i class="fa-solid fa-fire"></i> Trending</button>
            <button class="ti-tab" data-tab="buylows"  onclick="switchTITab('buylows')"><i class="fa-solid fa-arrow-trend-down"></i> Buy Low</button>
            <button class="ti-tab" data-tab="sellhigh" onclick="switchTITab('sellhigh')"><i class="fa-solid fa-arrow-trend-up"></i> Sell High</button>
          </div>
          <div class="ti-pos-filters">
            <button class="ti-pos active" data-pos="ALL" onclick="filterTI('ALL')">All</button>
            <button class="ti-pos" data-pos="QB"  onclick="filterTI('QB')">QB</button>
            <button class="ti-pos" data-pos="RB"  onclick="filterTI('RB')">RB</button>
            <button class="ti-pos" data-pos="WR"  onclick="filterTI('WR')">WR</button>
            <button class="ti-pos" data-pos="TE"  onclick="filterTI('TE')">TE</button>
          </div>
        </div>

        <div class="ti-key">
          <div class="ti-key-item">
            <span class="ti-key-swatch" style="background:#3b82f6;opacity:.7;border-radius:3px;"></span>
            <span><span class="ti-key-label">Market</span> Real Trade-weighted Median Value</span>
          </div>
          <div class="ti-key-item">
            <span class="ti-key-swatch" style="background:#8b5cf6;opacity:.7;border-radius:3px;"></span>
            <span><span class="ti-key-label">BR Model</span> BR Production Model Value</span>
          </div>
          <div class="ti-key-item">
            <span class="ti-key-swatch ti-key-delta"></span>
            <span><span class="ti-key-label">Delta</span> Market minus BR Model</span>
          </div>
          <div class="ti-key-item">
            <span style="display:inline-flex;align-items:center;vertical-align:middle;">
              <span style="width:8px;height:8px;border-radius:50%;color:#10b981;display:flex;align-items:center;line-height:1;">▲</span>
              <span style="width:8px;height:8px;border-radius:50%;color:#ef4444;display:inline-block;line-height:1;">▼</span>
            </span>
            <span><span class="ti-key-label">Momentum</span> Rising or Falling Market Price</span>
          </div>
        </div>

        <div id="tiPagination" class="ti-pagination" style="display:none;">
          <div class="ti-pagination-info">
            <span id="tiPaginationText">Showing 1-20 of 100 players</span>
          </div>
          <div class="ti-pagination-controls">
            <button id="tiPrevBtn" class="ti-pagination-btn" onclick="loadTIPage('prev')" disabled>
              <i class="fa-solid fa-chevron-left"></i> Previous
            </button>
            <div id="tiPageNumbers" class="ti-page-numbers"></div>
            <button id="tiNextBtn" class="ti-pagination-btn" onclick="loadTIPage('next')" disabled>
              Next <i class="fa-solid fa-chevron-right"></i>
            </button>
          </div>
        </div>

        <div id="tiLoading" style="text-align:center;padding:48px 0;color:var(--text-muted);">
          <div class="loading-spinner" style="margin:0 auto 12px;"></div>
          Loading trade data...
        </div>
        <div id="tiEmpty" style="display:none;text-align:center;padding:48px 0;color:var(--text-muted);">
          No data for this filter yet — analytics need to run to populate this view.
        </div>
        <div id="tiGrid" class="ti-grid" style="display:none;"></div>

      </div>
    </div>

    <style>
      .ti-controls {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
        flex-wrap: wrap;
      }}
      .ti-tabs {{
        display: flex;
        background: var(--bg-alt, #f1f5f9);
        border-radius: 10px;
        padding: 3px;
        gap: 2px;
      }}
      .ti-tab {{
        padding: 7px 16px;
        border-radius: 8px;
        border: none;
        background: transparent;
        color: var(--text-muted);
        cursor: pointer;
        font-size: 13px;
        font-weight: 500;
        transition: all .15s;
      }}
      .ti-tab.active {{
        background: var(--card-bg);
        color: var(--text-color);
        box-shadow: 0 1px 3px rgba(0,0,0,.12);
      }}
      .ti-pos-filters {{
        display: flex;
        gap: 6px;
      }}
      .ti-pos {{
        padding: 6px 13px;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        background: var(--card-bg);
        color: var(--text-muted);
        cursor: pointer;
        font-size: 12px;
        font-weight: 600;
        transition: all .15s;
      }}
      .ti-pos.active {{
        background: var(--text-color);
        color: var(--card-bg);
        border-color: var(--text-color);
      }}
      .ti-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
        gap: 12px;
      }}
      .ti-card {{
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 14px;
        cursor: pointer;
        transition: transform .12s, box-shadow .12s;
        background: var(--card-bg);
      }}
      .ti-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,.12); }}
      .ti-card-top {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px; }}
      .ti-name {{ font-weight:700; font-size:14px; line-height:1.3; }}
      .ti-meta {{ font-size:11px; color:var(--text-muted); margin-top:2px; }}
      .ti-chip {{
        font-size:11px; font-weight:700;
        padding:3px 9px; border-radius:10px; white-space:nowrap; flex-shrink:0;
      }}
      .ti-divider {{ height:1px; background:var(--border-color); margin:8px 0; }}
      .ti-row {{ display:flex; justify-content:space-between; font-size:12px; margin-top:5px; }}
      .ti-row-label {{ color:var(--text-muted); }}
      .ti-row-val {{ font-weight:600; }}
      .ti-delta-pos {{ color:#10b981; }}
      .ti-delta-neg {{ color:#ef4444; }}
      .ti-momentum {{ font-size:11px; font-weight:600; margin-top:6px; display:flex; align-items:center; gap:4px; }}
      .ti-key {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px 24px;
        font-size: 12px; color: var(--text-muted);
        background: var(--bg-alt, #f8fafc);
        border: 1px solid var(--border-color);
        border-radius: 10px; padding: 12px 16px;
        margin-bottom: 20px; line-height: 1.4;
      }}
      .ti-key-item {{
        display: flex; align-items: center; gap: 8px;
      }}
      .ti-key-swatch {{
        display: inline-block; width: 12px; height: 12px;
        flex-shrink: 0; margin-top: 1px;
      }}
      .ti-key-delta {{
        background: linear-gradient(135deg, #10b981 50%, #ef4444 50%);
        border-radius: 3px; opacity: .8;
      }}
      .ti-key-label {{
        font-weight: 600; color: var(--text-color);
        margin-right: 4px;
      }}
      .ti-pagination {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 12px 0;
        border-top: 1px solid var(--border-color);
      }}
      .ti-pagination-info {{
        font-size: 13px;
        color: var(--text-muted);
      }}
      .ti-pagination-controls {{
        display: flex;
        align-items: center;
        gap: 12px;
      }}
      .ti-pagination-btn {{
        padding: 6px 12px;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        background: var(--card-bg);
        color: var(--text-color);
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        transition: all .15s;
        display: flex;
        align-items: center;
        gap: 4px;
      }}
      .ti-pagination-btn:hover:not(:disabled) {{
        background: var(--bg-alt);
        border-color: var(--primary-color);
      }}
      .ti-pagination-btn:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
      }}
      .ti-page-numbers {{
        display: flex;
        gap: 4px;
      }}
      .ti-page-number {{
        padding: 4px 8px;
        border: 1px solid var(--border-color);
        border-radius: 4px;
        background: var(--card-bg);
        color: var(--text-color);
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        min-width: 28px;
        text-align: center;
      }}
      .ti-page-number:hover {{
        background: var(--bg-alt);
      }}
      .ti-page-number.active {{
        background: var(--accent-color);
        color: var(--card);
        border-color: var(--accent-color);
        font-weight: 700;
      }}
    </style>

    <script>
    (function() {{
      const TI_SEASON = {season};
      let currentPage = 1;
      let paginationData = null;
      let currentTab = 'trending';
      let currentPos = 'ALL';

      // Load initial page
      loadTIPage(1);

      function loadTIPage(page) {{
        if (typeof page === 'string') {{
          if (page === 'prev' && currentPage > 1) {{
            page = currentPage - 1;
          }} else if (page === 'next' && paginationData && paginationData.has_next) {{
            page = currentPage + 1;
          }} else {{
            return;
          }}
        }}
        
        currentPage = page;
        document.getElementById('tiLoading').style.display = '';
        document.getElementById('tiGrid').style.display = 'none';
        document.getElementById('tiPagination').style.display = 'none';
        
        fetch('/api/trade-intel/trending?season=' + TI_SEASON + '&page=' + page)
          .then(r => r.json())
          .then(data => {{
            if (data.error) {{
              throw new Error(data.error);
            }}
            paginationData = data.pagination;
            document.getElementById('tiLoading').style.display = 'none';
            document.getElementById('tiGrid').style.display = '';
            updatePaginationControls();
            renderTI(data.players || []);
          }})
          .catch(() => {{
            document.getElementById('tiLoading').innerHTML =
              '<div style="color:var(--text-muted)">Trade data unavailable.</div>';
          }});
      }}

      function updatePaginationControls() {{
        if (!paginationData) return;
        
        const prevBtn = document.getElementById('tiPrevBtn');
        const nextBtn = document.getElementById('tiNextBtn');
        const pageNumbers = document.getElementById('tiPageNumbers');
        const paginationText = document.getElementById('tiPaginationText');
        
        // Update button states
        prevBtn.disabled = !paginationData.has_prev;
        nextBtn.disabled = !paginationData.has_next;
        
        // Update text
        const start = (paginationData.current_page - 1) * paginationData.per_page + 1;
        const end = Math.min(paginationData.current_page * paginationData.per_page, paginationData.total_players);
        paginationText.textContent = `Showing ${{start}}-${{end}} of ${{paginationData.total_players}} players`;
        
        // Update page numbers
        pageNumbers.innerHTML = '';
        const maxPages = 5;
        let startPage = Math.max(1, paginationData.current_page - Math.floor(maxPages / 2));
        let endPage = Math.min(paginationData.total_pages, startPage + maxPages - 1);
        
        if (endPage - startPage < maxPages - 1) {{
          startPage = Math.max(1, endPage - maxPages + 1);
        }}
        
        for (let i = startPage; i <= endPage; i++) {{
          const pageBtn = document.createElement('button');
          pageBtn.className = 'ti-page-number' + (i === paginationData.current_page ? ' active' : '');
          pageBtn.textContent = i;
          pageBtn.onclick = () => loadTIPage(i);
          pageNumbers.appendChild(pageBtn);
        }}
        
        document.getElementById('tiPagination').style.display = 'flex';
      }}

      window.switchTITab = function(tab) {{
        currentTab = tab;
        document.querySelectorAll('.ti-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
        renderTI();
      }};

      window.filterTI = function(pos) {{
        currentPos = pos;
        document.querySelectorAll('.ti-pos').forEach(b => b.classList.toggle('active', b.dataset.pos === pos));
        loadTIPage(currentPage); // Reload current page with new filter
      }};

      function renderTI(players = null) {{
        // If no players provided, we need to load current page data
        if (!players) {{
          loadTIPage(currentPage);
          return;
        }}
        
        // Apply position filtering
        let filteredPlayers = currentPos === 'ALL' ? players : players.filter(p => p.position === currentPos);
        
        // Apply tab filtering for non-trending tabs
        if (currentTab !== 'trending') {{
          const withDelta = filteredPlayers.filter(p => p.value_delta != null && p.model_value > 0);
          if (currentTab === 'buylows') {{
            filteredPlayers = withDelta.filter(p => p.value_delta < -5).sort((a, b) => a.value_delta - b.value_delta);
          }} else if (currentTab === 'sellhigh') {{
            filteredPlayers = withDelta.filter(p => p.value_delta > 5).sort((a, b) => b.value_delta - a.value_delta);
          }}
        }}
        
        const grid  = document.getElementById('tiGrid');
        const empty = document.getElementById('tiEmpty');
        
        if (filteredPlayers.length === 0) {{
          grid.style.display = 'none';
          empty.style.display = '';
          return;
        }}
        empty.style.display = 'none';
        grid.style.display = '';

        grid.innerHTML = filteredPlayers.map(p => {{
          const name   = p.name || 'Unknown';
          const pos    = p.position || '?';
          const team   = p.team || '?';
          const cnt7   = p.trade_count_7d  || 0;
          const cnt30  = p.trade_count_30d || 0;
          const cntAll = p.trade_count_all || 0;
          const market = p.market_value != null ? p.market_value.toFixed(1) : '—';
          const model  = p.model_value  != null ? p.model_value.toFixed(1)  : '—';
          const delta  = p.value_delta;
          const trend  = p.market_trend;

          let chipBg, chipColor, chipText;
          if (currentTab === 'trending') {{
            chipBg = '#3b82f620'; chipColor = '#3b82f6';
            chipText = (cntAll) + ' trades';
          }} else if (currentTab === 'buylows') {{
            chipBg = '#10b98120'; chipColor = '#10b981';
            chipText = delta != null ? (delta > 0 ? '+' : '') + Math.round(delta) : '—';
          }} else {{
            chipBg = '#f59e0b20'; chipColor = '#f59e0b';
            chipText = delta != null ? (delta > 0 ? '+' : '') + Math.round(delta) : '—';
          }}

          const deltaHtml = delta != null
            ? `<span class="${{delta >= 0 ? 'ti-delta-pos' : 'ti-delta-neg'}}">${{delta >= 0 ? '+' : ''}}${{Math.round(delta)}}</span>`
            : '<span style="color:var(--text-muted)">—</span>';

          // Momentum: 14d median minus 90d median. Threshold ±5 to avoid noise.
          let momentumHtml = '';
          if (trend != null) {{
            if (trend >= 5) {{
              momentumHtml = '<span style="color:#10b981;">▲</span> Rising';
            }} else if (trend <= -5) {{
              momentumHtml = '<span style="color:#ef4444;">▼</span> Falling';
            }}
          }}

          // Pre-process strings to avoid backslashes
          const player_json = JSON.stringify(p).replace(/"/g, '\\"');
          const escaped_name = name.replace(/'/g, "\\'");
          const onclick_js = p.is_rookie && p.is_rookie !== 'False' ? `rkOpenModal(${{player_json}})` : `openPlayerModal(${{p.player_id}},'${{escaped_name}}')`;

          return `<div class="ti-card" onclick="${{onclick_js}}">
            <div class="ti-card-top">
              <div>
                <div class="ti-name">${{name}}</div>
                <div class="ti-meta">${{pos}} · ${{team}}</div>
              </div>
              <div class="ti-chip" style="background:${{chipBg}};color:${{chipColor}};">${{chipText}}</div>
            </div>
            <div class="ti-divider"></div>
            <div class="ti-row"><span class="ti-row-label">Market</span><span class="ti-row-val">${{market}}</span></div>
            <div class="ti-row"><span class="ti-row-label">BR Model</span><span class="ti-row-val">${{model}}</span></div>
            <div class="ti-row"><span class="ti-row-label">Delta</span><span class="ti-row-val">${{deltaHtml}}</span></div>
            <div class="ti-row"><span class="ti-row-label">Trades 7d/30d</span><span class="ti-row-val">${{cnt7}} / ${{cnt30}}</span></div>
            ${{momentumHtml ? `<div class="ti-momentum">${{momentumHtml}}</div>` : ''}}
          </div>`;
        }}).join('');
      }}
      
      // Expose functions to global scope for onclick handlers
      window.loadTIPage = loadTIPage;
    }})();
    </script>
    """
    return render_page("Trade Intelligence", league_id, "trade-intel", body_html, platform, season)


@app.route("/trade-intel")
def page_trade_intel_guest():
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_trade_intel(platform="sleeper", season=current_season, league_id=None)


@app.route("/<platform>/<int:season>/<league_id>/trade-database")
def page_trade_database(platform: str, season: int, league_id: str):
    body_html = f"""
    <div class="card central" style="max-width:960px;">
      <div class="card-header" style="border-bottom:1px solid var(--border-color);padding-bottom:16px;margin-bottom:0;">
        <h2 style="margin:0 0 4px;font-size:20px;">Trade Database</h2>
        <div style="font-size:13px;color:var(--text-muted);">
          Explore thousands of real dynasty trades to understand player values and market trends
        </div>
      </div>
      <div class="card-body" style="padding-top:20px;">

        <div class="tdb-toolbar">
          <div class="tdb-search-wrap">
            <span class="tdb-search-icon" aria-hidden="true"></span>
            <input id="tdbSearch" type="text" placeholder="Search by player name..." class="tdb-search">
          </div>
          <div class="tdb-lt-filters">
            <button class="tdb-lt active" data-lt="all" onclick="tdbFilter('all')">All</button>
            <button class="tdb-lt" data-lt="1qb" onclick="tdbFilter('1qb')">1QB</button>
            <button class="tdb-lt" data-lt="sf"  onclick="tdbFilter('sf')">SF</button>
          </div>
        </div>

        <div id="tdbStatus" class="tdb-status"></div>
        <div id="tdbList"   class="tdb-list"></div>
        
        <div id="tdbLoading" style="text-align:center;padding:48px 0;color:var(--text-muted);display:none;">
          <div class="loading-spinner" style="margin:0 auto 12px;"></div>
          Loading trade data...
        </div>
        
        <div id="tdbPagination" class="ti-pagination" style="display:none;">
          <div class="ti-pagination-info">
            <span id="tdbPaginationText">Showing 1-20 of 100 trades</span>
          </div>
          <div class="ti-pagination-controls">
            <button id="tdbPrevBtn" class="ti-pagination-btn" onclick="loadTDBPage('prev')" disabled>
              <i class="fa-solid fa-chevron-left"></i> Previous
            </button>
            <div id="tdbPageNumbers" class="ti-page-numbers"></div>
            <button id="tdbNextBtn" class="ti-pagination-btn" onclick="loadTDBPage('next')" disabled>
              Next <i class="fa-solid fa-chevron-right"></i>
            </button>
          </div>
        </div>

      </div>
    </div>

    <style>
      .tdb-toolbar {{
        display: flex; gap: 12px; margin-bottom: 16px;
        flex-wrap: wrap; align-items: center;
      }}
      .tdb-search-wrap {{
        flex: 1; min-width: 200px;
        display: flex; align-items: center;
        border: 1px solid var(--border-color); border-radius: 8px;
        background: var(--card-bg); padding: 0 12px; gap: 8px;
      }}
      .tdb-search-icon {{
        display: inline-block; width: 14px; height: 14px; flex-shrink: 0;
        background: url('/static/images/magnifying-glass-solid.png') no-repeat center / contain;
        filter: brightness(0) saturate(100%) invert(60%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(85%) contrast(90%);
        pointer-events: none;
      }}
      .tdb-search {{
        flex: 1; padding: 9px 0; border: none; background: transparent;
        color: var(--text-color); font-size: 14px; outline: none;
        min-width: 0;
      }}
      .tdb-search-wrap:focus-within {{ border-color: var(--accent-color, #3b82f6); }}
      .tdb-lt-filters {{ display: flex; gap: 4px; }}
      .tdb-lt {{
        padding: 7px 14px; border-radius: 8px; border: 1px solid var(--border-color);
        background: var(--card-bg); color: var(--text-muted); cursor: pointer;
        font-size: 13px; font-weight: 600; transition: all .15s;
      }}
      .tdb-lt.active {{
        background: var(--text-color); color: var(--card-bg); border-color: var(--text-color);
      }}
      .tdb-status {{ font-size: 12px; color: var(--text-muted); margin-bottom: 14px; min-height: 16px; }}
      .tdb-list {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
      @media(max-width: 600px) {{ .tdb-list {{ grid-template-columns: 1fr; }} }}
      .tdb-more-wrap {{ text-align: center; margin-top: 20px; grid-column: 1 / -1; }}
      .tdb-more-btn {{
        padding: 9px 28px; border-radius: 20px; border: 1px solid var(--border-color);
        background: var(--card-bg); color: var(--text-color); cursor: pointer; font-size: 13px;
      }}

      /* Trade card */
      .tdb-card {{
        border: 1px solid var(--border-color); border-radius: 12px;
        overflow: hidden; background: var(--card-bg);
      }}
      .tdb-card-head {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 14px; border-bottom: 1px solid var(--border-color);
        background: var(--bg-alt, rgba(0,0,0,.03));
      }}
      .tdb-card-date {{ font-size: 11px; color: var(--text-muted); font-weight: 500; }}
      .tdb-badges {{ display: flex; gap: 5px; flex-wrap: wrap; }}
      .tdb-badge {{
        font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 8px;
        background: var(--row, #1e293b); color: var(--text);
        border: 1px solid var(--border-color);
      }}
      .tdb-badge-sf {{ background: #7c3aed22; color: #a78bfa; border-color: #7c3aed44; }}
      .tdb-card-body {{
        display: grid; grid-template-columns: 1fr 1px 1fr;
      }}
      .tdb-col {{
        padding: 12px 14px; display: flex; flex-direction: column; gap: 5px;
      }}
      .tdb-col-divider {{ background: var(--border-color); }}
      .tdb-asset {{
        font-size: 14px; color: var(--text); font-weight: 500;
        display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
      }}
      .tdb-asset.tdb-match {{ font-weight: 800; color: var(--accent-color, #3b82f6); }}
      .tdb-asset.tdb-pick {{ color: var(--text-muted); font-size: 14px; font-weight: 500; }}
      .tdb-pos {{
        font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 4px;
        background: var(--row, #1e293b); color: var(--text); flex-shrink: 0;
      }}
      @media(max-width: 480px) {{
        .tdb-card-body {{ grid-template-columns: 1fr; }}
        .tdb-col-divider {{ height: 1px; width: auto; }}
      }}
      .ti-pagination {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 12px 0;
        border-top: 1px solid var(--border);
      }}
      .ti-pagination-info {{
        font-size: 13px;
        color: var(--text-muted);
      }}
      .ti-pagination-controls {{
        display: flex;
        align-items: center;
        gap: 12px;
      }}
      .ti-pagination-btn {{
        padding: 6px 12px;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: var(--card);
        color: var(--text);
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        transition: all .15s;
        display: flex;
        align-items: center;
        gap: 4px;
      }}
      .ti-pagination-btn:hover:not(:disabled) {{
        background: var(--bg-alt);
        border-color: var(--accent-color);
      }}
      .ti-pagination-btn:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
      }}
      .ti-page-numbers {{
        display: flex;
        gap: 4px;
      }}
      .ti-page-number {{
        padding: 4px 8px;
        border: 1px solid var(--border);
        border-radius: 4px;
        background: var(--card);
        color: var(--text);
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        min-width: 28px;
        text-align: center;
      }}
      .ti-page-number:hover {{
        background: var(--bg-alt);
      }}
      .ti-page-number.active {{
        background: var(--accent-color);
        color: var(--card);
        border-color: var(--accent-color);
        font-weight: 700;
      }}
    </style>

    <script>
    (function() {{
      const TDB_SEASON = {season};
      let currentPage = 1;
      let paginationData = null;
      let leagueType = 'all';
      let searchQuery = '';
      let loading = false;

      const listEl   = document.getElementById('tdbList');
      const statusEl = document.getElementById('tdbStatus');
      const searchEl = document.getElementById('tdbSearch');

      // Load initial page
      loadTDBPage(1);

      function loadTDBPage(page) {{
        if (loading) return;
        if (typeof page === 'string') {{
          if (page === 'prev' && currentPage > 1) {{
            page = currentPage - 1;
          }} else if (page === 'next' && paginationData && paginationData.has_next) {{
            page = currentPage + 1;
          }} else {{
            return;
          }}
        }}
        
        currentPage = page;
        loading = true;
        statusEl.textContent = '';
        listEl.style.display = 'none';
        document.getElementById('tdbLoading').style.display = '';
        document.getElementById('tdbPagination').style.display = 'none';
        
        const apiPage = page - 1; // Convert to 0-based for API
        const params = new URLSearchParams({{ page: apiPage, limit: 20, league_type: leagueType, season: TDB_SEASON }});
        if (searchQuery) params.set('q', searchQuery);

        fetch('/api/trade-database?' + params)
          .then(r => r.json())
          .then(data => {{
            if (data.error) {{
              throw new Error(data.error);
            }}
            const trades = data.trades || [];
            if (trades.length === 0) {{
              document.getElementById('tdbLoading').style.display = 'none';
              listEl.innerHTML = '<div style="color:var(--text-muted);padding:20px 0;text-align:center;grid-column:1/-1;">No trades found.</div>';
              statusEl.textContent = '';
              document.getElementById('tdbPagination').style.display = 'none';
              loading = false;
              return;
            }}
            
            document.getElementById('tdbLoading').style.display = 'none';
            paginationData = data.pagination;
            statusEl.textContent = '';
            listEl.style.display = '';
            updateTDBPaginationControls();
            renderTDBTrades(trades);
            loading = false;
          }})
          .catch(err => {{
            console.error('Error loading trades:', err);
            document.getElementById('tdbLoading').style.display = 'none';
            statusEl.textContent = 'Error loading trades';
            loading = false;
          }});
      }}

      function updateTDBPaginationControls() {{
        if (!paginationData) return;
        
        const prevBtn = document.getElementById('tdbPrevBtn');
        const nextBtn = document.getElementById('tdbNextBtn');
        const pageNumbers = document.getElementById('tdbPageNumbers');
        const paginationText = document.getElementById('tdbPaginationText');
        
        // Update button states
        prevBtn.disabled = !paginationData.has_prev;
        nextBtn.disabled = !paginationData.has_next;
        
        // Update text
        const start = (paginationData.current_page - 1) * paginationData.per_page + 1;
        const end = Math.min(paginationData.current_page * paginationData.per_page, paginationData.total_players);
        paginationText.textContent = `Showing ${{start}}-${{end}} of ${{paginationData.total_players}} trades`;
        
        // Update page numbers
        pageNumbers.innerHTML = '';
        const maxPages = 5;
        let startPage = Math.max(1, paginationData.current_page - Math.floor(maxPages / 2));
        let endPage = Math.min(paginationData.total_pages, startPage + maxPages - 1);
        
        if (endPage - startPage < maxPages - 1) {{
          startPage = Math.max(1, endPage - maxPages + 1);
        }}
        
        for (let i = startPage; i <= endPage; i++) {{
          const pageBtn = document.createElement('button');
          pageBtn.className = 'ti-page-number' + (i === paginationData.current_page ? ' active' : '');
          pageBtn.textContent = i;
          pageBtn.onclick = () => loadTDBPage(i);
          pageNumbers.appendChild(pageBtn);
        }}
        
        document.getElementById('tdbPagination').style.display = 'flex';
      }}

      function renderTDBTrades(trades) {{
        renderTrades(trades, searchQuery);
      }}

      function renderTrades(trades, q) {{
        const lq = (q || '').toLowerCase();
        trades.forEach(t => {{
          const sfBadge    = t.is_superflex === true  ? '<span class="tdb-badge tdb-badge-sf">SF</span>'
                           : t.is_superflex === false ? '<span class="tdb-badge">1QB</span>' : '';
          const teamsBadge = t.num_teams    ? `<span class="tdb-badge">${{t.num_teams}} Teams</span>` : '';
          const scoreBadge = t.scoring_type ? `<span class="tdb-badge">${{t.scoring_type.toUpperCase()}}</span>` : '';

          function renderAsset(a) {{
            const match = lq && a.name && a.name.toLowerCase().includes(lq);
            const pickCls = a.type === 'pick' ? ' tdb-pick' : '';
            const cls = 'tdb-asset' + pickCls + (match ? ' tdb-match' : '');
            const pos = a.position && a.type === 'player' ? `<span class="tdb-pos">${{a.position}}</span>` : '';
            return `<div class="${{cls}}">${{a.name}}${{pos}}</div>`;
          }}

          const sideA = (t.side_a || []).map(renderAsset).join('') || '<div class="tdb-asset" style="color:var(--text-muted)">—</div>';
          const sideB = (t.side_b || []).map(renderAsset).join('') || '<div class="tdb-asset" style="color:var(--text-muted)">—</div>';

          const card = document.createElement('div');
          card.className = 'tdb-card';
          card.innerHTML = `
            <div class="tdb-card-head">
              <span class="tdb-card-date">${{t.date || '—'}}</span>
              <div class="tdb-badges">${{sfBadge}}${{teamsBadge}}${{scoreBadge}}</div>
            </div>
            <div class="tdb-card-body">
              <div class="tdb-col">${{sideA}}</div>
              <div class="tdb-col-divider"></div>
              <div class="tdb-col">${{sideB}}</div>
            </div>`;
          listEl.appendChild(card);
        }});
      }}

      window.tdbFilter = function(lt) {{
        leagueType = lt;
        document.querySelectorAll('.tdb-lt').forEach(b => b.classList.toggle('active', b.dataset.lt === lt));
        loadTDBPage(1); // Reset to first page when filtering
      }};

      let debounce;
      searchEl.addEventListener('input', () => {{
        clearTimeout(debounce);
        debounce = setTimeout(() => {{ 
          searchQuery = searchEl.value.trim(); 
          loadTDBPage(1); // Reset to first page when searching
        }}, 350);
      }});

      // Expose pagination function to global scope
      window.loadTDBPage = loadTDBPage;
    }})();
    </script>
    """
    return render_page("Trade Database", league_id, "trade-database", body_html, platform, season)


@app.route("/trade-database")
def page_trade_database_guest():
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_trade_database(platform="sleeper", season=current_season, league_id=None)


@app.route("/prospects")
def page_prospects_guest():
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_prospects(platform="sleeper", season=current_season, league_id=None)


@app.route("/<platform>/<int:season>/<league_id>/teams")
def page_teams(platform: str, season: int, league_id: str):
    cached = get_page_html_from_cache(platform, season, league_id, "teams")
    if cached:
        return render_page("BR Fantasy Teams", league_id, "teams", cached, platform, season)

    ctx = get_league_ctx_from_cache(platform, league_id, season)
    body_html = build_teams_body(ctx)
    store_page_html(platform, season, league_id, "teams", body_html)
    return render_page("BR Fantasy Teams", league_id, "teams", body_html, platform, season)


def _ens(career_owners: dict, uid: str) -> None:
    """Ensure uid exists in career_owners with all default fields."""
    career_owners.setdefault(uid, {
        "Wins": 0, "Losses": 0, "Ties": 0,
        "PF": 0.0, "PA": 0.0, "seasons": 0, "weekly_pts": [],
    })


def _collect_all_season_data(platform: str, league_id: str, season: int):
    """
    Load ctx for every available historical season and return aggregated data:
      career_owners  – dict  user_id -> {Wins, Losses, Ties, PF, PA, seasons, weekly_pts}
      championships  – dict  user_id -> [season, ...]
      season_records – list  of per-season summary dicts
      user_id_to_name – dict user_id -> latest display name (for rendering)
    Keyed by user_id (stable) rather than team name (changes across seasons).
    """
    cached = get_awards_agg_from_cache(platform, season, league_id)
    if cached is not None:
        return cached

    available = get_available_history_seasons(platform, league_id, season)
    career_owners: dict = {}
    championships: dict = {}
    season_records: list = []
    user_id_to_name: dict = {}  # user_id → latest known display name

    for hist_s in available:
        rid = resolve_league_id_for_season(platform, league_id, season, hist_s)
        try:
            ctx = get_league_ctx_from_cache(platform, rid, hist_s)
        except Exception:
            continue

        df = ctx.get("df_weekly", pd.DataFrame())
        if df.empty or "owner" not in df.columns:
            continue

        # Build user_id → display_name for this season (later seasons overwrite earlier)
        season_users = ctx.get("users") or []
        for u in season_users:
            uid = str(u.get("user_id") or "").strip()
            if not uid:
                continue
            metadata = u.get("metadata") or {}
            name = (
                u.get("display_name")  # Prioritize actual username
                or metadata.get("team_name")  # Fall back to team name
                or metadata.get("username")  # Check username in metadata too
                or u.get("username")
                or uid
            )
            user_id_to_name[uid] = name

        # roster_id → user_id for this season
        season_rosters = ctx.get("rosters") or []
        roster_map = ctx.get("roster_map") or {}
        roster_to_uid: dict[str, str] = {
            str(r["roster_id"]): str(r.get("owner_id") or "")
            for r in season_rosters
        }
        # team_name → user_id reverse lookup (for championship mapping)
        name_to_uid: dict[str, str] = {
            roster_map.get(str(r["roster_id"]), ""): str(r.get("owner_id") or "")
            for r in season_rosters
        }

        # df_weekly has roster_id column — use it as the stable team key.
        # Team names can collide in larger leagues, which can hide/merge users.
        has_roster_id = "roster_id" in df.columns
        df_for_stats = df
        if has_roster_id:
            df_for_stats = df.copy()
            df_for_stats["owner"] = df_for_stats["roster_id"].astype(str)

        mock_league = ctx.get("league") or {}
        ts = build_regular_season_team_stats(df_for_stats, mock_league)

        for _, row in ts.iterrows():
            owner_key = str(row.get("owner", "Unknown"))
            if has_roster_id:
                uid = roster_to_uid.get(owner_key) or owner_key
            else:
                # Fallback path when roster_id is unavailable.
                uid = name_to_uid.get(owner_key) or owner_key
            if uid not in career_owners:
                career_owners[uid] = {
                    "Wins": 0, "Losses": 0, "Ties": 0,
                    "PF": 0.0, "PA": 0.0, "seasons": 0,
                    "weekly_pts": [],
                }
            career_owners[uid]["Wins"]   += int(row.get("Wins", 0))
            career_owners[uid]["Losses"] += int(row.get("Losses", 0))
            career_owners[uid]["Ties"]   += int(row.get("Ties", 0))
            career_owners[uid]["PF"]     += float(row.get("PF", 0))
            career_owners[uid]["PA"]     += float(row.get("PA", 0))
            career_owners[uid]["seasons"] += 1

        # Collect weekly scores per user_id
        sub_cols = ["owner", "points"] + (["roster_id"] if "roster_id" in df.columns else [])
        if {"owner", "points", "finalized"}.issubset(df.columns):
            sub = df[df["finalized"] == True][sub_cols].copy()
        elif {"owner", "points"}.issubset(df.columns):
            sub = df[sub_cols].copy()
        else:
            sub = pd.DataFrame()
        group_col = "roster_id" if has_roster_id and "roster_id" in sub.columns else "owner"
        for owner_key, grp in sub.groupby(group_col):
            if group_col == "roster_id":
                uid = roster_to_uid.get(str(owner_key)) or str(owner_key)
            else:
                uid = name_to_uid.get(str(owner_key)) or str(owner_key)
            career_owners.setdefault(uid, {
                "Wins": 0, "Losses": 0, "Ties": 0,
                "PF": 0.0, "PA": 0.0, "seasons": 0, "weekly_pts": [],
            })["weekly_pts"].extend(grp["points"].tolist())

        # ── Barely Breathing: wins by <5 pts ───────────────────────────────
        if {"owner", "points", "points_against"}.issubset(df.columns):
            margins = (df["points"] - df["points_against"]).astype(float)
            close_win_mask = (margins > 0) & (margins < 5)
            for owner, grp in df[close_win_mask].groupby("owner"):
                uid = name_to_uid.get(str(owner)) or str(owner)
                _ens(career_owners, uid)
                career_owners[uid]["close_wins"] = career_owners[uid].get("close_wins", 0) + len(grp)

        # ── Playoff Riser: avg pts regular season vs playoffs ──────────────
        _league_settings = mock_league.get("settings") or {}
        _po_start = int(_league_settings.get("playoff_week_start") or 15)
        if {"owner", "week", "points"}.issubset(df.columns):
            _reg = df[df["week"] < _po_start]
            _po = df[df["week"] >= _po_start]
            for owner, reg_grp in _reg.groupby("owner"):
                uid = name_to_uid.get(str(owner)) or str(owner)
                po_grp = _po[_po["owner"] == owner]
                if reg_grp.empty or po_grp.empty:
                    continue
                reg_avg = float(reg_grp["points"].mean())
                po_avg = float(po_grp["points"].mean())
                _ens(career_owners, uid)
                career_owners[uid]["playoff_delta_sum"] = career_owners[uid].get("playoff_delta_sum", 0.0) + (po_avg - reg_avg)
                career_owners[uid]["playoff_delta_n"] = career_owners[uid].get("playoff_delta_n", 0) + 1

        # ── Transactions: Main Character + Waiver Wire Demon ───────────────
        try:
            from dashboard_services.service import get_transactions_by_week as _gtxw
            _tx_data = _gtxw(rid, list(range(1, 19)), platform=platform, season=hist_s)
            for _week_txs in _tx_data.values():
                for _t in (_week_txs or []):
                    _ttype = _t.get("type")
                    if _ttype in ("waiver", "waiver_add", "free_agent"):
                        _adds = _t.get("adds") or {}
                        for _pid, _rid_t in _adds.items():
                            _uid = roster_to_uid.get(str(_rid_t), "")
                            if _uid:
                                _ens(career_owners, _uid)
                                career_owners[_uid]["waiver_adds"] = career_owners[_uid].get("waiver_adds", 0) + 1
                                career_owners[_uid]["activity"] = career_owners[_uid].get("activity", 0) + 1
                    elif _ttype == "trade":
                        _trade_rids = set(str(r) for r in (_t.get("roster_ids") or []))
                        _trade_rids |= {str(v) for v in (_t.get("adds") or {}).values()}
                        for _rid_t in _trade_rids:
                            _uid = roster_to_uid.get(_rid_t, "")
                            if _uid:
                                _ens(career_owners, _uid)
                                career_owners[_uid]["trade_count"] = career_owners[_uid].get("trade_count", 0) + 1
                                career_owners[_uid]["activity"] = career_owners[_uid].get("activity", 0) + 2
        except Exception as _e:
            pass  # transaction data unavailable; skip these awards

        # ── Bench Warmer MVP: points left on bench ─────────────────────────
        try:
            from dashboard_services.platform_api import get_matchups as _gmu
            for _w in range(1, _po_start):  # regular season only
                for _mu in (_gmu(platform, rid, _w, hist_s) or []):
                    _roster_id = str(_mu.get("roster_id", ""))
                    _uid = roster_to_uid.get(_roster_id, "")
                    if not _uid:
                        continue
                    _starters = {str(s) for s in (_mu.get("starters") or []) if s and str(s) != "0"}
                    _all_pts = _mu.get("players_points") or {}
                    _bench = sum(float(p) for pid, p in _all_pts.items() if str(pid) not in _starters and str(pid) != "0")
                    _ens(career_owners, _uid)
                    career_owners[_uid]["bench_pts"] = career_owners[_uid].get("bench_pts", 0.0) + _bench
        except Exception as _e:
            pass  # matchup data unavailable; skip bench award

        champ_name, runner_up_name = get_champion_and_runner_up(ctx)
        champ_uid = name_to_uid.get(champ_name) or champ_name
        runner_up_uid = name_to_uid.get(runner_up_name) or runner_up_name
        if champ_name != "—":
            championships.setdefault(champ_uid, []).append(hist_s)

        summary = _build_history_summary(ctx)
        season_records.append({
            "season": hist_s,
            "champion": champ_name,
            "champion_uid": champ_uid,
            "runner_up": runner_up_name,
            "runner_up_uid": runner_up_uid,
            "champion_record": summary.get("champion_record", "—"),
            "top_pf_team": summary.get("top_scorer_team", "—"),
            "top_pf": float(summary.get("top_scorer_value") or 0),
            "highest_week_team": summary.get("highest_week_team", "—"),
            "highest_week_value": float(summary.get("highest_week_value") or 0),
            "closest_matchup": summary.get("closest_matchup", "—"),
            "closest_margin": float(summary.get("closest_margin") or 0),
        })

    payload = (available, career_owners, championships, season_records, user_id_to_name)
    store_awards_agg(platform, season, league_id, payload)
    return payload


def _build_awards_html(career_owners: dict, championships: dict, season_records: list, user_id_to_name: Optional[dict] = None, platform: str = "", season: int = 0, league_id: str = "", league_name: str = "") -> str:
    """Render the All-Time Awards page body HTML."""
    name_map = user_id_to_name or {}

    def _display_name(uid: str) -> str:
        return name_map.get(uid) or uid


    # Build career standings DataFrame
    rows = []
    for uid, d in career_owners.items():
        games = d["Wins"] + d["Losses"] + d["Ties"]
        pts = d["weekly_pts"]
        n = len(pts)
        mean = sum(pts) / n if n > 0 else 0.0
        variance = sum((x - mean) ** 2 for x in pts) / (n - 1) if n > 1 else 0.0
        std = variance ** 0.5
        delta_n = d.get("playoff_delta_n", 0)
        rows.append({
            "owner": uid,
            "display_name": _display_name(uid),
            "Championships": len(championships.get(uid, [])),
            "Wins": d["Wins"],
            "Losses": d["Losses"],
            "PF": d["PF"],
            "PA": d["PA"],
            "Win%": d["Wins"] / games if games > 0 else 0.0,
            "AVG": d["PF"] / games if games > 0 else 0.0,
            "MAX": max(pts) if pts else 0.0,
            "Seasons": d["seasons"],
            "STD": std,
            "CloseWins": d.get("close_wins", 0),
            "BenchPts": d.get("bench_pts", 0.0),
            "WaiverAdds": d.get("waiver_adds", 0),
            "Activity": d.get("activity", 0),
            "PlayoffDelta": d.get("playoff_delta_sum", 0.0) / delta_n if delta_n > 0 else None,
        })

    career_df = pd.DataFrame(rows).sort_values(
        ["Championships", "Wins", "Win%", "PF"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    # ── Render helpers ──────────────────────────────────────────────────────
    _MEDALS = {
        1: '<span class="rank rank-first">1</span>',
        2: '<span class="rank rank-second">2</span>',
        3: '<span class="rank rank-third">3</span>',
    }

    def _rank_badge(i: int) -> str:
        return _MEDALS.get(i, f'<span class="rank-plain">{i}</span>')

    def _record_style(wins: int, losses: int) -> str:
        if wins > losses:
            return "color:#16a34a;font-weight:700;"
        if losses > wins:
            return "color:#ef4444;font-weight:700;"
        return "font-weight:700;"

    def _winpct_bar(pct: float) -> str:
        w = max(2, int(pct * 100))
        color = "#16a34a" if pct >= 0.5 else "#ef4444"
        return f'<div class="winpct-bar"><div class="winpct-fill" style="width:{w}%;background:{color};"></div></div>'

    def _hist_card(label: str, value: str, sub: str = "", icon: str = "") -> str:
        icon_html = f'<div class="history-card-icon">{icon}</div>' if icon else ""
        sub_html = f"<div class='history-card-sub'>{sub}</div>" if sub else ""
        return f"""
        <div class="history-card">
          {icon_html}
          <div class="history-card-label">{label}</div>
          <div class="history-card-value">{value}</div>
          {sub_html}
        </div>"""

    def _fun_award(title: str, icon: str, winner: str, sub: str, accent: str) -> str:
        return f"""
        <div class="fun-award-item" style="--award-accent:{accent};">
          <div class="fun-award-title">{title}</div>
          <div class="fun-award-icon">{icon}</div>
          <div class="fun-award-winner">{winner}</div>
          <div class="fun-award-sub">{sub}</div>
        </div>"""

    # ── Career standings table ──────────────────────────────────────────────
    table_rows_html = ""
    for i, (_, row) in enumerate(career_df.iterrows()):
        rank = i + 1
        rings = (
            '<i class="fa-solid fa-trophy" style="color:#f59e0b;" aria-hidden="true"></i> ' * int(row["Championships"])
        ) if row["Championships"] > 0 else ""
        rec_style = _record_style(int(row["Wins"]), int(row["Losses"]))
        table_rows_html += f"""
        <tr>
          <td>{_rank_badge(rank)}</td>
          <td>{html.escape(str(row['display_name']))} {rings}</td>
          <td style="font-weight:700;color:var(--accent);">{int(row['Championships'])}</td>
          <td style="{rec_style}">{int(row['Wins'])}-{int(row['Losses'])}</td>
          <td>
            <span>{row['Win%']:.1%}</span>
            {_winpct_bar(row['Win%'])}
          </td>
          <td>{row['PF']:,.1f}</td>
          <td>{row['PA']:,.1f}</td>
          <td>{row['AVG']:.1f}</td>
          <td>{row['MAX']:.1f}</td>
          <td>{int(row['Seasons'])}</td>
        </tr>"""

    standings_table = f"""
    <div class="card">
      <div class="card-header"><h2>All-Time Standings</h2></div>
      <div class="card-body" style="padding-top:0;">
        <div class="history-table-wrap">
          <table class="history-table">
            <thead><tr>
              <th>#</th><th>Team</th><th>Titles</th><th>Record</th>
              <th>Win%</th><th>PF</th><th>PA</th><th>Avg/Wk</th><th>Best Wk</th><th>Seasons</th>
            </tr></thead>
            <tbody>{table_rows_html}</tbody>
          </table>
        </div>
      </div>
    </div>"""

    # ── Championship timeline ───────────────────────────────────────────────
    def _cell(v: str) -> str:
        return html.escape(v) if v and v != "—" else "<span style='color:var(--text-muted)'>—</span>"

    sorted_records = sorted(season_records, key=lambda x: x["season"], reverse=True)
    most_recent_season = sorted_records[0]["season"] if sorted_records else None

    champ_rows_html = ""
    for rec in sorted_records:
        champ_display = _display_name(rec.get("champion_uid") or rec["champion"]) if rec.get("champion_uid") else rec["champion"]
        runner_display = _display_name(rec.get("runner_up_uid") or rec["runner_up"]) if rec.get("runner_up_uid") else rec["runner_up"]
        row_cls = ' class="champ-recent"' if rec["season"] == most_recent_season else ""
        champ_rows_html += f"""
        <tr{row_cls}>
          <td><strong>{rec['season']}</strong></td>
          <td style="font-weight:700;"><i class="fa-solid fa-trophy" style="color:#f59e0b;margin-right:5px;" aria-hidden="true"></i>{_cell(champ_display)}</td>
          <td>{html.escape(rec['champion_record'])}</td>
          <td>{_cell(runner_display)}</td>
        </tr>"""

    champ_table = f"""
    <div class="card champ-history-card">
      <div class="card-header"><h2>Championship History</h2></div>
      <div class="card-body champ-history-body" style="padding-top:0;">
        <div class="history-table-wrap champ-history-scroll">
          <table class="history-table">
            <thead><tr><th>Season</th><th>Champion</th><th>Record</th><th>Runner-Up</th></tr></thead>
            <tbody>{champ_rows_html}</tbody>
          </table>
        </div>
      </div>
    </div>"""

    # ── League Records cards ────────────────────────────────────────────────
    if season_records:
        best_pf_rec = max(season_records, key=lambda x: x["top_pf"])
        best_wk_rec = max(season_records, key=lambda x: x["highest_week_value"])
    else:
        best_pf_rec = best_wk_rec = None

    highlights_html = ""
    if best_pf_rec:
        highlights_html += _hist_card(
            "Highest Season PF",
            html.escape(best_pf_rec["top_pf_team"]),
            f"{best_pf_rec['top_pf']:.1f} pts in {best_pf_rec['season']}",
            '<i class="fa-solid fa-fire" style="color:#f97316;"></i>',
        )
    if best_wk_rec:
        highlights_html += _hist_card(
            "Highest Single Week",
            html.escape(best_wk_rec["highest_week_team"]),
            f"{best_wk_rec['highest_week_value']:.1f} pts in {best_wk_rec['season']}",
            '<i class="fa-solid fa-bolt" style="color:#facc15;"></i>',
        )

    if not career_df.empty and int(career_df.iloc[0]["Championships"]) > 0:
        most_champ_owner = str(career_df.iloc[0]["display_name"])
        most_champ_n = int(career_df.iloc[0]["Championships"])
        highlights_html += _hist_card(
            "Most Championships",
            html.escape(most_champ_owner),
            f"{most_champ_n} title{'s' if most_champ_n > 1 else ''}",
            '<i class="fa-solid fa-trophy" style="color:#f59e0b;"></i>',
        )

    # Best all-time win% (min 2 seasons)
    eligible = career_df[career_df["Seasons"] >= 2]
    if not eligible.empty:
        best_winpct_row = eligible.loc[eligible["Win%"].idxmax()]
        highlights_html += _hist_card(
            "Best Win%",
            html.escape(str(best_winpct_row["display_name"])),
            f"{best_winpct_row['Win%']:.1%} over {int(best_winpct_row['Seasons'])} seasons",
            '<i class="fa-solid fa-chart-line" style="color:#22c55e;"></i>',
        )

    # Most seasons played
    if not career_df.empty:
        most_seasons_row = career_df.loc[career_df["Seasons"].idxmax()]
        highlights_html += _hist_card(
            "Most Seasons",
            html.escape(str(most_seasons_row["display_name"])),
            f"{int(most_seasons_row['Seasons'])} seasons played",
            '<i class="fa-solid fa-calendar-days"></i>',
        )

    # Most points, no ring
    no_titles = career_df[career_df["Championships"] == 0]
    if not no_titles.empty:
        unlucky_row = no_titles.loc[no_titles["PF"].idxmax()]
        highlights_html += _hist_card(
            "Most Points, No Ring",
            html.escape(str(unlucky_row["display_name"])),
            f"{unlucky_row['PF']:,.1f} career points",
            '<i class="fa-solid fa-heart-crack"></i>',
        )

    highlights_section = ""
    if highlights_html:
        highlights_section = f"""
    <div class="card">
      <div class="card-header"><h2>League Records</h2></div>
      <div class="card-body">
        <div class="history-cards-grid awards-records-grid">{highlights_html}</div>
      </div>
    </div>"""

    # ── Fun Awards ──────────────────────────────────────────────────────────
    fun_awards_html = ""

    # The Bridesmaid — most runner-up appearances without a title
    runner_up_counts: dict = {}
    for rec in season_records:
        ru_uid = rec.get("runner_up_uid")
        ru_display = _display_name(ru_uid) if ru_uid else rec.get("runner_up", "")
        if ru_display and ru_display != "—":
            runner_up_counts[ru_display] = runner_up_counts.get(ru_display, 0) + 1
    no_title_names = set(career_df[career_df["Championships"] == 0]["display_name"].astype(str).tolist())
    bridesmaid_candidates = {k: v for k, v in runner_up_counts.items() if k in no_title_names and v >= 1}
    if bridesmaid_candidates:
        bridesmaid_name = max(bridesmaid_candidates, key=bridesmaid_candidates.get)
        bridesmaid_count = bridesmaid_candidates[bridesmaid_name]
        fun_awards_html += _fun_award(
            "The Bridesmaid",
            '<i class="fa-solid fa-ring"></i>',
            html.escape(bridesmaid_name),
            f"{bridesmaid_count}× runner-up, 0 titles",
            "#f59e0b",
        )

    # Most Dominant — best career win% (2+ seasons)
    if not eligible.empty:
        dominant_row = eligible.loc[eligible["Win%"].idxmax()]
        fun_awards_html += _fun_award(
            "Most Dominant",
            '<i class="fa-solid fa-crown"></i>',
            html.escape(str(dominant_row["display_name"])),
            f"{dominant_row['Win%']:.1%} all-time win rate",
            "#f59e0b",
        )

    # The Punching Bag — most PA with a losing record
    losing = career_df[career_df["Losses"] > career_df["Wins"]]
    if not losing.empty:
        punching_bag_row = losing.loc[losing["PA"].idxmax()]
        fun_awards_html += _fun_award(
            "The Punching Bag",
            '<i class="fa-solid fa-dumbbell"></i>',
            html.escape(str(punching_bag_row["display_name"])),
            f"{punching_bag_row['PA']:,.1f} points allowed",
            "#94a3b8",
        )

    # Boom or Bust — highest weekly score std dev
    boom_eligible = career_df[career_df["Seasons"] >= 1].copy()
    if not boom_eligible.empty:
        boom_row = boom_eligible.loc[boom_eligible["STD"].idxmax()]
        fun_awards_html += _fun_award(
            "Boom or Bust",
            '<i class="fa-solid fa-dice"></i>',
            html.escape(str(boom_row["display_name"])),
            f"σ {boom_row['STD']:.1f} pts/week variance",
            "#f97316",
        )

    # Barely Breathing — most wins by <5 points
    if "CloseWins" in career_df.columns and career_df["CloseWins"].sum() > 0:
        close_row = career_df.loc[career_df["CloseWins"].idxmax()]
        if int(close_row["CloseWins"]) > 0:
            fun_awards_html += _fun_award(
                "Barely Breathing",
                '<i class="fa-solid fa-heart-crack"></i>',
                html.escape(str(close_row["display_name"])),
                f"{int(close_row['CloseWins'])} wins by fewer than 5 pts",
                "#ef4444",
            )

    # Consistency King — lowest weekly score std dev (2+ seasons)
    if not eligible.empty:
        consistent_row = eligible.loc[eligible["STD"].idxmin()]
        fun_awards_html += _fun_award(
            "Consistency King",
            '<i class="fa-solid fa-snowflake"></i>',
            html.escape(str(consistent_row["display_name"])),
            f"σ {consistent_row['STD']:.1f} pts/week",
            "#60a5fa",
        )

    # Main Character — most total league activity (trades + pickups)
    if "Activity" in career_df.columns and career_df["Activity"].sum() > 0:
        main_row = career_df.loc[career_df["Activity"].idxmax()]
        if int(main_row["Activity"]) > 0:
            trades = int(main_row.get("Activity", 0) // 3)  # rough trade count from weighted activity
            pickups = int(main_row.get("WaiverAdds", 0))
            fun_awards_html += _fun_award(
                "Main Character",
                '<i class="fa-solid fa-star"></i>',
                html.escape(str(main_row["display_name"])),
                f"{pickups} pickups · {int(main_row['Activity'])} activity pts",
                "#a855f7",
            )

    # Bench Warmer MVP — most career points left on bench
    if "BenchPts" in career_df.columns and career_df["BenchPts"].sum() > 0:
        bench_row = career_df.loc[career_df["BenchPts"].idxmax()]
        if float(bench_row["BenchPts"]) > 0:
            fun_awards_html += _fun_award(
                "Bench Warmer MVP",
                '<i class="fa-solid fa-clipboard-list"></i>',
                html.escape(str(bench_row["display_name"])),
                f"{bench_row['BenchPts']:,.1f} pts left on bench",
                "#64748b",
            )

    # Waiver Wire Demon — most FA/waiver pickups
    if "WaiverAdds" in career_df.columns and career_df["WaiverAdds"].sum() > 0:
        waiver_row = career_df.loc[career_df["WaiverAdds"].idxmax()]
        if int(waiver_row["WaiverAdds"]) > 0:
            fun_awards_html += _fun_award(
                "Waiver Wire Demon",
                '<i class="fa-solid fa-magnifying-glass"></i>',
                html.escape(str(waiver_row["display_name"])),
                f"{int(waiver_row['WaiverAdds'])} career pickups",
                "#22c55e",
            )

    # Playoff Riser — biggest avg pts jump from regular season to playoffs
    po_eligible = career_df[career_df["PlayoffDelta"].notna()].copy() if "PlayoffDelta" in career_df.columns else pd.DataFrame()
    if not po_eligible.empty:
        riser_row = po_eligible.loc[po_eligible["PlayoffDelta"].idxmax()]
        delta = float(riser_row["PlayoffDelta"])
        if delta > 0:
            fun_awards_html += _fun_award(
                "Playoff Riser",
                '<i class="fa-solid fa-arrow-trend-up"></i>',
                html.escape(str(riser_row["display_name"])),
                f"+{delta:.1f} pts/wk in playoffs",
                "#16a34a",
            )

    fun_awards_section = ""
    if fun_awards_html:
        fun_awards_section = f"""
    <div class="card">
      <div class="card-header"><h2>League Superlatives</h2></div>
      <div class="card-body">
        <div class="fun-awards-grid">{fun_awards_html}</div>
      </div>
    </div>"""

    history_url = f"/{platform}/{season}/{league_id}/history" if platform and league_id else "#"

    nav_row = f"""
    <div style="display:flex;justify-content:flex-end;margin-bottom:4px;">
      <a href="{history_url}" class="awards-page-nav-link">
        <i class="fa-solid fa-clipboard-list"></i>
        Season History
      </a>
    </div>"""

    return f"""
    <div class="overview-layout">
      <div class="overview-main">
        {nav_row}
        {highlights_section}
        <div class="awards-two-col">
          {fun_awards_section}
          {champ_table}
        </div>
        {standings_table}
      </div>
    </div>"""


@app.route("/<platform>/<int:season>/<league_id>/awards")
def page_awards(platform: str, season: int, league_id: str):
    cached = get_page_html_from_cache(platform, season, league_id, "awards")
    if cached:
        return render_page("League Awards", league_id, "awards", cached, platform, season)

    available, career_owners, championships, season_records, user_id_to_name = \
        _collect_all_season_data(platform, league_id, season)

    if not available or not career_owners:
        body_html = """
        <div class="card central">
          <div class="card-body">
            <div class="bract-empty-state">
              <div class="bract-empty-title">No History Yet</div>
              <div class="bract-empty-copy">
                All-time awards will appear after your first completed season.
              </div>
            </div>
          </div>
        </div>"""
        store_page_html(platform, season, league_id, "awards", body_html)
        return render_page("League Awards", league_id, "awards", body_html, platform, season)

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        _league_name = (ctx.get("league") or {}).get("name") or ""
    except Exception:
        _league_name = ""
    body_html = _build_awards_html(
        career_owners, championships, season_records, user_id_to_name,
        platform=platform, season=season, league_id=league_id, league_name=_league_name,
    )
    store_page_html(platform, season, league_id, "awards", body_html)
    return render_page("League Awards", league_id, "awards", body_html, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/history")
def page_history(platform: str, season: int, league_id: str):
    # Tour preview: render with mock data, bypass real league fetch
    if request.args.get("tour"):
        try:
            mock_ctx = _build_tour_mock_history_ctx()
            body_html = build_history_body(
                history_ctx=mock_ctx,
                available_seasons=[2024],
                base_platform=platform,
                base_season=season,
                base_league_id=league_id,
                selected_history_season=2024,
                resolved_history_league_id="tour_mock",
            )
        except Exception as exc:
            body_html = f"<div class='card central'><div class='card-body'><p>History preview unavailable: {exc}</p></div></div>"
        return render_page("League History", league_id, "history", body_html, platform, season)

    selected_history_season_param = request.args.get("history_season")
    page_cache_key = f"history:{selected_history_season_param}" if selected_history_season_param else "history"
    cached = get_page_html_from_cache(platform, season, league_id, page_cache_key)
    if cached:
        return render_page("League History", league_id, "history", cached, platform, season)

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
    store_page_html(platform, season, league_id, page_cache_key, body_html)

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
    try:
        today_et: date = datetime.now(EASTERN).date()

        if daily_completed == today_et:
            return

        if daily_lock.acquire(blocking=False):
            try:
                if daily_completed != today_et:
                    logger.info("[daily] Running daily data process for %s (ET)...", today_et)

                    state = get_nfl_state() or {}
                    season = int(state.get("season") or datetime.now().year)
                    week = int(state.get("week") or 0)

                    daily_thread = threading.Thread(
                        target=run_daily_data_async,
                        args=(season, week),
                        daemon=True
                    )
                    daily_thread.start()

                    daily_completed = today_et
            finally:
                daily_lock.release()
    except Exception as _daily_exc:
        logger.warning("[daily] before_request check failed (non-fatal): %s", _daily_exc)


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
                if platform == "sleeper" and viewer.get("viewer_user_id"):
                    _background_seed_user(viewer["viewer_user_id"], viewer.get("viewer_username"))
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

        # Preload historical season contexts in the background so History/Awards/Graphs
        # pages are fast on first click.
        def _preload_history(p: str, lid: str, s: int) -> None:
            try:
                hist_seasons = get_available_history_seasons(p, lid, s)
                for hist_s in hist_seasons:
                    rid = resolve_league_id_for_season(p, lid, s, hist_s)
                    get_league_ctx_from_cache(p, rid, hist_s)
            except Exception:
                pass

        threading.Thread(
            target=_preload_history, args=(platform, league_id, season), daemon=True
        ).start()

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
    if platform == "sleeper" and viewer.get("viewer_user_id"):
        _background_seed_user(viewer["viewer_user_id"], viewer.get("viewer_username"))
    return redirect(url_for("page_dashboard", platform=platform, season=season, league_id=league_id))


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
    tbl = list(load_model_value_table() or [])

    # Append rookie prospects (mirrors /api/league-players logic)
    try:
        from data_building.rookie_pipeline.pipeline import get_rookie_rankings_from_db, get_active_rookie_class
        from utils.utils import normalize_name as _nn
        draft_year = get_active_rookie_class()
        for r in get_rookie_rankings_from_db(draft_year):
            name = r.get("name") or ""
            tbl.append({
                "id": r.get("player_id") or f"rookie_{name}",
                "name": name,
                "team": r.get("team") or "FA",
                "position": r.get("position") or "UNK",
                "age": r.get("age"),
                "value":    float(r.get("rookie_value") or 0),
                "sf_value": float(r.get("rookie_sf_value") or r.get("rookie_value") or 0),
                "is_rookie": True,
            })
    except Exception as e:
        print(f"[model-value-cache] rookies skipped: {e}")

    _MODEL_VALUE_CACHE = tbl
    _MODEL_VALUE_CACHE_TS = now
    return tbl


@app.route("/api/gm-memo", methods=["POST"])
@limiter.limit("10 per minute")
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
        logger.exception("[api-gm-memo] Error: %s", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/power-rankings", methods=["POST"])
@limiter.limit("6 per minute")
def api_power_rankings():
    payload = request.get_json(force=True)
    league_id = str(payload.get("league_id") or "").strip()
    season = int(payload.get("season") or datetime.now().year)
    platform = str(payload.get("platform") or "sleeper").strip()

    if not league_id:
        return jsonify({"error": "Missing league_id"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        html_out = get_power_rankings_html(ctx)
        return jsonify({"success": True, "html": html_out})
    except Exception as e:
        logger.exception("[api-power-rankings] Error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/trade-suggestions", methods=["POST"])
@limiter.limit("6 per minute")
def api_trade_suggestions():
    payload = request.get_json(force=True)
    league_id = str(payload.get("league_id") or "").strip()
    season = int(payload.get("season") or datetime.now().year)
    platform = str(payload.get("platform") or "sleeper").strip()
    viewer_roster_id = str(payload.get("viewer_roster_id") or "").strip()

    if not league_id or not viewer_roster_id:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        html_out = get_trade_suggestions_html(ctx, viewer_roster_id)
        return jsonify({"success": True, "html": html_out})
    except Exception as e:
        logger.exception("[api-trade-suggestions] Error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/roster-grade", methods=["POST"])
@limiter.limit("10 per minute")
def api_roster_grade():
    payload = request.get_json(force=True)
    league_id = str(payload.get("league_id") or "").strip()
    season = int(payload.get("season") or datetime.now().year)
    platform = str(payload.get("platform") or "sleeper").strip()
    viewer_roster_id = str(payload.get("viewer_roster_id") or "").strip()

    if not league_id or not viewer_roster_id:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        grade_data = get_roster_grade(ctx, viewer_roster_id)
        badge_html = render_roster_grade_badge(grade_data)
        return jsonify({"success": True, "grade_data": grade_data, "badge_html": badge_html})
    except Exception as e:
        logger.exception("[api-roster-grade] Error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/trade-outcome", methods=["POST"])
@limiter.limit("10 per minute")
def api_trade_outcome():
    """
    Compare the value delta of a past trade vs current values.
    Expects: {assets_received: [{id, name}], assets_sent: [{id, name}], trade_date: 'YYYY-MM-DD'}
    """
    from dashboard_services.player_value_history import get_player_value_history
    from dashboard_services.db import get_conn

    payload = request.get_json(force=True)
    assets_received = payload.get("assets_received") or []
    assets_sent = payload.get("assets_sent") or []

    if not assets_received and not assets_sent:
        logger.warning("[api-trade-outcome] 400 error: No assets provided. Payload: %s", payload)
        return jsonify({"error": "No assets provided", "debug": payload}), 400

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from dashboard_services.picks import load_pick_value_table
        
        # Load calibrated values: weighted_market_value_1qb from trade intel (primary),
        # falling back to calibrated_value_1qb / value_1qb from player_values.
        try:
            current_season = datetime.now().year
            with get_conn() as conn:
                cal_rows = conn.execute(
                    """
                    SELECT
                        pv.player_id,
                        COALESCE(
                            tips.weighted_market_value_1qb,
                            pv.calibrated_value_1qb,
                            pv.value_1qb
                        ) AS calibrated_value,
                        pv.position
                    FROM player_values pv
                    LEFT JOIN trade_intel_player_stats tips
                        ON tips.player_id = pv.player_id AND tips.season = %s
                    WHERE pv.player_id IS NOT NULL
                    """,
                    (current_season,),
                ).fetchall()
            if not cal_rows:
                raise ValueError("No calibrated values loaded")
            values_now = {}
            pick_count = 0
            player_count = 0
            for row in cal_rows:
                asset_id = str(row["player_id"] or "")
                if not asset_id:
                    continue
                value = float(row["calibrated_value"] or 0.0)
                values_now[asset_id] = value
                if str(row.get("position") or "").upper() == "PICK":
                    pick_count += 1
                    if "_" in asset_id:
                        space_format_id = asset_id.replace("_", " ", 1).replace("_", ".", 1)
                        values_now[space_format_id] = value
                else:
                    player_count += 1
        except Exception as db_error:
            logger.warning("[api-trade-outcome] Calibrated values unavailable, falling back to model table: %s", db_error)
            value_table = get_model_value_table_cached()
            values_now = {str(p["id"]): float(p.get("value") or 0) for p in value_table if isinstance(p, dict) and p.get("id")}
        
        # Load pick values for pick asset handling
        pick_values = load_pick_value_table()

        trade_date = str(payload.get("trade_date") or "")
        trade_month = trade_date[:7] if trade_date else ""

        def get_value_at_trade(pid: str) -> float:
            """Return the closest historical value to trade_date; returns 0.0 if no history."""
            if not trade_date:
                return 0.0
            from datetime import date as _date
            try:
                target = _date.fromisoformat(trade_date[:10])
            except ValueError:
                return 0.0
            history = get_player_value_history(pid, days=800)
            if not history:
                return 0.0
            best_val = 0.0
            best_diff = float("inf")
            for snap in history:
                snap_date_str = str(snap.get("as_of_date") or "")[:10]
                if not snap_date_str:
                    continue
                try:
                    diff = abs((_date.fromisoformat(snap_date_str) - target).days)
                    if diff < best_diff:
                        best_diff = diff
                        best_val = float(snap.get("value") or 0)
                except (ValueError, TypeError):
                    continue
            print(best_val)
            return best_val
        
        def get_pick_value(asset: dict) -> float:
            """Get current pick value, preferring WLS-derived bucket values."""
            try:
                rd = int(asset.get("pick_round") or 4)
            except (ValueError, TypeError):
                rd = 4
            try:
                year = int(asset.get("pick_season") or asset.get("pick_year") or
                           (trade_date[:4] if trade_date else datetime.now().year))
            except (ValueError, TypeError):
                year = datetime.now().year

            # Exact slot lookup (e.g. "2026_1_06")
            slot = asset.get("pick_slot")
            if slot:
                try:
                    key = f"{year}_{rd}_{int(slot):02d}"
                    if key in pick_values:
                        return float(pick_values[key])
                except (ValueError, TypeError):
                    pass

            # Bucket lookup — derive bucket from slot if pick_order is absent
            order = asset.get("pick_order")
            if not order and slot:
                try:
                    s = int(slot)
                    order = "early" if s <= 4 else ("mid" if s <= 8 else "late")
                except (ValueError, TypeError):
                    pass
            order = order or "mid"

            if order in ("early", "mid", "late"):
                key = f"{year}_{rd}_{order}"
                if key in pick_values:
                    return float(pick_values[key])

            key = f"{year}_{rd}"
            if key in pick_values:
                return float(pick_values[key])

            return 10.0

        all_assets = [("received", a) for a in assets_received] + [("sent", a) for a in assets_sent]
        all_pids = [(side, str(a.get("id") or ""), str(a.get("name") or a.get("id") or "")) for side, a in all_assets]

        # Fetch historical values in parallel
        then_values: dict[str, float] = {}
        if trade_date:
            with ThreadPoolExecutor(max_workers=min(len(all_pids), 8)) as pool:
                futures = {pool.submit(get_value_at_trade, pid): pid for _, pid, _ in all_pids if pid}
                for fut in as_completed(futures):
                    pid = futures[fut]
                    try:
                        then_values[pid] = fut.result()
                    except Exception:
                        then_values[pid] = 0.0

        received_rows = []
        sent_rows = []
        total_received_now = 0.0
        total_sent_now = 0.0
        total_received_then = 0.0
        total_sent_then = 0.0

        def _pick_now_value(pid: str, asset: dict) -> float:
            """Look up calibrated pick value trying multiple ID formats."""
            # Try as-is (e.g. "2026 1.01")
            v = values_now.get(pid, 0.0)
            if v:
                return v
            # Try underscore format (e.g. "2026_1_01")
            underscore_pid = pid.replace(" ", "_", 1).replace(".", "_", 1)
            v = values_now.get(underscore_pid, 0.0)
            if v:
                return v
            # Fall back to pick table
            return get_pick_value(asset)

        def _build_row(asset: dict, side: str) -> dict:
            pid = str(asset.get("id") or "")
            name = str(asset.get("name") or pid)
            is_pick = asset.get("asset_type") == "pick"
            if is_pick:
                now = _pick_now_value(pid, asset)
                then = None
            else:
                now = values_now.get(pid, 0.0)
                then = then_values.get(pid, None) if trade_date else None
            return {
                "id": pid, "name": name, "is_pick": is_pick,
                "value_now": round(now, 1),
                "value_then": round(then, 1) if then is not None else None,
                "delta": round(now - then, 1) if then is not None else None,
                "_now": now, "_then": then,
            }

        for asset in assets_received:
            row = _build_row(asset, "received")
            total_received_now += row["_now"]
            if row["_then"] is not None:
                total_received_then += row["_then"]
            del row["_now"], row["_then"]
            received_rows.append(row)

        for asset in assets_sent:
            row = _build_row(asset, "sent")
            total_sent_now += row["_now"]
            if row["_then"] is not None:
                total_sent_then += row["_then"]
            del row["_now"], row["_then"]
            sent_rows.append(row)

        net_delta_now = round(total_received_now - total_sent_now, 1)
        net_delta_then = round(total_received_then - total_sent_then, 1)

        if net_delta_now > 150:
            verdict = "WIN"
        elif net_delta_now < -150:
            verdict = "LOSS"
        else:
            verdict = "EVEN"

        return jsonify({
            "success": True,
            "verdict": verdict,
            "net_delta_now": net_delta_now,
            "net_delta_then": net_delta_then,
            "total_received_now": round(total_received_now, 1),
            "total_sent_now": round(total_sent_now, 1),
            "received": received_rows,
            "sent": sent_rows,
        })
    except Exception as e:
        logger.exception("[api-trade-outcome] Error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/trade-eval", methods=["POST"])
@limiter.limit("10 per minute")
def api_trade_eval():
    payload = request.get_json(force=True)

    league_id = str(payload.get("league_id") or "").strip()
    season = int(payload.get("season") or datetime.now().year)
    platform = str(payload.get("platform") or "sleeper").strip().lower()
    league_type = str(payload.get("league_type") or "1qb").strip().lower()
    league_size = int(payload.get("league_size") or 10)
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

            # Use size/type-specific value, falling back to 10-team then base value
            if league_type == "sf":
                size_key = "sf_value" if league_size == 10 else f"sf_value_{league_size}"
                val = float(player.get(size_key) or player.get("sf_value") or player.get("value") or 0.0)
            else:
                size_key = "value" if league_size == 10 else f"value_{league_size}"
                val = float(player.get(size_key) or player.get("value") or 0.0)

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
    depth_warnings = {}
    viewer_roster_id = payload.get("viewer_roster_id")
    viewer_team_name = payload.get("viewer_team_name")

    if league_id and viewer_roster_id:
        try:
            from dashboard_services.ai.context_builders import calculate_roster_depth_warning, build_model_value_lookup
            ctx = get_league_ctx_from_cache(platform=platform, league_id=league_id, season=season)
            analysis_html = get_trade_ai_analysis(
                ctx=ctx,
                viewer_roster_id=str(viewer_roster_id),
                viewer_side=viewer_side,
                side_a=side_a,
                side_b=side_b,
            )
            # Compute post-trade depth warnings for the viewer's roster
            rosters = ctx.get("rosters") or []
            viewer_roster = next((r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)), None)
            if viewer_roster:
                model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [])
                viewer_gives = side_b if viewer_side == "b" else side_a
                viewer_gets = side_a if viewer_side == "a" else side_b
                sending = [a for a in (viewer_gives.get("assets") or []) if str(a.get("position") or "").upper() != "PICK"]
                receiving = [a for a in (viewer_gets.get("assets") or []) if str(a.get("position") or "").upper() != "PICK"]
                depth_warnings = calculate_roster_depth_warning(viewer_roster, model_value_lookup, sending, receiving)
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
        "depth_warnings": depth_warnings,
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


@app.route("/api/players")
def api_players():
    """Compact player list for comparison search. No league context required.

    Query params:
        page  (int, default 1)   — 1-based page number
        limit (int, default 0)   — results per page; 0 = return all (legacy)
        q     (str, optional)    — prefix/substring filter applied before paging
    """
    try:
        from utils.utils import load_players_index, load_model_value_table
        players_index = load_players_index() or {}
        value_table = load_model_value_table() or []
        value_map = {str(p.get("id")): p for p in value_table}

        results = []
        for pid, meta in players_index.items():
            pos = meta.get("pos", "")
            if pos in ("K", "DEF"):
                continue
            v = value_map.get(str(pid), {})
            results.append({
                "player_id": pid,
                "name": meta.get("name", ""),
                "position": pos,
                "team": meta.get("team", ""),
                "value": v.get("value", 0),
                "sf_value": v.get("sf_value", 0),
                "pos_rank_label": v.get("pos_rank_label", ""),
                "rank_change_7d": v.get("rank_change_7d"),
                "espnHeadshot": meta.get("espnHeadshot", ""),
            })

        # Sort by value descending so most relevant players appear first
        results.sort(key=lambda x: x["value"] or 0, reverse=True)

        # Optional substring filter
        q = request.args.get("q", "").strip().lower()
        if q:
            results = [r for r in results if q in r["name"].lower()]

        # Pagination — limit=0 (default) returns the full list for backwards compat
        total = len(results)
        try:
            limit = max(0, int(request.args.get("limit", 0)))
            page  = max(1, int(request.args.get("page", 1)))
        except (TypeError, ValueError):
            limit, page = 0, 1

        if limit > 0:
            start   = (page - 1) * limit
            results = results[start: start + limit]
            return jsonify({
                "players": results,
                "total":   total,
                "page":    page,
                "pages":   math.ceil(total / limit),
            })

        return jsonify(results)
    except Exception as e:
        logger.exception("[api-players] Unexpected error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/league-players")
def api_league_players():
    # Force loading from database to get correct position ranks
    try:
        from dashboard_services.player_value_history import load_current_values_from_db
        model_value_table = load_current_values_from_db()
        if model_value_table:
            print(f"[api/league-players] Loaded {len(model_value_table)} players from database")
        else:
            print("[api/league-players] No data from database, falling back to JSON")
            model_value_table = list(load_model_value_table() or [])
    except Exception as e:
        print(f"[api/league-players] Database load failed: {e}, falling back to JSON")
        model_value_table = list(load_model_value_table() or [])
    
    if not isinstance(model_value_table, list):
        raise ValueError("model_value_table must be a list of player objects")

    # Compute rank_change_7d from player-only pool (QB/RB/WR/TE) so that picks
    # and newly-added rookies don't distort movement arrows on the rankings page.
    # Current rank = position in value-sorted player list; historical rank from DB snapshot.
    _PLAYER_POSITIONS = {"QB", "RB", "WR", "TE"}
    try:
        from data_building.update_player_values_with_rankings import _load_historical_ranks as _lhr
        from datetime import timedelta as _td

        # Cache historical ranks by date so we don't hit DB on every request
        _today = date.today()
        _hist_cache_key = f"_hist_ranks_{_today}"
        _hist_ranks = getattr(app, _hist_cache_key, None)
        if _hist_ranks is None:
            _hist_ranks = _lhr(_today - _td(days=7))
            setattr(app, _hist_cache_key, _hist_ranks)

        # Current player-only rank: sort QB/RB/WR/TE by value descending
        _player_rows = sorted(
            [p for p in model_value_table
             if isinstance(p, dict) and str(p.get("position", "")).upper() in _PLAYER_POSITIONS],
            key=lambda p: float(p.get("value") or 0),
            reverse=True,
        )
        _cur_rank_map = {str(p.get("id") or ""): idx + 1 for idx, p in enumerate(_player_rows)}

        for _p in model_value_table:
            _pid = str(_p.get("id") or "")
            _cur = _cur_rank_map.get(_pid)
            _hist = _hist_ranks.get(_pid)
            if _cur is not None and _hist:
                _p["rank_change_7d"] = _hist["overall_rank"] - _cur
            # leave rank_change_7d as-is (None or from JSON) for non-player-pos entries
    except Exception:
        # Fall back to DB-stored values if recomputation fails
        try:
            from dashboard_services.db import get_conn as _gc
            with _gc() as _rc:
                _rk_rows = _rc.execute(
                    "SELECT player_id, rank_change_7d FROM player_values WHERE rank_change_7d IS NOT NULL"
                ).fetchall()
            _rk_map = {str(r["player_id"]): r["rank_change_7d"] for r in _rk_rows}
            for _p in model_value_table:
                _pid = str(_p.get("id") or "")
                if _pid in _rk_map:
                    _p["rank_change_7d"] = _rk_map[_pid]
        except Exception:
            pass

    try:
        from data_building.rookie_pipeline.pipeline import (
            get_rookie_rankings_from_db,
            get_active_rookie_class,
        )
        from utils.utils import normalize_name as _nn
        draft_year = get_active_rookie_class()

        # Build a normalized-name → index map for the model table
        name_to_idx: dict = {}
        for i, p in enumerate(model_value_table):
            norm = _nn(p.get("name") or "")
            if norm:
                name_to_idx[norm] = i

        for r in get_rookie_rankings_from_db(draft_year):
            name = r.get("name") or ""
            norm = _nn(name)
            rookie_entry = {
                "id": r.get("player_id") or f"rookie_{name}",
                "name": name,
                "team": r.get("actual_nfl_team") or r.get("team") or "FA",
                "position": r.get("position") or "UNK",
                "age": r.get("age"),
                "value":       float(r.get("rookie_value")       or 0),
                "sf_value":    float(r.get("rookie_sf_value")    or r.get("rookie_value") or 0),
                "value_8":     float(r.get("rookie_value_8")     or r.get("rookie_value") or 0),
                "value_12":    float(r.get("rookie_value_12")    or r.get("rookie_value") or 0),
                "value_14":    float(r.get("rookie_value_14")    or r.get("rookie_value") or 0),
                "sf_value_8":  float(r.get("rookie_sf_value_8")  or r.get("rookie_sf_value") or 0),
                "sf_value_12": float(r.get("rookie_sf_value_12") or r.get("rookie_sf_value") or 0),
                "sf_value_14": float(r.get("rookie_sf_value_14") or r.get("rookie_sf_value") or 0),
                "pos_rank": None, "pos_rank_label": None,
                "sf_pos_rank": None, "sf_pos_rank_label": None,
                "search_name": norm,
                "is_rookie": True,
                "rank_change_7d": r.get("rank_change_7d"),
            }

            if norm and norm in name_to_idx:
                # Player already in model table (drafted + linked to Sleeper).
                # Mark as rookie and back-fill rookie values so they appear in
                # the ROOKIE tab with correct values; don't add a second entry.
                existing = model_value_table[name_to_idx[norm]]
                existing["is_rookie"] = True
                existing.setdefault("value",       rookie_entry["value"])
                existing.setdefault("sf_value",    rookie_entry["sf_value"])
                existing.setdefault("value_8",     rookie_entry["value_8"])
                existing.setdefault("value_12",    rookie_entry["value_12"])
                existing.setdefault("value_14",    rookie_entry["value_14"])
                existing.setdefault("sf_value_8",  rookie_entry["sf_value_8"])
                existing.setdefault("sf_value_12", rookie_entry["sf_value_12"])
                existing.setdefault("sf_value_14", rookie_entry["sf_value_14"])
            else:
                model_value_table.append(rookie_entry)

    except Exception as _e:
        print(f"[api/league-players] rookies skipped: {_e}")

    # --- Depth-decay pass ---
    # Recompute pos_rank from current calibrated values first so decay tiers
    # use correct ranks rather than stale DB values.
    from collections import defaultdict as _dd_prl
    _pos_groups: dict = _dd_prl(list)
    for _i, _p in enumerate(model_value_table):
        _pos = str(_p.get("position") or "").upper()
        if _pos and _pos != "PICK":
            _pos_groups[_pos].append(_i)
    for _pos, _idxs in _pos_groups.items():
        _idxs.sort(key=lambda _i: float(model_value_table[_i].get("value") or 0), reverse=True)
        for _rank, _i in enumerate(_idxs, 1):
            model_value_table[_i]["pos_rank"] = _rank
            model_value_table[_i]["pos_rank_label"] = f"{_pos}{_rank}"

    # Decay tables: (rank_threshold, multiplier). Beyond the last threshold the
    # final multiplier applies.  QB only decays 1QB value (sf_value untouched).
    # RB/WR/TE decay all value fields — depth penalty applies in any format.
    _DEPTH_DECAY = {
        "QB": [(12, 1.00), (18, 0.82), (24, 0.65), (36, 0.45), (48, 0.32), (9999, 0.22)],
        "RB": [(30, 1.00), (42, 0.88), (54, 0.72), (72, 0.55), (9999, 0.40)],
        "WR": [(36, 1.00), (48, 0.88), (60, 0.73), (80, 0.57), (9999, 0.42)],
        "TE": [(12, 1.00), (18, 0.85), (24, 0.68), (36, 0.50), (9999, 0.36)],
    }
    _QB_VAL_KEYS   = ["value"]
    _SKILL_VAL_KEYS = [
        "value", "sf_value",
        "value_8", "value_12", "value_14",
        "sf_value_8", "sf_value_12", "sf_value_14",
    ]
    for _p in model_value_table:
        _pos   = str(_p.get("position") or "").upper()
        _tiers = _DEPTH_DECAY.get(_pos)
        if not _tiers:
            continue
        _rank = int(_p.get("pos_rank") or 999)
        _keys = _QB_VAL_KEYS if _pos == "QB" else _SKILL_VAL_KEYS
        for _thresh, _factor in _tiers:
            if _rank <= _thresh:
                if _factor < 1.0:
                    for _vk in _keys:
                        if _p.get(_vk) is not None:
                            _p[_vk] = round(float(_p[_vk]) * _factor, 1)
                break

    # Recompute pos_rank / pos_rank_label a second time so ranks reflect
    # post-decay values (ordering within a tier is preserved, but this keeps
    # labels accurate if cross-tier compression shifts any players).
    _pos_groups2: dict = _dd_prl(list)
    for _i, _p in enumerate(model_value_table):
        _pos = str(_p.get("position") or "").upper()
        if _pos and _pos != "PICK":
            _pos_groups2[_pos].append(_i)
    for _pos, _idxs in _pos_groups2.items():
        _idxs.sort(key=lambda _i: float(model_value_table[_i].get("value") or 0), reverse=True)
        for _rank, _i in enumerate(_idxs, 1):
            model_value_table[_i]["pos_rank"] = _rank
            model_value_table[_i]["pos_rank_label"] = f"{_pos}{_rank}"

    # Sort players: first by value (descending), then by pos_rank (ascending for ties)
    model_value_table.sort(key=lambda p: (
        -(float(p.get("value") or 0)),  # Negative for descending sort by value
        int(p.get("pos_rank") or 9999)  # Lower pos_rank = better position
    ))

    return jsonify(_sanitize_for_json(model_value_table))


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
                             league_size=league_size, min_baseline_value=10) or {}

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

        # Load all players to check for rookies (years_exp 0 or 1 = first two seasons)
        players_index = load_players_index() or {}
        rookies = []

        for player_id, player_data in players_index.items():
            years_exp = player_data.get("years_exp")
            rookie_year = player_data.get("rookie_year")

            if years_exp in (0, 1, "0", "1"):
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

                offseason_candidates = get_offseason_breakout_candidates(current_season, min_score=25)
                breakouts = [str(c["player_id"]) for c in offseason_candidates]

            except Exception as e:
                print(f"[player-indicators] Offseason breakout detection failed: {e}")

        else:
            # During season, use in-season breakout detection
            try:
                from data_building.advanced_metrics import detect_breakout_candidates

                breakout_candidates = detect_breakout_candidates(lookback_days=14)
                breakouts = [str(b["player_id"]) for b in breakout_candidates]

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

        # Get elites based on positional rank cutoffs (12-man PPR dynasty)
        elites = []

        # Load model value table to get current player values
        value_table = load_model_value_table() or []
        value_map = {str(p.get("id")): p for p in value_table}

        # Top-N positional rank cutoffs for a 12-man PPR dynasty league
        elite_rank_cutoffs = {'QB': 5, 'RB': 6, 'WR': 6, 'TE': 5}

        from collections import defaultdict as _defaultdict
        pos_players: dict = _defaultdict(list)
        for player_id, player_data in value_map.items():
            pos = str(player_data.get("position", "")).upper()
            val = float(player_data.get("value", 0) or 0)
            if val > 0 and pos in elite_rank_cutoffs:
                pos_players[pos].append((val, str(player_id)))

        for pos, cutoff in elite_rank_cutoffs.items():
            for _, pid in sorted(pos_players[pos], reverse=True)[:cutoff]:
                elites.append(pid)

        # Prospects = only pre-draft class players (is_rookie=True in cached table)
        prospects = []
        try:
            model_tbl = get_model_value_table_cached() or []
            for entry in model_tbl:
                if entry.get("is_rookie") is True:
                    pid = str(entry.get("id") or "")
                    if pid:
                        prospects.append(pid)
        except Exception as _pe:
            print(f"[player-indicators] prospects skipped: {_pe}")

        return jsonify({
            "rookies": rookies,
            "breakouts": breakouts,
            "elites": elites,
            "prospects": prospects
        })

    except Exception as e:
        print(f"[player-indicators] Error: {e}")
        return jsonify({"rookies": [], "breakouts": [], "elites": [], "prospects": []})


@app.route("/api/prospect/<player_id>")
def api_prospect_profile(player_id: str):
    """Return full prospect profile data for a single pre-draft player."""
    try:
        from data_building.rookie_pipeline.pipeline import get_rookie_rankings_from_db, get_active_rookie_class
        draft_year = get_active_rookie_class()
        rows = get_rookie_rankings_from_db(draft_year)
        player_id = str(player_id).strip()
        for r in rows:
            if str(r.get("player_id") or "") == player_id:
                d = dict(r)
                d["draft_capital_label"] = (
                    f"Round {d['projected_round']} · Pick #{d['projected_pick']}"
                    if d.get("projected_round") and d.get("projected_pick") else None
                )
                return jsonify(_sanitize_for_json(d))
        return jsonify({"error": "not found"}), 404
    except Exception as e:
        print(f"[api/prospect] {e}")
        return jsonify({"error": str(e)}), 500


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


@app.route("/api/nfl-state")
def api_nfl_state():
    """Get current NFL state from Sleeper API."""
    try:
        from dashboard_services.api import get_nfl_state
        state = get_nfl_state()
        return jsonify(state or {})
    except Exception as e:
        print(f"[nfl-state] Error: {e}")
        return jsonify({}), 500


@app.route("/api/player-advanced-metrics/<player_id>")
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


@app.route("/api/player-details/<player_id>")
def api_player_details(player_id: str):
    """Get comprehensive player details for modal display."""
    try:
        from utils.utils import load_relevant_index, load_model_value_table
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
        _modal_lt = str(request.args.get("league_type", "1qb")).strip().lower()
        try:
            _modal_ls = int(request.args.get("league_size", 10))
            if _modal_ls not in (8, 10, 12, 14):
                _modal_ls = 10
        except (TypeError, ValueError):
            _modal_ls = 10

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

        players_index = load_relevant_index() or {}
        player_meta = players_index.get(player_id, {})

        if not player_meta:
            return jsonify({"error": "Player not found"}), 404

        player_team = player_meta.get("team", "")

        # Get value data
        value_table = load_model_value_table() or []
        player_value = next((p for p in value_table if str(p.get("id")) == str(player_id)), {})

        # Get FULL value history from database (not just 90 days)
        value_history = get_player_value_history(
            player_id, days=365,
            league_type=_modal_lt, league_size=_modal_ls,
        )

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

        # Try to attach prospect data — use game_logs to detect rookies rather than years_exp
        prospect_data = None
        has_game_logs = bool(game_logs_by_year)
        if not has_game_logs:
            try:
                import re as _re
                from dashboard_services.rookie_api import _cache as _rookie_cache
                from data_building.rookie_pipeline.pipeline import get_active_rookie_class
                from data_building.rookie_pipeline.value_translation import format_draft_capital

                active_year = get_active_rookie_class()
                found_row = None
                for check_year in [active_year, active_year - 1]:
                    if check_year not in _rookie_cache:
                        from data_building.rookie_pipeline.pipeline import get_rookie_rankings_from_db
                        _rookie_cache[check_year] = get_rookie_rankings_from_db(check_year)
                    for r in _rookie_cache.get(check_year, []):
                        if str(r.get("sleeper_id") or "") == str(player_id):
                            found_row = r
                            break
                    if found_row:
                        break

                # Fallback: match by name from players_index when sleeper_id not yet linked
                if not found_row:
                    def _norm_name(n):
                        n = n.lower()
                        n = _re.sub(r"['\.\-]", "", n)
                        n = _re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
                        return _re.sub(r"\s+", " ", n).strip()

                    players_idx = get_players_index_global() or {}
                    idx_entry = players_idx.get(str(player_id)) or {}
                    idx_name = idx_entry.get("name", "")
                    if idx_name:
                        norm_target = _norm_name(idx_name)
                        for check_year in [active_year, active_year - 1]:
                            for r in _rookie_cache.get(check_year, []):
                                if _norm_name(r.get("name", "")) == norm_target:
                                    found_row = r
                                    # Cache the link so future calls use sleeper_id
                                    r["sleeper_id"] = str(player_id)
                                    break
                            if found_row:
                                break

                if found_row:
                    def _sf(v):
                        try:
                            return float(v) if v is not None else None
                        except (TypeError, ValueError):
                            return None

                    prospect_data = {
                        "player_id":                     found_row.get("player_id"),
                        "draft_class_year":              found_row.get("draft_class_year"),
                        "school":                        found_row.get("school"),
                        "prospect_score":                _sf(found_row.get("prospect_score")),
                        "tier":                          found_row.get("tier"),
                        "tier_label":                    found_row.get("tier_label"),
                        "overall_rank":                  found_row.get("overall_rank"),
                        "position_rank":                 found_row.get("position_rank"),
                        "production_score":              _sf(found_row.get("production_score")),
                        "efficiency_score":              _sf(found_row.get("efficiency_score")),
                        "age_score":                     _sf(found_row.get("age_score")),
                        "breakout_profile_score":        _sf(found_row.get("breakout_profile_score")),
                        "athleticism_score":             _sf(found_row.get("athleticism_score")),
                        "competition_score":             _sf(found_row.get("competition_score")),
                        "projected_draft_capital_score": _sf(found_row.get("projected_draft_capital_score")),
                        "confidence_score":              _sf(found_row.get("confidence_score")),
                        "key_reasons":                   found_row.get("key_reasons"),
                        "rookie_value":                  _sf(found_row.get("rookie_value")),
                        "rookie_sf_value":               _sf(found_row.get("rookie_sf_value")),
                        "projected_round":               found_row.get("projected_round"),
                        "projected_pick":                found_row.get("projected_pick"),
                        "num_mocks_used":                found_row.get("num_mocks_used"),
                        "height_inches":                 found_row.get("height_inches"),
                        "weight_lbs":                    found_row.get("weight_lbs"),
                        "forty_yard":                    _sf(found_row.get("forty_yard")),
                        "ras_score":                     _sf(found_row.get("ras_score")),
                        "draft_capital_label":           format_draft_capital(
                            found_row.get("projected_round"),
                            found_row.get("projected_pick"),
                            found_row.get("projected_pick_low"),
                            found_row.get("projected_pick_high"),
                        ),
                    }
            except Exception as pe:
                print(f"[api_player_details] prospect lookup error: {pe}")

        response = {
            "player_id": player_id,
            "name": player_meta.get("name", "Unknown"),
            "position": player_meta.get("pos"),
            "team": player_meta.get("team"),
            "age": player_value.get("age"),
            "pos_rank": player_value.get("pos_rank"),
            "pos_rank_label": player_value.get("pos_rank_label"),
            "espnHeadshot": player_meta.get("espnHeadshot"),
            "stats": {
                "value": round(player_value.get("value", 0), 1) if player_value.get("value") else None,
                "sf_value": round(player_value.get("sf_value", 0), 1) if player_value.get("sf_value") else None,
                "pos_rank": player_value.get("pos_rank"),
                "pos_rank_label": player_value.get("pos_rank_label"),
                "years_exp": player_meta.get("years_exp"),
            },
            "value_history": value_history,
            "game_logs_by_year": game_logs_by_year,
            "prospect_data": prospect_data,
        }

        return jsonify(response)

    except Exception as e:
        print(f"[api_player_details] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def clean_nan_for_json(obj):
    """Recursively replace NaN values with None for JSON compatibility."""
    import math
    if isinstance(obj, dict):
        return {k: clean_nan_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_for_json(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj


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


            # Get league context for graphs
            ctx = get_league_ctx_from_cache(platform, league_id, graph_season)
            team_stats = ctx.get("team_stats")
            df_weekly = ctx.get("df_weekly")

            if df_weekly is not None and not df_weekly.empty:
                print(f"[api_team_details] df_weekly shape: {df_weekly.shape}")

            # If we don't have weekly data for the chosen season, try previous season
            if (df_weekly is None or df_weekly.empty) and graph_season > 2025:
                fallback_season = graph_season - 1

                # Resolve correct league_id for fallback season
                from dashboard_services.api import resolve_league_id_for_season
                fallback_league_id = resolve_league_id_for_season(
                    platform=platform,
                    league_id=league_id,
                    current_season=current_season,
                    target_season=fallback_season
                )

                ctx = get_league_ctx_from_cache(platform, fallback_league_id, fallback_season)
                team_stats = ctx.get("team_stats")
                df_weekly = ctx.get("df_weekly")
                graph_season = fallback_season
                if df_weekly is not None and not df_weekly.empty:
                    print(f"[api_team_details] Fallback df_weekly shape: {df_weekly.shape}")

            # Remove debug prints for cleaner logs
            if team_stats is not None and df_weekly is not None and not df_weekly.empty:
                # Filter to finalized weeks only (if finalized column exists)
                if "finalized" in df_weekly.columns:
                    df_weekly = df_weekly[df_weekly["finalized"] == True].copy()

                # Only build graphs if we have data after filtering
                if not df_weekly.empty:
                    # Get weekly scores for this team
                    team_weekly = df_weekly[df_weekly["owner"] == team_name] if team_name is not None else pd.DataFrame()
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

                    # Try exact match first
                    team_row = team_stats[team_stats["owner"] == team_name] if team_name is not None else pd.DataFrame()

                    # If no exact match, try fuzzy matching for team name variations
                    if team_row.empty and available_teams and team_name is not None:
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
                    else:
                        print(f"[api_team_details] No graphs generated - team_row is empty for team_name='{team_name}'") if team_name is not None else print("[api_team_details] No graphs generated - team_row is empty for team_name='None'")
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

        # Clean NaN values before JSON serialization
        cleaned_response = clean_nan_for_json(response)
        return jsonify(cleaned_response)

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


@app.route("/api/beat-the-market")
def api_beat_the_market():
    """
    For each roster in a league, compute the sum of 30-day player value deltas
    and compare it to the league-average delta over the same window.

    A positive 'vs_avg' means the roster gained more value than a typical roster.

    Query params: platform, league_id, season, days (default 30), league_type, league_size
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"error": "league_id required"}), 400

    nfl_state = get_nfl_state() or {}
    try:
        season = int(request.args.get("season") or nfl_state.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year

    try:
        days = max(1, int(request.args.get("days", 30)))
    except (TypeError, ValueError):
        days = 30

    league_type = str(request.args.get("league_type", "1qb")).strip().lower()
    try:
        league_size = int(request.args.get("league_size", 10))
        if league_size not in [8, 10, 12, 14]:
            league_size = 10
    except (TypeError, ValueError):
        league_size = 10

    try:
        # Determine the right value field
        if league_type == "sf" and league_size != 10:
            value_expr = f"COALESCE(sf_value_{league_size}, sf_value, value)"
        elif league_type == "sf":
            value_expr = "COALESCE(sf_value, value)"
        elif league_size != 10:
            value_expr = f"COALESCE(value_{league_size}, value)"
        else:
            value_expr = "value"

        from data_building.player_value_history import init_value_history_db, get_latest_snapshot_date
        from dashboard_services.db import get_conn
        from datetime import timedelta as _td

        init_value_history_db()
        latest_date = get_latest_snapshot_date(source="model")
        if not latest_date:
            return jsonify({"error": "No value history available"}), 404

        from datetime import date as _date
        if isinstance(latest_date, _date):
            latest_date_obj = latest_date
        else:
            latest_date_obj = _date.fromisoformat(str(latest_date))

        # Find the closest baseline date within the requested window
        from dashboard_services.db import get_conn
        baseline_date = None
        with get_conn() as conn:
            for candidate_days in range(days, 0, -1):
                target = latest_date_obj - _td(days=candidate_days)
                row = conn.execute(
                    "SELECT MAX(as_of_date) AS d FROM player_value_history "
                    "WHERE source = 'model' AND as_of_date <= %s",
                    (target,)
                ).fetchone()
                if row and row["d"] and row["d"] < latest_date_obj:
                    baseline_date = row["d"]
                    break

        if not baseline_date:
            return jsonify({"error": "Insufficient history for requested window"}), 404

        # Fetch deltas for all players in one query
        with get_conn() as conn:
            rows = conn.execute(
                f"""
                WITH latest AS (
                    SELECT DISTINCT ON (player_id) player_id,
                           {value_expr} AS val
                    FROM player_value_history
                    WHERE source = 'model' AND as_of_date = %s
                    ORDER BY player_id, as_of_date DESC
                ),
                baseline AS (
                    SELECT DISTINCT ON (player_id) player_id,
                           {value_expr} AS val
                    FROM player_value_history
                    WHERE source = 'model' AND as_of_date = %s
                    ORDER BY player_id, as_of_date DESC
                )
                SELECT l.player_id,
                       ROUND((l.val - b.val)::numeric, 1) AS delta
                FROM latest l
                JOIN baseline b ON b.player_id = l.player_id
                """,
                (latest_date_obj, baseline_date)
            ).fetchall()

        delta_by_pid = {str(r["player_id"]): float(r["delta"]) for r in rows}

        # Fetch rosters
        rosters = get_rosters(platform, league_id, season) or []
        roster_map = _build_roster_map(
            get_users(platform, league_id, season) or [],
            rosters
        )

        # Aggregate per roster (only QB/RB/WR/TE players, ignore picks/K/DEF)
        CORE_POS = {"QB", "RB", "WR", "TE"}
        players_index = load_players_index() or {}

        roster_totals: dict = {}
        for r in rosters:
            rid = str(r.get("roster_id", ""))
            players_on_roster = r.get("players") or []
            total_delta = 0.0
            player_deltas = []
            for pid in players_on_roster:
                meta = players_index.get(str(pid), {})
                pos = str(meta.get("pos", "")).upper()
                if pos not in CORE_POS:
                    continue
                d = delta_by_pid.get(str(pid), 0.0)
                total_delta += d
                if abs(d) >= 5:
                    player_deltas.append({
                        "player_id": str(pid),
                        "name": meta.get("name", str(pid)),
                        "position": pos,
                        "delta": d,
                    })
            player_deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)
            roster_totals[rid] = {
                "roster_id": rid,
                "team_name": roster_map.get(rid, f"Roster {rid}"),
                "total_delta": round(total_delta, 1),
                "top_movers": player_deltas[:5],
            }

        # Compute league average delta
        all_deltas = [v["total_delta"] for v in roster_totals.values()]
        avg_delta = round(sum(all_deltas) / len(all_deltas), 1) if all_deltas else 0.0

        # Add vs_avg
        results = []
        for rid, entry in roster_totals.items():
            results.append({
                **entry,
                "vs_avg": round(entry["total_delta"] - avg_delta, 1),
            })

        results.sort(key=lambda x: x["vs_avg"], reverse=True)

        def _iso(d) -> str:
            return d.isoformat() if hasattr(d, "isoformat") else str(d)

        return jsonify({
            "latest_date": _iso(latest_date_obj),
            "baseline_date": _iso(baseline_date),
            "league_avg_delta": avg_delta,
            "rosters": results,
        })

    except Exception:
        logger.exception("[beat-the-market] Unexpected error")
        return jsonify({"error": "Internal error"}), 500


@app.route("/api/schedule-strength")
def api_schedule_strength():
    """
    Compute schedule strength remaining for each team in a league.

    For each team's future matchups (weeks > current_week), look up their
    opponent's average points scored this season. The team with the hardest
    remaining schedule faces the highest-scoring opponents on average.

    Query params: platform, league_id, season
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"error": "league_id required"}), 400

    nfl_state = get_nfl_state() or {}
    try:
        season = int(request.args.get("season") or nfl_state.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year

    current_week = int(nfl_state.get("leg") or nfl_state.get("week") or 0)
    season_type = str(nfl_state.get("season_type") or "").lower()
    FULL_SEASON_WEEKS = 14  # typical fantasy regular season

    try:
        from dashboard_services.platform_api import get_matchups as pf_get_matchups

        rosters = get_rosters(platform, league_id, season) or []
        users = get_users(platform, league_id, season) or []
        roster_map = _build_roster_map(users, rosters)

        # Build per-roster average points from completed weeks
        avg_pts_by_rid: dict[str, float] = {}
        weekly_pts: dict[str, list] = {str(r.get("roster_id")): [] for r in rosters}

        for w in range(1, current_week + 1):
            try:
                week_data = pf_get_matchups(platform, league_id, w, season) or []
            except Exception:
                continue
            for m in week_data:
                rid = str(m.get("roster_id", ""))
                pts = float(m.get("points") or 0.0)
                if rid in weekly_pts:
                    weekly_pts[rid].append(pts)

        for rid, pts_list in weekly_pts.items():
            avg_pts_by_rid[rid] = round(sum(pts_list) / len(pts_list), 2) if pts_list else 0.0

        # When no games have been played, fall back to power rankings (roster value) as proxy
        games_played = sum(1 for pts in avg_pts_by_rid.values() if pts > 0)
        if games_played == 0:
            try:
                ctx = get_league_ctx_from_cache(platform, league_id, season)
                model_vals = ctx.get("model_value_table") or []
                picks_by_roster = ctx.get("picks_by_roster") or {}
                values_by_id = {str(p["id"]): float(p.get("value") or 0) for p in model_vals if p.get("id")}
                pick_values = load_pick_value_table() or {}
                standings_map = ctx.get("standings_map") or {}
                for r in rosters:
                    rid = str(r.get("roster_id", ""))
                    player_ids = [str(pid) for pid in (r.get("players") or [])]
                    roster_val = sum(values_by_id.get(pid, 0.0) for pid in player_ids)
                    # Normalize to a "projected points" scale (~100-160 range) for display consistency
                    avg_pts_by_rid[rid] = round(100.0 + roster_val / 50.0, 2)
            except Exception:
                pass

        # Build future matchups map: rid -> list of opponent roster_ids
        future_opponents: dict[str, list] = {str(r.get("roster_id")): [] for r in rosters}

        for w in range(current_week + 1, FULL_SEASON_WEEKS + 1):
            try:
                week_data = pf_get_matchups(platform, league_id, w, season) or []
            except Exception:
                continue
            if not week_data:
                break
            # Group by matchup_id
            by_mid: dict = {}
            for m in week_data:
                mid = m.get("matchup_id")
                if mid is None:
                    continue
                by_mid.setdefault(mid, []).append(str(m.get("roster_id", "")))
            for mid, rids in by_mid.items():
                if len(rids) == 2:
                    future_opponents[rids[0]].append(rids[1])
                    future_opponents[rids[1]].append(rids[0])

        results = []
        for r in rosters:
            rid = str(r.get("roster_id", ""))
            opp_rids = future_opponents.get(rid, [])
            opp_avgs = [avg_pts_by_rid.get(o, 0.0) for o in opp_rids]
            avg_opp = round(sum(opp_avgs) / len(opp_avgs), 2) if opp_avgs else 0.0
            results.append({
                "roster_id": rid,
                "team_name": roster_map.get(rid, f"Roster {rid}"),
                "games_remaining": len(opp_rids),
                "avg_opp_points": avg_opp,
                "my_avg_points": avg_pts_by_rid.get(rid, 0.0),
            })

        results.sort(key=lambda x: x["avg_opp_points"], reverse=True)

        return jsonify({
            "current_week": current_week,
            "weeks_remaining": max(0, FULL_SEASON_WEEKS - current_week),
            "teams": results,
            "using_power_rankings": games_played == 0,
        })

    except Exception:
        logger.exception("[schedule-strength] Unexpected error")
        return jsonify({"error": "Internal error"}), 500


def _fetch_fc_rookie_adp(is_sf: bool, season: int) -> dict:
    """
    Fetch dynasty rookie ADP from FantasyCalc and return a map of
    sleeper_id -> {adp_rank, pos_rank, position}.
    Caches per league type per day.
    """
    import json as _json
    from utils.paths import DATA_DIR
    key = f"fc_rookie_adp_{'sf' if is_sf else '1qb'}_{date.today().isoformat()}.json"
    cache_path = DATA_DIR / key
    if cache_path.exists():
        try:
            with open(cache_path) as _f:
                return _json.load(_f)
        except Exception:
            pass

    num_qbs = 2 if is_sf else 1
    url = f"https://fantasycalc.com/api/values/current?numQbs={num_qbs}&type=1&ppr=0.5"
    try:
        import requests as _req
        resp = _req.get(url, timeout=10, headers={"User-Agent": "fantasy-dashboard/1.0"})
        resp.raise_for_status()
        fc_data = resp.json()
    except Exception:
        fc_data = []

    fc_by_sleeper: dict = {}
    for entry in (fc_data or []):
        p = entry.get("player") or {}
        sid = str(p.get("sleeperId") or "")
        if sid and sid != "None":
            fc_by_sleeper[sid] = {
                "overall_rank": entry.get("overallRank"),
                "pos_rank":     entry.get("positionalRank"),
                "position":     str(p.get("position") or "").upper(),
                "name":         p.get("name") or "",
            }

    result: dict = {}
    try:
        from dashboard_services.db import get_conn
        with get_conn() as _conn:
            all_rows = _conn.execute(
                "SELECT sleeper_id, name, position FROM rookie_prospects "
                "WHERE draft_class_year = %s",
                (season,)
            ).fetchall()

        our_sids = {str(r["sleeper_id"]) for r in all_rows if r["sleeper_id"]}
        sid_matched = sorted(
            [(sid, fc_by_sleeper[sid]) for sid in our_sids if sid in fc_by_sleeper],
            key=lambda x: (x[1]["overall_rank"] or 9999)
        )
        for rookie_rank, (sid, info) in enumerate(sid_matched, start=1):
            result[sid] = {
                "adp_rank": rookie_rank,
                "fc_overall": info["overall_rank"],
                "pos_rank":   info["pos_rank"],
                "position":   info["position"],
            }

        our_names = {str(r["name"]).lower() for r in all_rows}
        name_matched = sorted(
            [entry for entry in (fc_data or [])
             if (entry.get("player") or {}).get("name", "").lower() in our_names],
            key=lambda e: (e.get("overallRank") or 9999)
        )
        for entry in name_matched:
            p = entry.get("player") or {}
            sid = str(p.get("sleeperId") or "")
            if not sid or sid in result:
                continue
            result[sid] = {
                "adp_rank": len(result) + 1,
                "fc_overall": entry.get("overallRank"),
                "pos_rank":   entry.get("positionalRank"),
                "position":   str(p.get("position") or "").upper(),
            }

        all_entries = sorted(result.items(), key=lambda kv: kv[1].get("fc_overall") or 9999)
        result = {sid: {**info, "adp_rank": rank}
                  for rank, (sid, info) in enumerate(all_entries, start=1)}

    except Exception:
        pass

    try:
        with open(cache_path, "w") as _f:
            _json.dump(result, _f)
    except Exception:
        pass
    return result


def _fetch_league_adp_from_db(
    is_sf: bool,
    season: int,
    draft_type: str,
    num_teams: int,
    min_samples: int = 20,
) -> dict:
    """
    Pull ADP from real league draft data stored by the draft ADP crawler.

    Tries an exact num_teams match first, then widens to ±2 if not enough
    samples exist.  Returns player_id -> {adp_rank, avg_pick, std_pick,
    sample_size, position} or empty dict when data is sparse.
    """
    try:
        from dashboard_services.db import get_conn
        from utils.paths import DATA_DIR
        import json as _json

        # Day-level cache so we don't hit the DB on every page load
        cache_key = f"league_adp_{draft_type}_{'sf' if is_sf else '1qb'}_{num_teams}t_{season}_{date.today().isoformat()}.json"
        cache_path = DATA_DIR / cache_key
        if cache_path.exists():
            try:
                return _json.load(open(cache_path))
            except Exception:
                pass

        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT da.player_id, da.avg_pick, da.std_pick, da.avg_round, da.sample_size
                FROM draft_adp da
                WHERE da.draft_type  = %s
                  AND da.season      = %s
                  AND da.is_superflex = %s
                  AND da.num_teams BETWEEN %s AND %s
                  AND da.sample_size >= %s
                ORDER BY da.avg_pick ASC
                """,
                (draft_type, season, is_sf, num_teams - 2, num_teams + 2, min_samples),
            ).fetchall()

        if not rows:
            return {}

        # Load position info from player_values for enrichment
        player_ids = [r["player_id"] for r in rows]
        pos_map: dict[str, str] = {}
        try:
            with get_conn() as conn:
                pv_rows = conn.execute(
                    "SELECT player_id, position FROM player_values WHERE player_id = ANY(%s)",
                    (player_ids,),
                ).fetchall()
                pos_map = {r["player_id"]: (r["position"] or "").upper() for r in pv_rows}
        except Exception:
            pass

        result: dict = {}
        pos_counters: dict[str, int] = {}
        for rank, row in enumerate(rows, start=1):
            pid = str(row["player_id"])
            pos = pos_map.get(pid, "")
            pos_counters[pos] = pos_counters.get(pos, 0) + 1
            result[pid] = {
                "adp_rank":   rank,
                "avg_pick":   float(row["avg_pick"] or rank),
                "std_pick":   float(row["std_pick"] or 0),
                "pos_rank":   pos_counters[pos],
                "position":   pos,
                "sample_size": int(row["sample_size"] or 0),
            }

        try:
            _json.dump(result, open(cache_path, "w"))
        except Exception:
            pass
        return result
    except Exception:
        return {}


def _build_model_adp_fallback(is_sf: bool, season: int, filter_undrafted: bool = False) -> dict:
    """
    Build a value-based board from our own model when external ADP is unavailable.
    Ranks this season's rookies by calibrated (or raw) model value so picks can
    still be graded relative to who was available at that spot.
    Returns sleeper_id -> {adp_rank, pos_rank, position}.
    """
    try:
        from dashboard_services.db import get_conn
        value_col = "COALESCE(calibrated_value_sf, value_sf)" if is_sf \
                    else "COALESCE(calibrated_value_1qb, value_1qb)"
        undrafted_clause = "AND rp.draft_confirmed = TRUE" if filter_undrafted else ""
        with get_conn() as _conn:
            rows = _conn.execute(
                f"""
                SELECT rp.sleeper_id, rp.position,
                       {value_col} AS val
                FROM rookie_prospects rp
                LEFT JOIN player_values pv ON pv.player_id = rp.sleeper_id
                WHERE rp.draft_class_year = %s
                  AND rp.sleeper_id IS NOT NULL
                  {undrafted_clause}
                ORDER BY {value_col} DESC NULLS LAST
                """,
                (season,)
            ).fetchall()

        result: dict = {}
        pos_counters: dict = {}
        for rank, row in enumerate(rows, start=1):
            sid = str(row["sleeper_id"])
            pos = str(row["position"] or "").upper()
            pos_counters[pos] = pos_counters.get(pos, 0) + 1
            result[sid] = {
                "adp_rank": rank,
                "pos_rank":  pos_counters[pos],
                "position":  pos,
            }
        return result
    except Exception:
        return {}


@app.route("/api/draft-grades")
def api_draft_grades():
    """
    Grade each team's rookie draft class using three signals:
      1. ADP value   — actual pick slot vs FantasyCalc rookie ADP (external)
      2. BPA / board — who was still available with better ADP at that pick
      3. Team need   — did the pick fill a positional need on the roster?

    Grading is rookie-draft-specific: only compares picks against other rookies
    in that draft, not against the full dynasty player pool.

    Query params: platform, league_id, season, league_type
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"error": "league_id required"}), 400

    nfl_state = get_nfl_state() or {}
    try:
        season = int(request.args.get("season") or nfl_state.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year

    league_type = str(request.args.get("league_type", "1qb")).strip().lower()
    is_sf = (league_type == "sf")

    try:
        from dashboard_services.api import fetch_json
        from collections import defaultdict as _defaultdict

        # ── Draft picks ─────────────────────────────────────────────────────
        drafts = get_drafts(platform, league_id, season) or []
        latest_draft = get_most_recent_valid_draft_for_season(drafts, season)
        if not latest_draft:
            return jsonify({"error": "No completed draft found for this season"}), 404

        draft_id = latest_draft.get("draft_id") or latest_draft.get("id")
        if not draft_id:
            return jsonify({"error": "Draft has no ID"}), 404

        picks_raw = fetch_json(f"/draft/{draft_id}/picks") or []
        if not isinstance(picks_raw, list) or not picks_raw:
            return jsonify({"error": "Draft has no picks yet"}), 404

        players_index = load_players_index() or {}

        # ── Draft type: startup (≥10 rounds) or rookie (1-5 rounds) ─────────
        _draft_rounds = int((latest_draft.get("settings") or {}).get("rounds") or 0)
        if _draft_rounds >= 10:
            _draft_type = "startup"
        elif 1 <= _draft_rounds <= 5:
            _draft_type = "rookie"
        else:
            _draft_type = "rookie"  # safe default

        # ── Rosters & users (needed for num_teams before ADP lookup) ─────────
        rosters = get_rosters(platform, league_id, season) or []
        _num_teams = len(rosters) or 12

        # ── NFL draft completion check ───────────────────────────────────────
        from data_building.rookie_pipeline.pipeline import is_draft_complete
        try:
            from dashboard_services.db import get_conn as _get_conn
            with _get_conn() as _dc:
                _nfl_draft_done = is_draft_complete(season, _dc)
        except Exception:
            _nfl_draft_done = is_draft_complete(season)

        # ── ADP: league data → FantasyCalc → model fallback ─────────────────
        # adp_info[sleeper_id] = {adp_rank, pos_rank, position, ...}
        adp_info = _fetch_league_adp_from_db(is_sf, season, _draft_type, _num_teams)
        if adp_info:
            adp_source = "league"
        else:
            adp_info = _fetch_fc_rookie_adp(is_sf, season)
            if adp_info:
                adp_source = "fantasycalc"
            else:
                adp_info = _build_model_adp_fallback(is_sf, season, filter_undrafted=_nfl_draft_done)
                adp_source = "model" if adp_info else "none"
        users   = get_users(platform, league_id, season) or []
        roster_map = _build_roster_map(users, rosters)

        # Pre-draft roster: each team's existing players by position
        # (excluding picks made in this draft — we'll account for those)
        roster_pos_counts: dict[str, dict[str, int]] = {}   # rid -> {pos: count}
        CORE_POS = {"QB", "RB", "WR", "TE"}
        # IDs that appear in any team's pre-draft roster (non-rookie veterans)
        drafted_player_ids = {str(p.get("player_id") or "") for p in picks_raw}
        for r in rosters:
            rid = str(r.get("roster_id", ""))
            players_on_roster = r.get("players") or []
            counts: dict[str, int] = {pos: 0 for pos in CORE_POS}
            for pid in players_on_roster:
                pid = str(pid)
                if pid in drafted_player_ids:
                    continue  # skip rookies from this very draft
                pos = str((players_index.get(pid) or {}).get("pos", "")).upper()
                if pos in CORE_POS:
                    counts[pos] = counts.get(pos, 0) + 1
            roster_pos_counts[rid] = counts

        # Need thresholds: below these counts the position is a clear need
        NEED_THRESHOLD = {"QB": 2, "RB": 4, "WR": 5, "TE": 2}

        def position_needed(rid: str, pos: str) -> bool:
            if pos not in CORE_POS:
                return False
            current = roster_pos_counts.get(rid, {}).get(pos, 0)
            return current < NEED_THRESHOLD.get(pos, 3)

        # ── Board simulation ─────────────────────────────────────────────────
        # Sort picks chronologically to simulate the board state at each selection
        picks_sorted = sorted(
            [p for p in picks_raw if isinstance(p, dict)],
            key=lambda p: int(p.get("pick_no") or 0)
        )

        # For rookie drafts, restrict the board to players who are actually
        # eligible — i.e., were actually picked in this draft or are confirmed
        # rookies for this season.  This prevents veterans (e.g. Isaiah Likely)
        # from appearing in adp_info (sourced from startup drafts) from polluting
        # the rookie draft board.
        eligible_sids: set[str] = set(drafted_player_ids)
        try:
            from dashboard_services.db import get_conn as _gcb
            with _gcb() as _cc:
                _rp = _cc.execute(
                    "SELECT sleeper_id FROM rookie_prospects "
                    "WHERE draft_class_year = %s AND sleeper_id IS NOT NULL",
                    (season,),
                ).fetchall()
            for _r in _rp:
                eligible_sids.add(str(_r["sleeper_id"]))
        except Exception:
            pass  # fall back to full board if DB unavailable

        # All eligible players with ADP data, sorted best → worst
        board_all: list[str] = sorted(
            [sid for sid in adp_info.keys()
             if adp_info[sid].get("avg_pick") is not None
             and (not eligible_sids or sid in eligible_sids)],
            key=lambda sid: adp_info[sid]["avg_pick"]
        )
        taken: set[str] = set()

        # ── Calculate positional rankings based on avg_pick ───────────────────────
        pos_rankings: dict[str, dict[str, int]] = {}
        for pos in CORE_POS:
            # Get all players with this position and valid avg_pick
            pos_players = [
                (sid, info["avg_pick"]) 
                for sid, info in adp_info.items() 
                if info.get("position") == pos and info.get("avg_pick") is not None
            ]
            # Sort by avg_pick and assign rankings
            pos_players.sort(key=lambda x: x[1])
            pos_rankings[pos] = {sid: rank + 1 for rank, (sid, _) in enumerate(pos_players)}

        def available_at_pick(exclude_sid: str) -> list[dict]:
            """Return top-3 remaining board players (by avg_pick) excluding the one just picked."""
            available_players = []
            for sid in board_all:
                if sid in taken or sid == exclude_sid:
                    continue
                info = adp_info[sid]
                # Position from adp_info first, fall back to players_index
                pos = info.get("position") or str((players_index.get(sid) or {}).get("pos", "")).upper()
                available_players.append({
                    "player_id": sid,
                    "name": (players_index.get(sid) or {}).get("name") or sid,
                    "position": pos,
                    "avg_pick": info.get("avg_pick"),
                    "pos_rank": pos_rankings.get(pos, {}).get(sid) if pos else None
                })
                if len(available_players) >= 3:
                    break
            return available_players

        def pick_grade(adp_diff: Optional[float], need: bool, bpa_gap: Optional[int], 
                    is_bpa: bool, pos: str, is_sf: bool, qb_count: int, name: str, num_teams: int) -> str:
            """
            Improved grading system that rewards BPA and accounts for league context.
            
            adp_diff  : actual_pick - avg_pick  (+= value, -= reach)
            need      : True if pick fills a positional need
            bpa_gap   : ADP gap between this pick and the best available player
                        (0 = BPA taken; positive = better players left on board)
            is_bpa    : True if this was the best player available at the pick
            pos       : Position of the player picked
            is_sf     : True if Superflex league, False if 1QB
            qb_count  : Current QB count on the roster
            name      : Player name for debugging
            """
            if adp_diff is None:
                return "N/A"

            # F should only trigger for a reach of more than ~1 full round.
            # In a 10-team draft 1 round = 10 picks, so -11 is the F threshold.
            # This prevents picks like -8 or -6 from grading F in small leagues.
            big_reach = -(num_teams * 1.1)

            if adp_diff >= 5:           score = 4   # clear value
            elif adp_diff >= 2:         score = 3   # good value
            elif adp_diff >= -1:        score = 2   # on ADP
            elif adp_diff >= big_reach: score = 1   # reach within 1 round → D
            else:                       score = 0   # > 1 round early → F

            # BPA bonus / penalty.
            # Only penalise when the pick was close to ADP (adp_diff >= -2):
            # for bigger reaches the adp_diff already captures the cost, so
            # applying BPA on top would double-count and turn a D into an F.
            if is_bpa:
                score += 2
            elif bpa_gap is not None and bpa_gap >= 5:
                score = max(score - 1, 0)   # better player available (was -2)
            # Moderate BPA gap (3-4) no longer penalises — adp_diff already
            # captures whether the pick was a reach

            # Need modifier with positional context
            if need:
                score += 1
            else:
                # Penalise redundant QB in 1QB
                if pos == "QB" and not is_sf and qb_count >= 2:
                    score = max(score - 2, 0)
                elif pos == "QB" and not is_sf and qb_count >= 1:
                    score = max(score - 1, 0)

            # ── Post-modifier floors ─────────────────────────────────────────
            if adp_diff >= -3:
                score = max(score, 1)   # tiny reach → at least D
            if need and adp_diff >= -4:
                score = max(score, 2)   # need pick within 4 → at least C

            return {5: "A+", 4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}.get(min(score, 5), "F")

        def team_grade(pick_scores: list[str]) -> str:
            if not pick_scores:
                return "N/A"
            grade_val = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1, "F": 0, "N/A": 2}
            avg = sum(grade_val.get(g, 2) for g in pick_scores) / len(pick_scores)
            if avg >= 4.5: return "A+"
            if avg >= 3.5: return "A"
            if avg >= 2.5: return "B"
            if avg >= 1.5: return "C"
            if avg >= 0.5: return "D"
            return "F"

        # ── Process picks in draft order ─────────────────────────────────────
        picks_by_roster: dict = _defaultdict(list)
        for p in picks_sorted:
            rid       = str(p.get("roster_id") or p.get("picked_by") or "")
            player_id = str(p.get("player_id") or "")
            pick_no   = int(p.get("pick_no") or 0)
            if not rid or not player_id or not pick_no:
                continue

            player_meta = players_index.get(player_id) or {}
            pos         = str(player_meta.get("pos") or "").upper()
            name        = player_meta.get("name") or f"Pick #{pick_no}"
            need        = position_needed(rid, pos)

            # ADP comparison
            info     = adp_info.get(player_id)
            avg_pick = info.get("avg_pick") if info else None
            adp_diff = (pick_no - avg_pick) if avg_pick is not None else None

            # BPA: who with a better ADP was still available?
            avail_better = [
                a for a in available_at_pick(player_id)
                if a.get("avg_pick", pick_no) < (avg_pick or pick_no)
            ]
            bpa_gap = (avg_pick - avail_better[0].get("avg_pick", pick_no)) if avail_better and avg_pick is not None else 0
            
            # Check if this was the best player available (BPA)
            is_bpa = len(avail_better) == 0

            # Could they have waited?
            # Estimate: next same-team pick ≈ pick_no + num_teams (snake round)
            num_teams = max(len(rosters), 1)
            could_wait = (adp_diff is not None and adp_diff < -2 and
                          avg_pick is not None and avg_pick > pick_no + num_teams)

            # Get current QB count for positional context
            qb_count = roster_pos_counts.get(rid, {}).get("QB", 0)

            grade = pick_grade(adp_diff, need, bpa_gap, is_bpa, pos, is_sf, qb_count, name, _num_teams)

            picks_by_roster[rid].append({
                "pick_no":          pick_no,
                "round":            (pick_no - 1) // max(num_teams, 1) + 1,
                "player_id":        player_id,
                "name":             name,
                "position":         pos,
                "team":             player_meta.get("team") or "",
                "avg_pick":         avg_pick,
                "adp_diff":         adp_diff,
                "pos_rank":         None,  # Will be calculated based on avg_pick
                "need":             need,
                "bpa":              avail_better[:2],   # top 2 available better options
                "could_wait":       could_wait,
                "grade":            grade,
            })

            # Mark this player as taken on the board
            taken.add(player_id)
            # Update this team's running positional count (so later picks reflect draft)
            if pos in CORE_POS:
                roster_pos_counts.setdefault(rid, {pos: 0 for pos in CORE_POS})
                roster_pos_counts[rid][pos] = roster_pos_counts[rid].get(pos, 0) + 1

        # ── Assemble results ─────────────────────────────────────────────────
        results = []
        for r in rosters:
            rid = str(r.get("roster_id", ""))
            team_picks = sorted(picks_by_roster.get(rid, []), key=lambda x: x["pick_no"])
            if not team_picks:
                continue
            
            # Update pos_rank for each pick based on avg_pick
            for pick in team_picks:
                if pick["position"] in pos_rankings and pick["player_id"] in pos_rankings[pick["position"]]:
                    pick["pos_rank"] = pos_rankings[pick["position"]][pick["player_id"]]
            
            tgrade = team_grade([p["grade"] for p in team_picks if p["grade"] != "N/A"])
            results.append({
                "roster_id":  rid,
                "team_name":  roster_map.get(rid, f"Roster {rid}"),
                "grade":      tgrade,
                "picks":      team_picks,
            })

        grade_order = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1, "F": 0, "N/A": 2}
        results.sort(key=lambda x: grade_order.get(x["grade"], 2), reverse=True)

        all_picks_flat = [p for t in results for p in t["picks"]]
        total_rounds = max((p["round"] for p in all_picks_flat), default=_draft_rounds or 1)

        return jsonify({
            "draft_id":     str(draft_id),
            "season":       season,
            "league_type":  league_type,
            "draft_type":   _draft_type,
            "adp_source":   adp_source,
            "num_teams":    _num_teams,
            "total_rounds": total_rounds,
            "teams":        results,
        })

    except Exception:
        logger.exception("[draft-grades] Unexpected error")
        return jsonify({"error": "Internal error"}), 500


# ---------------------------------------------------------------------------
# Trade Intelligence Engine API
# ---------------------------------------------------------------------------

@app.route("/api/trade-intel/trending")
def api_trade_intel_trending():
    """
    Most traded players in the last 7 days across all crawled leagues.
    Returns paginated players with trade counts and market vs model value delta.
    """
    from dashboard_services.db import get_conn
    try:
        season = int(request.args.get("season") or datetime.now().year)
        page = max(int(request.args.get("page") or 1), 1)
        per_page = 20  # Fixed at 20 per page
        offset = (page - 1) * per_page
        league_type = str(request.args.get("league_type") or "1qb").strip().lower()
        league_size = int(request.args.get("league_size") or 10)

        fmt = "sf" if league_type == "sf" else "1qb"
        model_col = f"value_{fmt}"

        # Map league_size to the size-bucketed column, fall back to all-leagues
        _SZ_MAP = {8: "8", 9: "8", 10: "10", 11: "10", 12: "12", 13: "14", 14: "14"}
        sz_suffix = _SZ_MAP.get(league_size, "")
        if sz_suffix:
            value_col_expr = f"COALESCE(s.market_value_{fmt}_{sz_suffix}, s.market_value_{fmt})"
        else:
            value_col_expr = f"s.market_value_{fmt}"

        # First get total count for pagination
        count_q = f"""
            SELECT COUNT(*) as total
            FROM trade_intel_player_stats s
            WHERE s.season = %s AND s.trade_count > 0
            """
        
        _q = f"""
            SELECT s.player_id, s.trade_count_7d, s.trade_count_30d, s.trade_count,
                   {value_col_expr} AS market_value, s.buy_sell_ratio,
                   s.market_trend_1qb,
                   pv.{model_col} AS model_value, pv.position, pv.team
            FROM trade_intel_player_stats s
            LEFT JOIN player_values pv ON pv.player_id = s.player_id
            WHERE s.season = %s AND s.trade_count > 0
            ORDER BY COALESCE(s.trade_count_7d, 0) DESC, s.trade_count DESC
            LIMIT %s OFFSET %s
            """
        with get_conn() as conn:
            # Get total count
            count_result = conn.execute(count_q, (season,)).fetchone()
            total_players = count_result["total"] if count_result else 0
            
            # Get paginated results
            rows = conn.execute(_q, (season, per_page, offset)).fetchall()
            # Fall back to most recent season that has data
            if not rows:
                fallback_season = conn.execute(
                    "SELECT season FROM trade_intel_player_stats WHERE trade_count > 0 ORDER BY season DESC LIMIT 1"
                ).fetchone()
                if fallback_season:
                    # Recalculate count for fallback season
                    count_result = conn.execute(count_q, (fallback_season["season"],)).fetchone()
                    total_players = count_result["total"] if count_result else 0
                    rows = conn.execute(_q, (fallback_season["season"], per_page, offset)).fetchall()

        from utils.utils import load_players_index
        players_map = load_players_index() or {}

        result = []
        for r in rows:
            pid = r["player_id"]
            info = players_map.get(pid, {})
            model_val = float(r["model_value"] or 0)
            market_val = float(r["market_value"] or 0)
            delta = round(market_val - model_val, 1) if model_val and market_val else None
            result.append({
                "player_id": pid,
                "name": info.get("name", pid),
                "position": r["position"] or info.get("pos"),
                "team": r["team"] or info.get("team"),
                "trade_count_7d": r["trade_count_7d"],
                "trade_count_30d": r["trade_count_30d"],
                "trade_count_all": r["trade_count"],
                "market_value": market_val or None,
                "model_value": model_val or None,
                "value_delta": delta,
                "market_trend": float(r["market_trend_1qb"]) if r["market_trend_1qb"] is not None else None,
            })

        # Calculate pagination info
        total_pages = (total_players + per_page - 1) // per_page
        
        return jsonify({
            "season": season, 
            "players": result,
            "pagination": {
                "current_page": page,
                "per_page": per_page,
                "total_players": total_players,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        })

    except Exception:
        logger.exception("[trade-intel/trending] error")
        return jsonify({"error": "Internal error"}), 500


@app.route("/api/trade-intel/player/<player_id>")
def api_trade_intel_player(player_id: str):
    """
    Trade market data for a specific player:
    - Real trade frequency
    - Market value implied by actual trades
    - Model value vs market delta
    - Common trade companions
    """
    from dashboard_services.db import get_conn
    try:
        season = int(request.args.get("season") or datetime.now().year)
        league_type = str(request.args.get("league_type") or "1qb").strip().lower()
        league_size = int(request.args.get("league_size") or 10)

        fmt     = "sf" if league_type == "sf" else "1qb"
        raw_col = f"value_{fmt}"
        cal_col = f"calibrated_value_{fmt}"

        _SZ_MAP = {8: "8", 9: "8", 10: "10", 11: "10", 12: "12", 13: "14", 14: "14"}
        sz_suffix = _SZ_MAP.get(league_size, "")
        if sz_suffix:
            value_col_expr = f"COALESCE(s.market_value_{fmt}_{sz_suffix}, s.market_value_{fmt})"
        else:
            value_col_expr = f"s.market_value_{fmt}"

        with get_conn() as conn:
            stat_row = conn.execute(
                f"""
                SELECT
                    s.*,
                    {value_col_expr}                            AS market_value,
                    pv.{raw_col}                                AS model_value,
                    COALESCE(pv.{cal_col}, pv.{raw_col})        AS calibrated_value,
                    pv.calibration_source,
                    pv.position, pv.team
                FROM trade_intel_player_stats s
                LEFT JOIN player_values pv ON pv.player_id = s.player_id
                WHERE s.player_id = %s AND s.season = %s
                """,
                (player_id, season)
            ).fetchone()

            package_rows = conn.execute(
                """
                SELECT package_key, occurrence_count
                FROM trade_intel_packages
                WHERE anchor_player_id = %s AND season = %s
                ORDER BY occurrence_count DESC
                LIMIT 10
                """,
                (player_id, season)
            ).fetchall()

        from utils.utils import load_players_index
        players_map = load_players_index() or {}

        def _resolve_package(pkg_key: str) -> list[dict]:
            companions = []
            for pid in pkg_key.split("|"):
                if not pid:
                    continue
                info = players_map.get(pid, {})
                companions.append({
                    "player_id": pid,
                    "name": info.get("name", pid),
                    "position": info.get("pos"),
                })
            return companions

        common_packages = [
            {
                "companions": _resolve_package(r["package_key"]),
                "occurrence_count": r["occurrence_count"],
            }
            for r in package_rows
            if r["package_key"]
        ]

        if not stat_row:
            return jsonify({
                "player_id": player_id,
                "season": season,
                "trade_count": 0,
                "common_packages": common_packages,
            })

        model_val = float(stat_row["model_value"] or 0)
        market_val = float(stat_row["market_value"] or 0)
        calibrated_val = float(stat_row["calibrated_value"] or 0)
        # Delta is market vs raw model — shows how much the model diverges from real trades
        delta = round(market_val - model_val, 1) if model_val and market_val else None

        return jsonify({
            "player_id": player_id,
            "season": season,
            "trade_count_7d": stat_row["trade_count_7d"],
            "trade_count_30d": stat_row["trade_count_30d"],
            "trade_count_all": stat_row["trade_count"],
            "market_value": market_val or None,
            "model_value": model_val or None,
            "calibrated_value": calibrated_val or None,
            "calibration_source": stat_row["calibration_source"],
            "value_delta": delta,
            "buy_sell_ratio": float(stat_row["buy_sell_ratio"]) if stat_row["buy_sell_ratio"] else None,
            "avg_package_value": float(stat_row["avg_package_value"]) if stat_row["avg_package_value"] else None,
            "common_packages": common_packages,
        })

    except Exception:
        logger.exception("[trade-intel/player] error")
        return jsonify({"error": "Internal error"}), 500


@app.route("/api/trade-database")
def api_trade_database():
    """
    Paginated, searchable real-trade log.
    ?q=<player name>  &page=<int>  &limit=<int>  &league_type=<all|1qb|sf>
    """
    try:
        q           = (request.args.get("q") or "").strip().lower()
        page        = max(0, int(request.args.get("page") or 0))
        limit       = min(int(request.args.get("limit") or 20), 50)
        league_type = (request.args.get("league_type") or "all").strip().lower()
        season      = int(request.args.get("season") or datetime.now().year)

        from dashboard_services.db import get_conn
        from utils.utils import load_players_index

        players_map = load_players_index() or {}

        # If searching by name, resolve to player_ids first
        match_ids: list[str] = []
        if q:
            match_ids = [
                pid for pid, info in players_map.items()
                if q in (info.get("name") or "").lower()
            ]
            if not match_ids:
                return jsonify({"trades": [], "total": 0, "has_more": False})

        sf_filter = ""
        if league_type == "sf":
            sf_filter = "AND l.is_superflex = TRUE"
        elif league_type == "1qb":
            sf_filter = "AND l.is_superflex = FALSE"

        with get_conn() as conn:
            if match_ids:
                count_row = conn.execute(
                    f"""
                    SELECT COUNT(DISTINCT t.id) AS n
                    FROM trade_intel_trades t
                    JOIN trade_intel_assets a ON a.trade_id = t.id
                    LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                    WHERE a.player_id = ANY(%s) AND a.asset_type = 'player'
                      AND t.season = %s {sf_filter}
                    """,
                    (match_ids, season),
                ).fetchone()
                total = int(count_row["n"]) if count_row else 0

                trade_rows = conn.execute(
                    f"""
                    SELECT DISTINCT
                        t.id, t.transaction_id, t.season, t.week, t.created_at,
                        l.scoring_type, l.is_superflex, l.num_teams
                    FROM trade_intel_trades t
                    JOIN trade_intel_assets a ON a.trade_id = t.id
                    LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                    WHERE a.player_id = ANY(%s) AND a.asset_type = 'player'
                      AND t.season = %s {sf_filter}
                    ORDER BY t.created_at DESC NULLS LAST
                    LIMIT %s OFFSET %s
                    """,
                    (match_ids, season, limit + 1, page * limit),
                ).fetchall()
            else:
                count_row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS n FROM trade_intel_trades t
                    LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                    WHERE t.season = %s {sf_filter}
                    """,
                    (season,),
                ).fetchone()
                total = int(count_row["n"]) if count_row else 0

                trade_rows = conn.execute(
                    f"""
                    SELECT t.id, t.transaction_id, t.season, t.week, t.created_at,
                           l.scoring_type, l.is_superflex, l.num_teams
                    FROM trade_intel_trades t
                    LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                    WHERE t.season = %s {sf_filter}
                    ORDER BY t.created_at DESC NULLS LAST
                    LIMIT %s OFFSET %s
                    """,
                    (season, limit + 1, page * limit),
                ).fetchall()

            has_more = len(trade_rows) > limit
            trade_rows = trade_rows[:limit]

            if not trade_rows:
                return jsonify({"trades": [], "total": total, "has_more": False})

            trade_ids = [r["id"] for r in trade_rows]
            asset_rows = conn.execute(
                """
                SELECT trade_id, side, asset_type, player_id,
                       pick_season, pick_round, pick_order, pick_slot, pick_roster_id
                FROM trade_intel_assets
                WHERE trade_id = ANY(%s)
                ORDER BY trade_id, side, id
                """,
                (trade_ids,),
            ).fetchall()

        assets_by_trade: dict = {}
        for a in asset_rows:
            tid = a["trade_id"]
            if tid not in assets_by_trade:
                assets_by_trade[tid] = {"a": [], "b": []}
            assets_by_trade[tid][a["side"]].append(a)

        def describe(a) -> dict:
            if a["asset_type"] == "player":
                pid  = a["player_id"]
                info = players_map.get(pid) or {}
                return {"type": "player", "player_id": pid,
                        "name": info.get("name") or pid,
                        "position": info.get("pos") or "?"}
            s    = str(a["pick_season"]) if a["pick_season"] else "?"
            r    = str(a["pick_round"])  if a["pick_round"]  else "?"
            slot = a["pick_slot"]
            if slot:
                name = f"{s} Pick {r}.{str(slot).zfill(2)}"
            else:
                order = a["pick_order"] or ""
                name  = f"{s} Round {r}" + (f" ({order})" if order else "")
            return {"type": "pick", "name": name}

        result = []
        for r in trade_rows:
            tid   = r["id"]
            sides = assets_by_trade.get(tid, {"a": [], "b": []})
            side_a_assets = [describe(a) for a in sides["a"]]
            side_b_assets = [describe(a) for a in sides["b"]]
            if not side_a_assets or not side_b_assets:
                continue
            trade_date = None
            if r["created_at"]:
                try:
                    trade_date = r["created_at"].strftime("%m/%d/%y")
                except Exception:
                    trade_date = str(r["created_at"])[:10]
            result.append({
                "trade_id":    r["transaction_id"],
                "date":        trade_date,
                "season":      r["season"],
                "scoring_type": r["scoring_type"],
                "is_superflex": r["is_superflex"],
                "num_teams":   r["num_teams"],
                "side_a":      side_a_assets,
                "side_b":      side_b_assets,
            })

        # Calculate pagination info
        current_page = page + 1  # Convert 0-based to 1-based
        per_page = limit
        total_pages = (total + per_page - 1) // per_page
        
        return jsonify({
            "trades": result, 
            "total": total, 
            "has_more": has_more,
            "pagination": {
                "current_page": current_page,
                "per_page": per_page,
                "total_players": total,  # Using total_players for consistency with trade intel
                "total_pages": total_pages,
                "has_next": has_more,
                "has_prev": current_page > 1
            }
        })

    except Exception:
        logger.exception("[trade-database] error")
        return jsonify({"error": "Internal error"}), 500


@app.route("/api/trade-intel/similar-trades")
def api_trade_intel_similar_trades():
    """
    Returns real trades where side-A players and side-B players appeared on
    OPPOSITE sides of the actual trade.  Falls back to any-side if only one
    side has players.
    """
    try:
        side_a_raw = request.args.get("side_a", "")
        side_b_raw = request.args.get("side_b", "")
        season     = int(request.args.get("season") or datetime.now().year)
        limit      = min(int(request.args.get("limit") or 10), 25)
        # Over-fetch so the empty-side filter doesn't starve the result set
        fetch_limit = limit * 6

        side_a_ids = [p.strip() for p in side_a_raw.split(",") if p.strip()]
        side_b_ids = [p.strip() for p in side_b_raw.split(",") if p.strip()]

        if not side_a_ids and not side_b_ids:
            return jsonify({"trades": []})

        from dashboard_services.db import get_conn
        from utils.utils import load_players_index

        with get_conn() as conn:
            if side_a_ids and side_b_ids:
                # Require players from each side to appear on OPPOSITE sides of the real trade
                trade_rows = conn.execute(
                    """
                    SELECT DISTINCT
                        t.id, t.transaction_id, t.season, t.week, t.created_at,
                        l.scoring_type, l.is_superflex, l.num_teams
                    FROM trade_intel_trades t
                    JOIN trade_intel_assets a1 ON a1.trade_id = t.id
                        AND a1.player_id = ANY(%s) AND a1.asset_type = 'player'
                    JOIN trade_intel_assets a2 ON a2.trade_id = t.id
                        AND a2.player_id = ANY(%s) AND a2.asset_type = 'player'
                        AND a2.side != a1.side
                    LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                    WHERE t.season = %s
                    ORDER BY t.created_at DESC NULLS LAST
                    LIMIT %s
                    """,
                    (side_a_ids, side_b_ids, season, fetch_limit),
                ).fetchall()
            else:
                # Only one side populated — match any trade with those players
                all_ids = side_a_ids or side_b_ids
                trade_rows = conn.execute(
                    """
                    SELECT DISTINCT
                        t.id, t.transaction_id, t.season, t.week, t.created_at,
                        l.scoring_type, l.is_superflex, l.num_teams
                    FROM trade_intel_trades t
                    JOIN trade_intel_assets a ON a.trade_id = t.id
                        AND a.player_id = ANY(%s) AND a.asset_type = 'player'
                    LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                    WHERE t.season = %s
                    ORDER BY t.created_at DESC NULLS LAST
                    LIMIT %s
                    """,
                    (all_ids, season, fetch_limit),
                ).fetchall()

            if not trade_rows:
                return jsonify({"trades": []})

            trade_ids = [r["id"] for r in trade_rows]
            asset_rows = conn.execute(
                """
                SELECT trade_id, side, asset_type, player_id,
                       pick_season, pick_round, pick_order, pick_slot, pick_roster_id
                FROM trade_intel_assets
                WHERE trade_id = ANY(%s)
                ORDER BY trade_id, side, id
                """,
                (trade_ids,),
            ).fetchall()

        assets_by_trade: dict = {}
        for a in asset_rows:
            tid = a["trade_id"]
            if tid not in assets_by_trade:
                assets_by_trade[tid] = {"a": [], "b": []}
            assets_by_trade[tid][a["side"]].append(a)

        players_map = load_players_index() or {}
        key_set     = set(side_a_ids + side_b_ids)

        def describe_asset(a) -> dict:
            if a["asset_type"] == "player":
                pid  = a["player_id"]
                info = players_map.get(pid) or {}
                return {
                    "type": "player", "player_id": pid,
                    "name": info.get("name") or pid,
                    "position": info.get("pos") or "?",
                    "is_key_player": pid in key_set,
                }
            s    = str(a["pick_season"]) if a["pick_season"] else "?"
            rd   = str(a["pick_round"])  if a["pick_round"]  else "?"
            slot = a["pick_slot"]
            if slot:
                name = f"{s} Pick {rd}.{str(slot).zfill(2)}"
            else:
                order = a["pick_order"] or ""
                name  = f"{s} Round {rd}" + (f" ({order})" if order else "")
            return {"type": "pick", "name": name, "is_key_player": False}

        side_a_ids_set = set(side_a_ids)

        result = []
        for r in trade_rows:
            tid   = r["id"]
            sides = assets_by_trade.get(tid, {"a": [], "b": []})
            side_a_raw = [describe_asset(a) for a in sides["a"]]
            side_b_raw = [describe_asset(a) for a in sides["b"]]
            # Skip trades missing one side (incomplete data)
            if not side_a_raw or not side_b_raw:
                continue
            # Orient so user's side_a players appear on the left column.
            # The cross-side SQL guarantees they're on opposite sides but the
            # DB's 'a'/'b' labeling might be inverted relative to the user's.
            if side_a_ids_set and any(
                a.get("player_id") in side_a_ids_set for a in side_b_raw
            ):
                side_a, side_b = side_b_raw, side_a_raw
            else:
                side_a, side_b = side_a_raw, side_b_raw
            trade_date = None
            if r["created_at"]:
                try:    trade_date = r["created_at"].strftime("%m/%d/%y")
                except: trade_date = str(r["created_at"])[:10]
            result.append({
                "trade_id":     r["transaction_id"],
                "date":         trade_date,
                "season":       r["season"],
                "scoring_type": r["scoring_type"],
                "is_superflex": r["is_superflex"],
                "num_teams":    r["num_teams"],
                "side_a":       side_a,
                "side_b":       side_b,
            })

        return jsonify({"trades": result[:limit]})

    except Exception:
        logger.exception("[trade-intel/similar-trades] error")
        return jsonify({"error": "Internal error"}), 500


@app.route("/api/trade-intel/run-crawl", methods=["POST"])
@limiter.limit("2 per hour")
def api_trade_intel_run_crawl():
    """
    Trigger a crawl batch manually (admin use). Runs discovery + crawl in background.
    In production this should be called by a cron job rather than the UI.
    """
    try:
        import threading
        from data_building.trade_intel.league_discovery import run_discovery
        from data_building.trade_intel.trade_crawler import run_crawl
        from data_building.trade_intel.analytics import run_analytics

        def _job():
            try:
                from data_building.trade_intel.trade_value_model import run_trade_value_model
                from data_building.build_daily_value_table import build_daily_model_values
                discovered = run_discovery(target=500)
                logger.info("[trade-intel] Discovered %d new leagues", discovered)
                crawl_result = run_crawl(batch_size=100)
                logger.info("[trade-intel] Crawl: %s", crawl_result)
                analytics_result = run_analytics()
                logger.info("[trade-intel] Analytics: %s", analytics_result)
                wls_result = run_trade_value_model()
                logger.info("[trade-intel] WLS: %s", wls_result)
                build_daily_model_values()
                logger.info("[trade-intel] Value table rebuilt with calibrated values")
            except Exception:
                logger.exception("[trade-intel] Background job failed")

        t = threading.Thread(target=_job, daemon=True)
        t.start()
        return jsonify({"status": "started"})

    except Exception:
        logger.exception("[trade-intel/run-crawl] error")
        return jsonify({"error": "Internal error"}), 500


@app.route("/api/roster-intel")
def api_roster_intel():
    """
    Keeper/cut signals for every rostered player in a league.
    Returns per-player hold/sell/buy/cut signals based on value trend, age, and position curve.
    """
    platform  = str(request.args.get("platform")  or "sleeper").strip()
    league_id = str(request.args.get("league_id") or "").strip()
    season    = int(request.args.get("season")    or datetime.now().year)
    league_type = str(request.args.get("league_type") or "1qb").strip().lower()

    if not league_id:
        return jsonify({"error": "league_id required"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform=platform, league_id=league_id, season=season)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    rosters    = ctx.get("rosters") or []
    roster_map = ctx.get("roster_map") or {}
    model_value_table = ctx.get("model_value_table") or []

    # Build value lookup keyed by player_id
    val_key = "sf_value" if league_type == "sf" else "value"
    values_by_id: dict = {}
    for row in model_value_table:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or "")
        if not pid:
            continue
        values_by_id[pid] = {
            "value":           float(row.get(val_key) or row.get("value") or 0),
            "age":             row.get("age"),
            "position":        str(row.get("position") or "").upper(),
            "pos_rank_label":  row.get("pos_rank_label") or "",
            "rank_change_7d":  row.get("rank_change_7d"),
            "name":            row.get("name") or "",
            "team":            row.get("team") or "",
        }

    # Bulk-fetch breakout scores
    all_rostered = [
        str(pid)
        for r in rosters
        for pid in (r.get("players") or [])
    ]
    breakout_scores: dict = {}
    from dashboard_services.db import get_conn
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (player_id)
                        player_id, breakout_opportunity_score
                    FROM breakout_opportunity_scores
                    WHERE player_id = ANY(%s)
                    ORDER BY player_id, as_of_date DESC
                    """,
                    (all_rostered,),
                )
                for r in cur.fetchall():
                    r = dict(r)
                    if r.get("breakout_opportunity_score") is not None:
                        breakout_scores[r["player_id"]] = float(r["breakout_opportunity_score"])
    except Exception:
        pass

    # Prime age ceilings by position
    prime_max = {"QB": 33, "RB": 26, "WR": 28, "TE": 29}

    def _signal(pid: str, info: dict) -> str:
        val        = info["value"]
        age        = float(info["age"] or 0)
        pos        = info["position"]
        rank_chg   = info["rank_change_7d"] or 0
        prime      = prime_max.get(pos, 28)
        bscore     = breakout_scores.get(pid, 0)
        past_prime = age > prime

        if val < 80:
            return "Cut"
        if val >= 400 and rank_chg >= 6:
            return "Sell High"
        if past_prime and val >= 300 and rank_chg >= 3:
            return "Sell High"
        if not past_prime and rank_chg <= -6 and val >= 200:
            return "Buy Window"
        if bscore >= 55 and not past_prime:
            return "Hold — Breakout"
        if val >= 500 and not past_prime:
            return "Core"
        if past_prime and val < 200:
            return "Cut"
        return "Hold"

    signal_order = {"Sell High": 0, "Core": 1, "Hold — Breakout": 2,
                    "Buy Window": 3, "Hold": 4, "Cut": 5}

    result = []
    players_index = ctx.get("players_index") or {}
    for roster in rosters:
        rid       = str(roster.get("roster_id"))
        team_name = roster_map.get(rid, f"Roster {rid}")
        players   = []
        for pid in (roster.get("players") or []):
            pid = str(pid)
            info = values_by_id.get(pid)
            if not info:
                continue
            pos = info["position"]
            if pos not in {"QB", "RB", "WR", "TE"}:
                continue
            sig = _signal(pid, info)
            players.append({
                "player_id":     pid,
                "name":          info["name"] or players_index.get(pid, {}).get("name", f"Player {pid}"),
                "position":      pos,
                "team":          info["team"],
                "age":           info["age"],
                "value":         round(info["value"], 0),
                "pos_rank_label": info["pos_rank_label"],
                "rank_change_7d": info["rank_change_7d"],
                "signal":        sig,
            })
        players.sort(key=lambda p: signal_order.get(p["signal"], 9))
        result.append({
            "roster_id": rid,
            "team_name": team_name,
            "players":   players,
        })

    result.sort(key=lambda t: t["team_name"])
    return jsonify({"teams": result})


@app.route("/api/trade-targets")
def api_trade_targets():
    """
    Suggest trade acquisition targets for the viewer's team based on positional needs.
    Compares viewer's positional value vs league average, surfaces best available from other teams.
    """
    platform        = str(request.args.get("platform")        or "sleeper").strip()
    league_id       = str(request.args.get("league_id")       or "").strip()
    season          = int(request.args.get("season")          or datetime.now().year)
    viewer_roster_id = str(request.args.get("viewer_roster_id") or "").strip()
    league_type     = str(request.args.get("league_type")     or "1qb").strip().lower()
    league_size     = int(request.args.get("league_size")     or 10)

    if not league_id or not viewer_roster_id:
        return jsonify({"error": "league_id and viewer_roster_id required"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform=platform, league_id=league_id, season=season)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    rosters           = ctx.get("rosters") or []
    roster_map        = ctx.get("roster_map") or {}
    model_value_table = ctx.get("model_value_table") or []
    players_index     = ctx.get("players_index") or {}
    picks_by_roster   = ctx.get("picks_by_roster") or {}

    if league_type == "sf":
        val_key = "sf_value" if league_size == 10 else f"sf_value_{league_size}"
        val_fallback = "sf_value"
    else:
        val_key = "value" if league_size == 10 else f"value_{league_size}"
        val_fallback = "value"
    values_by_id: dict = {}
    for row in model_value_table:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or "")
        if not pid:
            continue
        values_by_id[pid] = {
            "value":          float(row.get(val_key) or row.get(val_fallback) or row.get("value") or 0),
            "position":       str(row.get("position") or "").upper(),
            "pos_rank_label": (row.get("sf_pos_rank_label") or row.get("pos_rank_label") or "") if league_type == "sf" else (row.get("pos_rank_label") or ""),
            "rank_change_7d": row.get("rank_change_7d"),
            "name":           row.get("name") or players_index.get(pid, {}).get("name", f"Player {pid}"),
            "team":           row.get("team") or "",
            "age":            row.get("age"),
            "is_rookie":      bool(row.get("is_rookie")),
        }

    POSITIONS = ["QB", "RB", "WR", "TE"]

    # Compute positional value totals per roster
    def _pos_totals(player_ids: list) -> dict:
        totals = {p: 0.0 for p in POSITIONS}
        for pid in player_ids:
            info = values_by_id.get(str(pid))
            if info and info["position"] in POSITIONS:
                totals[info["position"]] += info["value"]
        return totals

    roster_totals = {str(r.get("roster_id")): _pos_totals(r.get("players") or []) for r in rosters}
    num_teams = max(len(rosters), 1)

    # Project viewer's upcoming picks to actual rookie positions/values using the
    # live rookie rankings, so need detection suppresses positions they'll draft.
    viewer_picks_list = picks_by_roster.get(viewer_roster_id, [])
    _cur_yr = datetime.now().year
    _top_rookies = sorted(
        [v for v in values_by_id.values()
         if v.get("is_rookie") and v.get("position") in POSITIONS],
        key=lambda r: float(r.get("value") or 0),
        reverse=True,
    )
    _rookie_idx = 0
    _projected_picks_out: list[dict] = []
    _pick_credits: dict[str, float] = {}
    for _rnd in [1, 2]:
        for _pk in sorted(
            [p for p in viewer_picks_list
             if p.get("round") == _rnd and int(p.get("season", 0)) <= _cur_yr + 1],
            key=lambda p: p.get("season", 9999),
        ):
            if _rookie_idx < len(_top_rookies):
                _proj = _top_rookies[_rookie_idx]
                _pos  = _proj.get("position", "")
                _val  = float(_proj.get("value") or 0)
                _pick_credits[_pos] = _pick_credits.get(_pos, 0.0) + _val
                _projected_picks_out.append({
                    "season":    _pk.get("season"),
                    "round":     _rnd,
                    "proj_name": _proj.get("name", ""),
                    "proj_pos":  _pos,
                    "proj_val":  round(_val, 1),
                })
                _rookie_idx += 1
    if _pick_credits and viewer_roster_id in roster_totals:
        vt = dict(roster_totals[viewer_roster_id])
        for _pos, _val in _pick_credits.items():
            vt[_pos] = vt.get(_pos, 0.0) + _val
        roster_totals[viewer_roster_id] = vt

    # Rank each roster by positional total (1 = best)
    pos_ranks: dict[str, dict] = {}  # pos -> {rid: rank}
    for pos in POSITIONS:
        sorted_rids = sorted(roster_totals.keys(), key=lambda rid: roster_totals[rid].get(pos, 0), reverse=True)
        pos_ranks[pos] = {rid: i + 1 for i, rid in enumerate(sorted_rids)}

    # Viewer is needy at a position if they rank in the bottom 35%
    need_cutoff = max(1, round(num_teams * 0.35))
    needed_positions = [
        pos for pos in POSITIONS
        if pos_ranks[pos].get(viewer_roster_id, num_teams) > num_teams - need_cutoff
    ]

    # Compute the viewer's realistic offer ceiling: best 2 players + pick value.
    # Targets are filtered to players the viewer could plausibly acquire.
    viewer_roster_obj = next((r for r in rosters if str(r.get("roster_id")) == viewer_roster_id), None)
    viewer_player_vals = sorted(
        [float(values_by_id[str(p)]["value"]) for p in (viewer_roster_obj.get("players") or [])
         if str(p) in values_by_id and values_by_id[str(p)]["value"] >= 150],
        reverse=True,
    ) if viewer_roster_obj else []
    _pick_val_est = sum(
        650 if p.get("round") == 1 else 220 if p.get("round") == 2 else 80
        for p in picks_by_roster.get(viewer_roster_id, [])
        if int(p.get("season", 0)) <= _cur_yr + 1
    )
    # Max realistic package: top 2 players + picks, with 20% premium the buyer might pay
    _offer_1for1  = (viewer_player_vals[0] * 1.25) if viewer_player_vals else 300
    _offer_package = (sum(viewer_player_vals[:2]) + _pick_val_est) * 1.2 if viewer_player_vals else 500
    _realistic_max = max(_offer_1for1, _offer_package)

    # Collect players from other teams, filtered by realistic acquire value
    all_collected: dict[str, list] = {pos: [] for pos in POSITIONS}
    for roster in rosters:
        rid = str(roster.get("roster_id"))
        if rid == viewer_roster_id:
            continue
        team_name = roster_map.get(rid, f"Roster {rid}")
        for pid in (roster.get("players") or []):
            pid = str(pid)
            info = values_by_id.get(pid)
            if not info or info["position"] not in POSITIONS or info["value"] < 150:
                continue
            if info["value"] > _realistic_max:
                continue  # not realistically acquirable given viewer's trade assets
            all_collected[info["position"]].append({
                "player_id":       pid,
                "name":            info["name"],
                "position":        info["position"],
                "nfl_team":        info["team"],
                "age":             info["age"],
                "value":           round(info["value"], 1),
                "pos_rank_label":  info["pos_rank_label"],
                "rank_change_7d":  info["rank_change_7d"],
                "owner_team":      team_name,
                "owner_roster_id": rid,
            })

    for pos in POSITIONS:
        all_collected[pos].sort(key=lambda p: p["value"], reverse=True)

    position_ranks_out = {pos: pos_ranks[pos].get(viewer_roster_id, num_teams) for pos in POSITIONS}

    if not needed_positions:
        # Balanced team: return top 2 per position as a discovery/browsing view
        all_positions = {pos: all_collected[pos][:2] for pos in POSITIONS if all_collected[pos]}
        return jsonify({
            "by_position": {}, "all_positions": all_positions,
            "position_ranks": position_ranks_out,
            "projected_picks": _projected_picks_out,
        })

    by_position = {pos: all_collected[pos][:4] for pos in needed_positions if all_collected[pos]}

    return jsonify({
        "by_position": by_position,
        "all_positions": {},
        "position_ranks": position_ranks_out,
        "projected_picks": _projected_picks_out,
    })


def _real_trade_packages_for_target(
    target_player_id: str,
    is_sf: bool,
    num_teams: int,
    viewer_players: list[dict],
    viewer_picks: list[dict],
    values_by_id: dict,
    max_packages: int = 3,
) -> dict:
    """
    Find real trades where target_player_id was acquired in comparable leagues,
    then match the sent-asset patterns against the viewer's roster.

    Returns {"packages": [...], "total_real_trades": N}
    Each package has the same shape as value-based packages plus "trades_like_this".
    """
    from collections import defaultdict
    try:
        from dashboard_services.db import get_conn as _gc
        with _gc() as conn:
            rows = conn.execute(
                """
                WITH acquisitions AS (
                    SELECT DISTINCT t.id AS trade_id, a_in.side AS recv_side
                    FROM trade_intel_trades t
                    JOIN trade_intel_leagues l ON l.league_id = t.league_id
                    JOIN trade_intel_assets a_in
                         ON a_in.trade_id = t.id
                        AND a_in.asset_type = 'player'
                        AND a_in.player_id = %s
                    WHERE l.league_type = 2
                      AND COALESCE(l.is_superflex, FALSE) = %s
                      AND COALESCE(l.num_teams, 12) BETWEEN %s AND %s
                      AND t.created_at > NOW() - INTERVAL '365 days'
                    LIMIT 300
                )
                SELECT
                    acq.trade_id,
                    a.asset_type,
                    a.player_id  AS sent_player_id,
                    a.pick_round,
                    a.pick_season,
                    a.pick_order
                FROM acquisitions acq
                JOIN trade_intel_assets a
                     ON a.trade_id = acq.trade_id
                    AND a.side != acq.recv_side
                ORDER BY acq.trade_id
                """,
                (target_player_id, is_sf, num_teams - 2, num_teams + 2),
            ).fetchall()
    except Exception:
        return {"packages": [], "total_real_trades": 0}

    # Group assets by trade_id → list of asset dicts
    trade_pkgs: dict = defaultdict(list)
    for row in rows:
        trade_pkgs[row["trade_id"]].append({
            "asset_type":     row["asset_type"],
            "sent_player_id": row["sent_player_id"],
            "pick_round":     row["pick_round"],
            "pick_season":    row["pick_season"],
            "pick_order":     row["pick_order"],
        })

    total_real_trades = len(trade_pkgs)
    if not total_real_trades:
        return {"packages": [], "total_real_trades": 0}

    # Build position/value signature for each trade package, then count frequencies
    def _sig(assets: list[dict]) -> Optional[tuple]:
        parts = []
        for a in sorted(assets, key=lambda x: x["asset_type"]):
            if a["asset_type"] == "player" and a["sent_player_id"]:
                info = values_by_id.get(str(a["sent_player_id"]))
                if not info:
                    continue
                pos = info["position"]
                val = info["value"]
                # Bucket value so similar-value swaps collapse to the same signature
                bucket = "elite" if val >= 900 else "high" if val >= 550 else "mid" if val >= 300 else "low"
                parts.append(f"P:{pos}:{bucket}")
            elif a["asset_type"] == "pick" and a["pick_round"]:
                parts.append(f"K:{a['pick_round']}")
        return tuple(sorted(parts)) if parts else None

    sig_counts: dict = defaultdict(list)
    for trade_id, assets in trade_pkgs.items():
        s = _sig(assets)
        if s:
            sig_counts[s].append(trade_id)

    # Viewer helpers: players by position, picks by round
    vp_by_pos: dict = defaultdict(list)
    for vp in viewer_players:
        vp_by_pos[vp["position"]].append(vp)

    vk_by_round: dict = defaultdict(list)
    for pk in viewer_picks:
        name = pk.get("name", "")
        for rnd, marker in ((1, "1st"), (2, "2nd"), (3, "3rd")):
            if marker in name:
                vk_by_round[rnd].append(pk)
                break

    VALUE_RANGES = {
        "elite": (700, 1400),
        "high":  (400, 800),
        "mid":   (220, 600),
        "low":   (100, 400),
    }

    result_packages = []
    used_pids: set = set()

    # Track which sigs we fall back on so we don't double-count
    fallback_packages = []

    for sig, trade_ids in sorted(sig_counts.items(), key=lambda x: -len(x[1])):
        trades_like_this = len(trade_ids)
        matched: list[dict] = []
        temp_used: set = set()
        ok = True

        for part in sig:
            kind, *rest = part.split(":")
            if kind == "P":
                pos, bucket = rest
                lo, hi = VALUE_RANGES.get(bucket, (100, 2000))
                candidates = [
                    vp for vp in vp_by_pos.get(pos, [])
                    if vp["player_id"] not in used_pids
                    and vp["player_id"] not in temp_used
                    and lo <= vp["value"] <= hi
                ]
                if not candidates:
                    ok = False
                    break
                mid_val = (lo + hi) / 2
                best = min(candidates, key=lambda p: abs(p["value"] - mid_val))
                matched.append(best)
                temp_used.add(best["player_id"])
            elif kind == "K":
                rnd = int(rest[0])
                available = [pk for pk in vk_by_round.get(rnd, [])]
                if not available:
                    ok = False
                    break
                matched.append(available[0])

        if not ok or not matched:
            # Build a fallback package describing the pattern (no viewer-specific players)
            fallback_assets = []
            for part in sig:
                kind, *rest = part.split(":")
                if kind == "P":
                    pos, bucket = rest
                    lo, hi = VALUE_RANGES.get(bucket, (100, 2000))
                    mid_val = (lo + hi) / 2
                    # Find any player in values_by_id matching this pos + value range
                    candidates = [
                        {"player_id": pid, "name": info["name"], "position": pos,
                         "value": info["value"], "pos_rank_label": info.get("pos_rank_label", ""),
                         "is_reference": True}
                        for pid, info in values_by_id.items()
                        if info["position"] == pos and lo <= info["value"] <= hi
                    ]
                    if candidates:
                        best = min(candidates, key=lambda p: abs(p["value"] - mid_val))
                        fallback_assets.append(best)
                elif kind == "K":
                    rnd = int(rest[0])
                    suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(rnd, f"{rnd}th")
                    fallback_assets.append({
                        "name": f"{suffix} Round Pick", "is_pick": True,
                        "value": 200 if rnd == 1 else 100,
                        "is_reference": True,
                    })
            if fallback_assets:
                fallback_packages.append({
                    "type":             "real-trade",
                    "trades_like_this": trades_like_this,
                    "send":             fallback_assets,
                    "send_value":       round(sum(a.get("value", 0) for a in fallback_assets), 1),
                    "is_reference":     True,
                })
            continue

        send_value = round(sum(a.get("value", 0) for a in matched), 1)

        result_packages.append({
            "type":             "real-trade",
            "trades_like_this": trades_like_this,
            "send":             matched,
            "send_value":       send_value,
        })
        for a in matched:
            if not a.get("is_pick"):
                used_pids.add(a["player_id"])

        if len(result_packages) >= max_packages:
            break

    # If no viewer-matched packages found, fall back to reference packages
    if not result_packages and fallback_packages:
        result_packages = sorted(fallback_packages, key=lambda x: -x["trades_like_this"])[:max_packages]

    return {"packages": result_packages, "total_real_trades": total_real_trades}


@app.route("/api/trade-ideas-for-target", methods=["POST"])
@limiter.limit("20 per minute")
def api_trade_ideas_for_target():
    """
    Given a specific target player, return packages the viewer could send to acquire them.
    Packages are value-matched (85–115% of target value) and never include absurd multi-star sends.
    """
    payload          = request.get_json(force=True)
    league_id        = str(payload.get("league_id")        or "").strip()
    season           = int(payload.get("season")           or datetime.now().year)
    platform         = str(payload.get("platform")         or "sleeper").strip()
    viewer_roster_id = str(payload.get("viewer_roster_id") or "").strip()
    target_player_id = str(payload.get("target_player_id") or "").strip()
    league_type      = str(payload.get("league_type")      or "1qb").strip()

    if not league_id or not viewer_roster_id or not target_player_id:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        from utils.utils import load_model_value_table
        ctx = get_league_ctx_from_cache(platform, league_id, season)

        val_key = "sf_value" if league_type == "sf" else "value"
        value_table = load_model_value_table() or []
        values_by_id = {}
        for p in value_table:
            pid = str(p.get("id") or "")
            if pid:
                values_by_id[pid] = {
                    "name":           p.get("name", ""),
                    "position":       str(p.get("position") or "").upper(),
                    "value":          float(p.get(val_key) or p.get("value") or 0),
                    "sf_value":       float(p.get("sf_value") or p.get("value") or 0),
                    "pos_rank":       int(p.get("pos_rank") or 99),
                    "pos_rank_label": p.get("pos_rank_label") or "",
                    "team":           p.get("team") or "",
                    "age":            p.get("age"),
                }

        target_info = values_by_id.get(target_player_id)
        if not target_info:
            return jsonify({"error": "Target player not found in value table"}), 404

        target_value = target_info["value"]

        # Dynasty premium: elite young skill-position players command a real-market overpay.
        # QBs in 1QB leagues are valued by current production, not aging curve, so they
        # get no premium. Only RB/WR/TE (and QBs in SF) warrant dynasty markup.
        #
        #   age_factor  = e^(-0.25 * max(0, age - 22))  → 1.0 @ 22, ~0.37 @ 26, 0 @ 30+
        #   rank_factor = e^(-0.12 * max(0, rank - 1))  → 1.0 @ rank 1, ~0.30 @ rank 10
        #   premium     = 1 + age_factor * rank_factor * 0.25
        #                 max ~1.25 (age 22, rank 1), fades to 1.0 for old/deep players
        import math as _math
        def _dynasty_premium(info: dict) -> float:
            pos = info.get("position", "")
            if pos in ("PICK", "K", "DEF"):
                return 1.0
            # QBs don't carry dynasty premium in 1QB — their value tracks production
            if pos == "QB" and league_type == "1qb":
                return 1.0
            age      = float(info.get("age") or 99)
            pos_rank = int(info.get("pos_rank") or 99)
            if age >= 30:
                return 1.0
            age_factor  = _math.exp(-0.25 * max(0.0, age - 22))
            rank_factor = _math.exp(-0.12 * max(0.0, pos_rank - 1))
            return round(1.0 + age_factor * rank_factor * 0.15, 3)

        premium       = _dynasty_premium(target_info)
        effective_target = target_value * premium   # what you actually need to send

        rosters      = ctx.get("rosters") or []
        roster_map   = ctx.get("roster_map") or {}
        picks_by_roster = ctx.get("picks_by_roster") or {}
        _cur_yr      = season

        # Find which roster owns the target
        target_owner_rid = None
        for r in rosters:
            if target_player_id in [str(p) for p in (r.get("players") or [])]:
                target_owner_rid = str(r.get("roster_id"))
                break
        if not target_owner_rid:
            return jsonify({"error": "Target player not on any roster"}), 404

        target_owner_name = roster_map.get(target_owner_rid, "Unknown")

        # Viewer's roster — players with value ≥50, sorted desc
        viewer_roster_obj = next(
            (r for r in rosters if str(r.get("roster_id")) == viewer_roster_id), None
        )
        if not viewer_roster_obj:
            return jsonify({"error": "Viewer roster not found"}), 404

        viewer_players = sorted(
            [
                {
                    "player_id":      pid,
                    "name":           values_by_id[pid]["name"],
                    "position":       values_by_id[pid]["position"],
                    "value":          values_by_id[pid]["value"],
                    "pos_rank_label": values_by_id[pid]["pos_rank_label"],
                }
                for pid in [str(p) for p in (viewer_roster_obj.get("players") or [])]
                if pid in values_by_id and values_by_id[pid]["value"] >= 50
            ],
            key=lambda x: x["value"],
            reverse=True,
        )

        # Viewer's picks (current + next season only) — use real values from value table
        pick_val_lookup = {
            str(p.get("id") or ""): float(p.get("value") or 0)
            for p in value_table
            if str(p.get("position") or "").upper() == "PICK"
        }

        def _resolve_pick(p: dict) -> dict:
            """Return {name, pick_id, value, is_pick} for a pick, resolving exact slot when possible.

            Accepts both picks_by_roster dicts (original_owner key) and raw Sleeper
            draft_picks dicts (roster_id = original team).
            """
            yr  = int(p.get("season") or _cur_yr)
            rnd = int(p.get("round") or 4)
            # draft_picks from Sleeper use roster_id for original team;
            # picks_by_roster uses original_owner
            original_owner = p.get("roster_id") or p.get("original_owner")

            slot = None
            if original_owner:
                slot = resolve_exact_pick_slot(
                    platform, league_id, season,
                    {"season": yr, "round": rnd, "previous_owner_id": int(original_owner)},
                )

            if slot is not None:
                label = f"{yr} {rnd}.{slot:02d}"
                bucket = "early" if slot <= 4 else "mid" if slot <= 8 else "late"
                # Prefer exact slot key (both zero-padded-round and plain), then bucket fallback
                val_keys = [
                    f"{yr}_{rnd:02d}_{slot:02d}",
                    f"{yr}_{rnd}_{slot:02d}",
                    f"{yr}_{rnd}_{bucket}",
                    f"{yr}_{rnd}",
                ]
            else:
                sfx = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
                label = f"{yr} {rnd}{sfx}"
                val_keys = [f"{yr}_{rnd}_early", f"{yr}_{rnd}_mid", f"{yr}_{rnd}_late", f"{yr}_{rnd}"]

            # Use whichever key is actually in the table so pick_id is always parseable
            pick_id = next((k for k in val_keys if k in pick_val_lookup), val_keys[0])
            value   = pick_val_lookup.get(pick_id)
            if value is None:
                value = 650.0 if rnd == 1 else 220.0 if rnd == 2 else 80.0

            return {"name": label, "pick_id": pick_id, "value": value, "is_pick": True}

        # Build the viewer's pick list from picks_by_roster (which accounts for
        # traded picks), then remove any picks that the traded list shows the viewer
        # no longer owns. This handles the case where build_picks_by_roster missed
        # a multi-hop trade.
        traded_list = ctx.get("traded") or []
        viewer_rid_int = int(viewer_roster_id)
        # Set of (season, round) the viewer has traded away to someone else
        traded_away = {
            (int(tp.get("season", 0)), int(tp.get("round", 0)))
            for tp in traded_list
            if (int(tp.get("roster_id", -1)) == viewer_rid_int
                and int(tp.get("owner_id", -1)) != viewer_rid_int)
        }
        viewer_picks = sorted(
            [
                _resolve_pick(p)
                for p in picks_by_roster.get(viewer_roster_id, [])
                if (int(p.get("season", 0)) <= _cur_yr + 1
                    and (int(p.get("season", 0)), int(p.get("round", 0))) not in traded_away)
            ],
            key=lambda x: x["value"],
            reverse=True,
        )

        # Match packages against effective_target (face value × dynasty premium).
        # Tight band so suggestions don't overshoot what the market actually demands.
        lo = effective_target * 0.93
        hi = effective_target * 1.06
        packages = []
        seen = set()

        def _key(*assets):
            return tuple(sorted(a.get("player_id") or a.get("name", "") for a in assets))

        # 1-for-1: single player in range
        for p in viewer_players:
            if lo <= p["value"] <= hi:
                k = _key(p)
                if k not in seen:
                    seen.add(k)
                    packages.append({"type": "1-for-1", "send": [p],
                                     "send_value": p["value"],
                                     "_delta": abs(p["value"] - effective_target)})

        # 2-for-1: neither player alone covers >75% of effective_target
        for i, p1 in enumerate(viewer_players):
            if p1["value"] > effective_target * 0.75:
                continue
            for p2 in viewer_players[i + 1:]:
                if p2["value"] < 60:
                    break
                combined = p1["value"] + p2["value"]
                if combined > hi:
                    continue
                if combined >= lo:
                    k = _key(p1, p2)
                    if k not in seen:
                        seen.add(k)
                        packages.append({"type": "2-for-1", "send": [p1, p2],
                                         "send_value": combined,
                                         "_delta": abs(combined - effective_target)})
                    break

        # Player + pick
        for p in viewer_players:
            if p["value"] > effective_target * 0.85:
                continue
            for pick in viewer_picks:
                combined = p["value"] + pick["value"]
                if lo <= combined <= hi:
                    k = _key(p, {"player_id": pick["name"]})
                    if k not in seen:
                        seen.add(k)
                        packages.append({"type": "player + pick", "send": [p, pick],
                                         "send_value": combined,
                                         "_delta": abs(combined - effective_target)})
                    break

        packages.sort(key=lambda x: (x["_delta"], len(x["send"])))
        for pkg in packages:
            del pkg["_delta"]

        # Include full player fields needed by the trade calculator
        target_calc = {
            "id":               target_player_id,
            "name":             target_info["name"],
            "position":         target_info["position"],
            "team":             target_info["team"],
            "value":            round(target_value, 1),
            "sf_value":         round(target_info["sf_value"], 1),
            "pos_rank_label":   target_info["pos_rank_label"],
            "sf_pos_rank_label": target_info["pos_rank_label"],
        }

        def _enrich_pkg_assets(pkg_list: list) -> None:
            for pkg in pkg_list:
                for asset in pkg["send"]:
                    if not asset.get("is_pick"):
                        info = values_by_id.get(asset.get("player_id") or "")
                        if info:
                            asset.update({
                                "id":               asset["player_id"],
                                "position":         info["position"],
                                "team":             info["team"],
                                "sf_value":         round(info.get("sf_value", info["value"]), 1),
                                "pos_rank_label":   info["pos_rank_label"],
                                "sf_pos_rank_label": info["pos_rank_label"],
                            })

        _enrich_pkg_assets(packages)

        # Real trade packages: what people with similar rosters actually sent
        roster_positions = ctx.get("roster_positions") or []
        _rp_list = [str(s).upper() for s in (roster_positions if isinstance(roster_positions, list) else [])]
        _is_sf = any(s in {"SUPER_FLEX", "SFLEX"} for s in _rp_list)

        real_result = _real_trade_packages_for_target(
            target_player_id=target_player_id,
            is_sf=_is_sf,
            num_teams=len(rosters) or 12,
            viewer_players=viewer_players,
            viewer_picks=viewer_picks,
            values_by_id=values_by_id,
        )
        _enrich_pkg_assets(real_result["packages"])

        return jsonify({
            "success":            True,
            "target":             target_calc,
            "owner":              target_owner_name,
            "packages":           packages[:4],
            "real_packages":      real_result["packages"],
            "total_real_trades":  real_result["total_real_trades"],
            "target_value":       round(target_value, 1),
            "effective_target":   round(effective_target, 1),
            "premium":            round(premium, 2),
        })

    except Exception as e:
        logger.exception("[api-trade-ideas-for-target] Error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/player-news/<player_id>")
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


@app.route("/api/nfl-news")
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


def _run_startup_daily() -> None:
    """Fire daily data build in the background immediately on startup."""
    global daily_completed
    try:
        today_et: date = datetime.now(EASTERN).date()
        if daily_lock.acquire(blocking=False):
            try:
                if daily_completed != today_et:
                    logger.info("[startup] Kicking off daily build for %s in background...", today_et)
                    state = get_nfl_state() or {}
                    season = int(state.get("season") or datetime.now().year)
                    week = int(state.get("week") or 0)
                    run_daily_data_async(season, week)
                    daily_completed = today_et
            finally:
                daily_lock.release()
    except Exception as e:
        logger.warning("[startup] Could not kick off daily build: %s", e)


threading.Thread(target=_run_startup_daily, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True)
