import hashlib
import html
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
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
    avatar_url
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
from dashboard_services.pages.waivers_page import build_waivers_body
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
from dashboard_services.subscriptions import (
    has_premium_access,
    create_league_subscription,
    create_user_subscription,
    cancel_subscription,
)
import stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
from dashboard_services.providers.espn_api import safe_float
from dashboard_services.service import (
    age_from_bday,
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
# Per-key locks prevent simultaneous first-loads for the same league from both
# running the full build_league_context (~40 API calls) at the same time.
_CTX_LOCKS: dict = {}
_CTX_LOCKS_LOCK = threading.Lock()

# How long a league context is considered fresh
CACHE_TTL = 60 * 60 * 12  # 12 hours

# How long value-table cache entries live
VALUE_CACHE_TTL = 60 * 60 * 6  # 6 hours

# How long to cache rendered page HTML (Teams, Activity, Graphs) per league
PAGE_HTML_TTL = 60 * 30  # 30 minutes

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

def _static_hash(filename: str) -> str:
    path = Path(__file__).parent / "static" / filename
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()[:8]
    except OSError:
        return "0"

_APP_JS_V = _static_hash("app.js")
_PAYWALL_JS_V = _static_hash("paywall.js")


@app.after_request
def _add_cache_headers(response):
    path = request.path
    if path.startswith("/static/"):
        # Versioned assets (e.g. ?v=123) can be cached aggressively; others get 1-day
        if request.args.get("v"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "public, max-age=86400"
    return response

_secret_key = os.environ.get('FLASK_SECRET_KEY', '')
if not _secret_key:
    logging.warning(
        "FLASK_SECRET_KEY is not set — using insecure default. "
        "Set this env var in production to protect session cookies."
    )
    _secret_key = 'dev-secret-key-change-in-production'
app.secret_key = _secret_key
del _secret_key

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

# ── Blueprint registrations ────────────────────────────────────────────────────
try:
    from routes.public_bp import public_bp
    app.register_blueprint(public_bp)
    logger.info("[public-bp] registered")
except Exception as e:
    logger.warning("[public-bp] skipped: %s", e)

try:
    from routes.auth_bp import auth_bp
    app.register_blueprint(auth_bp)
    logger.info("[auth-bp] registered")
except Exception as e:
    logger.warning("[auth-bp] skipped: %s", e)

try:
    from routes.billing_bp import billing_bp
    app.register_blueprint(billing_bp)
    logger.info("[billing-bp] registered")
except Exception as e:
    logger.warning("[billing-bp] skipped: %s", e)

try:
    from routes.trade_bp import trade_bp
    app.register_blueprint(trade_bp)
    logger.info("[trade-bp] registered")
except Exception as e:
    logger.warning("[trade-bp] skipped: %s", e)

try:
    from routes.players_bp import players_bp
    app.register_blueprint(players_bp)
    logger.info("[players-bp] registered")
except Exception as e:
    logger.warning("[players-bp] skipped: %s", e)



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
          <input type="hidden" name="next" id="formNext" value="{{ next_url or '' }}">

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
    <section class="home-feature-list-card">
    <section class="home-feature-list">
      <div class="home-feature-row">
        <div class="home-feature-row-icon"><svg style="width:20px;height:20px;color:#3b82f6;" viewBox="0 0 24 24" fill="currentColor"><path d="M5 9.2h3V19H5V9.2zM10.6 5h2.8v14h-2.8V5zm5.6 8H19v6h-2.8v-6z"/></svg></div>
        <div class="home-feature-row-body"><span class="home-feature-row-title">Trade Calculator</span><span class="home-feature-row-desc">AI-powered deal evaluation with counter suggestions and similar trade history</span></div>
      </div>
      <div class="home-feature-row">
        <div class="home-feature-row-icon"><svg style="width:20px;height:20px;color:#10b981;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg></div>
        <div class="home-feature-row-body"><span class="home-feature-row-title">Dynasty Rankings</span><span class="home-feature-row-desc">Daily-updated hybrid values blending consensus data with advanced metrics</span></div>
      </div>
      <div class="home-feature-row">
        <div class="home-feature-row-icon"><svg style="width:20px;height:20px;color:#a78bfa;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg></div>
        <div class="home-feature-row-body"><span class="home-feature-row-title">Rookie Prospects</span><span class="home-feature-row-desc">Prospect profiles with college metrics, athleticism scores, and ADP trends</span></div>
      </div>
      <div class="home-feature-row">
        <div class="home-feature-row-icon"><svg style="width:20px;height:20px;color:#f59e0b;" viewBox="0 0 24 24" fill="currentColor"><path d="M7 2v11h3v9l7-12h-4l4-8z"/></svg></div>
        <div class="home-feature-row-body"><span class="home-feature-row-title">Weekly Hub</span><span class="home-feature-row-desc">Live scoring, projections, injury news, and matchup context for gameday</span></div>
      </div>
      <div class="home-feature-row">
        <div class="home-feature-row-icon"><svg style="width:20px;height:20px;color:#06b6d4;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg></div>
        <div class="home-feature-row-body"><span class="home-feature-row-title">Breakout Candidates</span><span class="home-feature-row-desc">Spot usage spikes from depth chart shifts, target share, and opportunity data</span></div>
      </div>
      <div class="home-feature-row">
        <div class="home-feature-row-icon"><svg style="width:20px;height:20px;color:#f97316;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg></div>
        <div class="home-feature-row-body"><span class="home-feature-row-title">Waiver Wire</span><span class="home-feature-row-desc">Personalized pickups ranked by roster fit and positional need</span></div>
      </div>
      <div class="home-feature-row">
        <div class="home-feature-row-icon"><svg style="width:20px;height:20px;color:#ef4444;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg></div>
        <div class="home-feature-row-body"><span class="home-feature-row-title">Team Analytics</span><span class="home-feature-row-desc">Position grades, roster composition, and competitive advantages across the league</span></div>
      </div>
      <div class="home-feature-row">
        <div class="home-feature-row-icon"><svg style="width:20px;height:20px;color:#f43f5e;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg></div>
        <div class="home-feature-row-body"><span class="home-feature-row-title">Graphs & Trends</span><span class="home-feature-row-desc">PF/PA, SOS, playoff odds, luck metrics, and standings trajectory over time</span></div>
      </div>
      <div class="home-feature-row">
        <div class="home-feature-row-icon"><i class="fa-solid fa-trophy" style="font-size:18px;color:#eab308;" aria-hidden="true"></i></div>
        <div class="home-feature-row-body"><span class="home-feature-row-title">Historical Insights</span><span class="home-feature-row-desc">AI season recaps, rivalry records, draft grades, and full championship history</span></div>
      </div>
    </section>
    </section>

    <aside class="home-updates-sidebar">
      <h3 class="home-updates-sidebar-title">Recent Updates</h3>
      <div class="home-updates-list">
        {{ recent_updates | safe }}
      </div>
    </aside>
  </div>
</div>

<div class="fullscreen-loading-overlay" id="dashboardLoadingOverlay" style="display:none;">
  <div class="loading-spinner"></div>
  <div class="fullscreen-loading-text">Building your dashboard…</div>
  <div class="fullscreen-loading-subtext">This usually takes 10–20 seconds</div>
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
    {adsense_script}

    <link rel="icon" href="/static/BR_Logo.png" type="image/x-icon">
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#38bdf8">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="mobile-web-app-title" content="BR Fantasy">

    <link rel="stylesheet" href="/static/dashboard.css">
    <link rel="stylesheet" href="/static/icons.css">
    <link rel="stylesheet" href="/static/font-awesome.css">
    <link rel="stylesheet" href="/static/paywall.css">

    <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
    <script>
      if ('serviceWorker' in navigator) {{
        navigator.serviceWorker.register('/sw.js').catch(() => {{}});
      }}
    </script>
  </head>
  <body>
    <div id="app-scale">
      {nav}

      {ad_top}

      <main id="page-root" class="overview-layout">
        {body}
      </main>

      {ad_bottom}
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

    <!-- Page navigation loading overlay -->
    <div id="navLoadingOverlay" class="fullscreen-loading-overlay" style="display:none;">
      <div class="loading-spinner"></div>
      <div class="fullscreen-loading-text">Loading&hellip;</div>
    </div>

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

    <script src="/static/app.js?v={app_js_v}"></script>
    <script src="/static/paywall.js?v={paywall_js_v}"></script>
    <script>
      {adsense_init}

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

      // Page navigation loading spinner
      (function() {{
        var overlay = document.getElementById('navLoadingOverlay');
        if (!overlay) return;
        var shown = false;
        function showOverlay() {{
          if (shown) return;
          shown = true;
          overlay.style.display = '';
        }}
        document.addEventListener('click', function(e) {{
          var a = e.target.closest('a[href]');
          if (!a) return;
          var href = a.getAttribute('href');
          if (!href || href.startsWith('#') || href.startsWith('javascript') || href.startsWith('mailto')) return;
          if (a.target === '_blank') return;
          var url;
          try {{ url = new URL(href, window.location.href); }} catch(err) {{ return; }}
          if (url.origin !== window.location.origin) return;
          if (url.pathname === window.location.pathname && url.search === window.location.search) return;
          showOverlay();
        }});
        window.addEventListener('popstate', function() {{
          shown = false;
          overlay.style.display = 'none';
        }});
        window.addEventListener('pageshow', function() {{
          shown = false;
          overlay.style.display = 'none';
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


_SEED_SEMAPHORE = threading.Semaphore(1)


def _background_seed_user(user_id: str, username: Optional[str]) -> None:
    """Fire-and-forget: record login and seed dynasty leagues for a Sleeper user."""
    from data_building.trade_intel.league_discovery import _save_users as _tiu_save_users
    try:
        _tiu_save_users([user_id], source="login", usernames={user_id: username} if username else None)
    except Exception:
        pass

    def _run():
        if not _SEED_SEMAPHORE.acquire(blocking=False):
            return  # another seed already running; skip rather than pile up
        try:
            _seed_user_leagues(user_id, username=username)
        except Exception:
            pass
        finally:
            _SEED_SEMAPHORE.release()
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
_players_global_lock = threading.Lock()
_players_index_lock = threading.Lock()


def get_players_global():
    global _PLAYERS_GLOBAL
    if _PLAYERS_GLOBAL is None:
        with _players_global_lock:
            if _PLAYERS_GLOBAL is None:
                _PLAYERS_GLOBAL = get_nfl_players()
    return _PLAYERS_GLOBAL


def get_players_index_global():
    global _PLAYERS_INDEX_GLOBAL
    if _PLAYERS_INDEX_GLOBAL is None:
        with _players_index_lock:
            if _PLAYERS_INDEX_GLOBAL is None:
                _PLAYERS_INDEX_GLOBAL = load_players_index()
    return _PLAYERS_INDEX_GLOBAL


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
    # Refine active key based on ?tab= param so sub-tab nav items highlight correctly
    _tab_param = request.args.get("tab", "")
    if active == "trade" and _tab_param == "suggestions":
        active = "trade-suggestions"
    elif active == "prospects" and _tab_param == "draft":
        active = "prospects-draft"

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
                ("Suggestions <span class='nav-pro-badge'>PRO</span>", "/trade?tab=suggestions", "trade-suggestions"),
                ("Trade Database",   "/trade-database", "trade-database"),
                ("Trade Intel",      "/trade-intel",    "trade-intel"),
            ], ["trade", "trade-database", "trade-intel"], "tradesNavDropdown"),
            simple_dropdown("Players", [
                ("Player Rankings", "/players",   "players"),
                ("Prospects",       "/prospects",   "prospects"),
                ("Draft Assistant <span class='nav-pro-badge'>PRO</span>", "/prospects?tab=draft", "prospects-draft"),
                ("Breakouts",       "/breakouts", "breakouts"),
            ], ["players", "prospects", "breakouts"], "playersNavDropdown"),
        ]

        player_search_html = (
            "<div class='nav-search-wrapper' id='navSearchWrapper'>"
            "  <div class='nav-search-inner'>"
            "    <img src='/static/images/magnifying-glass-solid.png' class='nav-search-icon' alt='Search'>"
            "    <input type='text' id='navPlayerSearch' class='nav-search-input'"
            "           placeholder='Search players…' autocomplete='off' spellcheck='false' aria-label='Search players'/>"
            "    <button type='button' class='nav-search-clear' id='navSearchClear' aria-label='Clear search'>×</button>"
            "  </div>"
            "  <div class='nav-search-dropdown' id='navSearchDropdown'></div>"
            "</div>"
        )

        # Build utility bar for home screen (just settings gear with dark mode)
        home_utility_bar = (
            "<div class='nav-utility-bar'>"
            f"  {player_search_html}"
            f"  {changelog_bell}"
            f"  {settings_gear}"
            "  <button class='nav-hamburger utility-icon-btn' id='navToggle' aria-label='Menu'>"
            "    <i class='fa-solid fa-bars' aria-hidden='true'></i>"
            "  </button>"
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
        for item_tuple in items:
            item_label, endpoint, item_key = item_tuple[0], item_tuple[1], item_tuple[2]
            disabled = item_tuple[3] if len(item_tuple) > 3 else False
            href_suffix = item_tuple[4] if len(item_tuple) > 4 else ""
            if disabled:
                item_html += (
                    f"<span class='nav-pill-dropdown-item disabled'>"
                    f"{item_label} <span style='font-size:10px;margin-left:4px;'>Soon</span>"
                    f"</span>"
                )
            else:
                href = url_for(endpoint, platform=platform, season=season, league_id=league_id) + href_suffix
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
        ("Trade Calculator", "trade.page_trade",          "trade",          False),
        ("Suggestions <span class='nav-pro-badge'>PRO</span>", "trade.page_trade", "trade-suggestions", False, "?tab=suggestions"),
        ("Trade Database",   "trade.page_trade_database", "trade-database", False),
        ("Trade Intel",      "trade.page_trade_intel",    "trade-intel",    False),
    ], ["trade", "trade-database", "trade-intel"], "tradesNavDropdown"))
    # Weekly Hub only makes sense once games are being played
    draft_ended = has_draft_ended(league_id, platform, season)
    if not offseason_mode:
        nav_pills.append(nav_pill("Weekly Hub", "page_weekly", "weekly"))
    nav_pills.append(nav_pill_dropdown("League", [
        ("Standings", "page_standings", "standings", False),
        ("Teams",     "page_teams",     "teams",     False),
        ("Activity",  "page_activity",  "activity",  False),
        ("Waivers",   "page_waivers",   "waivers",   False),
    ], ["standings", "teams", "activity", "waivers"], "teamsNavDropdown"))
    nav_pills.append(nav_pill_dropdown("Players", [
        ("Player Rankings",   "page_players",   "players",   False),
        ("Prospect Rankings", "page_prospects",  "prospects", False),
        ("Draft Assistant <span class='nav-pro-badge'>PRO</span>", "page_prospects", "prospects-draft", False, "?tab=draft"),
        ("Breakout Engine",   "page_breakouts",  "breakouts", False),
    ], ["players", "prospects", "breakouts"], "playersNavDropdown"))
    nav_pills.append(nav_pill_dropdown("Stats", [
        ("Awards",  "page_awards",  "awards",  False),
        ("Graphs",  "page_graphs",  "graphs",  False),
        ("History", "page_history", "history", False),
    ], ["awards", "graphs", "history"], "statsNavDropdown"))

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

    player_search_html = (
        "<div class='nav-search-wrapper' id='navSearchWrapper'>"
        "  <div class='nav-search-inner'>"
        "    <img src='/static/images/magnifying-glass-solid.png' class='nav-search-icon' alt='Search'>"
        "    <input type='text' id='navPlayerSearch' class='nav-search-input'"
        "           placeholder='Search players…' autocomplete='off' spellcheck='false' aria-label='Search players'/>"
        "    <button type='button' class='nav-search-clear' id='navSearchClear' aria-label='Clear search'>×</button>"
        "  </div>"
        "  <div class='nav-search-dropdown' id='navSearchDropdown'></div>"
        "</div>"
    )

    # Build utility bar (desktop right side, mobile header)
    utility_bar = (
        "<div class='nav-utility-bar'>"
        f"  {player_search_html}"
        f"  {watchlist_btn}"
        f"  {changelog_bell}"
        f"  {settings_gear}"
        "  <button class='nav-hamburger utility-icon-btn' id='navToggle' aria-label='Menu'>"
        "    <i class='fa-solid fa-bars' aria-hidden='true'></i>"
        "  </button>"
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
        f"<div id='signinModal' class='signin-modal-overlay'>"
        f"  <div class='signin-modal-box'>"
        f"    <h3 class='signin-modal-title'>Sign in to your team</h3>"
        f"    <p class='signin-modal-sub'>Enter your Sleeper username to restore personalized features.</p>"
        f"    <form method='POST' action='/set-viewer'>"
        f"      <input type='hidden' name='platform' value='{platform}'>"
        f"      <input type='hidden' name='season' value='{season}'>"
        f"      <input type='hidden' name='league_id' value='{league_id}'>"
        f"      <input type='hidden' name='next' value='{request.path + ('?' + request.query_string.decode() if request.query_string else '')}'>"
        f"      <input class='signin-modal-input' type='text' name='username' placeholder='Sleeper username' autocomplete='username' autofocus>"
        f"      <div class='signin-modal-actions'>"
        f"        <button class='signin-modal-submit' type='submit'>Sign In</button>"
        f"        <button class='signin-modal-cancel' type='button'"
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
        f"    {pills_container}"
        "  </div>"
        "  <div class='nav-right'>"
        f"    {utility_bar}"
        "  </div>"
        "</nav>"
        f"{signin_modal}"
    )


_AD_SCRIPT = '<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-9164153092633845" crossorigin="anonymous"></script>'
_AD_TOP = """<div class="ad-container ad-top-banner"><ins class="adsbygoogle" style="display:block;max-height:90px;overflow:hidden;" data-ad-client="ca-pub-9164153092633845" data-ad-slot="5233061286" data-ad-format="horizontal"></ins></div>"""
_AD_BOTTOM = """<div class="ad-container ad-bottom-content"><ins class="adsbygoogle" style="display:block;max-height:90px;overflow:hidden;" data-ad-client="ca-pub-9164153092633845" data-ad-slot="5233061286" data-ad-format="horizontal"></ins></div>"""
_AD_INIT = """window.addEventListener('load', function() { setTimeout(function() { try { (adsbygoogle = window.adsbygoogle || []).push({}); (adsbygoogle = window.adsbygoogle || []).push({}); } catch(e) { console.warn('AdSense initialization error:', e); } }, 100); });"""


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

    user_id = session.get("viewer_username")
    is_premium = has_premium_access(user_id, league_id, platform or "sleeper")

    return BASE_HTML.format(
        title=title,
        nav=nav_html,
        body=wrapped_body,
        adsense_script="" if is_premium else _AD_SCRIPT,
        ad_top="" if is_premium else _AD_TOP,
        ad_bottom="" if is_premium else _AD_BOTTOM,
        adsense_init="" if is_premium else _AD_INIT,
        privacy_url=league_url("privacy", league_id),
        faq_url=league_url("faq", league_id),
        support_url=league_url("support", league_id),
        contact_url=league_url("contact", league_id),
        yt_url="https://youtube.com/@hoodiekj",
        app_js_v=_APP_JS_V,
        paywall_js_v=_PAYWALL_JS_V,
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

    # ── Fetch core league data and NFL state in parallel ─────────────────────
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac

    def _get_league():   return get_league(platform, resolved_league_id, season)
    def _get_users():    return get_users(platform, resolved_league_id, season)
    def _get_rosters():  return get_rosters(platform, resolved_league_id, season)
    def _get_traded():   return get_traded_picks(platform, resolved_league_id, season) if platform == "sleeper" else None
    def _get_drafts():
        try:
            return get_drafts(platform, resolved_league_id, season) or []
        except Exception as e:
            logger.warning("[build_league_context] failed to load drafts for league %s: %s", resolved_league_id, e)
            return []
    def _get_state():    return get_nfl_state() or {}

    _tasks = {
        "league":  _get_league,
        "users":   _get_users,
        "rosters": _get_rosters,
        "traded":  _get_traded,
        "drafts":  _get_drafts,
        "state":   _get_state,
    }
    _defaults = {"league": {}, "users": [], "rosters": [], "traded": [], "drafts": [], "state": {}}
    _results: dict = dict(_defaults)
    with _TPE(max_workers=len(_tasks)) as _pool:
        _fmap = {_pool.submit(fn): name for name, fn in _tasks.items()}
        for _fut in _ac(_fmap):
            _name = _fmap[_fut]
            try:
                _results[_name] = _fut.result()
            except Exception as _e:
                logger.warning("[build_league_context] task %s failed: %s", _name, _e)

    league  = _results["league"] or {}
    users   = _results["users"] or []
    rosters = _results["rosters"] or []
    traded  = _results["traded"] or []
    drafts  = _results["drafts"] or []
    current = _results["state"] or {}

    try:
        latest_draft = get_most_recent_valid_draft_for_season(drafts, season)
    except Exception as e:
        logger.warning("[build_league_context] failed to resolve latest draft: %s", e)
        drafts = []
        latest_draft = None

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

    # League settings — read directly from the league dict for Sleeper to avoid reading
    # module-level globals that may have been overwritten by a concurrent request.
    if platform == "sleeper":
        from dashboard_services.api import SCORING_DEFAULTS as _SCORING_DEFAULTS
        _raw_ss = (league or {}).get("scoring_settings") or {}
        scoring_settings = {**_SCORING_DEFAULTS, **_raw_ss}
        raw_scoring_settings = dict(_raw_ss)
        roster_positions = (league or {}).get("roster_positions") or []
        league_settings = (league or {}).get("settings") or {}
        total_rosters = int((league or {}).get("total_rosters") or 0)
    else:
        scoring_settings = get_effective_scoring_settings()
        raw_scoring_settings = {}
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
        # Compute draft-ended locally to avoid circular cache dependency
        _draft_ts_ms = None
        if isinstance(latest_draft, dict):
            _draft_ts_ms = _safe_int(latest_draft.get("start_time"))
        if _draft_ts_ms is None:
            _draft_ts_ms = _safe_int(league.get("draft_day"))
        _ctx_draft_ended = (
            datetime.now(EASTERN) > datetime.fromtimestamp(_draft_ts_ms / 1000, tz=EASTERN)
            if _draft_ts_ms else False
        )
        picks_by_roster = build_picks_by_roster(
            num_future_seasons=3,
            league=league,
            rosters=rosters,
            traded=traded,
            draft_ended=_ctx_draft_ended,
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
        age  = age_from_bday(pmeta.get("bDay")) or mv.get("age") or pmeta.get("age")

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
        logger.exception("[api_history_summary] Error")
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
        logger.exception("[api_history_standings] Error")
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
        logger.exception("[api_history_chart] Error")
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
                _ref_draft_ts = None
                if isinstance(latest_draft, dict):
                    _ref_draft_ts = _safe_int(latest_draft.get("start_time"))
                if _ref_draft_ts is None:
                    _ref_draft_ts = _safe_int(league.get("draft_day"))
                _ref_draft_ended = (
                    datetime.now(EASTERN) > datetime.fromtimestamp(_ref_draft_ts / 1000, tz=EASTERN)
                    if _ref_draft_ts else False
                )
                ctx["picks_by_roster"] = build_picks_by_roster(
                    num_future_seasons=3,
                    league=league,
                    rosters=rosters,
                    traded=traded,
                    draft_ended=_ref_draft_ended,
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

    gm_memo_html = ""
    front_office_html = ""

    if viewer_roster_id:
        try:
            gm_memo_html = get_team_gm_memo(ctx, str(viewer_roster_id))
        except Exception:
            pass
    else:
        try:
            front_office_html = get_front_office_briefing(ctx, str(viewer_roster_id))
        except Exception:
            pass

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


def render_power_and_playoffs(
    team_stats,
    roster_map: Dict[str, str],
    league_id: str,
    platform,
    season,
    bracket_override=None,
    seed_map_override=None,
) -> str:
    """
    Single card that shows:
      - Power Rankings (by PowerScore if present)
      - Playoff Picture (using bracket)
    bracket_override: pre-built bracket list; skips API fetch when provided.
    seed_map_override: {roster_id: seed_int}; skips seed_top6 calculation.
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
        span = abs(pmax - pmin)
        if span == 0:
            return 100.0
        return max(2.0, min(100.0, (float(v) - min(pmin, pmax)) / span * 100.0))

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
    wb = bracket_override if bracket_override is not None else get_bracket(platform, league_id, "winners", season)
    roster_avatar_map = {
        str(owner): av
        for owner, av in zip(team_stats["owner"], team_stats["avatar"])
        if pd.notna(owner)
    }

    seed_map = seed_map_override if seed_map_override is not None else seed_top6_from_team_stats(team_stats, roster_map)

    bracket_html = playoff_bracket(
        wb,
        roster_name_map=roster_map,
        roster_avatar_map=roster_avatar_map,
        seed_map=seed_map,
    )

    # Check if playoff bracket is available
    has_playoff_bracket = not bracket_html.strip().startswith('<div class=\'po-empty\'>')

    # Build tab buttons conditionally
    playoff_tab_html = ""
    playoff_panel_html = ""
    
    if has_playoff_bracket:
        playoff_tab_html = '<button class="tab-btn" data-tab="playoff">Playoff Picture</button>'
        playoff_panel_html = f'<div class="tab-panel" data-tab="playoff">{bracket_html}</div>'

    podium_card = f"""
          <div class="card power" data-section="overview">
            <div class="card-tabs" data-card="power">
              <div class="tab-strip">
                <button class="tab-btn active" data-tab="power">Power Rankings</button>
                {playoff_tab_html}
                <button class="tab-btn" data-tab="playoff-odds"
                        data-league-id="{league_id}"
                        data-platform="{platform}"
                        data-season="{season}">Playoff Odds</button>
              </div>
              <div class="tab-panels">
                <div class="tab-panel active" data-tab="power">
                  {podium_html}
                  {rankings_html}
                </div>
                {playoff_panel_html}
                <div class="tab-panel" data-tab="playoff-odds" id="playoffOddsPanel">
                  <div class="playoff-odds-loading">
                    <div class="loading-spinner-sm"></div>
                    <span>Running simulation…</span>
                  </div>
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


def _build_offseason_standings_body(ctx: dict) -> str:
    """
    Offseason standings: left = dynasty rankings table, right = power rankings
    cards (same order as the Teams page) + Playoff Odds.
    """
    roster_map        = ctx["roster_map"]
    rosters           = ctx["rosters"]
    users             = ctx.get("users") or []
    model_value_table = ctx.get("model_value_table") or []
    picks_by_roster   = ctx.get("picks_by_roster") or {}
    platform          = ctx["platform"]
    season            = ctx["season"]
    league_id_str     = str(ctx.get("resolved_league_id") or ctx.get("league_id") or "")

    # ── dynasty value lookup ──────────────────────────────────────────────────
    values_by_id: dict[str, float] = {}
    for row in model_value_table:
        if isinstance(row, dict) and row.get("id") is not None:
            try:
                values_by_id[str(row["id"])] = float(row.get("value") or 0)
            except Exception:
                pass

    pick_by_key: dict[str, float] = load_pick_value_table() or {}

    # ── avatar lookup from users ──────────────────────────────────────────────
    user_by_id = {str(u.get("user_id", "")): u for u in users}
    rid_to_avatar: dict[str, str] = {}
    for r in rosters:
        rid      = str(r.get("roster_id"))
        owner_id = str(r.get("owner_id") or "")
        u        = user_by_id.get(owner_id) or {}
        u_meta   = u.get("metadata") or {}
        u_av     = u.get("avatar") or ""
        av_raw   = u_meta.get("avatar") or (
            f"https://sleepercdn.com/avatars/{u_av}" if platform == "sleeper" and u_av else u_av
        )
        rid_to_avatar[rid] = avatar_url(av_raw) or ""

    # ── build per-team data ───────────────────────────────────────────────────
    team_rows: list[dict] = []
    for r in rosters:
        rid      = str(r.get("roster_id"))
        name     = roster_map.get(rid, f"Roster {rid}")
        pids     = [str(p) for p in (r.get("players") or [])]
        player_v = sum(values_by_id.get(p, 0.0) for p in pids)
        picks    = picks_by_roster.get(rid, []) if isinstance(picks_by_roster, dict) else []
        pick_v   = _team_pick_value(picks, pick_by_key, platform=platform,
                                    league_id=league_id_str, season=_safe_int(season, 0))
        total    = player_v + pick_v
        team_rows.append({
            "rid": rid, "name": name,
            "player_v": player_v, "pick_v": pick_v,
            "total": total, "n_players": len(pids), "n_picks": len(picks),
            "avatar": rid_to_avatar.get(rid, ""),
        })

    team_rows.sort(key=lambda x: x["total"], reverse=True)

    # ── Compute value share and projected production share ─────────────────────
    league_value_total = sum(r["total"] for r in team_rows) or 1.0
    # Offseason projected scoring from playoff odds simulation
    rid_to_proj: dict[str, float] = {}
    try:
        from data_building.simulate_playoff_odds import _estimate_from_rosters
        est_teams = _estimate_from_rosters(ctx)
        for t in est_teams:
            rid_to_proj[str(t["roster_id"])] = float(t.get("avg") or 0.0)
    except Exception:
        pass
    proj_total = sum(rid_to_proj.values()) or 1.0
    for r in team_rows:
        r["value_pct"] = r["total"] / league_value_total * 100
        r["prod_pct"]  = rid_to_proj.get(r["rid"], 0.0) / proj_total * 100

    # ── normalize to a PPG-like scale (100–160) matching Teams page formula ──
    raw_vals = [r["total"] for r in team_rows]
    raw_max  = max(raw_vals) if raw_vals else 1
    for r in team_rows:
        r["power_score"] = round(100.0 + r["total"] / max(raw_max, 1) * 60.0, 2)

    # ── synthetic team_stats DataFrame for render_power_and_playoffs ─────────
    df_rows = []
    for i, r in enumerate(team_rows):
        df_rows.append({
            "owner":      r["name"],
            "Wins":       0,
            "Losses":     0,
            "Ties":       0,
            "G":          0,
            "Win%":       0.0,
            "PF":         0.0,
            "PA":         0.0,
            "PowerScore": r["power_score"],
            "Streak":     "",
            "StreakLen":  0,
            "StreakType": "",
            "avatar":     r["avatar"],
        })
    synthetic_ts = pd.DataFrame(df_rows)

    # ── seed map from playoff odds simulation and projected bracket ──────────
    settings      = ctx.get("league_settings") or {}
    playoff_teams = int(settings.get("playoff_teams") or 6)

    # Run the Monte Carlo sim to get odds-based seedings (same data as the
    # Playoff Odds tab), sort by first_seed → bye → overall playoff probability
    try:
        from data_building.simulate_playoff_odds import simulate_playoff_odds
        odds_list = simulate_playoff_odds(ctx, platform=platform) or []
    except Exception:
        odds_list = []

    if odds_list:
        odds_list_sorted = sorted(
            odds_list,
            key=lambda o: (
                -o.get("first_seed_pct", 0),
                -o.get("bye_pct", 0),
                -o.get("playoff_pct", 0),
                -o.get("avg_final_wins", 0),
            ),
        )
        seeded_rids = [str(o["roster_id"]) for o in odds_list_sorted[:playoff_teams]]
    else:
        # Fallback: use dynasty value order
        name_to_rid = {v: k for k, v in roster_map.items()}
        seeded_rids = [
            str(name_to_rid[r["name"]])
            for r in team_rows
            if r["name"] in name_to_rid
        ][:playoff_teams]

    seed_map_override: dict = {rid: i + 1 for i, rid in enumerate(seeded_rids)}

    def _rid(seed: int):
        """Return int roster_id for 1-based seed, or None if not enough teams."""
        if seed <= len(seeded_rids):
            v = seeded_rids[seed - 1]
            try:
                return int(v)
            except Exception:
                return v
        return None

    # Build a standard projected bracket for the configured playoff size
    # Seeding: 1,2 get byes; R1 pairings are 3v(N), 4v(N-1), …
    synthetic_bracket: list[dict] = []
    match_id = 1

    if playoff_teams == 6:
        # R1: 3v6, 4v5 → R2 (Semis): 1 vs W(4v5), 2 vs W(3v6) → R3 Finals
        synthetic_bracket = [
            {"m": 1, "r": 1, "t1": _rid(3), "t2": _rid(6), "t1_from": None, "t2_from": None, "w": {"m": 3}},
            {"m": 2, "r": 1, "t1": _rid(4), "t2": _rid(5), "t1_from": None, "t2_from": None, "w": {"m": 4}},
            {"m": 3, "r": 2, "t1": _rid(1), "t2": None,    "t1_from": None, "t2_from": {"w": 2}, "w": {"m": 5}},
            {"m": 4, "r": 2, "t1": _rid(2), "t2": None,    "t1_from": None, "t2_from": {"w": 1}, "w": {"m": 5}},
            {"m": 5, "r": 3, "t1": None,    "t2": None,    "t1_from": {"w": 3}, "t2_from": {"w": 4}},
        ]
    elif playoff_teams == 4:
        # R1 (Semis): 1v4, 2v3 → Finals
        synthetic_bracket = [
            {"m": 1, "r": 1, "t1": _rid(1), "t2": _rid(4), "t1_from": None, "t2_from": None, "w": {"m": 3}},
            {"m": 2, "r": 1, "t1": _rid(2), "t2": _rid(3), "t1_from": None, "t2_from": None, "w": {"m": 3}},
            {"m": 3, "r": 2, "t1": None,    "t2": None,    "t1_from": {"w": 1}, "t2_from": {"w": 2}},
        ]
    elif playoff_teams == 8:
        # R1: 1v8, 4v5 (top half) + 2v7, 3v6 (bottom half) → Semis → Finals
        synthetic_bracket = [
            {"m": 1, "r": 1, "t1": _rid(1), "t2": _rid(8), "t1_from": None, "t2_from": None, "w": {"m": 5}},
            {"m": 2, "r": 1, "t1": _rid(4), "t2": _rid(5), "t1_from": None, "t2_from": None, "w": {"m": 5}},
            {"m": 3, "r": 1, "t1": _rid(2), "t2": _rid(7), "t1_from": None, "t2_from": None, "w": {"m": 6}},
            {"m": 4, "r": 1, "t1": _rid(3), "t2": _rid(6), "t1_from": None, "t2_from": None, "w": {"m": 6}},
            {"m": 5, "r": 2, "t1": None,    "t2": None,    "t1_from": {"w": 1}, "t2_from": {"w": 2}, "w": {"m": 7}},
            {"m": 6, "r": 2, "t1": None,    "t2": None,    "t1_from": {"w": 3}, "t2_from": {"w": 4}, "w": {"m": 7}},
            {"m": 7, "r": 3, "t1": None,    "t2": None,    "t1_from": {"w": 5}, "t2_from": {"w": 6}},
        ]
    else:
        # Generic: no bracket available
        synthetic_bracket = []

    power_playoffs_html = render_power_and_playoffs(
        synthetic_ts, roster_map, league_id_str, platform, season,
        bracket_override=synthetic_bracket,
        seed_map_override=seed_map_override,
    )

    # ── left-column dynasty rankings table ───────────────────────────────────
    table_rows_html = ""
    for i, row in enumerate(team_rows, 1):
        av = row["avatar"]
        img = (
            f"<img class='avatar sm' src='{av}' onerror=\"this.style.display='none'\">"
            if av else ""
        )
        first_rd = sum(
            1 for pk in (picks_by_roster.get(row["rid"], []) if isinstance(picks_by_roster, dict) else [])
            if int(pk.get("round") or 0) == 1
        )
        picks_label = f"{first_rd} 1st" if first_rd else f"{row['n_picks']} picks" if row["n_picks"] else "—"
        table_rows_html += (
            f"<tr>"
            f"<td class='num'>{i}</td>"
            f"<td class='team'>{img} {row['name']}</td>"
            f"<td class='num'>{row['total']:.0f}</td>"
            f"<td class='num'>{row['player_v']:.0f}</td>"
            f"<td class='num'>{picks_label}</td>"
            f"<td class='num'>{row['value_pct']:.1f}%</td>"
            f"<td class='num'>{row['prod_pct']:.1f}%</td>"
            f"</tr>"
        )

    table_html = f"""
        <div class="table-wrap">
          <table class="standings-table dynasty-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Team</th>
                <th>Value</th>
                <th>Players</th>
                <th>Draft Capital</th>
                <th>Val%</th>
                <th>Proj%</th>
              </tr>
            </thead>
            <tbody>{table_rows_html}</tbody>
          </table>
        </div>
        <div class="footer">Dynasty value · players + draft picks · no games played yet</div>
    """

    return f"""
    <div class="standings-main two-col-standings">
      <div class="standings-col">
        <div class="card">
          <div class="card-tabs">
            <div class="tab-strip">
              <button class="tab-btn active" data-tab="standings">Value Rankings</button>
              <div class="tab-panels">
                <div class="tab-panel active" data-tab="standings">
                  {table_html}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="standings-col">
        {power_playoffs_html}
      </div>
    </div>
    """


def render_share_rankings(ctx: dict) -> str:
    """
    Table showing each team's value share and production share within the league.
    Value share  = roster value / total league value (dynasty player_values).
    Production share = team PF / total league PF.
    """
    team_stats = ctx.get("team_stats")
    rosters    = ctx.get("rosters") or []
    roster_map = ctx.get("roster_map") or {}
    df_weekly  = ctx.get("df_weekly")

    if team_stats is None or team_stats.empty or not rosters:
        return '<div class="otc-movers-empty">Not enough data to compute shares.</div>'

    # ── owner → roster_id map ─────────────────────────────────────
    if df_weekly is not None and not df_weekly.empty and "roster_id" in df_weekly.columns and "owner" in df_weekly.columns:
        owner_to_rid = (df_weekly[["owner", "roster_id"]]
                        .drop_duplicates("owner")
                        .set_index("owner")["roster_id"]
                        .astype(str).to_dict())
    else:
        owner_to_rid = {v: k for k, v in roster_map.items()}

    # ── Value: load from player_values DB ─────────────────────────
    roster_positions = ctx.get("roster_positions") or []
    is_sf = any(str(p).upper() in {"SUPER_FLEX", "SFLEX"} for p in roster_positions)
    val_col = "value_sf" if is_sf else "value_1qb"

    values_by_pid: Dict[str, float] = {}
    try:
        from dashboard_services.db import get_conn
        with get_conn() as _conn:
            rows = _conn.execute(
                f"SELECT player_id, {val_col} AS v FROM player_values WHERE {val_col} IS NOT NULL"
            ).fetchall()
        for r in rows:
            values_by_pid[str(r["player_id"])] = float(r["v"] or 0)
    except Exception:
        pass

    rid_to_roster = {str(r.get("roster_id")): r for r in rosters}

    # ── Roster value per team ─────────────────────────────────────
    rid_to_value: Dict[str, float] = {}
    for rid, r in rid_to_roster.items():
        pids = [str(p) for p in (r.get("players") or [])]
        rid_to_value[rid] = sum(values_by_pid.get(pid, 0.0) for pid in pids)

    league_value_total = sum(rid_to_value.values()) or 1.0

    # ── Production: actual PF in-season, projected avg offseason ─────────────
    league_pf_total = float(team_stats["PF"].sum()) if "PF" in team_stats.columns else 0.0
    is_offseason    = league_pf_total == 0.0
    prod_label      = "Proj. Production Share" if is_offseason else "Production Share"

    # Offseason: use the same projected-avg pipeline as playoff odds
    rid_to_proj: Dict[str, float] = {}
    if is_offseason:
        try:
            from data_building.simulate_playoff_odds import _estimate_from_rosters
            est_teams = _estimate_from_rosters(ctx)
            for t in est_teams:
                rid_to_proj[str(t["roster_id"])] = float(t.get("avg") or 0.0)
        except Exception:
            pass

    # ── Build rows ────────────────────────────────────────────────────────────
    if is_offseason:
        proj_total = sum(rid_to_proj.values()) or 1.0
    else:
        proj_total = league_pf_total or 1.0

    rows_data = []
    for _, row in team_stats.iterrows():
        owner = row.get("owner", "")
        pf    = float(row.get("PF", 0) or 0)
        rid   = str(owner_to_rid.get(owner, ""))
        rval  = rid_to_value.get(rid, 0.0)

        if is_offseason:
            prod_val = rid_to_proj.get(rid, 0.0)
        else:
            prod_val = pf

        prod_pct  = prod_val / proj_total  * 100
        value_pct = rval     / league_value_total * 100
        rows_data.append({"owner": owner, "prod_val": prod_val, "roster_value": rval,
                          "prod_pct": prod_pct, "value_pct": value_pct})

    rows_data.sort(key=lambda x: -x["value_pct"])
    league_teams = len(rows_data) or 1
    fair_share   = 100.0 / league_teams

    def bar(pct: float) -> str:
        width = min(pct / (fair_share * 2) * 100, 100)
        color = "var(--accent)" if pct > fair_share else "var(--text-muted)"
        return (f'<div style="height:6px;border-radius:3px;background:var(--border);flex:1;">'
                f'<div style="height:100%;width:{width:.1f}%;border-radius:3px;background:{color};"></div></div>')

    rows_html = ""
    for i, d in enumerate(rows_data):
        rows_html += f"""
        <tr>
          <td style="width:24px;color:var(--text-muted);font-size:11px;text-align:center;">{i+1}</td>
          <td style="font-weight:600;font-size:13px;">{d['owner']}</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px;">
              {bar(d['value_pct'])}
              <span style="font-size:12px;font-weight:700;min-width:38px;text-align:right;">{d['value_pct']:.1f}%</span>
            </div>
          </td>
          <td>
            <div style="display:flex;align-items:center;gap:8px;">
              {bar(d['prod_pct'])}
              <span style="font-size:12px;font-weight:700;min-width:38px;text-align:right;">{d['prod_pct']:.1f}%</span>
            </div>
          </td>
        </tr>"""

    proj_note = " · projected from roster" if is_offseason else ""
    fair_pct  = f"{fair_share:.1f}%"
    return f"""
    <div style="padding:4px 0 8px;font-size:11px;color:var(--text-muted);">
      Fair share per team: {fair_pct}{proj_note} &nbsp;·&nbsp; bar fills to 2× fair share
    </div>
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;">
          <th style="width:24px;"></th>
          <th style="text-align:left;padding:0 8px 6px 0;">Team</th>
          <th style="text-align:left;padding:0 8px 6px 0;min-width:160px;">Value Share</th>
          <th style="text-align:left;padding:0 0 6px 0;min-width:160px;">{prod_label}</th>
        </tr>
      </thead>
      <tbody style="border-top:1px solid var(--border);">
        {rows_html}
      </tbody>
    </table>"""


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
    share_html = render_share_rankings(ctx)
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
          </div>
        </div>
      </div>
      <div class="standings-col">
        {power_playoffs_html}
      </div>
    </div>
    <div class="card standings-shares-card">
      <h3 class="standings-shares-title">Value &amp; Production Share</h3>
      {share_html}
    </div>
    <aside class="overview-sidebar">
      {sidebar_html}
    </aside>
    """

    return body


def build_draft_assistant_html(ctx: dict) -> str:
    """Rookie Draft Assistant section for the offseason hub."""
    import json as _json_da

    viewer = ctx.get("viewer") or {}
    viewer_roster_id = viewer.get("viewer_roster_id")
    roster_positions = ctx.get("roster_positions") or []
    is_sf = any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in roster_positions)
    league_type = "sf" if is_sf else "1qb"
    league_size = max(6, min(14, int(ctx.get("total_rosters") or 10)))
    current_year = int(ctx.get("season") or datetime.now().year)

    needs: dict = {}
    if viewer_roster_id:
        rosters = ctx.get("rosters") or []
        roster = next((r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)), None)
        if roster:
            players_index = ctx.get("players_index") or {}
            value_table = ctx.get("model_value_table") or []
            val_field = "sf_value" if is_sf else "value"
            values_by_id = {str(row["id"]): row for row in value_table if isinstance(row, dict) and row.get("id")}

            pos_counts: dict = {}
            pos_values: dict = {}
            for pid in (roster.get("players") or []):
                meta = players_index.get(str(pid)) or {}
                pos = str(meta.get("pos") or "").upper()
                if pos not in ("QB", "RB", "WR", "TE"):
                    continue
                vrow = values_by_id.get(str(pid)) or {}
                val = float(vrow.get(val_field) or vrow.get("value") or 0)
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
                pos_values[pos] = pos_values.get(pos, 0.0) + val

            thresholds = {
                "QB":  [(-2, 600), (-1, 400), (0, 200), (1, 50)],
                "RB":  [(-2, 1500), (-1, 1000), (0, 600), (1, 250)],
                "WR":  [(-2, 1500), (-1, 1000), (0, 600), (1, 250)],
                "TE":  [(-2, 600), (-1, 400), (0, 150), (1, 50)],
            }
            for pos in ("QB", "RB", "WR", "TE"):
                val = pos_values.get(pos, 0.0)
                needs[f"{pos}_count"] = pos_counts.get(pos, 0)
                needs[f"{pos}_value"] = round(val, 1)
                need_level = 2
                for level, cutoff in thresholds.get(pos, []):
                    if val >= cutoff:
                        need_level = level
                        break
                needs[pos] = need_level

    needs_json = _json_da.dumps(needs)

    return f"""
    <section class="os-card os-col-fill" id="draftAssistantCard"
             data-league-type="{league_type}"
             data-league-size="{league_size}"
             data-needs='{needs_json}'
             data-year="{current_year}">
      <div class="os-section-head">
        <div class="os-section-head-content">
          <h2 class="os-section-title">Rookie Draft Assistant</h2>
          <div class="os-section-subtitle">Personalized pick recommendations based on your roster</div>
        </div>
        <div class="os-section-head-actions">
          <button type="button" class="da-reset-btn" onclick="daReset()">Reset Board</button>
          <button type="button" class="card-collapse-toggle" data-target="draft-assistant-body">▼</button>
        </div>
      </div>
      <div class="card-collapsible-body" id="draft-assistant-body">
        <div class="da-toolbar">
          <button class="da-filter active" data-pos="ALL" onclick="daFilterPos('ALL')">All</button>
          <button class="da-filter" data-pos="QB" onclick="daFilterPos('QB')">QB</button>
          <button class="da-filter" data-pos="RB" onclick="daFilterPos('RB')">RB</button>
          <button class="da-filter" data-pos="WR" onclick="daFilterPos('WR')">WR</button>
          <button class="da-filter" data-pos="TE" onclick="daFilterPos('TE')">TE</button>
        </div>
        <div class="da-layout">
          <div class="da-board">
            <div class="da-board-header">
              <span>Prospect</span><span>Pos</span><span></span><span class="da-col-right">Value</span><span></span>
            </div>
            <div class="da-board-list" id="daBoardList">
              <div class="da-loading">
                <div class="loading-spinner" style="width:24px;height:24px;flex-shrink:0;"></div>
                <span>Loading prospects…</span>
              </div>
            </div>
          </div>
          <aside class="da-needs" id="daNeedsPanel">
          </aside>
        </div>
      </div>
    </section>
    """


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

        waiver_html.append(
            f"""
            <div class="os-waiver-row">
              <div class="os-waiver-main">
                <div class="os-waiver-name-row">
                  <span class="os-waiver-name player-clickable" style="cursor:pointer;font-weight:600;" data-player-id='{p['player_id']}' data-player-name='{p['name']}'>{p['name']}</span>
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


# Depth cap: each additional asset beyond the anchor is worth less (position 0 = anchor)
_DEPTH_MULTS = [1.0, 0.85, 0.72, 0.60, 0.50, 0.42]

# Tier cap: upper bound on how much a non-anchor asset can contribute, by individual tier.
# Spread linearly from 1.0 (T1, no cap) down to 0.38 (T9, fringe), extended for extra tiers.
def _build_tier_caps(num_tiers: int) -> dict:
    high, low = 1.0, 0.38
    if num_tiers <= 1:
        return {1: 1.0}
    return {t: round(high - (high - low) * (t - 1) / (num_tiers - 1), 3)
            for t in range(1, num_tiers + 1)}

_NUM_TIERS = 9
_TIER_CAPS  = _build_tier_caps(_NUM_TIERS)
_FALLBACK_THRESHOLDS = [850.0, 700.0, 550.0, 420.0, 300.0, 200.0, 120.0, 60.0]


def _market_scale(fmt: str = "1qb") -> float:
    """Return the normalization scale factor written by market_calibration."""
    try:
        from utils.paths import DATA_DIR
        import json as _json
        p = DATA_DIR / "market_calibration_scale.json"
        if p.exists():
            return float(_json.loads(p.read_text()).get(f"scale_{fmt}", 1.0))
    except Exception:
        pass
    return 1.0


def compute_tier_thresholds(value_table, league_type: str = "1qb", league_size: int = 10,
                             num_tiers: int = _NUM_TIERS) -> list:
    """
    Quantile-based tier boundaries with geometric growth and local gap snapping.

    Each tier is ~TIER_GROWTH_RATIO× larger than the previous, producing a
    pyramid: T1 is the smallest elite bucket; T9 (everything below the last
    boundary) is the large catch-all. COVERAGE controls what fraction of active
    players live in T1-T8; the rest fall into T9.

    Each boundary is placed at its target quantile position then snapped to
    the largest nearby gap within ±SNAP_WINDOW players, so boundaries prefer
    natural value breaks when they exist close to the target rank.

    Python _asset_tier and JS prGetTier both clamp at _NUM_TIERS (9).
    """
    TIER_GROWTH_RATIO = 1.6   # each tier is ~1.6× larger than the previous one
    COVERAGE          = 0.50  # fraction of active players in T1-T(num_tiers-1)
    SNAP_WINDOW       = 5     # snap each boundary to nearby gap (±this many players)

    if league_type == "sf":
        primary = "sf_value" if league_size == 10 else f"sf_value_{league_size}"
    else:
        primary = "value" if league_size == 10 else f"value_{league_size}"

    vals = []
    for p in (value_table or []):
        if not isinstance(p, dict):
            continue
        pos = (p.get("position") or "").upper()
        if pos in ("K", "DEF"):
            continue
        v = float(p.get(primary) or p.get("value") or 0)
        if v >= 5:
            vals.append(v)

    vals.sort(reverse=True)
    n = len(vals)

    if n < num_tiers * 2:
        return _FALLBACK_THRESHOLDS

    # Compute target tier sizes: geometric series scaled to cover COVERAGE of players
    n_bounds = num_tiers - 1
    raw = [TIER_GROWTH_RATIO ** i for i in range(n_bounds)]
    scale = (COVERAGE * n) / sum(raw)
    target_sizes = [max(2, round(r * scale)) for r in raw]

    # Gap sizes between adjacent players (for snapping)
    gaps = [vals[i] - vals[i + 1] for i in range(n - 1)]

    start = 0
    used  = set()
    boundaries = []
    for size in target_sizes:
        target_pos = min(start + size - 1, n - 2)
        lo = max(start, target_pos - SNAP_WINDOW)
        hi = min(n - 2, target_pos + SNAP_WINDOW)
        best_gap, best_pos = -1.0, target_pos
        for p in range(lo, hi + 1):
            if p not in used and gaps[p] > best_gap:
                best_gap, best_pos = gaps[p], p
        used.add(best_pos)
        boundaries.append(round((vals[best_pos] + vals[best_pos + 1]) / 2.0, 1))
        start = best_pos + 1

    thresholds = sorted(boundaries, reverse=True) if boundaries else _FALLBACK_THRESHOLDS

    # Enforce minimum tier size of 4.  Merge DOWN: a too-small group absorbs
    # into the tier below it (remove its lower boundary) so that, e.g., a
    # lone player at the top of what would be T2 stays at the head of T2
    # rather than being pulled up into T1.  Exception: if the small group is
    # the topmost one (value >= first threshold), merge it UP instead.
    MIN_TIER_SIZE = 4
    changed = True
    while changed:
        changed = False
        # Check the T1 group (players above the first threshold)
        if thresholds and sum(1 for v in vals if v >= thresholds[0]) < MIN_TIER_SIZE:
            thresholds = thresholds[1:]  # remove upper boundary → T1 merges down into T2
            changed = True
            continue
        # Check each intermediate group (between consecutive boundaries)
        for i in range(len(thresholds)):
            lo = thresholds[i + 1] if i + 1 < len(thresholds) else 0.0
            hi = thresholds[i]
            count = sum(1 for v in vals if lo <= v < hi)
            if count < MIN_TIER_SIZE:
                if i + 1 < len(thresholds):
                    # Merge DOWN: remove lower boundary so this group joins the tier below
                    thresholds = thresholds[:i + 1] + thresholds[i + 2:]
                else:
                    # Last intermediate group — merge UP instead
                    thresholds = thresholds[:i] + thresholds[i + 1:]
                changed = True
                break

    return thresholds if thresholds else _FALLBACK_THRESHOLDS

def _asset_tier(value: float, thresholds: list = None) -> int:
    t = thresholds if thresholds is not None else _FALLBACK_THRESHOLDS
    for i, threshold in enumerate(t):
        if value >= threshold:
            return min(i + 1, _NUM_TIERS)
    return _NUM_TIERS  # catch-all T9


def apply_tier_stack_adjustment(side_a: dict, side_b: dict,
                                 tier_thresholds: list = None,
                                 is_sf: bool = False) -> None:
    """
    Tier-aware trade evaluation.

    Each side's highest-value asset (the "anchor") counts at full face value.
    Every additional asset is worth min(depth_mult, tier_cap) × its face value,
    where tier_cap is driven by that player's individual tier derived from the
    live value-table gaps. Stacking lower-tier players against an elite anchor
    produces compounding discounts; adding a quality T2 player is barely penalised.

    In SF leagues, QBs fill up to 3 starting slots (QB / SFLEX / FLEX), so
    each QB in a package uses its own QB-specific depth index rather than its
    global value-rank index — preventing the second and third QBs from being
    over-penalised relative to their actual roster utility.
    """
    thresholds = tier_thresholds if tier_thresholds is not None else _FALLBACK_THRESHOLDS
    tier_caps  = _build_tier_caps(len(thresholds) + 1)
    # In SF, QBs have 3 starting slots before the depth penalty steepens
    qb_starter_slots = 3 if is_sf else 1

    def _compute_side(side):
        bd = side.get("breakdown") or []
        if bd:
            items = sorted(
                [(item.get("value", 0.0), str(item.get("position") or "").upper()) for item in bd],
                reverse=True,
            )
        else:
            # Fallback: no position info available
            raw_vals = sorted(list(side.get("player_values", []) or []), reverse=True)
            raw_picks = float(side.get("raw_picks_total", 0.0) or 0.0)
            if raw_picks > 0.0:
                raw_vals.append(raw_picks)
                raw_vals.sort(reverse=True)
            items = [(v, "") for v in raw_vals]

        if not items:
            return float(side.get("raw_total", 0.0) or 0.0), 0.0

        effective  = 0.0
        global_idx = 0
        qb_idx     = 0  # separate counter so SF QBs use their own depth slot

        for v, pos in items:
            if global_idx == 0:
                effective += v  # anchor always full value
            elif is_sf and pos == "QB" and qb_idx < qb_starter_slots:
                # QB still within SF starting depth — use QB-specific depth index
                depth_m = _DEPTH_MULTS[min(qb_idx, len(_DEPTH_MULTS) - 1)]
                tier_m  = tier_caps.get(_asset_tier(v, thresholds), 0.38)
                effective += v * min(depth_m, tier_m)
            else:
                depth_m = _DEPTH_MULTS[min(global_idx, len(_DEPTH_MULTS) - 1)]
                tier_m  = tier_caps.get(_asset_tier(v, thresholds), 0.38)
                effective += v * min(depth_m, tier_m)

            if pos == "QB":
                qb_idx += 1
            global_idx += 1

        return effective, effective - float(side.get("raw_total", 0.0) or 0.0)

    eff_a, adj_a = _compute_side(side_a)
    eff_b, adj_b = _compute_side(side_b)

    side_a["effective_total"] = eff_a
    side_b["effective_total"] = eff_b
    side_a["adjustment"] = adj_a
    side_b["adjustment"] = adj_b

    # Annotate each breakdown item with its individual tier + context-aware effective value
    for side in (side_a, side_b):
        bd = side.get("breakdown") or []
        if not bd:
            continue
        sorted_bd = sorted(bd, key=lambda x: x.get("value", 0.0), reverse=True)
        qb_idx = 0
        for idx, item in enumerate(sorted_bd):
            val  = item.get("value", 0.0)
            pos  = str(item.get("position") or "").upper()
            tier = _asset_tier(val, thresholds)
            if idx == 0:
                m = 1.0
            elif is_sf and pos == "QB" and qb_idx < qb_starter_slots:
                m = min(_DEPTH_MULTS[min(qb_idx, len(_DEPTH_MULTS) - 1)], tier_caps.get(tier, 0.38))
            else:
                m = min(_DEPTH_MULTS[min(idx, len(_DEPTH_MULTS) - 1)], tier_caps.get(tier, 0.38))
            if pos == "QB":
                qb_idx += 1
            item["tier"]            = tier
            item["stack_mult"]      = round(m, 3)
            item["effective_value"] = round(val * m, 1)


apply_multi_for_one_adjustment = apply_tier_stack_adjustment


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
        _chart_data_attr  = html.escape(_chart_data,   quote=True)
        _chart_layout_attr = html.escape(_chart_layout, quote=True)
        _chart_html = (
            f'<div id="{_chart_div_id}" class="team-value-chart team-chart-lazy"'
            f' data-chart="{_chart_data_attr}"'
            f' data-layout="{_chart_layout_attr}">'
            f'<div class="team-chart-skeleton"></div>'
            f'</div>'
        )

        _gdata = team_grades.get(rid, {})
        _grade = _gdata.get("grade", "?")
        _win_window = _gdata.get("win_window", "")
        _grade_cls = "grade-a" if _grade.startswith("A") else "grade-b" if _grade.startswith("B") else "grade-c" if _grade.startswith("C") else "grade-d"
        _grade_badge = f"<span class='roster-grade-inline {_grade_cls}' title='{_win_window}'>{_grade}</span>"

        # Numeric sort keys for client-side sorting
        _grade_num = {"A+":12,"A":11,"A-":10,"B+":9,"B":8,"B-":7,"C+":6,"C":5,"C-":4,"D+":3,"D":2,"D-":1,"F":0}.get(_grade, 0)
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
        </div>
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
                  ? '<span class="adp-value">+Steal (' + diff.toFixed(1) + ' vs ADP)</span>'
                  : diff < -1
                    ? '<span class="adp-reach">Reach (' + diff.toFixed(1) + ' vs ADP)</span>'
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
            var gradeA = Number(a.dataset.sortGrade) || 0;
            var gradeB = Number(b.dataset.sortGrade) || 0;
            // Higher grade numbers should come first (A+ = 10, A = 9, etc.)
            return gradeB - gradeA;
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

      // Lazy-render Plotly charts as they scroll into view
      (function() {{
        function renderChart(el) {{
          if (el.dataset.rendered) return;
          el.dataset.rendered = '1';
          try {{
            var trace  = JSON.parse(el.getAttribute('data-chart'));
            var layout = JSON.parse(el.getAttribute('data-layout'));
            el.innerHTML = '';
            Plotly.newPlot(el.id, trace, layout, {{responsive: true, displayModeBar: false}});
          }} catch(e) {{}}
        }}
        function tryRender(el) {{
          if (typeof Plotly !== 'undefined') {{
            renderChart(el);
          }} else {{
            var t = setInterval(function() {{
              if (typeof Plotly !== 'undefined') {{ clearInterval(t); renderChart(el); }}
            }}, 80);
          }}
        }}
        var charts = document.querySelectorAll('.team-chart-lazy');
        if ('IntersectionObserver' in window) {{
          var obs = new IntersectionObserver(function(entries) {{
            entries.forEach(function(e) {{
              if (e.isIntersecting) {{ tryRender(e.target); obs.unobserve(e.target); }}
            }});
          }}, {{ rootMargin: '300px' }});
          charts.forEach(function(el) {{ obs.observe(el); }});
        }} else {{
          charts.forEach(tryRender);
        }}
      }})();
    }})();
    </script>
    """












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
    if entry and (time.time() - entry.get("ts", 0) <= CACHE_TTL):
        ctx = entry["ctx"]
        ctx["viewer"] = get_viewer_session()
        return ctx

    with _CTX_LOCKS_LOCK:
        if key not in _CTX_LOCKS:
            _CTX_LOCKS[key] = threading.Lock()
        key_lock = _CTX_LOCKS[key]

    with key_lock:
        # Re-check after acquiring lock — another thread may have built it while we waited
        entry = DASHBOARD_CACHE.get(key)
        if entry and (time.time() - entry.get("ts", 0) <= CACHE_TTL):
            ctx = entry["ctx"]
            ctx["viewer"] = get_viewer_session()
            return ctx
        ctx = build_league_context(platform, league_id, season)
        DASHBOARD_CACHE[key] = {"ctx": ctx, "ts": time.time(), "page_html": {}}
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
        body = _build_offseason_standings_body(ctx)
    else:
        body = build_standings_body(ctx)

    return render_page("BR Fantasy Standings", league_id, "standings", body, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/waivers")
def page_waivers(platform: str, season: int, league_id: str):
    ctx = get_league_ctx_from_cache(platform, league_id, season)
    body = build_waivers_body(platform, season, league_id, ctx)
    return render_page("BR Fantasy Waivers", league_id, "waivers", body, platform, season)


@app.route("/api/waiver-candidates")
def api_waiver_candidates():
    """
    Returns scored waiver wire candidates for a league.
    Query params: platform, league_id, season, position (optional filter)
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    season = int(request.args.get("season") or datetime.now().year)
    position_filter = (request.args.get("position") or "").strip().upper()

    if not league_id:
        return jsonify({"error": "league_id required"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    rosters = ctx.get("rosters") or []
    rostered_ids = {
        str(pid)
        for r in rosters
        for pid in (r.get("players") or [])
    }

    players_index = ctx.get("players_index") or {}
    model_value_table = load_model_value_table()

    _rp_wv = ctx.get("roster_positions") or []
    _is_sf_wv = any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in _rp_wv)
    _vf_wv = "sf_value" if _is_sf_wv else "value"

    candidates = []
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
            val = float(row.get(_vf_wv) or row.get("value") or 0.0)
        except Exception:
            val = 0.0
        if val <= 0:
            continue
        try:
            age = float(row.get("age") or 0)
        except Exception:
            age = 0.0
        player_name = (
            row.get("name")
            or players_index.get(pid, {}).get("name")
            or f"Player {pid}"
        )
        candidates.append({
            "player_id": pid,
            "name": player_name,
            "position": pos,
            "team": row.get("team") or players_index.get(pid, {}).get("team") or "",
            "value": val,
            "age": age,
            "pos_rank_label": row.get("pos_rank_label") or "",
            "rank_change_7d": row.get("rank_change_7d"),
        })

    waiver_breakout: dict = {}
    try:
        _db_url = os.getenv("DATABASE_URL", "").strip()
        if _db_url and not any(t in _db_url for t in ("USER", "PASSWORD", "HOST")):
            from dashboard_services.db import get_conn as _gc
            _pids = [c["player_id"] for c in candidates[:100]]
            if _pids:
                with _gc() as _conn:
                    with _conn.cursor() as _cur:
                        _cur.execute(
                            """
                            SELECT DISTINCT ON (player_id)
                                player_id, breakout_opportunity_score
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

    _prime_max = {"QB": 33, "RB": 26, "WR": 28, "TE": 29}

    def _wv_score(c: dict) -> float:
        val = c["value"]
        age = c["age"] or 0
        pos = c["position"]
        rank_chg = c["rank_change_7d"] or 0
        bscore = waiver_breakout.get(c["player_id"], 0)
        prime = _prime_max.get(pos, 28)
        trend_bonus = min(rank_chg * 4, 60) if rank_chg and rank_chg > 0 else 0
        breakout_bonus = min(bscore * 0.5, 50)
        age_bonus = 30 - max(0, (age - prime) * 10) if age else 0
        return val + trend_bonus + breakout_bonus + age_bonus

    def _wv_signal(c: dict) -> tuple:
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

    candidates.sort(key=_wv_score, reverse=True)
    if position_filter and position_filter in {"QB", "RB", "WR", "TE"}:
        candidates = [c for c in candidates if c["position"] == position_filter]

    result = []
    for c in candidates[:30]:
        sig_cls, sig_label = _wv_signal(c)
        bscore = waiver_breakout.get(c["player_id"], 0.0)
        result.append({
            "player_id": c["player_id"],
            "name": c["name"],
            "position": c["position"],
            "team": c["team"],
            "value": c["value"],
            "age": c["age"],
            "pos_rank_label": c["pos_rank_label"],
            "rank_change_7d": c["rank_change_7d"],
            "breakout_score": bscore,
            "signal": sig_label,
            "signal_class": sig_cls,
            "composite_score": _wv_score(c),
        })

    return jsonify({"candidates": result, "total": len(result)})


@app.route("/api/start-sit-options")
def api_start_sit_options():
    """
    Returns roster options grouped by position for the viewing user.
    Query params: platform, league_id, season
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    season = int(request.args.get("season") or datetime.now().year)

    if not league_id:
        return jsonify({"error": "league_id required"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    viewer = ctx.get("viewer") or {}
    viewer_roster_id = viewer.get("viewer_roster_id")

    if not viewer_roster_id:
        return jsonify({"positions": {}})

    rosters = ctx.get("rosters") or []
    viewer_roster = next(
        (r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)),
        None,
    )
    if not viewer_roster:
        return jsonify({"positions": {}})

    player_ids = [str(pid) for pid in (viewer_roster.get("players") or [])]
    players_index = ctx.get("players_index") or {}
    players_full = ctx.get("players") or {}
    model_value_table = load_model_value_table()

    lineup_requirements: dict = ctx.get("lineup_requirements") or {}
    if not lineup_requirements:
        roster_positions = ctx.get("roster_positions") or []
        for slot in roster_positions:
            s = str(slot).upper()
            if s in {"QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX", "SFLEX", "K", "DEF"}:
                lineup_requirements[s] = lineup_requirements.get(s, 0) + 1

    _rp_ss = ctx.get("roster_positions") or []
    _is_sf_ss = any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in _rp_ss)
    _vf_ss = "sf_value" if _is_sf_ss else "value"

    rows_by_id: dict = {}
    for row in model_value_table:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or "")
        if pid:
            rows_by_id[pid] = row

    current_week = int(ctx.get("current_week") or 0)
    def_rush_allowed: dict = {}
    def_recv_allowed: dict = {}
    try:
        from utils.utils import load_week_stats as _lws
        check_weeks = [w for w in range(max(1, current_week - 4), current_week) if w > 0]
        for chk_week in check_weeks:
            wstat = _lws(season, chk_week) or {}
            for team, pos_data in wstat.items():
                if not isinstance(pos_data, dict):
                    continue
                rb_rush = sum(
                    float(p.get("rush_yds") or p.get("rushing_yds") or 0)
                    for p in (pos_data.get("RB") or {}).values()
                    if isinstance(p, dict)
                )
                wr_recv = sum(
                    float(p.get("rec_yds") or p.get("receiving_yds") or 0)
                    for p in (pos_data.get("WR") or {}).values()
                    if isinstance(p, dict)
                )
                te_recv = sum(
                    float(p.get("rec_yds") or p.get("receiving_yds") or 0)
                    for p in (pos_data.get("TE") or {}).values()
                    if isinstance(p, dict)
                )
                recv_allowed = wr_recv + te_recv
                if rb_rush > 0:
                    def_rush_allowed.setdefault(team, []).append(rb_rush)
                if recv_allowed > 0:
                    def_recv_allowed.setdefault(team, []).append(recv_allowed)
    except Exception:
        pass

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    all_rush_avgs = {t: _avg(v) for t, v in def_rush_allowed.items()}
    all_recv_avgs = {t: _avg(v) for t, v in def_recv_allowed.items()}

    def _matchup_adj(opponent_team: str, pos: str) -> float:
        if not opponent_team:
            return 1.0
        avgs = all_rush_avgs if pos == "RB" else all_recv_avgs
        if not avgs:
            return 1.0
        sorted_teams = sorted(avgs.keys(), key=lambda t: avgs[t])
        n = len(sorted_teams)
        try:
            rank = sorted_teams.index(opponent_team)
        except ValueError:
            return 1.0
        return 0.85 + 0.30 * (rank / max(n - 1, 1))

    opponent_map: dict = {}
    opp_label_map: dict = {}
    try:
        from utils.utils import load_week_sched as _lsched
        if current_week > 0:
            sched = _lsched(season, current_week) or []
            for game in sched:
                home = str(game.get("home") or "").upper()
                away = str(game.get("away") or "").upper()
                if home and away:
                    opponent_map[home] = away
                    opponent_map[away] = home
                    opp_label_map[home] = f"vs {away}"
                    opp_label_map[away] = f"@ {home}"
    except Exception:
        pass

    recent_pts: dict = {}
    try:
        import glob as _glob
        _stat_pattern = os.path.join("cache", "sleeper_stats", f"sleeper_stats_{season}_week_*.json")
        _week_files = sorted(_glob.glob(_stat_pattern))
        _recent_files = _week_files[-4:] if len(_week_files) >= 4 else _week_files
        _pts_by_player: dict = {}
        for _wf in _recent_files:
            try:
                with open(_wf) as _f:
                    _wdata = json.load(_f)
                for pid, stats in _wdata.items():
                    if not isinstance(stats, dict):
                        continue
                    pts = float(stats.get("pts_half_ppr") or stats.get("pts_ppr") or 0)
                    if pts > 0:
                        _pts_by_player.setdefault(pid, []).append(pts)
            except Exception:
                continue
        for pid, vals in _pts_by_player.items():
            recent_pts[pid] = _avg(vals)
    except Exception:
        pass

    positions_out: dict = {"QB": [], "RB": [], "WR": [], "TE": []}
    for pid in player_ids:
        row = rows_by_id.get(pid) or {}
        meta = players_index.get(pid) or {}
        pos = str(row.get("position") or row.get("pos") or meta.get("pos") or "").upper()
        if pos not in positions_out:
            continue
        player_name = row.get("name") or meta.get("name") or f"Player {pid}"
        team = (row.get("team") or meta.get("team") or "").upper()
        opponent = opponent_map.get(team, "")
        opp_label = opp_label_map.get(team, "BYE" if current_week > 0 else "")
        on_bye = current_week > 0 and team not in opponent_map
        avg_pts = recent_pts.get(pid, 0.0)
        adj = _matchup_adj(opponent, pos) if not on_bye else 0.5
        score = avg_pts * adj if avg_pts > 0 else float(row.get(_vf_ss) or row.get("value") or 0) * 0.01
        full_player = players_full.get(pid) or {}
        raw_status = str(full_player.get("injury_status") or full_player.get("status") or "").strip()
        injury_status = None if raw_status in {"", "active", "Active", "ACT"} else raw_status
        positions_out[pos].append({
            "player_id": pid,
            "name": player_name,
            "team": team,
            "opponent": opp_label,
            "on_bye": on_bye,
            "avg_pts": round(avg_pts, 1),
            "matchup_adj": round(adj, 2),
            "pos_rank_label": row.get("pos_rank_label") or "",
            "injury_status": injury_status,
            "_score": score,
        })

    flex_slots = lineup_requirements.get("FLEX") or 0
    sflex_slots = (lineup_requirements.get("SUPER_FLEX") or 0) + (lineup_requirements.get("SFLEX") or 0)

    for pos in positions_out:
        positions_out[pos].sort(key=lambda x: (not x["on_bye"], x["_score"]), reverse=True)
        n_start = lineup_requirements.get(pos, 1)
        eligible_for_flex = pos in ("RB", "WR", "TE")
        for i, p in enumerate(positions_out[pos]):
            p["start"] = i < n_start and not p["on_bye"]
            p["flex_start"] = False
            p["flex_eligible"] = eligible_for_flex and i >= n_start and not p["on_bye"]

    if flex_slots:
        all_flex = sorted(
            [p for pos in positions_out for p in positions_out[pos] if p["flex_eligible"]],
            key=lambda x: x["_score"], reverse=True
        )
        for p in all_flex[:flex_slots]:
            p["start"] = True
            p["flex_start"] = True

    if sflex_slots:
        sflex_cands = sorted(
            [p for pos in positions_out for p in positions_out[pos]
             if not p["start"] and not p["on_bye"] and pos in ("QB", "RB", "WR", "TE")],
            key=lambda x: x["_score"], reverse=True
        )
        for p in sflex_cands[:sflex_slots]:
            p["start"] = True
            p["flex_start"] = True

    for pos in positions_out:
        for p in positions_out[pos]:
            del p["_score"]

    return jsonify({
        "positions": positions_out,
        "lineup_requirements": lineup_requirements,
        "flex_slots": flex_slots,
        "sflex_slots": sflex_slots,
        "current_week": current_week,
    })


@app.route("/<platform>/<int:season>/<league_id>/weekly")
def page_weekly(platform: str, season: int, league_id: str):
    ctx = get_league_ctx_from_cache(platform, league_id, season)

    if ctx.get("offseason_mode"):
        body = """
        <div class="card central">
          <div class="card-header"><h2>Weekly Hub Unavailable</h2></div>
          <div class="card-body">
            <p>The Weekly Hub becomes active once the season begins and games are being played.</p>
            <p>Use the Dashboard, Teams, and Trade tools for offseason planning.</p>
          </div>
        </div>
        """
    elif not has_draft_ended(league_id, platform, season):
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




def page_auction_values(platform: str = None, season: int = None, league_id: str = None):
    user_id = session.get("viewer_username")
    has_premium = has_premium_access(user_id, league_id, platform or "sleeper")

    if not has_premium:
        # Show teaser with paywall
        body_html = """
    <div class="card central" style="max-width:900px;">
      <div class="card-header">
        <h2>Startup Auction Values</h2>
        <div style="font-size:14px;color:var(--text-muted);margin-top:4px;">
          Dynasty startup auction dollar values for every player — by league type, size, and budget
        </div>
      </div>
      <div class="card-body" style="text-align:center;padding:60px 24px;">
        <div style="font-size:40px;margin-bottom:16px;opacity:.3;"><i class="fa-solid fa-gavel"></i></div>
        <div style="font-weight:700;font-size:18px;margin-bottom:8px;">Premium Feature</div>
        <div style="color:var(--text-muted);font-size:14px;margin-bottom:24px;">
          Get precise auction dollar values for dynasty startup drafts,<br>
          customizable by league format and budget.
        </div>
        <button onclick="showPaywall('auction-values')"
          style="padding:12px 28px;border-radius:9px;border:none;background:linear-gradient(135deg,#667eea,#764ba2);color:white;font-size:15px;font-weight:700;cursor:pointer;">
          Unlock Auction Values
        </button>
      </div>
    </div>
    <script>
      // Pre-open paywall so user sees it immediately
      document.addEventListener('DOMContentLoaded', function() { showPaywall('auction-values'); });
    </script>
    """
        return render_page("Auction Values", league_id, "auction-values", body_html, platform, season)

    body_html = f"""
    <div class="card central" style="max-width:900px;">
      <div class="card-header" style="border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:0;">
        <h2 style="margin:0 0 4px;">Startup Auction Values</h2>
        <div style="font-size:13px;color:var(--text-muted);">
          Dynasty startup dollar values based on BR model — adjust format and budget below
        </div>
      </div>
      <div class="card-body" style="padding-top:20px;">

        <!-- Settings row -->
        <div style="display:flex;flex-wrap:wrap;gap:20px;align-items:flex-end;margin-bottom:20px;padding:14px 16px;background:var(--bg-alt,#f8fafc);border:1px solid var(--border);border-radius:10px;">
          <div>
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--text-muted);margin-bottom:6px;">League Type</div>
            <div style="display:flex;gap:4px;" id="avLeagueGroup">
              <button class="av-toggle active" data-val="1qb" onclick="avSetLeague('1qb')">1QB</button>
              <button class="av-toggle" data-val="sf" onclick="avSetLeague('sf')">SF</button>
            </div>
          </div>
          <div>
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--text-muted);margin-bottom:6px;">League Size</div>
            <div style="display:flex;gap:4px;" id="avSizeGroup">
              <button class="av-toggle" data-val="8" onclick="avSetSize(8)">8</button>
              <button class="av-toggle active" data-val="10" onclick="avSetSize(10)">10</button>
              <button class="av-toggle" data-val="12" onclick="avSetSize(12)">12</button>
              <button class="av-toggle" data-val="14" onclick="avSetSize(14)">14</button>
            </div>
          </div>
          <div>
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--text-muted);margin-bottom:6px;">Budget / Team ($)</div>
            <input id="avBudget" type="number" min="50" max="1000" step="10" value="200"
              style="width:80px;padding:5px 9px;border-radius:7px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:13px;font-weight:600;"
              oninput="avRender()">
          </div>
          <div style="margin-left:auto;">
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--text-muted);margin-bottom:6px;">Position</div>
            <div style="display:flex;gap:4px;">
              <button class="av-pos active" data-pos="ALL" onclick="avSetPos('ALL')">All</button>
              <button class="av-pos" data-pos="QB" onclick="avSetPos('QB')">QB</button>
              <button class="av-pos" data-pos="RB" onclick="avSetPos('RB')">RB</button>
              <button class="av-pos" data-pos="WR" onclick="avSetPos('WR')">WR</button>
              <button class="av-pos" data-pos="TE" onclick="avSetPos('TE')">TE</button>
            </div>
          </div>
        </div>

        <!-- Summary strip -->
        <div id="avSummary" style="font-size:12px;color:var(--text-muted);margin-bottom:12px;"></div>

        <!-- Table -->
        <div id="avLoading" style="text-align:center;padding:48px 0;color:var(--text-muted);">
          <div class="loading-spinner" style="margin:0 auto 12px;"></div>
          Loading player values…
        </div>
        <div id="avTableWrap" style="display:none;overflow-x:auto;">
          <table id="avTable" style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
              <tr style="border-bottom:2px solid var(--border);text-align:left;">
                <th style="padding:8px 6px;color:var(--text-muted);font-weight:600;cursor:pointer;" onclick="avSort('rank')"># <span id="avSortRank"></span></th>
                <th style="padding:8px 6px;color:var(--text-muted);font-weight:600;">Name</th>
                <th style="padding:8px 6px;color:var(--text-muted);font-weight:600;">Pos</th>
                <th style="padding:8px 6px;color:var(--text-muted);font-weight:600;">Team</th>
                <th style="padding:8px 6px;color:var(--text-muted);font-weight:600;cursor:pointer;" onclick="avSort('age')">Age <span id="avSortAge"></span></th>
                <th style="padding:8px 6px;color:var(--text-muted);font-weight:600;cursor:pointer;" onclick="avSort('value')">Value <span id="avSortValue"></span></th>
                <th style="padding:8px 10px 8px 6px;color:var(--text-muted);font-weight:600;cursor:pointer;" onclick="avSort('auction')">Auction $ <span id="avSortAuction"></span></th>
              </tr>
            </thead>
            <tbody id="avBody"></tbody>
          </table>
        </div>

      </div>
    </div>

    <style>
      .av-toggle, .av-pos {{
        padding:5px 12px;border-radius:7px;border:1px solid var(--border);
        background:var(--card);color:var(--text-muted);cursor:pointer;
        font-size:12px;font-weight:600;transition:all .15s;
      }}
      .av-toggle.active, .av-pos.active {{
        background:var(--text);color:var(--card);border-color:var(--text);
      }}
      #avTable tbody tr:hover {{ background:var(--bg-alt,#f8fafc); }}
      #avTable tbody td {{ padding:8px 6px;border-bottom:1px solid var(--border); }}
      .av-dollar {{ font-weight:800;font-size:15px; }}
      .av-dollar.top {{ color:#10b981; }}
      .av-dollar.mid {{ color:#3b82f6; }}
      .av-dollar.low {{ color:var(--text-muted); }}
      @media (max-width:600px) {{
        .av-col-team, .av-col-value {{ display:none; }}
      }}
    </style>

    <script>
    (function() {{
      const POS_COLORS = {{QB:'#3b82f6',RB:'#22c55e',WR:'#f59e0b',TE:'#8b5cf6'}};
      // Roster spots per team by league size (dynasty startup pool)
      const ROSTER = {{8:25, 10:24, 12:23, 14:22}};

      let allPlayers = [];
      let avLeague = '1qb';
      let avSize = 10;
      let avPos = 'ALL';
      let avSortCol = 'auction';
      let avSortDir = -1; // -1 = desc

      fetch('/api/league-players')
        .then(r => r.json())
        .then(data => {{
          allPlayers = (data.players || []).filter(p =>
            ['QB','RB','WR','TE'].includes((p.position || '').toUpperCase())
          );
          document.getElementById('avLoading').style.display = 'none';
          document.getElementById('avTableWrap').style.display = '';
          avRender();
        }})
        .catch(() => {{
          document.getElementById('avLoading').innerHTML =
            '<div style="color:var(--text-muted)">Could not load player data.</div>';
        }});

      function getVal(p) {{
        const sz = avSize;
        if (avLeague === 'sf') {{
          if (sz === 8)  return p.sf_value_8  || p.sf_value || 0;
          if (sz === 12) return p.sf_value_12 || p.sf_value || 0;
          if (sz === 14) return p.sf_value_14 || p.sf_value || 0;
          return p.sf_value || 0;
        }} else {{
          if (sz === 8)  return p.value_8  || p.value || 0;
          if (sz === 12) return p.value_12 || p.value || 0;
          if (sz === 14) return p.value_14 || p.value || 0;
          return p.value || 0;
        }}
      }}

      window.avRender = function() {{
        const budget = Math.max(50, parseInt(document.getElementById('avBudget').value) || 200);
        const totalBudget = budget * avSize;
        const poolSize = (ROSTER[avSize] || 24) * avSize;

        // Sort all players by value, take top poolSize
        const sorted = [...allPlayers]
          .map(p => ({{ ...p, _val: getVal(p) }}))
          .sort((a, b) => b._val - a._val)
          .slice(0, poolSize);

        const totalVal = sorted.reduce((s, p) => s + p._val, 0);

        // Calculate auction value: each rostered player gets at least $1
        // surplus budget distributed proportionally
        const minPerPlayer = 1;
        const surplusBudget = totalBudget - poolSize * minPerPlayer;
        const sortedWithAuction = sorted.map(p => ({{
          ...p,
          _auction: Math.max(1, Math.round(minPerPlayer + (p._val / totalVal) * surplusBudget)),
        }}));

        // Sort by selected column
        sortedWithAuction.sort((a, b) => {{
          let av, bv;
          if (avSortCol === 'auction') {{ av = a._auction; bv = b._auction; }}
          else if (avSortCol === 'value')  {{ av = a._val;    bv = b._val; }}
          else if (avSortCol === 'age')    {{ av = parseFloat(a.age) || 99; bv = parseFloat(b.age) || 99; }}
          else {{ av = a._auction; bv = b._auction; }} // rank = auction
          return avSortDir * (bv - av);
        }});

        // Apply position filter
        const display = avPos === 'ALL' ? sortedWithAuction
          : sortedWithAuction.filter(p => (p.position || '').toUpperCase() === avPos);

        // Update sort indicators
        ['Rank','Age','Value','Auction'].forEach(c => {{
          const el = document.getElementById('avSort' + c);
          if (el) el.textContent = '';
        }});
        const colKey = avSortCol === 'rank' ? 'Rank' :
                       avSortCol === 'age'  ? 'Age'  :
                       avSortCol === 'value'? 'Value': 'Auction';
        const sortEl = document.getElementById('avSort' + colKey);
        if (sortEl) sortEl.textContent = avSortDir === -1 ? ' ↓' : ' ↑';

        // Summary
        document.getElementById('avSummary').textContent =
          `${{avSize}}-team · ${{avLeague.toUpperCase()}} · ${{poolSize}} players in pool · ${{totalBudget}} total budget`;

        // Render rows
        const body = document.getElementById('avBody');
        body.innerHTML = display.map((p, i) => {{
          const pos = (p.position || '').toUpperCase();
          const col = POS_COLORS[pos] || 'var(--text-muted)';
          const age = p.age ? parseFloat(p.age).toFixed(1) : '—';
          const val = p._val ? p._val.toFixed(1) : '—';
          const auc = p._auction;
          const dollarClass = auc >= 40 ? 'top' : auc >= 10 ? 'mid' : 'low';
          return `<tr>
            <td style="color:var(--text-muted);">${{i + 1}}</td>
            <td style="font-weight:600;">${{p.name || '—'}}</td>
            <td><span style="font-size:11px;font-weight:700;padding:2px 6px;border-radius:4px;background:${{col}}20;color:${{col}};">${{pos}}</span></td>
            <td class="av-col-team" style="color:var(--text-muted);">${{p.team || '—'}}</td>
            <td>${{age}}</td>
            <td class="av-col-value" style="color:var(--text-muted);">${{val}}</td>
            <td style="padding-right:10px;"><span class="av-dollar ${{dollarClass}}">$${{auc}}</span></td>
          </tr>`;
        }}).join('');
      }};

      window.avSetLeague = function(val) {{
        avLeague = val;
        document.querySelectorAll('#avLeagueGroup .av-toggle').forEach(b =>
          b.classList.toggle('active', b.dataset.val === val));
        avRender();
      }};

      window.avSetSize = function(val) {{
        avSize = val;
        document.querySelectorAll('#avSizeGroup .av-toggle').forEach(b =>
          b.classList.toggle('active', b.dataset.val == val));
        avRender();
      }};

      window.avSetPos = function(val) {{
        avPos = val;
        document.querySelectorAll('.av-pos').forEach(b =>
          b.classList.toggle('active', b.dataset.pos === val));
        avRender();
      }};

      window.avSort = function(col) {{
        if (avSortCol === col) {{ avSortDir *= -1; }}
        else {{ avSortCol = col; avSortDir = col === 'age' ? 1 : -1; }}
        avRender();
      }};
    }})();
    </script>
    """
    return render_page("Auction Values", league_id, "auction-values", body_html, platform, season)


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
                Settings
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
              <select id="prSort" onchange="prPage=1;prRender()"
                style="padding:7px 10px;border-radius:8px;border:1px solid var(--border);
                       background:var(--card-bg);color:var(--text);font-size:12px;cursor:pointer;outline:none;min-height:34px;">
                <option value="rank">Overall Rank</option>
                <option value="value">Value</option>
                <option value="age">Age</option>
                <option value="pos_rank">Pos Rank</option>
                <option value="ppg">PPG</option>
                <option value="total_pts">Total Points</option>
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
             grid-template-columns:28px 18px 1fr 52px 46px 46px 60px;
             gap:0;padding:6px 12px;border-radius:6px;
             background:var(--accent-soft);font-size:11px;
             font-weight:700;color:var(--accent);letter-spacing:0.04em;
             text-transform:uppercase;" class="pr-grid-row">
          <span>#</span>
          <span style="text-align:center;"></span>
          <span>Player</span>
          <span style="text-align:center;">Pos</span>
          <span id="prAgeHeader" style="text-align:center;">Age</span>
          <span style="text-align:right;">Team</span>
          <span id="prSortHeader" style="text-align:right;">Value</span>
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
        grid-template-columns: 28px 18px 1fr 52px 46px 46px 60px;
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

      /* Pagination */
      .pr-pagination {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-top: 1px solid var(--border);
        margin-top: 12px;
      }
      .pr-pagination-info {
        font-size: 13px;
        color: var(--text-muted);
      }
      .pr-pagination-controls {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .pr-pagination-btn {
        padding: 6px 10px;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: var(--card-bg);
        color: var(--text);
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.12s, border-color 0.12s;
        display: flex;
        align-items: center;
        gap: 4px;
      }
      .pr-pagination-btn:hover:not(:disabled) {
        background: var(--bg-alt);
        border-color: var(--accent-color);
      }
      .pr-pagination-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      .pr-tier-divider {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 0 3px;
        pointer-events: none;
      }
      .pr-tier-divider-line {
        flex: 1;
        height: 1px;
        opacity: 0.5;
      }
      .pr-tier-divider-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.06em;
        opacity: 0.75;
        white-space: nowrap;
      }
      @media (max-width: 540px) {
        .pr-pagination {
          flex-direction: column;
          gap: 8px;
          align-items: center;
        }
        .pr-pagination-btn .pr-btn-label {
          display: none;
        }
        .pr-pagination-btn {
          padding: 6px 10px;
          min-width: 36px;
          justify-content: center;
        }
      }
      .pr-page-numbers {
        display: flex;
        gap: 4px;
      }
      .pr-page-num {
        min-width: 32px;
        height: 32px;
        padding: 0 6px;
        border-radius: 6px;
        border: 1px solid var(--border);
        background: var(--card-bg);
        color: var(--text);
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.12s, border-color 0.12s;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .pr-page-num:hover { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
      .pr-page-num.pr-page-active { background: var(--accent); color: #fff; border-color: var(--accent); }

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
        /* Table: hide Age on tablets — rank | arrow | name | pos | team | sort */
        .pr-grid-row { grid-template-columns: 28px 16px 1fr 44px 42px 56px !important; }
        .pr-age,  #prAgeHeader  { display: none !important; }
      }
      @media (max-width: 480px) {
        /* Phone: rank | arrow | name | sort — hide pos and team */
        .pr-grid-row { grid-template-columns: 28px 16px 1fr 56px !important; }
        .pr-pos-cell, #prTableHeader span:nth-child(4) { display: none !important; }
        .pr-team,     #prTableHeader span:nth-child(6) { display: none !important; }
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
      var prPage = 1;
      var prPageSize = 50;

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
        prPage = 1;
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
        prPage = 1;
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
        prPage = 1;
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
        prPage = 1;
        prRender();
      }

      function prClearSearch() {
        document.getElementById('prSearch').value = '';
        prSearchQuery = '';
        document.getElementById('prSearchClear').style.display = 'none';
        prPage = 1;
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

      // Map sort key → { header label, cell value function }
      const PR_SORT_META = {
        rank:      { label: 'Value',    cell: p => prFormatValue(prGetValue(p)) },
        value:     { label: 'Value',    cell: p => prFormatValue(prGetValue(p)) },
        age:       { label: 'Age',      cell: p => p.age != null ? Number(p.age).toFixed(1) : '—' },
        pos_rank:  { label: 'Pos Rank', cell: p => prLeagueType === 'sf'
          ? (p.sf_pos_rank_label || p.pos_rank_label || p.position)
          : (p.pos_rank_label || p.position) },
        ppg:       { label: 'PPG',       cell: p => p.ppg != null ? p.ppg.toFixed(1) : '—' },
        total_pts: { label: 'Total Pts', cell: p => p.total_pts != null ? p.total_pts.toFixed(1) : '—' },
      };

      // Sort and filter players, then render rows into the main table
      function prRender() {
        if (!prLoaded) return;
        const sortBy = document.getElementById('prSort').value;

        // On mobile (≤768px) the Age column is hidden, so switch the sort column
        // to show whatever is being sorted. On desktop all columns are visible.
        const isMobile = window.innerWidth <= 768;
        const _alwaysShowSort = sortBy === 'ppg' || sortBy === 'total_pts';
        const sortMeta = (isMobile || _alwaysShowSort) ? (PR_SORT_META[sortBy] || PR_SORT_META.rank) : PR_SORT_META.rank;
        const sortHeaderEl = document.getElementById('prSortHeader');
        if (sortHeaderEl) sortHeaderEl.textContent = sortMeta.label;
        // Hide age col only on mobile when sort=age (shown in sort col instead)
        const ageHeaderEl = document.getElementById('prAgeHeader');
        if (isMobile && ageHeaderEl) ageHeaderEl.style.visibility = sortBy === 'age' ? 'hidden' : '';
        const ageColEls = document.querySelectorAll('.pr-age');
        if (isMobile) ageColEls.forEach(el => el.style.visibility = sortBy === 'age' ? 'hidden' : '');
        else ageColEls.forEach(el => el.style.visibility = '');

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
            } else if (sortBy === 'ppg') {
              return (b.ppg != null ? b.ppg : -1) - (a.ppg != null ? a.ppg : -1);
            } else if (sortBy === 'total_pts') {
              return (b.total_pts != null ? b.total_pts : -1) - (a.total_pts != null ? a.total_pts : -1);
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
          prRenderPagination(0, 0);
          return;
        }

        const total = players.length;
        const totalPages = Math.max(1, Math.ceil(total / prPageSize));
        if (prPage > totalPages) prPage = totalPages;
        const start = (prPage - 1) * prPageSize;
        const end   = Math.min(start + prPageSize, total);
        const pageSlice = players.slice(start, end);
        
        // Store filtered players count for pagination navigation
        window.prFilteredPlayers = players;

        empty.style.display = 'none';
        header.style.display = 'grid';
        count.style.display = 'block';
        count.textContent = 'Showing ' + (start + 1) + '–' + end + ' of ' + total + ' player' + (total !== 1 ? 's' : '');

        const _PR_TIER_COLORS = ['','#10b981','#22d3ee','#3b82f6','#8b5cf6','#a855f7','#f59e0b','#f97316','#94a3b8','#64748b'];
        const _PR_TIER_LABELS = ['','Elite','Star','High-End Starter','Starter','Flex','Bench','Deep Bench','Handcuff','Fringe'];

        list.innerHTML = '';
        let prevTier = null;
        pageSlice.forEach((p, i) => {
          const _tier = (sortBy === 'value' || sortBy === 'rank') ? prGetTier(p) : null;
          if (_tier && _tier !== prevTier) {
            const tc = _PR_TIER_COLORS[_tier] || '#64748b';
            const tl = _PR_TIER_LABELS[_tier] || ('Tier ' + _tier);
            const div = document.createElement('div');
            div.className = 'pr-tier-divider';
            div.innerHTML =
              `<div class="pr-tier-divider-line" style="background:${tc};"></div>` +
              `<span class="pr-tier-divider-label" style="color:${tc};" title="${tl}">T${_tier}</span>` +
              `<div class="pr-tier-divider-line" style="background:${tc};"></div>`;
            list.appendChild(div);
          }
          if (_tier) prevTier = _tier;
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
          const displayRank = (p.position === 'PICK' || (p.is_rookie && !_drafted)) ? '' : (start + i + 1);
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
          }

          const sortDisplay = p.position === 'PICK' && sortBy === 'age' ? '—' : sortMeta.cell(p);
          let sortDisplayHTML;
          if (p.position !== 'PICK' && sortBy === 'ppg' && p.ppg != null) {
            const pRank = p.ppg_rank ? (p.position + p.ppg_rank) : '-';
            sortDisplayHTML = `<span style="display:flex;flex-direction:column;align-items:flex-end;line-height:1.2;">`
              + `<span>${sortDisplay}</span>`
              + `<span style="font-size:10px;font-weight:600;color:var(--text-muted);">${pRank}</span>`
              + `</span>`;
          } else if (p.position !== 'PICK' && sortBy === 'total_pts' && p.total_pts != null) {
            const tRank = p.total_pts_rank ? (p.position + p.total_pts_rank) : '-';
            sortDisplayHTML = `<span style="display:flex;flex-direction:column;align-items:flex-end;line-height:1.2;">`
              + `<span>${sortDisplay}</span>`
              + `<span style="font-size:10px;font-weight:600;color:var(--text-muted);">${tRank}</span>`
              + `</span>`;
          } else {
            sortDisplayHTML = sortDisplay;
          }

          row.innerHTML =
            '<span class="pr-rank">'  + (displayRank ? '#' + displayRank : '—') + '</span>' +
            '<span class="pr-arrows">' + rankArrow + '</span>' +
            '<span class="pr-name player-clickable">'  + (p.name || 'Unknown') + badges + '</span>' +
            '<span class="pr-pos-cell">' + posRank + '</span>' +
            '<span class="pr-age">'   + (p.position === 'PICK' ? '—' : age) + '</span>' +
            '<span class="pr-team">'  + (p.team || '—') + '</span>' +
            '<span class="pr-value">' + sortDisplayHTML + '</span>';

          list.appendChild(row);
        });

        prRenderPagination(prPage, totalPages);
      }

      function prRenderPagination(page, totalPages) {
        let bar = document.getElementById('prPagination');
        if (!bar) {
          bar = document.createElement('div');
          bar.id = 'prPagination';
          bar.className = 'pr-pagination';
          document.getElementById('prList').insertAdjacentElement('afterend', bar);
          bar.innerHTML = `
            <div class="pr-pagination-info">
              <span id="prPaginationText">Showing 1-50 of 100 players</span>
            </div>
            <div class="pr-pagination-controls">
              <button id="prPrevBtn" class="pr-pagination-btn" onclick="prGoPage('prev')" disabled>
                <i class="fa-solid fa-chevron-left"></i><span class="pr-btn-label"> Previous</span>
              </button>
              <div id="prPageNumbers" class="pr-page-numbers"></div>
              <button id="prNextBtn" class="pr-pagination-btn" onclick="prGoPage('next')" disabled>
                <span class="pr-btn-label">Next </span><i class="fa-solid fa-chevron-right"></i>
              </button>
            </div>
          `;
        }
        
        if (totalPages <= 1) { 
          bar.style.display = 'none'; 
          return;
        }

        bar.style.display = 'flex';

        // Update info text
        const start = (page - 1) * prPageSize + 1;
        const end = Math.min(page * prPageSize, window.prFilteredPlayers.length);
        const total = window.prFilteredPlayers.length;
        document.getElementById('prPaginationText').textContent = `Showing ${start}–${end} of ${total} players`;

        // Update Previous/Next buttons
        const prevBtn = document.getElementById('prPrevBtn');
        const nextBtn = document.getElementById('prNextBtn');
        prevBtn.disabled = page === 1;
        nextBtn.disabled = page === totalPages;

        // Update page numbers
        const pageNumbers = document.getElementById('prPageNumbers');
        const isMobile = window.innerWidth <= 540;
        const maxPages = isMobile ? 3 : 5;
        const wing = isMobile ? 1 : 2;
        let pages = [];

        if (totalPages <= maxPages) {
          for (let i = 1; i <= totalPages; i++) pages.push(i);
        } else {
          pages = [1];
          let lo = Math.max(2, page - wing), hi = Math.min(totalPages - 1, page + wing);
          if (lo > 2) pages.push('…');
          for (let i = lo; i <= hi; i++) pages.push(i);
          if (hi < totalPages - 1) pages.push('…');
          pages.push(totalPages);
        }

        pageNumbers.innerHTML = pages.map(p => {
          if (p === '…') return '<span style="color: var(--text-muted); font-size: 13px; padding: 0 4px; line-height: 32px;">…</span>';
          const active = p === page ? ' pr-page-active' : '';
          return `<button class="pr-page-num${active}" onclick="prGoPage(${p})">${p}</button>`;
        }).join('');
      }

      function prGoPage(p) {
        if (p === 'prev') {
          prPage = Math.max(1, prPage - 1);
        } else if (p === 'next') {
          const totalPages = Math.ceil(window.prFilteredPlayers.length / prPageSize);
          prPage = Math.min(totalPages, prPage + 1);
        } else {
          prPage = p;
        }
        prRender();
        // Scroll to top of player list
        const el = document.getElementById('prTableHeader');
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }

      // Wire up search input
      (function() {
        const inp   = document.getElementById('prSearch');
        const clear = document.getElementById('prSearchClear');
        if (!inp) return;

        inp.addEventListener('input', function() {
          prSearchQuery = inp.value.trim();
          clear.style.display = prSearchQuery.length > 0 ? 'block' : 'none';
          prPage = 1;
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

      var prTierThresholds = {};

      function prGetTier(p) {
        const lt  = prLeagueType || '1qb';
        const sz  = String(prLeagueSize || 10);
        const tbl = (prTierThresholds[lt] || {})[sz] || (prTierThresholds['1qb'] || {})['10'] || [];
        if (!tbl.length) return null;
        const val = prGetValue(p);
        for (let i = 0; i < tbl.length; i++) {
          if (val >= tbl[i]) return Math.min(i + 1, 9);
        }
        return 9; // T9 catch-all — matches Python _asset_tier clamp
      }

      // Load data
      Promise.all([
        fetch('/api/league-players', { cache: 'no-store' }).then(r => r.json()),
        fetch('/api/player-indicators?league_type=1qb&league_size=10', { cache: 'no-store' })
          .then(r => r.json()).catch(() => ({}))
      ]).then(([resp, indicators]) => {
        prIndicators = indicators || {};
        // Support both old (array) and new (object with players + tier_thresholds) format
        const rawPlayers = Array.isArray(resp) ? resp : (resp.players || []);
        prTierThresholds = (!Array.isArray(resp) && resp.tier_thresholds) ? resp.tier_thresholds : {};

        // Helper function to calculate precise age from birthday
        function calculateAgeFromBirthday(bDay) {
          if (!bDay) return null;
          try {
            const parts = bDay.split('/');
            if (parts.length !== 3) return null;
            const [month, day, year] = parts.map(Number);
            const birthDate = new Date(year, month - 1, day);
            const today = new Date();
            
            // Calculate precise age including partial years
            let age = today.getFullYear() - birthDate.getFullYear();
            const monthDiff = today.getMonth() - birthDate.getMonth();
            const dayDiff = today.getDate() - birthDate.getDate();
            
            if (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0)) {
              age--;
            }
            
            // Calculate partial year as decimal
            const lastBirthday = new Date(
              today.getFullYear() - (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0) ? 1 : 0),
              birthDate.getMonth(),
              birthDate.getDate()
            );
            const daysSinceBirthday = (today - lastBirthday) / (1000 * 60 * 60 * 24);
            const daysInYear = (today.getFullYear() % 4 === 0 && (today.getFullYear() % 100 !== 0 || today.getFullYear() % 400 === 0)) ? 366 : 365;
            
            age += daysSinceBirthday / daysInYear;
            return Math.round(age * 10) / 10; // Round to 1 decimal place
          } catch (e) {
            return null;
          }
        }

        prAllPlayers = rawPlayers
          .filter(p => p && p.id != null)
          .map(p => {
            // Calculate precise age from birthday if available
            const preciseAge = calculateAgeFromBirthday(p.bDay);
            
            return {
              id:               String(p.id),
              name:             p.name || p.full_name || 'Unknown',
              team:             p.team || '',
              position:         String(p.position || '').toUpperCase(),
              age:              preciseAge !== null ? preciseAge : (p.age != null ? Number(p.age) : null),
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
              ppg:              p.ppg != null ? Number(p.ppg) : null,
              total_pts:        p.total_pts != null ? Number(p.total_pts) : null,
              ppg_rank:         p.ppg_rank != null ? Number(p.ppg_rank) : null,
              total_pts_rank:   p.total_pts_rank != null ? Number(p.total_pts_rank) : null,
              ppg_season:       p.ppg_season || null,
            };
          })
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

    return render_page("Player Rankings", None, "players", body_html)


@app.route("/<platform>/<int:season>/<league_id>/prospects")
def page_prospects(platform: str, season: int, league_id: str):
    """Rookie prospect rankings page — active class auto-detected."""
    from dashboard_services.pages.rookies_page import build_prospects_body
    body_html = build_prospects_body()
    return render_page("Prospect Rankings", league_id, "prospects", body_html, platform, season)


@app.route("/<platform>/<int:season>/<league_id>/breakouts")
def page_breakouts(platform: str, season: int, league_id: str):
    """Dedicated page for breakout candidates with detailed projections."""
    user_id = session.get("viewer_username")
    has_premium = has_premium_access(user_id, league_id, platform)
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
      const BO_HAS_PREMIUM = {str(has_premium).lower()};
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

        const FREE_LIMIT = 3;
        const visible = BO_HAS_PREMIUM ? filtered : filtered.slice(0, FREE_LIMIT);
        const locked = !BO_HAS_PREMIUM && filtered.length > FREE_LIMIT;

        let html = '<div class="breakout-grid">';

        visible.forEach(candidate => {{
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
            <div class="breakout-card" style="cursor:pointer;" onclick="openPlayerModal('` + pid + `', '', {{tab:'breakout'}})">
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

        if (locked) {{
          html += `
            <div class="breakout-card" onclick="showPaywall('breakout-candidates')" style="cursor:pointer;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;min-height:180px;border:2px dashed var(--border);">
              <i class="fa-solid fa-lock" style="font-size:22px;color:var(--text-muted);"></i>
              <div style="font-weight:700;font-size:15px;">${{filtered.length - FREE_LIMIT}} more candidates locked</div>
              <div style="font-size:12px;color:var(--text-muted);text-align:center;">Upgrade to see all breakout<br>candidates and full details</div>
              <span style="font-size:11px;font-weight:700;padding:4px 12px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:12px;">Upgrade &rarr;</span>
            </div>
          `;
        }}
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






def _try_grant_from_stripe_success() -> None:
    """
    When a user returns from Stripe checkout, verify the session server-side
    and grant the subscription immediately. This is a reliable fallback for
    when the webhook is delayed or misconfigured.
    """
    if request.args.get("success") != "1":
        return
    checkout_session_id = request.args.get("session_id", "").strip()
    if not checkout_session_id:
        return
    try:
        cs = stripe.checkout.Session.retrieve(checkout_session_id)
        if cs.status != "complete":
            return

        meta      = cs.metadata.to_dict() if cs.metadata else {}
        plan      = meta.get("plan")
        user_id   = meta.get("user_id")
        league_id = meta.get("league_id") or ""
        sub_id    = cs.subscription
        cust_id   = cs.customer

        if plan not in ("league", "user", "combo"):
            return
        if plan == "user" and not user_id:
            return
        if plan == "league" and not league_id:
            return
        if plan == "combo" and not league_id and not user_id:
            return

        # Skip if already active (webhook may have already fired)
        if has_premium_access(user_id or None, league_id or None, "sleeper"):
            return

        try:
            sub        = stripe.Subscription.retrieve(sub_id) if sub_id else None
            expires_at = (
                datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)
                if sub else datetime.now(timezone.utc) + timedelta(days=366)
            )
        except Exception:
            expires_at = datetime.now(timezone.utc) + timedelta(days=366)

        if plan in ("league", "combo") and league_id:
            create_league_subscription(
                league_id, user_id or "", expires_at,
                stripe_subscription_id=sub_id,
                stripe_customer_id=cust_id,
            )
        if plan in ("user", "combo") and user_id:
            create_user_subscription(
                user_id, expires_at,
                stripe_subscription_id=sub_id,
                stripe_customer_id=cust_id,
            )
    except Exception:
        logger.exception("[stripe] success-page session verification failed")






def _pricing_body() -> str:
    plan      = request.args.get("plan", "")
    success   = request.args.get("success") == "1"
    canceled  = request.args.get("canceled") == "1"
    return_to = request.args.get("return_to", "").strip()

    if success:
        safe_return = html.escape(return_to) if return_to else ""
        return f"""
    <div class="card central" style="max-width:560px;text-align:center;">
      <div class="card-body" style="padding:48px 32px;">
        <div id="sub-icon" style="font-size:56px;margin-bottom:20px;">
          <i class="fa-solid fa-circle-check" style="color:#22c55e;"></i>
        </div>
        <h2 id="sub-heading" style="margin:0 0 10px;font-size:24px;">Payment confirmed!</h2>
        <p id="sub-msg" style="color:var(--text-muted);margin:0 0 28px;">
          Your premium access is activating&nbsp;&mdash; just a moment&hellip;
        </p>
        <div id="sub-spinner" style="margin:0 auto 24px;width:36px;height:36px;border:3px solid #e5e7eb;border-top-color:#667eea;border-radius:50%;animation:paywall-spin .8s linear infinite;"></div>
        {'<a id="sub-return" href="' + safe_return + '" style="display:none;padding:12px 28px;border-radius:9px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;font-weight:700;text-decoration:none;font-size:15px;">Continue</a>' if return_to else ''}
      </div>
    </div>
    <script>
    (function() {{
      var returnTo = {json.dumps(return_to)};
      var attempts = 0, maxAttempts = 20;

      // Extract league_id from return URL path (/{{platform}}/{{season}}/{{league_id}}/...)
      var leagueId = '';
      try {{
        if (returnTo) {{
          var parts = new URL(returnTo, window.location.origin).pathname.split('/').filter(Boolean);
          if (parts.length >= 3) leagueId = parts[2];
        }}
      }} catch(e) {{}}
      var statusUrl = '/api/subscription-status' + (leagueId ? '?league_id=' + encodeURIComponent(leagueId) : '');

      function activate() {{
        attempts++;
        fetch(statusUrl)
          .then(function(r) {{ return r.json(); }})
          .then(function(d) {{
            if (d.has_premium) {{
              document.getElementById('sub-spinner').style.display = 'none';
              document.getElementById('sub-msg').textContent = 'Premium is active on your account!';
              if (returnTo) {{
                setTimeout(function() {{ window.location.href = returnTo; }}, 1200);
              }} else {{
                document.getElementById('sub-heading').textContent = 'You\\'re all set!';
              }}
            }} else if (attempts < maxAttempts) {{
              setTimeout(activate, 2000);
            }} else {{
              document.getElementById('sub-spinner').style.display = 'none';
              document.getElementById('sub-msg').textContent =
                'Your access is being set up. If it isn\\'t active in a minute, try refreshing the page.';
              var btn = document.getElementById('sub-return');
              if (btn) btn.style.display = 'inline-block';
            }}
          }})
          .catch(function() {{
            if (attempts < maxAttempts) setTimeout(activate, 2000);
          }});
      }}

      setTimeout(activate, 1500);
    }})();
    </script>
    """

    league_highlight = "border-color:#667eea;box-shadow:0 8px 24px rgba(102,126,234,.2);" if plan == "league" else ""
    user_highlight   = "border-color:#667eea;box-shadow:0 8px 24px rgba(102,126,234,.2);" if plan == "user"   else ""
    canceled_banner = """
    <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:14px 18px;margin-bottom:20px;color:#dc2626;font-size:14px;">
      <i class="fa-solid fa-circle-xmark" style="margin-right:6px;"></i>
      Checkout was canceled. You have not been charged.
    </div>""" if canceled else ""
    return f"""
    {canceled_banner}
    <div class="card central" style="max-width:760px;">
      <div class="card-header" style="border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:0;text-align:center;">
        <h2 style="margin:0 0 6px;font-size:22px;">BR Fantasy Premium</h2>
        <div style="font-size:14px;color:var(--text-muted);">
          Unlock advanced analytics and insights for your dynasty league
        </div>
      </div>
      <div class="card-body" style="padding-top:28px;">

        <!-- Feature list -->
        <div style="margin-bottom:28px;">
          <div style="font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--text-muted);margin-bottom:12px;">What you get</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="display:flex;align-items:center;gap:8px;font-size:14px;">
              <i class="fa-solid fa-chart-line" style="color:#667eea;width:16px;text-align:center;"></i>
              Full Trade Intelligence feed
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:14px;">
              <i class="fa-solid fa-fire" style="color:#667eea;width:16px;text-align:center;"></i>
              All Breakout Engine candidates
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:14px;">
              <i class="fa-solid fa-clock-rotate-left" style="color:#667eea;width:16px;text-align:center;"></i>
              Player trade history
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:14px;">
              <i class="fa-solid fa-star" style="color:#667eea;width:16px;text-align:center;"></i>
              All future premium features
            </div>
          </div>
        </div>

        <!-- Pricing cards -->
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:28px;">

          <!-- League plan -->
          <div style="border:2px solid #e5e7eb;border-radius:14px;padding:24px;transition:all .2s;background:var(--card);">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;min-height:28px;">
              <div style="font-size:17px;font-weight:700;">League Plan</div>
            </div>
            <div style="font-size:38px;font-weight:800;line-height:1;margin-bottom:4px;">
              $10<span style="font-size:16px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">Premium for every manager in your league</div>
            <button onclick="initiatePurchase('league', this)" style="width:100%;padding:11px;border-radius:9px;border:2px solid #667eea;background:var(--card);color:#667eea;font-size:14px;font-weight:700;cursor:pointer;">
              Subscribe for League
            </button>
          </div>

          <!-- Combo plan -->
          <div style="border:2px solid #667eea;border-radius:14px;padding:24px;transition:all .2s;background:var(--card);{league_highlight}">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
              <div style="font-size:17px;font-weight:700;">League + Personal</div>
              <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;font-size:10px;font-weight:700;padding:3px 9px;border-radius:10px;text-transform:uppercase;letter-spacing:.4px;">Best Value</div>
            </div>
            <div style="font-size:38px;font-weight:800;line-height:1;margin-bottom:4px;">
              $12<span style="font-size:16px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">Premium for your league and all your personal leagues</div>
            <button onclick="initiatePurchase('combo', this)" style="width:100%;padding:11px;border-radius:9px;border:none;background:linear-gradient(135deg,#667eea,#764ba2);color:white;font-size:14px;font-weight:700;cursor:pointer;">
              Subscribe Both
            </button>
          </div>

          <!-- Personal plan -->
          <div style="border:2px solid #e5e7eb;border-radius:14px;padding:24px;transition:all .2s;background:var(--card);{user_highlight}">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;min-height:28px;">
              <div style="font-size:17px;font-weight:700;">Personal Plan</div>
            </div>
            <div style="font-size:38px;font-weight:800;line-height:1;margin-bottom:4px;">
              $5<span style="font-size:16px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">Premium for all your leagues, one account</div>
            <button onclick="initiatePurchase('user', this)" style="width:100%;padding:11px;border-radius:9px;border:2px solid #667eea;background:var(--card);color:#667eea;font-size:14px;font-weight:700;cursor:pointer;">
              Subscribe Personally
            </button>
          </div>

        </div>

        <!-- Free tier note -->
        <div style="text-align:center;font-size:13px;color:var(--text-muted);padding-top:12px;border-top:1px solid var(--border);">
          <i class="fa-solid fa-circle-info" style="margin-right:4px;"></i>
          ADP rankings and basic player data are always free.
        </div>

      </div>
    </div>

    <style>
      @media (max-width: 760px) {{
        .card-body > div:nth-child(2) {{ grid-template-columns: 1fr !important; }}
        .card-body > div:nth-child(3) {{ grid-template-columns: 1fr !important; }}
      }}
    </style>
    """


_STRIPE_LEAGUE_PRODUCT = "prod_USjDJYPhNGnmvM"
_STRIPE_USER_PRODUCT   = "prod_USjDRuVDcwH1xb"
_STRIPE_COMBO_PRODUCT  = "prod_UT5DaCA4u6hWgb"
_STRIPE_PRICES = {
    "league": {"unit_amount": 1000, "product": _STRIPE_LEAGUE_PRODUCT},
    "user":   {"unit_amount":  500, "product": _STRIPE_USER_PRODUCT},
    "combo":  {"unit_amount": 1200, "product": _STRIPE_COMBO_PRODUCT},
}










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
                time.sleep(3)  # let the foreground redirect complete first
                hist_seasons = get_available_history_seasons(p, lid, s)
                for hist_s in hist_seasons[:3]:  # cap at 3 most-recent seasons
                    rid = resolve_league_id_for_season(p, lid, s, hist_s)
                    get_league_ctx_from_cache(p, rid, hist_s)
                    time.sleep(2)  # spread out API load to avoid rate limiting
            except Exception:
                pass

        threading.Thread(
            target=_preload_history, args=(platform, league_id, season), daemon=True
        ).start()

        next_url = (request.form.get("next") or "").strip()
        if next_url and next_url.startswith("/") and not next_url.startswith("//"):
            return redirect(next_url)
        return redirect(url_for(
            "page_dashboard",
            platform=platform,
            season=season,
            league_id=league_id,
        ))

    next_url = (request.args.get("next") or "").strip()
    # Reject open redirects — only allow local paths
    if not (next_url.startswith("/") and not next_url.startswith("//")):
        next_url = ""
    body_html = render_template_string(
        FORM_BODY,
        username="",
        viewed_season=viewed_season,
        error=None,
        next_url=next_url,
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
            body_html = _build_offseason_standings_body(ctx) if ctx.get("offseason_mode") else build_standings_body(ctx)

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




# ---------- global cache for model value table used by trade eval ----------
_MODEL_VALUE_CACHE = None
_MODEL_VALUE_CACHE_TS = 0
_MODEL_VALUE_TTL = 60 * 60  # 1 hour

# Caches for advanced-metrics endpoints (10-minute TTL)
_ROLE_PLAYERS_CACHE: dict = {}
_ROLE_PLAYERS_CACHE_TS: dict = {}
_BREAKOUT_CACHE = None
_BREAKOUT_CACHE_TS = 0.0
_ADVANCED_METRICS_TTL = 600

# Cache for player-details available-years scan (5-minute TTL)
_PLAYER_DETAIL_YEARS_CACHE: set = set()
_PLAYER_DETAIL_YEARS_CACHE_TS = 0.0
_PLAYER_DETAIL_YEARS_TTL = 300

# Cache for bulk PPG stats (2-hour TTL) — computed from sleeper_stats files
_PPG_STATS_CACHE: dict = {}   # player_id → {ppg, total_pts, games, season}
_PPG_STATS_CACHE_TS = 0.0
_PPG_STATS_CACHE_TTL = 7200


def _score_stats(s: dict, ss: dict) -> float:
    """Compute fantasy points from a Sleeper stats dict using Sleeper key names."""
    p = 0.0
    p += (s.get("pass_yd") or 0) * (ss.get("pass_yd") or 0.04)
    p += (s.get("pass_td") or 0) * (ss.get("pass_td") or 4.0)
    p += (s.get("pass_int") or 0) * (ss.get("pass_int") or -2.0)
    p += (s.get("rush_yd") or 0) * (ss.get("rush_yd") or 0.1)
    p += (s.get("rush_td") or 0) * (ss.get("rush_td") or 6.0)
    p += (s.get("rec") or 0) * (ss.get("rec") or 0)
    p += (s.get("rec_yd") or 0) * (ss.get("rec_yd") or 0.1)
    p += (s.get("rec_td") or 0) * (ss.get("rec_td") or 6.0)
    p += (s.get("fum_lost") or 0) * (ss.get("fum_lost") or -2.0)
    py = s.get("pass_yd") or 0
    ry = s.get("rush_yd") or 0
    ey = s.get("rec_yd") or 0
    rr = ry + ey
    if py >= 400: p += (ss.get("bonus_pass_yd_400") or 0)
    elif py >= 300: p += (ss.get("bonus_pass_yd_300") or 0)
    if ry >= 200: p += (ss.get("bonus_rush_yd_200") or 0)
    elif ry >= 100: p += (ss.get("bonus_rush_yd_100") or 0)
    if ey >= 200: p += (ss.get("bonus_rec_yd_200") or 0)
    elif ey >= 100: p += (ss.get("bonus_rec_yd_100") or 0)
    if rr >= 200: p += (ss.get("bonus_rush_rec_yd_200") or 0)
    elif rr >= 100: p += (ss.get("bonus_rush_rec_yd_100") or 0)
    return p


def _compute_bulk_ppg_stats() -> dict:
    """Read all sleeper_stats files for the most recent season and compute PPG per player using current league scoring.

    Returns a dict keyed by player_id (str) with keys: ppg, total_pts, games, season.
    Uses standard PPR scoring so the numbers are consistent across the app.
    """
    global _PPG_STATS_CACHE, _PPG_STATS_CACHE_TS
    now = time.time()
    if _PPG_STATS_CACHE and now - _PPG_STATS_CACHE_TS < _PPG_STATS_CACHE_TTL:
        return _PPG_STATS_CACHE

    # Find the most recent season that has stats files
    stats_base = os.path.join("cache", "sleeper_stats")
    all_files = glob.glob(os.path.join(stats_base, "sleeper_stats_s*_w*.json"))
    season_years: set = set()
    for f in all_files:
        m = re.match(r'sleeper_stats_s(\d+)_w(\d+)', os.path.basename(f))
        if m:
            season_years.add(int(m.group(1)))

    if not season_years:
        return {}

    latest_season = max(season_years)

    try:
        from dashboard_services.api import get_effective_scoring_settings as _gess
        _ss = _gess()
    except Exception:
        _ss = {}

    # Aggregate pts per player across all weeks of the latest season
    player_totals: dict = {}  # player_id → {"pts": float, "games": int}
    week_files = glob.glob(os.path.join(stats_base, f"sleeper_stats_s{latest_season}_w*.json"))

    for wf in week_files:
        try:
            with open(wf) as fh:
                week_stats = json.load(fh)
        except Exception:
            continue
        if not isinstance(week_stats, dict):
            continue
        for pid, stats in week_stats.items():
            if not isinstance(stats, dict):
                continue
            pts = _score_stats(stats, _ss)
            if pts > 0:
                rec = player_totals.setdefault(pid, {"pts": 0.0, "games": 0})
                rec["pts"] += pts
                rec["games"] += 1

    result = {}
    for pid, d in player_totals.items():
        g = d["games"]
        if g > 0:
            result[str(pid)] = {
                "ppg":       round(d["pts"] / g, 1),
                "total_pts": round(d["pts"], 1),
                "games":     g,
                "season":    latest_season,
            }

    _PPG_STATS_CACHE = result
    _PPG_STATS_CACHE_TS = now
    return result

def get_model_value_table_cached():
    global _MODEL_VALUE_CACHE, _MODEL_VALUE_CACHE_TS
    now = time.time()
    if _MODEL_VALUE_CACHE is not None and now - _MODEL_VALUE_CACHE_TS < _MODEL_VALUE_TTL:
        return _MODEL_VALUE_CACHE
    # Prefer DB (player_values table) so all endpoints use the same source.
    tbl = None
    try:
        from dashboard_services.player_value_history import load_current_values_from_db
        tbl = load_current_values_from_db() or None
    except Exception as _e:
        print(f"[model-value-cache] DB load failed: {_e}")
    if not tbl:
        tbl = list(load_model_value_table() or [])

    # Apply FC/DP market corrections directly against the loaded values.
    # Compare model values already in tbl — no extra DB query needed.
    if tbl:
        try:
            from data_building.external_data.external_values_scraper import (
                load_fantasycalc_api_values, load_dynastyprocess_values,
            )
            from utils.utils import normalize_name as _nn
            from collections import defaultdict as _dd

            # Normalize FC to 0-999.9
            _fc_rows = load_fantasycalc_api_values() or []
            _fc_by_sid: dict = {}
            if _fc_rows:
                _max_fc = max((float(r.get("value") or 0) for r in _fc_rows if r.get("value")), default=0)
                if _max_fc > 0:
                    for _r in _fc_rows:
                        _sid = str(_r.get("sleeper_id") or "").strip()
                        if _sid:
                            try:
                                _fc_by_sid[_sid] = float(_r["value"]) * 999.9 / _max_fc
                            except (TypeError, ValueError):
                                pass

            # Normalize DP to 0-999.9
            _dp_rows = load_dynastyprocess_values() or []
            _dp_by_name: dict = {}
            if _dp_rows:
                _max_dp = max((float(r.get("value_1qb") or 0) for r in _dp_rows if r.get("value_1qb")), default=0)
                if _max_dp > 0:
                    for _r in _dp_rows:
                        _nm = str(_r.get("player") or "").strip()
                        if _nm:
                            try:
                                _dp_by_name[_nn(_nm)] = float(_r["value_1qb"]) * 999.9 / _max_dp
                            except (TypeError, ValueError):
                                pass

            # Build pid→name map from players_index so DP lookup works even when
            # player dicts from load_current_values_from_db() have no name field
            # (player_values table has no name column).
            try:
                from utils.utils import load_players_index as _lpi
                _pid_to_name = {str(k): (v.get("name") or "") for k, v in (_lpi() or {}).items()}
            except Exception:
                _pid_to_name = {}

            # Load trade counts (best-effort)
            _trade_counts: dict = {}
            try:
                from dashboard_services.db import get_conn as _gc
                with _gc() as _conn:
                    with _conn.cursor() as _cur:
                        _cur.execute("SELECT player_id, trade_count FROM trade_intel_player_stats")
                        for _r in _cur.fetchall():
                            _rd = dict(_r)
                            _trade_counts[str(_rd["player_id"])] = int(_rd.get("trade_count") or 0)
            except Exception:
                pass

            _corrected = 0
            if _fc_by_sid or _dp_by_name:
                for _p in tbl:
                    _pid     = str(_p.get("id") or "")
                    _mval    = float(_p.get("value") or 0)
                    if _mval <= 0:
                        continue
                    _msf     = float(_p.get("sf_value") or _mval)
                    _fc_val  = _fc_by_sid.get(_pid)
                    # Use name from player dict if present, else fall back to players_index
                    _name    = str(_p.get("name") or "") or _pid_to_name.get(_pid, "")
                    _dp_val  = _dp_by_name.get(_nn(_name))

                    if _fc_val is None and _dp_val is None:
                        continue

                    _avail   = [v for v in (_fc_val, _dp_val) if v is not None]
                    _ext_avg = sum(_avail) / len(_avail)

                    _inflated = (
                        (_fc_val is not None and _fc_val > 0 and _mval > 1.5 * _fc_val)
                        or (_dp_val is not None and _dp_val > 0 and _mval > 1.5 * _dp_val)
                    )
                    _low_trades = _trade_counts.get(_pid, 0) < 20

                    if _inflated or _low_trades:
                        _p["value"]    = round(_ext_avg, 2)
                        _p["sf_value"] = round(_ext_avg * (_msf / _mval), 2)
                        _corrected += 1

                print(f"[model-value-cache] applied {_corrected} FC/DP market corrections")

                # Recompute pos_rank / pos_rank_label after corrections
                _pos_idx: dict    = _dd(list)
                _sf_pos_idx: dict = _dd(list)
                for _i, _p in enumerate(tbl):
                    _pos = str(_p.get("position") or "").upper()
                    if _pos and _pos != "PICK":
                        _pos_idx[_pos].append(_i)
                        _sf_pos_idx[_pos].append(_i)
                for _pos, _idxs in _pos_idx.items():
                    _idxs.sort(key=lambda _i: float(tbl[_i].get("value") or 0), reverse=True)
                    for _rank, _i in enumerate(_idxs, 1):
                        tbl[_i]["pos_rank"]       = _rank
                        tbl[_i]["pos_rank_label"] = f"{_pos}{_rank}"
                for _pos, _idxs in _sf_pos_idx.items():
                    _idxs.sort(key=lambda _i: float(tbl[_i].get("sf_value") or 0), reverse=True)
                    for _rank, _i in enumerate(_idxs, 1):
                        tbl[_i]["sf_pos_rank"]       = _rank
                        tbl[_i]["sf_pos_rank_label"] = f"{_pos}{_rank}"
        except Exception as _e:
            print(f"[model-value-cache] market corrections skipped: {_e}")

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

    user_id = session.get("viewer_username")
    if not has_premium_access(user_id, league_id, platform):
        return jsonify({"paywall": True, "error": "Premium required"}), 403

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
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

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

    # Use DB as primary source (same as api_league_players) so chip values and
    # team totals always draw from the same number.  Fall back to JSON if DB unavailable.
    try:
        from dashboard_services.player_value_history import load_current_values_from_db as _lcvdb
        value_table = _lcvdb() or get_model_value_table_cached()
    except Exception:
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

    tier_thresholds = compute_tier_thresholds(value_table, league_type, league_size)
    # Only apply depth adjustment when asset counts differ — equal counts penalise both
    # sides symmetrically and just confuse the result.
    side_a_count = len(side_a_players) + len(side_a_picks)
    side_b_count = len(side_b_players) + len(side_b_picks)
    if side_a_count != side_b_count:
        apply_tier_stack_adjustment(side_a, side_b, tier_thresholds, is_sf=(league_type == "sf"))

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
            from dashboard_services.ai.context_builders import calculate_roster_depth_warning, build_model_value_lookup, _ctx_is_sf
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
                model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [], is_sf=_ctx_is_sf(ctx))
                viewer_gives = side_a if viewer_side == "b" else side_b
                viewer_gets  = side_b if viewer_side == "b" else side_a
                sending = [a for a in (viewer_gives.get("assets") or []) if str(a.get("position") or "").upper() != "PICK"]
                receiving = [a for a in (viewer_gets.get("assets") or []) if str(a.get("position") or "").upper() != "PICK"]
                depth_warnings = calculate_roster_depth_warning(
                    viewer_roster, model_value_lookup, sending, receiving,
                    roster_positions=ctx.get("roster_positions") or [],
                    num_teams=len(rosters) or 12,
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
        "depth_warnings": depth_warnings,
        "tier_thresholds": [round(t, 1) for t in tier_thresholds],
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
        page  (int, default 1)   -- 1-based page number
        limit (int, default 0)   -- results per page; 0 = return all (legacy)
        q     (str, optional)    -- prefix/substring filter applied before paging
    """
    try:
        from utils.utils import load_players_index
        players_index = load_players_index() or {}
        value_table = get_model_value_table_cached() or []
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
    try:
        model_value_table = list(get_model_value_table_cached() or [])
    except Exception as e:
        print(f"[api/league-players] Cache load failed: {e}")
        model_value_table = []

    if not isinstance(model_value_table, list):
        raise ValueError("model_value_table must be a list of player objects")

    # DB values are preferred (more accurate); WLS fills in bucket picks or any gaps.
    try:
        from dashboard_services.picks import load_pick_value_table as _lpvt
        _pick_values = _lpvt()
        _db_pick_ids: set = {
            str(p.get("id") or "") for p in model_value_table
            if str(p.get("position") or "").upper() == "PICK"
        }
        # Build pick-type sets per (year, round) — slot > bucket > generic hierarchy.
        _bucket_keywords = {"early", "mid", "late"}
        _slot_yr_rnd: set = set()
        _bucket_yr_rnd: set = set()
        for _pid in _db_pick_ids:
            _pp = _pid.split("_")
            if len(_pp) >= 3:
                _key = (_pp[0], _pp[1])
                if _pp[2].lower() in _bucket_keywords:
                    _bucket_yr_rnd.add(_key)
                else:
                    try:
                        int(_pp[2])
                        _slot_yr_rnd.add(_key)
                    except ValueError:
                        pass

        _injected_picks = []
        _seen_ids: set = set(_db_pick_ids)
        for _pk_id, _pk_val in _pick_values.items():
            if _pk_id in _seen_ids or float(_pk_val) <= 0:
                continue
            _parts = _pk_id.split("_")
            if len(_parts) >= 2:
                _key = (_parts[0], _parts[1])
                _is_generic = len(_parts) == 2
                _is_bucket = len(_parts) >= 3 and _parts[2].lower() in _bucket_keywords
                if _key in _slot_yr_rnd and (_is_generic or _is_bucket):
                    continue
                if _key in _bucket_yr_rnd and _is_generic:
                    continue
            _seen_ids.add(_pk_id)
            # Format display name: "2026_1_01" -> "2026 1.01", "2026_1_early" -> "2026 1st (Early)"
            if len(_parts) >= 3:
                _yr, _rnd_s, _third = _parts[0], _parts[1], "_".join(_parts[2:])
                try:
                    _rnd = int(_rnd_s)
                    _sfx = {1:"st",2:"nd",3:"rd"}.get(_rnd,"th")
                    _bkt = {"early":"Early","mid":"Mid","late":"Late"}.get(_third.lower())
                    if _bkt:
                        _name = f"{_yr} {_rnd}{_sfx} ({_bkt})"
                    else:
                        _name = f"{_yr} {_rnd}.{int(_third):02d}"
                except (ValueError, TypeError):
                    _name = _pk_id.replace("_", " ")
            else:
                _name = _pk_id.replace("_", " ")
            _injected_picks.append({
                "id": _pk_id, "name": _name, "position": "PICK",
                "value": round(float(_pk_val), 1), "team": "",
            })
        model_value_table.extend(_injected_picks)
        print(f"[api/league-players] DB picks: {len(_db_pick_ids)}, WLS fallback picks: {len(_injected_picks)}")

        # Rebuild slot/bucket sets to include WLS-injected picks so the model
        # injection below correctly suppresses bucket picks when slots exist.
        for _p in _injected_picks:
            _pp2 = str(_p.get("id") or "").split("_")
            if len(_pp2) >= 3:
                _k2 = (_pp2[0], _pp2[1])
                if _pp2[2].lower() in _bucket_keywords:
                    _bucket_yr_rnd.add(_k2)
                else:
                    try:
                        int(_pp2[2])
                        _slot_yr_rnd.add(_k2)
                    except ValueError:
                        pass

        # Inject model_values.json bucket picks, preferring them over WLS for
        # future years (WLS bucket values are in WLS units; model values are
        # pre-calibrated to the 0-999.9 model scale).
        # Current-year slot picks (already FC-normalized in picks.py) are kept.
        try:
            _cur_year = str(date.today().year)
            # Load picks directly from model_values.json — get_model_value_table_cached()
            # returns DB players (non-empty) so never falls back to model_values.json picks.
            _raw_model = load_model_value_table(apply_calibration=False) or []
            _json_picks = [p for p in _raw_model
                           if str(p.get("position") or "").upper() == "PICK"]
            _all_seen = {str(p.get("id") or "") for p in model_value_table
                         if str(p.get("position") or "").upper() == "PICK"}
            _json_injected = 0
            for _jp in _json_picks:
                _jid = str(_jp.get("id") or "")
                if not _jid:
                    continue
                _jparts = _jid.split("_")
                _jyear = _jparts[0] if _jparts else ""
                _jis_bucket = (len(_jparts) >= 3 and
                               _jparts[2].lower() in _bucket_keywords)
                _jis_generic = len(_jparts) == 2

                # Skip if slot picks for this year+round already exist
                if len(_jparts) >= 2:
                    _jkey = (_jparts[0], _jparts[1])
                    if _jkey in _slot_yr_rnd and (_jis_generic or _jis_bucket):
                        continue
                    if _jkey in _bucket_yr_rnd and _jis_generic:
                        continue

                # For future years, model bucket picks override WLS bucket picks.
                # For the current year, only add picks not already present.
                if _jid in _all_seen:
                    if _jyear == _cur_year or not _jis_bucket:
                        continue
                    # Replace the WLS entry with the model value
                    for _i, _existing in enumerate(model_value_table):
                        if str(_existing.get("id") or "") == _jid:
                            model_value_table[_i] = _jp
                            break
                else:
                    _all_seen.add(_jid)
                    model_value_table.append(_jp)
                _json_injected += 1
            if _json_injected:
                print(f"[api/league-players] model_values.json picks injected/overridden: {_json_injected}")
        except Exception as _e_json_picks:
            pass
    except Exception as _e_picks:
        print(f"[api/league-players] pick injection skipped: {_e_picks}")

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

    # --- Rank labels (no decay applied — DB values are source of truth) ---
    from collections import defaultdict as _dd_prl

    def _recompute_ranks(table, val_key, rank_key, label_key):
        _groups: dict = _dd_prl(list)
        for _i, _p in enumerate(table):
            _pos = str(_p.get("position") or "").upper()
            if _pos and _pos != "PICK":
                _groups[_pos].append(_i)
        for _pos, _idxs in _groups.items():
            _idxs.sort(key=lambda _i: float(table[_i].get(val_key) or 0), reverse=True)
            for _rank, _i in enumerate(_idxs, 1):
                table[_i][rank_key]  = _rank
                table[_i][label_key] = f"{_pos}{_rank}"

    _recompute_ranks(model_value_table, "value",    "pos_rank",    "pos_rank_label")
    _recompute_ranks(model_value_table, "sf_value", "sf_pos_rank", "sf_pos_rank_label")

    _recompute_ranks(model_value_table, "value",    "pos_rank",    "pos_rank_label")
    _recompute_ranks(model_value_table, "sf_value", "sf_pos_rank", "sf_pos_rank_label")

    # Sort players: first by value (descending), then by pos_rank (ascending for ties)
    model_value_table.sort(key=lambda p: (
        -(float(p.get("value") or 0)),
        int(p.get("pos_rank") or 9999)
    ))

    # Add birthday data for precise age calculation in frontend
    try:
        from utils.utils import load_players_index
        players_index = load_players_index() or {}
        
        # Enrich player data with birthday information
        for player in model_value_table:
            player_id = str(player.get("id") or "")
            player_data = players_index.get(player_id)
            if player_data:
                player["bDay"] = player_data.get("bDay")
    except Exception as e:
        print(f"[api/league-players] Could not add birthday data: {e}")

    # Enrich with PPG and total points from usage cache (full PPR, min 4 games)
    try:
        import os as _os_lp, json as _json_lp
        _season_lp = date.today().year
        _usage_data_lp = None
        _usage_season_lp = None
        for _s in [_season_lp, _season_lp - 1]:
            _up = _os_lp.path.join("cache", "player_history", f"usage_rows_{_s}.json")
            if _os_lp.path.exists(_up):
                with open(_up) as _f:
                    _usage_data_lp = _json_lp.load(_f)
                _usage_season_lp = _s
                break
        if _usage_data_lp:
            _usage_map_lp = {str(p.get("id")): p for p in _usage_data_lp if p.get("id")}
            # Build positional PPG and total_pts lists for ranking
            _pos_ppg_lp: dict = {}
            _pos_total_lp: dict = {}
            for _p in _usage_data_lp:
                _u = _p.get("usage") or {}
                _g = int(_u.get("games") or 0)
                if _g < 4:
                    continue
                _ppg_v = _u.get("ppr_ppg")
                if _ppg_v is None:
                    continue
                _ppg_v = round(float(_ppg_v), 1)
                _tot_v = round(_ppg_v * _g, 1)
                _pos_lp = str(_p.get("position") or "")
                _pos_ppg_lp.setdefault(_pos_lp, []).append(_ppg_v)
                _pos_total_lp.setdefault(_pos_lp, []).append(_tot_v)
            # Sort descending for rank lookup
            _pos_ppg_sorted_lp = {pos: sorted(vals, reverse=True) for pos, vals in _pos_ppg_lp.items()}
            _pos_total_sorted_lp = {pos: sorted(vals, reverse=True) for pos, vals in _pos_total_lp.items()}
            for _player in model_value_table:
                _pid = str(_player.get("id") or "")
                _entry = _usage_map_lp.get(_pid)
                if not _entry:
                    continue
                _u = _entry.get("usage") or {}
                _g = int(_u.get("games") or 0)
                if _g < 4:
                    continue
                _ppg_v = _u.get("ppr_ppg")
                if _ppg_v is None:
                    continue
                _ppg_v = round(float(_ppg_v), 1)
                _tot_v = round(_ppg_v * _g, 1)
                _pos_lp = str(_entry.get("position") or "")
                _ppg_sorted = _pos_ppg_sorted_lp.get(_pos_lp, [])
                _tot_sorted = _pos_total_sorted_lp.get(_pos_lp, [])
                _player["ppg"] = _ppg_v
                _player["total_pts"] = _tot_v
                _player["ppg_games"] = _g
                _player["ppg_season"] = _usage_season_lp
                _player["ppg_rank"] = (_ppg_sorted.index(_ppg_v) + 1) if _ppg_v in _ppg_sorted else None
                _player["total_pts_rank"] = (_tot_sorted.index(_tot_v) + 1) if _tot_v in _tot_sorted else None
    except Exception as _e_lp:
        print(f"[api/league-players] PPG enrichment skipped: {_e_lp}")

    # Compute tier thresholds for every league-type × size combination so the
    # frontend can display each player's tier badge without a second API call.
    _tier_thresholds_all = {}
    for _lt in ("1qb", "sf"):
        _tier_thresholds_all[_lt] = {}
        for _sz in (8, 10, 12, 14):
            _tier_thresholds_all[_lt][str(_sz)] = [
                round(v, 1) for v in
                compute_tier_thresholds(model_value_table, _lt, _sz)
            ]

    return jsonify(_sanitize_for_json({
        "players": model_value_table,
        "tier_thresholds": _tier_thresholds_all,
    }))


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
        value_table = get_model_value_table_cached() or []
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
                value_table = get_model_value_table_cached() or []
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
        logger.exception("[breakout-candidates] Error")
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
        global _ROLE_PLAYERS_CACHE, _ROLE_PLAYERS_CACHE_TS
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

        cache_key = (position, limit)
        now = time.time()
        if cache_key in _ROLE_PLAYERS_CACHE and now - _ROLE_PLAYERS_CACHE_TS.get(cache_key, 0) < _ADVANCED_METRICS_TTL:
            return jsonify(_ROLE_PLAYERS_CACHE[cache_key])

        players = get_top_role_players(position=position, limit=limit)

        # Clean up internal fields
        for player in players:
            player.pop("id", None)
            # Convert decimals to floats
            for k, v in player.items():
                if v is not None and k not in ("player_id", "position", "as_of_date"):
                    player[k] = float(v)

        _ROLE_PLAYERS_CACHE[cache_key] = players
        _ROLE_PLAYERS_CACHE_TS[cache_key] = now
        return jsonify(players)

    except Exception as e:
        logger.exception("[top-role-players] Error")
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
        global _BREAKOUT_CACHE, _BREAKOUT_CACHE_TS
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

        now = time.time()
        if _BREAKOUT_CACHE is not None and now - _BREAKOUT_CACHE_TS < _ADVANCED_METRICS_TTL:
            return jsonify(_BREAKOUT_CACHE)

        candidates = detect_breakout_candidates(
            lookback_days=lookback_days,
            min_games=min_games,
        )

        _BREAKOUT_CACHE = candidates
        _BREAKOUT_CACHE_TS = now
        return jsonify(candidates)

    except Exception as e:
        logger.exception("[breakout-candidates] Error")
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
    - Mike Evans leaves TB -> Egbuka gets targets
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
        model_values = get_model_value_table_cached() or []
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

        # Fall back to full players index if not found in relevant index
        if not player_meta:
            players_index_full = load_players_index() or {}
            player_meta = players_index_full.get(player_id, {})

        if not player_meta:
            return jsonify({"error": "Player not found"}), 404

        player_team = player_meta.get("team", "")

        # Get value data (use cache so FC/DP corrections are applied)
        value_table = get_model_value_table_cached() or []
        player_value = next((p for p in value_table if str(p.get("id")) == str(player_id)), {})

        # Get FULL value history from database (not just 90 days)
        value_history = get_player_value_history(
            player_id, days=365,
            league_type=_modal_lt, league_size=_modal_ls,
        )

        # Scale history to match the FC/DP-corrected current value so the chart
        # ends at the corrected value. Proportional scaling preserves trend shape.
        if value_history and player_value:
            _is_sf = _modal_lt == "sf"
            _corrected_cur = player_value.get("sf_value" if _is_sf else "value")
            if _corrected_cur is not None:
                _latest_raw = value_history[-1]["value"]
                if _latest_raw > 0 and abs(float(_corrected_cur) - _latest_raw) > 1.0:
                    _ratio = float(_corrected_cur) / _latest_raw
                    for _h in value_history:
                        _h["value"] = round(_h["value"] * _ratio, 1)
                        if _h.get("delta_from_prev") is not None:
                            _h["delta_from_prev"] = round(_h["delta_from_prev"] * _ratio, 1)

        # Load game logs from sleeper_stats for all available seasons
        game_logs_by_year = {}

        # Find all available season years — cached for 5 minutes to avoid
        # repeated glob scans on every player-details request.
        global _PLAYER_DETAIL_YEARS_CACHE, _PLAYER_DETAIL_YEARS_CACHE_TS
        now = time.time()
        if not _PLAYER_DETAIL_YEARS_CACHE or now - _PLAYER_DETAIL_YEARS_CACHE_TS > _PLAYER_DETAIL_YEARS_TTL:
            stats_files = glob.glob(os.path.join("cache", "sleeper_stats", "sleeper_stats_*.json"))
            _fresh_years: set = set()
            for stats_file in stats_files:
                try:
                    basename = os.path.basename(stats_file)
                    if basename.startswith("sleeper_stats_s"):
                        match = re.match(r'sleeper_stats_s(\d+)_w(\d+)', basename)
                        if match:
                            _fresh_years.add(int(match.group(1)))
                except Exception:
                    continue
            _PLAYER_DETAIL_YEARS_CACHE = _fresh_years
            _PLAYER_DETAIL_YEARS_CACHE_TS = now
        available_years = _PLAYER_DETAIL_YEARS_CACHE

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
                    logger.warning("[api_player_details] Error loading schedule %s: %s", schedule_file, e)
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

        # ── Fantasy team ownership (only when league context is provided) ──
        fantasy_team = None
        fantasy_team_owner = None
        if league_id:
            try:
                from dashboard_services.service import fantasy_team_and_roster_for_player as _ft_lookup
                _ctx = get_league_ctx_from_cache(platform, league_id, season)
                _rosters = _ctx.get("rosters") or []
                _users   = _ctx.get("users")   or []
                _rmap    = _build_roster_map(_users, _rosters)
                _team_name, _rid = _ft_lookup(str(player_id), _rosters, _rmap)
                if _team_name and _team_name != "Free Agent":
                    fantasy_team = _team_name
                    # Find the owner's username for the sub-label
                    _roster_obj = next((r for r in _rosters if str(r.get("roster_id")) == _rid), None)
                    if _roster_obj:
                        _owner_id = _roster_obj.get("owner_id")
                        _user = next((u for u in _users if u.get("user_id") == _owner_id), None)
                        fantasy_team_owner = (_user or {}).get("display_name") or (_user or {}).get("username")
            except Exception as _fe:
                logger.debug("[api_player_details] roster lookup failed: %s", _fe)

        # Compute PPG and positional scoring rank from usage cache
        _ppg = None
        _ppg_rank = None
        _ppg_games = None
        _ppg_season_used = None
        _total_pts = None
        _total_pts_rank = None
        try:
            import os as _os, json as _json2
            _ppr_val = scoring_settings.get("pointsPerReception", 0.5)
            def _pick_ppg(u):
                if _ppr_val >= 1.0:
                    return u.get("ppr_ppg")
                if _ppr_val <= 0:
                    return u.get("std_scoring_ppg") or u.get("std_ppg")
                return u.get("half_ppr_ppg")

            for _ppg_s in [season, season - 1]:
                _up = _os.path.join("cache", "player_history", f"usage_rows_{_ppg_s}.json")
                if not _os.path.exists(_up):
                    continue
                _ud = _json2.load(open(_up))
                _pe = next((p for p in _ud if str(p.get("id")) == str(player_id)), None)
                if not _pe:
                    continue
                _pu = _pe.get("usage") or {}
                _pg = int(_pu.get("games") or 0)
                if _pg < 4:
                    continue
                _ppg = _pick_ppg(_pu)
                if _ppg is None:
                    continue
                _ppg = round(float(_ppg), 1)
                _total_pts = round(_ppg * _pg, 1)
                _ppg_games = _pg
                _ppg_season_used = _ppg_s
                # Scoring rank within position (min 4 games)
                _pos_str = player_meta.get("pos", "")
                if _pos_str:
                    _pos_players = [
                        p for p in _ud
                        if p.get("position") == _pos_str
                        and int((p.get("usage") or {}).get("games") or 0) >= 4
                        and _pick_ppg(p.get("usage") or {}) is not None
                    ]
                    _all_ppg = sorted(
                        [round(float(_pick_ppg(p.get("usage") or {})), 1) for p in _pos_players],
                        reverse=True,
                    )
                    try:
                        _ppg_rank = _all_ppg.index(_ppg) + 1
                    except ValueError:
                        _ppg_rank = None
                    # Total points rank — round ppg before multiplying so values match
                    _all_total = sorted(
                        [round(round(float(_pick_ppg(p.get("usage") or {})), 1) * int((p.get("usage") or {}).get("games") or 0), 1)
                         for p in _pos_players],
                        reverse=True,
                    )
                    try:
                        _total_pts_rank = _all_total.index(_total_pts) + 1
                    except ValueError:
                        _total_pts_rank = None
                break
        except Exception:
            pass

        response = {
            "player_id": player_id,
            "name": player_meta.get("name", "Unknown"),
            "position": player_meta.get("pos"),
            "team": player_meta.get("team"),
            "age": player_value.get("age"),
            "pos_rank": player_value.get("pos_rank"),
            "pos_rank_label": player_value.get("pos_rank_label"),
            "espnHeadshot": player_meta.get("espnHeadshot"),
            "fantasy_team": fantasy_team,
            "fantasy_team_owner": fantasy_team_owner,
            "stats": {
                "value": round(player_value.get("value", 0), 1) if player_value.get("value") else None,
                "sf_value": round(player_value.get("sf_value", 0), 1) if player_value.get("sf_value") else None,
                "pos_rank": player_value.get("pos_rank"),
                "pos_rank_label": player_value.get("pos_rank_label"),
                "years_exp": player_meta.get("years_exp"),
                "ppg": _ppg,
                "ppg_rank": _ppg_rank,
                "ppg_season": _ppg_season_used,
                "ppg_games": _ppg_games,
                "total_pts": _total_pts,
                "total_pts_rank": _total_pts_rank,
            },
            "value_history": value_history,
            "game_logs_by_year": game_logs_by_year,
            "prospect_data": prospect_data,
        }

        return jsonify(response)

    except Exception as e:
        logger.exception("[api_player_details] Error")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/player-game-logs/<player_id>")
def api_player_game_logs(player_id: str):
    """Game logs for the Stats tab -- lazy-loaded separately from player-details."""
    try:
        import glob
        from utils.utils import load_relevant_index
        from dashboard_services.api import get_effective_scoring_settings
        from dashboard_services.platform_api import sync_league_globals

        league_id = request.args.get("league_id")
        platform  = request.args.get("platform", "sleeper")
        season    = int(request.args.get("season", datetime.now().year))

        if league_id:
            sync_league_globals(platform, league_id, season)
            scoring_settings = get_effective_scoring_settings()
        else:
            scoring_settings = {
                "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
                "rush_yd": 0.1,  "rush_td": 6.0, "rec": 1.0,
                "rec_yd": 0.1,   "rec_td": 6.0,  "fum_lost": -2.0,
            }

        players_index = load_relevant_index() or {}
        player_meta = players_index.get(player_id) or {}
        if not player_meta:
            players_index_full = load_players_index() or {}
            player_meta = players_index_full.get(player_id) or {}
        player_team = player_meta.get("team", "")

        # Reuse the years cache
        global _PLAYER_DETAIL_YEARS_CACHE, _PLAYER_DETAIL_YEARS_CACHE_TS
        now = time.time()
        if not _PLAYER_DETAIL_YEARS_CACHE or now - _PLAYER_DETAIL_YEARS_CACHE_TS > _PLAYER_DETAIL_YEARS_TTL:
            stats_files = glob.glob(os.path.join("cache", "sleeper_stats", "sleeper_stats_*.json"))
            _fresh_years: set = set()
            for sf in stats_files:
                bn = os.path.basename(sf)
                if bn.startswith("sleeper_stats_s"):
                    m = re.match(r'sleeper_stats_s(\d+)_w(\d+)', bn)
                    if m:
                        _fresh_years.add(int(m.group(1)))
            _PLAYER_DETAIL_YEARS_CACHE = _fresh_years
            _PLAYER_DETAIL_YEARS_CACHE_TS = now
        available_years = _PLAYER_DETAIL_YEARS_CACHE

        game_logs_by_year: dict = {}

        for season_year in sorted(available_years, reverse=True):
            game_logs = []

            schedule_by_week: dict = {}
            for schedule_file in glob.glob(os.path.join("cache", "schedule", f"schedule_s{season_year}_w*_d*.json")):
                try:
                    fn = os.path.basename(schedule_file)
                    week_num = int(fn.split('_w')[1].split('_')[0])
                    with open(schedule_file) as f:
                        games = json.load(f)
                    if isinstance(games, list) and week_num not in schedule_by_week:
                        schedule_by_week[week_num] = games
                except Exception:
                    continue

            stats_by_week: dict = {}
            for week_file in glob.glob(os.path.join("cache", "sleeper_stats", f"sleeper_stats_s{season_year}_w*.json")):
                try:
                    m = re.match(r'sleeper_stats_s(\d+)_w(\d+)', os.path.basename(week_file))
                    if m:
                        with open(week_file) as f:
                            stats_by_week[int(m.group(2))] = json.load(f)
                except Exception:
                    continue

            if not any(player_id in ws for ws in stats_by_week.values()):
                continue

            def _calc_pts(s):
                return round(_score_stats(s, scoring_settings), 2)

            def _stats_dict(s):
                return {k: s.get(k) for k in ["pass_yd","pass_td","pass_int","rush_att","rush_yd","rush_td","rec","rec_tgt","rec_yd","rec_td","fum_lost"]}

            if schedule_by_week:
                # Full path: schedule available — include opponent and date
                for week_num in sorted(schedule_by_week.keys()):
                    games = schedule_by_week[week_num]
                    if not isinstance(games, list):
                        continue
                    opponent = ""
                    is_away  = False
                    game_date = ""
                    for game in games:
                        if not isinstance(game, dict):
                            continue
                        home_team = game.get("home", "")
                        away_team = game.get("away", "")
                        if player_team and player_team == home_team:
                            opponent = away_team; is_away = False; game_date = game.get("gameDate", ""); break
                        elif player_team and player_team == away_team:
                            opponent = home_team; is_away = True;  game_date = game.get("gameDate", ""); break
                    stats = (stats_by_week.get(week_num) or {}).get(player_id)
                    if stats:
                        game_logs.append({"week": week_num, "date": game_date,
                            "opponent": f"@{opponent}" if is_away else opponent,
                            "fantasy_pts": _calc_pts(stats), "stats": _stats_dict(stats)})
                    else:
                        game_logs.append({"week": week_num, "date": game_date,
                            "opponent": f"@{opponent}" if is_away else opponent,
                            "fantasy_pts": None, "stats": None})
            else:
                # Fallback: no schedule files — show stats without opponent/date
                for week_num in sorted(stats_by_week.keys()):
                    stats = stats_by_week[week_num].get(player_id)
                    if stats:
                        game_logs.append({"week": week_num, "date": "",
                            "opponent": "", "fantasy_pts": _calc_pts(stats),
                            "stats": _stats_dict(stats)})
                    else:
                        game_logs.append({"week": week_num, "date": "",
                            "opponent": "", "fantasy_pts": None, "stats": None})

            if game_logs:
                game_logs.sort(key=lambda g: g.get("date", "") or "")
                game_logs_by_year[season_year] = game_logs

        return jsonify({"game_logs_by_year": game_logs_by_year})
    except Exception as e:
        logger.exception("[api_player_game_logs] error")
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
        _modal_draft_ended = has_draft_ended(league_id, platform, season)
        _modal_pick_start = 1 if _modal_draft_ended else 0

        # Build picks
        all_picks = []
        for offset in range(_modal_pick_start, _modal_pick_start + 3):  # Next 3 years
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
                    except Exception:
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
                        except Exception:
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
                        except Exception:
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
                            "via": via,
                            "original_owner": pick_info["original_owner"],
                        })

        # Sort picks by year then round
        all_picks.sort(key=lambda p: (p["year"], p["round"]))

        # Add pick values to total (match offseason snapshot calculation)
        try:
            from dashboard_services.picks import load_pick_value_table
            pick_by_key = load_pick_value_table() or {}
            picks_for_value = [
                {"season": p["year"], "round": p["round"], "original_owner": p.get("original_owner")}
                for p in all_picks
            ]
            total_value += _team_pick_value(picks_for_value, pick_by_key, platform=platform,
                                            league_id=league_id, season=season)
        except Exception:
            pass

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
            logger.warning("[api_team_details] Error getting graph data: %s", graph_err)
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
        logger.exception("[api_team_details] Error")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route("/api/team-trades/<roster_id>")
def api_team_trades(roster_id: str):
    """Return all trades for a specific team in the current league season."""
    try:
        from dashboard_services.service import get_transactions_by_week
        from utils.utils import load_players_index

        league_id = request.args.get("league_id")
        platform = request.args.get("platform", "sleeper")
        season = int(request.args.get("season") or datetime.now().year)

        if not league_id:
            return jsonify({"error": "league_id required"}), 400

        players_index = load_players_index() or {}

        weeks = list(range(1, 19))
        tx_by_week = get_transactions_by_week(league_id, weeks, platform=platform, season=season) or {}

        def _pinfo(pid):
            meta = players_index.get(str(pid)) or {}
            return {
                "player_id": str(pid),
                "name": meta.get("name") or str(pid),
                "position": meta.get("pos") or "",
            }

        trades = []
        for week in sorted(tx_by_week):
            for t in (tx_by_week[week] or []):
                if t.get("type") != "trade":
                    continue

                adds = t.get("adds") or {}
                drops = t.get("drops") or {}
                draft_picks = t.get("draft_picks") or []
                base_rids = set(str(r) for r in (t.get("roster_ids") or []))
                all_rids = base_rids | {str(v) for v in adds.values()} | {str(v) for v in drops.values()}

                if str(roster_id) not in all_rids:
                    continue

                ts_raw = t.get("status_updated") or t.get("created")
                date_str = ""
                if ts_raw:
                    from datetime import timezone as _tz
                    _dt = datetime.fromtimestamp(ts_raw / 1000.0, tz=_tz.utc)
                    date_str = _dt.strftime("%-m/%-d/%y")

                my_gets = [_pinfo(pid) for pid, to_rid in adds.items() if str(to_rid) == str(roster_id)]
                my_sends = [_pinfo(pid) for pid, from_rid in drops.items() if str(from_rid) == str(roster_id)]
                my_pick_gets = [p for p in draft_picks if str(p.get("owner_id")) == str(roster_id)]
                my_pick_sends = [p for p in draft_picks if str(p.get("previous_owner_id")) == str(roster_id)]

                trades.append({
                    "week": week,
                    "date": date_str,
                    "my_gets": my_gets,
                    "my_sends": my_sends,
                    "my_pick_gets": my_pick_gets,
                    "my_pick_sends": my_pick_sends,
                })

        trades.sort(key=lambda x: x["week"], reverse=True)
        return jsonify({"trades": trades, "total": len(trades)})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/draft-needs")
def api_draft_needs():
    """
    Returns positional needs for a team relative to league averages.
    Uses the same weighted positional strength + z-score approach as the Teams page.
    Need levels: -2 stacked, -1 depth, 0 neutral, 1 need, 2 major need.
    """
    try:
        from utils.utils import load_players_index, load_model_value_table
        league_id = request.args.get("league_id")
        platform  = request.args.get("platform", "sleeper")
        season    = int(request.args.get("season") or datetime.now().year)
        roster_id = request.args.get("roster_id") or ""

        # Fall back to the session viewer when roster_id is absent or the sentinel
        if not roster_id or roster_id == "viewer":
            roster_id = session.get("viewer_roster_id") or ""

        if not league_id or not roster_id:
            return jsonify({"error": "league_id and roster_id required"}), 400

        rosters = get_rosters(platform, league_id, season) or []
        league  = get_league(platform, league_id, season) or {}
        players_index = load_players_index() or {}
        value_table   = load_model_value_table() or []

        roster_positions = (league.get("roster_positions") or [])
        is_sf  = any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in roster_positions)
        vfield = "sf_value" if is_sf else "value"

        values_by_id = {str(r["id"]): r for r in value_table if isinstance(r, dict) and r.get("id")}

        CORE = ("QB", "RB", "WR", "TE")

        # Count roster slots for _weighted_pos_strength
        slot_counts: dict[str, int] = {}
        for rp in roster_positions:
            rp_str = str(rp).upper()
            slot_counts[rp_str] = slot_counts.get(rp_str, 0) + 1

        # Build per-roster positional value lists (same as teams page)
        roster_pos_vals: dict[str, dict[str, list]] = {}
        for r in rosters:
            rid = str(r.get("roster_id", ""))
            pv: dict[str, list] = {p: [] for p in CORE}
            for pid in (r.get("players") or []):
                meta = players_index.get(str(pid)) or {}
                pos  = str(meta.get("pos") or "").upper()
                if pos not in CORE:
                    continue
                vrow = values_by_id.get(str(pid)) or {}
                val  = float(vrow.get(vfield) or vrow.get("value") or 0)
                pv[pos].append(val)
            roster_pos_vals[rid] = pv

        if not roster_pos_vals:
            return jsonify({"needs": {}, "league_type": "sf" if is_sf else "1qb"})

        # Compute weighted positional strength per roster (mirrors _weighted_pos_strength)
        roster_strength: dict[str, dict[str, float]] = {}
        for rid, pv in roster_pos_vals.items():
            roster_strength[rid] = {
                pos: _weighted_pos_strength(pv[pos], pos, slot_counts)
                for pos in CORE
            }

        # League avg + std per position
        import math
        n = len(roster_strength)
        league_avg = {pos: sum(rv[pos] for rv in roster_strength.values()) / n for pos in CORE}
        league_std = {}
        for pos in CORE:
            variance = sum((rv[pos] - league_avg[pos]) ** 2 for rv in roster_strength.values()) / n
            league_std[pos] = math.sqrt(variance) if variance > 0 else 1.0

        viewer = roster_strength.get(str(roster_id), {p: 0.0 for p in CORE})

        needs: dict = {}
        for pos in CORE:
            mu    = league_avg[pos]
            sigma = league_std[pos]
            z     = (viewer[pos] - mu) / sigma if sigma > 0 else 0.0
            # Map z-score to need level (same thresholds as teams page z-score usage)
            if   z >= 1.0:  level = -2   # stacked
            elif z >= 0.35: level = -1   # depth
            elif z >= -0.35:level =  0   # neutral
            elif z >= -1.0: level =  1   # need
            else:           level =  2   # major need
            needs[pos]              = level
            needs[f"{pos}_count"]   = len(roster_pos_vals.get(str(roster_id), {}).get(pos, []))
            needs[f"{pos}_value"]   = round(viewer[pos], 1)
            needs[f"{pos}_avg"]     = round(league_avg[pos], 1)

        return jsonify({
            "needs": needs,
            "league_type": "sf" if is_sf else "1qb",
            "league_size": len(rosters),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# /api/subscription-status → routes/billing_bp.py :: api_subscription_status()

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


@app.route("/api/playoff-odds")
def api_playoff_odds():
    """
    Run Monte Carlo playoff odds simulation for a league.

    In-season: simulates remaining regular-season games (10 000 runs) using
    each team's historical scoring distribution and returns probabilities.
    Offseason / complete: returns 100 / 0 based on final standings.

    Query params: platform, league_id, season
    """
    platform  = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    season_s  = (request.args.get("season") or "").strip()

    if not league_id or not season_s:
        return jsonify({"error": "missing params"}), 400
    try:
        season = int(season_s)
    except ValueError:
        return jsonify({"error": "invalid season"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        if not ctx:
            return jsonify({"error": "league not found"}), 404

        from data_building.simulate_playoff_odds import simulate_playoff_odds
        odds = simulate_playoff_odds(ctx, platform=platform)

        settings           = ctx.get("league_settings") or {}
        playoff_week_start = int(settings.get("playoff_week_start") or 15)
        playoff_teams      = int(settings.get("playoff_teams") or 6)
        current_week       = int(ctx.get("current_week") or 0)
        is_complete        = bool(odds and odds[0].get("is_complete"))

        return jsonify({
            "odds":                odds,
            "season":              season,
            "current_week":        current_week,
            "playoff_week_start":  playoff_week_start,
            "playoff_teams":       playoff_teams,
            "is_complete":         is_complete,
        })
    except Exception as exc:
        logger.exception("[playoff-odds] %s", exc)
        return jsonify({"error": "Internal error"}), 500


from dashboard_services.adp_service import (
    fetch_fc_rookie_adp as _fetch_fc_rookie_adp,
    fetch_league_adp_from_db as _fetch_league_adp_from_db_impl,
    build_model_adp_fallback as _build_model_adp_fallback,
)


def _fetch_league_adp_from_db(
    is_sf: bool,
    season: int,
    draft_type: str,
    num_teams: int = 10,
    min_samples: int = 40,
) -> dict:
    return _fetch_league_adp_from_db_impl(is_sf, season, draft_type, min_samples)


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

            if adp_diff >= 4:           score = 4   # clear value
            elif adp_diff >= 2:         score = 3   # good value
            elif adp_diff >= -3:        score = 2   # on ADP
            elif adp_diff >= big_reach: score = 1   # reach within 1 round → D
            else:                       score = 0   # > 1 round early → F

            # BPA bonus / penalty.
            # Only penalise when the pick was close to ADP (adp_diff >= -2):
            # for bigger reaches the adp_diff already captures the cost, so
            # applying BPA on top would double-count and turn a D into an F.
            if is_bpa:
                # A reach is still a reach even if this was the best player left.
                # Cap the BPA bonus at +1 when the pick was already a meaningful
                # reach (adp_diff < -3) so BPA alone can't rescue a D to a B.
                score += 1 if adp_diff < -3 else 2
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
                "market_trend": round(float(r["market_trend_1qb"]) * _market_scale(), 1) if r["market_trend_1qb"] is not None else None,
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

        # ID-based filters (comma-separated player IDs)
        _pa_raw = (request.args.get("player_a") or "").strip()
        _pb_raw = (request.args.get("player_b") or "").strip()
        player_a_ids = [x for x in _pa_raw.split(",") if x] if _pa_raw else []
        player_b_ids = [x for x in _pb_raw.split(",") if x] if _pb_raw else []

        from dashboard_services.db import get_conn
        from utils.utils import load_players_index

        players_map = load_players_index() or {}

        # Resolve name search to IDs (legacy ?q= path)
        match_ids: list[str] = []
        if q:
            match_ids = [
                pid for pid, info in players_map.items()
                if q in (info.get("name") or "").lower()
            ]
            if not match_ids:
                return jsonify({"trades": [], "total": 0, "has_more": False})

        # Build superflex filter as a parameterized condition (no f-string interpolation)
        sf_param = None
        if league_type == "sf":
            sf_param = True
        elif league_type == "1qb":
            sf_param = False
        sf_clause = "AND l.is_superflex = %s" if sf_param is not None else ""

        # Build side-specific EXISTS filters (all static SQL, no user data in f-strings)
        filter_clauses: list = []
        filter_params:  list = []

        if player_a_ids:
            filter_clauses.append(
                "EXISTS (SELECT 1 FROM trade_intel_assets _fa WHERE _fa.trade_id = t.id"
                " AND _fa.side = 'a' AND _fa.asset_type = 'player' AND _fa.player_id = ANY(%s))"
            )
            filter_params.append(player_a_ids)

        if player_b_ids:
            filter_clauses.append(
                "EXISTS (SELECT 1 FROM trade_intel_assets _fb WHERE _fb.trade_id = t.id"
                " AND _fb.side = 'b' AND _fb.asset_type = 'player' AND _fb.player_id = ANY(%s))"
            )
            filter_params.append(player_b_ids)

        if match_ids and not (player_a_ids or player_b_ids):
            # Legacy name-search: any-side match
            filter_clauses.append(
                "EXISTS (SELECT 1 FROM trade_intel_assets _fm WHERE _fm.trade_id = t.id"
                " AND _fm.asset_type = 'player' AND _fm.player_id = ANY(%s))"
            )
            filter_params.append(match_ids)

        filter_sql = (" AND " + " AND ".join(filter_clauses)) if filter_clauses else ""
        sf_p       = [sf_param] if sf_param is not None else []

        with get_conn() as conn:
            count_params = [season] + sf_p + filter_params
            count_row = conn.execute(
                f"""
                SELECT COUNT(DISTINCT t.id) AS n
                FROM trade_intel_trades t
                LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                WHERE t.season = %s {sf_clause}{filter_sql}
                """,
                count_params,
            ).fetchone()
            total = int(count_row["n"]) if count_row else 0

            row_params = [season] + sf_p + filter_params + [limit + 1, page * limit]
            trade_rows = conn.execute(
                f"""
                SELECT DISTINCT t.id, t.transaction_id, t.season, t.week, t.created_at,
                       l.scoring_type, l.is_superflex, l.num_teams
                FROM trade_intel_trades t
                LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                WHERE t.season = %s {sf_clause}{filter_sql}
                ORDER BY t.created_at DESC NULLS LAST
                LIMIT %s OFFSET %s
                """,
                row_params,
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


@app.route("/api/trade-intel/player-trades/<player_id>")
def api_trade_intel_player_trades(player_id: str):
    """
    Paginated trade history for a single player.
    ?season=<int>  &league_type=all|sf|1qb  &page=<int>  &limit=<int>
    Returns each trade with both sides and is_focus=True on the queried player.
    """
    try:
        season      = int(request.args.get("season") or datetime.now().year)
        league_type = (request.args.get("league_type") or "all").strip().lower()
        page        = max(1, int(request.args.get("page") or 1))
        limit       = min(int(request.args.get("limit") or 15), 50)
        offset      = (page - 1) * limit

        sf_param = None
        if league_type == "sf":
            sf_param = True
        elif league_type == "1qb":
            sf_param = False
        sf_clause = "AND l.is_superflex = %s" if sf_param is not None else ""

        from dashboard_services.db import get_conn
        from utils.utils import load_players_index

        players_map = load_players_index() or {}

        with get_conn() as conn:
            base = [player_id, season] + ([sf_param] if sf_param is not None else [])

            count_row = conn.execute(
                f"""
                SELECT COUNT(DISTINCT t.id) AS n
                FROM trade_intel_trades t
                JOIN trade_intel_assets a ON a.trade_id = t.id
                LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                WHERE a.player_id = %s AND a.asset_type = 'player'
                  AND t.season = %s {sf_clause}
                """,
                base,
            ).fetchone()
            total = int(count_row["n"]) if count_row else 0

            trade_rows = conn.execute(
                f"""
                SELECT DISTINCT
                    t.id, t.transaction_id, t.season, t.week, t.created_at,
                    l.is_superflex, l.num_teams
                FROM trade_intel_trades t
                JOIN trade_intel_assets a ON a.trade_id = t.id
                LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
                WHERE a.player_id = %s AND a.asset_type = 'player'
                  AND t.season = %s {sf_clause}
                ORDER BY t.created_at DESC NULLS LAST
                LIMIT %s OFFSET %s
                """,
                base + [limit, offset],
            ).fetchall()

            if not trade_rows:
                return jsonify({"trades": [], "total": total, "page": page,
                                "total_pages": 0, "has_prev": False, "has_next": False})

            trade_ids = [r["id"] for r in trade_rows]
            asset_rows = conn.execute(
                """
                SELECT trade_id, side, asset_type, player_id,
                       pick_season, pick_round, pick_order, pick_slot
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
                pid  = str(a["player_id"])
                info = players_map.get(pid) or {}
                return {
                    "type":      "player",
                    "player_id": pid,
                    "name":      info.get("name") or pid,
                    "position":  info.get("pos") or "?",
                    "is_focus":  pid == str(player_id),
                }
            s     = str(a["pick_season"]) if a["pick_season"] else "?"
            r     = str(a["pick_round"])  if a["pick_round"]  else "?"
            slot  = a.get("pick_slot")
            if slot:
                name = f"{s} Pick {r}.{str(slot).zfill(2)}"
            else:
                order = a["pick_order"] or ""
                name  = f"{s} Round {r}" + (f" ({order})" if order else "")
            return {"type": "pick", "name": name, "is_focus": False}

        result = []
        for r in trade_rows:
            tid   = r["id"]
            sides = assets_by_trade.get(tid, {"a": [], "b": []})
            side_a = [describe(a) for a in sides["a"]]
            side_b = [describe(a) for a in sides["b"]]
            if not side_a or not side_b:
                continue
            trade_date = None
            if r["created_at"]:
                try:
                    trade_date = r["created_at"].strftime("%-m/%-d/%y")
                except Exception:
                    trade_date = str(r["created_at"])[:10]
            result.append({
                "trade_id":    r["transaction_id"],
                "date":        trade_date,
                "is_superflex": r["is_superflex"],
                "num_teams":   r["num_teams"],
                "side_a":      side_a,
                "side_b":      side_b,
            })

        total_pages = max(1, (total + limit - 1) // limit)
        return jsonify({
            "trades":      result,
            "total":       total,
            "page":        page,
            "total_pages": total_pages,
            "has_prev":    page > 1,
            "has_next":    page < total_pages,
        })

    except Exception:
        logger.exception("[player-trades] error")
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
                except Exception: trade_date = str(r["created_at"])[:10]
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
                try:
                    from data_building.trade_intel.analytics import run_trade_model
                    model_result = run_trade_model()
                    logger.info("[trade-intel] Trade pattern model: %s", model_result)
                except Exception:
                    logger.exception("[trade-intel] Trade pattern model training failed (non-fatal)")
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

    user_id = session.get("viewer_username") or None
    if not has_premium_access(user_id, league_id, platform):
        return jsonify({"paywall": True, "error": "Premium required"}), 403

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


@app.route("/api/trade-intel/player-packages/<player_id>")
@limiter.limit("30 per minute")
def api_trade_intel_player_packages(player_id: str):
    """
    GET handler called by the "Find packages" button on the player modal.

    Returns value-matched packages from the viewer's roster (if league context
    is provided) and real historical trade packages from the trade DB.

    Query params: season, league_type, league_id, platform, viewer_roster_id
    """
    season           = int(request.args.get("season") or datetime.now().year)
    league_type      = str(request.args.get("league_type") or "1qb").strip().lower()
    league_id        = str(request.args.get("league_id") or "").strip()
    platform         = str(request.args.get("platform") or "sleeper").strip()
    viewer_roster_id = str(request.args.get("viewer_roster_id") or "").strip()
    _untouchable_raw = str(request.args.get("untouchable_ids") or "").strip()
    untouchable_ids  = set(x.strip() for x in _untouchable_raw.split(",") if x.strip())

    user_id = session.get("viewer_username")
    if not has_premium_access(user_id, league_id, platform):
        return jsonify({"error": "Premium required"}), 403

    try:
        val_key = "sf_value" if league_type == "sf" else "value"
        # Always prefer live DB values; fall back to cached JSON only if DB unavailable
        value_table = []
        try:
            from dashboard_services.player_value_history import load_current_values_from_db as _load_db_vals
            value_table = _load_db_vals() or []
        except Exception:
            pass
        if not value_table:
            from utils.utils import load_model_value_table
            value_table = load_model_value_table() or []

        _use_sf_ranks = (val_key == "sf_value")
        values_by_id: dict = {}
        for p in value_table:
            pid = str(p.get("id") or "")
            if pid:
                values_by_id[pid] = {
                    "name":              p.get("name", ""),
                    "position":          str(p.get("position") or "").upper(),
                    "value":             float(p.get(val_key) or p.get("value") or 0),
                    "sf_value":          float(p.get("sf_value") or p.get("value") or 0),
                    "pos_rank":          int((p.get("sf_pos_rank") if _use_sf_ranks else p.get("pos_rank")) or 99),
                    "pos_rank_label":    (p.get("sf_pos_rank_label") if _use_sf_ranks else p.get("pos_rank_label")) or "",
                    "team":              p.get("team") or "",
                    "age":               p.get("age"),
                    "pos_rank_change_7d": p.get("pos_rank_change_7d"),
                }

        def _compute_profile(info: dict):
            pos = info.get("position", "")
            if pos in ("PICK", "K", "DEF"):
                return None
            age = float(info.get("age") or 0)
            if age <= 0:
                return None
            rank   = int(info.get("pos_rank") or 99)
            change = info.get("pos_rank_change_7d")  # negative = improving (rank went up)

            # Age bracket
            if age <= 24:
                bracket = "young"
            elif age <= 28:
                bracket = "prime"
            else:
                bracket = "vet"

            # Trend: prefer rank_change if available, otherwise infer from pos_rank
            if change is not None:
                try:
                    c = float(change)
                    if c <= -3:
                        trend = "rising"
                    elif c >= 3:
                        trend = "falling"
                    else:
                        trend = "stable"
                except (TypeError, ValueError):
                    trend = None

            if change is None or trend is None:
                # Infer from absolute rank (higher rank = still performing)
                if bracket == "young":
                    trend = "rising" if rank <= 5 else ("stable" if rank <= 15 else "falling")
                elif bracket == "prime":
                    trend = "rising" if rank <= 3 else ("stable" if rank <= 10 else "falling")
                else:  # vet
                    trend = "rising" if rank <= 3 else ("stable" if rank <= 7 else "falling")

            return f"{bracket}-{trend}"

        target_info = values_by_id.get(str(player_id))
        if not target_info:
            return jsonify({"error": "Player not found"}), 404

        focus_value  = round(target_info["value"], 1)
        player_name  = target_info["name"]

        # ── Viewer roster context (optional) ──────────────────────────────
        viewer_players: list[dict] = []
        viewer_picks:   list[dict] = []
        rosters: list = []
        ctx = {}
        if league_id and viewer_roster_id:
            try:
                ctx = get_league_ctx_from_cache(platform, league_id, season) or {}
                rosters = ctx.get("rosters") or []
                viewer_roster_obj = next(
                    (r for r in rosters if str(r.get("roster_id")) == viewer_roster_id), None
                )
                if viewer_roster_obj:
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
                    if untouchable_ids:
                        viewer_players = [p for p in viewer_players if p["player_id"] not in untouchable_ids]
                    # Use draft_ended=True if the fantasy rookie draft for this season
                    # is already complete — those picks no longer exist as assets.
                    try:
                        _drafts_for_check = ctx.get("drafts") or []
                        _latest_d = get_most_recent_valid_draft_for_season(_drafts_for_check, season)
                        _fantasy_draft_done = (
                            bool(_latest_d)
                            and str((_latest_d or {}).get("status") or "").lower() == "complete"
                            and int(((_latest_d or {}).get("settings") or {}).get("rounds") or 99) <= 5
                        )
                    except Exception:
                        _fantasy_draft_done = False
                    try:
                        from dashboard_services.service import build_picks_by_roster as _bpbr
                        _fresh_pbr = _bpbr(
                            num_future_seasons=3,
                            league=ctx.get("league") or {},
                            rosters=rosters,
                            traded=ctx.get("traded") or [],
                            draft_ended=_fantasy_draft_done,
                        )
                    except Exception as _pbr_err:
                        logger.warning("[trade-intel] build_picks_by_roster failed: %s", _pbr_err)
                        _fresh_pbr = ctx.get("picks_by_roster") or {}
                    raw_picks = _fresh_pbr.get(viewer_roster_id) or []
                    pick_val_lookup = {
                        str(p.get("id") or ""): float(p.get("value") or 0)
                        for p in value_table
                        if str(p.get("position") or "").upper() == "PICK"
                    }
                    # Slot map for current season: {original_roster_id -> slot_number}
                    _slot_map: dict = {}
                    try:
                        _slot_map = build_historical_pick_slot_map(
                            platform, league_id, season, season - 1
                        ) or {}
                    except Exception:
                        pass

                    def _pick_bucket(slot):
                        if not slot:
                            return ""
                        return "Early" if slot <= 4 else ("Mid" if slot <= 8 else "Late")

                    for pk in raw_picks:
                        yr  = int(pk.get("season") or season)
                        rnd = int(pk.get("round") or 4)
                        if yr > season + 1:
                            continue
                        orig = pk.get("original_owner")
                        # Slot only meaningful for the current season
                        slot = _slot_map.get(int(orig)) if (yr == season and orig) else None
                        suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(rnd, f"{rnd}th")
                        if slot:
                            pk_name = f"{yr} {rnd}.{slot:02d}"
                        else:
                            pk_name = f"{yr} {suffix}"
                        if slot:
                            pval = (
                                pick_val_lookup.get(f"{yr}_{rnd}_{slot:02d}")
                                or pick_val_lookup.get(f"{yr}_{rnd}", 0)
                            )
                        else:
                            pval = pick_val_lookup.get(f"{yr}_{rnd}", 0)
                        pval = pval or (220 if rnd == 1 else 130 if rnd == 2 else 70)
                        viewer_picks.append({
                            "name": pk_name, "value": pval, "is_pick": True,
                            "pick_season": yr, "pick_round": rnd,
                            "pick_order": slot,
                        })
            except Exception as _ctx_err:
                logger.warning("[trade-intel-picks] ctx error: %s", _ctx_err)

        # ── Archetype helpers ──────────────────────────────────────────────
        def _age_bracket(age) -> str:
            try:
                a = float(age)
            except (TypeError, ValueError):
                return "Prime"
            if a <= 24:
                return "Young"
            if a <= 28:
                return "Prime"
            return "Vet"

        def _asset_archetype_label(asset: dict) -> str:
            if asset.get("is_pick") or asset.get("type") == "pick":
                rnd = asset.get("pick_round") or asset.get("round") or ""
                try:
                    suffix = {1: "R1", 2: "R2", 3: "R3"}.get(int(rnd), f"R{rnd}")
                except (TypeError, ValueError):
                    suffix = None
                return f"PICK:{suffix}" if suffix else "PICK"
            info = values_by_id.get(asset.get("player_id") or asset.get("id") or "")
            if not info:
                return "?"
            pos   = info.get("position", "?")
            val   = info.get("value", 0.0)
            tier  = _asset_tier(val)
            return f"{pos}-T{tier}"

        THROW_IN_VALUE_THRESHOLD = 150.0

        def _pattern_sigs(assets: list) -> tuple:
            labeled = [
                (_asset_archetype_label(a), float(a.get("value") or a.get("send_value") or 0))
                for a in assets
            ]
            if len(labeled) >= 2:
                max_val = max(v for _, v in labeled)
                core    = [(lbl, v) for lbl, v in labeled if v >= THROW_IN_VALUE_THRESHOLD or v >= max_val * 0.35]
                throwin = [(lbl, v) for lbl, v in labeled if (lbl, v) not in core]
            else:
                core    = labeled
                throwin = []
            core.sort(key=lambda x: -x[1])
            return " + ".join(lbl for lbl, _ in core), " + ".join(lbl for lbl, _ in throwin) if throwin else ""

        # ── League context for DB queries ─────────────────────────────────
        roster_positions = ctx.get("roster_positions") or []
        _rp_list = [str(s).upper() for s in (roster_positions if isinstance(roster_positions, list) else [])]
        _is_sf   = (league_type == "sf") or any(s in {"SUPER_FLEX", "SFLEX"} for s in _rp_list)
        num_teams = len(ctx.get("rosters") or []) or 12

        # Resync val_key with the actual league format detected from roster_positions.
        # The initial val_key was set from the URL param; the real league may differ
        # (e.g. league has SUPER_FLEX but URL says league_type=1qb).
        _correct_val_key = "sf_value" if _is_sf else "value"
        if _correct_val_key != val_key and ctx:
            val_key = _correct_val_key
            _rsf = _is_sf
            _rebuilt: dict = {}
            for _p in value_table:
                _pid = str(_p.get("id") or "")
                if _pid:
                    _rebuilt[_pid] = {
                        "name":               _p.get("name", ""),
                        "position":           str(_p.get("position") or "").upper(),
                        "value":              float(_p.get(val_key) or _p.get("value") or 0),
                        "sf_value":           float(_p.get("sf_value") or _p.get("value") or 0),
                        "pos_rank":           int((_p.get("sf_pos_rank") if _rsf else _p.get("pos_rank")) or 99),
                        "pos_rank_label":     (_p.get("sf_pos_rank_label") if _rsf else _p.get("pos_rank_label")) or "",
                        "team":               _p.get("team") or "",
                        "age":                _p.get("age"),
                        "pos_rank_change_7d": _p.get("pos_rank_change_7d"),
                    }
            values_by_id = _rebuilt
            # Rebuild viewer_players with updated values
            _rvo = next((r for r in rosters if str(r.get("roster_id")) == viewer_roster_id), None)
            if _rvo:
                viewer_players = sorted(
                    [
                        {
                            "player_id":      _pid,
                            "name":           values_by_id[_pid]["name"],
                            "position":       values_by_id[_pid]["position"],
                            "value":          values_by_id[_pid]["value"],
                            "pos_rank_label": values_by_id[_pid]["pos_rank_label"],
                        }
                        for _pid in [str(_pp) for _pp in (_rvo.get("players") or [])]
                        if _pid in values_by_id and values_by_id[_pid]["value"] >= 50
                    ],
                    key=lambda x: x["value"],
                    reverse=True,
                )
                if untouchable_ids:
                    viewer_players = [p for p in viewer_players if p["player_id"] not in untouchable_ids]
            # Update focus_value and target_info with corrected values
            target_info = values_by_id.get(str(player_id))
            if target_info:
                focus_value = round(target_info["value"], 1)

        # ── ML model: primary package suggestions ─────────────────────────
        ml_pkgs: list = []
        _model_stale_days = None
        try:
            from data_building.trade_intel.trade_pattern_model import (
                load_model       as _tm_load,
                suggest_packages as _tm_suggest,
            )
            _tm_model = _tm_load()
            if _tm_model:
                _focus_pos = (values_by_id.get(str(player_id)) or {}).get("position", "WR")
                ml_pkgs = _tm_suggest(
                    model             = _tm_model,
                    target_player_id  = str(player_id),
                    target_pos        = _focus_pos,
                    target_value      = float(focus_value or 0),
                    viewer_players    = viewer_players,
                    viewer_picks      = viewer_picks,
                    values_by_id      = values_by_id,
                    n                 = 10,
                    num_teams         = num_teams,
                )
                logger.info("[api-trade-intel-player-packages] ML model: %d packages", len(ml_pkgs))
                _model_stale_days = _tm_model.get("model_stale_days")
        except Exception as _ml_err:
            logger.warning("[api-trade-intel-player-packages] ML model error: %s", _ml_err)

        # ── Rule-based DB result: archetype stats + fallback packages ─────
        real_result = _real_trade_packages_for_target(
            target_player_id=str(player_id),
            is_sf=_is_sf,
            num_teams=num_teams,
            viewer_players=viewer_players,
            viewer_picks=viewer_picks,
            values_by_id=values_by_id,
            focus_value=float(focus_value or 0),
        )

        # ML is primary; supplement with rule-based packages before value fallback
        primary_pkgs   = list(ml_pkgs) if ml_pkgs else list(real_result["packages"])
        package_source = "ml" if ml_pkgs else "rule"

        # Fill remaining slots (up to 5) with rule-based packages not already covered
        if ml_pkgs and len(primary_pkgs) < 5:
            _used_shapes = {
                frozenset(
                    str(a.get("player_id") or a.get("name") or "")
                    for a in pkg.get("send", [])
                )
                for pkg in primary_pkgs
            }
            for rule_pkg in real_result.get("packages") or []:
                if len(primary_pkgs) >= 5:
                    break
                rule_key = frozenset(
                    str(a.get("player_id") or a.get("name") or "")
                    for a in rule_pkg.get("send", [])
                )
                if rule_key not in _used_shapes:
                    _used_shapes.add(rule_key)
                    primary_pkgs.append(rule_pkg)

        # ── Value-based fallback: always fill to at least 5 packages ─────
        # Combines viewer players (and picks) whose total value sits within
        # [85%, 125%] of focus_value. Works even with no trade history.
        if viewer_players and len(primary_pkgs) < 5:
            _fv      = float(focus_value or 0)
            _lo, _hi = _fv * 0.82, _fv * 1.25
            _used_sets = {
                frozenset(
                    str(a.get("player_id") or a.get("name") or "")
                    for a in pkg.get("send", [])
                )
                for pkg in primary_pkgs
            }

            def _vb_packages(target: float, players: list, picks: list, limit: int) -> list:
                out: list = []
                sorted_p = sorted(players, key=lambda p: -float(p.get("value") or 0))
                sorted_k = sorted(picks,   key=lambda k:  int(k.get("pick_round") or 3))
                lo, hi   = target * 0.82, target * 1.25

                def _add(assets):
                    key = frozenset(
                        str(a.get("player_id") or a.get("name") or "") for a in assets
                    )
                    if key not in _used_sets:
                        _used_sets.add(key)
                        out.append(assets)
                        return True
                    return False

                # 1-player — at most 2 per position for variety
                pos_count: dict = {}
                for p in sorted_p:
                    if len(out) >= limit: break
                    if lo <= float(p.get("value") or 0) <= hi:
                        pos = str(p.get("position") or "")
                        if pos_count.get(pos, 0) < 2:
                            if _add([p]):
                                pos_count[pos] = pos_count.get(pos, 0) + 1

                # Single pick
                for pk in sorted_k:
                    if len(out) >= limit: break
                    if lo <= float(pk.get("value") or 0) <= hi:
                        _add([pk])

                # 2 picks
                for i, pk1 in enumerate(sorted_k):
                    if len(out) >= limit: break
                    v1 = float(pk1.get("value") or 0)
                    for pk2 in sorted_k[i + 1:]:
                        if len(out) >= limit: break
                        if lo <= v1 + float(pk2.get("value") or 0) <= hi:
                            _add([pk1, pk2])
                            break

                # 1 player + 1 pick
                for p in sorted_p:
                    if len(out) >= limit: break
                    v = float(p.get("value") or 0)
                    for pk in sorted_k:
                        if len(out) >= limit: break
                        if lo <= v + float(pk.get("value") or 0) <= hi:
                            _add([p, pk])
                            break

                # Throw-in: player slightly under floor + cheap sweetener (pick or player)
                # Main piece in [65%, 82%) of target — below floor on its own
                for p in sorted_p:
                    if len(out) >= limit: break
                    v = float(p.get("value") or 0)
                    if not (target * 0.65 <= v < lo): continue
                    needed = lo - v
                    # Try cheapest pick first
                    for pk in sorted_k:
                        pv = float(pk.get("value") or 0)
                        if lo <= v + pv <= hi:
                            _add([p, pk]); break
                    if len(out) >= limit: break
                    # Try cheapest player
                    for p2 in reversed(sorted_p):
                        if p2 is p: continue
                        p2v = float(p2.get("value") or 0)
                        if p2v < needed * 0.5: continue
                        if lo <= v + p2v <= hi:
                            _add([p, p2]); break

                # 2-player — prefer different positions
                for i, p1 in enumerate(sorted_p):
                    if len(out) >= limit: break
                    v1 = float(p1.get("value") or 0)
                    if v1 >= hi: continue
                    for p2 in sorted_p[i + 1:]:
                        if len(out) >= limit: break
                        if p2.get("position") == p1.get("position"):
                            continue  # prefer positional variety
                        if lo <= v1 + float(p2.get("value") or 0) <= hi:
                            _add([p1, p2])
                            break
                # 2-player same position (only if limit not met)
                for i, p1 in enumerate(sorted_p):
                    if len(out) >= limit: break
                    v1 = float(p1.get("value") or 0)
                    if v1 >= hi: continue
                    for p2 in sorted_p[i + 1:]:
                        if len(out) >= limit: break
                        if lo <= v1 + float(p2.get("value") or 0) <= hi:
                            _add([p1, p2])
                            break

                return out

            need = 5 - len(primary_pkgs)
            vb_asset_lists = _vb_packages(_fv, viewer_players, viewer_picks, need)
            for assets in vb_asset_lists:
                send = []
                for a in assets:
                    if a.get("is_pick"):
                        send.append({
                            "name":       a.get("name", ""),
                            "value":      float(a.get("value") or 0),
                            "send_value": float(a.get("value") or 0),
                            "is_pick":    True,
                            "pick_round": a.get("pick_round"),
                            "pick_season": a.get("pick_season"),
                        })
                    else:
                        send.append({
                            "player_id":  str(a.get("player_id") or ""),
                            "name":       a.get("name", ""),
                            "position":   a.get("position", ""),
                            "value":      float(a.get("value") or 0),
                            "send_value": float(a.get("value") or 0),
                            "is_pick":    False,
                        })
                primary_pkgs.append({
                    "send":             send,
                    "send_value":       round(sum(float(x.get("value") or 0) for x in send), 1),
                    "trades_like_this": 0,
                    "pattern_source":   "value",
                    "sig":              [],
                })

        # ── Shared enrichment ─────────────────────────────────────────────
        def _sig_to_archetype(sig_str: str) -> tuple:
            import ast as _ast
            try:
                parts = list(_ast.literal_eval(sig_str))
            except Exception:
                return "", ""
            labeled = []
            for part in parts:
                kind, *rest = part.split(":")
                if kind == "P" and len(rest) >= 2:
                    pos  = rest[0]
                    tier = rest[1]
                    labeled.append(f"{pos}-{tier}")
                elif kind == "K" and rest:
                    rnd_str  = rest[0]
                    slot_str = rest[1] if len(rest) > 1 else ""
                    rnd_lbl  = {"1": "R1", "2": "R2", "3": "R3"}.get(rnd_str, f"R{rnd_str}")
                    labeled.append(f"PICK:{rnd_lbl}:{slot_str}" if slot_str else f"PICK:{rnd_lbl}")
            players = sorted(lbl for lbl in labeled if not lbl.startswith("PICK"))
            picks   = sorted(lbl for lbl in labeled if lbl.startswith("PICK"))
            return " + ".join(players + picks), ""

        def _likely_takers(sent_positions):
            if not rosters or not sent_positions:
                return []
            need = {}
            names = {}
            for roster in rosters:
                rid = str(roster.get("roster_id", ""))
                if rid == viewer_roster_id:
                    continue
                owner = roster.get("display_name") or roster.get("owner_name") or f"Team {rid}"
                names[rid] = owner
                pos_vals: dict = {}
                for pid in (roster.get("players") or []):
                    info = values_by_id.get(str(pid))
                    if info:
                        p_pos = info.get("position", "")
                        pos_vals.setdefault(p_pos, []).append(float(info.get("value", 0)))
                score = 0
                for pos in set(sent_positions):
                    vals = sorted(pos_vals.get(pos, []), reverse=True)
                    if not vals:
                        score += 3
                    elif len(vals) < 2 or (len(vals) >= 2 and vals[1] < 150):
                        score += 2
                    elif vals[0] < 350:
                        score += 1
                need[rid] = score
            top = sorted(need.items(), key=lambda x: -x[1])[:3]
            return [names[rid] for rid, s in top if s > 0]

        # Pre-build a map of player_id → roster for ownership lookup
        _player_owner_map: dict = {}
        for _r in (rosters or []):
            for _pid in (_r.get("players") or []):
                _player_owner_map[str(_pid)] = _r

        from dashboard_services.ai.context_builders import (
            calculate_roster_grade as _calc_grade,
            detect_team_direction as _detect_direction,
        )

        def _roster_grade(roster: dict) -> dict:
            """Return calculate_roster_grade output for any roster,
            using the same logic as the Teams page."""
            rid = str(roster.get("roster_id", ""))
            flat = []
            for pid in (roster.get("players") or []):
                info = values_by_id.get(str(pid))
                if info and info.get("position") in ("QB", "RB", "WR", "TE"):
                    flat.append({
                        "position": info["position"],
                        "value":    float(info.get("value") or 0),
                        "age":      info.get("age"),
                    })
            flat.sort(key=lambda x: x["value"], reverse=True)
            picks = (_fresh_pbr or {}).get(rid, [])
            return _calc_grade(flat, picks)

        def _acceptance_prob(pkg: dict, total_trades: int) -> int:
            """
            Estimate acceptance probability (0–100) that the owner of the
            focus player accepts this package.

            Factors:
              1. Value balance      – from receiver's POV (overpay = high)
              2. Historical rate    – trades_like_this / total_real_trades
              3. Positional index   – tier of the receiver's best player at
                                      each position being sent, not just depth
              4. Win/rebuild window – rebuild teams prize youth + picks;
                                      win-now teams prize proven vets
            """
            # 1. Value base (receiver's perspective)
            grade = pkg.get("value_grade", "fair")
            base = {"steal": 14, "fair": 46, "overpay": 66, "big_overpay": 82}.get(grade, 46)

            # 2. Historical frequency boost
            count = pkg.get("trades_like_this", 0)
            freq  = count / max(total_trades, 1)
            freq_boost = min(int(freq * 60), 12)

            owner_roster = _player_owner_map.get(str(player_id))
            need_adj  = 0
            window_adj = 0

            if owner_roster:
                # Build positional value index for the receiving team
                pos_vals: dict[str, list[float]] = {}
                for pid in (owner_roster.get("players") or []):
                    info = values_by_id.get(str(pid))
                    if info and info.get("position") in ("QB", "RB", "WR", "TE"):
                        p = info["position"]
                        pos_vals.setdefault(p, []).append(float(info.get("value") or 0))
                for p in pos_vals:
                    pos_vals[p].sort(reverse=True)

                # 3. Positional index: score each sent position against receiver's depth
                sent_players = [a for a in pkg.get("send", []) if not a.get("is_pick")]
                for a in sent_players:
                    pos  = a.get("position", "")
                    aval = float(a.get("value") or a.get("send_value") or 0)
                    if not pos:
                        continue
                    existing = pos_vals.get(pos, [])
                    starter_val = existing[0] if existing else 0
                    depth_val   = existing[1] if len(existing) > 1 else 0

                    if not existing:
                        need_adj += 12              # hole at this position
                    elif starter_val < 200:
                        need_adj += 9               # barely-rostered starter
                    elif starter_val < 400:
                        need_adj += 5               # mediocre starter
                        if aval > starter_val * 1.1:
                            need_adj += 3           # upgrade on their best
                    elif depth_val < 150:
                        need_adj += 2               # strong starter, thin depth
                    else:
                        need_adj -= 3               # stacked at this pos — harder sell

                need_adj = max(-10, min(need_adj, 18))

                # 4. Win/rebuild window — use same grade as Teams page
                grade_data = _roster_grade(owner_roster)
                win_window = grade_data.get("win_window", "")
                # Map Teams-page labels to rebuild / win_now / competitive
                if win_window in ("Full Rebuild", "Retooling"):
                    window = "rebuild"
                elif win_window in ("Win-Now Window", "Aging Contender", "Contender Window"):
                    window = "win_now"
                else:
                    window = "competitive"

                sent_picks = [a for a in pkg.get("send", []) if a.get("is_pick")]
                sent_ages  = []
                sent_vals  = []
                for a in sent_players:
                    info = values_by_id.get(a.get("player_id") or "")
                    if info:
                        sent_ages.append(float(info.get("age") or 25))
                        sent_vals.append(float(info.get("value") or 0))

                avg_sent_age = sum(sent_ages) / len(sent_ages) if sent_ages else 25
                avg_sent_val = sum(sent_vals) / len(sent_vals) if sent_vals else 0
                n_picks      = len(sent_picks)

                if window == "rebuild":
                    # Rebuilding teams want youth and picks
                    youth_bonus  = max(0, int((26 - avg_sent_age) * 2.5))  # young = good
                    pick_bonus   = n_picks * 6
                    upside_bonus = min(int(avg_sent_val / 60), 8) if avg_sent_age < 24 else 0
                    window_adj   = min(youth_bonus + pick_bonus + upside_bonus, 20)
                elif window == "win_now":
                    # Win-now teams want proven contributors now
                    vet_bonus    = max(0, int((avg_sent_age - 23) * 2))
                    tier_bonus   = min(int(avg_sent_val / 100), 8)
                    pick_penalty = n_picks * -4  # picks less useful when chasing now
                    window_adj   = max(-15, min(vet_bonus + tier_bonus + pick_penalty, 12))
                # competitive: neutral (window_adj stays 0)

            prob = base + freq_boost + need_adj + window_adj
            return min(93, max(8, round(prob)))

        def _enrich_pkg(pkg: dict) -> None:
            for asset in pkg["send"]:
                if not asset.get("is_pick"):
                    info = values_by_id.get(asset.get("player_id") or "")
                    if info:
                        asset.update({
                            "id":             asset["player_id"],
                            "position":       info["position"],
                            "team":           info.get("team", ""),
                            "sf_value":       round(info.get("sf_value", info["value"]), 1),
                            "pos_rank_label": info["pos_rank_label"],
                        })
            raw_sig = pkg.get("sig")
            if raw_sig:
                core_sig, throw_sig = _sig_to_archetype(str(tuple(raw_sig)))
            else:
                core_sig, throw_sig = _pattern_sigs(pkg["send"])
            pkg["pattern_sig"]  = core_sig
            pkg["throw_in_sig"] = throw_sig
            send_total = sum(float(a.get("value") or a.get("send_value") or 0) for a in pkg["send"])
            pkg["send_value"]   = round(send_total, 1)
            pkg["value_delta"]  = round(send_total - focus_value, 1)
            _pct = (send_total - focus_value) / max(focus_value, 1) * 100
            if _pct <= -10:
                pkg["value_grade"] = "steal"
            elif _pct <= 8:
                pkg["value_grade"] = "fair"
            elif _pct <= 22:
                pkg["value_grade"] = "overpay"
            else:
                pkg["value_grade"] = "big_overpay"

        for pkg in primary_pkgs:
            _enrich_pkg(pkg)

        # Drop packages where the viewer is sending more than 2× the target's value.
        # Real trades include extreme overpays — those are not useful suggestions.
        _max_send = (focus_value or 1) * 1.3
        primary_pkgs = [p for p in primary_pkgs if p.get("send_value", 0) <= _max_send]

        _total_real = real_result.get("total_real_trades") or 1

        # Grade the receiving team once (same data as Teams page)
        _owner_roster = _player_owner_map.get(str(player_id))
        _receiver_grade = _roster_grade(_owner_roster) if _owner_roster else {}
        _receiver_win_window = _receiver_grade.get("win_window", "")

        for pkg in primary_pkgs:
            sent_pos = [a.get("position") for a in pkg.get("send", []) if not a.get("is_pick") and a.get("position")]
            pkg["likely_takers"]   = _likely_takers(sent_pos)
            pkg["acceptance_prob"] = _acceptance_prob(pkg, _total_real)

        # ── Archetype patterns — always from real DB sig_counts ───────────
        total_trade_count = real_result["total_real_trades"] or 1
        from collections import defaultdict as _dfd
        merged: dict = _dfd(int)
        for sig_str, cnt in (real_result.get("sig_counts") or {}).items():
            core_sig, throw_sig = _sig_to_archetype(sig_str)
            if core_sig:
                merged[f"{core_sig}|{throw_sig}"] += cnt

        # Full pct lookup across all patterns (not just top 6) so backfilled
        # entries can still show a real percentage when the DB has data.
        pct_lookup = {
            canon.split("|")[0]: round(cnt / total_trade_count * 100)
            for canon, cnt in merged.items()
        }

        pkg_sigs = {pkg.get("pattern_sig") or "" for pkg in primary_pkgs}

        archetype_patterns = sorted(
            [
                {
                    "pattern_sig":    canon.split("|")[0],
                    "throw_in_sig":   canon.split("|")[1],
                    "count":          cnt,
                    "pct":            round(cnt / total_trade_count * 100),
                    "fits_your_team": canon.split("|")[0] in pkg_sigs,
                }
                for canon, cnt in merged.items()
                if cnt >= 2
            ],
            key=lambda x: -x["count"],
        )[:4]

        # Backfill any suggestion patterns not already in the top 4.
        existing_sigs = {ap["pattern_sig"] for ap in archetype_patterns}
        for pkg in primary_pkgs:
            sig = pkg.get("pattern_sig") or ""
            if sig and sig not in existing_sigs:
                archetype_patterns.append({
                    "pattern_sig":    sig,
                    "throw_in_sig":   pkg.get("throw_in_sig") or "",
                    "count":          pkg.get("trades_like_this") or 1,
                    "pct":            pct_lookup.get(sig, 0),
                    "fits_your_team": True,
                })
                existing_sigs.add(sig)

        # Sort suggestions by archetype popularity so the most common pattern
        # surfaces first (e.g. single-pick before two-pick, etc.)
        if archetype_patterns and primary_pkgs:
            arch_rank = {ap["pattern_sig"]: i for i, ap in enumerate(archetype_patterns)}
            def _pkg_rank(pkg):
                sig = pkg.get("pattern_sig") or ""
                return (arch_rank.get(sig, len(archetype_patterns)), -pkg.get("trades_like_this", 0))
            primary_pkgs.sort(key=_pkg_rank)

        return jsonify({
            "player_name":          player_name,
            "focus_value":          focus_value,
            "packages":             [],
            "total_packages":       0,
            "real_packages":        primary_pkgs,
            "total_real_trades":    real_result["total_real_trades"],
            "archetype_patterns":   archetype_patterns,
            "package_source":       package_source,
            "model_stale_days":     _model_stale_days,
            "query_window_days":    730,
            "receiver_win_window":  _receiver_win_window,
        })

    except Exception as e:
        logger.exception("[api-trade-intel-player-packages] %s", e)
        return jsonify({"error": str(e)}), 500


def _real_trade_packages_for_target(
    target_player_id: str,
    is_sf: bool,
    num_teams: int,
    viewer_players: list[dict],
    viewer_picks: list[dict],
    values_by_id: dict,
    max_packages: int = 5,
    focus_value: float = 0,
) -> dict:
    """
    Find real trades where target_player_id was acquired in comparable leagues,
    then match the sent-asset patterns against the viewer's roster.

    Returns {"packages": [...], "total_real_trades": N, "sig_counts": {...}}
    Each package has the same shape as value-based packages plus "trades_like_this".
    sig_counts is the full pattern→trade_ids dict (used for archetype computation).
    """
    from collections import defaultdict

    # Build pick value lookup from values_by_id (position=="PICK", id like "2026_1")
    _pick_val_map = {
        pid: info["value"]
        for pid, info in values_by_id.items()
        if info.get("position") == "PICK" and info.get("value", 0) > 0
    }

    def _pick_val(rnd: int, season=None) -> float:
        if season:
            v = _pick_val_map.get(f"{season}_{rnd}") or _pick_val_map.get(f"{season}_{rnd:02d}")
            if v:
                return float(v)
        # fallback: find any key matching round
        for k, v in _pick_val_map.items():
            if k.endswith(f"_{rnd}") or k.endswith(f"_{rnd:02d}"):
                return float(v)
        return 450.0 if rnd == 1 else 175.0 if rnd == 2 else 70.0

    # Rough value by tier for pre-filtering low-value patterns
    _TIER_VAL_EST = {1: 1300, 2: 900, 3: 640, 4: 440, 5: 310, 6: 200, 7: 110, 8: 55, 9: 20}

    def _sig_estimate_value(sig: tuple) -> float:
        total = 0.0
        for part in sig:
            kind, *rest = part.split(":")
            if kind == "P" and rest:
                tier_str = rest[1] if len(rest) > 1 else "T5"
                req_tier = int(tier_str[1:]) if tier_str.startswith("T") else 5
                total += _TIER_VAL_EST.get(req_tier, 200)
            elif kind == "K" and rest:
                rnd = int(rest[0]) if rest[0].isdigit() else 3
                total += _pick_val(rnd)
        return total

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
                      AND t.created_at > NOW() - INTERVAL '730 days'
                    LIMIT 500
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
                (target_player_id, is_sf, num_teams - 4, num_teams + 4),
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

    # Build precise signature: P:{pos}:T{tier}:{bracket}  K:{round}:{slot_bucket}
    def _pick_slot_bucket(order) -> str:
        if order is None:
            return ""
        try:
            o = int(order)
        except (TypeError, ValueError):
            return ""
        return "Early" if o <= 4 else ("Mid" if o <= 8 else "Late")

    def _player_age_bracket(info: dict) -> str:
        try:
            age = float(info.get("age") or 0)
        except (TypeError, ValueError):
            age = 0
        if age <= 0:
            return "Unk"
        return "Young" if age <= 24 else ("Prime" if age <= 28 else "Vet")

    def _sig(assets: list[dict]) -> Optional[tuple]:
        parts = []
        for a in assets:
            if a["asset_type"] == "player" and a["sent_player_id"]:
                info = values_by_id.get(str(a["sent_player_id"]))
                if not info:
                    continue
                pos    = info["position"]
                tier   = _asset_tier(float(info.get("value") or 0))
                bracket = _player_age_bracket(info)
                parts.append(f"P:{pos}:T{tier}:{bracket}")
            elif a["asset_type"] == "pick" and a["pick_round"]:
                bucket = _pick_slot_bucket(a.get("pick_order"))
                parts.append(f"K:{a['pick_round']}:{bucket}")
        return tuple(sorted(parts)) if parts else None

    sig_counts: dict = defaultdict(list)
    for trade_id, assets in trade_pkgs.items():
        s = _sig(assets)
        if s:
            sig_counts[s].append(trade_id)

    # Viewer helpers
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

    # Pre-filter sig_counts: drop patterns whose total estimated value is < 55% of target
    _value_floor = focus_value * 0.85 if focus_value > 0 else 0
    if _value_floor > 0:
        sig_counts = {k: v for k, v in sig_counts.items()
                      if _sig_estimate_value(k) >= _value_floor}

    # Value range for matching packages against the viewer's roster
    max_send_value = focus_value * 1.30 if focus_value > 0 else float("inf")
    min_send_value = focus_value * 0.65 if focus_value > 0 else 0.0
    target_value   = focus_value  # alias used in anchor / tier checks below

    result_packages = []
    used_pids: set = set()
    fallback_packages = []

    for sig, trade_ids in sorted(sig_counts.items(), key=lambda x: -len(x[1])):
        trades_like_this = len(trade_ids)
        matched: list[dict] = []
        temp_used: set = set()
        ok = True

        for part in sig:
            kind, *rest = part.split(":")
            if kind == "P":
                pos, tier_str, bracket = rest[0], rest[1], rest[2]
                req_tier = int(tier_str[1:]) if tier_str.startswith("T") else 5
                # Find viewer player matching pos and within ±1 tier
                candidates = [
                    vp for vp in vp_by_pos.get(pos, [])
                    if vp["player_id"] not in used_pids
                    and vp["player_id"] not in temp_used
                    and abs(_asset_tier(vp["value"]) - req_tier) <= 1
                ]
                if not candidates:
                    ok = False
                    break
                # Pick closest tier match, break ties by closest age bracket
                best = min(candidates, key=lambda p: (
                    abs(_asset_tier(p["value"]) - req_tier),
                    abs(float(values_by_id.get(p.get("player_id",""), {}).get("age") or 99) - 26)
                ))
                matched.append(best)
                temp_used.add(best["player_id"])
            elif kind == "K":
                rnd = int(rest[0])
                available = vk_by_round.get(rnd, [])
                if not available:
                    ok = False
                    break
                matched.append(available[0])

        if not ok or not matched:
            # Fallback: reference players/picks describing the pattern
            fallback_assets = []
            for part in sig:
                kind, *rest = part.split(":")
                if kind == "P":
                    pos, tier_str, bracket = rest[0], rest[1], rest[2]
                    req_tier = int(tier_str[1:]) if tier_str.startswith("T") else 5
                    candidates = [
                        {"player_id": pid, "name": info["name"], "position": pos,
                         "value": info["value"], "pos_rank_label": info.get("pos_rank_label", ""),
                         "is_reference": True, **{k: info.get(k) for k in ("age","rank_change_7d","market_trend","buy_sell_ratio")}}
                        for pid, info in values_by_id.items()
                        if info["position"] == pos
                        and abs(_asset_tier(float(info.get("value") or 0)) - req_tier) <= 1
                    ]

                    if candidates:
                        best = min(candidates, key=lambda p: abs(_asset_tier(p["value"]) - req_tier))
                        fallback_assets.append(best)
                elif kind == "K":
                    rnd = int(rest[0])
                    slot_bucket = rest[1] if len(rest) > 1 else "Mid"
                    suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(rnd, f"{rnd}th")
                    pick_yr = datetime.now().year + 1
                    fallback_assets.append({
                        "name": f"{suffix} Round ({slot_bucket})", "is_pick": True,
                        "pick_round": rnd,
                        "pick_season": datetime.now().year,
                        "pick_order": slot_bucket,
                        "value": _pick_val(rnd),
                        "is_reference": True,
                    })
            if fallback_assets:
                fallback_packages.append({
                    "type":             "real-trade",
                    "trades_like_this": trades_like_this,
                    "send":             fallback_assets,
                    "send_value":       round(sum(a.get("value", 0) for a in fallback_assets), 1),
                    "is_reference":     True,
                    "sig":              list(sig),
                })
            continue

        send_value = round(sum(a.get("value", 0) for a in matched), 1)

        if send_value > max_send_value or send_value < min_send_value:
            continue

        # Anchor check: best single player sent must be ≥ 65% of target value.
        # Prevents historical multi-scraps patterns from mapping onto the viewer's roster.
        player_vals = sorted(
            [a.get("value", 0) for a in matched if not a.get("is_pick")],
            reverse=True,
        )
        if not player_vals or player_vals[0] < target_value * 0.65:
            continue

        # Secondary tier floor: add-ons must be at least one tier above target tier.
        # Mirrors the same rule in the value-based package generator.
        def _t(v): return (1 if v>=800 else 2 if v>=500 else 3 if v>=300 else
                           4 if v>=200 else 5 if v>=130 else 6 if v>=80 else 7 if v>=40 else 8)
        tgt_tier = _t(target_value)
        _sec_floor = {1: 300, 2: 200, 3: 130, 4: 80}.get(tgt_tier, 40)
        _ter_floor = {1: 200, 2: 130, 3: 80,  4: 40}.get(tgt_tier, 40)
        non_anchor_vals = player_vals[1:]  # everything after best player
        if len(non_anchor_vals) >= 1 and non_anchor_vals[0] < _sec_floor:
            continue
        if len(non_anchor_vals) >= 2 and non_anchor_vals[1] < _ter_floor:
            continue

        result_packages.append({
            "type":             "real-trade",
            "trades_like_this": trades_like_this,
            "send":             matched,
            "send_value":       send_value,
            "sig":              list(sig),
        })
        for a in matched:
            if not a.get("is_pick"):
                used_pids.add(a["player_id"])

        if len(result_packages) >= max_packages:
            break

    # If no viewer-matched packages found, fall back to reference packages
    if not result_packages and fallback_packages:
        result_packages = sorted(fallback_packages, key=lambda x: -x["trades_like_this"])[:max_packages]

    return {
        "packages":          result_packages,
        "total_real_trades": total_real_trades,
        "sig_counts":        {str(k): len(v) for k, v in sig_counts.items()},
    }


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

    user_id = session.get("viewer_username")
    if not has_premium_access(user_id, league_id, platform):
        return jsonify({"paywall": True, "error": "Premium required"}), 403

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
                    "rank_change_7d": p.get("rank_change_7d"),
                }

        # Enrich with market momentum from trade_intel_player_stats
        try:
            from dashboard_services.db import get_conn as _gc
            with _gc() as _conn:
                _trend_col = "market_trend_sf" if league_type == "sf" else "market_trend_1qb"
                _mkt_rows = _conn.execute(
                    f"SELECT player_id, buy_sell_ratio, {_trend_col} AS market_trend "
                    "FROM trade_intel_player_stats WHERE trade_count > 0"
                ).fetchall()
            for _r in _mkt_rows:
                _pid = str(_r["player_id"])
                if _pid in values_by_id:
                    values_by_id[_pid]["buy_sell_ratio"] = float(_r["buy_sell_ratio"] or 1.0)
                    values_by_id[_pid]["market_trend"]   = round(float(_r["market_trend"] or 0.0) * _market_scale(), 1)
        except Exception:
            pass  # market signals optional — fall back to rank_change_7d only

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
                    "age":            values_by_id[pid].get("age"),
                    "rank_change_7d": values_by_id[pid].get("rank_change_7d"),
                    "market_trend":   values_by_id[pid].get("market_trend"),
                    "buy_sell_ratio": values_by_id[pid].get("buy_sell_ratio"),
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

        def _tier(v: float) -> int:
            if v >= 800: return 1
            if v >= 500: return 2
            if v >= 300: return 3
            if v >= 200: return 4
            if v >= 130: return 5
            if v >= 80:  return 6
            if v >= 40:  return 7
            return 8

        target_tier = _tier(effective_target)
        # Anchor floors: lead player must be this fraction of target value.
        # 2-for-1: 75% — near-equal piece + sweetener
        # 3-for-1: 65% — strong piece + two contributors
        # player+pick: 65% — strong piece + pick
        ANCHOR_2 = effective_target * 0.75
        ANCHOR_3 = effective_target * 0.65

        # Per-tier minimum value for non-anchor (secondary/tertiary) assets.
        # The add-on must be one tier above what the target tier alone would imply,
        # making elite players progressively harder to acquire via multi-player deals.
        # T1 target → secondary ≥ T3 (300)
        # T2 target → secondary ≥ T4 (200)
        # T3 target → secondary ≥ T5 (130)
        # T4 target → secondary ≥ T6 (80)
        # T5+        → secondary ≥ T7 (40)
        _secondary_floor = {1: 300, 2: 200, 3: 130, 4: 80}.get(target_tier, 40)
        secondary_min = _secondary_floor
        # Tertiary (3rd player) drops one tier below secondary
        _tertiary_floor = {1: 200, 2: 130, 3: 80, 4: 40}.get(target_tier, 40)
        tertiary_min = _tertiary_floor

        # 1-for-1: single player in range
        for p in viewer_players:
            if lo <= p["value"] <= hi:
                k = _key(p)
                if k not in seen:
                    seen.add(k)
                    packages.append({"type": "1-for-1", "send": [p],
                                     "send_value": p["value"],
                                     "_delta": abs(p["value"] - effective_target)})

        # 2-for-1: anchor ≥ 75% of target, secondary meets tier-scaled floor
        for i, p1 in enumerate(viewer_players):
            if p1["value"] < ANCHOR_2:
                break
            if p1["value"] > effective_target * 0.93:
                continue  # close enough for 1-for-1
            for p2 in viewer_players[i + 1:]:
                if p2["value"] < secondary_min:
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

        # 3-for-1: anchor ≥ 65% of target, p2 ≥ secondary_min, p3 ≥ tertiary_min
        for i, p1 in enumerate(viewer_players):
            if p1["value"] < ANCHOR_3:
                break
            if p1["value"] >= ANCHOR_2:
                continue  # 2-for-1 territory
            for j, p2 in enumerate(viewer_players[i + 1:], i + 1):
                if p2["value"] < secondary_min:
                    break
                for p3 in viewer_players[j + 1:]:
                    if p3["value"] < tertiary_min:
                        break
                    combined = p1["value"] + p2["value"] + p3["value"]
                    if combined > hi:
                        continue
                    if combined >= lo:
                        k = _key(p1, p2, p3)
                        if k not in seen:
                            seen.add(k)
                            packages.append({"type": "3-for-1", "send": [p1, p2, p3],
                                             "send_value": combined,
                                             "_delta": abs(combined - effective_target)})
                        break
                else:
                    continue
                break

        # Player + pick: player ≥ 65% of target, pick value meets secondary_min
        for p in viewer_players:
            if p["value"] < ANCHOR_3:
                break
            if p["value"] > effective_target * 0.93:
                continue  # close enough for 1-for-1
            for pick in viewer_picks:
                if pick["value"] < secondary_min:
                    continue
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
                                "profile":          _player_profile(info),
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
