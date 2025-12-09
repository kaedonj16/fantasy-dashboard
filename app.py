import hashlib
import json
import math
import numpy as np
import os
import pandas as pd
import threading
import time
from collections import defaultdict
from datetime import date
from flask import Flask, request, render_template_string, redirect, url_for, jsonify, render_template
from pathlib import Path
from plotly.offline import plot as plotly_plot, get_plotlyjs
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

from dashboard_services.api import get_rosters, get_users, get_league, get_traded_picks, get_nfl_players, \
    get_nfl_state, get_bracket, avatar_from_users, get_nfl_scores_for_date, build_team_game_lookup, \
    get_effective_scoring_settings, get_roster_positions, get_league_settings, get_total_rosters
from dashboard_services.awards import compute_awards_season, render_awards_section
from dashboard_services.data_building.build_daily_value_table import build_daily_data
from dashboard_services.data_building.value_model_training import build_ml_value_table
from dashboard_services.graphs_page import build_graphs_body
from dashboard_services.injuries import build_injury_report, render_injury_accordion
from dashboard_services.matchups import render_matchup_slide, render_matchup_carousel_weeks, \
    compute_team_projections_for_weeks
from dashboard_services.picks import load_pick_value_table
from dashboard_services.players import get_players_map
from dashboard_services.service import build_tables, playoff_bracket, matchup_cards_last_week, render_top_three, \
    build_matchups_by_week, build_picks_by_roster, render_teams_sidebar, build_week_activity, pill, \
    seed_top6_from_team_stats, build_standings_map
from dashboard_services.trade_calculator_page import build_trade_calculator_body
from dashboard_services.utils import load_teams_index, streak_class, build_teams_overview, load_model_value_table, \
    load_players_index, \
    load_week_projection, bucket_for_slot, clear_activity_cache_for_league, clear_weekly_cache_for_league, \
    build_status_for_week, clear_teams_cache_for_league, get_week_projections_cached, \
    fetch_week_from_tank01, count_roster_positions, load_idp_index

daily_lock = threading.Lock()
daily_completed = None

DASHBOARD_CACHE = {}  # {league_id: {"ctx": ..., "ts": ..., "page_html": {page: (ts, html)}}}

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

FORM_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>BR Fantasy Dashboard</title>
    <style>
      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        background: white;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        flex-direction: column;
      }
      .shell {
        background: #122d4b;
        padding: 24px 28px;
        border-radius: 16px;
        box-shadow: 0 18px 40px rgba(0,0,0,.55);
        width: 380px;
      }
      h1 {
        font-size: 20px;
        margin-bottom: 16px;
      }
      label {
        display: block;
        font-size: 13px;
        margin-bottom: 4px;
        color: #9ca3af;
      }
      input[type="text"], input[type="number"] {
        width: 94.3%;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid #1f2937;
        background: #7d8895;
        color: white;
        font-size: 14px;
      }
      .row {
        margin-bottom: 12px;
      }
      button {
        width: 100%;
        padding: 9px 10px;
        border-radius: 10px;
        border: none;
        background: #38bdf8;
        color: #0f172a;
        font-weight: 600;
        cursor: pointer;
        font-size: 14px;
        margin-top: 4px;
      }
      button:hover {
        background: #0ea5e9;
      }
      .hint {
        font-size: 12px;
        color: #6b7280;
        margin-top: 6px;
      }
      .h1 {text-align: center;}
      .error-message {
        margin-top: 8px;
        font-size: 13px;
        color: #fecaca;
      }
    </style>
  </head>
  <body>
    <div><img src="/static/BR_Logo.png" alt="League Logo" class="site-logo" style="height:125px"/></div>
    <div class="shell">
      <form method="post">
        <div class="row">
          <label for="league">Sleeper League ID</label>
          <input type="text" id="league" name="league" required value="{{ league or '' }}">
        </div>
        <button type="submit">Generate Dashboard</button>
        {% if error %}
        <div class="error-message">
          {{ error }}
        </div>
        {% endif %}
        <div class="hint">Paste your Sleeper league ID, hit generate, and we’ll build the dashboard.</div>
      </form>
    </div>
  </body>
</html>
"""


BASE_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- use your existing dashboard CSS here -->
    <link rel="stylesheet" href="/static/dashboard.css">
    <script>
      {plotly_js}
    </script>
  </head>
  <body>
    {nav}
    <main id="page-root" class="overview-layout">
      {body}
    </main>
    <script src="/static/app.js"></script>
  </body>
</html>
"""

def timed(label: str, fn, *args, **kwargs):
    """
    Helper to log how long a block takes.
    Usage: result = timed("build_tables", build_tables, ...)
    """
    t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIMING] {label}: {dt:.2f}s")


def get_page_html_from_cache(league_id: str, page: str) -> str:
    entry = DASHBOARD_CACHE.get(league_id)
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


def store_page_html(league_id: str, page: str, html: str) -> None:
    entry = DASHBOARD_CACHE.setdefault(league_id, {})
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



def run_daily_data_async():
    """Start daily data build in a background thread."""
    thread = threading.Thread(
        target=build_daily_data,
        args=(CURRENT_SEASON,CURRENT_WEEK),
        daemon=True
    )
    thread.start()


def _weeks_hash(weeks):
    raw = ",".join(str(w) for w in weeks)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def get_cached_model_value_table(league_id: str, season: int, weeks: List[int]):
    key = f"values_{season}_{_weeks_hash(weeks)}"

    entry = DASHBOARD_CACHE.get(league_id, {})
    bundle = entry.get("value_tables", {})

    record = bundle.get(key)

    # Case 1: cache hit
    if record:
        ts, value_table = record
        if time.time() - ts < VALUE_CACHE_TTL:
            return value_table
        # cache exists but expired → fall through to load from disk

    # Case 2: no cache OR cache expired → try loading from disk
    disk_table = load_model_value_table()
    if disk_table:
        return disk_table

    # Case 3: nothing exists
    return None


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


def build_nav(league_id: str, active: str,) -> str:
    """
    active: one of 'dashboard','standings','power','weekly','teams','activity','injuries','trade','graphs'
    """

    def nav_pill(label: str, endpoint: str, key: str) -> str:
        cls = "nav-pill active" if key == active else "nav-pill"
        href = url_for(endpoint, league_id=league_id)
        return f"<a class='{cls}' href='{href}'>{label}</a>"

    # pages where a refresh makes sense
    refreshable_pages = {"dashboard", "weekly", "teams", "activity", "standings"}

    # label per page
    refresh_label_map = {
        "dashboard": "↻",
        "weekly": "↻",
        "teams": "↻",
        "activity": "↻",
        "standings": "↻",
    }
    refresh_label = refresh_label_map.get(active, "↻")

    # default season -> empty string if None

    refresh_btn = ""
    if active in refreshable_pages:
        refresh_btn = (
            f"<button type='button'"
            f"        id='refreshBtn'"
            f"        class='refresh-icon'"
            f"        data-page='{active}'"
            f"        data-league='{league_id}'"
            f"        style='display:inline-flex;gap:6px;color: #122d4b;font-size: x-large;background: white;border: white;transform: rotate(90deg);'>"
            f"{refresh_label}"
            f"</button>"
        )

    pills = [
        refresh_btn,  # may be empty string if not a refreshable page
        nav_pill("Dashboard", "page_dashboard", "dashboard"),
        nav_pill("Weekly Hub", "page_weekly", "weekly"),
        nav_pill("Trade Calc", "page_trade", "trade"),
        nav_pill("Teams", "page_teams", "teams"),
        nav_pill("Activity", "page_activity", "activity"),
        nav_pill("Standings", "page_standings", "standings"),
        nav_pill("Graphs", "page_graphs", "graphs"),
        "<a class='nav-pill logout-pill' href='/logout'>Logout</a>",
    ]

    return (
        "<nav class='top-nav'>"
        "  <div><img src='/static/Website_Logo.png' alt='League Logo' class='site-logo'/></div>"
        "  <div>"
        f"    {''.join(pills)}"
        "  </div>"
        "</nav>"
    )


def render_page(title: str, league_id: str, active: str, body_html: str) -> str:
    nav_html = build_nav(league_id, active)
    return BASE_HTML.format(title=title, nav=nav_html, body=body_html, plotly_js=plotly_js)


def validate_league_id(league_id: str) -> bool:
    league_id = (league_id or "").strip()
    if not league_id:
        return False

    try:
        league = get_league(league_id)
        print(league)
    except Exception as e:
        print(f"[validate_league_id] error checking league {league_id}: {e}")
        return False

    if not isinstance(league, dict):
        return False

    return bool(league.get("league_id"))


def build_league_context(league_id: str) -> dict:
    """
    Fetch all core data for a league once and reuse across pages.
    Heavy, but we do it rarely and cache the result.
    """

    # External data (with timing)
    rosters = timed("get_rosters", get_rosters, league_id)
    users = timed("get_users", get_users, league_id)
    league = timed("get_league", get_league, league_id)
    traded = timed("get_traded_picks", get_traded_picks, league_id)
    current = timed("get_nfl_state", get_nfl_state)
    scoring_settings = get_effective_scoring_settings()   # defaults overlaid with league scoring_settings
    raw_scoring_settings = get_scoring_settings() if 'get_scoring_settings' in globals() else None  # optional
    roster_positions = get_roster_positions()
    league_settings = get_league_settings()
    total_rosters = get_total_rosters()


    # Shared global data
    players = timed("get_nfl_players(global)", get_players_global)
    players_index = timed("load_players_index(global)", get_players_index_global)
    teams_index = timed("load_teams_index(global)", get_teams_index_global)
    players_map = get_players_map(players)

    current_season = current.get("season")
    current_week = current.get("week")
    weeks = 18

    # Core league tables (weekly DF + team_stats + roster_map)
    df_weekly, team_stats, roster_map = timed(
        "build_tables",
        build_tables,
        league_id,
        current_week,
        players,
        users,
        rosters,
    )

    # Activity / injury data
    activity_df = timed("build_week_activity", build_week_activity, league_id, players_map)
    injury_df = timed(
        "build_injury_report",
        build_injury_report,
        league_id,
        "America/New_York",
        False,
    )

    standings_map = build_standings_map(team_stats, roster_map)

    picks_by_roster = build_picks_by_roster(
        num_future_seasons=3,
        league=league,
        rosters=rosters,
        traded=traded,
    )

    scores_body = get_nfl_scores_for_date(date.today().strftime("%Y%m%d"))
    team_game_lookup = build_team_game_lookup(scores_body)

    # Model value table (used by multiple pages) – load once per ctx
    model_value_table = load_model_value_table() or []

    return {
        "league": league,
        "league_id": league_id,
        "rosters": rosters,
        "users": users,
        "traded": traded,
        "current_season": current_season,
        "current_week": current_week,
        "players": players,
        "players_map": players_map,
        "players_index": players_index,
        "teams_index": teams_index,
        "df_weekly": df_weekly,
        "team_stats": team_stats,
        "roster_map": roster_map,
        "weeks": weeks,
        "injury_df": injury_df,
        "activity_df": activity_df,
        "standings_map": standings_map,
        "picks_by_roster": picks_by_roster,
        "team_game_lookup": team_game_lookup,
        "model_value_table": model_value_table,
        "scoring_settings": scoring_settings,          # effective (defaults + league overrides)
        "raw_scoring_settings": raw_scoring_settings,  # if you expose the raw one, otherwise drop this
        "roster_positions": roster_positions,
        "league_settings": league_settings,
        "total_rosters": total_rosters,
    }


def ensure_weekly_bits(ctx: dict) -> None:
    """
    Lazily populate projections, statuses, matchups, and df_weekly['proj']
    into the ctx. Only used by Dashboard + Weekly Hub + related APIs.
    """
    # If we've already populated, nothing to do
    if all(k in ctx for k in ("proj_by_week", "statuses", "matchups_by_week", "proj_by_roster")):
        # still ensure df_weekly has 'proj' column
        df = ctx.get("df_weekly")
        if df is not None and "proj" not in df.columns:
            proj_by_roster = ctx["proj_by_roster"]
            key_series = list(zip(df["week"].astype(int), df["roster_id"].astype(str)))
            df["proj"] = [proj_by_roster.get(k, float("nan")) for k in key_series]
            ctx["df_weekly"] = df
        return

    current_season = ctx["current_season"]
    weeks = ctx["weeks"]
    league_id = ctx["league_id"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    roster_map = ctx["roster_map"]
    players_map = ctx["players_map"]

    proj_by_week = build_projections_by_week(current_season, weeks)
    if any(k in count_roster_positions(get_roster_positions()) for k in ["DL","LB", "DB","IDP_FLEX"]):
        statuses = build_status_by_week(current_season, weeks, players_index, teams_index, load_idp_index())
    else:
        statuses = build_status_by_week(current_season, weeks, players_index, teams_index)
    matchups_by_week = build_matchups_by_week(
        league_id,
        range(1, weeks),
        roster_map,
        players_map,
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

    # add/refresh proj column on df_weekly
    df = ctx["df_weekly"]
    key_series = list(zip(df["week"].astype(int), df["roster_id"].astype(str)))
    df["proj"] = [proj_by_roster.get(k, float("nan")) for k in key_series]
    ctx["df_weekly"] = df



def refresh_league_ctx_section(league_id: str, page: str) -> dict:
    """
    Refresh only the parts of the league context needed for a given page.
    Mutates the cached ctx in-place and updates the timestamp.
    """
    entry = DASHBOARD_CACHE.get(league_id)
    if not entry:
        # No cache yet: build everything once
        ctx = build_league_context(league_id)
        DASHBOARD_CACHE[league_id] = {"ctx": ctx, "ts": time.time()}
        return ctx

    ctx = entry["ctx"]

    # Common bits we’ll reuse
    rosters = ctx["rosters"]
    users = ctx["users"]
    players = ctx["players"]
    players_map = ctx["players_map"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    current_season = ctx["current_season"]
    current_week = ctx["current_week"]
    weeks = ctx["weeks"]  # this is your int (e.g. 18)

    # ---------- Standings / core weekly data ----------
    if page in ("standings", "dashboard", "weekly"):
        df_weekly, team_stats, roster_map = build_tables(
            league_id, current_week, players, users, rosters
        )

        ctx["df_weekly"] = df_weekly
        ctx["team_stats"] = team_stats
        ctx["roster_map"] = roster_map
        ctx["standings_map"] = build_standings_map(team_stats, roster_map)

    # ---------- Activity / injuries ----------
    if page in ("activity", "dashboard"):
        clear_activity_cache_for_league(league_id)
        ctx["activity_df"] = build_week_activity(league_id, players_map)

        ctx["injury_df"] = build_injury_report(
            league_id,
            local_tz="America/New_York",
            include_free_agents=False,
        )

    # ---------- Weekly projections, statuses, matchups ----------
    if page in ("weekly", "dashboard"):
        clear_weekly_cache_for_league()

        # make sure projections for current week are refreshed at the source
        get_week_projections_cached(
            current_season,
            current_week,
            fetch_week_from_tank01,
            True,  # force_refresh
        )

        ctx["proj_by_week"] = build_projections_by_week(current_season, weeks)

        ctx["statuses"] = build_status_by_week(
            current_season,
            weeks,
            players_index,
            teams_index,

        )

        ctx["matchups_by_week"] = build_matchups_by_week(
            league_id,
            range(1, weeks),
            ctx["roster_map"],
            players_map,
        )

        proj_by_roster = compute_team_projections_for_weeks(
            ctx["matchups_by_week"],
            ctx["statuses"],
            ctx["proj_by_week"],
            ctx["roster_map"],
        )
        ctx["proj_by_roster"] = proj_by_roster
        proj_by_roster = compute_team_projections_for_weeks(
            matchups_by_week,
            statuses,
            proj_by_week,
            roster_map,
        )

        # vectorized lookup for projections
        key_series = list(zip(df_weekly["week"].astype(int), df_weekly["roster_id"].astype(str)))
        proj_map = proj_by_roster  # already a dict keyed by (week, roster_id)
        df_weekly["proj"] = [proj_map.get(k, float("nan")) for k in key_series]

        # vectorized proj column update on df_weekly
        df = ctx["df_weekly"]
        key_series = list(zip(df["week"].astype(int), df["roster_id"].astype(str)))
        proj_map = proj_by_roster
        df["proj"] = [proj_map.get(k, float("nan")) for k in key_series]
        ctx["df_weekly"] = df

    # ---------- Teams page ----------
    if page == "teams":
        clear_teams_cache_for_league()
        # if rosters / users may change, re-pull them
        ctx["rosters"] = get_rosters(league_id)
        ctx["users"] = get_users(league_id)

    # Update timestamp
    entry["ts"] = time.time()
    return ctx


def get_league_ctx_from_cache(league_id: str) -> dict:
    now = time.time()
    entry = DASHBOARD_CACHE.get(league_id)
    if entry and (now - entry["ts"] < CACHE_TTL):
        return entry["ctx"]

    ctx = build_league_context(league_id)
    DASHBOARD_CACHE[league_id] = {"ctx": ctx, "ts": now}
    return ctx


def render_standings(team_stats, length) -> str:
    """
    Simple standings snapshot card: top N teams by record / PF.
    Adjust column names if your team_stats schema is different.
    """
    # adjust these column names to match your DataFrame
    # Example assumption:
    # team_stats has columns: 'owner', 'record', 'pf', 'pa'
    rows = []

    # Sort by Wins, Win%, PF, PA
    df = team_stats.copy()
    df["WinPct"] = df["Win%"].astype(float)
    df = (
        df.sort_values(
            by=["Wins", "PF", "PA"],
            ascending=[False, False, True],  # PA lower is better
        )
        .reset_index(drop=True)
    )

    # Add Rank column (1..N)
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
              <td>{row['past_sos']:.1f}</td>
              <td>{row['ros_sos']:.1f}</td>
            </tr>
        """)
    if len(rows) != length:
        total_rows = rows[:length]
    else:
        total_rows = rows
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

    # --- Standings snapshot (top card in main column) ---
    standings_html = render_standings(team_stats, 5)

    # --- Finalized games + last_final_week (for proj cutoff) ---
    finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())
    else:
        # fall back to current week if nothing is finalized yet
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

    # --- Awards / recap style info (season-level or last week) ---
    awards = compute_awards_season(finalized_df, players_map, league_id)
    awards_html = render_awards_section(awards)

    # --- Teams sidebar (right-hand side) ---

    teams_ctx = build_teams_overview(
        rosters=rosters,
        users_list=users,
        picks_by_roster=picks_by_roster,
        players=players_map,
        players_index=players_index,
        teams_index=teams_index,
    )

    teams_sidebar_html = render_teams_sidebar(teams_ctx)

    # --- compose into main + sidebar ---
    body = f"""
    <aside class="overview-sidebar-left">
      {awards_html}
    </aside>
    <div class="overview-main">
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


def render_power_and_playoffs(team_stats, roster_map: dict[str, str], league_id: str) -> str:
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
    wb = get_bracket(league_id, "winners")
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
    best = df_weekly.groupby("owner")["points"].max().rename("Best Week")
    worst = df_weekly.groupby("owner")["points"].min().rename("Worst Week")

    stats_tbl = (team_stats.rename(columns={"owner": "Team", "AVG": "Average", "STD": "Std Dev", "Win%": "Win %"})
                 .merge(best, left_on="Team", right_index=True, how="left")
                 .merge(worst, left_on="Team", right_index=True, how="left"))

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
    table_html = render_team_stats(team_stats, df_weekly[df_weekly["finalized"] == True].copy())
    power_playoffs_html = render_power_and_playoffs(team_stats, roster_map, ctx["league_id"])
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
                  {table_html}   <!-- your FULL sortable stats table -->
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


def _render_weekly_top_scorers_for_week(
        league_id: str,
        df_weekly: pd.DataFrame,
        roster_map: dict,
        players_map: dict,
        projections: dict,  # <–– pass ALL projections in once
        rosters: dict,
        w: int,
        users: list,
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
    for _, proj in projections[w].items():
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
    league_id = ctx["league_id"]
    season = ctx["current_season"]
    rosters = ctx["rosters"]
    users = ctx["users"]
    df_weekly = ctx["df_weekly"]
    roster_map = ctx["roster_map"]
    players_map = ctx["players_map"]
    current_week = ctx["current_week"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    proj_by_week = ctx["proj_by_week"]
    weeks = ctx["weeks"]
    statuses = ctx["statuses"]
    team_game_lookup = ctx["team_game_lookup"]
    matchups_by_week = ctx["matchups_by_week"]

    # last finalized week
    last_final_week = current_week
    finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())

    # Default week: current if valid, else last prior
    default_week = current_week if current_week in range(1, weeks) else (weeks - 1)

    # --- Matchups for default week only ---
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

    # --- Week dropdown ---
    options = []
    for w in range(1, weeks):
        sel = " selected" if w == default_week else ""
        options.append(f"<option value='{w}'{sel}>Week {w}</option>")
    week_select_html = "".join(options)

    # --- Only prebuild DEFAULT week’s left + sidebar panels ---
    top_scorers_html = _render_weekly_top_scorers_for_week(
        league_id,
        df_weekly,
        roster_map,
        players_map,
        proj_by_week,
        rosters,
        default_week,
        users,
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
  var leagueId = {league_js};

  var sel = document.getElementById('hubWeek');
  if (!sel) return;

  var matchupsContainer = document.getElementById('weeklyMatchupsContainer');
  var loadingOverlay    = document.getElementById('weeklyMatchupsLoading');

  function showLoading() {{
    if (loadingOverlay) {{
      loadingOverlay.classList.remove('hidden');
    }}
  }}

  function hideLoading() {{
    if (loadingOverlay) {{
      loadingOverlay.classList.add('hidden');
    }}
  }}

  sel.addEventListener('change', function() {{
    var w = this.value;

    showLoading();

    fetch('/api/weekly-week?league_id=' + encodeURIComponent(leagueId) +
          '&week=' + encodeURIComponent(w))
      .then(function(res) {{ return res.json(); }})
      .then(function(data) {{
        if (!data.ok) {{
          console.error('Failed to load week', w, data.error);
          hideLoading();
          return;
        }}

        var mainContainer = document.querySelector('.week-main-panels');
        var sideContainer = document.querySelector('.week-side-panels');

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
          // only replace inner carousel HTML, keep shell + overlay
          matchupsContainer.innerHTML = data.matchups_html;

          // re-align the carousel to the first slide + wire up buttons
          if (typeof window.resetMatchupCarousels === 'function') {{
            window.resetMatchupCarousels(matchupsContainer);
          }}
        }}

        hideLoading();
      }})
      .catch(function(err) {{
        console.error('Error fetching week', w, err);
        hideLoading();
      }});
  }});
}})();
</script>
"""


def build_projections_by_week(season: int, weeks: int):
    bundles = {}
    for w in range(1, weeks):
        try:
            projections = load_week_projection(season, w)
            bundles[w] = { "projections": projections, }
        except Exception as e:
            print(f"Error loading week {w} projections: {e}")
            bundles[w] = {"projections": {}}

    return bundles


def build_status_by_week(season: int, weeks: int, players_index, teams_index, idp_player_index: dict[str, dict] = None ):
    bundles = {}
    for w in range(1, weeks):
        try:
            statuses = build_status_for_week(season, w, players_index, teams_index, idp_player_index)
            bundles[w] = {
                "statuses": statuses,
            }
        except Exception as e:
            print(f"Error loading week {w} schedule: {e}")
            bundles[w] = {"statuses": {}}
    return bundles


def build_activity_body(ctx: dict) -> str:
    league_id = ctx["league_id"]
    activity_df = ctx["activity_df"]
    injury_df = ctx["injury_df"]
    standings_map = ctx["standings_map"]

    # ---------- VALUE SOURCES ----------
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
            player_val_by_key[(raw_name, pos, team)] = val
            player_val_by_key_np[(raw_name, pos)] = val

            lbl = row.get("pos_rank_label") or pos
            rank_label_by_name[name_lower] = str(lbl)

    def player_value(p: dict) -> tuple[float, str]:
        name = str(p.get("name") or "").strip()
        name_lower = name.lower()
        pos = str(p.get("pos") or "").strip().upper()
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

        bucket: str | None = None
        if seed is not None:
            if 1 <= seed <= 3:
                bucket = "early"
            elif 4 <= seed <= 7:
                bucket = "mid"
            elif 8 <= seed <= num_teams:
                bucket = "late"

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

    # ---------- ACTIVITY CARD HTML ----------
    activity_html = ""
    if activity_df is not None and not activity_df.empty:

        def html_trade(txrow):
            data = txrow["data"]
            teams = data["teams"]
            users = get_users(league_id)

            rid_to_name = {}
            for tm in teams:
                rid = tm.get("roster_id")
                if rid is not None:
                    rid_to_name[rid] = tm.get("name") or f"Team {rid}"

            def render_player_row(p, io_class):
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
                rnd_suffix = {1: "st", 2: "nd", 3: "rd"}.get(pick.get("round"), "th")
                round_label = f"{pick.get('round')}" + rnd_suffix
                season = str(pick.get("season") or "")
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
                subline = f"{orig_name}'s Pick" if orig_name else "Traded Pick"
                val = pick_value(pick, standings_map)
                val_txt = f"{val:.1f}" if val > 0 else ""
                val_html = f'<div class="player-trade-value">{val_txt}</div>' if val_txt else ""
                return (
                    "<div class='player-activity'>"
                    "<div style='gap: 10px;display: flex;align-items: center;'>"
                    f"<span class='io {io_class}'>"
                    f"{'+' if io_class == 'add' else '−'}</span>"
                    "<div>"
                    f"  <div style='font-weight:600'>{season} {round_label}</div>"
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

            # -------- build sides for multi-for-one (2-team trades) --------
            side_map: dict[int, dict] = {}
            for tm in teams:
                rid = tm.get("roster_id")
                if rid is None:
                    continue
                in_players = tm.get("gets") or []
                in_picks = picks_by_receiver.get(rid, []) or []

                # player_value returns (value, pos_rank_label)
                in_player_pairs = [player_value(p) for p in in_players]

                # just the numeric values for totals
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
                # this mutates side_a / side_b in-place
                apply_multi_for_one_adjustment(side_a, side_b)

            cols = []
            for tm in teams:
                roster_id = tm.get("roster_id")
                # incoming
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
                    gets = "<div style='color:#64748b;font-size:13px'>No players</div>"

                # outgoing
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

                # totals
                # effective "in" total (after multi-for-one)
                side_info = side_map.get(roster_id)
                eff_in = side_info["effective_total"] if side_info else 0.0

                # raw "out" total
                out_total = 0.0
                for p in (tm.get("sends") or []):
                    out_total += player_value(p)[0]
                if roster_id is not None:
                    for pick in picks_by_sender.get(roster_id, []):
                        out_total += pick_value(pick, standings_map)

                net_total = eff_in - out_total

                total_html = (
                    "<div class='trade-total-row'>"
                    "<hr style='margin-top:8px;margin-bottom:4px;border:none;border-top:1px solid #e2e8f0;'>"
                    "<div style='display:flex;justify-content:space-between;font-size:14px;font-weight:600;'>"
                    "<span>Total Value</span>"
                    f"<span>{net_total:.0f}</span>"
                    "</div>"
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
            d = txrow["data"]
            avatar = d.get("avatar") or ""
            img = (
                f"<img class='avatar' src='{avatar}' "
                "onerror=\"this.style.display='none'\">"
                if avatar
                else ""
            )
            adds_parts = []
            for p in d.get("adds", []):
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
            adds = "".join(adds_parts)

            when = (
                txrow["ts"].astimezone(ZoneInfo("America/New_York")).strftime("%b %d, %I:%M %p")
                if pd.notna(txrow["ts"])
                else ""
            )
            return (
                "<div class='tx activity-item' data-kind='waiver'>"
                f"  <div class='meta'>{pill('Waiver')} • {when}</div>"
                "  <div class='team-col'>"
                f"    <header>{img}<div class='team-name'>{d['name']}</div></header>"
                f"    <div class='plist'>{adds}</div>"
                "  </div>"
                "</div>"
            )

        cards = []
        for _, row in activity_df.iterrows():
            cards.append(html_trade(row) if row["kind"] == "trade" else html_waiver(row))

        activity_html = (
            "<div class='card activity-card' data-section='activity'>"
            "  <div class='card-header-row'>"
            "    <h2>Trades & Waiver Claims</h2>"
            "  </div>"
            "  <div class='scroll-box'>"
            "    <div class='feed'>"
            f"      {''.join(cards)}"
            "    </div>"
            "  </div>"
            "</div>"
        )

    # ---------- EXISTING INJURY ACCORDION ----------
    injury_html = ""
    if injury_df is not None and not injury_df.empty:
        injury_html = render_injury_accordion(injury_df)

    # ---------- PAGE LAYOUT WITH TWO COLUMNS + SIDEBAR ----------
    return f"""
    <div class="page-layout activity-page">
      <main class="page-main activity-main">
        <div class="activity-col">
          {activity_html or "<div class='card'><div class='card-body'><p>No activity yet.</p></div></div>"}
        </div>
        <div class="injury-col">
          {injury_html or "<div class='card'><div class='card-body'><p>No injury data.</p></div></div>"}
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

    <script>
    (function() {{
      // Activity filter (sidebar pills)
      document.querySelectorAll('.act-toggle').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          const kind = this.getAttribute('data-kind');
          const isActive = this.classList.toggle('active');

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

    # ----------------- Load value table -----------------
    # Expected rows like {id, name, position, team, value, search_name}
    model_vals = ctx.get("model_value_table") or []

    name_to_rank_label: dict[str, str] = {}
    name_to_age: dict[str, float | None] = {}

    for obj in model_vals:
        if not isinstance(obj, dict):
            continue
        nm = str(obj.get("search_name") or "").strip().lower()
        if not nm:
            continue
        pos_lbl = obj.get("pos_rank_label") or obj.get("position") or obj.get("pos") or ""
        name_to_rank_label[nm] = str(pos_lbl)
        age_val = obj.get("age")
        if age_val is not None:
            try:
                name_to_age[nm] = float(age_val)
            except Exception:
                name_to_age[nm] = None

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
        avatar = avatar_from_users(users, str(rid))
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
        "QB": slot_counts.get("QB"),
        "RB": slot_counts.get("RB"),
        "WR": slot_counts.get("WR"),
        "TE": slot_counts.get("TE"),
        "FLEX": slot_counts.get("FLEX"),
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
            name_raw = p.get('name', '')
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
                f"      <div style='font-weight:600'>{name_raw}</div>"
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


@app.route("/league/<league_id>/dashboard")
def page_dashboard(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    ensure_weekly_bits(ctx)  # dashboard needs projections & matchups
    body = timed("build_dashboard_body", build_dashboard_body, ctx)
    return render_page("Dashboard", league_id, "dashboard", body)


@app.route("/league/<league_id>/standings")
def page_standings(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body = timed("build_standings_body", build_standings_body, ctx)
    return render_page("Standings", league_id, "standings", body)


@app.route("/league/<league_id>/weekly")
def page_weekly(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    ensure_weekly_bits(ctx)
    body = timed("build_weekly_hub_body", build_weekly_hub_body, ctx)
    return render_page("Weekly Hub", league_id, "weekly", body)


@app.route("/league/<league_id>/trade")
def page_trade(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body = build_trade_calculator_body(ctx["league_id"], ctx["current_season"])
    return render_page("Trade Calculator", league_id, "trade", body)


@app.route("/league/<league_id>/activity")
def page_activity(league_id):
    cached = get_page_html_from_cache(league_id, "activity")
    if cached:
        return render_page("Activity", league_id, "activity", cached)

    ctx = get_league_ctx_from_cache(league_id)
    body = timed("build_activity_body", build_activity_body, ctx)
    store_page_html(league_id, "activity", body)
    return render_page("Activity", league_id, "activity", body)


@app.route("/league/<league_id>/graphs")
def page_graphs(league_id):
    cached = get_page_html_from_cache(league_id, "graphs")
    if cached:
        return render_page("Graphs", league_id, "graphs", cached)

    ctx = get_league_ctx_from_cache(league_id)
    body_html = timed("build_graphs_body", build_graphs_body, ctx)
    store_page_html(league_id, "graphs", body_html)
    return render_page("Graphs", league_id, "graphs", body_html)


@app.route("/league/<league_id>/teams")
def page_teams(league_id):
    cached = get_page_html_from_cache(league_id, "teams")
    if cached:
        return render_page("Teams", league_id, "teams", cached)

    ctx = get_league_ctx_from_cache(league_id)
    body_html = timed("build_teams_body", build_teams_body, ctx)
    store_page_html(league_id, "teams", body_html)
    return render_page("Teams", league_id, "teams", body_html)


@app.before_request
def maybe_run_daily():
    global daily_completed

    today = date.today().isoformat()

    # if already done for today, nothing to do
    if daily_completed == today:
        return

    # ensure only one thread runs it
    if daily_lock.acquire(blocking=False):
        try:
            if daily_completed != today:
                print(f"[startup] Running daily data process for {today}...")

                # run async (don’t block request)
                run_daily_data_async()

                # mark as done *immediately* so we don't spawn multiple threads
                daily_completed = today

        finally:
            daily_lock.release()


@app.route("/", methods=["GET", "POST"])
def index():
    default_weeks = get_nfl_state().get("week")

    if request.method == "POST":
        league_id = request.form.get("league", "").strip()

        # Validate the Sleeper league ID before doing any heavy work
        if not validate_league_id(league_id):
            return render_template_string(
                FORM_HTML,
                league=league_id,
                weeks=default_weeks,
                error="Invalid Sleeper league ID. Please check it and try again.",
            )

        cache_key = league_id
        now = time.time()
        print("cache keys currently:", list(DASHBOARD_CACHE.keys()))

        entry = DASHBOARD_CACHE.get(cache_key)
        if entry and (now - entry["ts"] < CACHE_TTL):
            # we already have a fresh context; jump to dashboard
            return redirect(url_for("page_dashboard", league_id=league_id))

        # build and cache context instead of HTML
        ctx = build_league_context(league_id)
        DASHBOARD_CACHE[cache_key] = {"ctx": ctx, "ts": now}

        return redirect(url_for("page_dashboard", league_id=league_id))

    # GET -> show the form
    return render_template_string(
        FORM_HTML,
        league=None,
        weeks=default_weeks,
        error=None,
    )



@app.route("/league/<league_id>")
def view_league(league_id):
    # warm or reuse league context cache
    _ = get_league_ctx_from_cache(league_id)
    return redirect(url_for("page_dashboard", league_id=league_id))


@app.route("/api/weekly-week")
def api_weekly_week():
    """
    Return HTML snippets for Weekly Hub:
      - top scorers (left main card)
      - highlights (sidebar)
      - matchups carousel (right main column)
    for a specific week.
    """
    league_id = (request.args.get("league_id") or "").strip()

    try:
        week = int(request.args.get("week") or 0)
    except ValueError:
        week = 0

    if not league_id or not week:
        return jsonify({"ok": False, "error": "Missing league_id or week"}), 400

    ctx = get_league_ctx_from_cache(league_id)
    ensure_weekly_bits(ctx)  # make sure proj_by_week / statuses / matchups_by_week exist

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
    season = ctx["current_season"]
    current_week = ctx["current_week"]
    max_weeks = ctx["weeks"]

    # guard against invalid weeks
    if week < 1 or week >= max_weeks:
        return jsonify({"ok": False, "error": f"Week {week} out of range"}), 400

    # --- last finalized week for proper matchup slide behavior ---
    finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())
    else:
        last_final_week = current_week

    # --- top scorers / highlights for this week ---
    top_html = _render_weekly_top_scorers_for_week(
        league_id,
        df_weekly,
        roster_map,
        players_map,
        proj_by_week,
        rosters,
        week,
        users,
    )
    highlights_html = _render_weekly_highlights(df_weekly, week)

    # --- matchups carousel HTML for this week only ---
    matchups = matchups_by_week.get(week, []) or []

    # use .get(...) to avoid KeyError if statuses[week] missing
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

    if slides:
        slides_html = "".join(slides)
    else:
        slides_html = "<div class='m-empty'>No matchups</div>"

    slides_by_week = {week: slides_html}

    # this version returns pure HTML (no <script>), matching app.js’ carousel logic
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
    page = (payload.get("page") or "").strip().lower()

    if not league_id or not page:
        return jsonify({"ok": False, "error": "Missing league_id or page"}), 400

    valid_pages = {"activity", "standings", "teams", "weekly", "dashboard"}
    if page not in valid_pages:
        return jsonify({"ok": False, "error": f"Unknown page '{page}'"}), 400

    try:
        # Refresh the ctx for this league + page
        ctx = refresh_league_ctx_section(league_id, page)

        # Re-render just the page body for this page
        if page == "dashboard":
            ensure_weekly_bits(ctx)
            body_html = build_dashboard_body(ctx)
        elif page == "standings":
            body_html = build_standings_body(ctx)
        elif page == "weekly":
            ensure_weekly_bits(ctx)
            body_html = build_weekly_hub_body(ctx)
        elif page == "activity":
            body_html = build_activity_body(ctx)
            store_page_html(league_id, "activity", body_html)
        elif page == "teams":
            body_html = build_teams_body(ctx)
            store_page_html(league_id, "teams", body_html)
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


CURRENT_SEASON = 2025
CURRENT_WEEK = get_nfl_state().get("week")

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

    # ---------- Build both sides + apply adjustment ----------
    side_a = build_side(side_a_players, side_a_picks)
    side_b = build_side(side_b_players, side_b_picks)

    apply_multi_for_one_adjustment(side_a, side_b)

    a_eff = side_a["effective_total"]
    b_eff = side_b["effective_total"]

    diff = a_eff - b_eff
    abs_diff = abs(diff)

    # ---------- Percentage-based fair threshold ----------
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
    league_id = (request.args.get("league_id") or "global").strip()
    season = int(request.args.get("season") or CURRENT_SEASON)
    weeks = list(range(1, 19))

    # ----- 1) Get / cache your model player values -----
    model_value_table = get_cached_model_value_table(league_id, season, weeks)
    if model_value_table is None:
        print(f"[league_players] cache MISS for league={league_id}, season={season}")
        model_value_table = build_ml_value_table()
        store_value_table(league_id, season, weeks, model_value_table)
    else:
        print(f"[league_players] cache HIT for league={league_id}, season={season}")

    # model_value_table is expected to be a list[dict] of players
    if not isinstance(model_value_table, list):
        # if your existing implementation returns a dict, adapt as needed
        raise ValueError("model_value_table must be a list of player objects")

    # ----- 2) Normalize players & mark them as type='player' -----
    players = []
    for obj in model_value_table:
        if not isinstance(obj, dict):
            continue
        p = dict(obj)  # shallow copy so we don’t mutate cache
        p.setdefault("type", "player")
        # make sure id is a string (helps frontend comparisons)
        if "id" in p and p["id"] is not None:
            p["id"] = str(p["id"])
        players.append(p)

    cleaned = _sanitize_for_json(players)
    return jsonify(cleaned)


if __name__ == "__main__":
    app.run(debug=False)
