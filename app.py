import hashlib
import json
import math
import numpy as np
import os
import pandas as pd
import threading
import time
from datetime import date
from datetime import date
from flask import Flask, request, render_template_string, redirect, url_for, jsonify, render_template
from pathlib import Path
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

from dashboard_services.api import get_rosters, get_users, get_league, get_traded_picks, get_nfl_players, \
    get_nfl_state, get_bracket
from dashboard_services.graphs_page import build_graphs_body
from dashboard_services.awards import compute_awards_season, render_awards_section
from dashboard_services.data_building.build_daily_value_table import build_daily_data
from dashboard_services.data_building.value_model_training import build_ml_value_table
from dashboard_services.injuries import build_injury_report, render_injury_accordion
from dashboard_services.matchups import render_matchup_slide, render_matchup_carousel_weeks, \
    compute_team_projections_for_weeks
from dashboard_services.players import get_players_map
from dashboard_services.service import build_tables, playoff_bracket, matchup_cards_last_week, render_top_three, \
    build_matchups_by_week, build_picks_by_roster, render_teams_sidebar, build_week_activity, pill
from dashboard_services.trade_calculator_page import build_trade_calculator_body
from dashboard_services.utils import load_teams_index, get_nfl_games_for_week, build_games_by_team, \
    build_status_by_pid, streak_class, build_teams_overview, load_model_value_table, load_players_index, \
    load_week_projection

daily_lock = threading.Lock()
daily_completed = None

DASHBOARD_CACHE = {}  # (league_id)
CACHE_TTL = 60 * 60
VALUE_CACHE_TTL = 60 * 60 * 3  # 3 hours
daily_init_done = False
# directory to hold value-table files
VALUE_TABLE_DIR = Path(__file__).resolve().parents[0] / "data"
VALUE_TABLE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder="static",  # points to site/static
    static_url_path="/static"  # URL base for static files
)

app.secret_key = os.urandom(32)

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
  </head>
  <body>
    {nav}
    <main class="overview-layout">
      {body}
    </main>
    <script src="/static/app.js"></script>
  </body>
</html>
"""

def run_daily_data_async():
    """Start daily data build in a background thread."""
    thread = threading.Thread(
        target=build_daily_data,
        args=(CURRENT_SEASON,),
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

    filename = f"value_table_{today_str}.json"
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


def build_nav(league_id: str, active: str) -> str:
    """
    active: one of 'dashboard','standings','power','weekly','teams','activity','injuries'
    """

    def nav_pill(label: str, endpoint: str, key: str) -> str:
        cls = "nav-pill active" if key == active else "nav-pill"
        href = url_for(endpoint, league_id=league_id)
        return f"<a class='{cls}' href='{href}'>{label}</a>"

    pills = [
        nav_pill("Dashboard", "page_dashboard", "dashboard"),
        nav_pill("Standings", "page_standings", "standings"),
        nav_pill("Weekly Hub", "page_weekly", "weekly"),
        nav_pill("Trade Calc", "page_trade", "trade"),
        nav_pill("Activity", "page_activity", "activity"),
        nav_pill("Graphs", "page_graphs", "graphs"),
        "<a class='nav-pill logout-pill' href='/logout'>Logout</a>"
    ]
    return "<nav class='top-nav'><div><img src='/static/Website_Logo.png' alt='League Logo' class='site-logo'/></div><div>" + "".join(
        pills) + "</div></nav>"


def render_page(title: str, league_id: str, active: str, body_html: str) -> str:
    nav_html = build_nav(league_id, active)
    return BASE_HTML.format(title=title, nav=nav_html, body=body_html)


def build_league_context(league_id: str) -> dict:
    """
    Fetch all core data for a league once and reuse across pages.
    """
    rosters = get_rosters(league_id)
    users = get_users(league_id)
    league = get_league(league_id)
    traded = get_traded_picks(league_id)
    current = get_nfl_state()
    players = get_nfl_players()
    players_index = load_players_index()
    teams_index = load_teams_index()
    players_map = get_players_map(players)

    current_season = current.get("season")
    current_week = current.get("week")
    weeks = 18

    df_weekly, team_stats, roster_map = build_tables(
        league_id, current_week, players, users, rosters
    )
    activity_df = build_week_activity(league_id, df_weekly.get("week").max(),
                                      players_map)
    injury_df = build_injury_report(league_id, local_tz="America/New_York", include_free_agents=False)
    statuses = build_status_by_week(
        current_season,
        weeks,
        players_index,
        teams_index,
    )

    proj_by_week = build_projections_by_week(current_season, weeks)

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

    # add 'proj' column – looks up by (week, roster_id)
    df_weekly["proj"] = df_weekly.apply(
        lambda row: proj_by_roster.get(
            (int(row["week"]), str(row["roster_id"])),
            float("nan")
        ),
        axis=1,
    )

    return {
        "league_id": league_id,
        "league": league,
        "rosters": rosters,
        "users": users,
        "traded": traded,
        "current": current,
        "current_season": current_season,
        "current_week": current_week,
        "players": players,
        "players_map": players_map,
        "players_index": players_index,
        "teams_index": teams_index,
        "df_weekly": df_weekly,
        "team_stats": team_stats,
        "roster_map": roster_map,
        "statuses": statuses,
        "matchups_by_week": matchups_by_week,
        "proj_by_week": proj_by_week,
        "weeks": weeks,
        "injury_df": injury_df,
        "activity_df": activity_df,
    }


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
        <table class="standings-table">
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


def build_status_for_week(season, week, players_index, teams_index):
    games = get_nfl_games_for_week(week, season)
    games_by_team = build_games_by_team(games)
    return build_status_by_pid(players_index, games_by_team, teams_index, week)


def build_dashboard_body(ctx: dict) -> str:
    league_id = ctx["league_id"]
    league = ctx["league"]
    rosters = ctx["rosters"]
    users = ctx["users"]
    traded = ctx["traded"]
    current_week = ctx["current_week"]
    players_map = ctx["players_map"]
    df_weekly = ctx["df_weekly"]
    team_stats = ctx["team_stats"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    statuses = ctx["statuses"]
    proj_by_week = ctx["proj_by_week"]
    matchups_by_week = ctx["matchups_by_week"]

    # --- Standings snapshot (top card in main column) ---
    standings_html = render_standings(team_stats, 5)

    # --- Finalized games + last_final_week (for proj cutoff) ---
    finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())
    else:
        # fall back to current week if nothing is finalized yet
        last_final_week = current_week

    slides = []
    for m in matchups_by_week.get(current_week, []):
        slides.append(
            render_matchup_slide(
                m,
                current_week,
                last_final_week,
                status_by_pid=statuses[current_week].get("statuses", {}),
                projections=proj_by_week,
                players=players_index,
                teams=teams_index,
            )
        )

    slides_by_week = {current_week: "".join(slides)}
    matchup_html = render_matchup_carousel_weeks(slides_by_week, True)

    # --- Awards / recap style info (season-level or last week) ---
    awards = compute_awards_season(finalized_df, players_map, league_id)
    awards_html = render_awards_section(awards)

    # --- Teams sidebar (right-hand side) ---
    picks_by_roster = build_picks_by_roster(
        num_future_seasons=3,
        league=league,
        rosters=rosters,
        traded=traded,
    )

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
    <div class="overview-main">
      <div class="card central">
        <h2>Standings</h2>
        {standings_html}
      </div>
      {matchup_html}
      {awards_html}
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

    bracket_html = playoff_bracket(
        wb,
        roster_name_map=roster_map,
        roster_avatar_map=roster_avatar_map,
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

    stats_tbl = (team_stats.rename(columns={"owner": "Team", "AVG": "Average", "STD": "Std Dev", "WinPct": "Win %"})
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

    standings_html = render_standings(team_stats, 10)
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
    df_weekly = ctx["df_weekly"]
    roster_map = ctx["roster_map"]
    players_map = ctx["players_map"]
    current_week = ctx["current_week"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    rosters = ctx["rosters"]
    users = ctx["users"]
    proj_by_week = ctx["proj_by_week"]
    weeks = ctx["weeks"]
    statuses = ctx["statuses"]

    finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())
    else:
        last_final_week = current_week

    default_week = current_week if current_week in range(1, weeks) else weeks[-1]
    # Build all matchup slides once
    matchups_by_week = build_matchups_by_week(
        league_id, range(1, weeks), roster_map, players_map
    )
    slides_by_week: dict[int, str] = {
        w: "".join(
            render_matchup_slide(
                m,
                w,
                last_final_week,
                status_by_pid=statuses[w].get("statuses", {}),
                projections=proj_by_week,
                players=players_index,
                teams=teams_index,
            )
            for m in matchups_by_week.get(w, [])
        )
        for w in range(1, weeks)
    }

    # One global carousel (will listen to #hubWeek)
    matchup_html = render_matchup_carousel_weeks(slides_by_week, False)

    # Week dropdown HTML
    options = []
    for w in range(1, weeks):
        sel = " selected" if w == default_week else ""
        options.append(f"<option value='{w}'{sel}>Week {w}</option>")
    week_select_html = "".join(options)

    main_panels = []
    side_panels = []

    for w in range(1, weeks):
        active_cls = " active" if w == default_week else ""

        # Top scorers are truly per-week
        top_scorers_html = _render_weekly_top_scorers_for_week(
            league_id,
            df_weekly,
            roster_map,
            players_map,
            proj_by_week,
            rosters,
            w,
            users
        )
        highlights_html = _render_weekly_highlights(df_weekly, w)

        # Left-side weekly pane (this will swap)
        main_panels.append(f"""
              <div class="week-main-panel{active_cls}" data-week="{w}">
                {top_scorers_html}
              </div>
        """)

        # Sidebar weekly pane (already working)
        side_panels.append(f"""
              <div class="week-side-panel{active_cls}" data-week="{w}">
                {highlights_html}
              </div>
        """)

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

        <!-- two-column main area like your standings hub -->
        <div class="standings-main two-col-standings">
          <div class="standings-col">
            <div class="week-main-panels">
              {''.join(main_panels)}
            </div>
          </div>
          <div class="standings-col">
            {matchup_html}
          </div>
        </div>
      </main>

      <aside class="page-sidebar">
        <div class="week-side-panels">
          {''.join(side_panels)}
        </div>
      </aside>
    </div>

    <script>
    (function() {{
      var sel = document.getElementById('hubWeek');
      if (!sel) return;

      sel.addEventListener('change', function() {{
        var w = this.value;

        // toggle main & side weekly panels
        document.querySelectorAll('.week-main-panel').forEach(function(el) {{
          el.classList.toggle('active', el.getAttribute('data-week') === w);
        }});
        document.querySelectorAll('.week-side-panel').forEach(function(el) {{
          el.classList.toggle('active', el.getAttribute('data-week') === w);
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
            bundles[w] = {
                "projections": projections,
            }
        except Exception as e:
            print(f"Error loading week {w} projections: {e}")
            bundles[w] = {"projections": {}}
    return bundles


def build_status_by_week(season: int, weeks: int, players_index, teams_index):
    bundles = {}
    for w in range(1, weeks):
        try:
            statuses = build_status_for_week(season, w, players_index, teams_index)
            bundles[w] = {
                "statuses": statuses,
            }
        except Exception as e:
            print(f"Error loading week {w} schedule: {e}")
            bundles[w] = {"statuses": {}}
    return bundles

def build_activity_body(ctx: dict) -> str:
    league_id = ctx["league_id"]
    activity_df = ctx.get("activity_df")
    injury_df = ctx.get("injury_df")  # from build_injury_report(...)

    # ---------- EXISTING ACTIVITY CARD HTML ----------
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
                return (
                    f"<div class='player'><span class='io {io_class}'>"
                    f"{'+' if io_class == 'add' else '−'}</span>"
                    f"<div><div style='font-weight:600'>{season} {round_label}</div>"
                    f"<div style='color:#64748b;font-size:12px'>{subline}</div></div></div>"
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

            cols = []
            for tm in teams:
                roster_id = tm.get("roster_id")

                gets_players = "".join(
                    f"<div class='player'><span class='io add'>+</span>"
                    f"<div><div style='font-weight:600'>{p['name']}</div>"
                    f"<div style='color:#64748b;font-size:12px'>{p['pos']} • {p['team']}</div></div></div>"
                    for p in (tm.get("gets") or [])
                )

                gets_picks = "".join(
                    render_pick_row(pick, "add")
                    for pick in (picks_by_receiver.get(roster_id, []) if roster_id is not None else [])
                )
                gets = gets_players + gets_picks
                if not gets:
                    gets = "<div style='color:#64748b;font-size:13px'>No players</div>"

                sends_players = "".join(
                    f"<div class='player'><span class='io drop'>−</span>"
                    f"<div><div style='font-weight:600'>{p['name']}</div>"
                    f"<div style='color:#64748b;font-size:12px'>{p['pos']} • {p['team']}</div></div></div>"
                    for p in (tm.get("sends") or [])
                )

                sends_picks = "".join(
                    render_pick_row(pick, "drop")
                    for pick in (picks_by_sender.get(roster_id, []) if roster_id is not None else [])
                )
                sends = sends_players + sends_picks

                avatar = tm.get("avatar") or ""
                img = (
                    f"<img class='avatar' src='{avatar}' "
                    f"onerror=\"this.style.display='none'\">"
                    if avatar else ""
                )
                cols.append(
                    f"<div class='team-col'>"
                    f"  <header>{img}<div class='team-name'>{tm.get('name', '')}</div></header>"
                    f"  <div class='plist'>{gets}{sends}</div>"
                    f"</div>"
                )

            when = (
                txrow["ts"].astimezone(ZoneInfo("America/New_York")).strftime("%b %d, %I:%M %p")
                if pd.notna(txrow["ts"])
                else ""
            )
            return (
                f"<div class='tx trade-card activity-item' data-kind='trade'>"
                f"  <div class='meta'>{pill('Trade completed')} • {when}</div>"
                f"  <div class='teams'>{''.join(cols)}</div>"
                f"</div>"
            )

        def html_waiver(txrow):
            d = txrow["data"]
            avatar = d.get("avatar") or ""
            img = (
                f"<img class='avatar' src='{avatar}' "
                f"onerror=\"this.style.display='none'\">"
                if avatar else ""
            )
            adds = "".join(
                f"<div class='player'><span class='io add'>+</span>"
                f"<div><div style='font-weight:600'>{p['name']}</div>"
                f"<div style='color:#64748b;font-size:12px'>{p['pos']} • {p['team']}</div></div></div>"
                for p in d.get("adds", [])
            )
            when = (
                txrow["ts"].astimezone(ZoneInfo("America/New_York")).strftime("%b %d, %I:%M %p")
                if pd.notna(txrow["ts"])
                else ""
            )
            return (
                f"<div class='tx activity-item' data-kind='waiver'>"
                f"  <div class='meta'>{pill('Waiver')} • {when}</div>"
                f"  <div class='team-col'>"
                f"    <header>{img}<div class='team-name'>{d['name']}</div></header>"
                f"    <div class='plist'>{adds}</div>"
                f"  </div>"
                f"</div>"
            )

        cards = []
        for _, row in activity_df.iterrows():
            cards.append(html_trade(row) if row["kind"] == "trade" else html_waiver(row))

        # optional: keep your existing pills at top of the card


        activity_html = (
            "<div class='card activity-card' data-section='activity'>"
            "  <div class='card-header-row'>"
            "    <h2>This Week: Trades & Waiver Claims</h2>"
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

      // Injury filter (sidebar pills)
      document.querySelectorAll('.inj-toggle').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          // make this behave like radio buttons
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



@app.route("/league/<league_id>/dashboard")
def page_dashboard(league_id):
    # grab context (this will use cache or rebuild if expired)
    ctx = get_league_ctx_from_cache(league_id)
    body = build_dashboard_body(ctx)
    return render_page("Dashboard", league_id, "dashboard", body)


@app.route("/league/<league_id>/standings")
def page_standings(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body = build_standings_body(ctx)
    return render_page("Standings", league_id, "standings", body)


@app.route("/league/<league_id>/weekly")
def page_weekly(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body = build_weekly_hub_body(ctx)
    return render_page("Weekly Hub", league_id, "weekly", body)


@app.route("/league/<league_id>/trade")
def page_trade(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body = build_trade_calculator_body(ctx["league_id"], ctx["current_season"])
    return render_page("Trade Calculator", league_id, "trade", body)


@app.route("/league/<league_id>/activity")
def page_activity(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body = build_activity_body(ctx)
    return render_page("Activity", league_id, "activity", body)


@app.route("/league/<league_id>/graphs")
def page_graphs(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body_html = build_graphs_body(ctx)
    return render_page("Graphs", league_id, "graphs", body_html)


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
    if request.method == "POST":
        league_id = request.form.get("league", "").strip()
        cache_key = league_id
        now = time.time()
        print("league_id:", repr(league_id))
        print("cache keys currently:", list(DASHBOARD_CACHE.keys()))

        entry = DASHBOARD_CACHE.get(cache_key)
        if entry and (now - entry["ts"] < CACHE_TTL):
            # we already have a fresh context; jump to dashboard
            return redirect(url_for("page_dashboard", league_id=league_id))

        # build and cache context instead of HTML
        ctx = build_league_context(league_id)
        DASHBOARD_CACHE[cache_key] = {"ctx": ctx, "ts": now}

        return redirect(url_for("page_dashboard", league_id=league_id))

    # GET -> show the form (unchanged)
    default_weeks = get_nfl_state().get("week")
    return render_template_string(FORM_HTML, league=None, weeks=default_weeks)


@app.route("/league/<league_id>")
def view_league(league_id):
    # warm or reuse league context cache
    _ = get_league_ctx_from_cache(league_id)
    return redirect(url_for("page_dashboard", league_id=league_id))


@app.route("/logout")
def logout():
    # Clear the session + cached league context
    from flask import session
    session.clear()
    return redirect(url_for("index"))


# in your Flask app

CURRENT_SEASON = 2025  # or derive dynamically


@app.route("/api/trade-eval", methods=["POST"])
def api_trade_eval():
    payload = request.get_json(force=True)

    league_id = str(payload.get("league_id") or "global")
    season = int(payload.get("season") or CURRENT_SEASON)

    side_a_players = [str(pid) for pid in payload.get("side_a_players", [])]
    side_b_players = [str(pid) for pid in payload.get("side_b_players", [])]
    side_a_picks = payload.get("side_a_picks", [])
    side_b_picks = payload.get("side_b_picks", [])

    weeks = list(range(1, 19))

    # Cached table load
    value_table = load_model_value_table()
    if value_table is not None:
        value_table = build_ml_value_table(season)
        store_model_values(league_id, season, weeks, value_table)

    # ---------- Helpers ----------

    # Simple rookie pick curve
    def value_pick(pk: str) -> float:
        try:
            yr, rnd, slot = pk.split("_")
            rnd = int(rnd)
            slot = int(slot)
        except Exception:
            return 0.0

        base = 40 if rnd == 1 else 20 if rnd == 2 else 10
        decay = max(0.1, 1 - (slot - 1) * 0.04)
        return round(base * decay, 1)

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
                # Fallback if missing
                breakdown.append({
                    "type": "player",
                    "id": pid_str,
                    "name": f"Player {pid_str}",
                    "value": 0.0,
                    "position": None,
                    "team": None,
                })
                continue

            val = float(player.get("value", 0.0))
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
            val = float(value_pick(pk))
            breakdown.append({
                "type": "pick",
                "id": pk,
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
            "effective_total": raw_total,
            "adjustment": 0.0,
        }

    def apply_multi_for_one_adjustment(side_a: dict, side_b: dict) -> None:
        """
        Apply a 'value adjustment' only to the side with FEWER players.

        Intuition:
          - Look at the gap in raw value between the two sides.
          - Look at how much of a 'stud' the best player is on the fewer side.
          - Look at how many extra pieces the other side has.
          - Reverse-engineer an adjustment that moves the trade toward even
            but caps it so you don't blow things up.
        """
        vals_a = side_a["player_values"]
        vals_b = side_b["player_values"]
        n_a = len(vals_a)
        n_b = len(vals_b)

        # need players on both sides and a true imbalance
        if n_a == 0 or n_b == 0 or n_a == n_b:
            side_a["effective_total"] = side_a["raw_total"]
            side_b["effective_total"] = side_b["raw_total"]
            return

        # Decide which side is the "consolidation" side (fewer players)
        if n_a < n_b:
            fewer = side_a
            more = side_b
            fewer_is_a = True
        else:
            fewer = side_b
            more = side_a
            fewer_is_a = False

        fewer_vals = fewer["player_values"]
        more_vals = more["player_values"]
        fewer_raw = fewer["raw_total"]
        more_raw = more["raw_total"]

        # safety
        if not fewer_vals or fewer_raw <= 0:
            side_a["effective_total"] = side_a["raw_total"]
            side_b["effective_total"] = side_b["raw_total"]
            return

        # How many extra pieces is the other side sending?
        extra_pieces = max(len(more_vals) - len(fewer_vals), 0)

        # "Stud" score: best player on the consolidation side
        stud_val = max(fewer_vals)
        stud_score = min(stud_val / 1000.0, 1.0)  # 0–1

        # How far apart are they in raw value?
        gap = abs(more_raw - fewer_raw)

        # Base adj comes from the gap – we don't usually bridge 100% of the gap,
        # but we move it a good chunk of the way.
        base_from_gap = gap * (0.45 + 0.25 * stud_score)  # ~45–70% of gap

        # Extra bonus for each additional piece on the "more" side
        piece_bonus = fewer_raw * 0.03 * extra_pieces  # 3% per extra piece

        # Small extra weight for true monsters
        stud_bonus = fewer_raw * 0.10 * stud_score  # up to +10% of side

        raw_adj = base_from_gap + piece_bonus + stud_bonus

        # Cap overall adjustment so it can't exceed ~25% of the consolidation side
        cap = fewer_raw * 0.25
        adj = min(raw_adj, cap)

        if fewer_is_a:
            side_a["adjustment"] = adj
            side_b["adjustment"] = 0.0
        else:
            side_a["adjustment"] = 0.0
            side_b["adjustment"] = adj

        side_a["effective_total"] = side_a["raw_total"] + side_a["adjustment"]
        side_b["effective_total"] = side_b["raw_total"] + side_b["adjustment"]

    # ---------- Build both sides (raw) ----------
    players_by_id = {str(p["id"]): p for p in value_table}

    side_a = build_side(side_a_players, side_a_picks)
    side_b = build_side(side_b_players, side_b_picks)

    # ---------- Apply multi-for-one consolidation adjustment ----------
    apply_multi_for_one_adjustment(side_a, side_b)

    a_eff = side_a["effective_total"]
    b_eff = side_b["effective_total"]

    diff = a_eff - b_eff
    abs_diff = abs(diff)

    # ---------- Percentage-based fair threshold ----------
    FAIR_PCT = 0.015  # 8% band; tweak as needed
    baseline = max(a_eff, b_eff, 1.0)
    fair_band = baseline * FAIR_PCT

    if abs_diff <= fair_band:
        pct = (abs_diff / baseline) * 100.0
        verdict = f"This trade looks very fair (about {pct:.1f}% apart)."
    elif diff > 0:
        verdict = f"Side A is favored by about {abs_diff:.1f} value."
    else:
        verdict = f"Side B is favored by about {abs_diff:.1f} value."

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

    # Try to use the cached value table
    model_value_table = get_cached_model_value_table(league_id, season, weeks)
    if model_value_table is None:
        print(f"[league_players] cache MISS for league={league_id}, season={season}")
        model_value_table = build_ml_value_table(season)
        store_value_table(league_id, season, weeks, model_value_table)
    else:
        print(f"[league_players] cache HIT for league={league_id}, season={season}")

    cleaned_players = _sanitize_for_json(model_value_table)
    return jsonify(cleaned_players)


if __name__ == "__main__":
    app.run(debug=True)
