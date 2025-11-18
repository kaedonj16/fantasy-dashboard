import json
import numpy as np
import pandas as pd
import time
from flask import Flask, request, render_template_string, redirect, url_for
from pathlib import Path

from dashboard_services.api import get_rosters, get_users, get_league, get_traded_picks, get_nfl_players, \
    get_nfl_state, get_bracket
from dashboard_services.players import get_players_map
from dashboard_services.service import build_tables, playoff_bracket
from dashboard_services.utils import load_players_index, load_teams_index, get_nfl_games_for_week, build_games_by_team, \
    build_status_by_pid, path_week_proj, _streak_class
from league_dashboard import build_matchups_by_week, _render_matchup_slide, render_matchup_carousel_weeks, \
    compute_awards_season, render_awards_section, build_picks_by_roster, build_teams_overview, render_teams_sidebar

DASHBOARD_CACHE = {}  # (league_id)
CACHE_TTL = 60 * 60  # seconds (10 minutes)
app = Flask(
    __name__,
    static_folder="static",  # points to site/static
    static_url_path="/static"  # URL base for static files
)

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
    <div><img src="/static/BR_Logo.png" alt="League Logo" class="site-logo"/></div>
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


def build_nav(league_id: str, active: str) -> str:
    """
    active: one of 'dashboard','standings','power','weekly','teams','activity','injuries'
    """

    def pill(label: str, endpoint: str, key: str) -> str:
        cls = "nav-pill active" if key == active else "nav-pill"
        href = url_for(endpoint, league_id=league_id)
        return f"<a class='{cls}' href='{href}'>{label}</a>"

    pills = [
        pill("Dashboard", "page_dashboard", "dashboard"),
        pill("Standings", "page_standings", "standings"),
        pill("Weekly Hub", "page_weekly", "weekly"),
        pill("Teams", "page_teams", "teams"),
        pill("Activity", "page_activity", "activity"),
        pill("Injuries", "page_injuries", "injuries"),
    ]
    return "<nav class='top-nav'><div><img src='/static/BR_Logo.png' alt='League Logo' class='site-logo' style='height:50px;'/></div><div>" + "".join(
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

    df_weekly, team_stats, roster_map = build_tables(
        league_id, current_week, players, users, rosters
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
    }


def get_league_ctx_from_cache(league_id: str) -> dict:
    now = time.time()
    entry = DASHBOARD_CACHE.get(league_id)
    if entry and (now - entry["ts"] < CACHE_TTL):
        return entry["ctx"]

    ctx = build_league_context(league_id)
    DASHBOARD_CACHE[league_id] = {"ctx": ctx, "ts": now}
    return ctx


def render_standings_snapshot(team_stats) -> str:
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

    return f"""
    <div class="card central">
        <h2>Standings</h2>
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
            {''.join(rows[:5])}
          </tbody>
        </table>
    </div>
    """


def _load_week_projection_bundle(season: int, w: int):
    # projections
    proj_path = Path(path_week_proj(season, w))
    if not proj_path.exists():
        get_week_projections_cached(season, w, fetch_week_from_tank01)

    with open(proj_path, "r", encoding="utf-8") as f:
        projections = json.load(f)

    # players / teams index
    player_path = Path("cache/players_index.json")
    team_path = Path("cache/teams_index.json")
    with open(player_path, "r", encoding="utf-8") as f:
        players = json.load(f)
    with open(team_path, "r", encoding="utf-8") as f:
        teams = json.load(f)

    return projections, players, teams


def _build_status_for_week(season, week, players_index, teams_index):
    games = get_nfl_games_for_week(week=week, season=season)
    games_by_team = build_games_by_team(games)
    return build_status_by_pid(players_index, games_by_team, teams_index, week)


def build_dashboard_body(ctx: dict) -> str:
    league_id = ctx["league_id"]
    league = ctx["league"]
    rosters = ctx["rosters"]
    users = ctx["users"]
    traded = ctx["traded"]
    current_season = ctx["current_season"]
    current_week = ctx["current_week"]
    players_map = ctx["players_map"]
    df_weekly = ctx["df_weekly"]
    team_stats = ctx["team_stats"]
    roster_map = ctx["roster_map"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]

    # --- Standings snapshot (top card in main column) ---
    standings_html = render_standings_snapshot(team_stats)

    # --- Finalized games + last_final_week (for proj cutoff) ---
    finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())
    else:
        # fall back to current week if nothing is finalized yet
        last_final_week = current_week

    # --- Matchup carousel (ONLY current week for dashboard) ---
    weeks = [current_week]
    matchups_by_week = build_matchups_by_week(
        league_id, weeks, roster_map, players_map
    )

    # 1. Status per player for this week (do this ONCE)
    # if your helper only takes (season, week), drop players_index/teams_index
    status_by_pid = _build_status_for_week(
        current_season,
        current_week,
        players_index,
        teams_index,
    )

    # 2. Load projections / players / teams for this week (ONCE)
    projections, players, teams = _load_week_projection_bundle(current_season, current_week)

    # 3. Build slides for ONLY current week
    slides = []
    for m in matchups_by_week.get(current_week, []):
        slides.append(
            _render_matchup_slide(
                m,
                current_week,
                last_final_week,
                status_by_pid=status_by_pid,
                projections=projections,
                players=players,
                teams=teams,
            )
        )

    slides_by_week = {current_week: "".join(slides)}
    matchup_html = render_matchup_carousel_weeks(slides_by_week)

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
      {standings_html}
      {matchup_html}
      {awards_html}
    </div>
    <aside class="overview-sidebar">
      {teams_sidebar_html}
    </aside>
    """

    return body


def render_full_standings(team_stats) -> str:
    """
    Render a full standings table.

    Tries to be robust to different column names in team_stats.
    Expected-ish columns:
      - 'owner' or 'team_name' or 'display_name'
      - 'record' OR wins/losses/ties
      - 'pf' or 'points_for'
      - 'pa' or 'points_against'
      - 'avg' or 'points_avg'
      - 'stdev' or 'std'
    """

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

    return f"""
        <div class="card">
            <h2>Standings</h2>
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
                {''.join(rows)}
              </tbody>
            </table>
        </div>
        """


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
        streak_frame_cls = _streak_class(row)  # assumes you already have this helper
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
        css_cls = _streak_class(row)
        print(css_cls)
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

    # Best offense (highest PF)
    best_off = None
    if "PF" in ts.columns:
        best_off = ts.loc[ts["PF"].idxmax()]

    # Best defense (lowest PA)
    best_def = None
    if "PA" in ts.columns:
        best_def = ts.loc[ts["PA"].idxmin()]

    # Longest active streak (by StreakLen, if you have it)
    hottest = None
    coldest = None
    if "StreakLen" in ts.columns and "StreakType" in ts.columns:
        # positive streaks
        hot_df = ts[ts["StreakType"] == "W"]
        if not hot_df.empty:
            hottest = hot_df.loc[hot_df["StreakLen"].idxmax()]

        # negative streaks
        cold_df = ts[ts["StreakType"] == "L"]
        if not cold_df.empty:
            coldest = cold_df.loc[cold_df["StreakLen"].idxmax()]

    sections = []

    # League Snapshot card
    snapshot_rows = []
    if best_off is not None:
        snapshot_rows.append(
            f"""
            <div class='mini-row'>
              <div class='mini-label'>Best Offense</div>
              <div class='mini-value'>
                <span class='mini-team'>{best_off['owner']}</span>
                <span class='mini-stat'>{best_off['PF']:.1f} PF</span>
              </div>
            </div>
            """
        )
    if best_def is not None:
        snapshot_rows.append(
            f"""
            <div class='mini-row'>
              <div class='mini-label'>Best Defense</div>
              <div class='mini-value'>
                <span class='mini-team'>{best_def['owner']}</span>
                <span class='mini-stat'>{best_def['PA']:.1f} PA</span>
              </div>
            </div>
            """
        )

    # Streaks card
    streak_rows = []
    if hottest is not None:
        streak_rows.append(
            f"""
            <div class='mini-row hot gradient-hot'>
              <div class='mini-label'>Hottest Team</div>
              <div class='mini-value'>
                <span class='mini-team'>{hottest['owner']}</span>
                <span class='streak-pill w'>{hottest['Streak']}</span>
              </div>
            </div>
            """
        )

    if coldest is not None:
        streak_rows.append(
            f"""
            <div class='mini-row cold gradient-cold'>
              <div class='mini-label'>Coldest Team</div>
              <div class='mini-value'>
                <span class='mini-team'>{coldest['owner']}</span>
                <span class='streak-pill l'>{coldest['Streak']}</span>
              </div>
            </div>
            """
        )

    if snapshot_rows:
        sections.append(
            "<div class='card mini-card'>"
            "<div class='card-header'><h3>League Snapshot</h3></div>"
            "<div class='mini-body'>"
            + "".join(snapshot_rows) +
            "</div></div>"
        )

    if streak_rows:
        sections.append(
            "<div class='card mini-card'>"
            "<div class='card-header'><h3>Streaks</h3></div>"
            "<div class='mini-body'>"
            + "".join(streak_rows) +
            "</div></div>"
        )

    return "".join(sections)


def build_standings_body(ctx: dict) -> str:
    team_stats = ctx["team_stats"]
    roster_map = ctx["roster_map"]

    standings_html = render_full_standings(team_stats)
    power_playoffs_html = render_power_and_playoffs(team_stats, roster_map, ctx["league_id"])
    sidebar_html = render_standings_sidebar(team_stats)

    body = f"""
    <div class="standings-main two-col-standings">
      <div class="standings-col">
        {standings_html}
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
    body = """
    <div class="overview-main">
      <div class="card"><h2>Weekly Hub</h2></div>
    </div>
    <aside class="overview-sidebar"></aside>
    """
    return render_page("Weekly Hub", league_id, "weekly", body)


@app.route("/league/<league_id>/teams")
def page_teams(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body = """
    <div class="overview-main">
      <div class="card"><h2>Teams & Rosters</h2></div>
    </div>
    <aside class="overview-sidebar"></aside>
    """
    return render_page("Teams", league_id, "teams", body)


@app.route("/league/<league_id>/activity")
def page_activity(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body = """
    <div class="overview-main">
      <div class="card"><h2>Transactions & Activity</h2></div>
    </div>
    <aside class="overview-sidebar"></aside>
    """
    return render_page("Activity", league_id, "activity", body)


@app.route("/league/<league_id>/injuries")
def page_injuries(league_id):
    ctx = get_league_ctx_from_cache(league_id)
    body = """
    <div class="overview-main">
      <div class="card"><h2>Injuries & News</h2></div>
    </div>
    <aside class="overview-sidebar"></aside>
    """
    return render_page("Injuries", league_id, "injuries", body)


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
    return redirect(url_for("page_dashboard", league_id=league_id))


if __name__ == "__main__":
    app.run(debug=True)
