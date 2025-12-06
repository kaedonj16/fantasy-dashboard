#!/usr/bin/env python3
"""
League Blueprint Report Generator for Sleeper Fantasy Football

This script generates comprehensive analytics and visualizations for Sleeper fantasy football leagues.
It pulls data from the Sleeper API and creates both PDF reports and interactive dashboards.

Usage:
    python league_dashboard.py --league <league_id> --weeks <num_weeks>

Features:
    - Pulls users, rosters, and matchups from Sleeper API (read-only)
    - Calculates team metrics (PF, PA, AVG, MAX, MIN, STD, Win%)
    - Generates visualizations in a consistent theme
    - Exports data to CSV and interactive HTML
    - Creates PDF reports with multiple visualizations
"""

# Standard library imports
import argparse
import json
import math
# Third-party imports
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import pathlib
import plotly.graph_objs as go
import random
import requests
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from plotly.offline import plot as plotly_plot, get_plotlyjs
from reportlab.lib import colors
from typing import Dict, Any, Iterable, Tuple, Optional, List, Union, Callable
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

from dashboard_services.api import get_rosters, get_users, get_league, get_nfl_players, \
    get_nfl_state, get_bracket
from dashboard_services.awards import compute_weekly_highlights, compute_awards_season, render_awards_section
from dashboard_services.injuries import render_injury_accordion, build_injury_report
from dashboard_services.matchups import render_matchup_carousel_weeks, render_matchup_slide
from dashboard_services.players import get_players_map
from dashboard_services.service import build_tables, playoff_bracket, pill
from dashboard_services.service import render_weekly_highlight_ticker, render_week_recap_tab, \
    build_week_activity, render_standings_table, build_matchups_by_week
from dashboard_services.styles import activity_css, logoCss, injury_script, js_sort_filter, \
    activity_filter_js, NAV_JS, NAV_CSS
from dashboard_services.utils import streak_class, z_better_outward, load_teams_index, load_model_value_table, \
    load_week_schedule, build_status_for_week


# === WEEK RECAP + TOP-3 SIDEBAR ==============================================

def build_interactive_site(df_weekly: pd.DataFrame,
                           team_stats: pd.DataFrame,
                           out_dir: str = "site",
                           activity_df: Optional[pd.DataFrame] = None,
                           injury_df: Optional[pd.DataFrame] = None,
                           matchup_html: Optional[pd.DataFrame] = None,
                           recap_html: Optional[pd.DataFrame] = None,
                           awards_html: Optional[pd.DataFrame] = None,
                           league_id: Optional[str] = None,
                           roster_map: Optional[Dict[str, str]] = None,
                           teams_html: Optional[pd.DataFrame] = None
                           ):
    """
    Creates /site/index.html with:
      - STATS (top): sortable, multi-sort table (no Record column)
      - POWER RANKINGS: podium top-3 (team + record)
      - PF vs PA (interactive scatter + trendline)
      - Weekly Scores (interactive line + league avg)
      - Score Distribution (interactive box, VERTICAL by team)
      - Radar Comparison (two dropdowns to compare any two teams)
      - This Week: Trades & Waiver Adds (using Sleeper team names, player names with NFL teams)
    """
    os.makedirs(out_dir, exist_ok=True)
    owners = team_stats["owner"].tolist()

    def _z(series):
        s = pd.Series(series, dtype="float64")
        sd = float(s.std(ddof=0))
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / sd

    if "Win%" in team_stats.columns:
        win_pct = team_stats["Win%"].fillna(0.0)
    else:
        ties = team_stats["Ties"] if "Ties" in team_stats.columns else 0
        win_pct = ((team_stats["Wins"] + 0.5 * ties)
                   / team_stats["G"].replace(0, np.nan)).fillna(0.0)

    if "Last3" not in team_stats.columns:
        last3 = (
            df_weekly[df_weekly["finalized"] == True].copy().sort_values(["owner", "week"])
            .groupby("owner")["points"]
            .apply(lambda s: s.tail(3).mean() if len(s) else 0.0)
            .rename("Last3")
            .reset_index()
        )
        team_stats = team_stats.merge(last3, on="owner", how="left")
    team_stats["Last3"] = team_stats["Last3"].fillna(0.0)

    avg_pts = team_stats["AVG"].fillna(0.0)
    last3 = team_stats["Last3"].fillna(0.0)
    cons_inv = -team_stats["STD"].fillna(team_stats["STD"].mean())
    ceiling = team_stats["MAX"].fillna(0.0)

    team_stats["Z_WinPercentage"] = _z(win_pct)
    team_stats["Z_Avg"] = _z(avg_pts)
    team_stats["Z_Last3"] = _z(last3)
    team_stats["Z_Consistency"] = _z(cons_inv)
    team_stats["Z_Ceiling"] = _z(ceiling)

    W_WIN, W_AVG, W_LAST3, W_CONS, W_CEIL = 0.2, 0.3, 0.15, 0.20, 0.15
    team_stats["Win%"] = win_pct
    team_stats["PowerScore"] = (
            W_WIN * team_stats["Z_WinPercentage"] +
            W_AVG * team_stats["Z_Avg"] +
            W_LAST3 * team_stats["Z_Last3"] +
            W_CONS * team_stats["Z_Consistency"] +
            W_CEIL * team_stats["Z_Ceiling"]
    )

    pr_sorted = team_stats.sort_values(["PowerScore", "PF"], ascending=[False, False]).reset_index(drop=True)
    top3 = pr_sorted.head(3)

    wk_avg = df_weekly.groupby("week")["points"].mean().reset_index()

    metrics = ["PF", "PA", "MAX", "MIN", "AVG", "STD"]
    Z = z_better_outward(team_stats, metrics)
    theta = metrics
    z_map = {team_stats.loc[i, "owner"]: Z.iloc[i].values.astype(float).tolist() for i in range(len(team_stats))}

    figs = {}

    scatter_traces = []
    for _, r in team_stats.iterrows():
        scatter_traces.append(go.Scatter(
            x=[r["PA"]], y=[r["PF"]],
            mode="markers+text",
            text=[r["owner"]], textposition="top center",
            marker=dict(size=12, line=dict(color="black", width=1)),
            name=r["owner"]
        ))
    x = team_stats["PA"].values
    y = team_stats["PF"].values
    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        m = ((x - x.mean()) * (y - y.mean())).sum() / max(((x - x.mean()) ** 2).sum(), 1e-9)
        b = y.mean() - m * x.mean()
        xs = [float(min(x) * 0.95), float(max(x) * 1.05)]
        ys = [m * xs[0] + b, m * xs[1] + b]
        scatter_traces.append(go.Scatter(x=xs, y=ys, mode="lines",
                                         line=dict(dash="dash"), name="Trend"))
    figs["pf_pa"] = go.Figure(scatter_traces)
    figs["pf_pa"].update_layout(title="PF vs PA",
                                xaxis_title="Points Against (PA)",
                                yaxis_title="Points For (PF)",
                                hovermode="closest")

    line_traces = [go.Scatter(x=wk_avg["week"], y=wk_avg["points"],
                              mode="lines", name="League Avg",
                              line=dict(dash="dash", width=3), opacity=0.7)]
    for owner, g in df_weekly.sort_values("week").groupby("owner"):
        line_traces.append(go.Scatter(x=g["week"], y=g["points"],
                                      mode="lines+markers", name=owner))
    figs["scores_line"] = go.Figure(line_traces)
    figs["scores_line"].update_layout(title="Weekly Scores by Team (interactive)",
                                      xaxis_title="Week", yaxis_title="Points",
                                      hovermode="x unified")

    order = df_weekly.groupby("owner")["points"].median().sort_values(ascending=False).index.tolist()
    box_traces = []
    for o in order:
        pts = df_weekly.loc[df_weekly["owner"] == o, "points"]
        box_traces.append(go.Box(y=pts, name=o, boxmean=True, orientation="v",
                                 hoveron="boxes", boxpoints=False))
    figs["scores_box"] = go.Figure(box_traces)
    figs["scores_box"].update_layout(title="Scores Distribution by Team (interactive)",
                                     xaxis_title="Team", yaxis_title="Points", hovermode="closest")

    t1 = pr_sorted.iloc[0]["owner"]
    t2 = pr_sorted.iloc[1]["owner"] if len(pr_sorted) > 1 else pr_sorted.iloc[0]["owner"]

    def radar_compare_fig(a, b):
        return go.Figure([
            go.Scatterpolar(r=[0] * len(theta) + [0], theta=theta + theta[:1],
                            name="League Avg", line=dict(dash="dash"), opacity=0.8),
            go.Scatterpolar(r=z_map[a] + [z_map[a][0]], theta=theta + theta[:1],
                            fill="toself", name=a, opacity=0.45),
            go.Scatterpolar(r=z_map[b] + [z_map[b][0]], theta=theta + theta[:1],
                            fill="toself", name=b, opacity=0.45),
        ])

    figs["radar_cmp"] = radar_compare_fig(t1, t2)
    figs["radar_cmp"].update_layout(title="Radar Comparison (select two teams)",
                                    polar=dict(radialaxis=dict(visible=False)),
                                    showlegend=True)
    plotly_js = f'<script>{get_plotlyjs()}</script>'

    div_pfpa = plotly_plot(figs["pf_pa"], include_plotlyjs=False, output_type="div")
    div_line = plotly_plot(figs["scores_line"], include_plotlyjs=False, output_type="div")
    div_box = plotly_plot(figs["scores_box"], include_plotlyjs=False, output_type="div")

    best = df_weekly.groupby("owner")["points"].max().rename("Best Week")
    worst = df_weekly.groupby("owner")["points"].min().rename("Worst Week")
    stats_tbl = (team_stats.rename(columns={"owner": "Team", "AVG": "Average", "STD": "Std Dev", "WinPct": "Win %"})
                 .merge(best, left_on="Team", right_index=True, how="left")
                 .merge(worst, left_on="Team", right_index=True, how="left"))

    cols = ["Team", "Win %", "PF", "PA", "Average", "Std Dev", "Best Week", "Worst Week", "avatar"]
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

    standings_html = render_standings_table(team_stats, 10)
    cols = ["Team", "Win %", "PF", "PA", "Average", "Std Dev", "Best Week", "Worst Week"]

    table_html = f"""
      <div class="table-wrap">
        <table id="stats" class="standings-table">
          <thead><tr>{"".join([f"<th data-col='{i}'>{c}</th>" for i, c in enumerate(cols)])}</tr></thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
      </div>
    """

    div_stats = f"""
      <div class="card stats" data-section="overview">
        <div class="card-tabs" data-card="stats-tabs">
          <div class="tab-strip">
            <button class="tab-btn active" data-tab="standings">Standings</button>
            <button class="tab-btn" data-tab="details">Detailed Stats</button>
          </div>
    
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
      </div>
    """

    p = pr_sorted["PowerScore"];
    pmin, pmax = float(p.min()), float(p.max())

    def pct(v):
        return 100.0 if pmax == pmin else max(2.0, (v - pmin) / (pmax - pmin) * 100.0)

    pfpg = (pr_sorted["PF"] / pr_sorted["G"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    papg = (pr_sorted["PA"] / pr_sorted["G"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    diff = pfpg - papg

    others = pr_sorted.iloc[3:].reset_index(drop=True)
    rank_cards = []
    for i, row in others.iterrows():
        pos = i + 4
        team = row["owner"]
        record = f"{int(row['Wins'])}-{int(row['G'] - row['Wins'])}" + (
            f"-{int(row.get('Ties', 0))}" if int(row.get('Ties', 0)) else "")
        bar_w = pct(row["PowerScore"])
        diff_val = diff.iloc[i + 3]
        diff_class = "diff-pos" if diff_val > 0 else "diff-neg" if diff_val < 0 else ""
        streak_chip = row.get("Streak", "")  # e.g., "W3", "L2" (computed earlier)
        chips_html = (
            f"<span class='chip'>PF/G {pfpg.iloc[i + 3]:.1f}</span>"
            f"<span class='chip'>PA/G {papg.iloc[i + 3]:.1f}</span>"
        )
        chips_html += f"<span class='chip chip-streak'>{streak_chip}</span>"
        chips_html += f"<span class='chip {diff_class}'>{diff_val:+.1f}</span>"
        if streak_chip:
            pass
        avatar_url = row.get("avatar")
        img = f"<img class='avatar sm' src='{avatar_url}' onerror=\"this.style.display='none'\">" if avatar_url else ""
        css_cls = streak_class(row)
        rank_cards.append(
            f"<div class='rank-item {css_cls} '>"
            f"<span class='pos'>#{pos}</span>"
            f"<span class='name'>{img}&nbsp;{team}</span>"
            f"<span class='rec'>{record}</span>"
            f"<div class='power-row'><div class='bar'><div style='width:{bar_w:.1f}%'></div></div><div class='chips'>{chips_html}</div></div>"
            f"</div>"
        )
    rankings_html = "<div class='rank-grid'>" + "".join(rank_cards) + "</div>"

    def podium_slot(rank, row):
        name = row['owner']
        rec = f"{int(row['Wins'])}-{int(row['G'] - row['Wins'])}" + (
            f"-{int(row.get('Ties', 0))}" if int(row.get('Ties', 0)) else ""
        )
        size = {"1": "38px", "2": "32px", "3": "32px"}[str(rank)]
        base_cls = {1: "first", 2: "second", 3: "third"}[rank]
        w = pct(row["PowerScore"])

        # NEW: streak bits
        streak_chip = row.get("Streak", "")  # e.g., "W3", "L2" (computed earlier)
        streak_frame_cls = streak_class(row)  # "streak-hot"/"streak-cold"/""

        avatar_url = row.get("avatar")
        avatar_html = f"<img class='avatar' src='{avatar_url}' onerror=\"this.style.display='none'\">" if avatar_url else ""

        pfpg_v = row["PF"] / row["G"] if row["G"] else 0
        papg_v = row["PA"] / row["G"] if row["G"] else 0
        diff_v = pfpg_v - papg_v
        diff_class = "diff-pos" if diff_v > 0 else "diff-neg" if diff_v < 0 else ""

        chips_html = "<div class='chips'>"
        chips_html += f"<span class='chip'>PF/G {pfpg_v:.1f}</span>"
        chips_html += f"<span class='chip'>PA/G {papg_v:.1f}</span>"
        chips_html += f"<span class='chip {diff_class}'>{diff_v:+.1f}</span>"
        if streak_chip:
            streak = f"<span class='chip chip-streak'>{streak_chip}</span>"
        chips_html += "</div>"

        # Add streak class to the slot’s outer div
        return f"""
          <div class="slot {base_cls} {streak_frame_cls}">
            <div class="wrap">
              <div class='podium-header'>
                <h3 style="font-size:{size}">#{rank}</h3>
                {avatar_html}
              </div>
              <div class="name">{name}</div>
              <div class="rec">{rec} {streak}</div>
              <div class="bar"><div style="width:{w:.1f}%"></div></div>
              {chips_html}
            </div>
          </div>
        """

    podium_html = f"""
      <div class="podium">
        {podium_slot(1, top3.iloc[0])}
        {podium_slot(2, top3.iloc[1]) if len(top3) > 1 else ''}
        {podium_slot(3, top3.iloc[2]) if len(top3) > 2 else ''}
      </div>
    """

    try:
        # if you wire winners_bracket into build_interactive_site(...)
        wb = get_bracket(league_id, 'winners')  # or use a real parameter
        if wb:
            # build a simple id -> avatar map from team_stats if you don't already have one
            roster_avatar_map = {
                str(owner): av
                for owner, av in zip(team_stats["owner"], team_stats["avatar"])
                if pd.notna(owner)
            }
            # roster_map in main() is exactly the roster_id -> owner map
            # so you can pass that into build_interactive_site as roster_name_map
            bracket_html = playoff_bracket(
                wb,
                roster_name_map=roster_map,  # pass from caller
                roster_avatar_map=roster_avatar_map,
            )
        else:
            bracket_html = "<div class='card po-empty'>No playoff bracket available.</div>"
    except Exception:
        bracket_html = "<div class='card po-empty'>No playoff bracket available.</div>"

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

    def _opts(selected):
        return "".join([f"<option value='{o}'{' selected' if o == selected else ''}>{o}</option>" for o in owners])

    radar_cmp_card = f"""
      <div class="card" data-section="graphs">
        <h2>Radar Comparison</h2>
        <div id="radar-controls" style="display:flex;gap:10px;align-items:center;margin-bottom:8px;flex-wrap:wrap">
          <label for="radarA">Team A</label>
          <select id="radarA">{_opts(t1)}</select>
          <label for="radarB">Team B</label>
          <select id="radarB">{_opts(t2)}</select>
        </div>
        <div id="radar-cmp"></div>
      </div>
    """

    def html_trade(txrow):
        data = txrow["data"]
        teams = data["teams"]
        users = get_users(league_id)

        # Build a lookup: roster_id -> team name (for "(from Team X)" on picks)
        rid_to_name = {}
        for tm in teams:
            rid = tm.get("roster_id")
            if rid is not None:
                rid_to_name[rid] = tm.get("name") or f"Team {rid}"

        # Helper to render a single draft pick row in the same visual style as players
        def render_pick_row(pick, io_class):
            # Example label: "2027 1st" / "2027 2nd"
            rnd_suffix = {1: "st", 2: "nd", 3: "rd"}.get(pick.get("round"), "th")
            round_label = f"{pick.get('round')}" + rnd_suffix
            season = str(pick.get("season") or "")
            # original owner (roster that the pick originates from)
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
            # e.g., subline: "Pick • from Caleb"
            subline = f"{orig_name}'s Pick"
            # mimic the player block structure
            return (
                f"<div class='player'><span class='io {io_class}'>"
                f"{'+' if io_class == 'add' else '−'}</span>"
                f"<div><div style='font-weight:600'>{season} {round_label}</div>"
                f"<div style='color:#64748b;font-size:12px'>{subline}</div></div></div>"
            )

        # Index draft picks by receiver and sender for quick lookup
        draft_picks = data.get("draft_picks", []) or []
        picks_by_receiver = {}
        picks_by_sender = {}
        for dp in draft_picks:
            recv = dp.get("owner_id")  # who ends up with the pick
            send = dp.get("previous_owner_id")  # who gave the pick away
            if recv is not None:
                picks_by_receiver.setdefault(recv, []).append(dp)
            if send is not None:
                picks_by_sender.setdefault(send, []).append(dp)

        cols = []
        for tm in teams:
            roster_id = tm.get("roster_id")
            # Players received
            gets_players = "".join([
                f"<div class='player'><span class='io add'>+</span>"
                f"<div><div style='font-weight:600'>{p['name']}</div>"
                f"<div style='color:#64748b;font-size:12px'>{p['pos']} • {p['team']}</div></div></div>"
                for p in tm.get("gets", []) or []
            ])
            # Picks received
            gets_picks = "".join([
                render_pick_row(pick, "add")
                for pick in (picks_by_receiver.get(roster_id, []) if roster_id is not None else [])
            ])
            gets = gets_players + gets_picks
            if not gets:
                gets = "<div style='color:#64748b;font-size:13px'>No players</div>"

            # Players sent
            sends_players = "".join([
                f"<div class='player'><span class='io drop'>−</span>"
                f"<div><div style='font-weight:600'>{p['name']}</div>"
                f"<div style='color:#64748b;font-size:12px'>{p['pos']} • {p['team']}</div></div></div>"
                for p in tm.get("sends", []) or []
            ])
            # Picks sent
            sends_picks = "".join([
                render_pick_row(pick, "drop")
                for pick in (picks_by_sender.get(roster_id, []) if roster_id is not None else [])
            ])
            sends = sends_players + sends_picks

            avatar = tm.get("avatar") or ""
            img = f"<img class='avatar' src='{avatar}' onerror=\"this.style.display='none'\">" if avatar else ""
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
        img = f"<img class='avatar' src='{avatar}' onerror=\"this.style.display='none'\">" if avatar else ""
        adds = "".join([
            f"<div class='player'><span class='io add'>+</span>"
            f"<div><div style='font-weight:600'>{p['name']}</div>"
            f"<div style='color:#64748b;font-size:12px'>{p['pos']} • {p['team']}</div></div></div>"
            for p in d.get("adds", [])
        ])
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

    if activity_df is not None and not activity_df.empty:
        cards = []
        filter_controls_html = """
        <div class="activity-filter">
          <span class="activity-pill filter-pill active" data-kind="waiver">Waivers</span>
          <span class="activity-pill filter-pill active" data-kind="trade">Trades</span>
        </div>
        """
        for _, row in activity_df.iterrows():
            cards.append(html_trade(row) if row["kind"] == "trade" else html_waiver(row))

        activity_html = (
            f"<div class='card activity-card' data-section='activity'>"
            f"  <div style='display:grid;grid-template-columns:.63fr 1fr;align-items:center;'>"
            f"    <h2>This Week: Trades & Waiver Claims</h2>"
            f"    {filter_controls_html}"
            f"  </div>"
            f"  <div class='scroll-box'>"
            f"    <ul class='activity-list'>"
            f"      {activity_css}"  # ensure li items have data-kind + activity-item if you want them filtered
            f"    </ul>"
            f"    <div class='feed'>{''.join(cards)}</div>"  # moved OUTSIDE the <ul>
            f"  </div>"
            f"</div>"
        )

    else:
        activity_html = (
            "<div class='card activity-card' data-section='activity'><h2>This Week: Trades & Waiver Adds</h2>"
            "<div class='feed'>No completed trades or waiver adds this week.</div></div>"
        )

    js_radar = f"""
    <script>
    const ZMAP = {json.dumps(z_map)};
    const METRICS = {json.dumps(theta)};
    const closeRing = arr => arr.concat(arr[0]);
    function makeRadarData(teamA, teamB) {{
      const a = (ZMAP[teamA] || METRICS.map(()=>0));
      const b = (ZMAP[teamB] || METRICS.map(()=>0));
      return [
        {{type:'scatterpolar', r: closeRing(METRICS.map(()=>0)), theta: closeRing(METRICS), name:'League Avg', line: {{ dash:'dash' }}, opacity: 0.8}},
        {{type:'scatterpolar', r: closeRing(a), theta: closeRing(METRICS), name: teamA, fill:'toself', opacity: 0.45}},
        {{type:'scatterpolar', r: closeRing(b), theta: closeRing(METRICS), name: teamB, fill:'toself', opacity: 0.45}}
      ];
    }}
    function renderRadar(teamA, teamB) {{
      const el = document.getElementById('radar-cmp'); if (!el) return;
      const layout = {{ title: 'Radar Comparison (select two teams)', polar: {{ radialaxis: {{ visible: false }} }}, showlegend: true }};
      const data = makeRadarData(teamA, teamB);
      if (!el._plotted) {{ Plotly.newPlot(el, data, layout); el._plotted = true; }}
      else {{ Plotly.react(el, data, layout); }}
    }}
    document.addEventListener('DOMContentLoaded', () => {{
      const selA = document.getElementById('radarA'); const selB = document.getElementById('radarB'); if (!selA || !selB) return;
      renderRadar(selA.value, selB.value);
      selA.addEventListener('change', () => renderRadar(selA.value, selB.value));
      selB.addEventListener('change', () => renderRadar(selA.value, selB.value));
    }});
    </script>
    """

    NAV_HTML = f"""
    <div class="topnav">
      <div class="brand">{get_league(league_id).get('name')} League Dashboard</div>
      <div class="nav-links">
        <button class="nav-btn active" data-target="overview">Overview</button>
        <button class="nav-btn" data-target="recap">Recap</button>
        <button class="nav-btn" data-target="activity">Activity</button>
        <button class="nav-btn" data-target="matchups">Matchups</button>
        <button class="nav-btn" data-target="graphs">Graphs</button>
        <button class="nav-btn" data-target="awards">Awards</button>
      </div>
    </div>
    <div class="nav-spacer"></div>
    """

    tabs_js = """
    <script>
    document.addEventListener('DOMContentLoaded', () => {
      document.querySelectorAll('.card-tabs').forEach(card => {
        const tabs = card.querySelectorAll('.tab-btn');
        const panels = card.querySelectorAll('.tab-panel');
    
        tabs.forEach(tab => {
          tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
    
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
    
            panels.forEach(p => {
              p.classList.toggle('active', p.dataset.tab === target);
            });
          });
        });
      });
    });
    
    document.addEventListener("DOMContentLoaded", function () {
      const tabs = document.querySelectorAll(".team-tab");
      const panels = document.querySelectorAll(".team-panel");
    
      if (!tabs.length) return;
    
      tabs.forEach(tab => {
        tab.addEventListener("click", () => {
          const id = tab.getAttribute("data-team-id");
    
          tabs.forEach(t => t.classList.remove("active"));
          tab.classList.add("active");
    
          panels.forEach(p => {
            if (p.getAttribute("data-team-id") === id) {
              p.classList.add("active");
            } else {
              p.classList.remove("active");
            }
          });
        });
      });
    });
    </script>

    """

    teams_rankings_html = f"""
    <div class="card" data-section="overview">
        {div_stats}
    </div>
    """

    # --- Injury relevance section ---
    accordion_html = ""
    if injury_df is not None and not injury_df.empty:
        accordion_html = render_injury_accordion(injury_df)

    last_week = int(df_weekly["week"].max())
    high = compute_weekly_highlights(df_weekly, last_week)
    ticker_html = render_weekly_highlight_ticker(high, last_week)

    html = f"""<!doctype html>
        <html lang="en"><head>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <div class="header">
            <img src="/static/BR_Logo.png" alt="League Logo" class="site-logo" />
        </div>
        {logoCss}
        {NAV_CSS}
        {plotly_js}
        </head>
        <body>
        {NAV_HTML}
          <div class="grid">
            {teams_rankings_html}
            {podium_card}
            {activity_html}
            {accordion_html}
            {ticker_html}
            {recap_html}
            {matchup_html}
            {awards_html}
            <div class="card" data-section="graphs"><h2>PF vs PA</h2>{div_pfpa}</div>
            <div class="card" data-section="graphs"><h2>Weekly Scores</h2>{div_line}</div>
            <div class="card" data-section="graphs"><h2>Score Distribution</h2>{div_box}</div>
            {radar_cmp_card}
          </div>
        {js_sort_filter}
        {js_radar}
        {activity_filter_js}
        {injury_script}
        {NAV_JS}
        {tabs_js}
        </body>
    </html>
"""

    out_file = os.path.join(out_dir, "index.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    return out_file


def generate_dashboard(league_id: str, rosters: list[dict], users: list[dict], traded: list[dict],
                       players: list[dict], players_map: dict, current_season: int, df_weekly: pd.DataFrame,
                       team_stats: pd.DataFrame, roster_map: dict, league: dict, players_index: dict, teams_index: dict
                       ) -> str:
    weeks = list(range(1, 18))

    matchups_by_week = build_matchups_by_week(
        league_id, weeks, roster_map, players_map
    )
    # picks_by_roster = build_picks_by_roster(league=league, rosters=rosters,
    #                                         traded=traded)  # { roster_id_str: [pick_dict, ...] }
    # teams_ctx = build_teams_overview(
    #     rosters=rosters,
    #     users_list=users,
    #     picks_by_roster=picks_by_roster,
    #     players=players,
    #     players_index=players_index,
    #     teams_index=teams_index,
    # )
    # teams_sidebar_html = render_teams_sidebar(teams_ctx)

    activity_df = build_week_activity(league_id, df_weekly.get("week").max(),
                                      players_map)
    injury_df = build_injury_report(league_id, local_tz="America/New_York", include_free_agents=False,
                                    players=players_map, roster_map=roster_map)

    # after df_weekly/team_stats are computed:
    projections, players, teams = load_week_schedule(current_season, current_week)
    recap_html = render_week_recap_tab(
        league_id, df_weekly[df_weekly["finalized"] == True].copy(), roster_map, players_map
    )

    # Pre-render HTML slides per week using your existing _render_matchup_slide
    slides_by_week: dict[int, str] = {
        w: "".join(
            render_matchup_slide(m, w, df_weekly[df_weekly["finalized"] == True].copy().get("week").max(),
                                 current_season, projections, players, teams) for
            m in
            matchups_by_week.get(w, []))
        for w in range(1, weeks)
    }
    matchup_html = render_matchup_carousel_weeks(slides_by_week)

    awards = compute_awards_season(df_weekly[df_weekly["finalized"] == True].copy(), players_map, league_id)
    awards_html = render_awards_section(awards)

    # you already have draft pick logic; plug that in here

    site_path = build_interactive_site(
        df_weekly[df_weekly["finalized"] == True].copy(),
        team_stats,
        out_dir="site",
        activity_df=activity_df,
        injury_df=injury_df,
        matchup_html=matchup_html,
        recap_html=recap_html,
        awards_html=awards_html,
        league_id=league_id,
        roster_map=roster_map,
        teams_html=teams_sidebar_html
    )

    return site_path


# --------------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True, help="Sleeper league ID")
    args = parser.parse_args()
    rosters = get_rosters(args.league)
    users = get_users(args.league)
    current = get_nfl_state()
    players = get_nfl_players()
    players_map = get_players_map(players)
    players_index = load_model_value_table()
    teams_index = load_teams_index()

    current_season = current.get("season")
    current_week = current.get("week")

    df_weekly, team_stats, roster_map = build_tables(
        args.league, current_week, players, users, rosters
    )
    activity_df = build_week_activity(args.league, df_weekly.get("week").max(),
                                      players_map)
    injury_df = build_injury_report(args.league, local_tz="America/New_York", include_free_agents=False,
                                    players=players_map, roster_map=roster_map)
    projections, players, teams = load_week_schedule(current_season, current_week)
    recap_html = render_week_recap_tab(
        args.league, df_weekly[df_weekly["finalized"] == True].copy(), roster_map, players_map
    )
    weeks = list(range(1, 18))
    matchups_by_week = build_matchups_by_week(
        args.league, weeks, roster_map, players_map
    )
    finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())
    else:
        # fall back to current week if nothing is finalized yet
        last_final_week = current_week

    status_by_pid = build_status_for_week(
        current_season,
        current_week,
        players_index,
        teams_index,
    )

    # Pre-render HTML slides per week using your existing _render_matchup_slide
    slides_by_week: dict[int, str] = {
        w: "".join(
            render_matchup_slide(
                m,
                current_week,
                last_final_week,
                status_by_pid=status_by_pid,
                projections=projections,
                players=players,
                teams=teams, ) for
            m in matchups_by_week.get(w, []))
        for w in weeks
    }
    matchup_html = render_matchup_carousel_weeks(slides_by_week)

    awards = compute_awards_season(df_weekly[df_weekly["finalized"] == True].copy(), players_map, args.league)
    awards_html = render_awards_section(awards)

    site_path = build_interactive_site(
        df_weekly[df_weekly["finalized"] == True].copy(),
        team_stats,
        out_dir="site",
        activity_df=activity_df,
        injury_df=injury_df,
        matchup_html=matchup_html,
        recap_html=recap_html,
        awards_html=awards_html,
        league_id=args.league,
        roster_map=roster_map,
    )
    print(f"Interactive site saved to: {site_path}")


if __name__ == "__main__":
    main()
