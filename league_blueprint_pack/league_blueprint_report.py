#!/usr/bin/env python3
"""
League Blueprint Report Generator for Sleeper Fantasy Football

This script generates comprehensive analytics and visualizations for Sleeper fantasy football leagues.
It pulls data from the Sleeper API and creates both PDF reports and interactive dashboards.

Usage:
    python league_blueprint_report.py --league <league_id> --weeks <num_weeks>

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
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer
from requests.adapters import HTTPAdapter
from textwrap import shorten
from textwrap import shorten as _shorten
from typing import Dict, Any, Iterable, Tuple, Optional, List, Union, Callable
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

from league_blueprint.api import get_league_name, get_nfl_state
from league_blueprint.awards import compute_weekly_highlights, compute_awards_season, render_awards_section
from league_blueprint.injuries import render_injury_accordion, build_injury_report
from league_blueprint.matchups import _render_matchup_slide, render_matchup_carousel_weeks
from league_blueprint.players import get_players_map
from league_blueprint.service import build_tables, render_weekly_highlight_ticker, render_week_recap_tab, \
    build_matchups_by_week, build_week_activity
from league_blueprint.styles import activity_css, logoCss, injury_script, js_sort_filter, \
    activity_filter_js, NAV_JS, NAV_CSS
from league_blueprint.utils import z_better_outward, _streak_class

try:
    from rapidfuzz import fuzz, process

    HAVE_RF = True
except Exception:
    HAVE_RF = False


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
    team_stats["WinPct"] = win_pct
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
    x = team_stats["PA"].values;
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

    cols = ["Team", "Win %", "PF", "PA", "Average", "Std Dev", "Best Week", "Worst Week"]
    stats_tbl = stats_tbl[cols].copy()
    for c in ["Win %", "PF", "PA", "Average", "Std Dev", "Best Week", "Worst Week"]:
        stats_tbl[c] = stats_tbl[c].astype(float).round(3 if c == "Win %" else 2)

    body_rows = []
    for _, r in stats_tbl[cols].iterrows():
        body_rows.append("<tr>" + "".join([
            f"<td>{r['Team']}</td>",
            f"<td class='num'>{r['Win %']:.3f}</td>",
            f"<td class='num'>{float(r['PF']):.2f}</td>",
            f"<td class='num'>{float(r['PA']):.2f}</td>",
            f"<td class='num'>{float(r['Average']):.2f}</td>",
            f"<td class='num'>{float(r['Std Dev']):.2f}</td>",
            f"<td class='num'>{float(r['Best Week']):.2f}</td>",
            f"<td class='num'>{float(r['Worst Week']):.2f}</td>",
        ]) + "</tr>")

    table_html = f"""
      <div class="table-wrap">
        <table id="stats" class="table-stats">
          <thead><tr>{"".join([f"<th data-col='{i}'>{c}</th>" for i, c in enumerate(cols)])}</tr></thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
      </div>
    """

    div_stats = f"""
      <div class="card stats" data-section="overview">
        <h2>Team Stats</h2>
        {table_html}
        <div class="footer">Default sort: Win% ↓ then PF ↓. Click a header to sort; Shift+Click to add secondary sorts.</div>
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
        if streak_chip:
            chips_html += f"<span class='chip chip-streak'>{streak_chip}</span>"
        chips_html += f"<span class='chip {diff_class}'>{diff_val:+.1f}</span>"
        avatar_url = row.get("avatar")
        img = f"<img class='avatar sm' src='{avatar_url}' onerror=\"this.style.display='none'\">" if avatar_url else ""
        css_cls = _streak_class(row)
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
        streak_frame_cls = _streak_class(row)  # "streak-hot"/"streak-cold"/""

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

    podium_card = f"<div class='card power' data-section='overview'><h2>Power Rankings</h2>{podium_html}{rankings_html}</div>"

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

    def pill(s):
        return f"<span class='badge'>{s}</span>"

    def html_trade(txrow):
        teams = txrow["data"]["teams"]
        cols = []
        for tm in teams:
            gets = "".join([
                f"<div class='player'><span class='io add'>+</span>"
                f"<div><div style='font-weight:600'>{p['name']}</div>"
                f"<div style='color:#64748b;font-size:12px'>{p['pos']} • {p['team']}</div></div></div>"
                for p in tm.get("gets", [])
            ]) or "<div style='color:#64748b;font-size:13px'>No players</div>"

            sends = "".join([
                f"<div class='player'><span class='io drop'>−</span>"
                f"<div><div style='font-weight:600'>{p['name']}</div>"
                f"<div style='color:#64748b;font-size:12px'>{p['pos']} • {p['team']}</div></div></div>"
                for p in tm.get("sends", [])
            ])

            avatar = tm.get("avatar") or ""
            img = f"<img class='avatar' src='{avatar}' onerror=\"this.style.display='none'\">" if avatar else ""
            cols.append(
                f"<div class='team-col'>"
                f"  <header>{img}<div class='team-name'>{tm['name']}</div></header>"
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
      <div class="brand">{get_league_name(league_id).get('name')} League Dashboard</div>
      <div class="nav-links">
        <button class="nav-btn active" data-target="overview">Overview</button>
        <button class="nav-btn" data-target="recap">Recap</button>
        <button class="nav-btn" data-target="activity">Activity</button>
        <button class="nav-btn" data-target="graphs">Graphs</button>
        <button class="nav-btn" data-target="matchups">Matchups</button>
        <button class="nav-btn" data-target="awards">Awards</button>
      </div>
    </div>
    <div class="nav-spacer"></div>
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
            <img src="BR_Logo.png" alt="League Logo" class="site-logo" />
        </div>
        {logoCss}
        {NAV_CSS}
        {plotly_js}
        </head>
        <body>
        {NAV_HTML}
          <div class="grid">
            {div_stats}
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
        </body></html>"""

    out_file = os.path.join(out_dir, "index.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    return out_file


# --------------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True, help="Sleeper league ID")
    parser.add_argument("--weeks", type=int, default=get_nfl_state().get("week"), help="Number of weeks to pull (1..N)")
    args = parser.parse_args()
    # rapidApiKey = "a31667ff00msh6d542faa96aa36bp1513aajsn612c819feca4"
    season = get_nfl_state().get("season")

    weeks = list(range(1, 15))
    players_map = get_players_map()
    df_weekly, team_stats, roster_map = build_tables(args.league, args.weeks)
    matchups_by_week = build_matchups_by_week(
        args.league, weeks, roster_map, players_map
    )
    roster_map = {str(rid): owner for rid, owner in zip(df_weekly["roster_id"].unique(), df_weekly["owner"].unique())}

    activity_df = build_week_activity(args.league, df_weekly[df_weekly["finalized"] == True].copy().get("week").max(),

                                      players_map)
    injury_df = build_injury_report(args.league, local_tz="America/New_York", include_free_agents=False)

    # after df_weekly/team_stats are computed:
    recap_html = render_week_recap_tab(
        args.league, df_weekly[df_weekly["finalized"] == True].copy(), roster_map, players_map
    )

    # Pre-render HTML slides per week using your existing _render_matchup_slide
    slides_by_week: dict[int, str] = {
        w: "".join(
            _render_matchup_slide(m, w, df_weekly[df_weekly["finalized"] == True].copy().get("week").max(), season) for
            m in
            matchups_by_week.get(w, []))
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
        league_id=args.league
    )
    print(f"Interactive site saved to: {site_path}")


if __name__ == "__main__":
    main()
