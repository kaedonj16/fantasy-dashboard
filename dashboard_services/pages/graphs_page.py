import json
from typing import Dict

import numpy as np
import plotly.graph_objs as go

from utils.utils import z_better_outward
from utils.all_play import all_play_analysis
from utils.standings_viz import luck_quadrant_svg, value_age_svg


def _luck_and_value_age_cards(ctx: dict, df_weekly_finalized) -> str:
    """Two league-wide SVG scatters for the graphs page: a performance-vs-luck
    quadrant (all-play win rate vs actual) and a dynasty-value-vs-age quadrant.
    Both are server-rendered SVG (no Plotly), and return '' individually when
    there isn't enough data, so the cards simply don't appear."""
    viewer_owner = str((ctx.get("viewer") or {}).get("viewer_team_name") or "")

    luck_svg = ""
    try:
        if df_weekly_finalized is not None and not df_weekly_finalized.empty:
            weekly_scores: dict = {}
            actual_wins: dict = {}
            for _, r in df_weekly_finalized.iterrows():
                wk = int(r["week"])
                owner = str(r["owner"])
                weekly_scores.setdefault(wk, {})[owner] = float(r["points"] or 0)
                actual_wins[owner] = actual_wins.get(owner, 0.0) + float(r.get("win") or 0)
            luck_svg = luck_quadrant_svg(all_play_analysis(weekly_scores, actual_wins), viewer_owner)
    except Exception:
        luck_svg = ""

    value_age_svg_str = ""
    try:
        from dashboard_services.ai.context_builders import team_value_age_rows
        value_age_svg_str = value_age_svg(team_value_age_rows(ctx), viewer_owner)
    except Exception:
        value_age_svg_str = ""

    cards = ""
    if luck_svg:
        cards += f"""
            <div class="card">
              <div class="card-header-row">
                <h2>Performance vs Luck</h2>
              </div>
              <div class="card-body graph-body svg-graph-body">
                <div class="svg-graph-note">All-play win rate (how you would do against the whole league each week) vs your actual win rate. Above the dashed line means you have won more than your scoring earned.</div>
                {luck_svg}
              </div>
            </div>"""
    if value_age_svg_str:
        cards += f"""
            <div class="card">
              <div class="card-header-row">
                <h2>Dynasty Value vs Age</h2>
              </div>
              <div class="card-body graph-body svg-graph-body">
                <div class="svg-graph-note">Each team by total roster value and average age. Top-left is young and loaded; top-right is a closing win-now window.</div>
                {value_age_svg_str}
              </div>
            </div>"""
    return cards


def build_graphs_body(ctx: dict) -> str:
    team_stats = ctx["team_stats"]
    df_weekly = ctx["df_weekly"]
    df_weekly = df_weekly[df_weekly["finalized"] == True].copy()

    # ---------- Core aggregates ----------
    pr_sorted = (
        team_stats.sort_values(["PowerScore", "PF"], ascending=[False, False])
        .reset_index(drop=True)
    )
    top3 = pr_sorted.head(3)
    wk_avg = df_weekly.groupby("week")["points"].mean().reset_index()

    metrics = ["PF", "PA", "MAX", "MIN", "AVG", "STD"]
    Z = z_better_outward(team_stats, metrics)
    theta = metrics
    z_map: Dict[str, list] = {
        team_stats.loc[i, "owner"]: Z.iloc[i].values.astype(float).tolist()
        for i in range(len(team_stats))
    }

    owners = team_stats["owner"].tolist()

    # ---------- ONE shared color map ----------
    COLOR_CYCLE = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692",
        "#B6E880", "#FF97FF", "#FECB52",
    ]
    owner_colors: Dict[str, str] = {}
    for idx, o in enumerate(owners):
        owner_colors[o] = COLOR_CYCLE[idx % len(COLOR_CYCLE)]

    figs: Dict[str, go.Figure] = {}

    # ---------- PF vs PA scatter ----------
    scatter_traces = []
    for _, r in team_stats.iterrows():
        owner = r["owner"]
        scatter_traces.append(
            go.Scatter(
                x=[r["PA"]],
                y=[r["PF"]],
                mode="markers+text",
                text=[owner],
                textposition="top center",
                marker=dict(
                    size=11,
                    line=dict(color="black", width=1),
                    color=owner_colors.get(owner),
                ),
                name=owner,
                showlegend=False,
            )
        )
    x = team_stats["PA"].values
    y = team_stats["PF"].values
    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        m = ((x - x.mean()) * (y - y.mean())).sum() / max(
            ((x - x.mean()) ** 2).sum(), 1e-9
        )
        b = y.mean() - m * x.mean()
        xs = [float(min(x) * 0.95), float(max(x) * 1.05)]
        ys = [m * xs[0] + b, m * xs[1] + b]
        scatter_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(dash="dash", color="#9ca3af"),
                name="Trend",
                showlegend=False,
            )
        )

    figs["pf_pa"] = go.Figure(scatter_traces)
    figs["pf_pa"].update_layout(
        xaxis_title=dict(text="Points Against (PA)", standoff=12),
        yaxis_title=dict(text="Points For (PF)"),
        hovermode="closest",
        margin=dict(l=40, r=20, t=10, b=45),
        showlegend=False,
    )

    # ---------- Weekly scores line chart ----------
    line_traces = [
        go.Scatter(
            x=wk_avg["week"],
            y=wk_avg["points"],
            mode="lines",
            name="League Avg",
            line=dict(dash="dash", width=3, color="#9ca3af"),
            opacity=0.7,
            showlegend=False,
        )
    ]
    for owner, g in df_weekly.sort_values("week").groupby("owner"):
        line_traces.append(
            go.Scatter(
                x=g["week"],
                y=g["points"],
                mode="lines+markers",
                name=owner,
                line=dict(color=owner_colors.get(owner)),
                marker=dict(size=6),
                showlegend=False,
            )
        )

    figs["scores_line"] = go.Figure(line_traces)
    figs["scores_line"].update_layout(
        xaxis_title=dict(text="Week", standoff=12),
        yaxis_title=dict(text="Points"),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=10, b=90),
        showlegend=False,
    )

    # ---------- Boxplot of scores by team ----------
    order = (
        df_weekly.groupby("owner")["points"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    box_traces = []
    for o in order:
        pts = df_weekly.loc[df_weekly["owner"] == o, "points"]
        box_traces.append(
            go.Box(
                y=pts,
                name=o,
                boxmean=True,
                orientation="v",
                hoveron="boxes",
                boxpoints=False,
                marker=dict(color=owner_colors.get(o)),
                showlegend=False,
            )
        )

    figs["scores_box"] = go.Figure(box_traces)
    figs["scores_box"].update_layout(
        xaxis_title=dict(text="Team", standoff=12),
        yaxis_title=dict(text="Points"),
        hovermode="closest",
        margin=dict(l=40, r=20, t=10, b=120),
        showlegend=False,
    )

    # ---------- Radar selectors ----------
    if not owners:
        owners = ["Team A", "Team B"]

    opts_a = []
    opts_b = []
    for i, o in enumerate(owners):
        sel_a = " selected" if i == 0 else ""
        sel_b = " selected" if i == 1 else ""
        opts_a.append(f"<option value='{o}'{sel_a}>{o}</option>")
        opts_b.append(f"<option value='{o}'{sel_b}>{o}</option>")

    opts_a_html = "".join(opts_a)
    opts_b_html = "".join(opts_b)

    # ---------- Convert figs to JS-safe JSON for deferred rendering ----------
    def _fig_json(fig):
        return fig.to_json().replace("</", "<\\/")

    pfpa_json = _fig_json(figs["pf_pa"])
    line_json  = _fig_json(figs["scores_line"])
    box_json   = _fig_json(figs["scores_box"])

    # League-wide SVG scatters (luck quadrant + dynasty value vs age).
    svg_cards_html = _luck_and_value_age_cards(ctx, df_weekly)

    # ---------- Sidebar: top teams + metrics + unified legend ----------
    top_rows = []
    for _, r in top3.iterrows():
        top_rows.append(
            f"<div class='mini-row'>"
            f"  <div class='mini-label'>{r['owner']}</div>"
            f"  <div class='mini-value'>"
            f"    <span class='mini-stat'>Power {r['PowerScore']:.1f}</span>"
            f"    <span class='mini-stat'>PF {r['PF']:.1f}</span>"
            f"  </div>"
            f"</div>"
        )
    top3_html = "".join(top_rows)

    legend_items = []
    for o in owners:
        color = owner_colors.get(o, "#9ca3af")
        legend_items.append(
            f"""
            <div class="legend-row">
              <span class="legend-dot" style="background:{color};"></span>
              <span class="legend-label">{o}</span>
            </div>
            """
        )
    legend_html = "".join(legend_items)

    sidebar_html = f"""
        <div class="card small">
          <div class="card-header">
            <h3>Legend</h3>
          </div>
          <div class="card-body mini-body">
            {legend_html}
          </div>
        </div>

        <div class="card small">
          <div class="card-header">
            <h3>Metrics Key</h3>
          </div>
          <div class="card-body">
            <ul class="ticker-list">
              <li><span class="mini-label">PF</span> - Points For</li>
              <li><span class="mini-label">PA</span> - Points Against</li>
              <li><span class="mini-label">MAX</span> - Best weekly score</li>
              <li><span class="mini-label">MIN</span> - Worst weekly score</li>
              <li><span class="mini-label">AVG</span> - Average weekly score</li>
              <li><span class="mini-label">STD</span> - Volatility of scores</li>
            </ul>
          </div>
        </div>
    """

    # ---------- Radar JS (uses same owner_colors) ----------
    js_radar = f"""
    <script>
    const ZMAP = {json.dumps(z_map)};
    const METRICS = {json.dumps(theta)};
    const COLORS = {json.dumps(owner_colors)};
    const closeRing = arr => arr.concat(arr[0]);

    function makeRadarData(teamA, teamB) {{
      const a = (ZMAP[teamA] || METRICS.map(() => 0));
      const b = (ZMAP[teamB] || METRICS.map(() => 0));

      const colorA = COLORS[teamA] || '#1f77b4';
      const colorB = COLORS[teamB] || '#ff7f0e';

      return [
        {{
          type: 'scatterpolar',
          r: closeRing(METRICS.map(() => 0)),
          theta: closeRing(METRICS),
          name: 'League Avg',
          line: {{ dash: 'dash', color: '#9ca3af' }},
          opacity: 0.8
        }},
        {{
          type: 'scatterpolar',
          r: closeRing(a),
          theta: closeRing(METRICS),
          name: teamA,
          fill: 'toself',
          opacity: 0.45,
          line: {{ color: colorA }},
          fillcolor: colorA
        }},
        {{
          type: 'scatterpolar',
          r: closeRing(b),
          theta: closeRing(METRICS),
          name: teamB,
          fill: 'toself',
          opacity: 0.45,
          line: {{ color: colorB }},
          fillcolor: colorB
        }}
      ];
    }}

    function renderRadar(teamA, teamB) {{
      const el = document.getElementById('radar-cmp');
      if (!el) return;

      const layout = {{
        title: 'Radar Comparison (select two teams)',
        polar: {{ radialaxis: {{ visible: false }} }},
        showlegend: false,
        margin: {{ l: 40, r: 20, t: 40, b: 30 }}
      }};

      const data = makeRadarData(teamA, teamB);

      (window.ensurePlotly ? window.ensurePlotly() : Promise.resolve(window.Plotly)).then(function(P) {{
        if (!P) return;
        if (!el._plotted) {{
          P.newPlot(el, data, layout);
          el._plotted = true;
        }} else {{
          P.react(el, data, layout);
        }}
      }});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      const selA = document.getElementById('radarTeamA');
      const selB = document.getElementById('radarTeamB');
      if (!selA || !selB) return;

      renderRadar(selA.value, selB.value);

      selA.addEventListener('change', () => renderRadar(selA.value, selB.value));
      selB.addEventListener('change', () => renderRadar(selA.value, selB.value));
    }});
    </script>
    """

    # ---------- Deferred chart rendering (Plotly loaded on demand) ----------
    js_charts = (
        '<script>(function(){'
        'var _FIGS={'
        '"chart-pfpa":' + pfpa_json + ','
        '"chart-line":'  + line_json + ','
        '"chart-box":'   + box_json  +
        '};'
        'var _CFG={responsive:true,displayModeBar:false};'
        '(window.ensurePlotly?window.ensurePlotly():Promise.resolve(window.Plotly)).then(function(P){'
        'if(!P)return;'
        'Object.keys(_FIGS).forEach(function(id){'
        'var f=_FIGS[id];var el=document.getElementById(id);'
        'if(el)P.newPlot(el,f.data,f.layout,_CFG);'
        '});});})();</script>'
    )

    # ---------- Main layout ----------
    main_html = f"""
      <div class="page-layout" data-page="graphs">
        <main class="page-main">
          <div class="graphs-page">

            <div class="card">
              <div class="card-header-row">
                <h2>PF vs PA Scatter</h2>
              </div>
              <div class="card-body graph-body">
                <div id="chart-pfpa" style="width:100%;min-height:350px;"></div>
              </div>
            </div>

            <div class="card">
              <div class="card-header-row">
                <h2>Weekly Scores by Team</h2>
              </div>
              <div class="card-body graph-body">
                <div id="chart-line" style="width:100%;min-height:350px;"></div>
              </div>
            </div>

            <div class="card">
              <div class="card-header-row">
                <h2>Score Distribution</h2>
              </div>
              <div class="card-body graph-body">
                <div id="chart-box" style="width:100%;min-height:350px;"></div>
              </div>
            </div>

            {svg_cards_html}

            <div class="card">
              <div class="card-header-row">
                <h2>Radar Comparison</h2>
                <div class="radar-selectors">
                  <label>
                    Team A
                    <select id="radarTeamA" class="search">
                      {opts_a_html}
                    </select>
                  </label>
                  <label>
                    Team B
                    <select id="radarTeamB" class="search">
                      {opts_b_html}
                    </select>
                  </label>
                </div>
              </div>
              <div class="card-body graph-body">
                <div id="radar-cmp" style="width:100%;min-height:380px;"></div>
              </div>
            </div>

          </div>
        </main>

        <aside class="page-sidebar">
          {sidebar_html}
        </aside>
      </div>
      {js_charts}
      {js_radar}
    """

    return main_html


def build_career_graphs_body(career_ctx: dict) -> str:
    """
    Build Plotly graphs for the career (all-seasons aggregate) view.
    career_ctx keys: team_stats, df_weekly (combined with 'season' col), season_pf_df, is_career
    """
    import pandas as pd  # local import fine since module already imports it at top
    team_stats = career_ctx.get("team_stats", pd.DataFrame())
    df_all = career_ctx.get("df_weekly", pd.DataFrame())
    season_pf_df = career_ctx.get("season_pf_df", pd.DataFrame())

    if team_stats.empty:
        return "<div class='card central'><div class='card-body'><p>No career data available.</p></div></div>"

    owners = team_stats["owner"].tolist()
    owner_colors: dict = {}
    for idx, o in enumerate(owners):
        owner_colors[o] = COLOR_CYCLE[idx % len(COLOR_CYCLE)]

    figs: dict = {}

    # ── 1. Career PF vs PA scatter ─────────────────────────────────────────
    scatter_traces = []
    for _, r in team_stats.iterrows():
        owner = r["owner"]
        scatter_traces.append(
            go.Scatter(
                x=[r["PA"]],
                y=[r["PF"]],
                mode="markers+text",
                text=[owner],
                textposition="top center",
                marker=dict(size=12, color=owner_colors.get(owner), line=dict(color="black", width=1)),
                name=owner,
                showlegend=False,
            )
        )
    figs["pf_pa"] = go.Figure(scatter_traces)
    figs["pf_pa"].update_layout(
        xaxis_title=dict(text="Career Points Against", standoff=12),
        yaxis_title=dict(text="Career Points For"),
        hovermode="closest",
        margin=dict(l=40, r=20, t=10, b=45),
        showlegend=False,
    )

    # ── 2. Season-by-season points per team (line chart) ──────────────────
    season_fig_traces = []
    if not season_pf_df.empty and {"season", "owner", "pf"}.issubset(season_pf_df.columns):
        for owner, grp in season_pf_df.groupby("owner"):
            grp = grp.sort_values("season")
            season_fig_traces.append(
                go.Scatter(
                    x=grp["season"],
                    y=grp["pf"],
                    mode="lines+markers",
                    name=str(owner),
                    line=dict(color=owner_colors.get(str(owner))),
                    marker=dict(size=7),
                    showlegend=False,
                    hovertemplate="%{fullData.name}<br>%{x}: %{y:.1f} PF<extra></extra>",
                )
            )
    figs["season_pf"] = go.Figure(season_fig_traces)
    figs["season_pf"].update_layout(
        xaxis_title=dict(text="Season", standoff=12),
        yaxis_title=dict(text="Total Points (regular season)"),
        xaxis=dict(dtick=1),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=10, b=60),
        showlegend=False,
    )

    # ── 3. Career score distribution (all weekly scores, box plot) ─────────
    finalized_df = df_all[df_all["finalized"] == True].copy() if "finalized" in df_all.columns else df_all.copy()
    box_order = (
        finalized_df.groupby("owner")["points"].median()
        .sort_values(ascending=False).index.tolist()
    ) if not finalized_df.empty and "points" in finalized_df.columns else owners

    box_traces = []
    for o in box_order:
        pts = finalized_df.loc[finalized_df["owner"] == o, "points"] if not finalized_df.empty else pd.Series()
        box_traces.append(
            go.Box(
                y=pts,
                name=o,
                boxmean=True,
                boxpoints=False,
                marker=dict(color=owner_colors.get(o)),
                showlegend=False,
            )
        )
    figs["box"] = go.Figure(box_traces)
    figs["box"].update_layout(
        xaxis_title=dict(text="Team", standoff=12),
        yaxis_title=dict(text="Weekly Points"),
        hovermode="closest",
        margin=dict(l=40, r=20, t=10, b=120),
        showlegend=False,
    )

    def _fig_json(fig):
        return fig.to_json().replace("</", "<\\/")

    scatter_json = _fig_json(figs["pf_pa"])
    season_json  = _fig_json(figs["season_pf"])
    box_c_json   = _fig_json(figs["box"])

    # ── Sidebar: career standings table ───────────────────────────────────
    ts_sorted = team_stats.sort_values("PF", ascending=False).reset_index(drop=True)
    sidebar_rows = ""
    for _, r in ts_sorted.iterrows():
        owner = r["owner"]
        color = owner_colors.get(owner, "#9ca3af")
        games = int(r.get("Wins", 0)) + int(r.get("Losses", 0)) + int(r.get("Ties", 0))
        sidebar_rows += (
            f"<div class='mini-row'>"
            f"  <div class='mini-label' style='display:flex;align-items:center;gap:6px;'>"
            f"    <span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};'></span>"
            f"    {owner}"
            f"  </div>"
            f"  <div class='mini-value'>"
            f"    <span class='mini-stat'>{int(r.get('Wins',0))}-{int(r.get('Losses',0))}</span>"
            f"    <span class='mini-stat'>{r['PF']:.0f} PF</span>"
            f"  </div>"
            f"</div>"
        )

    sidebar_html = f"""
        <div class="card small">
          <div class="card-header"><h3>Career Standings</h3></div>
          <div class="card-body mini-body">{sidebar_rows}</div>
        </div>
        <div class="card small">
          <div class="card-header"><h3>Metrics Key</h3></div>
          <div class="card-body">
            <ul class="ticker-list">
              <li><span class="mini-label">PF</span> - Career points for</li>
              <li><span class="mini-label">PA</span> - Career points against</li>
              <li><span class="mini-label">W-L</span> - Career record</li>
            </ul>
          </div>
        </div>"""

    js_career = (
        '<script>(function(){'
        'var _FIGS={'
        '"chart-scatter":' + scatter_json + ','
        '"chart-season":'  + season_json  + ','
        '"chart-box-c":'   + box_c_json   +
        '};'
        'var _CFG={responsive:true,displayModeBar:false};'
        '(window.ensurePlotly?window.ensurePlotly():Promise.resolve(window.Plotly)).then(function(P){'
        'if(!P)return;'
        'Object.keys(_FIGS).forEach(function(id){'
        'var f=_FIGS[id];var el=document.getElementById(id);'
        'if(el)P.newPlot(el,f.data,f.layout,_CFG);'
        '});});})();</script>'
    )
    return f"""
      <div class="page-layout" data-page="graphs">
        <main class="page-main">
          <div class="graphs-page">
            <div class="card">
              <div class="card-header-row"><h2>Career PF vs PA</h2></div>
              <div class="card-body graph-body"><div id="chart-scatter" style="width:100%;min-height:350px;"></div></div>
            </div>
            <div class="card">
              <div class="card-header-row"><h2>Points Per Season by Team</h2></div>
              <div class="card-body graph-body"><div id="chart-season" style="width:100%;min-height:350px;"></div></div>
            </div>
            <div class="card">
              <div class="card-header-row"><h2>Career Score Distribution</h2></div>
              <div class="card-body graph-body"><div id="chart-box-c" style="width:100%;min-height:350px;"></div></div>
            </div>
          </div>
        </main>
        <aside class="page-sidebar">{sidebar_html}</aside>
      </div>{js_career}"""


COLOR_CYCLE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692",
    "#B6E880", "#FF97FF", "#FECB52",
]
