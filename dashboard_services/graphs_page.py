import json
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot
from typing import Dict

from dashboard_services.utils import z_better_outward


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

    # ---------- Convert figs to divs ----------
    div_pfpa = plotly_plot(figs["pf_pa"], include_plotlyjs=False, output_type="div")
    div_line = plotly_plot(figs["scores_line"], include_plotlyjs=False, output_type="div")
    div_box = plotly_plot(figs["scores_box"], include_plotlyjs=False, output_type="div")

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
              <li><span class="mini-label">PF</span> — Points For</li>
              <li><span class="mini-label">PA</span> — Points Against</li>
              <li><span class="mini-label">MAX</span> — Best weekly score</li>
              <li><span class="mini-label">MIN</span> — Worst weekly score</li>
              <li><span class="mini-label">AVG</span> — Average weekly score</li>
              <li><span class="mini-label">STD</span> — Volatility of scores</li>
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
      if (!el || !window.Plotly) return;

      const layout = {{
        title: 'Radar Comparison (select two teams)',
        polar: {{ radialaxis: {{ visible: false }} }},
        showlegend: false,
        margin: {{ l: 40, r: 20, t: 40, b: 30 }}
      }};

      const data = makeRadarData(teamA, teamB);

      if (!el._plotted) {{
        Plotly.newPlot(el, data, layout);
        el._plotted = true;
      }} else {{
        Plotly.react(el, data, layout);
      }}
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

    # ---------- Main layout ----------
    main_html = f"""
      <div class="page-layout">
        <main class="page-main">
          <div class="graphs-page">

            <div class="card">
              <div class="card-header-row">
                <h2>PF vs PA Scatter</h2>
              </div>
              <div class="card-body graph-body">
                {div_pfpa}
              </div>
            </div>

            <div class="card">
              <div class="card-header-row">
                <h2>Weekly Scores by Team</h2>
              </div>
              <div class="card-body graph-body">
                {div_line}
              </div>
            </div>

            <div class="card">
              <div class="card-header-row">
                <h2>Score Distribution</h2>
              </div>
              <div class="card-body graph-body">
                {div_box}
              </div>
            </div>

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
      {js_radar}
    """

    return main_html
