import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot, get_plotlyjs
from plotly.offline import plot as plotly_plot
from typing import Dict

from dashboard_services.utils import z_better_outward


def build_graphs_body(ctx: dict) -> str:
    team_stats = ctx["team_stats"]
    df_weekly = ctx["df_weekly"]

    # ---------- your existing analytics ----------

    pr_sorted = team_stats.sort_values(["PowerScore", "PF"], ascending=[False, False]).reset_index(drop=True)
    top3 = pr_sorted.head(3)

    wk_avg = df_weekly.groupby("week")["points"].mean().reset_index()

    metrics = ["PF", "PA", "MAX", "MIN", "AVG", "STD"]
    Z = z_better_outward(team_stats, metrics)
    theta = metrics
    z_map = {team_stats.loc[i, "owner"]: Z.iloc[i].values.astype(float).tolist()
             for i in range(len(team_stats))}

    figs = {}

    # PF vs PA scatter
    scatter_traces = []
    for _, r in team_stats.iterrows():
        scatter_traces.append(go.Scatter(
            x=[r["PA"]], y=[r["PF"]],
            mode="markers+text",
            text=[r["owner"]],
            textposition="top center",
            marker=dict(size=12, line=dict(color="black", width=1)),
            name=r["owner"],
        ))

    x = team_stats["PA"].values
    y = team_stats["PF"].values
    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        m = ((x - x.mean()) * (y - y.mean())).sum() / max(((x - x.mean()) ** 2).sum(), 1e-9)
        b = y.mean() - m * x.mean()
        xs = [float(min(x) * 0.95), float(max(x) * 1.05)]
        ys = [m * xs[0] + b, m * xs[1] + b]
        scatter_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(dash="dash"),
            name="Trend"
        ))

    figs["pf_pa"] = go.Figure(scatter_traces)
    figs["pf_pa"].update_layout(
        title="PF vs PA",
        xaxis_title="Points Against (PA)",
        yaxis_title="Points For (PF)",
        hovermode="closest",
    )

    # Weekly scores line chart
    line_traces = [
        go.Scatter(
            x=wk_avg["week"],
            y=wk_avg["points"],
            mode="lines",
            name="League Avg",
            line=dict(dash="dash", width=3),
            opacity=0.7,
        )
    ]
    for owner, g in df_weekly.sort_values("week").groupby("owner"):
        line_traces.append(go.Scatter(
            x=g["week"],
            y=g["points"],
            mode="lines+markers",
            name=owner,
        ))
    figs["scores_line"] = go.Figure(line_traces)
    figs["scores_line"].update_layout(
        title="Weekly Scores by Team",
        xaxis_title="Week",
        yaxis_title="Points",
        hovermode="x unified",
    )

    # Boxplot of scores by team
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
        box_traces.append(go.Box(
            y=pts,
            name=o,
            boxmean=True,
            orientation="v",
            hoveron="boxes",
            boxpoints=False,
        ))
    figs["scores_box"] = go.Figure(box_traces)
    figs["scores_box"].update_layout(
        title="Score Distribution by Team",
        xaxis_title="Team",
        yaxis_title="Points",
        hovermode="closest",
    )

    # Radar: top 2 power ranking teams
    t1 = pr_sorted.iloc[0]["owner"]
    t2 = pr_sorted.iloc[1]["owner"] if len(pr_sorted) > 1 else pr_sorted.iloc[0]["owner"]

    def radar_compare_fig(a, b):
        return go.Figure([
            go.Scatterpolar(
                r=[0] * len(theta) + [0],
                theta=theta + theta[:1],
                name="League Avg",
                line=dict(dash="dash"),
                opacity=0.8,
            ),
            go.Scatterpolar(
                r=z_map[a] + [z_map[a][0]],
                theta=theta + theta[:1],
                fill="toself",
                name=a,
                opacity=0.45,
            ),
            go.Scatterpolar(
                r=z_map[b] + [z_map[b][0]],
                theta=theta + theta[:1],
                fill="toself",
                name=b,
                opacity=0.45,
            ),
        ])

    figs["radar_cmp"] = radar_compare_fig(t1, t2)
    figs["radar_cmp"].update_layout(
        title="Radar Comparison (Top Power Teams)",
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=True,
    )

    # ---------- turn figs into HTML divs ----------
    div_pfpa = plotly_plot(figs["pf_pa"], include_plotlyjs=False, output_type="div")
    div_line = plotly_plot(figs["scores_line"], include_plotlyjs=False, output_type="div")
    div_box = plotly_plot(figs["scores_box"], include_plotlyjs=False, output_type="div")
    div_radar = plotly_plot(figs["radar_cmp"], include_plotlyjs=False, output_type="div")

    # One-time Plotly JS blob
    plotly_js = f'<script>{get_plotlyjs()}</script>'

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

    # ---------- build page body (for {body} in BASE_HTML) ----------
    sidebar_html = f"""
        <div class="card small">
          <div class="card-header">
            <h3>Top Power Ranked Teams</h3>
          </div>
          <div class="card-body mini-body">
            {top3_html}
          </div>
        </div>

        <div class="card small">
          <div class="card-header">
            <h3>Metrics Key</h3>
          </div>
          <div class="card-body">
            <ul class="ticker-list">
              <li><span class="mini-label">PF</span> &mdash; Points For</li>
              <li><span class="mini-label">PA</span> &mdash; Points Against</li>
              <li><span class="mini-label">MAX</span> &mdash; Best weekly score</li>
              <li><span class="mini-label">MIN</span> &mdash; Worst weekly score</li>
              <li><span class="mini-label">AVG</span> &mdash; Average weekly score</li>
              <li><span class="mini-label">STD</span> &mdash; Volatility of scores</li>
            </ul>
          </div>
        </div>
    """

    main_html = f"""
      <div class="page-layout">
        <main class="page-main">
            <div class="graphs-page">
          <div class="card">
            <div class="card-header-row">
              <h2>PF vs PA Scatter</h2>
            </div>
            <div class="card-body">
              {div_pfpa}
            </div>
          </div>

          <div class="card">
            <div class="card-header-row">
              <h2>Weekly Scores by Team</h2>
            </div>
            <div class="card-body">
              {div_line}
            </div>
          </div>

          <div class="card">
            <div class="card-header-row">
              <h2>Score Distribution</h2>
            </div>
            <div class="card-body">
              {div_box}
            </div>
          </div>

          <div class="card">
            <div class="card-header-row">
              <h2>Radar Comparison</h2>
            </div>
            <div class="card-body">
              {div_radar}
            </div>
          </div>
          </div>
        </main>

        <aside class="page-sidebar">
          {sidebar_html}
        </aside>
      </div>

      <!-- Plotly library -->
      {plotly_js}
    """

    return main_html
