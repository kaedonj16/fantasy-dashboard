import html
import json
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from utils.utils import z_better_outward
from dashboard_services.plotly_theme import apply_brand_layout
from utils.all_play import all_play_analysis
from utils.standings_viz import luck_quadrant_svg, value_age_svg


def owner_color_map(owners) -> Dict[str, str]:
    """Per-owner color from the shared graphs COLOR_CYCLE, so a team is the same
    color across every chart on the page (and the new SVG scatters)."""
    m: Dict[str, str] = {}
    for idx, o in enumerate(owners or []):
        m[str(o)] = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
    return m


def _luck_and_value_age_cards(ctx: dict, df_weekly_finalized, owner_colors: dict = None) -> str:
    """Two league-wide SVG scatters for the graphs page: a performance-vs-luck
    quadrant (all-play win rate vs actual) and a dynasty-value-vs-age quadrant.
    Both are server-rendered SVG (no Plotly), and return '' individually when
    there isn't enough data, so the cards simply don't appear. ``owner_colors``
    is the shared per-team color map so points match the other charts."""
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
            luck_svg = luck_quadrant_svg(all_play_analysis(weekly_scores, actual_wins), viewer_owner, owner_colors)
    except Exception:
        luck_svg = ""

    value_age_svg_str = ""
    try:
        from dashboard_services.ai.context_builders import team_value_age_rows
        value_age_svg_str = value_age_svg(team_value_age_rows(ctx), viewer_owner, owner_colors)
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


def _consistency_card(team_stats, owner_colors: dict) -> str:
    """Boom/bust consistency ranking from each team's weekly scoring spread.

    Uses the coefficient of variation (STD / AVG): a lower ratio means a team
    posts a similar score every week (steady), a higher ratio means big swings
    (boom/bust). Teams are split into thirds by that ratio so the Steady /
    Balanced / Boom-Bust label is percentile-ranked within the league rather
    than tied to absolute thresholds that vary by scoring format."""
    rows = []
    for _, r in team_stats.iterrows():
        try:
            owner = r["owner"]
            avg = float(r.get("AVG") or 0)
            std = float(r.get("STD") or 0)
        except Exception:
            continue
        if avg <= 0:
            continue
        rows.append((owner, avg, std, std / avg))
    if len(rows) < 2:
        return ""
    rows.sort(key=lambda x: x[3])              # steadiest (lowest CV) first
    cvs = [x[3] for x in rows]
    lo, hi = min(cvs), max(cvs)
    rng = (hi - lo) or 1.0
    n = len(rows)

    def _band(i: int):
        # Percentile terciles: steadiest third, middle third, most volatile third.
        if i < n / 3:
            return "Steady", "#22c55e"
        if i < 2 * n / 3:
            return "Balanced", "#f59e0b"
        return "Boom / Bust", "#ef4444"

    body = ""
    for i, (owner, avg, std, cv) in enumerate(rows):
        label, col = _band(i)
        dot = owner_colors.get(owner, "#9ca3af")
        bar_w = 10 + (cv - lo) / rng * 90      # 10..100% of the track
        body += (
            f"<div class='cons-row'>"
            f"<span class='cons-dot' style='background:{dot};'></span>"
            f"<span class='cons-name'>{html.escape(str(owner))}</span>"
            f"<span class='cons-track'><span class='cons-fill' style='width:{bar_w:.0f}%;background:{col};'></span></span>"
            f"<span class='cons-vol' title='Week-to-week volatility (std / avg)'>{cv * 100:.0f}%</span>"
            f"<span class='cons-band' style='color:{col};background:{col}1a;'>{label}</span>"
            f"</div>"
        )
    return f"""
    <div class="card">
      <div class="card-header-row"><h2>Consistency (Boom / Bust)</h2></div>
      <div class="card-body">
        <div class="svg-graph-note">Week-to-week scoring volatility (standard deviation as a share of a team's average). Lower is steadier; higher swings between booms and busts. Ranked steadiest to most volatile.</div>
        <style>
          .cons-row {{ display:grid; grid-template-columns:12px minmax(90px,1.4fr) 3fr auto auto; align-items:center; gap:10px; padding:7px 2px; border-bottom:1px solid var(--border); }}
          .cons-row:last-child {{ border-bottom:none; }}
          .cons-dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
          .cons-name {{ font-weight:600; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
          .cons-track {{ height:8px; border-radius:5px; background:var(--row,rgba(127,127,127,.12)); overflow:hidden; }}
          .cons-fill {{ display:block; height:100%; border-radius:5px; }}
          .cons-vol {{ font-size:12px; font-weight:700; font-variant-numeric:tabular-nums; color:var(--muted); min-width:34px; text-align:right; }}
          .cons-band {{ font-size:10.5px; font-weight:800; letter-spacing:.03em; padding:2px 8px; border-radius:999px; white-space:nowrap; }}
          @media (max-width:520px) {{ .cons-row {{ grid-template-columns:10px minmax(70px,1fr) 2fr auto; }} .cons-vol {{ display:none; }} }}
        </style>
        {body}
      </div>
    </div>"""


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
    # Validated categorical palette (dataviz skill): worst adjacent CVD ΔE 9.1,
    # normal-vision ΔE 19.6; the lighter hues' sub-3:1 surface contrast is relieved
    # by the direct team labels + legend every chart carries. Leagues past 8 teams
    # cycle - no 10+ hue set can stay CVD-distinct, so 8 validated beats 10 raw.
    COLOR_CYCLE = [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
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
                cliponaxis=False,
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
    _xr = None
    if len(x) and np.isfinite(x).all():
        _pad = max((float(max(x)) - float(min(x))) * 0.12, 1.0)
        _xr = [float(min(x)) - _pad, float(max(x)) + _pad]
    figs["pf_pa"].update_layout(
        xaxis_title=dict(text="Points Against (PA)", standoff=12),
        xaxis=dict(range=_xr, automargin=True) if _xr else dict(automargin=True),
        yaxis_title=dict(text="Points For (PF)"),
        yaxis=dict(automargin=True),
        hovermode="closest",
        margin=dict(l=52, r=40, t=10, b=45),
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

    for _f in figs.values():
        apply_brand_layout(_f)

    pfpa_json = _fig_json(figs["pf_pa"])
    line_json  = _fig_json(figs["scores_line"])
    box_json   = _fig_json(figs["scores_box"])

    # League-wide SVG scatters (luck quadrant + dynasty value vs age).
    svg_cards_html = _luck_and_value_age_cards(ctx, df_weekly, owner_colors)
    consistency_html = _consistency_card(team_stats, owner_colors)

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
        font: {{ family: (window.brandPlotlyFont || 'system-ui, sans-serif'), size: 12.5, color: '#7c8798' }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        polar: {{ radialaxis: {{ visible: false }}, bgcolor: 'rgba(0,0,0,0)' }},
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

            {consistency_html}

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
                cliponaxis=False,
                marker=dict(size=12, color=owner_colors.get(owner), line=dict(color="black", width=1)),
                name=owner,
                showlegend=False,
            )
        )
    figs["pf_pa"] = go.Figure(scatter_traces)
    _cx = team_stats["PA"].values
    _cxr = None
    if len(_cx) and np.isfinite(_cx).all():
        _cpad = max((float(max(_cx)) - float(min(_cx))) * 0.12, 1.0)
        _cxr = [float(min(_cx)) - _cpad, float(max(_cx)) + _cpad]
    figs["pf_pa"].update_layout(
        xaxis_title=dict(text="Career Points Against", standoff=12),
        xaxis=dict(range=_cxr, automargin=True) if _cxr else dict(automargin=True),
        yaxis_title=dict(text="Career Points For"),
        yaxis=dict(automargin=True),
        hovermode="closest",
        margin=dict(l=52, r=40, t=10, b=45),
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

    for _f in figs.values():
        apply_brand_layout(_f)

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
        <div class="card small career-standings">
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


# Validated categorical palette (dataviz skill) - shared by the career graphs.
# Same set as build_graphs_body so a team keeps one color across season/career.
COLOR_CYCLE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]


# ── Page orchestration (moved out of app.py) ──────────────────────────────────
# These build the full /graphs page body. They take every app-level input as a
# parameter (the current league context, the completed-season list, a context
# provider for other seasons, the live value table, and the prebuilt page URL)
# so this module never imports app.py — the route wires the accessors in.

def build_tour_mock_graphs_ctx(df_weekly) -> dict:
    """Graphs ctx for the tour/demo, from a pre-built mock weekly frame."""
    from dashboard_services.pages.history_page import (
        build_regular_season_team_stats, sort_team_stats)
    df = df_weekly
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


def build_career_graphs_ctx(
    platform, league_id, season, available_seasons, get_ctx, only_owners=None,
) -> dict:
    """Aggregate team_stats and df_weekly across completed seasons for career graphs.

    ``get_ctx(platform, rid, season)`` fetches a league context (injected so this
    module stays independent of app.py). When ``only_owners`` is a non-empty set of
    owner names, the aggregate is restricted to those members (used by the current-
    vs-all-time toggle to drop owners no longer in the league)."""
    from dashboard_services.api import resolve_league_id_for_season
    from dashboard_services.pages.history_page import build_regular_season_team_stats

    career: dict = {}  # owner -> {Wins, Losses, Ties, PF, PA, weekly_pts, season_pf}
    season_pf_rows: list = []  # rows for per-season bar chart: {season, owner, pf}

    for hist_s in available_seasons:
        rid = resolve_league_id_for_season(platform, league_id, season, hist_s)
        try:
            hctx = get_ctx(platform, rid, hist_s)
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
            hctx = get_ctx(platform, rid, hist_s)
            df = hctx.get("df_weekly", pd.DataFrame()).copy()
            if not df.empty and "owner" in df.columns:
                df["season"] = hist_s
                combined_dfs.append(df)
        except Exception:
            continue

    df_combined = pd.concat(combined_dfs, ignore_index=True) if combined_dfs else pd.DataFrame()
    season_pf_df = pd.DataFrame(season_pf_rows) if season_pf_rows else pd.DataFrame()

    # Restrict to current members when the toggle asks for it.
    if only_owners:
        _keep = {str(o) for o in only_owners}
        if not team_stats.empty and "owner" in team_stats.columns:
            team_stats = team_stats[team_stats["owner"].astype(str).isin(_keep)].reset_index(drop=True)
        if not df_combined.empty and "owner" in df_combined.columns:
            df_combined = df_combined[df_combined["owner"].astype(str).isin(_keep)].reset_index(drop=True)
        if not season_pf_df.empty and "owner" in season_pf_df.columns:
            season_pf_df = season_pf_df[season_pf_df["owner"].astype(str).isin(_keep)].reset_index(drop=True)

    return {
        "team_stats": team_stats,
        "df_weekly": df_combined,
        "season_pf_df": season_pf_df,
        "is_career": True,
    }


def render_graphs_html(
    platform, season, league_id, view, members, *,
    ctx, available_seasons, get_ctx, model_value_table, graphs_base_url,
) -> str:
    """Build the /graphs page body for a given view ("career" or a season) and
    member filter. Every app-level input is injected:

      ctx                current league context
      available_seasons  completed seasons with data (newest-first)
      get_ctx            callable (platform, rid, season) -> league context
      model_value_table  live dynasty value rows (for the value/age scatter)
      graphs_base_url    the /graphs URL for this league (selector links)

    Pure of request args (view/members are passed in) so the heavy career view -
    which aggregates every past season - can build in a background thread."""
    import logging
    from dashboard_services.api import resolve_league_id_for_season
    logger = logging.getLogger(__name__)

    offseason = bool(ctx.get("offseason_mode"))

    # Current members = current rosters' owner names. roster_map is populated even
    # in the offseason (team_stats can be empty then), so prefer it; the career
    # aggregate is keyed by these same owner names.
    current_owners = set()
    _rmap = ctx.get("roster_map") or {}
    if _rmap:
        current_owners = {str(v) for v in _rmap.values() if v}
    if not current_owners:
        _cur_ts = ctx.get("team_stats")
        if _cur_ts is not None and not _cur_ts.empty and "owner" in _cur_ts.columns:
            current_owners = {str(o) for o in _cur_ts["owner"].tolist()}

    # Build the season-selector dropdown (navigate via URL query param)
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

    # Current-vs-all-time member toggle (only meaningful for the career view,
    # where former members would otherwise appear across every chart).
    members_toggle_html = ""
    if view == "career":
        _cur_url = f"{graphs_base_url}?view=career&members=current"
        _all_url = f"{graphs_base_url}?view=career&members=all"
        members_toggle_html = f"""
        <div class="members-toggle" role="group" aria-label="Member filter">
          <a class="members-toggle-btn {'active' if members == 'current' else ''}" href="{_cur_url}">Current members</a>
          <a class="members-toggle-btn {'active' if members == 'all' else ''}" href="{_all_url}">All-time</a>
        </div>"""

    season_selector_html = f"""
    <div class="graphs-season-selector">
      <label class="graphs-season-label">View:</label>
      <select class="graphs-season-select" onchange="window.location.href=this.value">
        {"".join(selector_opts)}
      </select>
      {members_toggle_html}
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
                career_ctx = build_career_graphs_ctx(
                    platform, league_id, season, available_seasons, get_ctx,
                    only_owners=(current_owners if members == "current" else None),
                )
                charts_html = build_career_graphs_body(career_ctx)

                # The luck + value/age scatters aren't season-aggregates, so the
                # career builder doesn't emit them. Inject them here: value/age
                # from the current rosters, luck from the most recent completed
                # season's weekly results.
                try:
                    _co = None
                    _cts = career_ctx.get("team_stats")
                    if _cts is not None and not _cts.empty and "owner" in _cts.columns:
                        _co = owner_color_map(_cts["owner"].tolist())
                    _luck_df = None
                    _latest = max(available_seasons)
                    _lrid = resolve_league_id_for_season(platform, league_id, season, _latest)
                    _lctx = get_ctx(platform, _lrid, _latest)
                    _ldf = _lctx.get("df_weekly")
                    if _ldf is not None and not _ldf.empty and "finalized" in _ldf.columns:
                        _luck_df = _ldf[_ldf["finalized"] == True]
                        if members == "current" and current_owners and "owner" in _luck_df.columns:
                            _luck_df = _luck_df[_luck_df["owner"].astype(str).isin(current_owners)]
                    # Use the live value table (the ctx copy can be stale/empty),
                    # so the dynasty-value-vs-age scatter always has real values.
                    _val_ctx = {**ctx, "model_value_table": (model_value_table or ctx.get("model_value_table") or [])}
                    _svg_cards = _luck_and_value_age_cards(_val_ctx, _luck_df, _co)
                    if _svg_cards and '<div class="graphs-page">' in charts_html:
                        charts_html = charts_html.replace(
                            '<div class="graphs-page">',
                            '<div class="graphs-page">' + _svg_cards, 1,
                        )
                except Exception:
                    logger.debug("career graphs scatter injection failed", exc_info=True)
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
            season_ctx = get_ctx(platform, rid, target_season)

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

    return season_selector_html + charts_html
