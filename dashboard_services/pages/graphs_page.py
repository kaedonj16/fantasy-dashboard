"""Graphs page: league value/performance charts ("League Stats").

Data contract for ``build_graphs_body(ctx)`` / ``render_graphs_html``:
- ``ctx["team_stats"]``: PF/PA/AVG/STD per owner
- ``ctx["df_weekly"]`` with ``finalized == True`` for in-season plots
- ``ctx["viewer"]["viewer_team_name"]``: the viewer's team, highlighted in every chart
- Empty weekly data returns a static "No weekly data" card (never a spinner hang)
- Cold-cache Graphs requests use a chart-shaped ``.graphs-skeleton`` (not a generic list)
- Career view may render a skeleton while a background aggregation fills in

Layout: a four-tab shell (Performance | Value | Trends | Career). Each tab is a
2-column chart grid (1 column on mobile). Every chart card carries a title, a
subtitle, the chart, and a plain-English insight computed from the real
underlying data.
"""
import html
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dashboard_services.plotly_theme import apply_brand_layout
from utils.standings import all_play_analysis
from utils.standings_viz import luck_quadrant_svg, value_age_svg


# Validated categorical palette (dataviz skill) - shared by the season and
# career graphs so a team keeps one color across every chart on the page.
# Leagues past 8 teams cycle - no 10+ hue set can stay CVD-distinct, so 8
# validated beats 10 raw.
COLOR_CYCLE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

_YOU_BLUE = "#1e3a5f"
_LEAGUE_GRAY = "#94a3b8"
_ABOVE_AMBER = "#f59e0b"
_BELOW_BLUE = "#60a5fa"


def owner_color_map(owners) -> Dict[str, str]:
    """Per-owner color from the shared graphs COLOR_CYCLE, so a team is the same
    color across every chart on the page."""
    m: Dict[str, str] = {}
    for idx, o in enumerate(owners or []):
        m[str(o)] = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
    return m


def _viewer_owner(ctx: dict) -> str:
    return str((ctx.get("viewer") or {}).get("viewer_team_name") or "")


def _league_name(ctx: dict) -> str:
    return str((ctx.get("league") or {}).get("name") or "")


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ── Shared tab shell ─────────────────────────────────────────────────────────

def _graphs_style() -> str:
    """Scoped CSS for the tabbed League Stats page. Uses theme vars with
    light fallbacks so cards read in both themes."""
    return """<style>
.gs-head{margin:0 0 4px;}
.gs-head h1{font-size:22px;font-weight:800;margin:0 0 2px;}
.gs-head p{font-size:13px;color:var(--text-muted,#64748b);margin:0 0 14px;}
.gs-tabs{display:flex;gap:8px;margin-bottom:16px;overflow-x:auto;padding-bottom:2px;}
.gs-tab{padding:8px 16px;border-radius:8px;font-size:13px;font-weight:700;background:var(--card,#fff);border:1px solid var(--border,#e2e8f0);color:var(--text-muted,#64748b);white-space:nowrap;cursor:pointer;font-family:inherit;}
.gs-tab.active{background:#1e3a5f;color:#fff;border-color:#1e3a5f;}
.gs-pane{display:none;}
.gs-pane.active{display:block;}
.gs-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;}
@media (max-width:700px){.gs-grid{grid-template-columns:1fr;}}
.gs-sub{font-size:12px;color:var(--text-muted,#94a3b8);margin-bottom:10px;}
.gs-insight{margin-top:12px;padding:10px 12px;background:rgba(30,58,95,.07);border-radius:8px;font-size:13px;line-height:1.5;}
.gs-legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:11px;color:var(--text-muted,#64748b);}
.gs-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:baseline;}
.gs-bar-row{display:grid;grid-template-columns:minmax(90px,130px) 1fr 52px;gap:8px;align-items:center;padding:5px 6px;margin:0 -6px;font-size:12px;border-radius:6px;}
.gs-bar-row .nm{font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.gs-bar-row.you{background:rgba(30,58,95,.08);}
.gs-track{height:10px;background:rgba(127,127,127,.14);border-radius:5px;overflow:hidden;}
.gs-fill{height:100%;border-radius:5px;}
.gs-bar-row .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--text-muted,#475569);}
.gs-btn{display:inline-block;margin-top:10px;padding:9px 18px;border-radius:8px;background:#1e3a5f;color:#fff;font-weight:700;font-size:13px;text-decoration:none;}
.cons-row{display:grid;grid-template-columns:12px minmax(90px,1.4fr) 3fr auto auto;align-items:center;gap:10px;padding:7px 2px;border-bottom:1px solid var(--border,#f1f5f9);}
.cons-row:last-child{border-bottom:none;}
.cons-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.cons-name{font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.cons-track{height:8px;border-radius:5px;background:rgba(127,127,127,.14);overflow:hidden;}
.cons-fill{display:block;height:100%;border-radius:5px;}
.cons-vol{font-size:13px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--text-muted,#64748b);min-width:34px;text-align:right;}
.cons-band{font-size:11px;font-weight:800;letter-spacing:.03em;padding:2px 8px;border-radius:8px;white-space:nowrap;}
@media (max-width:520px){.cons-row{grid-template-columns:10px minmax(70px,1fr) 2fr auto;}.cons-vol{display:none;}}
.gs-details{margin-top:12px;border-top:1px solid var(--border,#f1f5f9);padding-top:10px;}
.gs-details summary{cursor:pointer;font-size:13px;font-weight:700;color:#1e3a5f;}
.gs-select{background:var(--card,#fff);border:1px solid var(--border,#e2e8f0);border-radius:8px;padding:6px 10px;font-size:13px;font-family:inherit;max-width:160px;}
.h2h-wrap{overflow-x:auto;}
.h2h-table{border-collapse:collapse;font-size:11px;min-width:100%;}
.h2h-table th,.h2h-table td{padding:6px 8px;text-align:center;border:1px solid var(--border,#eef2f6);white-space:nowrap;}
.h2h-table thead th{font-size:10px;color:var(--text-muted,#64748b);}
.h2h-table tbody th{text-align:left;font-weight:700;font-size:12px;position:sticky;left:0;background:var(--card,#fff);}
.h2h-table tr.h2h-you th,.h2h-table tr.h2h-you td{background:rgba(30,58,95,.06);}
.h2h-w{color:#16a34a;font-weight:700;background:rgba(22,163,74,.08);}
.h2h-l{color:#ef4444;background:rgba(239,68,68,.06);}
.h2h-t{color:var(--text-muted,#64748b);}
.h2h-na{color:var(--text-muted,#cbd5e1);}
.pos-legend{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;font-size:11px;color:var(--text-muted,#64748b);}
.pos-row{display:grid;grid-template-columns:minmax(90px,130px) 1fr 52px;gap:8px;align-items:center;padding:5px 6px;margin:0 -6px;font-size:12px;border-radius:6px;}
.pos-row .nm{font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.pos-row.you{background:rgba(30,58,95,.08);}
.pos-track{height:14px;border-radius:7px;overflow:hidden;display:flex;background:rgba(127,127,127,.14);}
.pos-seg{height:100%;}
.pos-row .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--text-muted,#475569);}
</style>"""


def _page_header(league_name: str, season_label: str) -> str:
    sub = " · ".join(p for p in (league_name, season_label) if p)
    sub_html = f"<p>{html.escape(sub)}</p>" if sub else ""
    return f'<div class="gs-head"><h1>League Stats</h1>{sub_html}</div>'


def _tab_shell(active_tab: str, panes: Dict[str, tuple]) -> str:
    """Tab bar + panes. ``panes`` maps key -> (label, cards_html). The active
    pane's cards sit in the 2-column grid."""
    tabs = []
    bodies = []
    for key, (label, cards_html) in panes.items():
        cls = "gs-tab active" if key == active_tab else "gs-tab"
        tabs.append(
            f'<button class="{cls}" data-tab="{key}" role="tab" '
            f'aria-selected="{str(key == active_tab).lower()}">{html.escape(label)}</button>'
        )
        pane_cls = "gs-pane active" if key == active_tab else "gs-pane"
        bodies.append(
            f'<div class="{pane_cls}" id="gs-pane-{key}" role="tabpanel">'
            f'<div class="gs-grid">{cards_html}</div></div>'
        )
    return (
        f'<div class="gs-tabs" role="tablist">{"".join(tabs)}</div>'
        f'{"".join(bodies)}'
    )


def _tabs_js() -> str:
    """Client-side tab switching. Plotly charts render while their pane is
    hidden at 0 width, so resize any plot in the pane being revealed."""
    return """<script>(function(){
var tabs=document.querySelectorAll('.gs-tabs .gs-tab[data-tab]');
tabs.forEach(function(t){t.addEventListener('click',function(){
tabs.forEach(function(x){x.classList.remove('active');x.setAttribute('aria-selected','false');});
t.classList.add('active');t.setAttribute('aria-selected','true');
document.querySelectorAll('.gs-pane').forEach(function(p){p.classList.remove('active');});
var pane=document.getElementById('gs-pane-'+t.getAttribute('data-tab'));
if(pane){pane.classList.add('active');
if(window.Plotly){pane.querySelectorAll('.js-plotly-plot').forEach(function(el){try{window.Plotly.Plots.resize(el);}catch(e){}});}}
});});})();</script>"""


def _insight(text: str) -> str:
    return f'<div class="gs-insight">{text}</div>' if text else ""


def _card(title: str, subtitle: str, body_html: str, insight: str = "") -> str:
    sub = f'<div class="gs-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    return (
        f'<div class="card"><div class="card-header-row"><h2>{html.escape(title)}</h2></div>'
        f'<div class="card-body graph-body">{sub}{body_html}{_insight(insight)}</div></div>'
    )


def _empty_card(message: str) -> str:
    return (
        f'<div class="card"><div class="card-body">'
        f'<p style="color:var(--text-muted);">{html.escape(message)}</p>'
        f"</div></div>"
    )


def _legend(items: List[tuple]) -> str:
    spans = "".join(
        f'<span><span class="gs-dot" style="background:{color}"></span>{html.escape(label)}</span>'
        for color, label in items
    )
    return f'<div class="gs-legend">{spans}</div>'


def _deferred_plotly_js(figs: Dict[str, str]) -> str:
    """Render registered Plotly figs once Plotly loads. ``figs`` maps element
    id -> fig JSON (already escaped)."""
    if not figs:
        return ""
    entries = ",".join(f'"{cid}":' + fjson for cid, fjson in figs.items())
    return (
        "<script>(function(){"
        "var _FIGS={" + entries + "};"
        "var _CFG={responsive:true,displayModeBar:false};"
        "(window.ensurePlotly?window.ensurePlotly():Promise.resolve(window.Plotly)).then(function(P){"
        "if(!P)return;"
        "Object.keys(_FIGS).forEach(function(id){"
        "var f=_FIGS[id];var el=document.getElementById(id);"
        "if(el)P.newPlot(el,f.data,f.layout,_CFG);"
        "});});})();</script>"
    )


def _fig_json(fig) -> str:
    return fig.to_json().replace("</", "<\\/")


# ── Performance tab ──────────────────────────────────────────────────────────

def _luck_analysis(df_weekly_finalized) -> dict:
    """{owner: all-play row} from finalized weekly scores."""
    weekly_scores: dict = {}
    actual_wins: dict = {}
    for _, r in df_weekly_finalized.iterrows():
        try:
            wk = int(r["week"])
            owner = str(r["owner"])
        except Exception:
            continue
        weekly_scores.setdefault(wk, {})[owner] = float(r["points"] or 0)
        actual_wins[owner] = actual_wins.get(owner, 0.0) + float(r.get("win") or 0)
    return all_play_analysis(weekly_scores, actual_wins)


def _luck_card(df_weekly_finalized, viewer_owner: str, owner_colors: dict) -> str:
    """Performance vs Luck scatter (all-play win rate vs actual), with a
    plain-English read of the viewer's luck delta."""
    try:
        analysis = _luck_analysis(df_weekly_finalized)
        svg = luck_quadrant_svg(analysis, viewer_owner, owner_colors)
    except Exception:
        return ""
    if not svg:
        return ""
    insight = ""
    me = analysis.get(viewer_owner) if viewer_owner else None
    if me and me.get("luck_delta") is not None and me.get("games"):
        aw = float(me["actual_wins"])
        exp = float(me["expected_wins"])
        delta = float(me["luck_delta"])
        games = int(me["games"])
        losses = games - aw
        if float(aw).is_integer() and float(losses).is_integer():
            rec = f"{int(aw)}-{int(losses)}"
        else:
            rec = f"{aw:.1f} wins in {games} games"
        if delta >= 1.0:
            verdict = "You've been lucky."
        elif delta <= -1.0:
            verdict = "You've been unlucky."
        else:
            verdict = "Your record matches your scoring."
        insight = (
            f"You're {rec} but your all-play record says {exp:.1f} wins. {verdict}"
        )
    you_color = owner_colors.get(viewer_owner, _YOU_BLUE) if viewer_owner else _YOU_BLUE
    legend = _legend([(you_color, "You"), (_LEAGUE_GRAY, "League")])
    return _card(
        "Performance vs Luck",
        "All-play win rate (true strength) vs actual win rate. Above the dashed line means more wins than your scoring earned.",
        f'<div class="svg-graph-body">{svg}</div>{legend}',
        insight,
    )


def _consistency_card(team_stats, owner_colors: dict, viewer_owner: str, detail_html: str = "") -> str:
    """Boom/bust consistency ranking from each team's weekly scoring spread.

    Uses the coefficient of variation (STD / AVG). Teams are split into thirds
    by that ratio so the Steady / Balanced / Boom-Bust label is percentile-ranked
    within the league rather than tied to absolute thresholds that vary by
    scoring format."""
    rows = []
    for _, r in team_stats.iterrows():
        try:
            owner = str(r["owner"])
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
    viewer_rank = None
    viewer_band = ""
    for i, (owner, avg, std, cv) in enumerate(rows):
        label, col = _band(i)
        is_you = viewer_owner and owner == viewer_owner
        if is_you:
            viewer_rank = i + 1
            viewer_band = label
        dot = owner_colors.get(owner, "#9ca3af")
        bar_w = 10 + (cv - lo) / rng * 90      # 10..100% of the track
        name = html.escape(owner) + (" (you)" if is_you else "")
        body += (
            f"<div class='cons-row'>"
            f"<span class='cons-dot' style='background:{dot};'></span>"
            f"<span class='cons-name'>{name}</span>"
            f"<span class='cons-track'><span class='cons-fill' style='width:{bar_w:.0f}%;background:{col};'></span></span>"
            f"<span class='cons-vol' title='Week-to-week volatility (std / avg)'>{cv * 100:.0f}%</span>"
            f"<span class='cons-band' style='color:{col};background:{col}1a;'>{label}</span>"
            f"</div>"
        )
    insight = ""
    if viewer_rank is not None:
        cv = rows[viewer_rank - 1][3]
        insight = (
            f"You rank {_ordinal(viewer_rank)} of {n} in steadiness "
            f"({cv * 100:.0f}% spread), in the {viewer_band} third."
        )
    return _card(
        "Consistency Ranking",
        "Weekly scoring spread (coefficient of variation). Lower is steadier; higher swings between booms and busts.",
        body + (detail_html or ""),
        insight,
    )


def _pf_pa_card(team_stats, owner_colors: dict, viewer_owner: str, figs: dict) -> str:
    """PF vs PA scatter: the classic good-offense / bad-defense quadrants, with
    a trend line and smart label placement for dense clusters."""
    try:
        from utils.scatter_labels import scatter_label_placements
    except Exception:
        scatter_label_placements = None
    try:
        owners = [str(r["owner"]) for _, r in team_stats.iterrows()]
        pas = [float(r["PA"]) for _, r in team_stats.iterrows()]
        pfs = [float(r["PF"]) for _, r in team_stats.iterrows()]
        if len(owners) < 2:
            return ""
        label_plan = (
            scatter_label_placements(pas, pfs, owners)
            if scatter_label_placements
            else [(o, "top center") for o in owners]
        )
        traces = []
        for i, owner in enumerate(owners):
            text, textposition = label_plan[i]
            traces.append(
                go.Scatter(
                    x=[pas[i]], y=[pfs[i]],
                    mode="markers+text" if text else "markers",
                    text=[text] if text else None,
                    textposition=textposition if text else None,
                    cliponaxis=False,
                    marker=dict(
                        size=13,
                        line=dict(color="black", width=1),
                        color=owner_colors.get(owner, "#9ca3af"),
                    ),
                    name=owner,
                    showlegend=False,
                    hovertemplate=f"{html.escape(owner)}<br>PA: %{{x:.1f}}<br>PF: %{{y:.1f}}<extra></extra>",
                )
            )
        x = np.array(pas)
        y = np.array(pfs)
        if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
            m = ((x - x.mean()) * (y - y.mean())).sum() / max(
                ((x - x.mean()) ** 2).sum(), 1e-9
            )
            b = y.mean() - m * x.mean()
            xs = [float(min(x) * 0.95), float(max(x) * 1.05)]
            ys = [m * xs[0] + b, m * xs[1] + b]
            traces.append(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(dash="dash", color="#9ca3af"),
                    name="Trend", showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig = go.Figure(traces)
        pad = max((float(max(x)) - float(min(x))) * 0.12, 1.0)
        fig.update_layout(
            xaxis_title=dict(text="Points Against (PA)", standoff=12),
            xaxis=dict(range=[float(min(x)) - pad, float(max(x)) + pad], automargin=True),
            yaxis_title=dict(text="Points For (PF)"),
            yaxis=dict(automargin=True),
            hovermode="closest",
            margin=dict(l=52, r=40, t=10, b=45),
            showlegend=False,
        )
        apply_brand_layout(fig)
        figs["chart-pfpa"] = _fig_json(fig)

        insight = ""
        if viewer_owner and viewer_owner in owners:
            i = owners.index(viewer_owner)
            my_pf, my_pa = pfs[i], pas[i]
            avg_pf = float(np.mean(pfs))
            avg_pa = float(np.mean(pas))
            good_o = my_pf >= avg_pf
            good_d = my_pa <= avg_pa
            if good_o and good_d:
                read = "Good offense, good defense. A complete team."
            elif good_o and not good_d:
                read = "Good offense, leaky defense. You win shootouts and lose them too."
            elif not good_o and good_d:
                read = "Stingy defense, quiet offense. You need more scoring."
            else:
                read = "Below average on both sides. The roster needs work."
            insight = f"You're at {my_pf:.0f} PF / {my_pa:.0f} PA vs league averages of {avg_pf:.0f} / {avg_pa:.0f}. {read}"
        return _card(
            "PF vs PA",
            "Total points scored vs allowed. Top-left is the promised land: outscoring everyone while nobody scores on you.",
            '<div id="chart-pfpa" style="width:100%;min-height:350px;"></div>',
            insight,
        )
    except Exception:
        return ""


def _boxplot_detail_html(df_weekly, owner_colors: dict, figs: dict) -> str:
    """Score distribution boxplot, returned as a <details> expander for the
    Consistency Ranking card."""
    try:
        order = (
            df_weekly.groupby("owner")["points"]
            .median().sort_values(ascending=False)
            .index.tolist()
        )
        if len(order) < 2:
            return ""
        traces = []
        for o in order:
            pts = df_weekly.loc[df_weekly["owner"] == o, "points"]
            traces.append(
                go.Box(
                    y=[float(v) for v in pts.tolist()],
                    name=str(o),
                    boxmean=True,
                    orientation="v",
                    hoveron="boxes",
                    boxpoints=False,
                    marker=dict(color=owner_colors.get(str(o), "#9ca3af")),
                    showlegend=False,
                )
            )
        fig = go.Figure(traces)
        fig.update_layout(
            xaxis_title=dict(text="Team", standoff=12),
            yaxis_title=dict(text="Points"),
            hovermode="closest",
            margin=dict(l=40, r=20, t=10, b=120),
            showlegend=False,
        )
        apply_brand_layout(fig)
        figs["chart-boxdist"] = _fig_json(fig)
        return (
            "<details class='gs-details'><summary>Show score distributions</summary>"
            '<div id="chart-boxdist" style="width:100%;min-height:350px;"></div>'
            "<p class='gs-sub'>Median, quartiles, and mean (x) per team. Wide boxes mean boom-or-bust weeks.</p>"
            "</details>"
        )
    except Exception:
        return ""


def _radar_card(team_stats, owner_colors: dict, figs: dict) -> str:
    """Radar comparison: two selectable teams across PF/PA/MAX/MIN/AVG/STD
    z-scores, with the league average as a dashed ring."""
    try:
        from utils.core import z_better_outward
        import json as _json
    except Exception:
        return ""
    try:
        metrics = ["PF", "PA", "MAX", "MIN", "AVG", "STD"]
        owners = [str(r["owner"]) for _, r in team_stats.iterrows()]
        if len(owners) < 2:
            return ""
        Z = z_better_outward(team_stats, metrics)
        z_map = {
            str(team_stats.loc[i, "owner"]): Z.iloc[i].values.astype(float).tolist()
            for i in range(len(team_stats))
        }
        opts_a, opts_b = [], []
        for i, o in enumerate(owners):
            esc = html.escape(o)
            opts_a.append(f"<option value='{esc}'{' selected' if i == 0 else ''}>{esc}</option>")
            opts_b.append(f"<option value='{esc}'{' selected' if i == 1 else ''}>{esc}</option>")
        js = (
            "<script>(function(){"
            f"var ZMAP={_json.dumps(z_map)};"
            f"var METRICS={_json.dumps(metrics)};"
            f"var COLORS={_json.dumps(owner_colors)};"
            "var closeRing=function(a){return a.concat(a[0]);};"
            "function makeRadarData(a,b){"
            "var za=ZMAP[a]||METRICS.map(function(){return 0;});"
            "var zb=ZMAP[b]||METRICS.map(function(){return 0;});"
            "return[{type:'scatterpolar',r:closeRing(METRICS.map(function(){return 0;}))"
            ",theta:closeRing(METRICS),name:'League avg',"
            "line:{dash:'dash',color:'#9ca3af'},opacity:.8},"
            "{type:'scatterpolar',r:closeRing(za),theta:closeRing(METRICS),name:a,"
            "fill:'toself',opacity:.45,line:{color:COLORS[a]||'#1f77b4'},"
            "fillcolor:COLORS[a]||'#1f77b4'},"
            "{type:'scatterpolar',r:closeRing(zb),theta:closeRing(METRICS),name:b,"
            "fill:'toself',opacity:.45,line:{color:COLORS[b]||'#ff7f0e'},"
            "fillcolor:COLORS[b]||'#ff7f0e'}];}"
            "function renderRadar(a,b){var el=document.getElementById('radar-cmp');"
            "if(!el)return;"
            "var layout={font:{family:'system-ui,sans-serif',size:12,color:'#7c8798'},"
            "paper_bgcolor:'rgba(0,0,0,0)',"
            "polar:{radialaxis:{visible:false},bgcolor:'rgba(0,0,0,0)'},"
            "showlegend:true,legend:{orientation:'h',y:-0.15},"
            "margin:{l:40,r:20,t:30,b:60}};"
            "(window.ensurePlotly?window.ensurePlotly():Promise.resolve(window.Plotly))"
            ".then(function(P){if(!P)return;"
            "if(!el._plotted){P.newPlot(el,makeRadarData(a,b),layout,"
            "{responsive:true,displayModeBar:false});el._plotted=true;}"
            "else{P.react(el,makeRadarData(a,b),layout);}});}"
            "function _initRadar(){var sA=document.getElementById('radarTeamA');"
            "var sB=document.getElementById('radarTeamB');if(!sA||!sB)return;"
            "renderRadar(sA.value,sB.value);"
            "sA.addEventListener('change',function(){renderRadar(sA.value,sB.value);});"
            "sB.addEventListener('change',function(){renderRadar(sA.value,sB.value);});}"
            "if(document.readyState==='loading'){"
            "document.addEventListener('DOMContentLoaded',_initRadar);}"
            "else{_initRadar();}})();</script>"
        )
        body = (
            "<div class='radar-selectors' style='display:flex;gap:10px;margin-bottom:8px;'>"
            f"<label style='font-size:12px;'>Team A <select id='radarTeamA' class='gs-select'>{''.join(opts_a)}</select></label>"
            f"<label style='font-size:12px;'>Team B <select id='radarTeamB' class='gs-select'>{''.join(opts_b)}</select></label>"
            "</div>"
            '<div id="radar-cmp" style="width:100%;min-height:380px;"></div>'
            + js
        )
        return _card(
            "Radar Comparison",
            "Two teams across six scoring metrics, as z-scores vs the league. Further out is better on every axis.",
            body,
            "",
        )
    except Exception:
        return ""


def _margin_card(df_weekly, owner_colors: dict, viewer_owner: str) -> str:
    """Margin of victory / defeat: average blowout size in wins vs average
    heartbreak size in losses, per team."""
    try:
        if "points_against" not in getattr(df_weekly, "columns", []):
            return ""
        sub = df_weekly.dropna(subset=["points_against"]).copy()
        if sub.empty:
            return ""
        sub["margin"] = sub["points"].astype(float) - sub["points_against"].astype(float)
        rows = []
        for owner, g in sub.groupby("owner"):
            wins = g[g["margin"] > 0]["margin"]
            losses = g[g["margin"] < 0]["margin"]
            rows.append((
                str(owner),
                float(wins.mean()) if not wins.empty else 0.0,
                float(losses.mean()) if not losses.empty else 0.0,
                len(wins), len(losses),
            ))
        if len(rows) < 2:
            return ""
        rows.sort(key=lambda r: r[1] - abs(r[2]), reverse=True)
        body = ""
        viewer_note = ""
        for owner, wavg, lavg, nw, nl in rows:
            is_you = viewer_owner and owner == viewer_owner
            cls = "gs-bar-row you" if is_you else "gs-bar-row"
            name = html.escape(owner) + (" (you)" if is_you else "")
            color = owner_colors.get(owner, "#9ca3af")
            w_txt = f"+{wavg:.1f}" if nw else "0-0"
            l_txt = f"{lavg:.1f}" if nl else "0-0"
            body += (
                f'<div class="{cls}"><span class="nm">{name}</span>'
                f'<span class="v" style="text-align:left;">'
                f'<span style="color:#16a34a;font-weight:700;">{w_txt}</span>'
                f'<span style="color:var(--text-muted,#94a3b8);"> / </span>'
                f'<span style="color:#ef4444;font-weight:700;">{l_txt}</span>'
                f"</span>"
                f'<span class="v">{nw}W-{nl}L</span></div>'
            )
            if is_you:
                if wavg >= abs(lavg):
                    viewer_note = (
                        f"When you win, you win big (+{wavg:.1f} avg). "
                        f"When you lose, it's closer ({lavg:.1f} avg)."
                    )
                else:
                    viewer_note = (
                        f"Your wins are tight (+{wavg:.1f} avg) but your losses "
                        f"sting ({lavg:.1f} avg)."
                    )
        return _card(
            "Margin of Victory / Defeat",
            "Average blowout size in wins (green) vs average heartbreak size in losses (red).",
            body,
            viewer_note,
        )
    except Exception:
        return ""


def _h2h_card(df_weekly, owner_colors: dict, viewer_owner: str) -> str:
    """Head-to-head matrix: every team's all-time record against every other
    team this season, from weekly matchup pairings."""
    try:
        from dashboard_services.service import owner_pairs_from_weekly
    except Exception:
        return ""
    try:
        pairs = owner_pairs_from_weekly(df_weekly)
        if not pairs:
            return ""
        pts = {}
        for _, r in df_weekly.iterrows():
            try:
                pts[(int(r["week"]), str(r["owner"]))] = float(r["points"] or 0)
            except Exception:
                continue
        owners = sorted({str(o) for _, a, b in pairs for o in (a, b)})
        if len(owners) < 2:
            return ""
        rec = {a: {b: [0, 0] for b in owners if b != a} for a in owners}
        for wk, a, b in pairs:
            pa = pts.get((wk, a))
            pb = pts.get((wk, b))
            if pa is None or pb is None or pa == pb:
                continue
            if pa > pb:
                rec[a][b][0] += 1
                rec[b][a][1] += 1
            else:
                rec[b][a][0] += 1
                rec[a][b][1] += 1
        # Table: rows = team, cols = opponent, cell = W-L
        head = "".join(
            f"<th title='{html.escape(o)}'>{html.escape(o[:3])}</th>" for o in owners
        )
        body_rows = ""
        for a in owners:
            cells = ""
            for b in owners:
                if a == b:
                    cells += "<td class='h2h-self'>-</td>"
                    continue
                w, l = rec[a][b]
                if w == 0 and l == 0:
                    cells += "<td class='h2h-na' title='Haven&apos;t played'>-</td>"
                    continue
                cls = "h2h-w" if w > l else ("h2h-l" if l > w else "h2h-t")
                cells += f"<td class='{cls}' title='{html.escape(a)} vs {html.escape(b)}'>{w}-{l}</td>"
            row_cls = "h2h-you" if (viewer_owner and a == viewer_owner) else ""
            body_rows += (
                f"<tr class='{row_cls}'><th>{html.escape(a)}</th>{cells}</tr>"
            )
        table = (
            "<div class='h2h-wrap'><table class='h2h-table'>"
            f"<thead><tr><th></th>{head}</tr></thead>"
            f"<tbody>{body_rows}</tbody></table></div>"
        )
        insight = ""
        if viewer_owner and viewer_owner in owners:
            nemesis, best = None, None
            for b in owners:
                if b == viewer_owner:
                    continue
                w, l = rec[viewer_owner][b]
                if w + l == 0:
                    continue
                if best is None or w - l > best[1] - best[2]:
                    best = (b, w, l)
                if nemesis is None or l - w > nemesis[2] - nemesis[1]:
                    nemesis = (b, w, l)
            parts = []
            if nemesis and nemesis[2] > nemesis[1]:
                parts.append(f"{html.escape(nemesis[0])} owns you ({nemesis[1]}-{nemesis[2]}).")
            if best and best[1] > best[2]:
                parts.append(f"You dominate {html.escape(best[0])} ({best[1]}-{best[2]}).")
            insight = " ".join(parts)
        return _card(
            "Head-to-Head Matrix",
            "Your record against every other team this season. Green cells are winning records.",
            table,
            insight,
        )
    except Exception:
        return ""


# ── Value tab ────────────────────────────────────────────────────────────────

def _value_age_card(rows: list, viewer_owner: str, owner_colors: dict) -> str:
    """Dynasty Value vs Age scatter, with the viewer's window read from real
    value rank and age vs the league average."""
    try:
        svg = value_age_svg(rows, viewer_owner, owner_colors)
    except Exception:
        return ""
    if not svg:
        return ""
    insight = ""
    me = next((r for r in rows if str(r.get("owner")) == viewer_owner), None) if viewer_owner else None
    if me:
        vals = sorted((float(r.get("total_value") or 0) for r in rows), reverse=True)
        ages = [float(r.get("avg_age") or 0) for r in rows if (r.get("avg_age") or 0) > 0]
        rank = vals.index(float(me.get("total_value") or 0)) + 1
        my_age = float(me.get("avg_age") or 0)
        league_age = sum(ages) / len(ages) if ages else 0
        younger = my_age < league_age
        insight = (
            f"Your roster (age {my_age:.1f}) is "
            f"{'younger than' if younger else 'older than'} the league average "
            f"({league_age:.1f}) and ranks {_ordinal(rank)} of {len(vals)} in dynasty value."
        )
    you_color = owner_colors.get(viewer_owner, _YOU_BLUE) if viewer_owner else _YOU_BLUE
    legend = _legend([(you_color, "You"), (_LEAGUE_GRAY, "League")])
    return _card(
        "Dynasty Value vs Age",
        "Each team by total roster value and average age. Top-left is young and loaded; top-right is a closing win-now window.",
        f'<div class="svg-graph-body">{svg}</div>{legend}',
        insight,
    )


def _roster_value_card(rows: list, viewer_owner: str, owner_colors: dict) -> str:
    """Roster Value by Team: horizontal bars, viewer row highlighted."""
    valued = [r for r in (rows or []) if (r.get("total_value") or 0) > 0]
    if len(valued) < 2:
        return ""
    valued.sort(key=lambda r: -float(r.get("total_value") or 0))
    top = float(valued[0].get("total_value") or 0) or 1.0
    body = ""
    viewer_rank = None
    for i, r in enumerate(valued):
        owner = str(r.get("owner"))
        v = float(r.get("total_value") or 0)
        is_you = viewer_owner and owner == viewer_owner
        if is_you:
            viewer_rank = i + 1
        cls = "gs-bar-row you" if is_you else "gs-bar-row"
        name = html.escape(owner) + (" (you)" if is_you else "")
        color = owner_colors.get(owner, "#9ca3af")
        body += (
            f'<div class="{cls}"><span class="nm">{name}</span>'
            f'<div class="gs-track"><div class="gs-fill" '
            f'style="width:{v / top * 100:.1f}%;background:{color};"></div></div>'
            f'<span class="v">{v:,.0f}</span></div>'
        )
    insight = ""
    if viewer_rank is not None:
        mine = float(valued[viewer_rank - 1].get("total_value") or 0)
        if viewer_rank == 1:
            insight = f"You have the most valuable roster in the league ({mine:,.0f})."
        else:
            gap = top - mine
            leader = html.escape(str(valued[0].get("owner")))
            insight = (
                f"You rank {_ordinal(viewer_rank)} of {len(valued)} in roster value, "
                f"{gap:,.0f} behind {leader}."
            )
    return _card(
        "Roster Value by Team",
        "Total dynasty value on each roster, highest first.",
        body,
        insight,
    )


def _positional_value_card(value_ctx: dict, viewer_owner: str, owner_colors: dict) -> str:
    """Positional Value Breakdown: stacked bars showing where each team's
    dynasty value lives (QB/RB/WR/TE), highest-value roster first."""
    try:
        from dashboard_services.ai.context_builders import (
            build_model_value_lookup, ctx_scoring_type, summarize_roster_players,
            _ctx_is_sf, _safe_float,
        )
    except Exception:
        return ""
    try:
        rosters = value_ctx.get("rosters") or []
        if not rosters:
            return ""
        roster_map = value_ctx.get("roster_map") or {}
        players_index = value_ctx.get("players_index") or {}
        players_map = value_ctx.get("players_map") or {}
        lookup = build_model_value_lookup(
            value_ctx.get("model_value_table") or [],
            is_sf=_ctx_is_sf(value_ctx),
            scoring_type=ctx_scoring_type(value_ctx),
        )
        POS_ORDER = ["QB", "RB", "WR", "TE"]
        POS_COLORS = {"QB": "#2a78d6", "RB": "#1baf7a", "WR": "#eda100", "TE": "#e87ba4"}
        team_pos = []
        for r in rosters:
            rid = str(r.get("roster_id"))
            owner = roster_map.get(rid) or f"Roster {rid}"
            players = summarize_roster_players(r, players_index, players_map, lookup)
            buckets = {p: 0.0 for p in POS_ORDER}
            other = 0.0
            for pl in players:
                pos = str(pl.get("position") or "?").upper()
                v = _safe_float(pl.get("value"))
                if pos in buckets:
                    buckets[pos] += v
                else:
                    other += v
            total = sum(buckets.values()) + other
            if total <= 0:
                continue
            team_pos.append((owner, buckets, other, total))
        if len(team_pos) < 2:
            return ""
        team_pos.sort(key=lambda t: -t[3])
        top = team_pos[0][3]
        body = ""
        viewer_weak = ""
        for owner, buckets, other, total in team_pos:
            is_you = viewer_owner and owner == viewer_owner
            cls = "pos-row you" if is_you else "pos-row"
            name = html.escape(owner) + (" (you)" if is_you else "")
            segs = "".join(
                f"<span class='pos-seg' style='width:{buckets[p] / total * 100:.1f}%;"
                f"background:{POS_COLORS[p]};' title='{p}: {buckets[p]:,.0f}'></span>"
                for p in POS_ORDER
            )
            body += (
                f'<div class="{cls}"><span class="nm">{name}</span>'
                f'<div class="pos-track">{segs}</div>'
                f'<span class="v">{total:,.0f}</span></div>'
            )
            if is_you:
                weakest = min(POS_ORDER, key=lambda p: buckets[p])
                strongest = max(POS_ORDER, key=lambda p: buckets[p])
                viewer_weak = (
                    f"Your value is concentrated at {strongest} ({buckets[strongest]:,.0f}); "
                    f"{weakest} is your thinnest spot ({buckets[weakest]:,.0f})."
                )
        legend = "".join(
            f"<span><span class='gs-dot' style='background:{POS_COLORS[p]}'></span>{p}</span>"
            for p in POS_ORDER
        )
        return _card(
            "Positional Value Breakdown",
            "Where each roster's dynasty value lives, by position.",
            body + f"<div class='pos-legend'>{legend}</div>",
            viewer_weak,
        )
    except Exception:
        return ""


def _value_pane(value_ctx: dict, viewer_owner: str, owner_colors: dict) -> str:
    """The Value tab: value-vs-age scatter plus roster-value bars. Dynasty-only;
    redraft leagues get an honest empty state instead of empty charts."""
    try:
        from dashboard_services.ai.context_builders import ctx_scoring_type, team_value_age_rows
        if ctx_scoring_type(value_ctx) == "redraft":
            return _empty_card(
                "Dynasty value charts are for dynasty and keeper leagues. "
                "This league is redraft."
            )
        rows = team_value_age_rows(value_ctx)
    except Exception:
        return _empty_card("Roster value data is unavailable right now.")
    cards = _value_age_card(rows, viewer_owner, owner_colors)
    cards += _roster_value_card(rows, viewer_owner, owner_colors)
    cards += _positional_value_card(value_ctx, viewer_owner, owner_colors)
    if not cards:
        return _empty_card("No roster value data available.")
    return cards


# ── Trends tab ───────────────────────────────────────────────────────────────

def _trend_card(df_weekly_finalized, viewer_owner: str, owner_colors: dict, figs: dict) -> str:
    """Weekly Scoring Trend: the viewer's line (thick, team color) against the
    league average (dashed). Falls back to all teams when the viewer is unknown."""
    try:
        weeks = sorted(int(w) for w in df_weekly_finalized["week"].unique())
        if not weeks:
            return ""
        wk_avg = df_weekly_finalized.groupby("week")["points"].mean()
        avg_line = [float(wk_avg[w]) for w in weeks]
        traces = []
        me = None
        owners = {str(o) for o in df_weekly_finalized["owner"].astype(str).tolist()}
        if viewer_owner and viewer_owner in owners:
            g = df_weekly_finalized[
                df_weekly_finalized["owner"].astype(str) == viewer_owner
            ].sort_values("week")
            me = [float(v) for v in g["points"].tolist()]
            me_weeks = [int(w) for w in g["week"].tolist()]
            traces.append(
                go.Scatter(
                    x=me_weeks, y=me,
                    mode="lines+markers",
                    name=viewer_owner,
                    line=dict(color=owner_colors.get(viewer_owner, _YOU_BLUE), width=3.5),
                    marker=dict(size=7),
                    showlegend=False,
                    hovertemplate="Wk %{x}: %{y:.1f}<extra></extra>",
                )
            )
        else:
            for owner, gg in df_weekly_finalized.sort_values("week").groupby("owner"):
                gg = gg.sort_values("week")
                traces.append(
                    go.Scatter(
                        x=[int(w) for w in gg["week"].tolist()],
                        y=[float(v) for v in gg["points"].tolist()],
                        mode="lines",
                        name=str(owner),
                        line=dict(color=owner_colors.get(str(owner)), width=1.5),
                        opacity=0.55,
                        showlegend=False,
                    )
                )
        traces.append(
            go.Scatter(
                x=weeks, y=avg_line,
                mode="lines",
                name="League avg",
                line=dict(dash="dash", width=2, color=_LEAGUE_GRAY),
                showlegend=False,
                hovertemplate="Wk %{x}: %{y:.1f} avg<extra></extra>",
            )
        )
        fig = go.Figure(traces)
        fig.update_layout(
            xaxis_title=dict(text="Week", standoff=12),
            xaxis=dict(dtick=1),
            yaxis_title=dict(text="Points"),
            hovermode="x unified",
            margin=dict(l=44, r=16, t=10, b=45),
            showlegend=False,
        )
        apply_brand_layout(fig)
        figs["chart-trend"] = _fig_json(fig)

        insight = ""
        if me:
            parts = []
            if len(me) >= 3:
                last3 = me[-3:]
                direction = (
                    "up" if last3[-1] > last3[0]
                    else "down" if last3[-1] < last3[0]
                    else "flat"
                )
                parts.append(
                    "Your scoring is trending "
                    + direction
                    + ": "
                    + " → ".join(f"{v:.0f}" for v in last3)
                    + " over the last 3 weeks."
                )
            vavg = sum(me) / len(me)
            lavg = float(np.mean(avg_line))
            diff = vavg - lavg
            parts.append(
                f"You're averaging {abs(diff):.1f} points "
                f"{'above' if diff >= 0 else 'below'} the league."
            )
            insight = " ".join(parts)
        else:
            avgs = df_weekly_finalized.groupby("owner")["points"].mean().sort_values(ascending=False)
            if not avgs.empty:
                insight = (
                    f"{html.escape(str(avgs.index[0]))} leads the league "
                    f"at {float(avgs.iloc[0]):.1f} points per week."
                )
        you_color = owner_colors.get(viewer_owner, _YOU_BLUE) if me else _LEAGUE_GRAY
        legend = _legend([(you_color, "You"), (_LEAGUE_GRAY, "League avg")])
        return _card(
            "Weekly Scoring Trend",
            "Points for by week: your team against the league average.",
            '<div id="chart-trend" style="width:100%;min-height:350px;"></div>' + legend,
            insight,
        )
    except Exception:
        return ""


def _sos_fig(df_weekly_finalized, owner_colors: dict):
    """Strength-of-schedule bar chart: average opponent score faced per team.
    Fallback for the SOS card when the viewer can't be identified. Returns a
    Plotly figure, or None when points_against is unavailable."""
    try:
        if "points_against" not in getattr(df_weekly_finalized, "columns", []):
            return None
        sub = df_weekly_finalized.dropna(subset=["points_against"])
        if sub.empty:
            return None
        avg = sub.groupby("owner")["points_against"].mean().sort_values(ascending=True)
        owners = [str(o) for o in avg.index.tolist()]
        vals = [float(v) for v in avg.values.tolist()]
        if not owners:
            return None
        league_avg = float(np.mean(vals))
        fig = go.Figure(
            go.Bar(
                x=vals,
                y=owners,
                orientation="h",
                marker=dict(color=[owner_colors.get(o, "#9ca3af") for o in owners]),
                showlegend=False,
                hovertemplate="%{y}: %{x:.1f} opp avg<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title=dict(text="Avg opponent score", standoff=12),
            yaxis=dict(automargin=True),
            margin=dict(l=140, r=20, t=10, b=45),
            showlegend=False,
            shapes=[dict(
                type="line",
                x0=league_avg, x1=league_avg,
                y0=-0.6, y1=len(owners) - 0.4,
                line=dict(dash="dash", color="#9ca3af"),
            )],
        )
        return fig
    except Exception:
        return None


def _sos_card(df_weekly_finalized, viewer_owner: str, owner_colors: dict, figs: dict) -> str:
    """Strength of Schedule: the viewer's opponent points by week, with weeks
    above the league average highlighted. Falls back to the per-team average
    chart when the viewer can't be identified."""
    try:
        if "points_against" not in getattr(df_weekly_finalized, "columns", []):
            return ""
        owners = {str(o) for o in df_weekly_finalized["owner"].astype(str).tolist()}
        if viewer_owner and viewer_owner in owners:
            g = df_weekly_finalized[
                df_weekly_finalized["owner"].astype(str) == viewer_owner
            ].sort_values("week")
            weeks = [int(w) for w in g["week"].tolist()]
            opp = [float(v) for v in g["points_against"].tolist()]
            if not weeks:
                return ""
            wk_avg = df_weekly_finalized.groupby("week")["points_against"].mean()
            league_avg = float(wk_avg.mean())
            colors = [
                _ABOVE_AMBER if v > float(wk_avg[w]) else _BELOW_BLUE
                for v, w in zip(opp, weeks)
            ]
            fig = go.Figure(
                go.Bar(
                    x=weeks, y=opp,
                    marker=dict(color=colors),
                    showlegend=False,
                    hovertemplate="Wk %{x}: %{y:.1f} opp pts<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title=dict(text="Week", standoff=12),
                xaxis=dict(dtick=1),
                yaxis_title=dict(text="Opponent points"),
                margin=dict(l=44, r=16, t=10, b=45),
                showlegend=False,
                shapes=[dict(
                    type="line",
                    x0=min(weeks) - 0.6, x1=max(weeks) + 0.6,
                    y0=league_avg, y1=league_avg,
                    line=dict(dash="dash", color=_LEAGUE_GRAY),
                )],
            )
            apply_brand_layout(fig)
            figs["chart-sos"] = _fig_json(fig)
            vavg = sum(opp) / len(opp)
            insight = (
                f"Opponents have averaged {vavg:.1f} against you vs "
                f"{league_avg:.1f} league-wide. You've faced a "
                f"{'tougher' if vavg > league_avg else 'softer'} than average schedule."
            )
            legend = _legend([(_ABOVE_AMBER, "Above avg week"), (_BELOW_BLUE, "Below avg week")])
            return _card(
                "Strength of Schedule",
                "Opponent points faced by week. Weeks you faced above-average opponents are highlighted.",
                '<div id="chart-sos" style="width:100%;min-height:350px;"></div>' + legend,
                insight,
            )
        # Fallback: per-team average opponent score.
        fig = _sos_fig(df_weekly_finalized, owner_colors)
        if fig is None:
            return ""
        apply_brand_layout(fig)
        figs["chart-sos"] = _fig_json(fig)
        avgs = (
            df_weekly_finalized.dropna(subset=["points_against"])
            .groupby("owner")["points_against"].mean().sort_values(ascending=False)
        )
        insight = ""
        if not avgs.empty:
            insight = (
                f"{html.escape(str(avgs.index[0]))} has faced the toughest schedule "
                f"({float(avgs.iloc[0]):.1f} avg opponent score)."
            )
        return _card(
            "Strength of Schedule",
            "Average opponent score each team has faced. Dashed line is the league average.",
            '<div id="chart-sos" style="width:100%;min-height:350px;"></div>',
            insight,
        )
    except Exception:
        return ""


def _bump_card(df_weekly, owner_colors: dict, viewer_owner: str, figs: dict) -> str:
    """Standings bump chart: every team's rank by week, one line each. Rank 1
    sits at the top. Built on utils.standings.seed_series_for (cumulative wins,
    then PF)."""
    try:
        from utils.standings import seed_series_for
    except Exception:
        return ""
    try:
        owners = sorted({str(o) for o in df_weekly["owner"].astype(str).tolist()})
        if len(owners) < 2:
            return ""
        weeks = sorted(int(w) for w in df_weekly["week"].unique())
        if not weeks:
            return ""
        traces = []
        for owner in owners:
            series = seed_series_for(df_weekly, owner)
            if not series:
                continue
            xs = [w for w, _ in series]
            ys = [r for _, r in series]
            is_you = viewer_owner and owner == viewer_owner
            traces.append(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    name=owner + (" (you)" if is_you else ""),
                    line=dict(
                        color=owner_colors.get(owner, "#9ca3af"),
                        width=3.5 if is_you else 1.8,
                    ),
                    marker=dict(size=7 if is_you else 5),
                    opacity=1.0 if is_you else 0.7,
                    showlegend=False,
                    hovertemplate=f"{html.escape(owner)}<br>Wk %{{x}}: rank %{{y}}<extra></extra>",
                )
            )
        fig = go.Figure(traces)
        fig.update_layout(
            xaxis_title=dict(text="Week", standoff=12),
            xaxis=dict(dtick=1),
            yaxis_title=dict(text="Standings rank"),
            yaxis=dict(autorange="reversed", dtick=1),
            hovermode="x unified",
            margin=dict(l=44, r=16, t=10, b=45),
            showlegend=False,
        )
        apply_brand_layout(fig)
        figs["chart-bump"] = _fig_json(fig)

        insight = ""
        if viewer_owner:
            series = seed_series_for(df_weekly, viewer_owner)
            if len(series) >= 2:
                first, last = series[0][1], series[-1][1]
                move = first - last
                if move >= 3:
                    insight = f"You're climbing: {_ordinal(first)} after week 1 to {_ordinal(last)} now."
                elif move <= -3:
                    insight = f"You're sliding: {_ordinal(first)} after week 1 down to {_ordinal(last)} now."
                elif last == 1:
                    insight = "You've held 1st place."
                else:
                    insight = f"You're {_ordinal(last)} (started {_ordinal(first)})."
        legend_items = [
            (owner_colors.get(o, "#9ca3af"), o + (" (you)" if viewer_owner and o == viewer_owner else ""))
            for o in owners
        ]
        return _card(
            "Standings Bump Chart",
            "Every team's rank by week. Rising lines are climbing; falling lines are collapsing.",
            '<div id="chart-bump" style="width:100%;min-height:380px;"></div>' + _legend(legend_items),
            insight,
        )
    except Exception:
        return ""


def _all_teams_trend_card(df_weekly, owner_colors: dict, figs: dict) -> str:
    """Weekly Scoring Trend (all teams): every team's line plus the league
    average, for spotting who's hot."""
    try:
        weeks = sorted(int(w) for w in df_weekly["week"].unique())
        if not weeks:
            return ""
        wk_avg = df_weekly.groupby("week")["points"].mean()
        avg_line = [float(wk_avg[w]) for w in weeks]
        traces = [
            go.Scatter(
                x=weeks, y=avg_line,
                mode="lines",
                name="League avg",
                line=dict(dash="dash", width=2.5, color=_LEAGUE_GRAY),
                showlegend=False,
                hovertemplate="Wk %{x}: %{y:.1f} avg<extra></extra>",
            )
        ]
        for owner, gg in df_weekly.sort_values("week").groupby("owner"):
            gg = gg.sort_values("week")
            traces.append(
                go.Scatter(
                    x=[int(w) for w in gg["week"].tolist()],
                    y=[float(v) for v in gg["points"].tolist()],
                    mode="lines+markers",
                    name=str(owner),
                    line=dict(color=owner_colors.get(str(owner), "#9ca3af"), width=1.8),
                    marker=dict(size=5),
                    opacity=0.75,
                    showlegend=False,
                    hovertemplate=f"{html.escape(str(owner))}<br>Wk %{{x}}: %{{y:.1f}}<extra></extra>",
                )
            )
        fig = go.Figure(traces)
        fig.update_layout(
            xaxis_title=dict(text="Week", standoff=12),
            xaxis=dict(dtick=1),
            yaxis_title=dict(text="Points"),
            hovermode="x unified",
            margin=dict(l=44, r=16, t=10, b=45),
            showlegend=False,
        )
        apply_brand_layout(fig)
        figs["chart-allteams"] = _fig_json(fig)

        avgs = df_weekly.groupby("owner")["points"].mean().sort_values(ascending=False)
        insight = ""
        if not avgs.empty:
            top, top_v = str(avgs.index[0]), float(avgs.iloc[0])
            bot, bot_v = str(avgs.index[-1]), float(avgs.iloc[-1])
            insight = (
                f"{html.escape(top)} leads at {top_v:.1f} per week; "
                f"{html.escape(bot)} trails at {bot_v:.1f}."
            )
        owners = [str(o) for o in avgs.index.tolist()]
        legend = _legend([(owner_colors.get(o, "#9ca3af"), o) for o in owners])
        return _card(
            "Weekly Scoring: All Teams",
            "Every team's weekly score. Find who's heating up and who's fading.",
            '<div id="chart-allteams" style="width:100%;min-height:380px;"></div>' + legend,
            insight,
        )
    except Exception:
        return ""


# ── Season panes ─────────────────────────────────────────────────────────────

def _finalized_weekly(df_weekly):
    if df_weekly is None or "finalized" not in getattr(df_weekly, "columns", []):
        return None
    sub = df_weekly[df_weekly["finalized"] == True].copy()
    return sub if not sub.empty else None


def _season_panes(perf_ctx: dict, value_ctx: dict, viewer_owner: str, figs: dict) -> Dict[str, tuple]:
    """The Performance / Value / Trends panes for one season. ``perf_ctx``
    carries the weekly results (luck, consistency, trends); ``value_ctx``
    carries the rosters used for the dynasty value cards (usually the same ctx,
    but the career view passes current rosters for value)."""
    team_stats = perf_ctx.get("team_stats") if perf_ctx else None
    df_weekly = _finalized_weekly(perf_ctx.get("df_weekly") if perf_ctx else None)
    if team_stats is None or getattr(team_stats, "empty", True) or df_weekly is None:
        empty = (
            "<div class='card central'><div class='card-body'>"
            "<p style='color:var(--text-muted);'>No weekly data available for this season.</p>"
            "</div></div>"
        )
        return {
            "perf": ("Performance", empty),
            "value": ("Value", empty),
            "trends": ("Trends", empty),
        }

    owners = team_stats["owner"].tolist()
    owner_colors = owner_color_map(owners)

    perf_html = _luck_card(df_weekly, viewer_owner, owner_colors)
    box_detail = _boxplot_detail_html(df_weekly, owner_colors, figs)
    perf_html += _consistency_card(team_stats, owner_colors, viewer_owner, box_detail)
    perf_html += _pf_pa_card(team_stats, owner_colors, viewer_owner, figs)
    perf_html += _margin_card(df_weekly, owner_colors, viewer_owner)
    perf_html += _h2h_card(df_weekly, owner_colors, viewer_owner)
    perf_html += _radar_card(team_stats, owner_colors, figs)
    if not perf_html:
        perf_html = _empty_card("Not enough weekly data for performance charts.")

    v_owners = []
    vts = (value_ctx or {}).get("team_stats")
    if vts is not None and not getattr(vts, "empty", True) and "owner" in vts.columns:
        v_owners = vts["owner"].tolist()
    value_html = _value_pane(value_ctx or {}, viewer_owner, owner_color_map(v_owners or owners))

    trends_html = _trend_card(df_weekly, viewer_owner, owner_colors, figs)
    trends_html += _all_teams_trend_card(df_weekly, owner_colors, figs)
    trends_html += _bump_card(df_weekly, owner_colors, viewer_owner, figs)
    trends_html += _sos_card(df_weekly, viewer_owner, owner_colors, figs)
    if not trends_html:
        trends_html = _empty_card("Not enough weekly data for trend charts.")

    return {
        "perf": ("Performance", perf_html),
        "value": ("Value", value_html),
        "trends": ("Trends", trends_html),
    }


def build_graphs_body(ctx: dict, *, tab: str = "perf", career_url: str = "",
                       season_label: str = "") -> str:
    """Seasonal League Stats body: header, the four tabs, and the tab panes.

    ``tab`` selects the initially active tab ("perf" | "value" | "trends").
    ``career_url`` feeds the Career pane's call-to-action. ``season_label``
    (e.g. "2026 season") goes under the page title next to the league name.
    """
    team_stats = ctx["team_stats"]
    df_weekly = _finalized_weekly(ctx["df_weekly"])
    if (
        team_stats is None or getattr(team_stats, "empty", True)
        or df_weekly is None
    ):
        return """
            <div class="card central graphs-empty">
              <div class="card-body">
                <p style="color:var(--text-muted);">
                  No weekly data available for this season.
                </p>
              </div>
            </div>"""

    viewer_owner = _viewer_owner(ctx)
    figs: Dict[str, str] = {}
    panes = _season_panes(ctx, ctx, viewer_owner, figs)

    career_cta = (
        "<p style='color:var(--text-muted);margin:0;'>"
        "Career stats aggregate every completed season: your franchise win% "
        "trajectory and points by season.</p>"
    )
    if career_url:
        career_cta += f'<a class="gs-btn" href="{html.escape(career_url)}">View career stats</a>'
    panes["career"] = ("Career", _card("Career", "", career_cta))

    active = tab if tab in panes else "perf"
    return (
        _graphs_style()
        + _page_header(_league_name(ctx), season_label)
        + _tab_shell(active, panes)
        + _deferred_plotly_js(figs)
        + _tabs_js()
    )


# ── Career tab ───────────────────────────────────────────────────────────────

def _winpct_streak_line(seasons: List[int], pcts: List[float]) -> str:
    """Plain-English read of a franchise win% trajectory, from real values."""
    n = len(pcts)
    if n == 0:
        return ""
    if n == 1:
        return f"One season on record: {pcts[0]:.0%} in {seasons[0]}."
    # Consecutive season-over-season climbs ending at the latest season.
    up = 0
    i = n - 1
    while i > 0 and pcts[i] > pcts[i - 1]:
        up += 1
        i -= 1
    if up >= 2:
        return f"Your franchise win% has climbed {up} straight seasons."
    if pcts[-1] < pcts[-2]:
        k = 0
        j = n - 2
        while j > 0 and pcts[j] > pcts[j - 1]:
            k += 1
            j -= 1
        if k >= 1:
            return (
                f"Your win% dipped in {seasons[-1]} after climbing "
                f"{k} straight season{'s' if k > 1 else ''}."
            )
        return f"Your win% dipped in {seasons[-1]} to {pcts[-1]:.0%}."
    if pcts[-1] == pcts[-2]:
        return f"Your win% held steady in {seasons[-1]} at {pcts[-1]:.0%}."
    if up == 1:
        return f"Your win% rose in {seasons[-1]} to {pcts[-1]:.0%}."
    best_i = max(range(n), key=lambda i: pcts[i])
    return (
        f"Your win% has been up and down. {seasons[best_i]} was your best "
        f"at {pcts[best_i]:.0%}."
    )


def _career_winpct_card(season_record_df, viewer_owner: str, owner_colors: dict, figs: dict) -> str:
    """Career Win % by Season: the viewer's franchise trajectory line."""
    if season_record_df is None or getattr(season_record_df, "empty", True):
        return _empty_card("No career season records available.")
    if not viewer_owner:
        return _empty_card("Sign in to see your franchise trajectory.")
    sub = season_record_df[
        season_record_df["owner"].astype(str) == viewer_owner
    ].copy()
    if sub.empty:
        return _empty_card("Your team wasn't found in the career records.")
    try:
        sub["season"] = sub["season"].astype(int)
    except Exception:
        return _empty_card("Your team wasn't found in the career records.")
    sub = sub.sort_values("season")
    seasons = sub["season"].tolist()
    pcts = []
    for _, r in sub.iterrows():
        w = float(r.get("wins", 0) or 0)
        l = float(r.get("losses", 0) or 0)
        t = float(r.get("ties", 0) or 0)
        g = w + l + t
        pcts.append((w + 0.5 * t) / g if g else 0.0)
    try:
        fig = go.Figure(
            go.Scatter(
                x=seasons,
                y=[p * 100 for p in pcts],
                mode="lines+markers",
                name=viewer_owner,
                line=dict(color=owner_colors.get(viewer_owner, _YOU_BLUE), width=3),
                marker=dict(size=8),
                showlegend=False,
                hovertemplate="%{x}: %{y:.0f}%<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title=dict(text="Season", standoff=12),
            xaxis=dict(dtick=1),
            yaxis_title=dict(text="Win %"),
            margin=dict(l=44, r=16, t=10, b=45),
            showlegend=False,
        )
        apply_brand_layout(fig)
        figs["chart-career-wpct"] = _fig_json(fig)
        chart = '<div id="chart-career-wpct" style="width:100%;min-height:350px;"></div>'
    except Exception:
        chart = ""
    return _card(
        "Career Win % by Season",
        "Your all-time franchise trajectory.",
        chart,
        _winpct_streak_line(seasons, pcts),
    )


def _career_pf_card(season_pf_df, viewer_owner: str, owner_colors: dict) -> str:
    """Career Points For: the viewer's total points by season, as bars."""
    if season_pf_df is None or getattr(season_pf_df, "empty", True):
        return _empty_card("No career scoring data available.")
    if not viewer_owner:
        return _empty_card("Sign in to see your franchise scoring history.")
    sub = season_pf_df[
        season_pf_df["owner"].astype(str) == viewer_owner
    ].copy()
    if sub.empty:
        return _empty_card("Your team wasn't found in the career records.")
    try:
        sub["season"] = sub["season"].astype(int)
    except Exception:
        return _empty_card("Your team wasn't found in the career records.")
    sub = sub.sort_values("season")
    top = float(sub["pf"].max()) or 1.0
    color = owner_colors.get(viewer_owner, _YOU_BLUE)
    body = ""
    for _, r in sub.iterrows():
        v = float(r.get("pf") or 0)
        body += (
            f'<div class="gs-bar-row"><span class="nm">{int(r["season"])}</span>'
            f'<div class="gs-track"><div class="gs-fill" '
            f'style="width:{v / top * 100:.1f}%;background:{color};"></div></div>'
            f'<span class="v">{v:,.0f}</span></div>'
        )
    best_idx = sub["pf"].astype(float).idxmax()
    best_season = int(sub.loc[best_idx, "season"])
    best_pf = float(sub.loc[best_idx, "pf"])
    insight = (
        f"{best_season} was your highest-scoring season ever ({best_pf:,.0f} points)."
    )
    return _card(
        "Career Points For",
        "Your total points by season.",
        body,
        insight,
    )


def build_career_graphs_body(career_ctx: dict, *, season_ctx: dict = None,
                             value_ctx: dict = None, viewer_owner: str = "",
                             league_name: str = "") -> str:
    """Career League Stats body: the four-tab shell with the Career tab active.

    ``season_ctx`` feeds the Performance/Value/Trends panes (usually the latest
    completed season); ``value_ctx`` feeds the dynasty value cards (usually the
    current rosters). Either may be None, in which case those panes show an
    empty state.
    """
    team_stats = career_ctx.get("team_stats", pd.DataFrame())
    if team_stats.empty:
        return "<div class='card central'><div class='card-body'><p>No career data available.</p></div></div>"

    owners = team_stats["owner"].tolist()
    owner_colors = owner_color_map(owners)
    figs: Dict[str, str] = {}

    panes: Dict[str, tuple] = {}
    if season_ctx is not None:
        panes = _season_panes(season_ctx, value_ctx or {}, viewer_owner, figs)

    career_html = _career_winpct_card(
        career_ctx.get("season_record_df"), viewer_owner, owner_colors, figs
    )
    career_html += _career_pf_card(
        career_ctx.get("season_pf_df"), viewer_owner, owner_colors
    )
    panes["career"] = ("Career", career_html)

    return (
        _graphs_style()
        + _page_header(league_name, "Career (all seasons)")
        + _tab_shell("career", panes)
        + _deferred_plotly_js(figs)
        + _tabs_js()
    )


# ── Page orchestration (moved out of app.py) ──────────────────────────────────
# These build the full /graphs page body. They take every app-level input as a
# parameter (the current league context, the completed-season list, a context
# provider for other seasons, the live value table, and the prebuilt page URL)
# so this module never imports app.py -- the route wires the accessors in.

def build_tour_mock_graphs_ctx(df_weekly) -> dict:
    """Graphs ctx for the tour/demo, from a pre-built mock weekly frame."""
    from dashboard_services.pages.history_page import (
        build_regular_season_team_stats, sort_team_stats)
    df = df_weekly
    mock_league: dict = {"settings": {"playoff_week_start": 14}}
    team_stats = build_regular_season_team_stats(df, mock_league)
    team_stats = sort_team_stats(team_stats)

    if not team_stats.empty and "PF" in team_stats.columns:
        from dashboard_services.power_score import approximate_power_score_frame
        team_stats = approximate_power_score_frame(team_stats)
        # Z-score columns required by z_better_outward
        for col in ["PF", "PA", "MAX", "MIN", "AVG", "STD"]:
            zc = f"Z_{col}"
            if col in team_stats.columns and zc not in team_stats.columns:
                col_vals = team_stats[col]
                std_val = float(col_vals.std()) if len(col_vals) else 1.0
                team_stats[zc] = (col_vals - col_vals.mean()) / max(std_val, 1.0)

    return {"team_stats": team_stats, "df_weekly": df}


def build_career_graphs_ctx(
    platform, league_id, season, available_seasons, get_ctx, only_owners=None,
) -> dict:
    """Aggregate team_stats and df_weekly across seasons for career graphs.

    Includes the current in-progress season so career totals reflect this
    year's games too (same pattern as awards all-time standings). Completed
    seasons come from ``available_seasons``; the current season is appended
    when missing.

    ``get_ctx(platform, rid, season)`` fetches a league context (injected so this
    module stays independent of app.py). When ``only_owners`` is a non-empty set of
    owner names, the aggregate is restricted to those members (used by the current-
    vs-all-time toggle to drop owners no longer in the league)."""
    from dashboard_services.api import resolve_league_id_for_season
    from dashboard_services.pages.history_page import build_regular_season_team_stats

    from dashboard_services.historical_identity import canonicalize_weekly_owners, season_owner_index
    career: dict = {}  # stable owner id -> career totals
    season_pf_rows: list = []  # rows for per-season bar chart: {season, owner, pf}
    season_record_rows: list = []  # rows for career win% line: {season, owner, wins, losses, ties}
    season_frames: list = []
    labels: dict[str, tuple[int, str]] = {}

    # Include the current in-progress season so career graphs reflect this
    # year's games too. Current season goes last so its display names win
    # over historical ones.
    _seasons_to_process = list(available_seasons or [])
    if int(season) not in [int(s) for s in _seasons_to_process]:
        _seasons_to_process.append(int(season))

    for hist_s in _seasons_to_process:
        rid = resolve_league_id_for_season(platform, league_id, season, hist_s)
        try:
            hctx = get_ctx(platform, rid, hist_s)
        except Exception:
            continue

        df = canonicalize_weekly_owners(hctx.get("df_weekly", pd.DataFrame()), hctx)
        if df.empty or "owner" not in df.columns:
            continue

        _, season_labels = season_owner_index(hctx)
        for owner_id, label in season_labels.items():
            if owner_id not in labels or int(hist_s) > labels[owner_id][0]:
                labels[owner_id] = (int(hist_s), label)

        mock_lg = hctx.get("league") or {}
        stats_df = df.copy()
        stats_df["owner"] = stats_df["owner_key"]
        ts = build_regular_season_team_stats(stats_df, mock_lg)

        for _, row in ts.iterrows():
            owner_id = str(row.get("owner", "?"))
            owner = labels.get(owner_id, (0, owner_id))[1]
            if owner_id not in career:
                career[owner_id] = {
                    "Wins": 0, "Losses": 0, "Ties": 0,
                    "PF": 0.0, "PA": 0.0, "weekly_pts": [],
                }
            wins = int(row.get("Wins", 0))
            losses = int(row.get("Losses", 0))
            ties = int(row.get("Ties", 0))
            career[owner_id]["Wins"] += wins
            career[owner_id]["Losses"] += losses
            career[owner_id]["Ties"] += ties
            career[owner_id]["PF"] += float(row.get("PF", 0))
            career[owner_id]["PA"] += float(row.get("PA", 0))
            season_pf_rows.append({"season": hist_s, "owner_key": owner_id, "owner": owner, "pf": float(row.get("PF", 0))})
            season_record_rows.append({
                "season": hist_s, "owner_key": owner_id, "owner": owner,
                "wins": wins, "losses": losses, "ties": ties,
            })

        sub = df[df["finalized"] == True] if "finalized" in df.columns else df
        for owner_id, grp in sub.groupby("owner_key"):
            career.setdefault(str(owner_id), {
                "Wins": 0, "Losses": 0, "Ties": 0, "PF": 0.0, "PA": 0.0, "weekly_pts": [],
            })["weekly_pts"].extend(grp["points"].tolist() if "points" in grp else [])
        df["season"] = hist_s
        season_frames.append(df)

    # Build career team_stats DataFrame
    stat_rows = []
    for owner_id, d in career.items():
        owner = labels.get(owner_id, (0, owner_id))[1]
        pts = d["weekly_pts"]
        games = d["Wins"] + d["Losses"] + d["Ties"]
        stat_rows.append({
            "owner": owner,
            "owner_key": owner_id,
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
        from dashboard_services.power_score import approximate_power_score_frame
        team_stats = approximate_power_score_frame(team_stats)
        for col in ["PF", "PA", "MAX", "MIN", "AVG", "STD"]:
            zc = f"Z_{col}"
            if col in team_stats.columns and zc not in team_stats.columns:
                cv = team_stats[col]
                sd = max(float(cv.std()) if len(cv) else 1.0, 1.0)
                team_stats[zc] = (cv - cv.mean()) / sd

    # Combined df_weekly (with season column) for box/line charts
    # Relabel every season with the newest known team name while retaining the
    # stable owner key used by filters and aggregation.
    for frame in season_frames:
        frame["owner"] = frame["owner_key"].map(lambda key: labels.get(str(key), (0, str(key)))[1])
    df_combined = pd.concat(season_frames, ignore_index=True) if season_frames else pd.DataFrame()
    season_pf_df = pd.DataFrame(season_pf_rows) if season_pf_rows else pd.DataFrame()
    if not season_pf_df.empty:
        season_pf_df["owner"] = season_pf_df["owner_key"].map(
            lambda key: labels.get(str(key), (0, str(key)))[1]
        )
    season_record_df = pd.DataFrame(season_record_rows) if season_record_rows else pd.DataFrame()
    if not season_record_df.empty:
        season_record_df["owner"] = season_record_df["owner_key"].map(
            lambda key: labels.get(str(key), (0, str(key)))[1]
        )

    # Restrict to current members when the toggle asks for it.
    if only_owners:
        _keep = {str(o) for o in only_owners}
        key_col = "owner_key"
        if not team_stats.empty:
            team_stats = team_stats[team_stats[key_col].astype(str).isin(_keep)].reset_index(drop=True)
        if not df_combined.empty:
            df_combined = df_combined[df_combined[key_col].astype(str).isin(_keep)].reset_index(drop=True)
        if not season_pf_df.empty:
            season_pf_df = season_pf_df[season_pf_df[key_col].astype(str).isin(_keep)].reset_index(drop=True)
        if not season_record_df.empty:
            season_record_df = season_record_df[season_record_df[key_col].astype(str).isin(_keep)].reset_index(drop=True)

    return {
        "team_stats": team_stats,
        "df_weekly": df_combined,
        "season_pf_df": season_pf_df,
        "season_record_df": season_record_df,
        "is_career": True,
    }


def render_graphs_html(
    platform, season, league_id, view, members, *,
    ctx, available_seasons, get_ctx, model_value_table, graphs_base_url,
    tab: str = "perf",
) -> str:
    """Build the /graphs page body for a given view ("career" or a season),
    member filter, and initially-active tab. Every app-level input is injected:

      ctx                current league context
      available_seasons  completed seasons with data (newest-first)
      get_ctx            callable (platform, rid, season) -> league context
      model_value_table  live dynasty value rows (for the value/age scatter)
      graphs_base_url    the /graphs URL for this league (selector links)
      tab                initially active tab ("perf" | "value" | "trends")

    Pure of request args (view/members/tab are passed in) so the heavy career
    view - which aggregates every past season - can build in a background thread.
    """
    import logging
    from dashboard_services.api import resolve_league_id_for_season
    logger = logging.getLogger(__name__)

    offseason = bool(ctx.get("offseason_mode"))

    # Current members are stable provider owner ids, matching the career keys.
    current_owners = set()
    _rmap = ctx.get("roster_map") or {}
    if ctx.get("rosters"):
        current_owners = {str(r.get("owner_id")) for r in ctx["rosters"] if r.get("owner_id") is not None}
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
        _career_seasons = list(available_seasons or [])
        if int(season) not in [int(s) for s in _career_seasons]:
            _career_seasons.append(int(season))
        if not _career_seasons:
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
                # Seasonal panes come from the latest completed season; the
                # value cards use the current rosters with the live value table.
                latest_ctx = None
                try:
                    _latest = max(available_seasons)
                    _lrid = resolve_league_id_for_season(platform, league_id, season, _latest)
                    latest_ctx = get_ctx(platform, _lrid, _latest)
                except Exception:
                    logger.debug("career graphs latest-season ctx failed", exc_info=True)
                _val_ctx = {**ctx, "model_value_table": (model_value_table or ctx.get("model_value_table") or [])}
                charts_html = build_career_graphs_body(
                    career_ctx,
                    season_ctx=latest_ctx,
                    value_ctx=_val_ctx,
                    viewer_owner=_viewer_owner(ctx),
                    league_name=_league_name(ctx),
                )
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
            charts_html = build_graphs_body(
                season_ctx,
                tab=tab,
                career_url=f"{graphs_base_url}?view=career",
                season_label=f"{target_season} season",
            )

    return season_selector_html + charts_html
