"""Public NFL Teams rankings page.

A league-free research page: every NFL team ranked on honest per-game
offense numbers (see utils/team_offense_ranks.py), with a lazy-loaded team
profile (environment, O-line, depth chart, schedule) after a team is picked.

Data arrives from /api/nfl-team-rankings (one league-wide request) and
/api/nfl-team-details (per-team, on selection). All numbers are real measured
values or labeled projections: true per-game denominators, competition
ranking (1, 2, 2, 4), legitimate zeroes ranked, N/A only when truly missing.
There is no overall offense score; O-line ratings carry their own season
label because only the prior season is available in-season.
"""
from __future__ import annotations

from html import escape as _esc


VIEW_DEFS = (
    ("overview", "Overview"),
    ("passing", "Passing"),
    ("rushing", "Rushing"),
    ("oline", "Offensive Line"),
    ("defense", "Defense"),
)


def build_nfl_teams_body(
    season: int,
    team: str = "",
    view: str = "overview",
    available_seasons: "list | None" = None,
) -> str:
    """HTML shell for /nfl-teams. Table and profile render client-side."""
    season = int(season or 0)
    team = (_esc(str(team or "").upper()))
    view = str(view or "overview")
    if view not in {v for v, _ in VIEW_DEFS}:
        view = "overview"
    seasons = [int(s) for s in (available_seasons or [season]) if s]
    season_opts = "".join(
        f'<option value="{s}"{" selected" if s == season else ""}>{s}</option>'
        for s in seasons
    )
    tabs = "".join(
        f'<button type="button" role="tab" data-view="{v}" aria-selected="{str(v == view).lower()}">{label}</button>'
        for v, label in VIEW_DEFS
    )
    return f"""
<div class="nt-page" data-season="{season}" data-team="{team}" data-view="{view}">
<style>
.nt-page{{max-width:1060px;margin:0 auto;padding:18px 16px 28px;color:var(--text)}}
.nt-card{{margin-bottom:16px}}
.nt-cbody{{padding:0 16px 16px}}
.nt-chead{{padding:14px 16px 10px}}
.nt-chead h1,.nt-chead h2{{font-size:18px;margin:0}}
.nt-sub{{color:var(--text-muted);margin:4px 0 0;font-size:13px}}
.nt-controls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.nt-hbtn{{flex-shrink:0;display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border:1px solid var(--border);border-radius:8px;background:var(--card);color:var(--text-muted);font-size:12px;font-weight:600;cursor:pointer;white-space:nowrap;font-family:inherit}}
.nt-hbtn:hover{{background:var(--row)}}
.nt-seaslab{{font-size:12px;color:var(--text-muted)}}
.nt-csel select{{appearance:none;background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 30px 8px 12px;font-size:13px;min-height:38px;cursor:pointer;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23888' fill='none' stroke-width='1.5'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center}}
.nt-tabs{{display:flex;gap:8px;margin:12px 0 2px;flex-wrap:wrap}}
.nt-tabs button{{padding:7px 14px;border:1px solid var(--border);border-radius:999px;background:var(--card);color:var(--text);font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;transition:background .14s,border-color .14s;min-height:36px;font-family:inherit}}
.nt-tabs button:hover{{background:var(--row)}}
.nt-tabs button[aria-selected="true"]{{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}}
.nt-tcard{{overflow:hidden;margin-top:12px}}
.nt-tscroll{{overflow-x:auto}}
table.nt-rank{{border-collapse:separate;border-spacing:0;width:100%;min-width:680px}}
table.nt-rank thead th{{padding:8px 10px;border-bottom:1px solid var(--border);text-align:right;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--text-muted);white-space:nowrap}}
table.nt-rank thead th.nt-teamcol,table.nt-rank thead th.nt-rankcol{{text-align:left}}
table.nt-rank th .nt-thbtn{{background:none;border:0;color:inherit;font:inherit;padding:0;cursor:pointer;white-space:nowrap}}
table.nt-rank th .nt-thbtn:hover{{color:var(--text)}}
table.nt-rank th .nt-arr{{color:var(--text-muted);margin-left:4px}}
table.nt-rank td{{border-top:1px solid var(--border);padding:9px 10px;vertical-align:middle;font-size:14px}}
table.nt-rank tbody tr{{cursor:pointer}}
table.nt-rank tbody tr:hover td{{background:var(--row)}}
table.nt-rank tbody tr.nt-sel td{{background:var(--accent-soft)}}
.nt-rbadge{{display:inline-block;min-width:32px;text-align:center;font-size:11px;font-weight:800;padding:2px 6px;border-radius:999px;background:var(--row);color:var(--text-muted)}}
.nt-rbadge.nt-gold{{background:color-mix(in srgb,var(--gold) 16%,transparent);color:var(--gold)}}
.nt-rbadge.nt-tier-g{{background:color-mix(in srgb,var(--win) 15%,transparent);color:var(--win)}}
.nt-rbadge.nt-tier-b{{background:color-mix(in srgb,var(--loss) 15%,transparent);color:var(--loss)}}
.nt-tid{{display:flex;align-items:center;justify-content:space-between;gap:8px}}
.nt-tleft{{display:flex;align-items:center;gap:10px;min-width:0}}
.nt-logo{{width:28px;height:28px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;color:#fff;flex:none;overflow:hidden;background:var(--card-soft)}}
.nt-logo img{{width:100%;height:100%;object-fit:cover}}
.nt-logo.nt-lg{{width:44px;height:44px;font-size:13px}}
.nt-tid .nt-tn{{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.nt-tright{{display:flex;align-items:center;gap:8px;flex:none}}
.nt-gp{{font-size:11px;color:var(--text-muted);white-space:nowrap}}
table.nt-rank thead th.nt-teamcol{{position:sticky;left:0;z-index:3;background:var(--bg)}}
table.nt-rank tbody td.nt-teamcol{{position:sticky;left:0;z-index:1;background:var(--bg)}}
table.nt-rank tbody tr:first-child td{{border-top:0}}
table.nt-rank tbody tr:hover td.nt-teamcol{{background:var(--row)}}
table.nt-rank tbody tr.nt-sel td.nt-teamcol{{background:var(--accent-soft)}}
.nt-metric{{display:flex;align-items:center;gap:10px;justify-content:flex-end}}
.nt-mbar{{flex:1;min-width:64px;max-width:170px}}
.nt-mtrack{{position:relative;background:rgba(128,128,128,.18);border-radius:6px;height:10px;width:100%}}
.nt-mfill{{position:absolute;left:0;top:0;bottom:0;border-radius:6px}}
.nt-val{{font-weight:700;font-size:13px;white-space:nowrap;min-width:48px;text-align:right;font-variant-numeric:tabular-nums}}
.nt-val.nt-na{{color:var(--text-muted);font-weight:500}}
.nt-vwrap{{display:flex;flex-direction:column;align-items:flex-end;line-height:1.25}}
.nt-eff{{font-size:10px;color:var(--text-muted);white-space:nowrap;font-variant-numeric:tabular-nums}}
.nt-tnote{{margin:0;padding:10px 14px;font-size:12px;color:var(--text-muted);border-top:1px solid var(--border)}}
@media(max-width:600px){{.nt-mbar{{display:none}}.nt-metric{{gap:0}}.nt-val{{min-width:40px;font-size:12px}}table.nt-rank{{min-width:520px}}}}
.nt-pcard{{background:var(--card);border:1px solid var(--border);border-radius:12px;margin-bottom:16px;overflow:hidden}}
.nt-pcard.nt-empty{{padding:28px 20px;text-align:center;color:var(--text-muted)}}
.nt-pcard.nt-empty h2{{color:var(--text);margin:0 0 6px;font-size:17px}}
.nt-backrow{{margin:0 0 12px}}
.nt-back{{background:var(--card);border:1px solid var(--border);color:var(--text);border-radius:999px;padding:10px 18px;font-size:14px;font-weight:700;cursor:pointer;min-height:44px;font-family:inherit}}
.nt-back:active{{transform:scale(.98)}}
.nt-phead2{{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--border)}}
.nt-pid{{display:flex;align-items:center;gap:12px}}
.nt-pid h2{{margin:0;font-size:18px}}
.nt-pid .nt-meta{{margin:2px 0 0;font-size:12px;color:var(--text-muted)}}
.nt-herohead{{border-radius:12px 12px 0 0}}
.nt-record{{display:inline-block;font-size:12px;font-weight:800;background:var(--card-soft);border:1px solid var(--border);border-radius:6px;padding:2px 8px;margin-left:8px;vertical-align:2px;letter-spacing:.03em;white-space:nowrap}}
.nt-pbody{{padding:16px}}
.nt-pgrid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
@media(max-width:700px){{.nt-pgrid{{grid-template-columns:1fr}}}}
.nt-psec{{margin-bottom:16px}}
.nt-psec h3{{font-size:14px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted)}}
.nt-erow{{display:grid;grid-template-columns:150px 1fr 44px;gap:10px;align-items:center;padding:7px 0;border-top:1px solid var(--border)}}
.nt-erow:first-of-type{{border-top:0}}
.nt-elab{{font-size:13px}}
.nt-eval{{display:block;font-size:11px;color:var(--text-muted)}}
.nt-bar{{height:8px;background:rgba(128,128,128,.18);border-radius:4px;overflow:hidden}}
.nt-bar i{{display:block;height:100%;background:var(--accent);border-radius:4px}}
.nt-erk{{font-size:12px;font-weight:700;text-align:right}}
.nt-fine{{font-size:12px;color:var(--text-muted);margin:10px 0 0}}
.nt-chart{{width:100%;height:auto;display:block}}
.nt-chart text{{font-family:inherit}}
.nt-cmp{{display:flex;align-items:center;gap:8px;margin:0 0 8px;font-size:12px;color:var(--text-muted)}}
.nt-cmp select{{font:inherit;color:var(--text);background:var(--card-soft);border:1px solid var(--border);border-radius:8px;padding:8px 10px;min-height:40px;max-width:100%}}
.nt-cmp-legend{{display:flex;gap:14px;margin:8px 0 0;font-size:12px;color:var(--text-muted)}}
.nt-cmp-legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.nt-chips{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px}}
.nt-chip{{background:var(--card-soft);border-radius:10px;padding:12px;text-align:center}}
.nt-chip .nt-clab{{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em}}
.nt-chip .nt-cval{{font-size:26px;font-weight:800;margin:2px 0}}
.nt-chip .nt-crk{{font-size:11px;color:var(--text-muted)}}
details.nt-method{{border:1px solid var(--border);border-radius:8px;padding:10px 12px;font-size:13px}}
details.nt-method summary{{cursor:pointer;font-weight:700;min-height:32px}}
.nt-mrow{{display:flex;justify-content:space-between;padding:6px 0;border-top:1px solid var(--border);font-size:13px}}
.nt-def4{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:6px 0 10px}}
/* Defense vs position: each position is its own card so the four groups read
   as separate blocks instead of two continuous columns of numbers. */
.nt-def4cell{{text-align:center;min-width:0;background:var(--card-soft);border:1px solid var(--border);border-radius:12px;padding:12px 8px 10px}}
.nt-def4pos{{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px;white-space:nowrap}}
.nt-def4val{{font-size:17px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1.2}}
.nt-def4cell .nt-rbadge{{margin-top:4px}}
@media(max-width:420px){{.nt-def4{{grid-template-columns:repeat(2,1fr)}}}}
/* Collapsible page sections (Graphs, Schedule): native details/summary so no
   JS wiring is needed and the open state survives re-renders of inner HTML. */
details.nt-collapse{{margin-bottom:16px}}
details.nt-collapse>summary{{list-style:none;cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:44px}}
details.nt-collapse>summary::-webkit-details-marker{{display:none}}
details.nt-collapse>summary h3{{margin:0}}
.nt-chev{{width:9px;height:9px;border-right:2px solid var(--text-muted);border-bottom:2px solid var(--text-muted);transform:rotate(-45deg);transition:transform .18s ease;flex:none;margin-right:6px}}
details.nt-collapse[open]>summary .nt-chev{{transform:rotate(45deg)}}
.nt-seg{{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}}
.nt-seg button{{border:1px solid var(--border);background:var(--card);color:var(--text-muted);border-radius:8px;padding:8px 14px;font-size:13px;font-weight:700;cursor:pointer;min-height:40px;font-family:inherit}}
.nt-seg button[aria-pressed="true"]{{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}}
table.nt-depth{{border-collapse:collapse;width:100%}}
table.nt-depth th{{text-align:left;font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;padding:8px;border-bottom:1px solid var(--border)}}
table.nt-depth th.nt-num,table.nt-depth td.nt-num{{text-align:right}}
table.nt-depth td{{padding:9px 8px;border-top:1px solid var(--border);font-size:13px}}
table.nt-depth>thead>tr>th:first-child,table.nt-depth>tbody>tr>td:first-child{{position:sticky;left:0;z-index:1;background:var(--card);border-right:1px solid var(--border)}}
table.nt-depth>thead>tr>th:first-child{{z-index:2}}
.nt-pname{{background:none;border:0;color:var(--accent);font:inherit;font-weight:600;cursor:pointer;padding:0;text-align:left;min-height:32px}}
.nt-dno{{display:inline-flex;width:22px;height:22px;border-radius:50%;background:var(--card-soft);align-items:center;justify-content:center;font-size:11px;font-weight:700;margin-right:8px;color:var(--text-muted)}}
.nt-inj{{display:inline-block;background:#8a5a00;color:#fff;font-size:10px;font-weight:800;border-radius:4px;padding:1px 5px;margin-left:6px}}
.nt-sched{{border:1px solid var(--border);border-radius:10px;overflow:hidden}}
.nt-wrow{{display:flex;align-items:center;gap:10px;width:100%;background:none;border:0;border-top:1px solid var(--border);color:var(--text);font:inherit;padding:11px 12px;cursor:pointer;text-align:left;min-height:48px}}
.nt-wrow:first-child{{border-top:0}}
.nt-wrow .nt-wk{{width:44px;color:var(--text-muted);font-size:12px;flex:none}}
.nt-wrow .nt-opp{{flex:1;display:flex;align-items:center;gap:8px;font-weight:600}}
.nt-wrow .nt-res{{font-size:13px;color:var(--text-muted)}}
.nt-wrow .nt-res.nt-w{{color:var(--win);font-weight:700;background:color-mix(in srgb,var(--win) 14%,transparent);border-radius:6px;padding:3px 8px;font-size:12px;white-space:nowrap}}
.nt-wrow .nt-res.nt-l{{color:var(--loss);font-weight:700;background:color-mix(in srgb,var(--loss) 14%,transparent);border-radius:6px;padding:3px 8px;font-size:12px;white-space:nowrap}}
.nt-opp-logo{{width:22px;height:22px;border-radius:50%;flex:none;background:var(--card-soft)}}
.nt-wrow.nt-bye{{cursor:default;color:var(--text-muted)}}
.nt-box{{padding:4px 12px 12px;font-size:13px}}
.nt-box-head{{display:flex;align-items:center;justify-content:center;gap:14px;padding:10px 4px 12px}}
.nt-box-side{{display:flex;align-items:center;gap:8px;min-width:0}}
.nt-box-logo{{width:30px;height:30px;object-fit:contain;flex:none}}
.nt-box-abbr{{font-weight:700;font-size:13px;letter-spacing:.02em}}
.nt-box-score{{font-size:22px;font-weight:800;font-variant-numeric:tabular-nums}}
.nt-box-score.dim{{color:var(--text-muted);font-weight:700}}
.nt-box-status{{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;white-space:nowrap}}
.nt-boxscore-teams{{display:grid;grid-template-columns:1fr 1fr;gap:8px;align-items:start}}
.nt-boxscore-team{{margin-bottom:0;min-width:0}}
.nt-boxscore-team>b{{display:block;font-size:13px;margin-bottom:4px}}
.nt-boxscore-team table.nt-depth th[colspan]{{background:var(--card-soft);font-size:11px;padding:5px 6px}}
.nt-boxscore table.nt-depth{{border-collapse:collapse}}
.nt-boxscore table.nt-depth th{{font-size:10px;padding:5px 6px;white-space:nowrap}}
.nt-boxscore table.nt-depth th.nt-stat{{width:1%;text-align:right}}
.nt-boxscore table.nt-depth td{{padding:5px 6px;font-size:12px;white-space:nowrap}}
.nt-boxscore table.nt-depth td.nt-stat{{text-align:right;font-variant-numeric:tabular-nums}}
.nt-boxscore table.nt-depth td.nt-pcell{{max-width:220px;overflow:hidden;text-overflow:ellipsis}}
.nt-boxscore .nt-pname{{font-size:12px}}
.nt-sheet-backdrop{{position:fixed;inset:0;z-index:calc(var(--z-modal) - 1);background:rgba(0,0,0,.6);opacity:0;transition:opacity .25s ease}}
.nt-sheet-backdrop.open{{opacity:1}}
.nt-sheet{{position:fixed;left:0;right:0;bottom:0;z-index:var(--z-modal);max-width:480px;margin:0 auto;max-height:88vh;max-height:88dvh;display:flex;flex-direction:column;background:var(--card);border:1px solid var(--border);border-bottom:none;border-radius:18px 18px 0 0;transform:translateY(102%);transition:transform .3s cubic-bezier(.32,.72,.28,1);color:var(--text)}}
.nt-sheet.open{{transform:translateY(0)}}
.nt-sheet-handle{{width:40px;height:4px;border-radius:2px;background:var(--border);margin:10px auto 2px;flex:0 0 auto}}
.nt-sheet-head{{display:flex;align-items:center;gap:10px;padding:8px 16px 10px;border-bottom:1px solid var(--border);flex:0 0 auto}}
.nt-sheet-title{{font-size:16px;font-weight:800}}
.nt-sheet-sub{{font-size:11.5px;color:var(--text-muted);font-weight:600;margin-top:1px;font-variant-numeric:tabular-nums}}
.nt-sheet-x{{margin-left:auto;appearance:none;cursor:pointer;width:32px;height:32px;border-radius:10px;border:1px solid var(--border);background:var(--card-soft);color:var(--text-muted);font-size:15px;font-weight:700;line-height:1;flex:none}}
.nt-sheet-body{{flex:1 1 auto;min-height:0;overflow-y:auto;padding:12px 16px calc(28px + env(safe-area-inset-bottom));-webkit-overflow-scrolling:touch}}
.nt-sheet-teams{{display:flex;gap:6px;margin-bottom:6px;background:var(--card-soft);border:1px solid var(--border);border-radius:12px;padding:4px}}
.nt-sheet-team{{flex:1;appearance:none;cursor:pointer;border:none;background:transparent;color:var(--text-muted);font:inherit;font-size:13px;font-weight:800;padding:9px 6px;border-radius:9px;min-height:40px}}
.nt-sheet-team.is-on{{background:var(--bg);color:var(--text);box-shadow:inset 0 0 0 1px var(--border)}}
.nt-sheet .nt-boxscore-teams{{grid-template-columns:1fr}}
.nt-sheet .nt-boxscore-team>b{{font-size:14px}}
@media(prefers-reduced-motion:reduce){{.nt-sheet-backdrop,.nt-sheet{{transition:none}}}}
.nt-err{{padding:24px 16px;text-align:center;color:var(--text-muted)}}
.nt-err button{{margin-top:10px}}
.nt-load{{padding:32px 16px;text-align:center;color:var(--text-muted)}}

</style>
<div class="card nt-card">
  <div class="card-header nt-chead">
    <div><h1>NFL Team Rankings</h1><p class="nt-sub" id="ntSeasonSub">Loading team data.</p></div>
    <div class="nt-controls">
      <button type="button" class="nt-hbtn" id="ntHowBtn">How ranks work</button>
      <button type="button" class="nt-hbtn" id="ntCsvBtn">CSV</button>
      <label class="nt-seaslab" for="ntSeasonSel">Season</label>
      <span class="nt-csel"><select id="ntSeasonSel" aria-label="Season">{season_opts}</select></span>
    </div>
  </div>
  <div class="nt-cbody">
    <div class="nt-tabs" role="tablist" id="ntTabs" aria-label="Ranking views">{tabs}</div>
    <div class="nt-tcard" id="ntListWrap">
      <div class="nt-tscroll"><table class="nt-rank" id="ntTbl" aria-label="NFL team rankings"></table></div>
      <p class="nt-tnote" id="ntTableNote"></p>
    </div>
  </div>
</div>
<section id="ntProfile" aria-live="polite"></section>
<script>
(function(){{
"use strict";
var $=function(s,r){{return (r||document).querySelector(s);}};
var DEF={{view:"{view}",sortKey:null,sortDir:null,team:"{team}",season:"{season}",room:"QB",expanded:{{}}}};
var state=Object.assign({{}},DEF);
function esc(s){{return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}}
function teamColor(t){{return (t&&(t.color||""))||"var(--accent)";}}
function ntEnvRank(rank){{
  if(!rank)return '<span class="nt-rbadge">N/A</span>';
  var tier=rank<=10?" nt-tier-g":(rank>=23?" nt-tier-b":"");
  return '<span class="nt-rbadge'+tier+'">#'+rank+'</span>';
}}
function rankBadge(rank,fg,bg){{
  if(!rank)return "";
  var style=(fg&&bg)?' style="background:'+bg+';color:'+fg+'"':"";
  var gold=(!fg&&rank===1)?" nt-gold":"";
  return '<span class="nt-rbadge'+gold+'"'+style+">#"+rank+"</span>";
}}
/* Matchup-ease colors mirror utils/schedule_ease.py sched_rank_color
   (rank 1 = most allowed = easiest). Quartile tiers, not a new scale. */
function easeTier(rank,total){{
  if(!rank||!total)return 0;
  var pct=rank/total;
  return pct<=0.25?1:pct<=0.50?2:pct<=0.75?3:4;
}}
var EASE_FG={{1:"#22c55e",2:"#84cc16",3:"#f59e0b",4:"#ef4444"}};
var EASE_BG={{1:"#22c55e18",2:"#84cc1618",3:"#f59e0b18",4:"#ef444418"}};
function shortEff(label){{
  var l=String(label||"").toLowerCase();
  if(l.indexOf("attempt")>=0)return "Y/A";
  if(l.indexOf("carry")>=0)return "Y/C";
  if(l.indexOf("target")>=0)return "Y/T";
  return label||"";
}}
/* Columns actually shown for the current view (projection mode hides scoring). */
function visibleCols(){{
  var cols=VIEWS[state.view].cols;
  if(state.view==="defense")return cols;
  return cols.filter(function(c){{return !(state.view==="overview"&&c.k==="points_pg"&&DATA.data_mode==="projection");}});
}}
/* Defense-vs-position payload, fetched lazily from /api/defense-vs-position
   (PR #1908). {{season, teams}} on success, {{failed:true}} when the endpoint is
   unavailable, so the tab degrades instead of breaking. */
var DPOS=null;
function ensureDefense(){{
  if(DPOS&&(String(DPOS.season)===String(state.season)||DPOS.failed))return;
  var season=state.season;
  api("/api/defense-vs-position?season="+encodeURIComponent(season))
    .then(function(d){{
      var teams={{}};
      var poss=d.positions||["QB","RB","WR","TE"];
      Object.keys(d.teams||{{}}).forEach(function(ab){{
        var src=d.teams[ab]||{{}},tp={{}};
        poss.forEach(function(p){{
          var c=src[p];
          if(c&&c.fpts_ppr_pg!=null)tp[p]={{v:c.fpts_ppr_pg,rank:c.rank,total:c.total,eff:c.eff,eff_label:c.eff_label}};
        }});
        teams[ab]=tp;
      }});
      DPOS={{season:String(d.season||season),teams:teams}};
      render();
    }})
    .catch(function(){{DPOS={{failed:true}};render();}});
}}

var VIEWS={{
overview:{{label:"Overview",def:"points_pg",cols:[
 {{k:"points_pg",t:"Scoring",tip:"Points scored per game, actual results. Hidden for projections: scoring cannot be projected from offense stats alone.",f:1,hb:true}},
 {{k:"plays_pg",t:"Plays / game",tip:"Offensive plays per game.",f:1,hb:true}},
 {{k:"pass_yds_pg",t:"Pass yds",tip:"Passing yards per game.",f:0,hb:true}},
 {{k:"rush_yds_pg",t:"Rush yds",tip:"Rushing yards per game.",f:0,hb:true}},
 {{k:"total_yds_pg",t:"Total yds",tip:"Pass yards plus rush yards per game.",f:0,hb:true}},
 {{k:"pass_rate",t:"Pass rate",tip:"Pass attempts as a share of pass attempts plus rush attempts.",f:"pct",hb:true}}]}},
passing:{{label:"Passing",def:"pass_yds_pg",cols:[
 {{k:"pass_yds_pg",t:"Pass yds",tip:"Passing yards per game.",f:0,hb:true}},
 {{k:"pass_att_pg",t:"Pass att.",tip:"Pass attempts per game.",f:1,hb:true}},
 {{k:"pass_tds_pg",t:"Pass TD",tip:"Passing touchdowns per game.",f:1,hb:true}},
 {{k:"pass_rate",t:"Pass rate",tip:"Pass attempts as a share of pass attempts plus rush attempts.",f:"pct",hb:true}}]}},
rushing:{{label:"Rushing",def:"rush_yds_pg",cols:[
 {{k:"rush_yds_pg",t:"Rush yds",tip:"Rushing yards per game.",f:0,hb:true}},
 {{k:"rush_att_pg",t:"Rush att.",tip:"Rush attempts per game.",f:1,hb:true}},
 {{k:"rush_tds_pg",t:"Rush TD",tip:"Rushing touchdowns per game.",f:1,hb:true}}]}},
oline:{{label:"Offensive Line",def:"oline_composite",cols:[
 {{k:"oline_composite",t:"Overall",tip:"0-100 overall line unit rating. Season labeled below the table.",f:0,hb:true}},
 {{k:"oline_pass_block",t:"Pass block",tip:"0-100 pass protection unit rating.",f:0,hb:true}},
 {{k:"oline_run_block",t:"Run block",tip:"0-100 run blocking unit rating.",f:0,hb:true}},
 {{k:"oline_pressure_rate",t:"Pressure %",tip:"Share of dropbacks under pressure. Lower is better.",f:"pct1",hb:false}},
 {{k:"oline_sack_rate",t:"Sack %",tip:"Share of dropbacks ending in a sack. Lower is better.",f:"pct1",hb:false}},
 {{k:"oline_line_yards",t:"Line yds",tip:"Adjusted line yards per carry.",f:1,hb:true}}]}},
/* Defense vs position: fantasy points allowed per game by each defense,
   from /api/defense-vs-position (PR #1908). Rank 1 = most allowed =
   easiest matchup. Loaded lazily; the tab degrades gracefully if the
   endpoint is unavailable. */
defense:{{label:"Defense",def:"QB",cols:[
 {{k:"QB",t:"vs QB",tip:"Fantasy points allowed per game to QBs (PPR). Rank 1 = most allowed = easiest matchup.",f:1,hb:true}},
 {{k:"RB",t:"vs RB",tip:"Fantasy points allowed per game to RBs (PPR). Rank 1 = most allowed = easiest matchup.",f:1,hb:true}},
 {{k:"WR",t:"vs WR",tip:"Fantasy points allowed per game to WRs (PPR). Rank 1 = most allowed = easiest matchup.",f:1,hb:true}},
 {{k:"TE",t:"vs TE",tip:"Fantasy points allowed per game to TEs (PPR). Rank 1 = most allowed = easiest matchup.",f:1,hb:true}}]}}
}};
var ROOMCOLS={{
QB:[["snap_pct","Snap %"],["games","G"],["ppg","PPR PPG"]],
RB:[["snap_pct","Snap %"],["tgt_share","Tgt %"],["carry_share","Carry %"],["touch_share","Touch %"],["ppg","PPR PPG"]],
WR:[["tgt_share","Tgt %"],["snap_pct","Snap %"],["ppg","PPR PPG"]],
TE:[["tgt_share","Tgt %"],["snap_pct","Snap %"],["ppg","PPR PPG"]]}};

var DATA=null;   /* /api/nfl-team-rankings payload */
var DETAIL=null; /* /api/nfl-team-details payload for state.team */
var DETAIL_TEAM=null;
var FPRINT_CMP=null; /* {{team, abbr}} fingerprint compare selection */

function api(u){{return fetch(u,{{credentials:"same-origin"}}).then(function(r){{if(!r.ok)throw new Error("http "+r.status);return r.json();}});}}
function update(patch,push){{Object.assign(state,patch);render();syncUrl(push);}}
function cellFor(t,col){{
  var src;
  if(state.view==="defense"){{src=((DPOS&&DPOS.teams||{{}})[t.team]||{{}})[col.k]||null;}}
  else if(col.k.indexOf("oline_")===0){{src=(t.oline||{{}})[col.k.slice(6)]||null;}}
  else{{src=(t.ranks||{{}})[col.k]||null;}}
  return src;
}}
/* The rankings APIs send entries as {{rank, total, value}}. Normalize value->v
   once per payload so every renderer below can rely on entry.v. */
function normEntry(e){{
  if(e&&typeof e==="object"&&e.v===undefined&&e.value!==undefined){{e.v=e.value;}}
  return e;
}}
function normOline(ol){{
  if(!ol)return;
  Object.keys(ol).forEach(function(k){{if(k!=="season")normEntry(ol[k]);}});
}}
function normRankings(d){{
  (d.teams||[]).forEach(function(t){{
    var r=t.ranks||{{}};Object.keys(r).forEach(function(k){{normEntry(r[k]);}});
    normOline(t.oline);
  }});
}}
function fmtVal(col,v){{
  if(v==null)return null;
  if(col.f==="pct")return Math.round(v*100)+"%";
  if(col.f==="pct1")return Number(v).toFixed(1)+"%";
  if(col.f===1)return Number(v).toFixed(1);
  return String(Math.round(v));
}}
function defaultDir(col){{return col.hb===false?"asc":"desc";}}
function logoHTML(t,lg){{
  var cls="nt-logo"+(lg?" nt-lg":"");
  if(t.logo){{return '<span class="'+cls+'"><img src="'+esc(t.logo)+'" alt="" loading="lazy" onerror="this.remove()"></span>';}}
  return '<span class="'+cls+'">'+esc(t.team.slice(0,2))+'</span>';
}}

function renderTabs(){{
  var el=$("#ntTabs");
  el.querySelectorAll("button").forEach(function(b){{
    var v=b.getAttribute("data-view");
    var on=(v===state.view);
    b.setAttribute("aria-selected",on?"true":"false");
    if(b.__ntv!==v){{b.__ntv=v;b.onclick=function(){{update({{view:v,sortKey:VIEWS[v].def,sortDir:defaultDir(VIEWS[v].cols[0])}},true);}};}}
  }});
}}

function downloadCsv(){{
  if(!DATA||!DATA.teams)return;
  var cols=visibleCols();
  function q(x){{return '"'+String(x==null?"":x).replace(/"/g,'""')+'"';}}
  var lines=["Team,"+cols.map(function(c){{return q(c.t);}}).join(",")];
  DATA.teams.slice().sort(function(a,b){{return a.team<b.team?-1:1;}}).forEach(function(t){{
    var row=[q(t.city+" "+t.name)];
    cols.forEach(function(c){{
      var cell=cellFor(t,c);
      row.push(q(cell&&cell.v!=null?cell.v:""));
    }});
    lines.push(row.join(","));
  }});
  var blob=new Blob([lines.join("\\n")],{{type:"text/csv"}});
  var a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download="nfl-team-rankings-"+state.view+"-"+state.season+".csv";
  document.body.appendChild(a);a.click();
  setTimeout(function(){{URL.revokeObjectURL(a.href);a.remove();}},200);
}}

function renderTable(){{
  var tbl=$("#ntTbl");
  if(!DATA||!DATA.teams){{tbl.innerHTML="";$("#ntTableNote").textContent="";return;}}
  var view=VIEWS[state.view];
  var cols=visibleCols();
  if(!state.sortKey||!cols.some(function(c){{return c.k===state.sortKey;}})){{state.sortKey=view.def;state.sortDir=defaultDir(view.cols[0]);}}
  var skey=state.sortKey,sdir=state.sortDir||"desc";
  var scol=cols.filter(function(c){{return c.k===skey;}})[0]||cols[0];
  var rows=DATA.teams.slice();
  rows.sort(function(a,b){{
    var av=cellFor(a,scol),bv=cellFor(b,scol);
    av=av?av.v:null;bv=bv?bv.v:null;
    if(av==null&&bv==null)return a.team<b.team?-1:1;
    if(av==null)return 1;if(bv==null)return -1;
    var d=sdir==="asc"?(av-bv):(bv-av);
    return d!==0?d:(a.team<b.team?-1:1);
  }});
  var h='<thead><tr><th class="nt-rankcol" scope="col"><span>Rank</span></th><th class="nt-teamcol" scope="col"><span>Team</span></th>';
  cols.forEach(function(c){{
    var active=(c.k===skey);
    var arrow=active?(sdir==="asc"?"&#8593;":"&#8595;"):"";
    h+='<th scope="col" aria-sort="'+(active?(sdir==="asc"?"ascending":"descending"):"none")+'">'+
      '<button class="nt-thbtn" data-col="'+c.k+'" title="'+esc(c.tip)+'" aria-label="Sort by '+esc(c.t)+'">'+esc(c.t)+'<span class="nt-arr">'+arrow+'</span></button></th>';
  }});
  h+='</tr></thead><tbody>';
  rows.forEach(function(t){{
    var tc=teamColor(t);
    var sc=cellFor(t,scol);
    var isDef=(state.view==="defense");
    var badge=rankBadge(sc&&sc.rank);
    if(isDef&&sc&&sc.rank){{
      var bt=easeTier(sc.rank,sc.total);
      if(bt)badge=rankBadge(sc.rank,EASE_FG[bt],EASE_BG[bt]);
    }}
    h+='<tr data-abbr="'+t.team+'" class="'+(state.team===t.team?"nt-sel":"")+'">'+
      '<td class="nt-rankcol">'+badge+'</td>'+
      '<td class="nt-teamcol"><span class="nt-tid"><span class="nt-tleft">'+logoHTML(t)+'<span class="nt-tn">'+esc(t.city)+' '+esc(t.name)+'</span></span>'+
      '<span class="nt-tright">'+
      '<span class="nt-gp">'+(t.games==null?"":t.games+" GP")+'</span></span></span></td>';
    cols.forEach(function(c){{
      var cell=cellFor(t,c);
      var v=cell?fmtVal(c,cell.v):null;
      if(v==null){{h+='<td><div class="nt-metric"><span class="nt-val nt-na">N/A</span></div></td>';}}
      else if(isDef){{
        var tier=easeTier(cell.rank,cell.total);
        var w2=cell.rank?Math.max(3,Math.round((33-cell.rank)/32*100)):0;
        var fg=tier?EASE_FG[tier]:"var(--text)";
        var effTxt=cell.eff!=null?'<span class="nt-eff">'+Number(cell.eff).toFixed(1)+' '+esc(shortEff(cell.eff_label))+'</span>':"";
        h+='<td><div class="nt-metric"><span class="nt-mbar"><span class="nt-mtrack"><span class="nt-mfill" style="width:'+w2+'%;background:'+fg+'"></span></span></span>'+
          '<span class="nt-vwrap"><span class="nt-val" style="color:'+fg+'">'+v+'</span>'+effTxt+'</span></div></td>';
      }}
      else{{
        var w=cell.rank?Math.max(3,Math.round((33-cell.rank)/32*100)):0;
        h+='<td><div class="nt-metric"><span class="nt-mbar"><span class="nt-mtrack"><span class="nt-mfill" style="width:'+w+'%;background:'+esc(tc)+'"></span></span></span><span class="nt-val">'+v+'</span></div></td>';
      }}
    }});
    h+='</tr>';
  }});
  h+='</tbody>';
  tbl.innerHTML=h;
  tbl.querySelectorAll(".nt-thbtn").forEach(function(b){{
    b.addEventListener("click",function(){{
      var k=b.getAttribute("data-col");
      var col=cols.filter(function(c){{return c.k===k;}})[0];
      if(state.sortKey===k){{update({{sortDir:state.sortDir==="asc"?"desc":"asc"}},false);}}
      else{{update({{sortKey:k,sortDir:defaultDir(col)}},false);}}
    }});
  }});
  tbl.querySelectorAll("tbody tr").forEach(function(tr){{
    tr.addEventListener("click",function(){{update({{team:tr.getAttribute("data-abbr")}},true);scrollTop();}});
  }});
  var note="Ranks use competition ranking (1, 2, 2, 4) across all 32 teams. N/A means no data, not zero. "+
    "Lower pressure and sack rates rank better. O-line ratings are 0-100 unit scores from public play-by-play ("+esc(DATA.oline_note||"prior season")+").";
  if(state.view==="defense"){{
    note="Fantasy points allowed per game (PPR). Rank 1 = most allowed = easiest matchup. Only completed games count; defenses with no completed games are omitted.";
    if(DPOS&&DPOS.failed)note+=" Defensive matchup data is currently unavailable.";
  }}
  if(DATA.data_mode==="projection"){{note="Projection mode: per-game values are projected season totals divided by 17. Scoring is hidden because NFL points cannot be honestly projected from offense stats alone. "+note;}}
  $("#ntTableNote").textContent=note;
  $("#ntSeasonSub").textContent="Team-level research, not fantasy roster rankings. "+(DATA.season_label||"");
}}

function envRows(t){{
  var R=t.ranks||{{}};
  function na(v){{return v==null||isNaN(Number(v));}}
  function e(label,key,fmt){{
    var c=R[key];if(!c)return null;
    return {{label:label,val:fmt(c.v),rank:c.rank}};
  }}
  return [
    e("Scoring","points_pg",function(v){{return na(v)?"N/A":Number(v).toFixed(1)+" pts/g";}}),
    e("Plays / game","plays_pg",function(v){{return na(v)?"N/A":Number(v).toFixed(1);}}),
    e("Pass yards","pass_yds_pg",function(v){{return na(v)?"N/A":Math.round(v)+" /g";}}),
    e("Rush yards","rush_yds_pg",function(v){{return na(v)?"N/A":Math.round(v)+" /g";}}),
    e("Pass rate","pass_rate",function(v){{return na(v)?"N/A":Math.round(v*100)+"%";}})
  ].filter(Boolean);
}}

/* Team profile charts: dependency-free SVG. Percentiles come from competition
   ranks (rank 1 = best), so pct = (total - rank + 1) / total * 100. */
function ntPct(entry){{
  if(!entry||entry.rank==null||!entry.total)return null;
  return Math.round((entry.total-entry.rank+1)/entry.total*100);
}}
/* #3: grouped bars of points scored vs allowed, home and away, from completed
   games in the details schedule payload. */
function splitsSVG(d,t){{
  var finals=(d.schedule||[]).filter(function(g){{
    return !g.bye&&g.status==="final"&&g.team_pts!=null&&g.opp_pts!=null;
  }});
  function grp(home){{
    var gs=finals.filter(function(g){{return !!g.is_home===home;}});
    function avg(k){{
      if(!gs.length)return null;
      return gs.reduce(function(s,g){{return s+Number(g[k]);}},0)/gs.length;
    }}
    return {{
      n:gs.length,
      w:gs.filter(function(g){{return g.result==="W";}}).length,
      l:gs.filter(function(g){{return g.result==="L";}}).length,
      sc:avg("team_pts"),al:avg("opp_pts")
    }};
  }}
  var groups=[{{label:"Home",g:grp(true)}},{{label:"Away",g:grp(false)}}];
  if(!groups[0].g.n&&!groups[1].g.n)
    return '<p class="nt-fine">No completed games yet.</p>';
  var max=0;
  groups.forEach(function(gr){{
    [gr.g.sc,gr.g.al].forEach(function(v){{if(v!=null)max=Math.max(max,v);}});
  }});
  max=Math.max(10,Math.ceil(max/5)*5);
  var W=360,bx=76,bw=W-bx-44,rowH=62,top=6;
  var H=top+groups.length*rowH+8;
  var tcolor=teamColor(t);
  function barW(v){{return Math.max(2,v/max*bw);}}
  var s='<svg class="nt-chart" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Points scored and allowed at home and away">';
  groups.forEach(function(gr,i){{
    var y=top+i*rowH,g=gr.g;
    s+='<text x="0" y="'+(y+14)+'" font-size="12" font-weight="700" fill="var(--text)">'+esc(gr.label)+
      (g.n?' <tspan font-weight="400" fill="var(--text-muted)">'+g.w+'-'+g.l+'</tspan>':'')+'</text>';
    if(!g.n){{
      s+='<text x="0" y="'+(y+36)+'" font-size="11" fill="var(--text-muted)">No games yet.</text>';
      return;
    }}
    [{{lab:"Scored",v:g.sc,c:tcolor}},{{lab:"Allowed",v:g.al,c:"var(--text-muted)"}}].forEach(function(r,j){{
      var by=y+20+j*20,wpx=barW(r.v);
      s+='<text x="0" y="'+(by+10)+'" font-size="11" fill="var(--text-muted)">'+r.lab+'</text>';
      s+='<rect x="'+bx+'" y="'+by+'" width="'+wpx.toFixed(1)+'" height="12" rx="3" fill="'+r.c+'"/>';
      s+='<text x="'+(bx+wpx+6).toFixed(1)+'" y="'+(by+10)+'" font-size="11" font-weight="700" fill="var(--text)">'+r.v.toFixed(1)+'</text>';
    }});
  }});
  return s+'</svg>';
}}
/* #4: radar of league percentiles across six higher-is-better offensive metrics. */
var FPRINT_AXES=[
  ["Scoring","points_pg"],["Pass yds","pass_yds_pg"],["Rush yds","rush_yds_pg"],
  ["Total yds","total_yds_pg"],["Pass TD","pass_tds_pg"],["Rush TD","rush_tds_pg"]
];
function fingerprintSVG(t,u){{
  var R=t.ranks||{{}};
  var pts=FPRINT_AXES.map(function(a){{
    return {{label:a[0],key:a[1],pct:ntPct(R[a[1]])}};
  }}).filter(function(p){{return p.pct!=null;}});
  if(pts.length<3)
    return '<p class="nt-fine">Not enough ranked metrics yet.</p>';
  var upts=null;
  if(u&&u.team&&u.team!==t.team){{
    var UR=u.ranks||{{}};
    var q=pts.map(function(p){{return ntPct(UR[p.key]);}});
    if(q.every(function(v){{return v!=null;}}))upts=q;
  }}
  var W=320,cx=160,cy=132,Rr=86,n=pts.length;
  function pos(i,rad){{
    var ang=-Math.PI/2+i*2*Math.PI/n;
    return [cx+rad*Math.cos(ang),cy+rad*Math.sin(ang)];
  }}
  function ring(rad){{
    var d=[];
    for(var i=0;i<n;i++){{var p=pos(i,rad);d.push(p[0].toFixed(1)+' '+p[1].toFixed(1));}}
    return d.join(' ');
  }}
  var s='<svg class="nt-chart" viewBox="0 0 '+W+' 264" role="img" aria-label="Team fingerprint: league percentiles">';
  [25,50,75,100].forEach(function(r){{
    s+='<polygon points="'+ring(Rr*r/100)+'" fill="none" stroke="var(--border)" stroke-width="1"/>';
  }});
  for(var i=0;i<n;i++){{
    var pe=pos(i,Rr);
    s+='<line x1="'+cx+'" y1="'+cy+'" x2="'+pe[0].toFixed(1)+'" y2="'+pe[1].toFixed(1)+'" stroke="var(--border)" stroke-width="1"/>';
  }}
  var tcolor=teamColor(t);
  if(upts){{
    var uv=upts.map(function(pct,i){{
      var up=pos(i,Rr*pct/100);
      return up[0].toFixed(1)+' '+up[1].toFixed(1);
    }});
    s+='<polygon points="'+uv.join(' ')+'" fill="var(--text-muted)" fill-opacity="0.15" stroke="var(--text-muted)" stroke-width="1.5"/>';
    upts.forEach(function(pct,i){{
      var up=pos(i,Rr*pct/100);
      s+='<circle cx="'+up[0].toFixed(1)+'" cy="'+up[1].toFixed(1)+'" r="3" fill="var(--text-muted)"/>';
    }});
  }}
  var vpts=pts.map(function(p,i){{
    var q=pos(i,Rr*p.pct/100);
    return q[0].toFixed(1)+' '+q[1].toFixed(1);
  }});
  s+='<polygon points="'+vpts.join(' ')+'" fill="'+tcolor+'" fill-opacity="0.25" stroke="'+tcolor+'" stroke-width="2"/>';
  pts.forEach(function(p,i){{
    var q=pos(i,Rr*p.pct/100);
    s+='<circle cx="'+q[0].toFixed(1)+'" cy="'+q[1].toFixed(1)+'" r="3.5" fill="'+tcolor+'"/>';
    var lp=pos(i,Rr+16);
    var anchor=Math.abs(lp[0]-cx)<8?"middle":(lp[0]>cx?"start":"end");
    s+='<text x="'+lp[0].toFixed(1)+'" y="'+lp[1].toFixed(1)+'" font-size="11" text-anchor="'+anchor+'" fill="var(--text)">'+
      esc(p.label)+' <tspan font-weight="700">'+p.pct+'</tspan></text>';
  }});
  return s+'</svg>';
}}
/* Fingerprint section with a compare-team picker. The compare team renders as
   a gray overlay polygon; the selection is ephemeral UI state (not in the URL)
   and resets when the profile team changes. */
function fingerprintSection(t){{
  var teams=((DATA&&DATA.teams)||[]).slice().sort(function(a,b){{
    var an=(a.city||"")+" "+(a.name||""),bn=(b.city||"")+" "+(b.name||"");
    return an<bn?-1:an>bn?1:0;
  }});
  var cmpAbbr=(FPRINT_CMP&&FPRINT_CMP.team===state.team)?FPRINT_CMP.abbr:null;
  var cmpRow=null;
  var opts='<option value="">None</option>';
  teams.forEach(function(x){{
    if(x.team===t.team)return;
    if(x.team===cmpAbbr)cmpRow=x;
    opts+='<option value="'+esc(x.team)+'"'+(x.team===cmpAbbr?" selected":"")+'>'+
      esc((((x.city||"")+" "+(x.name||"")).trim())||x.team)+'</option>';
  }});
  var h='<section class="nt-psec"><h3>Team fingerprint</h3>';
  h+='<div class="nt-cmp"><label for="ntCmpSel">Compare</label><select id="ntCmpSel" aria-label="Compare fingerprint with another team">'+opts+'</select></div>';
  h+=fingerprintSVG(t,cmpRow);
  if(cmpRow){{
    h+='<div class="nt-cmp-legend"><span><i style="background:'+esc(teamColor(t))+'"></i>'+esc(t.team)+
      '</span><span><i style="background:var(--text-muted)"></i>'+esc(cmpRow.team)+'</span></div>';
  }}
  h+='<p class="nt-fine">League percentile by rank. Higher is better on every axis.</p></section>';
  return h;
}}
/* Pass block vs run block as a 2D scatter over all 32 teams, from the already
   loaded rankings payload. The selected team is highlighted. */
function ev(e){{var v=e&&e.v;return (v==null||!isFinite(Number(v)))?null:Number(v);}}
/* Generic 2D league scatter with median splits and quadrant labels.
   cfg: pts [{{abbr,x,y}}], xLab, yLab, quads [topRight,topLeft,bottomRight,bottomLeft],
   xTick, yTick, tip(p), aria, emptyMsg. The selected team is highlighted. */
function scatterSVG(t,cfg){{
  var pts=(cfg.pts||[]).filter(function(p){{return p&&p.x!=null&&p.y!=null&&isFinite(p.x)&&isFinite(p.y);}});
  if(pts.length<4)
    return '<p class="nt-fine">'+esc(cfg.emptyMsg||"Not enough league data.")+'</p>';
  var xs=pts.map(function(p){{return p.x;}}),
      ys=pts.map(function(p){{return p.y;}});
  var xlo=Math.min.apply(null,xs),xhi=Math.max.apply(null,xs);
  var xpad=(xhi-xlo)*0.08||1;
  var x0=xlo-xpad,x1=xhi+xpad;
  var ylo=Math.min.apply(null,ys),yhi=Math.max.apply(null,ys);
  var ypad=(yhi-ylo)*0.08||1;
  var y0=ylo-ypad,y1=yhi+ypad;
  var W=360,H=300,ml=36,mr=14,mt=12,mb=32;
  var pw=W-ml-mr,ph=H-mt-mb;
  function X(v){{return ml+(v-x0)/(x1-x0)*pw;}}
  function Y(v){{return mt+ph-(v-y0)/(y1-y0)*ph;}}
  function med(a){{
    var b=a.slice().sort(function(p,q){{return p-q;}});
    var m=b.length>>1;
    return b.length%2?b[m]:(b[m-1]+b[m])/2;
  }}
  var xTick=cfg.xTick||function(v){{return String(Math.round(v));}};
  var yTick=cfg.yTick||xTick;
  var tcolor=teamColor(t);
  var s='<svg class="nt-chart" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+esc(cfg.aria||"Team scatter plot")+'">';
  [x0,(x0+x1)/2,x1].forEach(function(tv){{
    var x=X(tv);
    s+='<line x1="'+x.toFixed(1)+'" y1="'+mt+'" x2="'+x.toFixed(1)+'" y2="'+(mt+ph)+'" stroke="var(--border)" stroke-width="1" stroke-dasharray="2 3" opacity="0.6"/>';
    s+='<text x="'+x.toFixed(1)+'" y="'+(H-16)+'" font-size="10" text-anchor="middle" fill="var(--text-muted)">'+esc(xTick(tv))+'</text>';
  }});
  [y0,(y0+y1)/2,y1].forEach(function(tv){{
    var y=Y(tv);
    s+='<text x="'+(ml-6)+'" y="'+(y+3).toFixed(1)+'" font-size="10" text-anchor="end" fill="var(--text-muted)">'+esc(yTick(tv))+'</text>';
  }});
  var mx=X(med(xs)),my=Y(med(ys));
  s+='<line x1="'+mx.toFixed(1)+'" y1="'+mt+'" x2="'+mx.toFixed(1)+'" y2="'+(mt+ph)+'" stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>';
  s+='<line x1="'+ml+'" y1="'+my.toFixed(1)+'" x2="'+(ml+pw)+'" y2="'+my.toFixed(1)+'" stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>';
  /* Quadrant labels, split by the median on each axis. Drawn before the dots. */
  var qxL=ml+pw*0.25,qxR=ml+pw*0.75,qyT=mt+ph*0.22,qyB=mt+ph*0.78;
  var q=cfg.quads||[];
  [{{x:qxR,y:qyT,t:q[0]}},{{x:qxL,y:qyT,t:q[1]}},
   {{x:qxR,y:qyB,t:q[2]}},{{x:qxL,y:qyB,t:q[3]}}].forEach(function(qq){{
    if(!qq.t)return;
    s+='<text x="'+qq.x.toFixed(1)+'" y="'+qq.y.toFixed(1)+'" font-size="10" text-anchor="middle" fill="var(--text-muted)" opacity="0.85" letter-spacing="1">'+esc(qq.t)+'</text>';
  }});
  pts.forEach(function(p){{
    var x=X(p.x),y=Y(p.y),me=p.abbr===t.team;
    s+='<circle cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="'+(me?7:4.5)+'" fill="'+(me?tcolor:"var(--text-muted)")+'" '+
      (me?'stroke="#fff" stroke-width="1.5" ':'')+'fill-opacity="'+(me?1:0.55)+'"><title>'+
      esc(cfg.tip?cfg.tip(p):p.abbr)+'</title></circle>';
    if(me){{
      var lx=x+11>W-34?x-36:x+11;
      s+='<text x="'+lx.toFixed(1)+'" y="'+(y+4).toFixed(1)+'" font-size="12" font-weight="800" fill="'+tcolor+'">'+esc(p.abbr)+'</text>';
    }}
  }});
  s+='<text x="'+(ml+pw/2)+'" y="'+(H-2)+'" font-size="11" text-anchor="middle" fill="var(--text-muted)">'+esc(cfg.xLab||"")+'</text>';
  s+='<text x="10" y="'+(mt+ph/2)+'" font-size="11" text-anchor="middle" fill="var(--text-muted)" transform="rotate(-90 10 '+(mt+ph/2)+')">'+esc(cfg.yLab||"")+'</text>';
  return s+'</svg>';
}}
function olineScatterSVG(t){{
  var pts=((DATA&&DATA.teams)||[]).map(function(x){{
    var ol=x.oline||{{}};
    return {{abbr:x.team,x:ev(ol.pass_block),y:ev(ol.run_block)}};
  }});
  return scatterSVG(t,{{pts:pts,xLab:"Pass block grade",yLab:"Run block grade",
    quads:["Elite","Run-first","Pass-first","Rebuilding"],
    tip:function(p){{return p.abbr+": pass "+Math.round(p.x)+", run "+Math.round(p.y);}},
    aria:"Pass block vs run block grades, all teams",
    emptyMsg:"Not enough league O-line data."}});
}}
function pressureScatterSVG(t){{
  var pts=((DATA&&DATA.teams)||[]).map(function(x){{
    var ol=x.oline||{{}};
    return {{abbr:x.team,x:ev(ol.pressure_rate),y:ev(ol.sack_rate)}};
  }});
  var pct1=function(v){{return v.toFixed(1)+"%";}};
  return scatterSVG(t,{{pts:pts,xLab:"Pressure %",yLab:"Sack %",xTick:pct1,yTick:pct1,
    quads:["Under siege","Sitting ducks","Escape artists","Clean pocket"],
    tip:function(p){{return p.abbr+": pressure "+p.x.toFixed(1)+"%, sacks "+p.y.toFixed(1)+"%";}},
    aria:"Pressure rate vs sack rate, all teams",
    emptyMsg:"Not enough league O-line data."}});
}}
function identityScatterSVG(t){{
  var pts=((DATA&&DATA.teams)||[]).map(function(x){{
    var r=x.ranks||{{}};
    return {{abbr:x.team,x:ntPct(r.pass_yds_pg),y:ntPct(r.rush_yds_pg)}};
  }});
  return scatterSVG(t,{{pts:pts,xLab:"Pass yards percentile",yLab:"Rush yards percentile",
    quads:["Balanced","Run-first","Pass-first","One-dimensional"],
    tip:function(p){{return p.abbr+": pass "+Math.round(p.x)+"th pct, rush "+Math.round(p.y)+"th pct";}},
    aria:"Pass offense vs rush offense percentile, all teams",
    emptyMsg:"Not enough ranked team metrics yet."}});
}}
function softSpotsSVG(t){{
  if(!DPOS){{ensureDefense();return '<p class="nt-fine">Loading defensive matchup data.</p>';}}
  if(DPOS.failed||!DPOS.teams)return '<p class="nt-fine">Defensive matchup data is not available.</p>';
  var pts=Object.keys(DPOS.teams).map(function(ab){{
    var tp=DPOS.teams[ab]||{{}};
    return {{abbr:ab,x:ev(tp.RB),y:ev(tp.WR)}};
  }});
  var f1=function(v){{return v.toFixed(1);}};
  return scatterSVG(t,{{pts:pts,xLab:"FPTS allowed to RB",yLab:"FPTS allowed to WR",xTick:f1,yTick:f1,
    quads:["Soft","Air funnel","Ground funnel","Lockdown"],
    tip:function(p){{return p.abbr+": "+p.x.toFixed(1)+" to RB, "+p.y.toFixed(1)+" to WR";}},
    aria:"Fantasy points allowed to running backs vs wide receivers, all defenses",
    emptyMsg:"Not enough defensive matchup data."}});
}}

function backRow(){{
  return '<div class="nt-backrow"><button type="button" class="nt-back" id="ntBack">&larr; All teams</button></div>';
}}
function wireBack(){{
  var b=$("#ntBack");
  if(b&&!b.__wired){{b.__wired=true;b.addEventListener("click",function(){{update({{team:""}},true);scrollTop();}});}}
}}
function defenseSection(){{
  var h='<section class="nt-psec"><h3>Defense vs position</h3>';
  var dp=(DPOS&&DPOS.teams||{{}})[state.team];
  if(!DPOS){{ensureDefense();h+='<p class="nt-fine">Loading defensive matchup data.</p></section>';return h;}}
  if(DPOS.failed||!dp){{h+='<p class="nt-fine">Defensive matchup data is not available for this team and season.</p></section>';return h;}}
  var cells="";
  ["QB","RB","WR","TE"].forEach(function(p){{
    var c=dp[p];if(!c||c.v==null)return;
    var tier=easeTier(c.rank,c.total);
    var fg=tier?EASE_FG[tier]:"inherit",bg=tier?EASE_BG[tier]:null;
    var effTxt=c.eff!=null?'<div class="nt-fine">'+Number(c.eff).toFixed(1)+' '+esc(shortEff(c.eff_label))+'</div>':"";
    cells+='<div class="nt-def4cell"><div class="nt-def4pos">vs '+p+'</div>'+
      '<div class="nt-def4val"><b style="color:'+fg+'">'+Number(c.v).toFixed(1)+'</b></div>'+
      '<div class="nt-fine">FPTS/G</div>'+
      '<div>'+rankBadge(c.rank,bg?fg:null,bg)+'</div>'+effTxt+'</div>';
  }});
  h+='<div class="nt-def4">'+cells+'</div>';
  h+='<p class="nt-fine">Fantasy points allowed per game (PPR). Rank 1 = most allowed = easiest matchup. Only completed games count.</p></section>';
  return h;
}}
function renderProfile(){{
  var el=$("#ntProfile");
  if(!state.team){{el.innerHTML="";return;}}
  var t=(DATA&&DATA.teams||[]).filter(function(x){{return x.team===state.team;}})[0];
  if(!t){{el.innerHTML="";return;}}
  if(DETAIL_TEAM!==state.team||!DETAIL){{
    el.innerHTML=backRow()+'<div class="nt-pcard"><div class="nt-load">Loading '+esc(state.team)+' details.</div></div>';
    wireBack();
    var want=state.team,season=state.season;
    api("/api/nfl-team-details?team="+encodeURIComponent(want)+"&season="+encodeURIComponent(season))
      .then(function(d){{normOline(d.oline);DETAIL=d;DETAIL_TEAM=want;if(state.team===want)renderProfile();}})
      .catch(function(){{if(state.team===want){{el.innerHTML=backRow()+'<div class="nt-pcard"><div class="nt-err">Could not load team details.<br><button type="button" id="ntRetry">Retry</button></div></div>';wireBack();var rb=$("#ntRetry");if(rb)rb.addEventListener("click",function(){{DETAIL=null;renderProfile();}});}}}});
    return;
  }}
  var d=DETAIL;
  var nextG=null;(d.schedule||[]).forEach(function(g){{if(!nextG&&!g.bye&&g.status!=="final"&&g.status!=="live")nextG=g;}});
  var nextTxt=nextG?((nextG.is_home?"vs ":"at ")+nextG.opponent+(", "+(nextG.kickoff||nextG.date_label||"")).replace(/, $/,"")):"none remaining";
  var tcolor=teamColor(t);
  var rec={{w:0,l:0,t:0}};
  (d.schedule||[]).forEach(function(g){{
    if(g.bye||g.status!=="final")return;
    if(g.result==="W")rec.w++;else if(g.result==="L")rec.l++;else rec.t++;
  }});
  var recHtml=(rec.w+rec.l+rec.t)>0?'<span class="nt-record">'+esc(rec.w+"-"+rec.l+(rec.t?"-"+rec.t:""))+'</span>':"";
  var h=backRow()+'<div class="nt-pcard"><div class="nt-phead2 nt-herohead" style="background:color-mix(in srgb,'+esc(tcolor)+' 10%,transparent);box-shadow:inset 0 3px 0 '+esc(tcolor)+'">'+
    '<div class="nt-pid">'+logoHTML(t,true)+
    '<div><h2>'+esc(t.city)+' '+esc(t.name)+recHtml+'</h2><p class="nt-meta">'+esc(d.season_label||"")+' &middot; Bye week '+esc(String(t.bye_week==null?"?":t.bye_week))+' &middot; Next: '+esc(nextTxt)+'</p></div></div></div>';
  h+='<div class="nt-pbody"><div class="nt-pgrid">';
  h+='<section class="nt-psec"><h3>Offensive environment</h3>';
  envRows(t).forEach(function(e2){{
    var w=e2.rank?Math.max(4,Math.round((33-e2.rank)/32*100)):4;
    h+='<div class="nt-erow"><div class="nt-elab">'+esc(e2.label)+'<span class="nt-eval">'+esc(e2.val)+'</span></div>'+
      '<div class="nt-bar"><i style="width:'+w+'%;background:'+esc(tcolor)+'"></i></div><div class="nt-erk">'+ntEnvRank(e2.rank)+'</div></div>';
  }});
  h+='<p class="nt-fine">Volume and tendency ranks describe the offense, not player quality.</p></section>';
  var ol=d.oline;
  h+='<section class="nt-psec"><h3>Offensive line</h3>';
  if(!ol){{h+='<p class="nt-fine" style="margin:0 0 10px">Line ratings unavailable. Shown as N/A, no rank.</p>';}}
  else{{
    h+='<div class="nt-chips">';
    [["Overall","composite"],["Pass block","pass_block"],["Run block","run_block"]].forEach(function(c){{
      var cell=ol[c[1]];
      h+='<div class="nt-chip"><div class="nt-clab">'+c[0]+'</div><div class="nt-cval">'+(cell&&cell.v!=null?Math.round(cell.v):"N/A")+'</div>'+
        '<div class="nt-crk">'+(cell&&cell.rank?("#"+cell.rank+" &middot; "+ol.season+" season"):"no rank")+'</div></div>';
    }});
    h+='</div><details class="nt-method"><summary>Methodology and underlying metrics</summary>';
    [["Pressure rate","pressure_rate",true],["Sack rate","sack_rate",true],["Line yards","line_yards",false]].forEach(function(m){{
      var cell=ol[m[1]];var disp="N/A";
      if(cell&&cell.v!=null){{disp=m[1]==="line_yards"?Number(cell.v).toFixed(1):Number(cell.v).toFixed(1)+"%";}}
      h+='<div class="nt-mrow"><span>'+m[0]+'</span><span>'+(cell&&cell.rank?"<b>"+disp+"</b> #"+cell.rank+" ("+(m[2]?"lower is better":"higher is better")+")":disp)+'</span></div>';
    }});
    h+='<p class="nt-fine">Ratings are 0-100 unit scores from public play-by-play, not commercial blocker grades. O-line ratings use the '+ol.season+' season (latest available); all other metrics use the selected season.</p></details>';
  }}
  h+='</section>';
  h+='</div>';
  h+=defenseSection();
  /* Graphs live in one collapsible section below the depth chart so the
     profile stats stay up top. */
  h+='<section class="nt-psec"><h3>Depth chart / competition</h3><div class="nt-seg" role="group" aria-label="Position room">';
  ["QB","RB","WR","TE"].forEach(function(p){{
    h+='<button type="button" data-room="'+p+'" aria-pressed="'+(state.room===p)+'">'+p+'</button>';
  }});
  h+='</div><div class="nt-tscroll"><table class="nt-depth"><thead><tr><th scope="col">Player</th>';
  ROOMCOLS[state.room].forEach(function(c){{h+='<th scope="col" class="nt-num">'+c[1]+'</th>';}});
  h+='</tr></thead><tbody>';
  var room=((d.depth_chart||{{}})[state.room])||[];
  if(!room.length){{h+='<tr><td colspan="'+(ROOMCOLS[state.room].length+1)+'">No role data available.</td></tr>';}}
  room.forEach(function(p,i){{
    h+='<tr><td><span class="nt-dno">'+(p.order||i+1)+'</span><button type="button" class="nt-pname" data-pid="'+esc(p.id||"")+'" data-pname="'+esc(p.name||"")+'">'+esc(p.name||"")+'</button>'+(p.injury?'<span class="nt-inj">'+esc(p.injury)+'</span>':"")+'</td>';
    ROOMCOLS[state.room].forEach(function(c){{
      var v=p[c[0]];var txt="N/A";
      if(v!=null){{txt=(c[0]==="games")?String(v):(c[0]==="ppg"?Number(v).toFixed(1):v+"%");}}
      h+='<td class="nt-num">'+txt+'</td>';
    }});
    h+='</tr>';
  }});
  h+='</tbody></table></div><p class="nt-fine">'+esc(d.roster_note||"Current roster")+ (d.usage_note?(" "+esc(d.usage_note)):"") +' Tap a player to open their card.</p></section>';
  h+='<details class="nt-psec nt-collapse" open><summary><h3>Graphs</h3><span class="nt-chev" aria-hidden="true"></span></summary>';
  h+='<section class="nt-psec"><h3>Home vs away</h3>'+splitsSVG(d,t)+'</section>';
  h+=fingerprintSection(t);
  h+='<section class="nt-psec"><h3>Pass block vs run block</h3>'+olineScatterSVG(t)+'<p class="nt-fine">Every dot is an NFL team, the highlighted dot is this team. O-line grades use the '+esc(String((DATA&&DATA.oline_season)||"latest"))+' season (latest available).</p></section>';
  h+='<section class="nt-psec"><h3>Pressure vs sacks</h3>'+pressureScatterSVG(t)+'<p class="nt-fine">Every dot is an NFL team. Lower is better on both axes. Rates use the '+esc(String((DATA&&DATA.oline_season)||"latest"))+' season (latest available).</p></section>';
  h+='<section class="nt-psec"><h3>Pass vs run identity</h3>'+identityScatterSVG(t)+'<p class="nt-fine">League percentile by rank, all teams. Higher is better on both axes.</p></section>';
  h+='<section class="nt-psec"><h3>Defensive soft spots</h3>'+softSpotsSVG(t)+'<p class="nt-fine">Fantasy points allowed per game (PPR). Higher means a softer matchup. Only completed games count.</p></section>';
  h+='</details>';
  h+='<details class="nt-psec nt-collapse" open><summary><h3>Schedule</h3><span class="nt-chev" aria-hidden="true"></span></summary><div class="nt-sched">';
  (d.schedule||[]).forEach(function(g){{
    if(g.bye){{h+='<div class="nt-wrow nt-bye"><span class="nt-wk">'+esc(g.week_label||"")+'</span><span class="nt-opp">Bye week</span></div>';return;}}
    var key=state.team+"-"+g.week+"-"+(g.season_type||"reg");
    var oppAbbr=g.opponent||"";
    var oppLogo=g.opponent_logo?'<img class="nt-opp-logo" src="'+esc(g.opponent_logo)+'" alt="" loading="lazy" onerror="this.remove()">':"";
    if(g.status==="final"&&g.game_id&&g.expandable!==false){{
      var won=g.result==="W";
      h+='<button type="button" class="nt-wrow" data-w="'+esc(key)+'"><span class="nt-wk">'+esc(g.week_label||"")+'</span><span class="nt-opp">'+oppLogo+(g.is_home?"vs ":"at ")+esc(oppAbbr)+'</span><span class="nt-res '+(won?"nt-w":"nt-l")+'">'+(won?"W":"L")+" "+g.team_pts+"-"+g.opp_pts+'</span></button>';
      if(state.expanded[key]){{h+='<div class="nt-box" id="ntBox-'+esc(key)+'"><div class="nt-load">Loading box score.</div></div>';}}
    }}else if(g.status==="final"){{
      var won2=g.result==="W";
      h+='<div class="nt-wrow nt-bye"><span class="nt-wk">'+esc(g.week_label||"")+'</span><span class="nt-opp">'+oppLogo+(g.is_home?"vs ":"at ")+esc(oppAbbr)+'</span><span class="nt-res '+(won2?"nt-w":"nt-l")+'">'+(won2?"W":"L")+" "+g.team_pts+"-"+g.opp_pts+'</span></div>';
    }}else{{
      h+='<div class="nt-wrow nt-bye"><span class="nt-wk">'+esc(g.week_label||"")+'</span><span class="nt-opp">'+oppLogo+(g.is_home?"vs ":"at ")+esc(oppAbbr)+'</span><span class="nt-res">'+esc(g.kickoff||g.date_label||"")+'</span></div>';
    }}
  }});
  h+='</div><p class="nt-fine">Completed scores, kickoff times, and byes reuse the shared NFL game-data service.</p></details>';
  h+='</div></div>';
  el.innerHTML=h;
  wireBack();
  var cmpSel=el.querySelector("#ntCmpSel");
  if(cmpSel){{cmpSel.addEventListener("change",function(){{
    FPRINT_CMP={{team:state.team,abbr:cmpSel.value||null}};
    renderProfile();
  }});}}
  el.querySelectorAll(".nt-seg button").forEach(function(b){{
    b.addEventListener("click",function(){{update({{room:b.getAttribute("data-room")}},false);}});
  }});
  el.querySelectorAll(".nt-pname").forEach(function(b){{
    b.addEventListener("click",function(e){{
      e.stopPropagation();
      var pid=b.getAttribute("data-pid"),pname=b.getAttribute("data-pname");
      if(pid&&typeof window.openPlayerModal==="function"){{window.openPlayerModal(pid,pname,{{}});}}
    }});
  }});
  el.querySelectorAll(".nt-wrow[data-w]").forEach(function(b){{
    b.addEventListener("click",function(){{
      var key=b.getAttribute("data-w");
      if(isMobileBox()){{openBoxSheet(key);return;}}
      var ex=Object.assign({{}},state.expanded);
      if(ex[key])delete ex[key];else ex[key]=1;
      update({{expanded:ex}},false);
      if(ex[key])loadBox(key);
    }});
  }});
  Object.keys(state.expanded).forEach(function(key){{loadBox(key);}});
}}

var BOX_CACHE={{}};
function isMobileBox(){{return !!(window.matchMedia&&window.matchMedia("(max-width: 640px)").matches);}}
function findWkRow(key){{
  var parts=key.split("-");var abbr=parts[0];
  var wkRow=null;(DETAIL.schedule||[]).forEach(function(g){{if(abbr+"-"+g.week+"-"+(g.season_type||"reg")===key)wkRow=g;}});
  return {{abbr:abbr,wkRow:wkRow}};
}}
function fetchBox(key,ok,no){{
  if(BOX_CACHE[key]){{ok(BOX_CACHE[key],true);return;}}
  var found=findWkRow(key),abbr=found.abbr,wkRow=found.wkRow;
  if(!wkRow||!wkRow.game_id){{no();return;}}
  api("/api/player-team-boxscore?game_id="+encodeURIComponent(wkRow.game_id)+"&team="+encodeURIComponent(abbr))
    .then(function(bx){{BOX_CACHE[key]=bx;ok(bx,false);}})
    .catch(no);
}}
function wirePnameModal(host){{
  host.querySelectorAll(".nt-pname").forEach(function(b){{
    b.addEventListener("click",function(e){{
      e.stopPropagation();
      var pid=b.getAttribute("data-pid"),pname=b.getAttribute("data-pname");
      if(pid&&typeof window.openPlayerModal==="function"){{window.openPlayerModal(pid,pname,{{}});}}
    }});
  }});
}}
function loadBox(key){{
  var host=document.getElementById("ntBox-"+CSS.escape(key));
  if(!host||host.__done)return;host.__done=true;
  fetchBox(key,function(bx){{
    var found=findWkRow(key);
    host.innerHTML=boxHTML(bx,found.abbr,found.wkRow);
    wirePnameModal(host);
  }},function(){{host.innerHTML='<div class="nt-fine">Box score unavailable.</div>';host.__done=false;}});
}}
// ── Mobile box-score sheet (Redzone-style bottom sheet) ──────────────
// On phones the inline two-column box score becomes a bottom sheet showing
// one team at a time with a toggle, mirroring the Redzone box score sheet.
var _ntSheet=null,_ntSheetEls=null;
function _ntSheetStatus(bx){{
  if(!bx||bx.started===false)return "Not started";
  if(bx.status==="final")return "Final";
  if(bx.quarter)return "Q"+bx.quarter+(bx.clock?" "+bx.clock:"");
  return bx.status||"Live";
}}
function _ntSheetRender(){{
  if(!_ntSheet||!_ntSheetEls)return;
  var s=_ntSheet,bx=s.data;
  var home=(bx&&bx.home)||{{}},away=(bx&&bx.away)||{{}};
  var title="Box score",sub="Loading.";
  if(bx){{
    var ha=home.team||"",aa=away.team||"";
    if(aa||ha)title=(aa||"-")+" @ "+(ha||"-");
    var hp=home.pts,ap=away.pts;
    var score=(ap==null||hp==null)?"":ap+"-"+hp;
    sub=_ntSheetStatus(bx)+(score?" · "+score:"");
  }}else if(s.error){{sub="Unavailable";}}
  _ntSheetEls.title.textContent=title;
  _ntSheetEls.sub.textContent=sub;
  var body=_ntSheetEls.body;
  if(s.error||!bx){{body.innerHTML='<div class="nt-fine">'+esc((bx&&bx.message)||"Box score unavailable.")+'</div>';return;}}
  var teams=[away.team,home.team].filter(Boolean);
  var team=s.team&&bx.teams&&bx.teams[s.team]?s.team:teams[0];
  s.team=team;
  var toggle='<div class="nt-sheet-teams" role="group" aria-label="Team">'+teams.map(function(ab){{
    return '<button type="button" class="nt-sheet-team'+(ab===team?" is-on":"")+'" data-ntsteam="'+esc(ab)+'" aria-pressed="'+(ab===team)+'">'+esc(ab)+'</button>';
  }}).join("")+'</div>';
  body.innerHTML=toggle+boxHTML(bx,team,null,team);
  wirePnameModal(body);
}}
function _ntCloseSheet(){{
  if(_ntSheetEls){{
    if(_ntSheetEls.backdrop.parentNode)_ntSheetEls.backdrop.parentNode.removeChild(_ntSheetEls.backdrop);
    if(_ntSheetEls.sheet.parentNode)_ntSheetEls.sheet.parentNode.removeChild(_ntSheetEls.sheet);
    _ntSheetEls=null;
  }}
  _ntSheet=null;
  document.removeEventListener("keydown",_ntSheetKey);
  document.body.style.overflow="";
}}
function _ntSheetKey(e){{if(e&&e.key==="Escape")_ntCloseSheet();}}
function openBoxSheet(key){{
  _ntCloseSheet();
  var backdrop=document.createElement("div");backdrop.className="nt-sheet-backdrop";
  var sheet=document.createElement("div");sheet.className="nt-sheet";
  sheet.setAttribute("role","dialog");sheet.setAttribute("aria-modal","true");sheet.setAttribute("aria-label","Game box score");
  sheet.innerHTML='<div class="nt-sheet-handle" aria-hidden="true"></div>'
    +'<div class="nt-sheet-head"><div><div class="nt-sheet-title">Box score</div>'
    +'<div class="nt-sheet-sub">Loading.</div></div>'
    +'<button type="button" class="nt-sheet-x" data-ntsclose="1" aria-label="Close box score">✕</button></div>'
    +'<div class="nt-sheet-body"><div class="nt-load">Loading box score.</div></div>';
  document.body.appendChild(backdrop);document.body.appendChild(sheet);
  _ntSheetEls={{backdrop:backdrop,sheet:sheet,
    title:sheet.querySelector(".nt-sheet-title"),sub:sheet.querySelector(".nt-sheet-sub"),
    body:sheet.querySelector(".nt-sheet-body")}};
  _ntSheet={{key:key,team:"",data:null,error:false}};
  requestAnimationFrame(function(){{requestAnimationFrame(function(){{
    backdrop.classList.add("open");sheet.classList.add("open");
  }});}});
  document.body.style.overflow="hidden";
  document.addEventListener("keydown",_ntSheetKey);
  backdrop.addEventListener("click",_ntCloseSheet);
  sheet.addEventListener("click",function(e){{
    var t=e.target,closest=function(sel){{return t&&t.closest?t.closest(sel):null;}};
    if(closest("[data-ntsclose]")){{_ntCloseSheet();return;}}
    var tb=closest("[data-ntsteam]");
    if(tb&&_ntSheet){{_ntSheet.team=tb.getAttribute("data-ntsteam");_ntSheetRender();}}
  }});
  fetchBox(key,function(bx){{if(!_ntSheet||_ntSheet.key!==key)return;_ntSheet.data=bx;_ntSheetRender();}},
    function(){{if(!_ntSheet||_ntSheet.key!==key)return;_ntSheet.error=true;_ntSheetRender();}});
}}
function _boxCellEmpty(v){{
  return v===null||v===undefined||v===""||v===0||v==="0"||v==="0/0";
}}
function _boxRowEmpty(p,cols){{
  if(p.is_focus)return false;
  var cells=p.cells||{{}};
  for(var i=0;i<cols.length;i++){{if(!_boxCellEmpty(cells[cols[i].key]))return false;}}
  return true;
}}
function boxHTML(bx,abbr,wkRow,onlyAbbr){{
  if(!bx||bx.available===false){{return '<div class="nt-fine">'+esc((bx&&bx.message)||"Box score unavailable.")+'</div>';}}
  var home=bx.home||{{}},away=bx.away||{{}};
  var h='<div class="nt-boxscore">';
  if(!onlyAbbr){{
    var aPts=away.pts,hPts=home.pts;
    var aWin=aPts!=null&&hPts!=null&&Number(aPts)>Number(hPts);
    var hWin=aPts!=null&&hPts!=null&&Number(hPts)>Number(aPts);
    var stTxt=bx.status==="live"?((bx.quarter?("Q"+bx.quarter+(bx.clock?" "+bx.clock:"")):"Live")):(bx.status==="final"?"Final":"Scheduled");
    h+='<div class="nt-box-head">'
      +'<div class="nt-box-side">'+(away.logo?'<img class="nt-box-logo" src="'+esc(away.logo)+'" alt="" loading="lazy">':"")
      +'<span class="nt-box-abbr" title="'+esc(away.name||"")+'">'+esc(away.team||"")+'</span>'
      +'<span class="nt-box-score'+(aWin?"":" dim")+'">'+esc(aPts==null?"":aPts)+'</span></div>'
      +'<span class="nt-box-status">'+esc(stTxt)+'</span>'
      +'<div class="nt-box-side">'
      +'<span class="nt-box-score'+(hWin?"":" dim")+'">'+esc(hPts==null?"":hPts)+'</span>'
      +'<span class="nt-box-abbr" title="'+esc(home.name||"")+'">'+esc(home.team||"")+'</span>'
      +(home.logo?'<img class="nt-box-logo" src="'+esc(home.logo)+'" alt="" loading="lazy">':"")+'</div>'
      +'</div>';
  }}
  h+='<div class="nt-boxscore-teams">';
  var sides=[[home,away],[away,home]];
  sides.forEach(function(pair){{
    var info=pair[0];
    var ab=info.team||"";
    if(onlyAbbr&&ab!==onlyAbbr)return;
    var grp=((bx.teams&&bx.teams[ab]&&bx.teams[ab].groups)||[]).filter(function(g){{return g.pos!=="DEF"&&g.pos!=="ST";}});
    if(!grp.length)return;
    h+='<div class="nt-boxscore-team"><b>'+esc(info.name||ab)+'</b><div class="nt-tscroll"><table class="nt-depth">';
    grp.forEach(function(g){{
      var cols=g.columns||[];
      var vis=(g.players||[]).filter(function(p){{return !_boxRowEmpty(p,cols);}});
      if(!vis.length)return;
      h+='<tr><th colspan="'+(cols.length+1)+'">'+esc(g.pos||"")+'</th></tr>';
      h+='<tr><th scope="col">Player</th>'+cols.map(function(c){{return '<th scope="col" class="nt-stat">'+esc(c.label||c.key)+'</th>';}}).join("")+'</tr>';
      vis.forEach(function(p){{
        h+='<tr><td class="nt-pcell">'+(p.id?'<button type="button" class="nt-pname" data-pid="'+esc(p.id)+'" data-pname="'+esc(p.name)+'">'+esc(p.name)+'</button>':esc(p.name))+'</td>';
        cols.forEach(function(c){{var v=(p.cells||{{}})[c.key];h+='<td class="nt-stat">'+(v==null?"":esc(v))+'</td>';}});
        h+='</tr>';
      }});
    }});
    h+='</table></div></div>';
  }});
  h+='</div></div>';
  return h;
}}

function render(){{
  renderTabs();
  var inDetail=!!state.team;
  var list=$("#ntListWrap"),tabs=$("#ntTabs");
  if(list)list.style.display=inDetail?"none":"";
  if(tabs)tabs.style.display=inDetail?"none":"";
  if(!DATA){{loadRankings();return;}}
  if(inDetail){{renderProfile();return;}}
  if(state.view==="defense"){{
    if(!DPOS||(!DPOS.failed&&String(DPOS.season)!==String(state.season))){{ensureDefense();showDefenseLoading();return;}}
  }}
  renderTable();$("#ntProfile").innerHTML="";
}}
function showDefenseLoading(){{
  var tbl=$("#ntTbl");
  if(tbl)tbl.innerHTML="";
  $("#ntTableNote").textContent="";
  $("#ntSeasonSub").textContent="Team-level research, not fantasy roster rankings. "+(DATA.season_label||"");
  $("#ntProfile").innerHTML='<div class="nt-pcard"><div class="nt-load">Loading defensive matchup data.</div></div>';
}}
function scrollTop(){{try{{window.scrollTo(0,0);}}catch(e){{}}}}
function loadRankings(){{
  var tbl=$("#ntTbl");
  tbl.innerHTML="";$("#ntTableNote").textContent="";
  $("#ntProfile").innerHTML='<div class="nt-pcard"><div class="nt-load">Loading team data.</div></div>';
  api("/api/nfl-team-rankings?season="+encodeURIComponent(state.season))
    .then(function(d){{
      DATA=d;normRankings(DATA);DETAIL=null;DETAIL_TEAM=null;
      var seasons=d.available_seasons||[];
      var sel=$("#ntSeasonSel");
      if(sel&&seasons.length){{sel.innerHTML=seasons.map(function(s){{return '<option value="'+s+'"'+(String(s)===String(d.season)?" selected":"")+'>'+s+'</option>';}}).join("");}}
      render();
    }})
    .catch(function(){{$("#ntProfile").innerHTML='<div class="nt-pcard"><div class="nt-err">Could not load team data.<br><button type="button" id="ntRetryAll">Retry</button></div></div>';var rb=$("#ntRetryAll");if(rb)rb.addEventListener("click",loadRankings);}});
}}

function syncUrl(push){{
  var p="?season="+encodeURIComponent(state.season)+"&view="+encodeURIComponent(state.view);
  if(state.team)p+="&team="+encodeURIComponent(state.team);
  if(state.sortKey)p+="&sort="+encodeURIComponent(state.sortKey);
  if(state.sortDir)p+="&dir="+encodeURIComponent(state.sortDir);
  try{{
    var base=location.pathname||"/nfl-teams";
    if(push&&history.pushState)history.pushState(null,"",base+p);
    else if(history.replaceState)history.replaceState(null,"",base+p);
  }}catch(e){{}}
}}
function readUrl(){{
  var q=new URLSearchParams(location.search);
  var s=q.get("season");if(s)state.season=s;
  var v=q.get("view");if(v&&VIEWS[v])state.view=v;
  var t=(q.get("team")||"").toUpperCase();state.team=t;
  var view=VIEWS[state.view];
  var k=q.get("sort");var d=q.get("dir");
  if(k&&view.cols.some(function(c){{return c.k===k;}})){{state.sortKey=k;state.sortDir=(d==="asc"?"asc":"desc");}}
  else{{state.sortKey=view.def;state.sortDir=defaultDir(view.cols[0]);}}
  var r=q.get("room");if(r&&ROOMCOLS[r])state.room=r;
}}

function init(){{
  readUrl();
  var sel=$("#ntSeasonSel");
  if(sel){{sel.value=state.season;sel.addEventListener("change",function(){{DATA=null;DETAIL=null;DETAIL_TEAM=null;DPOS=null;update({{season:sel.value}},false);}});}}
  var how=$("#ntHowBtn");
  if(how){{how.addEventListener("click",function(){{
    var n=$("#ntTableNote");
    if(n)n.style.display=(n.style.display==="none")?"":"none";
  }});}}
  var csv=$("#ntCsvBtn");
  if(csv){{csv.addEventListener("click",downloadCsv);}}
  if(window.addEventListener){{window.addEventListener("popstate",function(){{readUrl();DETAIL=null;DETAIL_TEAM=null;render();}});}}
  render();
}}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",init);else init();
}})();
</script>
</div>
"""
