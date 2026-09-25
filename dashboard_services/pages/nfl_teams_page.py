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
.nt-chead h2{{font-size:18px;margin:0}}
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
.nt-chips{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px}}
.nt-chip{{background:var(--card-soft);border-radius:10px;padding:12px;text-align:center}}
.nt-chip .nt-clab{{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em}}
.nt-chip .nt-cval{{font-size:26px;font-weight:800;margin:2px 0}}
.nt-chip .nt-crk{{font-size:11px;color:var(--text-muted)}}
details.nt-method{{border:1px solid var(--border);border-radius:8px;padding:10px 12px;font-size:13px}}
details.nt-method summary{{cursor:pointer;font-weight:700;min-height:32px}}
.nt-mrow{{display:flex;justify-content:space-between;padding:6px 0;border-top:1px solid var(--border);font-size:13px}}
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
.nt-wrow .nt-res.nt-w{{color:var(--win);font-weight:700}}
.nt-wrow .nt-res.nt-l{{color:var(--loss);font-weight:700}}
.nt-wrow.nt-bye{{cursor:default;color:var(--text-muted)}}
.nt-box{{padding:4px 12px 12px;font-size:13px}}
.nt-boxscore-line{{font-size:15px;margin:8px 0 10px}}
.nt-boxscore-team{{margin-bottom:12px}}
.nt-boxscore-team table.nt-depth th[colspan]{{background:var(--card-soft)}}
.nt-err{{padding:24px 16px;text-align:center;color:var(--text-muted)}}
.nt-err button{{margin-top:10px}}
.nt-load{{padding:32px 16px;text-align:center;color:var(--text-muted)}}

</style>
<div class="card nt-card">
  <div class="card-header nt-chead">
    <div><h2>NFL Team Rankings</h2><p class="nt-sub" id="ntSeasonSub">Loading team data.</p></div>
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
  ["QB","RB","WR","TE"].forEach(function(p){{
    var c=dp[p];if(!c||c.v==null)return;
    var tier=easeTier(c.rank,c.total);
    var fg=tier?EASE_FG[tier]:"inherit",bg=tier?EASE_BG[tier]:null;
    var effTxt=c.eff!=null?'<span class="nt-fine"> '+Number(c.eff).toFixed(1)+' '+esc(shortEff(c.eff_label))+'</span>':"";
    h+='<div class="nt-mrow"><span>vs '+p+'</span><span><b style="color:'+fg+'">'+Number(c.v).toFixed(1)+' FPTS/G</b> '+
      rankBadge(c.rank,bg?fg:null,bg)+effTxt+'</span></div>';
  }});
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
  var h=backRow()+'<div class="nt-pcard"><div class="nt-phead2"><div class="nt-pid">'+logoHTML(t,true)+
    '<div><h2>'+esc(t.city)+' '+esc(t.name)+'</h2><p class="nt-meta">'+esc(d.season_label||"")+' &middot; Bye week '+esc(String(t.bye_week==null?"?":t.bye_week))+' &middot; Next: '+esc(nextTxt)+'</p></div></div></div>';
  h+='<div class="nt-pbody"><div class="nt-pgrid">';
  h+='<section class="nt-psec"><h3>Offensive environment</h3>';
  var tcolor=teamColor(t);
  envRows(t).forEach(function(e2){{
    var w=e2.rank?Math.max(4,Math.round((33-e2.rank)/32*100)):4;
    h+='<div class="nt-erow"><div class="nt-elab">'+esc(e2.label)+'<span class="nt-eval">'+esc(e2.val)+'</span></div>'+
      '<div class="nt-bar"><i style="width:'+w+'%;background:'+esc(tcolor)+'"></i></div><div class="nt-erk">'+(e2.rank?"#"+e2.rank:"N/A")+'</div></div>';
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
  h+='</section></div>';
  h+=defenseSection();
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
  h+='<section class="nt-psec"><h3>Schedule</h3><div class="nt-sched">';
  (d.schedule||[]).forEach(function(g){{
    if(g.bye){{h+='<div class="nt-wrow nt-bye"><span class="nt-wk">'+esc(g.week_label||"")+'</span><span class="nt-opp">Bye week</span></div>';return;}}
    var key=state.team+"-"+g.week+"-"+(g.season_type||"reg");
    var oppAbbr=g.opponent||"";
    if(g.status==="final"&&g.game_id&&g.expandable!==false){{
      var won=g.result==="W";
      h+='<button type="button" class="nt-wrow" data-w="'+esc(key)+'"><span class="nt-wk">'+esc(g.week_label||"")+'</span><span class="nt-opp">'+(g.is_home?"vs ":"at ")+esc(oppAbbr)+'</span><span class="nt-res '+(won?"nt-w":"nt-l")+'">'+(won?"W":"L")+" "+g.team_pts+"-"+g.opp_pts+'</span></button>';
      if(state.expanded[key]){{h+='<div class="nt-box" id="ntBox-'+esc(key)+'"><div class="nt-load">Loading box score.</div></div>';}}
    }}else if(g.status==="final"){{
      var won2=g.result==="W";
      h+='<div class="nt-wrow nt-bye"><span class="nt-wk">'+esc(g.week_label||"")+'</span><span class="nt-opp">'+(g.is_home?"vs ":"at ")+esc(oppAbbr)+'</span><span class="nt-res '+(won2?"nt-w":"nt-l")+'">'+(won2?"W":"L")+" "+g.team_pts+"-"+g.opp_pts+'</span></div>';
    }}else{{
      h+='<div class="nt-wrow nt-bye"><span class="nt-wk">'+esc(g.week_label||"")+'</span><span class="nt-opp">'+(g.is_home?"vs ":"at ")+esc(oppAbbr)+'</span><span class="nt-res">'+esc(g.kickoff||g.date_label||"")+'</span></div>';
    }}
  }});
  h+='</div><p class="nt-fine">Completed scores, kickoff times, and byes reuse the shared NFL game-data service.</p></section>';
  h+='</div></div>';
  el.innerHTML=h;
  wireBack();
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
      var ex=Object.assign({{}},state.expanded);
      if(ex[key])delete ex[key];else ex[key]=1;
      update({{expanded:ex}},false);
      if(ex[key])loadBox(key);
    }});
  }});
  Object.keys(state.expanded).forEach(function(key){{loadBox(key);}});
}}

var BOX_CACHE={{}};
function loadBox(key){{
  var host=document.getElementById("ntBox-"+CSS.escape(key));
  if(!host||host.__done)return;host.__done=true;
  if(BOX_CACHE[key]){{host.innerHTML=BOX_CACHE[key];return;}}
  var parts=key.split("-");var abbr=parts[0];
  var wkRow=null;(DETAIL.schedule||[]).forEach(function(g){{if(abbr+"-"+g.week+"-"+(g.season_type||"reg")===key)wkRow=g;}});
  if(!wkRow||!wkRow.game_id){{host.innerHTML='<div class="nt-fine">Box score unavailable.</div>';return;}}
  api("/api/player-team-boxscore?game_id="+encodeURIComponent(wkRow.game_id)+"&team="+encodeURIComponent(abbr))
    .then(function(bx){{
      var html=boxHTML(bx,abbr,wkRow);
      BOX_CACHE[key]=html;host.innerHTML=html;
      host.querySelectorAll(".nt-pname").forEach(function(b){{
        b.addEventListener("click",function(e){{
          e.stopPropagation();
          var pid=b.getAttribute("data-pid"),pname=b.getAttribute("data-pname");
          if(pid&&typeof window.openPlayerModal==="function"){{window.openPlayerModal(pid,pname,{{}});}}
        }});
      }});
    }})
    .catch(function(){{host.innerHTML='<div class="nt-fine">Box score unavailable.</div>';host.__done=false;}});
}}
function boxHTML(bx,abbr,wkRow){{
  if(!bx||bx.available===false){{return '<div class="nt-fine">'+esc((bx&&bx.message)||"Box score unavailable.")+'</div>';}}
  var home=bx.home||{{}},away=bx.away||{{}};
  var h='<div class="nt-boxscore"><div class="nt-boxscore-line"><b>'+esc(home.name||"")+'</b> '+esc(home.pts==null?"":home.pts)+
    ' &middot; <b>'+esc(away.name||"")+'</b> '+esc(away.pts==null?"":away.pts)+
    (bx.quarter?(' &middot; '+esc("Q"+bx.quarter+(bx.clock?" "+bx.clock:""))):"")+'</div>';
  var sides=[[home,away],[away,home]];
  sides.forEach(function(pair){{
    var info=pair[0];
    var ab=info.team||"";
    var grp=(bx.teams&&bx.teams[ab]&&bx.teams[ab].groups)||[];
    if(!grp.length)return;
    h+='<div class="nt-boxscore-team"><b>'+esc(info.name||ab)+'</b><div class="nt-tscroll"><table class="nt-depth">';
    grp.forEach(function(g){{
      var cols=g.columns||[];
      h+='<tr><th colspan="'+(cols.length+1)+'">'+esc(g.pos||"")+'</th></tr>';
      h+='<tr><th scope="col">Player</th>'+cols.map(function(c){{return '<th scope="col" class="nt-num">'+esc(c.label||c.key)+'</th>';}}).join("")+'</tr>';
      (g.players||[]).forEach(function(p){{
        h+='<tr><td>'+(p.id?'<button type="button" class="nt-pname" data-pid="'+esc(p.id)+'" data-pname="'+esc(p.name)+'">'+esc(p.name)+'</button>':esc(p.name))+'</td>';
        cols.forEach(function(c){{var v=(p.cells||{{}})[c.key];h+='<td class="nt-num">'+(v==null?"":esc(v))+'</td>';}});
        h+='</tr>';
      }});
    }});
    h+='</table></div></div>';
  }});
  h+='</div>';
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
