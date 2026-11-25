"""Autopilot page body: the real Phase 1 page.

Renders the acting account's real Autopilot state for one league:
the stored mode and guardrail settings (autopilot_store), the live
plan's swap groups (autopilot_plan, built from the site's own start
score engines against a fresh ESPN roster read), the real roster in
the protected-players editor, and the append-only audit log as the
activity feed. Actions post to routes/autopilot_bp.py; execution runs
through autopilot_execute with every guardrail re-checked live.

The page has no header block: it starts with the hero card (Kaedon,
2026-11-25, from his phone screenshot of the earlier mock). Access is
gated by autopilot_access in the route before this builder runs.
"""
from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

_CSS = """
.ap-wrap{max-width:880px;margin:0 auto;padding:6px 2px 96px}
.ap-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
box-shadow:0 1px 3px rgba(18,45,75,.07);padding:12px;margin-bottom:12px}
:root[data-theme='dark'] .ap-card{box-shadow:0 2px 10px rgba(0,0,0,.35)}
.ap-h2{font-size:16px;font-weight:800;color:var(--text);margin:0 0 2px}
.ap-card-sub{font-size:12.5px;color:var(--text-muted);margin:0 0 8px}
.ap-hero-head{display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap}
.ap-hero-title{font-size:19px;font-weight:900;color:var(--text);margin:0}
.ap-hero-sub{font-size:13px;color:var(--text-muted);margin:4px 0 0;max-width:520px}
.ap-pill{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:700;
color:var(--accent);background:var(--accent-soft);border-radius:999px;padding:4px 10px;white-space:nowrap}
.ap-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);flex:0 0 auto}
.ap-banner{font-size:13px;font-weight:700;color:var(--text);background:var(--accent-soft);
border-radius:10px;padding:10px 12px;margin-top:12px}
.ap-banner-warn{background:var(--row);border:1px solid var(--grid)}
.ap-seg{position:relative;overflow:hidden;display:flex;background:var(--row);border:1px solid var(--grid);
border-radius:10px;padding:3px;gap:3px;margin:14px 0 12px;max-width:420px}
.ap-seg button{flex:1;border:0;background:transparent;color:var(--text-muted);font:inherit;
font-size:13.5px;font-weight:700;padding:7px 0;border-radius:8px;cursor:pointer;
transition:background .2s ease,color .2s ease}
.ap-seg button.ap-on{background:var(--accent);color:var(--on-accent)}
.ap-seg.ap-arming::after{content:"";position:absolute;inset:0;pointer-events:none;
background:linear-gradient(100deg,transparent 20%,color-mix(in srgb,var(--accent) 28%,transparent) 50%,transparent 80%);
transform:translateX(-100%);animation:apSweep .38s ease-out forwards}
@keyframes apSweep{to{transform:translateX(100%)}}
.ap-hero{transition:box-shadow .3s ease}
.ap-hero.ap-arming{animation:apArmPulse .38s ease-out}
@keyframes apArmPulse{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--accent) 45%,transparent)}
100%{box-shadow:0 0 0 14px transparent}}
.ap-quiet{font-size:13.5px;color:var(--text-muted);background:var(--row);border:1px solid var(--grid);
border-radius:10px;padding:12px 14px}
.ap-quiet b{color:var(--text)}
.ap-error{font-size:13.5px;color:var(--text);background:var(--row);border:1px solid var(--grid);
border-radius:10px;padding:12px 14px}
.ap-move{position:relative;padding:12px 0 10px;border-top:1px solid var(--grid)}
.ap-move::before{content:"";position:absolute;left:-12px;top:0;bottom:0;width:3px;
background:var(--win);transform:scaleY(0);transition:transform .3s ease}
.ap-move.ap-done::before{transform:scaleY(1)}
.ap-move.ap-failed::before{background:var(--loss,var(--text-subtle))}
.ap-move-top{display:flex;align-items:flex-start;gap:10px}
.ap-tag{font-size:10px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;color:var(--accent);
background:var(--accent-soft);border-radius:6px;padding:3px 7px;flex:0 0 auto;margin-top:2px}
.ap-move-main{flex:1;min-width:0}
.ap-move-title{font-size:14.5px;font-weight:700;color:var(--text);line-height:1.3}
.ap-move-why{font-size:12.5px;color:var(--text-muted);margin-top:2px;line-height:1.35}
.ap-move-actions{display:flex;align-items:center;gap:10px;margin-top:9px}
.ap-delta{font-size:15px;font-weight:800;color:var(--win);flex:0 0 auto}
.ap-delta-neg{color:var(--text-muted)}
.ap-move-actions .ap-spacer{flex:1}
.ap-dismiss{border:0;background:transparent;color:var(--text-subtle);font:inherit;font-size:16px;
line-height:1;cursor:pointer;padding:4px 5px;flex:0 0 auto}
.ap-wrap [hidden]{display:none!important}
.ap-btn{display:inline-flex;align-items:center;justify-content:center;height:32px;padding:0 14px;
background:var(--accent);color:var(--on-accent);border:1px solid var(--accent);font:inherit;
font-size:13px;font-weight:700;border-radius:var(--radius-pill);cursor:pointer;white-space:nowrap;
transition:opacity .2s ease,background .2s ease}
.ap-btn[disabled]{opacity:.55;cursor:default}
.ap-btn-ghost{background:var(--card);color:var(--text);border-color:var(--border)}
.ap-actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:12px}
.ap-flash{font-size:12.5px;color:var(--accent);font-weight:600}
.ap-status{font-size:10.5px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;
border-radius:6px;padding:3px 7px;flex:0 0 auto;display:inline-flex;align-items:center;gap:5px}
.ap-st-ok{color:var(--win);background:var(--row)}
.ap-st-fail{color:var(--text);background:var(--row)}
.ap-st-skip{color:var(--text-muted);background:var(--row)}
.ap-st-queue{color:var(--accent);background:var(--accent-soft)}
.ap-check{display:inline-block;animation:apCheckIn .3s ease-out}
@keyframes apCheckIn{0%{opacity:0;transform:scale(.5)}100%{opacity:1;transform:scale(1)}}
.ap-autonext{background:var(--accent-soft);border-radius:10px;padding:12px 14px;margin-bottom:6px}
.ap-autonext-lab{font-size:10.5px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:var(--accent)}
.ap-autonext-main{font-size:15px;font-weight:800;color:var(--text);margin-top:2px}
.ap-autonext-sub{font-size:12.5px;color:var(--text-muted);margin-top:2px}
.ap-sumrow{display:flex;justify-content:space-between;gap:14px;padding:8px 0;border-top:1px solid var(--grid);
font-size:13px;color:var(--text-muted)}
.ap-sumrow b{color:var(--text);text-align:right}
.ap-rowline{display:flex;align-items:center;justify-content:space-between;gap:14px;
padding:9px 0;border-top:1px solid var(--grid)}
.ap-rowline:first-of-type{border-top:none}
.ap-lab{font-size:13.5px;font-weight:700;color:var(--text)}
.ap-lab small{display:block;font-size:12px;font-weight:400;color:var(--text-muted);margin-top:1px}
.ap-chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.ap-chip{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;font-weight:700;color:var(--text);
background:var(--row);border:1px solid var(--grid);border-radius:999px;padding:5px 8px 5px 11px}
.ap-chip button{border:0;background:transparent;color:var(--text-subtle);font-size:14px;cursor:pointer;
padding:0 2px;line-height:1}
.ap-roster{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.ap-roster button{font:inherit;font-size:12.5px;font-weight:600;color:var(--text-muted);
background:var(--card);border:1px solid var(--border);border-radius:999px;padding:5px 11px;cursor:pointer}
.ap-roster button:hover{color:var(--accent);border-color:var(--accent)}
.ap-note{font-size:12.5px;color:var(--text-muted);border-top:1px solid var(--grid);
margin:14px 0 0;padding-top:10px}
.ap-grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px 26px}
@media(max-width:700px){.ap-grid2{grid-template-columns:1fr}}
.ap-run{display:flex;gap:10px;padding:8px 0;border-top:1px solid var(--grid);font-size:13px}
.ap-run:first-of-type{border-top:none}
.ap-when{font-weight:800;color:var(--text);flex:0 0 86px}
.ap-what{color:var(--text-muted)}
.ap-what b{color:var(--text)}
.ap-log{display:flex;gap:10px;padding:9px 0;border-top:1px solid var(--grid);align-items:flex-start}
.ap-log:first-of-type{border-top:none}
.ap-log-ic{flex:0 0 auto;margin-top:1px}
.ap-log-ok{color:var(--win)}
.ap-log-fail{color:var(--text)}
.ap-log-skip{color:var(--text-subtle)}
.ap-log-queue{color:var(--accent)}
.ap-log-main{flex:1;font-size:13px;color:var(--text);min-width:0}
.ap-log-main small{display:block;font-size:12px;color:var(--text-muted);margin-top:1px}
@media (prefers-reduced-motion: reduce){
.ap-wrap *, .ap-wrap *::before, .ap-wrap *::after{animation:none!important;transition:none!important}}
"""

_SLOT_NAMES = {
    0: "QB", 2: "RB", 3: "RB/WR", 4: "WR", 5: "WR/TE", 6: "TE",
    7: "Superflex", 16: "DEF", 17: "K", 20: "Bench", 21: "IR", 23: "FLEX",
}

_CHECK_SVG = (
    '<svg class="ap-check" width="12" height="12" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="3.2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M4.5 12.5l5 5L19.5 7"/></svg>'
)


def _slot_name(slot_id) -> str:
    try:
        return _SLOT_NAMES.get(int(slot_id), f"slot {slot_id}")
    except (TypeError, ValueError):
        return ""


def _fmt_score(value) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return ""


def _fmt_delta(value) -> str:
    try:
        return f"{float(value):+.1f}"
    except (TypeError, ValueError):
        return ""


def _swap_title(swap: dict) -> str:
    ins = swap.get("in") or []
    outs = swap.get("out") or []

    def label(entry):
        pos = f" ({entry['pos']})" if entry.get("pos") else ""
        return f"{entry.get('name') or 'Player'}{pos}"

    if len(ins) == 1 and len(outs) == 1:
        slot = _slot_name(ins[0].get("to_slot"))
        return f"Start {label(ins[0])} at {slot} over {label(outs[0])}"
    if ins and outs:
        return (
            "Start " + ", ".join(label(x) for x in ins)
            + " over " + ", ".join(label(x) for x in outs)
        )
    if ins:
        slot = _slot_name(ins[0].get("to_slot"))
        return f"Start {label(ins[0])} at {slot}"
    if outs:
        return "Bench " + ", ".join(label(x) for x in outs)
    return "Lineup move"


def _swap_why(swap: dict, issues_by_canon: Dict[str, dict]) -> str:
    ins = swap.get("in") or []
    outs = swap.get("out") or []
    parts: List[str] = []
    if len(ins) == 1 and len(outs) == 1:
        si, so = _fmt_score(ins[0].get("score")), _fmt_score(outs[0].get("score"))
        if si and so:
            parts.append(f"Start score <b>{si}</b> vs {so}")
    elif ins:
        parts.append("Raises the lineup's blended start score")
    for entry in list(outs) + list(ins):
        issue = issues_by_canon.get(str(entry.get("canon_id") or ""))
        if issue and issue.get("detail"):
            parts.append(html.escape(str(issue["detail"])))
            break
    return " &middot; ".join(parts)


def _audit_icon(action: str) -> str:
    if action == "executed":
        return (
            '<svg class="ap-log-ic ap-log-ok" width="18" height="18" viewBox="0 0 24 24" '
            'fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" '
            'r="10" stroke-opacity=".35"/><path d="M8 12.5l2.6 2.6L16 9.5"/></svg>'
        )
    if action == "failed":
        return (
            '<svg class="ap-log-ic ap-log-fail" width="18" height="18" viewBox="0 0 24 24" '
            'fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" '
            'r="10" stroke-opacity=".4"/><path d="M12 7v6M12 16.5v.5"/></svg>'
        )
    if action == "skipped":
        return (
            '<svg class="ap-log-ic ap-log-skip" width="18" height="18" viewBox="0 0 24 24" '
            'fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" '
            'r="10" stroke-opacity=".4"/><path d="M8 8l8 8M16 8l-8 8"/></svg>'
        )
    return (
        '<svg class="ap-log-ic ap-log-queue" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" '
        'r="10" stroke-opacity=".4"/><path d="M12 7v5l3.5 2"/></svg>'
    )


_AUDIT_LABEL = {
    "executed": ("Executed", "ap-st-ok"),
    "planned": ("Planned", "ap-st-queue"),
    "approved": ("Approved", "ap-st-queue"),
    "skipped": ("Skipped", "ap-st-skip"),
    "failed": ("Failed", "ap-st-fail"),
}


def _fmt_time(value) -> str:
    try:
        return value.strftime("%b %-d, %-I:%M %p")
    except Exception:
        return str(value or "")[:16]


_JS = """
<script>
(function(){
  var root=document.getElementById('apRoot'); if(!root){return;}
  var PLATFORM=root.getAttribute('data-ap-platform'),SEASON=root.getAttribute('data-ap-season'),
      LEAGUE=root.getAttribute('data-ap-league');
  var base='/api/autopilot/'+PLATFORM+'/'+SEASON+'/'+LEAGUE;
  var hero=document.getElementById('apHero');
  function post(url,body){
    return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body||{})}).then(function(r){
        return r.json().catch(function(){return{};}).then(function(d){return {ok:r.ok,data:d};});
      });
  }
  function flash(msg){
    root.querySelectorAll('.ap-flash').forEach(function(t){t.textContent=msg||'';});
  }
  // --- Mode segmented control -------------------------------------------
  var seg=document.getElementById('apSeg');
  if(seg){
    seg.querySelectorAll('[data-ap-mode]').forEach(function(b){
      b.addEventListener('click',function(){
        var m=b.getAttribute('data-ap-mode');
        if(b.classList.contains('ap-on')){return;}
        if(m==='auto'){seg.classList.add('ap-arming'); if(hero){hero.classList.add('ap-arming');}}
        b.disabled=true;
        post(base+'/mode',{mode:m}).then(function(res){
          if(res.ok){setTimeout(function(){location.reload();}, m==='auto'?380:60);}
          else{seg.classList.remove('ap-arming'); if(hero){hero.classList.remove('ap-arming');}
            b.disabled=false; flash('Could not change the mode. Try again.');}
        }).catch(function(){seg.classList.remove('ap-arming'); b.disabled=false;});
      });
    });
  }
  // --- Guardrail switches -------------------------------------------------
  root.querySelectorAll('[data-ap-setting]').forEach(function(input){
    input.addEventListener('change',function(){
      var key=input.getAttribute('data-ap-setting'), body={}; body[key]=input.checked;
      post(base+'/settings',body).then(function(res){
        if(!res.ok){input.checked=!input.checked; flash('Could not save that setting.');}
      }).catch(function(){input.checked=!input.checked;});
    });
  });
  // --- Pause / resume -------------------------------------------------------
  root.querySelectorAll('[data-ap-pause]').forEach(function(b){
    b.addEventListener('click',function(){
      var paused=b.getAttribute('data-ap-pause')==='1';
      post(base+'/settings',{paused:paused}).then(function(res){
        if(res.ok){location.reload();} else {flash('Could not change the pause state.');}
      });
    });
  });
  // --- Account kill switch --------------------------------------------------
  var kill=document.getElementById('apKillSwitch');
  if(kill){
    kill.addEventListener('change',function(){
      post('/api/autopilot/kill-switch',{on:kill.checked}).then(function(res){
        if(res.ok){location.reload();} else {kill.checked=!kill.checked;}
      }).catch(function(){kill.checked=!kill.checked;});
    });
  }
  // --- Queue: accept / dismiss ---------------------------------------------
  var queue=document.getElementById('apQueue');
  function runGroups(indexes, rows){
    if(!queue||!indexes.length){return;}
    var planId=queue.getAttribute('data-plan-id');
    rows.forEach(function(r){r.classList.add('ap-running');
      r.querySelectorAll('button').forEach(function(x){x.disabled=true;});});
    post(base+'/execute',{plan_id:parseInt(planId,10),groups:indexes}).then(function(res){
      var result=(res.data&&res.data.result)||{};
      var byIndex={};
      (result.groups||[]).forEach(function(g){byIndex[g.index]=g;});
      var settled=false;
      rows.forEach(function(r){
        var g=byIndex[parseInt(r.getAttribute('data-ap-group'),10)];
        var out=r.querySelector('[data-ap-outcome]'); if(!out){return;}
        out.hidden=false; settled=true;
        if(g&&(g.outcome==='executed'||g.outcome==='already_applied')){
          r.classList.add('ap-done');
          out.className='ap-status ap-st-ok';
          out.innerHTML=__AP_CHECK__+(g.outcome==='executed'?'Executed &middot; Verified':'Already in place');
        } else {
          r.classList.add('ap-failed');
          out.className='ap-status ap-st-fail';
          out.textContent='Failed: '+((g&&g.detail)||result.detail||'The move did not run.');
        }
      });
      if(result.status==='superseded'){
        flash('Your roster changed since this plan was built. Refreshing with a fresh plan.');
      }
      if(settled||result.status){setTimeout(function(){location.reload();},1800);}
    }).catch(function(){flash('The run did not complete. Refresh and try again.');});
  }
  if(queue){
    queue.querySelectorAll('[data-ap-accept]').forEach(function(b){
      b.addEventListener('click',function(){
        var row=b.closest('.ap-move'); if(!row){return;}
        runGroups([parseInt(row.getAttribute('data-ap-group'),10)],[row]);
      });
    });
    queue.querySelectorAll('[data-ap-dismiss]').forEach(function(b){
      b.addEventListener('click',function(){
        var row=b.closest('.ap-move'); if(row){row.hidden=true;}
        flash('Move dismissed. It stays unexecuted; running the rest logs it as skipped.');
      });
    });
    var all=document.getElementById('apAcceptAll');
    if(all){all.addEventListener('click',function(){
      var rows=[].slice.call(queue.querySelectorAll('.ap-move')).filter(function(r){return !r.hidden;});
      runGroups(rows.map(function(r){return parseInt(r.getAttribute('data-ap-group'),10);}),rows);
    });}
    var replan=document.getElementById('apReplan');
    if(replan){replan.addEventListener('click',function(){
      replan.disabled=true;
      post(base+'/replan',{}).then(function(){location.reload();})
        .catch(function(){replan.disabled=false;});
    });}
  }
  // --- Protected players ------------------------------------------------------
  var roster={};
  try{roster=JSON.parse(root.getAttribute('data-ap-roster')||'{}');}catch(e){roster={};}
  var protectedIds=[];
  try{protectedIds=JSON.parse(root.getAttribute('data-ap-protected')||'[]');}catch(e){protectedIds=[];}
  var chipsBox=document.getElementById('apProtectedChips');
  var rosterBox=document.getElementById('apRosterList');
  function renderProtected(){
    if(!chipsBox||!rosterBox){return;}
    chipsBox.innerHTML='';
    if(!protectedIds.length){
      chipsBox.innerHTML='<span class="ap-card-sub" style="margin:0">No protected players yet.</span>';
    }
    protectedIds.forEach(function(pid){
      var info=roster[String(pid)]||{};
      var chip=document.createElement('span'); chip.className='ap-chip';
      chip.textContent=(info.name||('Player '+pid))+' ';
      var x=document.createElement('button'); x.type='button'; x.textContent='\\u00d7';
      x.setAttribute('aria-label','Remove protection');
      x.addEventListener('click',function(){
        saveProtected(protectedIds.filter(function(p){return p!==pid;}));});
      chip.appendChild(x); chipsBox.appendChild(chip);
    });
    rosterBox.innerHTML='';
    Object.keys(roster).forEach(function(pid){
      if(protectedIds.indexOf(parseInt(pid,10))>=0){return;}
      var info=roster[pid];
      var b=document.createElement('button'); b.type='button';
      b.textContent='+ '+(info.name||('Player '+pid));
      b.addEventListener('click',function(){
        saveProtected(protectedIds.concat([parseInt(pid,10)]));});
      rosterBox.appendChild(b);
    });
  }
  function saveProtected(ids){
    post(base+'/settings',{protected_player_ids:ids}).then(function(res){
      if(res.ok){protectedIds=ids; renderProtected();}
      else{flash('Could not save the protected list.');}
    });
  }
  renderProtected();
})();
</script>
"""


def build_autopilot_body(platform: str, season: int, league_id: str,
                         ctx: dict | None = None,
                         data: dict | None = None) -> str:
    """Render the real Autopilot page from route-gathered state.

    ``data`` keys (all optional; the route fills them from the store,
    the plan builder, and a fresh roster read): settings, kill_switch,
    plan, plan_status, plan_error, roster (espn_writes entries),
    team_name, scoring_period, audit.
    """
    ctx = ctx or {}
    data = data or {}
    league = ctx.get("league") or {}
    viewer = ctx.get("viewer") or {}
    league_name = str(data.get("league_name") or league.get("name") or "").strip()
    team_name = str(
        data.get("team_name") or viewer.get("viewer_team_name") or ""
    ).strip()
    settings: Dict[str, Any] = data.get("settings") or {}
    mode = str(settings.get("mode") or "off")
    paused = bool(settings.get("paused"))
    kill = bool(data.get("kill_switch"))
    plan: Optional[dict] = data.get("plan")
    plan_status = str(data.get("plan_status") or "")
    plan_error = data.get("plan_error")
    roster: List[dict] = data.get("roster") or []
    audit: List[dict] = data.get("audit") or []
    period = data.get("scoring_period") or (plan or {}).get("scoring_period")
    week_label = f"Week {int(period)}" if period else "this week"

    protected_ids = [int(p) for p in (settings.get("protected_player_ids") or [])]
    roster_map = {
        str(e["player_id"]): {
            "name": e.get("name") or f"Player {e['player_id']}",
            "pos": e.get("position") or "",
            "slot": e.get("slot"),
        }
        for e in roster
    }
    issues_by_canon: Dict[str, dict] = {}
    for issue in (plan or {}).get("issues") or []:
        if issue.get("pid") is not None:
            issues_by_canon[str(issue["pid"])] = issue

    e = html.escape
    swaps = (plan or {}).get("swaps") or []

    # --- Queue (Approve) ------------------------------------------------
    queue_rows = ""
    for i, swap in enumerate(swaps):
        delta = swap.get("delta")
        delta_cls = "ap-delta" + (" ap-delta-neg" if (delta or 0) < 0 else "")
        queue_rows += f"""
      <div class="ap-move" data-ap-group="{i}">
        <div class="ap-move-top">
          <span class="ap-tag">Lineup</span>
          <div class="ap-move-main">
            <div class="ap-move-title">{e(_swap_title(swap))}</div>
            <div class="ap-move-why">{_swap_why(swap, issues_by_canon)}</div>
          </div>
        </div>
        <div class="ap-move-actions">
          <span class="{delta_cls}">{e(_fmt_delta(delta))}</span>
          <span class="ap-spacer"></span>
          <span class="ap-status ap-st-ok" data-ap-outcome hidden></span>
          <button class="ap-btn" type="button" data-ap-accept>Accept</button>
          <button class="ap-dismiss" type="button" data-ap-dismiss aria-label="Dismiss this move">&times;</button>
        </div>
      </div>"""
    plan_id_attr = f' data-plan-id="{int(plan["id"])}"' if plan and plan.get("id") else ""
    if plan_error:
        queue_inner = f"""
      <div class="ap-error"><b>Autopilot could not read your ESPN roster.</b><br>{e(str(plan_error))}</div>
      <div class="ap-actions">
        <button class="ap-btn ap-btn-ghost" type="button" id="apReplan">Try again</button>
        <span class="ap-flash" role="status"></span>
      </div>"""
    elif swaps:
        queue_inner = f"""
      <p class="ap-card-sub" style="margin-bottom:2px"><b style="color:var(--text)">{len(swaps)} lineup move{'s' if len(swaps) != 1 else ''} queued for {e(week_label)}.</b> Nothing runs until you accept it. Accept the moves you want, dismiss the ones you don't.</p>
      {queue_rows}
      <div class="ap-actions">
        <button class="ap-btn" type="button" id="apAcceptAll">Accept all</button>
        <button class="ap-btn ap-btn-ghost" type="button" id="apReplan">Rebuild plan</button>
        <span class="ap-flash" role="status"></span>
      </div>"""
    else:
        queue_inner = f"""
      <div class="ap-quiet"><b>Your lineup already matches the recommended optimal lineup for {e(week_label)}.</b><br>No moves are queued. A fresh plan is built at the Thursday and Sunday checks, or rebuild one now.</div>
      <div class="ap-actions">
        <button class="ap-btn ap-btn-ghost" type="button" id="apReplan">Rebuild plan</button>
        <span class="ap-flash" role="status"></span>
      </div>"""
    queue_html = f'<div id="apQueue"{plan_id_attr}>{queue_inner}</div>'

    # --- Mode views -------------------------------------------------------
    seg = f"""
    <div class="ap-seg" id="apSeg" role="tablist" aria-label="Autopilot mode">
      <button type="button" data-ap-mode="off" role="tab" class="{'ap-on' if mode == 'off' else ''}">Off</button>
      <button type="button" data-ap-mode="approve" role="tab" class="{'ap-on' if mode == 'approve' else ''}">Approve</button>
      <button type="button" data-ap-mode="auto" role="tab" class="{'ap-on' if mode == 'auto' else ''}">Auto</button>
    </div>"""
    banners = ""
    if kill:
        banners += (
            '<div class="ap-banner ap-banner-warn">Your account-wide Autopilot '
            "kill switch is on. Nothing plans or executes in any league until "
            "you turn it off below.</div>"
        )
    if paused:
        banners += (
            '<div class="ap-banner ap-banner-warn">This league is paused. '
            "Plan runs and executions are halted until you resume.</div>"
        )
    if mode == "off":
        mode_view = """
      <div class="ap-quiet"><b>Autopilot is off for this league.</b><br>BR keeps making recommendations. Nothing is planned or executed. Switch to Approve to queue moves, or Auto to run them on schedule.</div>"""
    elif mode == "approve":
        mode_view = queue_html
    else:  # auto
        auto_queue = ""
        if swaps:
            rows = "".join(
                f"""
      <div class="ap-move">
        <div class="ap-move-top">
          <span class="ap-tag">Lineup</span>
          <div class="ap-move-main">
            <div class="ap-move-title">{e(_swap_title(swap))}</div>
            <div class="ap-move-why">{_swap_why(swap, issues_by_canon)}</div>
          </div>
          <span class="ap-delta{' ap-delta-neg' if (swap.get('delta') or 0) < 0 else ''}">{e(_fmt_delta(swap.get('delta')))}</span>
        </div>
      </div>"""
                for swap in swaps
            )
            auto_queue = f"""
      <p class="ap-card-sub" style="margin:10px 0 2px"><b style="color:var(--text)">Queued for the next run:</b></p>{rows}"""
        mode_view = f"""
      <p class="ap-card-sub" style="margin-bottom:10px"><b style="color:var(--text)">Auto is on. Full autonomy, inside the guardrails below.</b> Plans execute on schedule with no approval step. You get a push and an email after every executed move.</p>
      <div class="ap-autonext">
        <div class="ap-autonext-lab">Next execution</div>
        <div class="ap-autonext-main">Thursday evening lineup check &middot; executes automatically</div>
        <div class="ap-autonext-sub">Then Sunday, 12:30 PM ET final check.</div>
      </div>
      <div class="ap-sumrow"><span>Lineup executions</span><b>Automatic, Thu and Sun runs</b></div>
      <div class="ap-sumrow"><span>Scope</span><b>{'Problem slots only' if settings.get('problem_slots_only') else 'Full optimal lineup'}</b></div>
      <div class="ap-sumrow"><span>Protected players</span><b>Never benched, never dropped</b></div>
      <div class="ap-sumrow"><span>Notifications</span><b>Push and email, after every move</b></div>
      <div class="ap-sumrow"><span>Waivers</span><b>Not included yet. Lineups only for now.</b></div>
      {auto_queue}
      <div class="ap-actions">
        <button class="ap-btn ap-btn-ghost" type="button" data-ap-pause="1">Pause Autopilot</button>
        <span class="ap-flash" role="status"></span>
      </div>"""

    hero_sub = (
        "Runs the lineup moves BR already recommends, on your ESPN account, "
        "inside the guardrails you set below. Every write is verified by "
        "re-reading your roster."
    )
    context_line = " &middot; ".join(
        x for x in [e(league_name), e(team_name), e(week_label)] if x
    )
    hero = f"""
  <section class="ap-card ap-hero" id="apHero" aria-label="Autopilot">
    <div class="ap-hero-head">
      <div style="flex:1;min-width:220px">
        <h2 class="ap-hero-title">Autopilot<span class="nav-pro-badge">PRO</span></h2>
        <p class="ap-hero-sub">{hero_sub}</p>
        {f'<p class="ap-hero-sub">{context_line}</p>' if context_line else ''}
      </div>
      <span class="ap-pill"><span class="ap-dot"></span>ESPN: writes proven</span>
    </div>
    {banners}
    {seg}
    {mode_view}
  </section>"""

    # --- Guardrails ---------------------------------------------------------
    def switch(key, checked, label):
        return (
            f'<label class="otc-pill-switch"><input type="checkbox" '
            f'data-ap-setting="{key}"{" checked" if checked else ""} '
            f'aria-label="{e(label)}"><span class="otc-pill-track"></span></label>'
        )

    pause_btn = (
        '<button class="ap-btn ap-btn-ghost" type="button" data-ap-pause="0">Resume</button>'
        if paused else
        '<button class="ap-btn ap-btn-ghost" type="button" data-ap-pause="1">Pause</button>'
    )
    guardrails = f"""
  <section class="ap-card" aria-label="Guardrails">
    <h2 class="ap-h2">Guardrails</h2>
    <p class="ap-card-sub">Hard rules, checked again at execution time.</p>
    <div class="ap-rowline"><div class="ap-lab">Lineups<small>Autopilot may set your starting lineup</small></div>
      {switch('lineups_enabled', settings.get('lineups_enabled', True), 'Lineups on')}</div>
    <div class="ap-rowline"><div class="ap-lab">Problem slots only<small>In Auto, fix only empty, bye, or injured starters</small></div>
      {switch('problem_slots_only', settings.get('problem_slots_only', False), 'Problem slots only on')}</div>
    <div class="ap-rowline"><div class="ap-lab">Notifications: push<small>After every executed or failed move</small></div>
      {switch('notify_push', settings.get('notify_push', True), 'Push notifications on')}</div>
    <div class="ap-rowline"><div class="ap-lab">Notifications: email<small>After every executed or failed move</small></div>
      {switch('notify_email', settings.get('notify_email', True), 'Email notifications on')}</div>
    <div class="ap-rowline"><div class="ap-lab">Pause this league<small>Halts plan runs and executions here until you resume</small></div>
      {pause_btn}</div>
    <div class="ap-rowline"><div class="ap-lab">Kill switch: all leagues<small>Stops Autopilot everywhere on your account at once</small></div>
      <label class="otc-pill-switch"><input type="checkbox" id="apKillSwitch"{" checked" if kill else ""} aria-label="Account-wide Autopilot kill switch"><span class="otc-pill-track"></span></label></div>
    <div style="margin-top:14px">
      <div style="font-size:13.5px;font-weight:700;color:var(--text)">Protected players</div>
      <p class="ap-card-sub" style="margin-top:2px">A protected player is never benched by a plan and never dropped, in any mode.</p>
      <div class="ap-chips" id="apProtectedChips"></div>
      <p class="ap-card-sub" style="margin:12px 0 0">Tap a roster player to protect him:</p>
      <div class="ap-roster" id="apRosterList"></div>
    </div>
    <p class="ap-note">Waivers and FAAB arrive in Phase 1.5. Autopilot never makes trades.</p>
  </section>"""

    # --- Schedule + activity --------------------------------------------------
    log_rows = ""
    for row in audit[:12]:
        action = str(row.get("action") or "")
        label, cls = _AUDIT_LABEL.get(action, (action.title(), "ap-st-queue"))
        name = row.get("player_name") or ""
        detail = row.get("detail") or ""
        main = e(f"{name}: {detail}" if name and detail else (detail or name))
        bits = []
        if row.get("mode"):
            bits.append(f"{row['mode']} mode")
        if row.get("verification"):
            bits.append(e(str(row["verification"])))
        if row.get("created_at"):
            bits.append(e(_fmt_time(row["created_at"])))
        small = " &middot; ".join(bits)
        log_rows += f"""
        <div class="ap-log">
          {_audit_icon(action)}
          <div class="ap-log-main">{main}{f'<small>{small}</small>' if small else ''}</div>
          <span class="ap-status {cls}">{label}</span>
        </div>"""
    if not log_rows:
        log_rows = (
            '<p class="ap-card-sub" style="margin:0">Nothing yet. Plans, '
            "executions, and skips will land here.</p>"
        )
    schedule = f"""
  <section class="ap-card" aria-label="Schedule and activity">
    <div class="ap-grid2">
      <div>
        <h2 class="ap-h2">Next runs</h2>
        <p class="ap-card-sub">Eastern time. In Approve, runs queue moves for your tap. In Auto, runs execute.</p>
        <div class="ap-run"><span class="ap-when">Thursday</span><span class="ap-what"><b>Lineup check.</b> Evening pass before the week's first games.</span></div>
        <div class="ap-run"><span class="ap-when">Sunday</span><span class="ap-what"><b>Final check, 12:30 PM.</b> Late inactives, then lineups lock.</span></div>
      </div>
      <div>
        <h2 class="ap-h2">Activity</h2>
        <p class="ap-card-sub">Audit log. Every planned, executed, and skipped move lands here.</p>
        {log_rows}
      </div>
    </div>
  </section>"""

    roster_json = html.escape(json.dumps(roster_map), quote=True)
    protected_json = html.escape(json.dumps(protected_ids), quote=True)
    js = _JS.replace("__AP_CHECK__", json.dumps(_CHECK_SVG))
    body = f"""
<div class="ap-wrap" id="apRoot" data-ap-platform="{e(str(platform))}" data-ap-season="{int(season)}" data-ap-league="{e(str(league_id))}" data-ap-roster="{roster_json}" data-ap-protected="{protected_json}">
<style>{_CSS}</style>
{hero}
{guardrails}
{schedule}
</div>
{js}
"""
    return body
