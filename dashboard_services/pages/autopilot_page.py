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
transform:translateX(-100%);animation:apSweep .5s ease-out forwards}
@keyframes apSweep{to{transform:translateX(100%)}}
.ap-seg::before{content:"";position:absolute;inset:0;border-radius:10px;pointer-events:none;
box-shadow:inset 0 0 0 2px color-mix(in srgb,var(--accent) 60%,transparent);opacity:0}
.ap-seg.ap-arming::before{animation:apRingFade .55s ease-out}
@keyframes apRingFade{0%{opacity:.9}100%{opacity:0}}
.ap-hero{position:relative;overflow:hidden}
.ap-hero::after{content:"";position:absolute;inset:0;border-radius:var(--radius);pointer-events:none;
background:linear-gradient(100deg,transparent 25%,color-mix(in srgb,var(--accent) 15%,transparent) 50%,transparent 75%);
transform:translateX(-100%);opacity:0}
.ap-hero.ap-arming::after{animation:apHeroSheen .55s ease-out}
@keyframes apHeroSheen{0%{transform:translateX(-100%);opacity:1}100%{transform:translateX(100%);opacity:1}}
.ap-live .ap-view:not([hidden]){animation:apViewIn .3s ease-out}
@keyframes apViewIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.ap-swap-out{opacity:0;transform:translateY(4px);transition:opacity .18s ease,transform .18s ease}
.ap-swap-in{animation:apViewIn .32s ease-out}
.ap-nextrun{display:flex;align-items:center;gap:12px;background:var(--accent-soft);
border-radius:10px;padding:10px 12px;margin:0 0 12px}
.ap-nextrun-main{font-size:14px;font-weight:800;color:var(--text);margin-top:1px}
.ap-nextrun .ap-btn{margin-left:auto;flex:0 0 auto}
.ap-quiet{font-size:13.5px;color:var(--text-muted);background:var(--row);border:1px solid var(--grid);
border-radius:10px;padding:12px 14px}
.ap-quiet b{color:var(--text)}
.ap-error{font-size:13.5px;color:var(--text);background:var(--row);border:1px solid var(--grid);
border-radius:10px;padding:12px 14px}
.ap-move{position:relative;padding:12px 0 10px;border-top:1px solid var(--grid);
transition:opacity .2s ease}
.ap-move.ap-running{opacity:.78}
.ap-sheen{position:absolute;inset:0;overflow:hidden;pointer-events:none}
.ap-sheen i{display:block;width:100%;height:100%;
background:linear-gradient(100deg,transparent 30%,color-mix(in srgb,var(--accent) 14%,transparent) 50%,transparent 70%);
transform:translateX(-100%)}
.ap-move.ap-done .ap-sheen i{animation:apRowSweep .5s ease-out .1s}
@keyframes apRowSweep{to{transform:translateX(100%)}}
.ap-move.ap-shake{animation:apShake .3s ease-out}
@keyframes apShake{0%,100%{transform:translateX(0)}20%{transform:translateX(-4px)}60%{transform:translateX(4px)}}
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
.ap-delta{display:inline-block;font-size:15px;font-weight:800;color:var(--win);flex:0 0 auto}
.ap-move.ap-done .ap-delta{animation:apDeltaSettle .35s ease-out .1s}
@keyframes apDeltaSettle{0%{transform:scale(1.22)}100%{transform:scale(1)}}
.ap-qcount{display:inline-block}
.ap-qcount.ap-tick{animation:apTick .4s ease-out}
@keyframes apTick{0%{transform:scale(1)}35%{transform:scale(1.35)}100%{transform:scale(1)}}
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
.ap-btn.ap-busy::before{content:"";width:12px;height:12px;margin-right:7px;border-radius:50%;
border:2px solid color-mix(in srgb,currentColor 35%,transparent);border-top-color:currentColor;
animation:apSpin .7s linear infinite;flex:0 0 auto}
@keyframes apSpin{to{transform:rotate(360deg)}}
.ap-btn-ghost{background:var(--card);color:var(--text);border-color:var(--border)}
.ap-actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:12px}
.ap-flash{font-size:12.5px;color:var(--accent);font-weight:600}
.ap-status{font-size:10.5px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;
border-radius:6px;padding:3px 7px;flex:0 0 auto;display:inline-flex;align-items:center;gap:5px}
.ap-st-ok{color:var(--win);background:var(--row)}
.ap-st-fail{color:var(--text);background:var(--row)}
.ap-st-skip{color:var(--text-muted);background:var(--row)}
.ap-st-queue{color:var(--accent);background:var(--accent-soft)}
.ap-check{display:inline-block}
.ap-check path{stroke-dasharray:100;stroke-dashoffset:100}
.ap-done .ap-check path{animation:apDraw .45s ease-out .05s forwards}
@keyframes apDraw{to{stroke-dashoffset:0}}
.ap-autonext-lab{font-size:10.5px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:var(--accent)}
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
    'stroke-linejoin="round"><path pathLength="100" d="M4.5 12.5l5 5L19.5 7"/></svg>'
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
  root.classList.add('ap-live');
  var PLATFORM=root.getAttribute('data-ap-platform'),SEASON=root.getAttribute('data-ap-season'),
      LEAGUE=root.getAttribute('data-ap-league');
  var base='/api/autopilot/'+PLATFORM+'/'+SEASON+'/'+LEAGUE;
  var currentMode=root.getAttribute('data-ap-mode')||'off';
  function $all(sel,el){return [].slice.call((el||root).querySelectorAll(sel));}
  function post(url,body){
    return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body||{})}).then(function(r){
        return r.json().catch(function(){return{};}).then(function(d){return {ok:r.ok,data:d};});
      });
  }
  function flash(msg){
    $all('.ap-flash').forEach(function(t){t.textContent=msg||'';});
  }
  function setBusy(btn,on,label){
    if(!btn){return;}
    if(on){btn.setAttribute('data-ap-label',btn.textContent);
      btn.classList.add('ap-busy'); btn.disabled=true;
      if(label){btn.textContent=label;}}
    else{btn.classList.remove('ap-busy'); btn.disabled=false;
      var l=btn.getAttribute('data-ap-label'); if(l){btn.textContent=l;}}
  }
  // --- Soft in-place refresh ---------------------------------------------
  // Mode changes, executions, replans and manual runs all refresh the
  // mode region + activity card by swapping in the freshly rendered
  // fragments: no document reload, no scroll loss, no full repaint.
  var refreshing=false;
  function softRefresh(done){
    if(refreshing){if(done){done(false);} return;}
    refreshing=true;
    fetch(window.location.href).then(function(r){return r.text();}).then(function(text){
      var doc=new DOMParser().parseFromString(text,'text/html');
      var oldCount=root.querySelector('.ap-qcount');
      var oldN=oldCount?parseInt(oldCount.getAttribute('data-ap-qcount'),10):null;
      ['apModeRegion','apActivity'].forEach(function(id){
        var fresh=doc.getElementById(id), cur=document.getElementById(id);
        if(fresh&&cur){
          cur.classList.add('ap-swap-out');
          setTimeout(function(){
            var imp=document.importNode(fresh,true);
            imp.classList.add('ap-swap-in');
            cur.replaceWith(imp);
          },180);
        }
      });
      var freshRoot=doc.getElementById('apRoot');
      if(freshRoot){
        root.setAttribute('data-ap-roster',freshRoot.getAttribute('data-ap-roster')||'{}');
        root.setAttribute('data-ap-protected',freshRoot.getAttribute('data-ap-protected')||'[]');
        currentMode=freshRoot.getAttribute('data-ap-mode')||currentMode;
        root.setAttribute('data-ap-mode',currentMode);
        syncProtectedFromRoot();
      }
      setTimeout(function(){
        var nc=root.querySelector('.ap-qcount');
        if(nc&&oldN!==null&&parseInt(nc.getAttribute('data-ap-qcount'),10)!==oldN){
          nc.classList.add('ap-tick');
        }
        refreshing=false; if(done){done(true);}
      },300);
    }).catch(function(){refreshing=false; if(done){done(false);}});
  }
  // --- Mode segmented control ---------------------------------------------
  // The mode POST only persists the mode (no plan build, no ESPN read),
  // so the switch flips in place instantly: segment, status view, and
  // the Auto arming animation all run off the optimistic update. When
  // the target view was not rendered (the page renders the off view
  // plus the active mode's view), the region soft-refreshes behind
  // the animation instead of reloading the page.
  function segEl(){return document.getElementById('apSeg');}
  function markSeg(m){
    var s=segEl(); if(!s){return;}
    $all('[data-ap-mode]',s).forEach(function(b){
      b.classList.toggle('ap-on',b.getAttribute('data-ap-mode')===m);});
  }
  function showView(m){
    $all('.ap-view').forEach(function(v){
      v.hidden=v.getAttribute('data-ap-view')!==m;});
    var nr=document.getElementById('apNextRun');
    if(nr){nr.hidden=(m==='off');}
  }
  function viewExists(m){
    return !!root.querySelector('.ap-view[data-ap-view="'+m+'"]');
  }
  function clearArming(){
    var s=segEl(), h=document.getElementById('apHero');
    if(s){s.classList.remove('ap-arming');}
    if(h){h.classList.remove('ap-arming');}
  }
  var modeBusy=false;
  function chooseMode(m){
    if(modeBusy||m===currentMode){return;}
    modeBusy=true;
    var prev=currentMode;
    currentMode=m; markSeg(m);
    if(m==='auto'){
      var s=segEl(), h=document.getElementById('apHero');
      if(s){s.classList.add('ap-arming');}
      if(h){h.classList.add('ap-arming');}
    }
    if(viewExists(m)){showView(m);}
    post(base+'/mode',{mode:m}).then(function(res){
      modeBusy=false;
      setTimeout(clearArming,560);
      if(!res.ok){
        currentMode=prev; markSeg(prev);
        if(viewExists(prev)){showView(prev);}
        flash('Could not change the mode. Try again.');
        return;
      }
      if(!viewExists(m)){softRefresh();}
    }).catch(function(){
      modeBusy=false; clearArming();
      currentMode=prev; markSeg(prev);
      if(viewExists(prev)){showView(prev);}
      flash('Could not change the mode. Try again.');
    });
  }
  // --- Pause / resume (in place: banner + every pause button) -------------
  function togglePause(btn){
    var target=btn.getAttribute('data-ap-pause')==='1';
    btn.disabled=true;
    post(base+'/settings',{paused:target}).then(function(res){
      btn.disabled=false;
      if(!res.ok){flash('Could not change the pause state.'); return;}
      var banner=document.getElementById('apPauseBanner');
      if(banner){banner.hidden=!target;}
      $all('[data-ap-pause]').forEach(function(b){
        b.setAttribute('data-ap-pause',target?'0':'1');
        b.textContent=target?(b.getAttribute('data-label-resume')||'Resume')
                            :(b.getAttribute('data-label-pause')||'Pause');
      });
    }).catch(function(){btn.disabled=false;});
  }
  // --- Queue: accept / dismiss ---------------------------------------------
  function runGroups(indexes, rows, btn){
    var queue=document.getElementById('apQueue');
    if(!queue||!indexes.length){return;}
    var planId=queue.getAttribute('data-plan-id');
    rows.forEach(function(r){r.classList.add('ap-running');
      $all('button',r).forEach(function(x){x.disabled=true;});});
    setBusy(btn,true,'Running');
    post(base+'/execute',{plan_id:parseInt(planId,10),groups:indexes}).then(function(res){
      setBusy(btn,false);
      var result=(res.data&&res.data.result)||{};
      var byIndex={};
      (result.groups||[]).forEach(function(g){byIndex[g.index]=g;});
      var anyExecuted=false, anySettled=false;
      rows.forEach(function(r){
        r.classList.remove('ap-running');
        var g=byIndex[parseInt(r.getAttribute('data-ap-group'),10)];
        var out=r.querySelector('[data-ap-outcome]');
        if(!out||!g){
          $all('button',r).forEach(function(x){x.disabled=false;});
          return;
        }
        anySettled=true; out.hidden=false;
        if(g.outcome==='executed'||g.outcome==='already_applied'){
          anyExecuted=true;
          r.classList.add('ap-done');
          out.className='ap-status ap-st-ok';
          out.innerHTML=__AP_CHECK__+(g.outcome==='executed'?'Executed &middot; Verified':'Already in place');
        } else {
          r.classList.add('ap-failed','ap-shake');
          out.className='ap-status ap-st-fail';
          out.textContent='Failed: '+((g&&g.detail)||result.detail||'The move did not run.');
          setTimeout(function(){r.classList.remove('ap-shake');},350);
        }
      });
      if(result.status==='superseded'){
        flash('Your roster changed since this plan was built. Refreshing with a fresh plan.');
        setTimeout(function(){softRefresh();},700);
      } else if(anyExecuted){
        // Let the check draw + sweep play, then fade the fresh queue in.
        setTimeout(function(){softRefresh();},1150);
      } else if(!anySettled){
        rows.forEach(function(r){
          $all('button',r).forEach(function(x){x.disabled=false;});});
        flash('The run did not complete. Try again.');
      }
    }).catch(function(){
      setBusy(btn,false);
      rows.forEach(function(r){r.classList.remove('ap-running');
        $all('button',r).forEach(function(x){x.disabled=false;});});
      flash('The run did not complete. Try again.');
    });
  }
  // --- Run now ---------------------------------------------------------------
  function runNow(btn){
    setBusy(btn,true,'Running');
    post(base+'/run-now',{}).then(function(res){
      setBusy(btn,false);
      var d=res.data||{};
      if(!res.ok){flash(d.error||'The run did not complete.'); return;}
      if(d.status==='executed'){flash('Run complete. Moves executed and verified.');}
      else if(d.status==='queued'){flash('Plan built. Moves are waiting for your approval.');}
      else if(d.status==='no_moves'){flash('Run complete. Your lineup already matches the recommended lineup.');}
      else{flash('Run complete.');}
      softRefresh();
    }).catch(function(){
      setBusy(btn,false);
      flash('The run did not complete. Try again.');
    });
  }
  // --- Delegated clicks (survive soft-refresh fragment swaps) ---------------
  root.addEventListener('click',function(ev){
    var t=ev.target;
    var modeBtn=t.closest('[data-ap-mode]');
    if(modeBtn&&segEl()&&segEl().contains(modeBtn)){
      chooseMode(modeBtn.getAttribute('data-ap-mode')); return;
    }
    var pauseBtn=t.closest('[data-ap-pause]');
    if(pauseBtn){togglePause(pauseBtn); return;}
    var acceptBtn=t.closest('[data-ap-accept]');
    if(acceptBtn){
      var row=acceptBtn.closest('.ap-move'); if(!row){return;}
      runGroups([parseInt(row.getAttribute('data-ap-group'),10)],[row],acceptBtn);
      return;
    }
    var dismissBtn=t.closest('[data-ap-dismiss]');
    if(dismissBtn){
      var drow=dismissBtn.closest('.ap-move'); if(drow){drow.hidden=true;}
      flash('Move dismissed. It stays unexecuted; running the rest logs it as skipped.');
      return;
    }
    if(t.closest('#apAcceptAll')){
      var queue=document.getElementById('apQueue'); if(!queue){return;}
      var rows=$all('.ap-move',queue).filter(function(r){return !r.hidden;});
      runGroups(rows.map(function(r){
        return parseInt(r.getAttribute('data-ap-group'),10);}),rows,t.closest('#apAcceptAll'));
      return;
    }
    if(t.closest('#apRunNow')){runNow(t.closest('#apRunNow')); return;}
    var replanBtn=t.closest('#apReplan');
    if(replanBtn){
      setBusy(replanBtn,true,'Rebuilding');
      post(base+'/replan',{}).then(function(res){
        if(res.ok){softRefresh(function(){setBusy(replanBtn,false);});}
        else{setBusy(replanBtn,false); flash('Could not rebuild the plan.');}
      }).catch(function(){setBusy(replanBtn,false);});
      return;
    }
  });
  // --- Delegated changes: guardrail switches + kill switch ------------------
  root.addEventListener('change',function(ev){
    var t=ev.target;
    if(t.id==='apKillSwitch'){
      post('/api/autopilot/kill-switch',{on:t.checked}).then(function(res){
        if(res.ok){
          var b=document.getElementById('apKillBanner');
          if(b){b.hidden=!t.checked;}
        } else {t.checked=!t.checked;}
      }).catch(function(){t.checked=!t.checked;});
      return;
    }
    if(t.hasAttribute&&t.hasAttribute('data-ap-setting')){
      var key=t.getAttribute('data-ap-setting'), body={}; body[key]=t.checked;
      post(base+'/settings',body).then(function(res){
        if(!res.ok){t.checked=!t.checked; flash('Could not save that setting.');}
      }).catch(function(){t.checked=!t.checked;});
    }
  });
  // --- Protected players ------------------------------------------------------
  var roster={};
  var protectedIds=[];
  function syncProtectedFromRoot(){
    try{roster=JSON.parse(root.getAttribute('data-ap-roster')||'{}');}catch(e){roster={};}
    try{protectedIds=JSON.parse(root.getAttribute('data-ap-protected')||'[]');}catch(e){protectedIds=[];}
    renderProtected();
  }
  function renderProtected(){
    var chipsBox=document.getElementById('apProtectedChips');
    var rosterBox=document.getElementById('apRosterList');
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
  syncProtectedFromRoot();
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
        <span class="ap-sheen"><i></i></span>
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
      <p class="ap-card-sub" style="margin-bottom:2px"><b style="color:var(--text)"><span class="ap-qcount" data-ap-qcount="{len(swaps)}">{len(swaps)}</span> lineup move{'s' if len(swaps) != 1 else ''} queued for {e(week_label)}.</b> Nothing runs until you accept it. Accept the moves you want, dismiss the ones you don't.</p>
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
    # Banners render in both states (hidden when inactive) so the page
    # JS can toggle them in place on pause / kill-switch changes
    # without a reload.
    banners = (
        '<div class="ap-banner ap-banner-warn" id="apKillBanner"'
        f'{" hidden" if not kill else ""}>Your account-wide Autopilot '
        "kill switch is on. Nothing plans or executes in any league until "
        "you turn it off below.</div>"
        '<div class="ap-banner ap-banner-warn" id="apPauseBanner"'
        f'{" hidden" if not paused else ""}>This league is paused. '
        "Plan runs and executions are halted until you resume.</div>"
    )
    off_view = f"""
      <div class="ap-view" data-ap-view="off"{'' if mode == 'off' else ' hidden'}>
        <div class="ap-quiet"><b>Autopilot is off for this league.</b><br>BR keeps making recommendations. Nothing is planned or executed. Switch to Approve to queue moves, or Auto to run them on schedule.</div>
      </div>"""
    next_run_label = str(data.get("next_run_label") or "")
    next_run_bar = ""
    if mode != "off" and next_run_label:
        next_run_bar = f"""
      <div class="ap-nextrun" id="apNextRun">
        <div>
          <div class="ap-autonext-lab">Next run</div>
          <div class="ap-nextrun-main">{e(next_run_label)}</div>
        </div>
        <button class="ap-btn ap-btn-ghost" type="button" id="apRunNow">Run now</button>
      </div>"""
    if mode == "approve":
        mode_view = f"""
      <div class="ap-view" data-ap-view="approve">{queue_html}
      </div>"""
    elif mode == "auto":
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
      <div class="ap-view" data-ap-view="auto">
      <p class="ap-card-sub" style="margin-bottom:10px"><b style="color:var(--text)">Auto is on. Full autonomy, inside the guardrails below.</b> Plans execute on schedule with no approval step. You get a push and an email after every executed move.</p>
      <div class="ap-sumrow"><span>Lineup executions</span><b>Automatic, Thu and Sun runs</b></div>
      <div class="ap-sumrow"><span>Scope</span><b>{'Problem slots only' if settings.get('problem_slots_only') else 'Full optimal lineup'}</b></div>
      <div class="ap-sumrow"><span>Protected players</span><b>Never benched, never dropped</b></div>
      <div class="ap-sumrow"><span>Notifications</span><b>Push and email, after every move</b></div>
      <div class="ap-sumrow"><span>Waivers</span><b>Not included yet. Lineups only for now.</b></div>
      {auto_queue}
      <div class="ap-actions">
        <button class="ap-btn ap-btn-ghost" type="button" data-ap-pause="{'0' if paused else '1'}" data-label-pause="Pause Autopilot" data-label-resume="Resume Autopilot">{'Resume Autopilot' if paused else 'Pause Autopilot'}</button>
        <span class="ap-flash" role="status"></span>
      </div>
      </div>"""
    else:
        mode_view = ""

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
    <div id="apModeRegion">
    {banners}
    {seg}
    {next_run_bar}
    {off_view}
    {mode_view}
    </div>
  </section>"""

    # --- Guardrails ---------------------------------------------------------
    def switch(key, checked, label):
        return (
            f'<label class="otc-pill-switch"><input type="checkbox" '
            f'data-ap-setting="{key}"{" checked" if checked else ""} '
            f'aria-label="{e(label)}"><span class="otc-pill-track"></span></label>'
        )

    pause_btn = (
        '<button class="ap-btn ap-btn-ghost" type="button" data-ap-pause="0" '
        'data-label-pause="Pause" data-label-resume="Resume">Resume</button>'
        if paused else
        '<button class="ap-btn ap-btn-ghost" type="button" data-ap-pause="1" '
        'data-label-pause="Pause" data-label-resume="Resume">Pause</button>'
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
  <section class="ap-card" id="apActivity" aria-label="Schedule and activity">
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
<div class="ap-wrap" id="apRoot" data-ap-platform="{e(str(platform))}" data-ap-season="{int(season)}" data-ap-league="{e(str(league_id))}" data-ap-mode="{e(mode)}" data-ap-roster="{roster_json}" data-ap-protected="{protected_json}">
<style>{_CSS}</style>
{hero}
{guardrails}
{schedule}
</div>
{js}
"""
    return body
