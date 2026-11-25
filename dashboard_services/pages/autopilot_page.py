"""Autopilot page body: an in-site UI mock of the unbuilt Autopilot feature.

Design source: the approved artifact mock (~/workspace/dashboard-mocks/
autopilot-controls/), re-skinned to the real site's card grammar and theme
variables. Everything on this page is sample data, labeled as such: there is
no execute endpoint, no write client, and no settings persistence behind it.
Mode switches, guardrail toggles, and the per-move Accept buttons are
client-side presentation only; the flash copy says so.

Access is gated by dashboard_services.autopilot_preview (owner preview,
pilot ESPN league only); the route enforces the gate before this builder
runs.
"""
from __future__ import annotations

import html

_CSS = """
.ap-wrap{max-width:880px;margin:0 auto;padding:6px 2px 96px}
.ap-h1{font-size:22px;font-weight:800;letter-spacing:-.02em;color:var(--text);margin:4px 0 2px;text-align:left}
.ap-sub{font-size:13px;color:var(--text-muted);margin:0 0 14px}
.ap-sample{color:var(--accent);font-weight:700}
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
.ap-seg{display:flex;background:var(--row);border:1px solid var(--grid);border-radius:10px;
padding:3px;gap:3px;margin:14px 0 12px;max-width:420px}
.ap-seg button{flex:1;border:0;background:transparent;color:var(--text-muted);font:inherit;
font-size:13.5px;font-weight:700;padding:7px 0;border-radius:8px;cursor:pointer}
.ap-seg button.ap-on{background:var(--accent);color:var(--on-accent)}
.ap-quiet{font-size:13.5px;color:var(--text-muted);background:var(--row);border:1px solid var(--grid);
border-radius:10px;padding:12px 14px}
.ap-quiet b{color:var(--text)}
.ap-move{display:flex;align-items:center;gap:10px;padding:11px 0;border-top:1px solid var(--grid)}
.ap-wrap [hidden]{display:none!important}
.ap-dismiss{border:0;background:transparent;color:var(--text-subtle);font:inherit;font-size:15px;
line-height:1;cursor:pointer;padding:2px 3px;flex:0 0 auto}
.ap-tag{font-size:10px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;color:var(--accent);
background:var(--accent-soft);border-radius:6px;padding:3px 7px;flex:0 0 auto}
.ap-move-main{flex:1;min-width:0}
.ap-move-title{font-size:14px;font-weight:700;color:var(--text)}
.ap-move-why{font-size:12.5px;color:var(--text-muted);margin-top:1px}
.ap-delta{font-size:15px;font-weight:800;color:var(--win);flex:0 0 auto}
.ap-actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:12px}
.ap-btn{display:inline-flex;align-items:center;justify-content:center;height:32px;padding:0 14px;
background:var(--accent);color:var(--on-accent);border:1px solid var(--accent);font:inherit;
font-size:13px;font-weight:700;border-radius:var(--radius-pill);cursor:pointer;white-space:nowrap}
.ap-btn-ghost{background:var(--card);color:var(--text);border-color:var(--border)}
.ap-flash{font-size:12.5px;color:var(--accent);font-weight:600}
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
.ap-addrow{display:flex;gap:8px;margin-top:10px;max-width:420px}
.ap-addrow input{flex:1;font:inherit;font-size:13px;padding:8px 11px;border:1px solid var(--border);
border-radius:9px;background:var(--card);color:var(--text);min-width:0}
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
.ap-log-skip{color:var(--text-subtle)}
.ap-log-queue{color:var(--accent)}
.ap-log-main{flex:1;font-size:13px;color:var(--text);min-width:0}
.ap-log-main small{display:block;font-size:12px;color:var(--text-muted);margin-top:1px}
.ap-status{font-size:10.5px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;
border-radius:6px;padding:3px 7px;flex:0 0 auto}
.ap-st-ok{color:var(--win);background:var(--row)}
.ap-st-skip{color:var(--text-muted);background:var(--row)}
.ap-st-queue{color:var(--accent);background:var(--accent-soft)}
.ap-strip{border:1px solid var(--border);border-radius:12px;padding:12px 16px;margin-bottom:12px;
background:var(--card)}
.ap-strip-row{display:flex;gap:8px;font-size:13px;color:var(--text-muted);padding:4px 0}
.ap-strip-row b{color:var(--text);flex:0 0 62px}
.ap-foot{font-size:12px;color:var(--text-subtle);text-align:center;margin:4px 0 0}
"""

_JS = """
<script>
(function(){
  var root=document.getElementById('apRoot'); if(!root){return;}
  var views={off:document.getElementById('apViewOff'),approve:document.getElementById('apViewApprove'),auto:document.getElementById('apViewAuto')};
  var btns=root.querySelectorAll('[data-ap-mode]');
  btns.forEach(function(b){b.addEventListener('click',function(){
    btns.forEach(function(x){x.classList.toggle('ap-on',x===b);});
    var m=b.getAttribute('data-ap-mode');
    Object.keys(views).forEach(function(k){if(views[k]){views[k].hidden=(k!==m);}});
  });});
  root.querySelectorAll('[data-ap-flash]').forEach(function(b){b.addEventListener('click',function(){
    var view=b.closest('.ap-view'); var t=view?view.querySelector('.ap-flash'):null;
    if(t){t.textContent=b.getAttribute('data-ap-flash');}
  });});
  function apFlash(el,msg){var view=el.closest('.ap-view'); var t=view?view.querySelector('.ap-flash'):null;
    if(t){t.textContent=msg;}}
  root.querySelectorAll('[data-ap-accept]').forEach(function(b){b.addEventListener('click',function(){
    var row=b.closest('.ap-move'); if(!row){return;}
    row.querySelectorAll('[data-ap-accept],[data-ap-dismiss]').forEach(function(x){x.hidden=true;});
    var st=row.querySelector('[data-ap-accepted]'); if(st){st.hidden=false;}
    apFlash(b,'Preview mock: nothing was submitted. In the real build, accepted moves run and verify against your ESPN roster.');
  });});
  root.querySelectorAll('[data-ap-dismiss]').forEach(function(b){b.addEventListener('click',function(){
    var row=b.closest('.ap-move'); if(row){row.hidden=true;}
    apFlash(b,'Preview mock: move dismissed. Nothing was submitted.');
  });});
  var pb=document.getElementById('apPauseBtn');
  if(pb){pb.addEventListener('click',function(){pb.textContent=(pb.textContent==='Pause')?'Resume':'Pause';});}
})();
</script>
"""

_MOCK_FLASH = (
    "Preview mock: nothing was submitted. In the real build, accepted moves "
    "run and verify against your ESPN roster."
)


def build_autopilot_body(platform: str, season: int, league_id: str, ctx: dict | None = None) -> str:
    ctx = ctx or {}
    league = ctx.get("league") or {}
    viewer = ctx.get("viewer") or {}
    league_name = str(league.get("name") or "").strip()
    team_name = str(viewer.get("viewer_team_name") or "").strip()
    week = ctx.get("current_week")
    # Sample context fallback: the pilot league this mock is scoped to.
    if not league_name:
        league_name = "Josh Allen's Twitter"
    if not team_name:
        team_name = "Cook and Ball Torture"
    try:
        week_label = f"Week {int(week)}" if week else "Week 13"
    except (TypeError, ValueError):
        week_label = "Week 13"

    e = html.escape
    body = f"""
<div class="ap-wrap" id="apRoot">
<style>{_CSS}</style>
  <h1 class="ap-h1">Autopilot</h1>
  <p class="ap-sub"><b>{e(league_name)}</b> &middot; ESPN league {e(str(league_id))} &middot; {e(team_name)} &middot; {e(week_label)} &middot; <span class="ap-sample">Preview mock: all player and move data is sample</span></p>

  <section class="ap-card" aria-label="Autopilot">
    <div class="ap-hero-head">
      <div style="flex:1;min-width:220px">
        <h2 class="ap-hero-title">Autopilot<span class="nav-pro-badge">PRO</span></h2>
        <p class="ap-hero-sub">Runs the lineup moves BR already recommends, on your ESPN account, inside the guardrails you set below. Every write is verified by re-reading your roster.</p>
      </div>
      <span class="ap-pill"><span class="ap-dot"></span>ESPN: writes proven</span>
    </div>

    <div class="ap-seg" role="tablist" aria-label="Autopilot mode">
      <button type="button" data-ap-mode="off" role="tab">Off</button>
      <button type="button" data-ap-mode="approve" role="tab" class="ap-on">Approve</button>
      <button type="button" data-ap-mode="auto" role="tab">Auto</button>
    </div>

    <div class="ap-view" id="apViewOff" hidden>
      <div class="ap-quiet"><b>Autopilot is off for this league.</b><br>BR keeps making recommendations. Nothing is planned or executed. Switch to Approve to queue moves, or Auto to run them on schedule.</div>
    </div>

    <div class="ap-view" id="apViewApprove">
      <p class="ap-card-sub" style="margin-bottom:2px"><b style="color:var(--text)">2 lineup moves queued for {e(week_label)} (sample).</b> Nothing runs until you accept it. Accept the moves you want, dismiss the ones you don't.</p>

      <div class="ap-move">
        <span class="ap-tag">Lineup</span>
        <div class="ap-move-main">
          <div class="ap-move-title">Start Christian Watson (WR) at FLEX over Chase Brown (RB)</div>
          <div class="ap-move-why">Start score <b>16.4</b> vs 14.1 &middot; Watson is the higher blended score for the FLEX slot this week</div>
        </div>
        <span class="ap-delta">+2.3</span>
        <span class="ap-status ap-st-ok" data-ap-accepted hidden>Accepted</span>
        <button class="ap-btn" type="button" data-ap-accept>Accept</button>
        <button class="ap-dismiss" type="button" data-ap-dismiss aria-label="Dismiss lineup swap Watson over Brown">&times;</button>
      </div>

      <div class="ap-move">
        <span class="ap-tag">Lineup</span>
        <div class="ap-move-main">
          <div class="ap-move-title">Start George Pickens (WR) over Drake London (WR)</div>
          <div class="ap-move-why">Start score <b>17.7</b> vs 15.9 &middot; London is Questionable and limited all week</div>
        </div>
        <span class="ap-delta">+1.8</span>
        <span class="ap-status ap-st-ok" data-ap-accepted hidden>Accepted</span>
        <button class="ap-btn" type="button" data-ap-accept>Accept</button>
        <button class="ap-dismiss" type="button" data-ap-dismiss aria-label="Dismiss lineup swap Pickens over London">&times;</button>
      </div>

      <div class="ap-actions">
        <button class="ap-btn" type="button" data-ap-flash="{_MOCK_FLASH}">Accept all</button>
        <button class="ap-btn ap-btn-ghost" type="button" data-ap-flash="{_MOCK_FLASH}">Run accepted</button>
        <span class="ap-flash" role="status"></span>
      </div>
    </div>

    <div class="ap-view" id="apViewAuto" hidden>
      <p class="ap-card-sub" style="margin-bottom:10px"><b style="color:var(--text)">Auto is on. Full autonomy, inside the guardrails below.</b> Plans execute on schedule with no approval step. You get a push and an email after every executed move.</p>
      <div class="ap-autonext">
        <div class="ap-autonext-lab">Next execution</div>
        <div class="ap-autonext-main">Thu evening lineup check &middot; executes automatically</div>
        <div class="ap-autonext-sub">Then Sunday, 12:30 PM ET final check. Last run: Sunday final check executed 1 swap, verified by re-read (sample).</div>
      </div>
      <div class="ap-sumrow"><span>Lineup executions</span><b>Automatic, Thu and Sun runs</b></div>
      <div class="ap-sumrow"><span>Scope</span><b>Problem slots only (setting below)</b></div>
      <div class="ap-sumrow"><span>Protected players</span><b>Never benched, never dropped</b></div>
      <div class="ap-sumrow"><span>Notifications</span><b>Push and email, after every move</b></div>
      <div class="ap-sumrow"><span>Waivers</span><b>Not included yet. Lineups only for now.</b></div>
      <div class="ap-actions">
        <button class="ap-btn ap-btn-ghost" type="button" data-ap-flash="Preview mock: in the real build, scheduled runs for this league halt until you resume.">Pause Autopilot</button>
        <span class="ap-flash" role="status"></span>
      </div>
    </div>
  </section>

  <section class="ap-card" aria-label="Guardrails">
    <h2 class="ap-h2">Guardrails</h2>
    <p class="ap-card-sub">Hard rules, checked again at execution time. Sample settings.</p>
    <div class="ap-rowline"><div class="ap-lab">Lineups<small>Autopilot may set your starting lineup</small></div>
      <label class="otc-pill-switch"><input type="checkbox" checked aria-label="Lineups on"><span class="otc-pill-track"></span></label></div>
    <div class="ap-rowline"><div class="ap-lab">Problem slots only<small>In Auto, fix only empty, bye, or injured starters</small></div>
      <label class="otc-pill-switch"><input type="checkbox" checked aria-label="Problem slots only on"><span class="otc-pill-track"></span></label></div>
    <div class="ap-rowline"><div class="ap-lab">Notifications: push<small>After every executed or failed move</small></div>
      <label class="otc-pill-switch"><input type="checkbox" checked aria-label="Push notifications on"><span class="otc-pill-track"></span></label></div>
    <div class="ap-rowline"><div class="ap-lab">Notifications: email<small>After every executed or failed move</small></div>
      <label class="otc-pill-switch"><input type="checkbox" checked aria-label="Email notifications on"><span class="otc-pill-track"></span></label></div>
    <div class="ap-rowline"><div class="ap-lab">Pause this league<small>Halts plan runs and executions here until you resume</small></div>
      <button class="ap-btn ap-btn-ghost" type="button" id="apPauseBtn">Pause</button></div>
    <div style="margin-top:14px">
      <div style="font-size:13.5px;font-weight:700;color:var(--text)">Protected players <span class="ap-sample">(sample)</span></div>
      <p class="ap-card-sub" style="margin-top:2px">A protected player is never benched by a plan and never dropped, in any mode.</p>
      <div class="ap-chips">
        <span class="ap-chip">Jahmyr Gibbs <button type="button" aria-label="Remove Gibbs">&times;</button></span>
        <span class="ap-chip">Justin Jefferson <button type="button" aria-label="Remove Jefferson">&times;</button></span>
        <span class="ap-chip">Josh Allen <button type="button" aria-label="Remove Allen">&times;</button></span>
      </div>
      <div class="ap-addrow"><input type="text" placeholder="Add a player to protect" aria-label="Add a protected player"><button class="ap-btn ap-btn-ghost" type="button">Add</button></div>
    </div>
    <p class="ap-note">Waivers and FAAB arrive in Phase 1.5. Autopilot never makes trades.</p>
  </section>

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
        <div class="ap-log">
          <svg class="ap-log-ic ap-log-ok" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" r="10" stroke-opacity=".35"/><path d="M8 12.5l2.6 2.6L16 9.5"/></svg>
          <div class="ap-log-main">Lineup swap executed: Watson in at FLEX, Brown out (sample)<small>Auto mode &middot; Verified by re-reading your ESPN roster</small></div>
          <span class="ap-status ap-st-ok">Verified</span>
        </div>
        <div class="ap-log">
          <svg class="ap-log-ic ap-log-skip" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" r="10" stroke-opacity=".4"/><path d="M8 8l8 8M16 8l-8 8"/></svg>
          <div class="ap-log-main">Swap skipped: Pickens over London (sample)<small>You dismissed this move in Approve</small></div>
          <span class="ap-status ap-st-skip">Skipped</span>
        </div>
        <div class="ap-log">
          <svg class="ap-log-ic ap-log-queue" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" r="10" stroke-opacity=".4"/><path d="M12 7v5l3.5 2"/></svg>
          <div class="ap-log-main">Plan queued for {e(week_label)} (sample)<small>2 lineup moves waiting in Approve</small></div>
          <span class="ap-status ap-st-queue">Queued</span>
        </div>
      </div>
    </div>
  </section>

  <div class="ap-strip">
    <div class="ap-strip-row"><b>ESPN:</b><span>Available. Lineup writes proven on this league.</span></div>
    <div class="ap-strip-row"><b>Yahoo:</b><span>In the pilot for planning and UI. Execution stays off until the Yahoo write probe passes.</span></div>
    <div class="ap-strip-row"><b>Sleeper:</b><span>Unavailable. Sleeper's API is read-only, so BR can plan the moves but cannot place them.</span></div>
  </div>

  <p class="ap-foot">Preview mock only. Sample data throughout. No moves are real and nothing here touches a league.</p>
</div>
{_JS}
"""
    return body
