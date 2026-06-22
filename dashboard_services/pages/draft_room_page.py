"""
Standalone Draft Room (Draft Assistant) page.

Phase 2: a dedicated, self-contained draft board that supersedes the old
Prospects→Draft tab. Supports manual drafting for both startup (all players)
and rookie drafts, with snake / linear / third-round-reversal pick order.
Live Sleeper sync, persistence/history, and the full command-center panels
land in later phases; this establishes the standalone page + board grid +
best-available picker + the pickOrder foundation.

The page is self-contained: its CSS/JS are inlined here so nothing depends on
the prospects-page IIFE. Server values are passed via a small window.__draftCfg
JSON blob so the JS body needs no f-string brace escaping.
"""
from __future__ import annotations

import json
from typing import Optional


def build_draft_room_body(
    league_id: Optional[str],
    season: Optional[int],
    platform: Optional[str] = None,
    *,
    is_guest: bool = False,
    num_teams: Optional[int] = None,
    is_superflex: bool = False,
    viewer_user_id: Optional[str] = None,
) -> str:
    cfg = {
        "leagueId": league_id or "",
        "season": int(season) if season else None,
        "platform": platform or "sleeper",
        "isGuest": bool(is_guest),
        "numTeams": int(num_teams) if num_teams else None,
        "isSuperflex": bool(is_superflex),
        "viewerUserId": str(viewer_user_id) if viewer_user_id else "",
    }
    cfg_json = json.dumps(cfg)
    return (
        f'<script>window.__draftCfg = {cfg_json};</script>\n'
        + _DRAFT_ROOM_HTML
    )


# Plain (non-f) string — safe to contain { } freely.
_DRAFT_ROOM_HTML = r"""
<div class="dr-wrap">
  <div class="dr-hero">
    <h1 class="dr-title">Draft Assistant</h1>
    <p class="dr-sub">Build your board, draft manually, and see the best available in real time.
      Live draft sync is coming soon.</p>
  </div>

  <!-- Setup -->
  <div class="dr-setup card" id="drSetup">
    <div class="dr-setup-grid">
      <label class="dr-field"><span>Draft Type</span>
        <select id="drType">
          <option value="startup">Startup (Dynasty)</option>
          <option value="rookie">Rookie (Dynasty)</option>
          <option value="redraft">Redraft</option>
        </select>
      </label>
      <label class="dr-field"><span>Teams</span>
        <select id="drTeams">
          <option>8</option><option>10</option><option selected>12</option><option>14</option>
        </select>
      </label>
      <label class="dr-field"><span>Rounds</span>
        <input id="drRounds" type="number" min="1" max="40" value="15">
      </label>
      <label class="dr-field"><span>Your Pick</span>
        <select id="drSlot"></select>
      </label>
      <label class="dr-field"><span>Format</span>
        <select id="drSf">
          <option value="0">1QB</option>
          <option value="1">Superflex</option>
        </select>
      </label>
      <label class="dr-field"><span>Order</span>
        <select id="drOrder">
          <option value="snake">Snake</option>
          <option value="linear">Linear</option>
          <option value="3rr">3rd Round Reversal</option>
        </select>
      </label>
    </div>
    <div class="dr-setup-actions">
      <button class="dr-btn dr-btn-primary" id="drStart">Start Manual Draft</button>
      <button class="dr-btn" id="drConnect">Connect Live Draft</button>
      <span class="dr-setup-note" id="drResumeNote" style="display:none;"></span>
    </div>
    <div class="dr-live-list" id="drLiveList" style="display:none;"></div>
  </div>

  <!-- Board + side -->
  <div class="dr-main" id="drMain" style="display:none;">
    <div class="dr-statusbar">
      <div class="dr-status-left">
        <span class="dr-pill" id="drRoundPill">Round 1</span>
        <span class="dr-pill" id="drPickPill">Pick 1</span>
        <span class="dr-onclock">On the clock: <b id="drOnClock">Team 1</b></span>
        <span class="dr-pill dr-pill-you" id="drNextPill" style="display:none;"></span>
        <span class="dr-pill dr-pill-live" id="drLiveBadge" style="display:none;">&#9679; LIVE</span>
        <span class="dr-progress" id="drProgress"></span>
        <span class="dr-save" id="drSave"></span>
      </div>
      <div class="dr-status-right">
        <button class="dr-btn dr-btn-ghost" id="drUndo">Undo</button>
        <button class="dr-btn dr-btn-ghost" id="drEdit">Edit Setup</button>
        <button class="dr-btn dr-btn-ghost dr-btn-danger" id="drReset">Reset</button>
      </div>
    </div>

    <div class="dr-cols">
      <div class="dr-board-wrap">
        <div class="dr-board" id="drBoard"></div>
      </div>
      <aside class="dr-side">
        <div class="dr-side-head">
          <div class="dr-side-title">Best Available</div>
          <div class="dr-side-controls">
            <input id="drSearch" type="search" placeholder="Search…" autocomplete="off">
            <select id="drBaSort">
              <option value="value">Value</option>
              <option value="adp">ADP</option>
            </select>
          </div>
          <div class="dr-pos-filters" id="drPosFilters">
            <button class="dr-pos active" data-pos="ALL">All</button>
            <button class="dr-pos" data-pos="QB">QB</button>
            <button class="dr-pos" data-pos="RB">RB</button>
            <button class="dr-pos" data-pos="WR">WR</button>
            <button class="dr-pos" data-pos="TE">TE</button>
          </div>
        </div>
        <div class="dr-ba-list" id="drBaList">
          <div class="dr-loading"><div class="loading-spinner" style="width:22px;height:22px;"></div><span>Loading players…</span></div>
        </div>
      </aside>
    </div>
  </div>
</div>

<style>
  .dr-wrap { max-width: 1200px; margin: 0 auto; padding: 12px 14px 48px; }
  .dr-hero { margin-bottom: 14px; }
  .dr-title { font-size: clamp(20px,4vw,28px); font-weight: 800; color: var(--text); margin: 0 0 4px; }
  .dr-sub { font-size: 14px; color: var(--text-muted); margin: 0; }
  .dr-setup { padding: 16px; }
  .dr-setup-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(140px,1fr)); gap: 12px; }
  .dr-field { display: flex; flex-direction: column; gap: 5px; font-size: 12px; font-weight: 600; color: var(--text-muted); }
  .dr-field select, .dr-field input {
    padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--card); color: var(--text); font-size: 13px; outline: none; min-height: 36px;
  }
  .dr-setup-actions { margin-top: 14px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .dr-setup-note { font-size: 12px; color: var(--text-muted); }
  .dr-btn {
    padding: 9px 16px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); white-space: nowrap;
  }
  .dr-btn-primary { background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); color: #fff; }
  .dr-btn-ghost { background: transparent; font-weight: 600; }
  .dr-btn-danger { color: #ef4444; border-color: rgba(239,68,68,.4); }
  .dr-statusbar {
    display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    padding: 10px 12px; margin-bottom: 12px; border: 1px solid var(--border); border-radius: 10px;
    background: var(--card);
    position: sticky; top: 56px; z-index: 30;
  }
  .dr-status-left { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .dr-status-right { display: flex; align-items: center; gap: 6px; }
  .dr-pill { font-size: 12px; font-weight: 700; padding: 3px 9px; border-radius: 999px;
    background: rgba(56,189,248,.14); color: var(--accent,#38bdf8); }
  .dr-pill-you { background: rgba(34,197,94,.16); color: #22c55e; }
  .dr-pill-live { background: rgba(239,68,68,.16); color: #ef4444; animation: drPulse 1.6s ease-in-out infinite; }
  .dr-progress { font-size: 12px; color: var(--text-muted); }
  .dr-save { font-size: 11px; color: #22c55e; }
  .dr-live-list { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
  .dr-live-head { font-size: 12px; font-weight: 700; color: var(--text-muted); }
  .dr-live-item { text-align: left; padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 13px; cursor: pointer; }
  .dr-live-item:hover { border-color: var(--accent,#38bdf8); }
  .dr-live-status { font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 1px 6px; border-radius: 999px; margin-right: 6px; }
  .dr-ls-drafting { background: rgba(239,68,68,.16); color: #ef4444; }
  .dr-ls-pre_draft { background: rgba(245,158,11,.16); color: #f59e0b; }
  .dr-ls-complete { background: rgba(148,163,184,.16); color: #94a3b8; }
  .dr-onclock { font-size: 13px; color: var(--text-muted); }
  .dr-onclock b { color: var(--text); }
  .dr-cols { display: grid; grid-template-columns: 1fr 340px; gap: 14px; align-items: start; }
  .dr-board-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--card); padding: 8px; }
  .dr-board { display: grid; gap: 6px; min-width: max-content; }
  .dr-cell {
    border: 1px solid var(--border); border-radius: 8px; padding: 6px; min-height: 52px;
    background: var(--bg); display: flex; align-items: center; gap: 7px; position: relative;
  }
  .dr-cell-empty { opacity: .45; }
  .dr-cell-filled { background: linear-gradient(180deg, rgba(56,189,248,.05), var(--bg)); }
  .dr-cell-current { outline: 2px solid var(--accent,#38bdf8); animation: drPulse 1.6s ease-in-out infinite; }
  @keyframes drPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(56,189,248,.0); } 50% { box-shadow: 0 0 0 3px rgba(56,189,248,.18); } }
  .dr-cell-mine { box-shadow: inset 3px 0 0 var(--accent,#38bdf8); }
  .dr-cell-just { animation: drPop .35s ease; }
  @keyframes drPop { 0% { transform: scale(.92); opacity: .3; } 100% { transform: scale(1); opacity: 1; } }
  .dr-cell-val { position: absolute; top: 2px; right: 5px; font-size: 9px; font-weight: 800; color: var(--accent,#38bdf8); }
  .dr-cell-num { position: absolute; top: 2px; left: 5px; font-size: 9px; font-weight: 700; color: var(--text-muted); }
  .dr-hs { width: 34px; height: 34px; border-radius: 6px; object-fit: cover; flex-shrink: 0; background: rgba(0,0,0,.15); }
  .dr-cell-body { min-width: 0; line-height: 1.2; }
  .dr-cell-name { font-size: 12px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 96px; }
  .dr-cell-meta { font-size: 10px; color: var(--text-muted); }
  .dr-posbadge { font-size: 9px; font-weight: 700; color: #fff; border-radius: 3px; padding: 1px 4px; }
  .dr-colhead { font-size: 11px; font-weight: 700; color: var(--text-muted); text-align: center; padding: 2px 0; white-space: nowrap; }
  .dr-colhead-you { color: var(--accent,#38bdf8); }
  .dr-side { border: 1px solid var(--border); border-radius: 10px; background: var(--card); display: flex; flex-direction: column;
    position: sticky; top: 120px; align-self: start; max-height: calc(100vh - 134px); z-index: 20; }
  .dr-side-head { padding: 10px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }
  .dr-side-title { font-size: 14px; font-weight: 800; color: var(--text); }
  .dr-side-controls { display: flex; gap: 6px; }
  .dr-side-controls input { flex: 1; min-width: 0; padding: 7px 9px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; }
  .dr-side-controls select { padding: 7px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; }
  .dr-pos-filters { display: flex; gap: 4px; flex-wrap: wrap; }
  .dr-pos { font-size: 11px; font-weight: 700; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--border); background: var(--bg); color: var(--text-muted); cursor: pointer; }
  .dr-pos.active { background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); color: #fff; }
  .dr-ba-list { overflow-y: auto; flex: 1; }
  .dr-ba-row { display: flex; align-items: center; gap: 8px; padding: 7px 10px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .dr-ba-row:hover { background: rgba(56,189,248,.08); }
  .dr-ba-hs { width: 30px; height: 30px; border-radius: 5px; object-fit: cover; flex-shrink: 0; background: rgba(0,0,0,.15); }
  .dr-ba-body { min-width: 0; flex: 1; line-height: 1.25; }
  .dr-ba-name { font-size: 13px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-ba-meta { font-size: 11px; color: var(--text-muted); }
  .dr-ba-right { text-align: right; flex-shrink: 0; }
  .dr-ba-val { font-size: 13px; font-weight: 800; color: var(--text); }
  .dr-ba-adp { font-size: 10px; color: var(--text-muted); }
  .dr-loading { display: flex; align-items: center; gap: 10px; padding: 24px; color: var(--text-muted); font-size: 13px; justify-content: center; }
  @media (max-width: 900px) {
    .dr-cols { grid-template-columns: 1fr; }
    .dr-side { position: static; max-height: none; order: -1; }
    .dr-statusbar { top: 0; }
  }
</style>

<script>
(function(){
  var cfg = window.__draftCfg || {};
  var POS_COLOR = { QB:'#f59e0b', RB:'#22c55e', WR:'#3b82f6', TE:'#8b5cf6', K:'#94a3b8', DEF:'#64748b' };
  var posColor = function(p){ return POS_COLOR[(p||'').toUpperCase()] || '#94a3b8'; };
  var hsUrl = function(id){ return 'https://sleepercdn.com/content/nfl/players/' + id + '.jpg'; };

  var sessKey = 'dr_' + location.pathname;
  var state = null;        // { type, teams, rounds, sf, slot, order, picks:{}, current }
  var players = [];        // best-available pool
  var drafted = {};        // id -> true
  var posFilter = 'ALL';
  var justPick = null;     // pick # filled this render (for the pop-in animation)
  var playersById = {};    // id -> player (value lookup for live picks)
  var lastLivePicks = null;// last picks payload from the live feed
  var saveTimer = null;    // debounce for DB autosave
  var pollTimer = null;    // live-draft poll interval

  // ── Pick-order helper (snake / linear / 3rr) ───────────────────────────────
  function pickDir(r, order){            // true = forward (slot 1 → N)
    if (order === 'linear') return true;
    if (order === '3rr') {
      if (r === 1) return true;
      if (r === 2 || r === 3) return false;
      return (r % 2 === 0);
    }
    return (r % 2 === 1);                 // snake
  }
  function pickNum(r, slot, teams, order){
    var inRound = pickDir(r, order) ? slot : (teams - slot + 1);
    return (r - 1) * teams + inRound;
  }
  function slotOnClock(pickNo, teams, order){
    var r = Math.ceil(pickNo / teams);
    var posInRound = pickNo - (r - 1) * teams;
    return pickDir(r, order) ? posInRound : (teams - posInRound + 1);
  }
  window.draftPickOrder = { pickDir: pickDir, pickNum: pickNum, slotOnClock: slotOnClock };

  // ── Persistence ────────────────────────────────────────────────────────────
  function save(){ try { sessionStorage.setItem(sessKey, JSON.stringify(state)); } catch(e){} }
  function load(){ try { return JSON.parse(sessionStorage.getItem(sessKey) || 'null'); } catch(e){ return null; } }

  function fillSlotOptions(teams){
    var sel = document.getElementById('drSlot');
    sel.innerHTML = '';
    for (var i = 1; i <= teams; i++){
      var o = document.createElement('option');
      o.value = i; o.textContent = 'Pick ' + i;
      sel.appendChild(o);
    }
  }

  // ── Setup ────────────────────────────────────────────────────────────────
  function applyCfgDefaults(){
    if (cfg.numTeams) {
      var t = document.getElementById('drTeams');
      var want = String(Math.min(14, Math.max(8, cfg.numTeams)));
      for (var i=0;i<t.options.length;i++){ if (t.options[i].value === want || t.options[i].text === want){ t.selectedIndex = i; break; } }
    }
    if (cfg.isSuperflex) document.getElementById('drSf').value = '1';
    fillSlotOptions(parseInt(document.getElementById('drTeams').value, 10));
  }

  document.getElementById('drTeams').addEventListener('change', function(){
    fillSlotOptions(parseInt(this.value, 10));
  });
  document.getElementById('drType').addEventListener('change', function(){
    document.getElementById('drRounds').value = (this.value === 'rookie') ? '4' : '15';
  });

  function readSetup(){
    var teams = parseInt(document.getElementById('drTeams').value, 10);
    return {
      type:   document.getElementById('drType').value,
      teams:  teams,
      rounds: Math.max(1, Math.min(40, parseInt(document.getElementById('drRounds').value, 10) || 15)),
      sf:     document.getElementById('drSf').value === '1',
      slot:   Math.min(teams, Math.max(1, parseInt(document.getElementById('drSlot').value, 10) || 1)),
      order:  document.getElementById('drOrder').value,
      picks:  {},
      current: 1
    };
  }

  function startDraft(){
    state = readSetup();
    drafted = {};
    save();
    showMain();
    loadPlayers();
  }

  function showMain(){
    document.getElementById('drSetup').style.display = 'none';
    document.getElementById('drMain').style.display = '';
  }
  function showSetup(){
    document.getElementById('drMain').style.display = 'none';
    document.getElementById('drSetup').style.display = '';
  }

  // ── Data ─────────────────────────────────────────────────────────────────
  function redraftVal(p){
    return (state.sf ? (p.redraft_value_sf != null ? p.redraft_value_sf : p.redraft_value_1qb)
                     : p.redraft_value_1qb) || 0;
  }
  function valOf(p){
    if (state.type === 'redraft') return redraftVal(p);
    return state.sf ? (p.sf_value || p.value || 0) : (p.value || 0);
  }
  function adpOf(p){
    // Redraft has no market ADP feed, so derive a rank from redraft value.
    if (state.type === 'redraft') return (p._radp != null ? p._radp : null);
    return state.sf ? p.sf_avg_pick : p.avg_pick;
  }

  function loadPlayers(){
    fetch('/api/league-players', { cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(function(resp){
        var raw = Array.isArray(resp) ? resp : (resp.players || []);
        players = raw.filter(function(p){
          if (!p || p.id == null) return false;
          var pos = String(p.position || '').toUpperCase();
          if (pos === 'PICK') return false;
          if (state.type === 'rookie' && !p.is_rookie) return false;
          if (state.type === 'redraft') return redraftVal(p) > 0;  // must have a redraft value
          return ['QB','RB','WR','TE'].indexOf(pos) >= 0 || p.is_rookie;
        });
        // Derive a stable redraft ADP rank (1 = top redraft value).
        if (state.type === 'redraft'){
          players.slice().sort(function(a, b){ return redraftVal(b) - redraftVal(a); })
            .forEach(function(p, i){ p._radp = i + 1; });
        }
        playersById = {};
        players.forEach(function(p){ playersById[String(p.id)] = p; });
        // Live mode: re-apply picks now that values are available; else rebuild
        // the drafted set from saved picks.
        if (state.mode === 'live' && lastLivePicks){
          applyLivePicks(lastLivePicks);
        } else {
          drafted = {};
          Object.keys(state.picks).forEach(function(k){ var pp = state.picks[k]; if (pp) drafted[String(pp.id)] = true; });
        }
        render();
      })
      .catch(function(){
        document.getElementById('drBaList').innerHTML =
          '<div class="dr-loading">Could not load players. Refresh to retry.</div>';
      });
  }

  // ── Render ───────────────────────────────────────────────────────────────
  function render(){ renderStatus(); renderBoard(); renderBA(); justPick = null; save(); }

  // ── Live draft (P5, Sleeper) ────────────────────────────────────────────────
  function valLookup(id){ var p = playersById[String(id)]; return (p && state) ? Math.round(valOf(p)) : null; }
  function applyLivePicks(picks){
    lastLivePicks = picks;
    state.picks = {}; drafted = {};
    picks.forEach(function(p){
      if (p.pick_no == null) return;
      state.picks[p.pick_no] = { id: p.player_id, name: p.name, position: p.position, team: p.team, val: valLookup(p.player_id) };
      if (p.player_id) drafted[String(p.player_id)] = true;
    });
    state.current = (picks.length || 0) + 1;
  }
  function detectLive(){
    if (cfg.isGuest || !cfg.leagueId){
      alert('Live draft sync requires opening the Draft Room from your league.');
      return;
    }
    var box = document.getElementById('drLiveList');
    box.style.display = ''; box.innerHTML = '<div class="dr-live-head">Detecting drafts…</div>';
    fetch('/api/draft/detect?platform=' + encodeURIComponent(cfg.platform) + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || ''))
      .then(function(r){ return r.json(); })
      .then(function(resp){
        if (resp.unsupported){ box.innerHTML = '<div class="dr-live-head">Live sync currently supports Sleeper leagues.</div>'; return; }
        var ds = resp.drafts || [];
        if (!ds.length){ box.innerHTML = '<div class="dr-live-head">No drafts found for this league yet.</div>'; return; }
        var html = '<div class="dr-live-head">Detected drafts — pick one to connect</div>';
        ds.forEach(function(d){
          html += '<button class="dr-live-item" data-id="' + esc(d.draft_id) + '">'
            + '<span class="dr-live-status dr-ls-' + esc(d.status || '') + '">' + esc(d.status || '') + '</span>'
            + esc((d.type || 'snake') + ' · ' + (d.teams || '?') + ' teams · ' + (d.rounds || '?') + ' rounds') + '</button>';
        });
        box.innerHTML = html;
      })
      .catch(function(){ box.innerHTML = '<div class="dr-live-head">Could not detect drafts.</div>'; });
  }
  function connectLive(draftId){
    fetch('/api/draft/live?platform=' + encodeURIComponent(cfg.platform) + '&draft_id=' + encodeURIComponent(draftId))
      .then(function(r){ return r.json(); })
      .then(function(d){
        if (!d || d.error){ alert('Could not load that draft.'); return; }
        var teams = parseInt(d.teams || 0, 10) || (cfg.numTeams || 12);
        var rounds = parseInt(d.rounds || 0, 10) || 15;
        var slot = 0;
        if (cfg.viewerUserId && d.draft_order && d.draft_order[cfg.viewerUserId]) {
          slot = parseInt(d.draft_order[cfg.viewerUserId], 10) || 0;
        }
        state = {
          type: 'startup', teams: teams, rounds: rounds, sf: !!cfg.isSuperflex,
          slot: slot, order: d.order || 'snake', picks: {}, current: 1,
          mode: 'live', sourceDraftId: draftId
        };
        applyLivePicks(d.picks || []);
        showMain();
        document.getElementById('drLiveBadge').style.display = '';
        document.getElementById('drUndo').style.display = 'none';
        loadPlayers();
        startPolling();
        if (String(d.status) === 'complete') stopPolling();
      })
      .catch(function(){ alert('Could not connect to the live draft.'); });
  }
  function startPolling(){
    stopPolling();
    pollTimer = setInterval(function(){
      if (!state || state.mode !== 'live') { stopPolling(); return; }
      fetch('/api/draft/live?platform=' + encodeURIComponent(cfg.platform) + '&draft_id=' + encodeURIComponent(state.sourceDraftId))
        .then(function(r){ return r.json(); })
        .then(function(d){
          if (d && d.picks){ applyLivePicks(d.picks); render(); if (String(d.status) === 'complete') stopPolling(); }
        })
        .catch(function(){});
    }, 5000);
  }
  function stopPolling(){ if (pollTimer){ clearInterval(pollTimer); pollTimer = null; } }


  function userNextPick(){
    var total = state.teams * state.rounds;
    for (var pn = state.current; pn <= total; pn++){
      if (slotOnClock(pn, state.teams, state.order) === state.slot) return pn;
    }
    return null;
  }

  function renderStatus(){
    var total = state.teams * state.rounds;
    var done = state.current > total;
    var r = Math.ceil(state.current / state.teams);
    document.getElementById('drRoundPill').textContent = done ? 'Complete' : ('Round ' + r);
    document.getElementById('drPickPill').textContent  = done ? (total + ' picks') : ('Pick ' + state.current);
    var oc = document.getElementById('drOnClock');
    if (done) { oc.textContent = 'Draft complete'; }
    else {
      var slot = slotOnClock(state.current, state.teams, state.order);
      oc.textContent = (slot === state.slot) ? ('Team ' + slot + ' (You)') : ('Team ' + slot);
    }
    var nextPill = document.getElementById('drNextPill');
    var np = done ? null : userNextPick();
    if (np){ nextPill.style.display = ''; nextPill.textContent = 'Your next: #' + np + ' (R' + Math.ceil(np / state.teams) + ')'; }
    else { nextPill.style.display = 'none'; }
    document.getElementById('drProgress').textContent = Math.min(state.current - 1, total) + ' / ' + total + ' picks';
  }

  function renderBoard(){
    var board = document.getElementById('drBoard');
    var teams = state.teams, rounds = state.rounds;
    board.style.gridTemplateColumns = '34px repeat(' + teams + ', minmax(116px, 1fr))';
    var html = '';
    // header row
    html += '<div class="dr-colhead"></div>';
    for (var s = 1; s <= teams; s++){
      var you = (s === state.slot) ? ' dr-colhead-you' : '';
      html += '<div class="dr-colhead' + you + '">Team ' + s + (s === state.slot ? ' ★' : '') + '</div>';
    }
    for (var rnd = 1; rnd <= rounds; rnd++){
      html += '<div class="dr-colhead">R' + rnd + '</div>';
      for (var slot = 1; slot <= teams; slot++){
        var pn = pickNum(rnd, slot, teams, state.order);
        var pl = state.picks[pn];
        var isCurrent = (pn === state.current);
        var mine = (slot === state.slot);
        var cls = 'dr-cell' + (pl ? ' dr-cell-filled' : ' dr-cell-empty')
          + (isCurrent ? ' dr-cell-current' : '') + (mine ? ' dr-cell-mine' : '')
          + (pn === justPick ? ' dr-cell-just' : '');
        html += '<div class="' + cls + '">';
        html += '<span class="dr-cell-num">' + pn + '</span>';
        if (pl){
          if (pl.val != null) html += '<span class="dr-cell-val">' + Math.round(pl.val) + '</span>';
          html += '<img class="dr-hs" src="' + hsUrl(pl.id) + '" alt="" onerror="this.style.visibility=\'hidden\'">';
          html += '<div class="dr-cell-body">';
          html += '<div class="dr-cell-name">' + esc(pl.name) + '</div>';
          html += '<div class="dr-cell-meta"><span class="dr-posbadge" style="background:' + posColor(pl.position) + '">' + esc(pl.position) + '</span> ' + esc(pl.team || '') + '</div>';
          html += '</div>';
        }
        html += '</div>';
      }
    }
    board.innerHTML = html;
  }

  function renderBA(){
    var listEl = document.getElementById('drBaList');
    var sortBy = document.getElementById('drBaSort').value;
    var q = (document.getElementById('drSearch').value || '').trim().toLowerCase();
    var pool = players.filter(function(p){
      if (drafted[String(p.id)]) return false;
      if (posFilter !== 'ALL' && String(p.position||'').toUpperCase() !== posFilter) return false;
      if (q && String(p.name||'').toLowerCase().indexOf(q) < 0) return false;
      return true;
    });
    pool.sort(function(a, b){
      if (sortBy === 'adp'){
        var aa = adpOf(a), ba = adpOf(b);
        return (aa != null ? aa : 99999) - (ba != null ? ba : 99999);
      }
      return valOf(b) - valOf(a);
    });
    if (!pool.length){ listEl.innerHTML = '<div class="dr-loading">No players match.</div>'; return; }
    var html = '';
    for (var i = 0; i < Math.min(pool.length, 200); i++){
      var p = pool[i];
      var adp = adpOf(p);
      html += '<div class="dr-ba-row" data-id="' + esc(String(p.id)) + '">';
      html += '<img class="dr-ba-hs" src="' + hsUrl(p.id) + '" alt="" onerror="this.style.visibility=\'hidden\'">';
      html += '<div class="dr-ba-body"><div class="dr-ba-name">' + esc(p.name) + '</div>';
      html += '<div class="dr-ba-meta"><span class="dr-posbadge" style="background:' + posColor(p.position) + '">' + esc(p.position) + '</span> ' + esc(p.team || '') + '</div></div>';
      html += '<div class="dr-ba-right"><div class="dr-ba-val">' + Math.round(valOf(p)) + '</div>';
      html += '<div class="dr-ba-adp">' + (adp != null ? ('ADP ' + Number(adp).toFixed(1)) : '') + '</div></div>';
      html += '</div>';
    }
    listEl.innerHTML = html;
  }

  function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]; }); }

  // ── Actions ──────────────────────────────────────────────────────────────
  function draftPlayer(id){
    if (state.mode === 'live') return;   // live board is driven by the platform
    var total = state.teams * state.rounds;
    if (state.current > total) return;
    var p = players.filter(function(x){ return String(x.id) === String(id); })[0];
    if (!p || drafted[String(id)]) return;
    state.picks[state.current] = { id: p.id, name: p.name, position: p.position, team: p.team, val: Math.round(valOf(p)) };
    drafted[String(id)] = true;
    justPick = state.current;
    state.current++;
    render();
  }
  function undo(){
    if (state.current <= 1) return;
    state.current--;
    var p = state.picks[state.current];
    if (p) { delete drafted[String(p.id)]; delete state.picks[state.current]; }
    render();
  }
  function resetDraft(){
    if (!confirm('Reset the draft board?')) return;
    try { sessionStorage.removeItem(sessKey); } catch(e){}
    state = null;
    showSetup();
  }

  // ── Wire up ──────────────────────────────────────────────────────────────
  document.getElementById('drStart').addEventListener('click', startDraft);
  document.getElementById('drConnect').addEventListener('click', detectLive);
  document.getElementById('drLiveList').addEventListener('click', function(e){
    var b = e.target.closest('.dr-live-item'); if (b) connectLive(b.getAttribute('data-id'));
  });
  document.getElementById('drUndo').addEventListener('click', undo);
  document.getElementById('drReset').addEventListener('click', resetDraft);
  document.getElementById('drEdit').addEventListener('click', showSetup);
  document.getElementById('drBaSort').addEventListener('change', renderBA);
  document.getElementById('drSearch').addEventListener('input', renderBA);
  document.getElementById('drBaList').addEventListener('click', function(e){
    var row = e.target.closest('.dr-ba-row');
    if (row) draftPlayer(row.getAttribute('data-id'));
  });
  document.getElementById('drPosFilters').addEventListener('click', function(e){
    var b = e.target.closest('.dr-pos'); if (!b) return;
    posFilter = b.getAttribute('data-pos');
    this.querySelectorAll('.dr-pos').forEach(function(x){ x.classList.toggle('active', x === b); });
    renderBA();
  });

  applyCfgDefaults();

  function resumeFromSession(){
    var saved = load();
    if (saved && saved.teams && saved.picks){
      state = saved;
      if (state.mode === 'live'){
        document.getElementById('drLiveBadge').style.display = '';
        document.getElementById('drUndo').style.display = 'none';
      }
      showMain();
      loadPlayers();
      if (state.mode === 'live' && state.sourceDraftId) startPolling();
    }
  }

  // Open a specific league draft from history (?live=<draft_id>), else resume
  // the in-progress session draft.
  var urlLive = new URLSearchParams(location.search).get('live');
  if (urlLive){
    connectLive(urlLive);
  } else {
    resumeFromSession();
  }
})();
</script>
"""


def build_draft_history_body(
    league_id: Optional[str],
    season: Optional[int],
    platform: Optional[str] = None,
) -> str:
    """Draft History page: the league's real drafts (from Sleeper), openable by
    any league member to review the board."""
    has_league = bool(league_id and platform and season)
    base = f"/{platform}/{int(season)}/{league_id}/draft" if has_league else "/draft"
    cfg = {
        "base": base,
        "leagueId": league_id or "",
        "platform": platform or "sleeper",
        "season": int(season) if season else None,
        "hasLeague": has_league,
    }
    cfg_json = json.dumps(cfg)
    return (
        f'<script>window.__draftHistCfg = {cfg_json};</script>\n'
        + _DRAFT_HISTORY_HTML
    )


_DRAFT_HISTORY_HTML = r"""
<div class="dr-wrap">
  <div class="dr-hero">
    <h1 class="dr-title">Draft History</h1>
    <p class="dr-sub">Your league's drafts. Open any board to review the picks pick-by-pick.</p>
  </div>
  <div id="drHistList" class="dr-hist-list">
    <div class="dr-loading"><div class="loading-spinner" style="width:22px;height:22px;"></div><span>Loading…</span></div>
  </div>
</div>

<style>
  .dr-wrap { max-width: 900px; margin: 0 auto; padding: 12px 14px 48px; }
  .dr-hero { margin-bottom: 14px; }
  .dr-title { font-size: clamp(20px,4vw,28px); font-weight: 800; color: var(--text); margin: 0 0 4px; }
  .dr-sub { font-size: 14px; color: var(--text-muted); margin: 0; }
  .dr-hist-list { display: flex; flex-direction: column; gap: 10px; }
  .dr-hist-card { display: flex; align-items: center; gap: 12px; padding: 14px 16px; border: 1px solid var(--border);
    border-radius: 10px; background: var(--card); }
  .dr-hist-body { flex: 1; min-width: 0; }
  .dr-hist-title { font-size: 15px; font-weight: 700; color: var(--text); }
  .dr-hist-meta { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
  .dr-hist-tag { font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 1px 7px; border-radius: 999px;
    background: rgba(56,189,248,.14); color: var(--accent,#38bdf8); margin-right: 6px; }
  .dr-hist-tag-live { background: rgba(239,68,68,.16); color: #ef4444; }
  .dr-hist-tag-complete { background: rgba(148,163,184,.16); color: #94a3b8; }
  .dr-hist-actions { display: flex; gap: 6px; flex-shrink: 0; }
  .dr-btn { padding: 8px 14px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); text-decoration: none; }
  .dr-btn-primary { background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); color: #fff; }
  .dr-btn-danger { color: #ef4444; border-color: rgba(239,68,68,.4); background: transparent; }
  .dr-loading { display: flex; align-items: center; gap: 10px; padding: 24px; color: var(--text-muted); font-size: 13px; justify-content: center; }
  .dr-hist-empty { padding: 28px; text-align: center; color: var(--text-muted); font-size: 14px; }
</style>

<script>
(function(){
  var cfg = window.__draftHistCfg || { base: '/draft', hasLeague: false };
  var listEl = document.getElementById('drHistList');

  function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]; }); }

  function statusTag(s){
    var c = (s === 'drafting') ? 'dr-hist-tag-live' : (s === 'complete' ? 'dr-hist-tag-complete' : '');
    var label = (s === 'drafting') ? 'Live now' : (s === 'pre_draft' ? 'Upcoming' : (s === 'complete' ? 'Complete' : (s || '')));
    return '<span class="dr-hist-tag ' + c + '">' + esc(label) + '</span>';
  }

  function render(drafts){
    if (!drafts.length){
      listEl.innerHTML = '<div class="dr-hist-empty">No drafts found for this league yet.</div>';
      return;
    }
    // Live/upcoming first, then completed.
    var rank = { drafting: 0, pre_draft: 1, complete: 2 };
    drafts.sort(function(a, b){ return (rank[a.status] != null ? rank[a.status] : 3) - (rank[b.status] != null ? rank[b.status] : 3); });
    var html = '';
    drafts.forEach(function(d){
      var title = (d.type ? (d.type.charAt(0).toUpperCase() + d.type.slice(1)) : 'Draft')
        + ' · ' + (d.teams || '?') + ' teams · ' + (d.rounds || '?') + ' rounds';
      html += '<div class="dr-hist-card">'
        + '<div class="dr-hist-body"><div class="dr-hist-title">' + statusTag(d.status) + esc(title) + '</div>'
        + '<div class="dr-hist-meta">' + esc((d.order || 'snake')) + ' order' + (d.season ? (' · ' + esc(String(d.season))) : '') + '</div></div>'
        + '<div class="dr-hist-actions">'
        + '<a class="dr-btn dr-btn-primary" href="' + esc(cfg.base) + '?live=' + encodeURIComponent(d.draft_id) + '">Open board</a>'
        + '</div></div>';
    });
    listEl.innerHTML = html;
  }

  function loadList(){
    if (!cfg.hasLeague){
      listEl.innerHTML = '<div class="dr-hist-empty">Open Draft History from your league to see its drafts. '
        + 'You can still run a mock in the <a href="' + esc(cfg.base) + '">Draft Room</a>.</div>';
      return;
    }
    fetch('/api/draft/detect?platform=' + encodeURIComponent(cfg.platform)
        + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || ''), { cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(function(resp){
        if (resp.unsupported){ listEl.innerHTML = '<div class="dr-hist-empty">Draft history is available for Sleeper leagues.</div>'; return; }
        render(resp.drafts || []);
      })
      .catch(function(){ listEl.innerHTML = '<div class="dr-hist-empty">Could not load drafts.</div>'; });
  }

  loadList();
})();
</script>
"""
