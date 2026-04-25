"""
Rookies page HTML builder.

Returns the full body HTML for the /<platform>/<season>/<league_id>/rookies route.
Data is loaded client-side from /api/prospects/rankings so the page is fast and
the filters/sorts are instant without round-trips.
"""
from __future__ import annotations


def build_prospects_body(platform: str, season: int, league_id: str) -> str:
    return """
<div class="card central">
  <div class="card-header">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px;">
      <div>
        <h2 id="rookiesTitle">Prospect Rankings</h2>
        <div style="font-size: 14px; color: var(--text-muted); margin-top: 4px;">
          Dynasty prospect rankings — production, athleticism, and draft capital combined
        </div>
      </div>
    </div>
  </div>

  <div class="card-body" style="padding-top:0;">

    <!-- Controls -->
    <div class="filter-controls-container">
      <!-- Row 1: Search -->
      <div class="filter-row">
        <div class="filter-search">
          <input id="rookieSearch" type="text" placeholder="Search prospects…" autocomplete="off"
            style="width:100%;padding:8px 32px 8px 34px;border-radius:8px;
                   border:1px solid var(--border);background:var(--card-bg);
                   color:var(--text);font-size:13px;outline:none;box-sizing:border-box;">
          <span style="position:absolute;left:10px;top:50%;transform:translateY(-50%);
                       color:var(--text-muted);font-size:14px;pointer-events:none;"><i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i></span>
          <button id="rookieSearchClear" onclick="rkClearSearch()"
            style="display:none;position:absolute;right:8px;top:50%;transform:translateY(-50%);
                   background:none;border:none;cursor:pointer;color:var(--text-muted);
                   font-size:16px;padding:2px;">&#x2715;</button>
        </div>
        <div class="filter-row rk-pills-row">
        <div class="filter-positions">
          <button class="pos-pill active" data-pos="ALL" onclick="rkTogglePos('ALL')">All</button>
          <button class="pos-pill" data-pos="QB"  onclick="rkTogglePos('QB')">QB</button>
          <button class="pos-pill" data-pos="RB"  onclick="rkTogglePos('RB')">RB</button>
          <button class="pos-pill" data-pos="WR"  onclick="rkTogglePos('WR')">WR</button>
          <button class="pos-pill" data-pos="TE"  onclick="rkTogglePos('TE')">TE</button>
        </div>
        <div class="rk-settings-wrapper">
          <button id="rkSettingsBtn" class="filter-settings-btn" onclick="rkToggleSettings()">
            League️ Settings
          </button>
          <div id="rkSettingsPanel" class="filter-settings-panel" style="display:none;">
            <div class="settings-section">
              <span class="settings-section-label">League Format</span>
              <div class="settings-toggle-group">
                <button class="settings-toggle active" data-value="1qb" onclick="rkSetLeague('1qb')">1QB</button>
                <button class="settings-toggle" data-value="sf" onclick="rkSetLeague('sf')">SF</button>
              </div>
            </div>
            <div class="settings-section">
              <span class="settings-section-label">League Size</span>
              <div class="settings-toggle-group">
                <button class="settings-toggle" data-value="8" onclick="rkSetSize(8)">8</button>
                <button class="settings-toggle active" data-value="10" onclick="rkSetSize(10)">10</button>
                <button class="settings-toggle" data-value="12" onclick="rkSetSize(12)">12</button>
                <button class="settings-toggle" data-value="14" onclick="rkSetSize(14)">14</button>
              </div>
            </div>
          </div>
        </div>
      </div>
      </div>


      <!-- Row 3: Sort + Active settings -->
      <div class="filter-row filter-row-secondary">
        <div id="rkActiveSettings" class="active-settings-indicator">
          <span class="active-setting-tag">10-Team</span>
          <span class="active-setting-tag">1QB</span>
        </div>
        <div class="filter-sort">
          <label class="filter-label">Sort by</label>
          <select id="rkSort" onchange="rkRender()"
            style="padding:7px 10px;border-radius:8px;border:1px solid var(--border);
                   background:var(--card-bg);color:var(--text);font-size:12px;
                   cursor:pointer;outline:none;min-height:34px;width:140px;">
            <option value="rank">Overall Rank</option>
            <option value="value">Value</option>
            <option value="score">Prospect Score</option>
            <option value="age">Age</option>
            <option value="pick">Draft Pick</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Count -->
    <div id="rkCount" style="font-size:12px;color:var(--text-muted);margin-bottom:8px;display:none;"></div>

    <!-- Table header -->
    <div id="rkHeader" class="rk-grid-row rk-header" style="display:none;">
      <span>#</span>
      <span>Prospect</span>
      <span style="text-align:center;">Pos</span>
      <span style="text-align:center;">Age</span>
      <span style="text-align:right;">Draft</span>
      <span style="text-align:right;">Score</span>
      <span style="text-align:right;">Value</span>
    </div>

    <!-- Loading -->
    <div id="rkLoading" style="text-align:center;padding:40px;color:var(--text-muted);">
      <div class="loading-spinner" style="margin:0 auto 12px;"></div>
      Loading rookie prospects…
    </div>

    <!-- Rows -->
    <div id="rkList"></div>

    <!-- Empty -->
    <div id="rkEmpty" style="display:none;text-align:center;padding:40px;color:var(--text-muted);">
      <div style="font-size:24px;margin-bottom:8px;"><i class="fa-solid fa-football" aria-hidden="true"></i></div>
      No prospects match your filters
    </div>

  </div>
</div>

<!-- Prospect detail modal -->
<div id="rkModal" style="display:none;position:fixed;inset:0;z-index:10500;
     display:none;align-items:center;justify-content:center;padding:20px;
     background:rgba(15,23,42,0.7);backdrop-filter:blur(4px);">
  <div id="rkModalContent"
    style="background:var(--card);border-radius:16px;max-width:680px;width:100%;
           max-height:90vh;overflow-y:auto;
           box-shadow:0 24px 48px rgba(15,23,42,0.25);">
    <!-- filled by JS -->
  </div>
</div>

<style>
  /* Filter Controls */
  .filter-controls-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 16px 0 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  .filter-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: space-between;
  }
  .filter-search {
    position: relative;
    flex: 1;
    min-width: 0;
  }
  .rk-pills-row {
    justify-content: space-between;
    flex-wrap: nowrap;
    gap: 12px;
  }
  .rk-settings-wrapper {
    position: relative;
    flex-shrink: 0;
  }
  .filter-positions {
    display: flex;
    gap: 3px;
    flex-wrap: wrap;
  }
  .pos-pill {
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--card-bg);
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.12s;
    white-space: nowrap;
  }
  .pos-pill.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .filter-settings-btn {
    padding: 7px 14px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--card-bg);
    color: var(--text);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    transition: all 0.12s;
  }
  .filter-settings-btn:hover {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent);
  }
  .filter-settings-panel {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 8px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    padding: 16px;
    min-width: 260px;
    z-index: 1000;
  }
  .settings-section {
    margin-bottom: 16px;
  }
  .settings-section:last-of-type {
    margin-bottom: 0;
  }
  .settings-section-label {
    display: block;
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
  }
  .settings-toggle-group {
    display: flex;
    gap: 6px;
  }
  .settings-toggle {
    flex: 1;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--card-bg);
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.12s;
  }
  .settings-toggle.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .active-settings-indicator {
    display: flex;
    gap: 6px;
    align-items: center;
    flex-wrap: wrap;
  }
  .active-setting-tag {
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
  }
  .filter-sort {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .filter-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  /* Mobile responsive */
  @media (max-width: 768px) {
    .rk-pills-row {
      flex-wrap: wrap;
    }
    .filter-row-secondary {
      flex-wrap: wrap;
      gap: 8px;
    }
  }

  /* Table grid */
  .rk-grid-row {
    display: grid;
    grid-template-columns: 40px 1fr 52px 48px 72px 58px 60px;
    align-items: center;
    gap: 0;
  }
  .rk-header {
    padding: 6px 12px;
    border-radius: 6px;
    background: var(--accent-soft);
    font-size: 11px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .rk-row {
    padding: 10px 12px;
    cursor: pointer;
    transition: background 0.12s;
    border-top: 1px solid var(--border);
  }
  .rk-row:hover { background: var(--accent-soft); }
  .rk-row:first-child { border-top: none; }

  .rk-rank { font-size: 12px; font-weight: 700; color: var(--text-muted); }
  .rk-name-cell { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .rk-name { font-size: 13px; font-weight: 600; color: var(--text); }
  .rk-name:hover { text-decoration: underline; }
  .rk-meta { font-size: 11px; color: var(--text-muted); }
  .rk-pos  { text-align: center; font-size: 11px; font-weight: 700; color: var(--text-muted); }
  .rk-age  { text-align: center; font-size: 12px; color: var(--text-muted); }
  .rk-draft { text-align: right; font-size: 11px; color: var(--text-muted); white-space: nowrap; }
  .rk-score { text-align: right; font-size: 12px; font-weight: 600; }
  .rk-value { text-align: right; font-size: 13px; font-weight: 700; color: var(--accent); }

  /* Tier badge */
  .rk-tier {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 4px;
    margin-left: 4px;
    vertical-align: middle;
  }
  .rk-tier-1 { background: rgba(16,185,129,0.15); color: #10b981; }
  .rk-tier-2 { background: rgba(59,130,246,0.15); color: #3b82f6; }
  .rk-tier-3 { background: rgba(139,92,246,0.15); color: #8b5cf6; }
  .rk-tier-4 { background: rgba(245,158,11,0.15); color: #f59e0b; }
  .rk-tier-5 { background: rgba(107,114,128,0.15); color: #6b7280; }
  .rk-tier-6 { background: rgba(107,114,128,0.10); color: #9ca3af; }

  /* Score bar */
  .rk-score-bar {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .rk-score-dot {
    width: 6px; height: 6px; border-radius: 50%;
  }

  /* Modal */
  .rk-modal-header {
    padding: 24px 24px 0;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }
  .rk-modal-close {
    background: var(--accent-soft);
    border: none;
    width: 32px; height: 32px;
    border-radius: 8px;
    cursor: pointer;
    color: var(--accent);
    font-size: 18px;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
  }
  .rk-modal-body { padding: 16px 24px 24px; }

  /* Hero row */
  .rk-hero-row {
    display: grid;
    grid-template-columns: 1.2fr 1fr 1fr;
    gap: 8px;
    margin-bottom: 10px;
  }
  .rk-hero-stat {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
  }
  .rk-hero-primary {
    background: var(--accent-soft);
    border-color: transparent;
  }
  .rk-hero-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .rk-hero-val {
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
  }
  .rk-hero-sub {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Draft + measurables */
  .rk-info-row {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 14px;
    margin-bottom: 10px;
  }
  .rk-meas-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 4px;
  }
  .rk-meas-cell {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 8px;
    text-align: center;
  }
  .rk-meas-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 4px;
  }
  .rk-meas-val {
    font-size: 14px;
    font-weight: 700;
    color: var(--text);
  }

  /* Section divider */
  .rk-section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 14px 0;
  }

  /* Component breakdown with bars */
  .rk-comp-list {
    display: flex;
    flex-direction: column;
    gap: 9px;
  }
  .rk-comp-row {
    display: grid;
    grid-template-columns: 90px 1fr 32px;
    align-items: center;
    gap: 10px;
  }
  .rk-comp-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
  }
  .rk-comp-bar-wrap {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .rk-comp-bar {
    height: 100%;
    border-radius: 3px;
  }
  .rk-comp-val {
    font-size: 12px;
    font-weight: 700;
    text-align: right;
  }

  /* Modal mobile */
  @media (max-width: 480px) {
    .rk-hero-row { grid-template-columns: 1fr 1fr; }
    .rk-hero-primary { grid-column: 1 / -1; }
    .rk-meas-grid { grid-template-columns: repeat(2, 1fr); }
    .rk-comp-row { grid-template-columns: 76px 1fr 28px; gap: 8px; }
  }

  /* Mobile */
  @media (max-width: 768px) {
    /* Show: rank | name | pos | draft | value — hide age and score */
    .rk-grid-row { grid-template-columns: 34px 1fr 46px 56px 54px !important; }
    .rk-score, #rkHeader span:nth-child(6) { display: none; }
    .rk-age,   #rkHeader span:nth-child(4) { display: none; }
    .rk-row { padding: 8px 10px; }
  }
  @media (max-width: 480px) {
    /* Show: rank | name | pos | value — also hide draft */
    .rk-grid-row { grid-template-columns: 30px 1fr 44px 52px !important; }
    .rk-draft, #rkHeader span:nth-child(5) { display: none; }
    .rk-draft { font-size: 10px; }
  }
</style>

<script>
  var rkAllPlayers = [];
  var rkLeague    = '1qb';
  var rkSize      = 10;
  var rkPosFilters = new Set();    // empty = all
  var rkSearch    = '';
  var rkLoaded    = false;
  var rkDraftYear = null;

  // Size-specific value keys
  function rkGetValue(r) {
    var key;
    if (rkLeague === 'sf') {
      key = rkSize === 10 ? 'rookie_sf_value' : 'rookie_sf_value_' + rkSize;
      return parseFloat(r[key] || r['rookie_sf_value'] || 0);
    } else {
      key = rkSize === 10 ? 'rookie_value' : 'rookie_value_' + rkSize;
      return parseFloat(r[key] || r['rookie_value'] || 0);
    }
  }

  // Settings panel toggle
  function rkToggleSettings() {
    var panel = document.getElementById('rkSettingsPanel');
    var btn = document.getElementById('rkSettingsBtn');
    if (!panel || !btn) return;

    var isOpen = panel.style.display === 'block';
    panel.style.display = isOpen ? 'none' : 'block';
    btn.classList.toggle('active', !isOpen);
  }

  // Update active settings indicator tags
  function updateRookieSettingsIndicator() {
    var indicator = document.getElementById('rkActiveSettings');
    if (!indicator) return;

    var sizeTag = indicator.querySelector('.active-setting-tag:first-child');
    var formatTag = indicator.querySelector('.active-setting-tag:last-child');

    if (sizeTag) sizeTag.textContent = rkSize + '-Team';
    if (formatTag) formatTag.textContent = rkLeague.toUpperCase();
  }

  function rkSetLeague(type) {
    rkLeague = type;

    // Update settings panel toggles
    document.querySelectorAll('#rkSettingsPanel .settings-toggle[data-value]').forEach(function(btn) {
      var section = btn.closest('.settings-section');
      if (section && section.querySelector('.settings-section-label').textContent.includes('Format')) {
        btn.classList.toggle('active', btn.getAttribute('data-value') === type);
      }
    });

    updateRookieSettingsIndicator();
    rkRender();
  }

  function rkSetSize(sz) {
    rkSize = sz;

    // Update settings panel toggles
    document.querySelectorAll('#rkSettingsPanel .settings-toggle[data-value]').forEach(function(btn) {
      var section = btn.closest('.settings-section');
      if (section && section.querySelector('.settings-section-label').textContent.includes('Size')) {
        var btnSize = parseInt(btn.getAttribute('data-value'));
        btn.classList.toggle('active', btnSize === sz);
      }
    });

    updateRookieSettingsIndicator();
    rkRender();
  }

  function rkTogglePos(pos) {
    if (pos === 'ALL') {
      rkPosFilters.clear();
    } else {
      if (rkPosFilters.has(pos)) rkPosFilters.delete(pos);
      else rkPosFilters.add(pos);
    }
    document.querySelectorAll('.pos-pill').forEach(function(b) {
      var p = b.getAttribute('data-pos');
      b.classList.toggle('active', p === 'ALL' ? rkPosFilters.size === 0 : rkPosFilters.has(p));
    });
    rkRender();
  }

  function rkClearSearch() {
    document.getElementById('rookieSearch').value = '';
    rkSearch = '';
    document.getElementById('rookieSearchClear').style.display = 'none';
    rkRender();
  }

  function rkFuzzy(name, q) {
    if (!name || !q) return 0;
    var n = name.toLowerCase(), qL = q.toLowerCase();
    if (n.includes(qL)) return 100 + (100 - n.indexOf(qL));
    var nw = n.split(/[\s\-]+/);
    if (nw.some(function(w){ return w.startsWith(qL); })) return 60;
    return 0;
  }

  function rkScoreColor(score) {
    if (score >= 80) return '#10b981';
    if (score >= 65) return '#3b82f6';
    if (score >= 50) return '#f59e0b';
    return '#6b7280';
  }

  function rkTierBadge(tier, label) {
    return '<span class="rk-tier rk-tier-' + (tier||6) + '">' +
      'T' + (tier||'?') + '</span>';
  }

  function rkRender() {
    if (!rkLoaded) return;
    var sortBy = document.getElementById('rkSort').value;

    var players = rkAllPlayers.slice();

    // Position filter
    if (rkPosFilters.size > 0) {
      players = players.filter(function(r) {
        return rkPosFilters.has((r.position||'').toUpperCase());
      });
    }

    // Search
    if (rkSearch.length > 0) {
      var q = rkSearch;
      players = players
        .map(function(r) { return {r:r, s: rkFuzzy(r.name, q)}; })
        .filter(function(x) { return x.s > 0; })
        .sort(function(a,b) { return b.s - a.s || rkGetValue(b.r) - rkGetValue(a.r); })
        .map(function(x) { return x.r; });
    } else {
      players.sort(function(a, b) {
        if (sortBy === 'value')  return rkGetValue(b) - rkGetValue(a);
        if (sortBy === 'score')  return (b.prospect_score||0) - (a.prospect_score||0);
        if (sortBy === 'age')    return (a.age||99) - (b.age||99);
        if (sortBy === 'pick')   return (a.projected_pick||999) - (b.projected_pick||999);
        // Default rank: in SF mode use SF value so QBs are ranked correctly
        if (rkLeague === 'sf')   return rkGetValue(b) - rkGetValue(a);
        return (a.overall_rank||999) - (b.overall_rank||999);
      });
    }

    var list   = document.getElementById('rkList');
    var empty  = document.getElementById('rkEmpty');
    var count  = document.getElementById('rkCount');
    var header = document.getElementById('rkHeader');

    if (players.length === 0) {
      list.innerHTML = '';
      empty.style.display = 'block';
      header.style.display = 'none';
      count.style.display  = 'none';
      return;
    }

    empty.style.display  = 'none';
    header.style.display = 'grid';
    count.style.display  = 'block';
    count.textContent    = players.length + ' prospect' + (players.length !== 1 ? 's' : '');

    list.innerHTML = '';
    players.forEach(function(r, idx) {
      var row = document.createElement('div');
      row.className = 'rk-row rk-grid-row';
      row.setAttribute('data-pid', r.player_id||'');

      var val   = rkGetValue(r);
      var score = parseFloat(r.prospect_score||0);
      var age   = r.age != null ? parseFloat(r.age).toFixed(1) : '—';
      var draft = r.draft_capital_label || (r.projected_pick ? '#'+r.projected_pick : '?');
      var posRk = r.position || '';
      if (r.position_rank) posRk += r.position_rank;

      var tierHtml = rkTierBadge(r.tier, r.tier_label);
      var earlyTag = r.early_declare ? '<span style="font-size:10px;color:var(--text-muted);margin-left:4px;">Early</span>' : '';
      var scoreColor = rkScoreColor(score);

      row.innerHTML =
        '<span class="rk-rank">' + (r.overall_rank ? '#' + r.overall_rank : idx+1) + '</span>' +
        '<div class="rk-name-cell">' +
          '<div class="rk-name">' + (r.name||'Unknown') + tierHtml + earlyTag + '</div>' +
          '<div class="rk-meta">' + (r.school||'') + (r.school && r.position ? ' • ' : '') + (posRk||'') + '</div>' +
        '</div>' +
        '<span class="rk-pos">' + (r.position||'') + '</span>' +
        '<span class="rk-age">' + age + '</span>' +
        '<span class="rk-draft">' + draft + '</span>' +
        '<span class="rk-score"><span class="rk-score-bar">' +
          '<span class="rk-score-dot" style="background:' + scoreColor + ';"></span>' +
          score.toFixed(0) + '</span></span>' +
        '<span class="rk-value">' + (val > 0 ? val.toFixed(1) : '—') + '</span>';

      row.addEventListener('click', function() { rkOpenModal(r); });
      list.appendChild(row);
    });
  }

  // ── Modal ──────────────────────────────────────────────────────────────────
  function rkOpenModal(r) {
    var modal   = document.getElementById('rkModal');
    var content = document.getElementById('rkModalContent');

    var val1qb = parseFloat(r.rookie_value||0);
    var valsf  = parseFloat(r.rookie_sf_value||0);
    var score  = parseFloat(r.prospect_score||0);
    var conf   = parseFloat(r.confidence_score||0);
    var age    = r.age != null ? parseFloat(r.age).toFixed(1) : '—';
    var tier   = r.tier || '?';
    var tierColors = ['','#10b981','#3b82f6','#8b5cf6','#f59e0b','#6b7280','#9ca3af'];
    var tierColor  = tierColors[tier] || '#9ca3af';

    var reasons = (r.key_reasons||'').split('\\n').filter(function(l){ return l.trim(); });

    // Measurables
    var ht = r.height_inches;
    var heightStr = ht ? (Math.floor(ht/12) + "'" + (ht%12) + '"') : '—';
    var weightStr = r.weight_lbs ? r.weight_lbs + ' lbs' : '—';
    var fortyStr  = r.forty_yard ? r.forty_yard + 's' : '—';
    var rasStr    = r.ras_score  ? parseFloat(r.ras_score).toFixed(1) + '/10' : '—';

    // Draft info — single consolidated line
    var draftCapLabel = r.draft_capital_label || (r.projected_pick ? 'Pick #' + r.projected_pick : null);
    var draftStr = draftCapLabel
      ? draftCapLabel + (r.num_mocks_used ? '  ·  ' + r.num_mocks_used + ' mocks' : '')
      : 'Undrafted / Unknown';

    // Component scores (Confidence lives in the section header, not here)
    var components = [
      {label:'Production',  val: r.production_score,              color:'#10b981'},
      {label:'Efficiency',  val: r.efficiency_score,              color:'#3b82f6'},
      {label:'Age',         val: r.age_score,                     color:'#8b5cf6'},
      {label:'Breakout',    val: r.breakout_profile_score,        color:'#f59e0b'},
      {label:'Athleticism', val: r.athleticism_score,             color:'#ef4444'},
      {label:'Competition', val: r.competition_score,             color:'#06b6d4'},
      {label:'Draft Cap.',  val: r.projected_draft_capital_score, color:'#f97316'},
    ];

    var compsHtml = components.map(function(c) {
      var v = parseFloat(c.val||0);
      return '<div class="rk-comp-row">' +
        '<div class="rk-comp-label">' + c.label + '</div>' +
        '<div class="rk-comp-bar-wrap"><div class="rk-comp-bar" style="width:' + Math.round(v) + '%;background:' + c.color + ';"></div></div>' +
        '<div class="rk-comp-val" style="color:' + c.color + ';">' + v.toFixed(0) + '</div>' +
      '</div>';
    }).join('');

    var reasonsHtml = reasons.length > 0
      ? '<div class="rk-section-divider"></div>' +
        '<div style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:10px;">Scouting Notes</div>' +
        '<div style="font-size:13px;color:var(--text-muted);line-height:1.7;">' +
          reasons.map(function(l){ return '<div style="padding:2px 0;">' + l + '</div>'; }).join('') +
        '</div>'
      : '';

    content.innerHTML =
      // ── Header: name + tier badge + close ───────────────────────────────
      '<div class="rk-modal-header">' +
        '<div>' +
          '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">' +
            '<span style="font-size:22px;font-weight:700;color:var(--text);">' + (r.name||'') + '</span>' +
            '<span style="padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;' +
                  'background:' + tierColor + '22;color:' + tierColor + ';border:1px solid ' + tierColor + '44;">' +
              'Tier ' + tier +
            '</span>' +
          '</div>' +
          '<div style="font-size:13px;color:var(--text-muted);margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;">' +
            '<span style="font-weight:600;color:var(--text);">' + (r.position||'') + (r.position_rank ? ' #'+r.position_rank : '') + '</span>' +
            (r.school ? '<span style="opacity:.4;">·</span><span>' + r.school + '</span>' : '') +
            '<span style="opacity:.4;">·</span><span>' + age + ' yrs</span>' +
            (r.draft_class_year ? '<span style="opacity:.4;">·</span><span>' + r.draft_class_year + ' Draft</span>' : '') +
          '</div>' +
        '</div>' +
        '<button class="rk-modal-close" onclick="rkCloseModal()">✕</button>' +
      '</div>' +

      '<div class="rk-modal-body">' +

        // ── Hero: Prospect Score + 1QB Value + SF Value ──────────────────
        '<div class="rk-hero-row">' +
          '<div class="rk-hero-stat rk-hero-primary">' +
            '<div class="rk-hero-label">Prospect Score</div>' +
            '<div class="rk-hero-val" style="color:var(--accent);">' + score.toFixed(1) + '</div>' +
            '<div class="rk-hero-sub">' + (r.tier_label||'') + '</div>' +
          '</div>' +
          '<div class="rk-hero-stat">' +
            '<div class="rk-hero-label">1QB Value</div>' +
            '<div class="rk-hero-val">' + (val1qb > 0 ? val1qb.toFixed(1) : '—') + '</div>' +
            '<div class="rk-hero-sub">10-team</div>' +
          '</div>' +
          '<div class="rk-hero-stat">' +
            '<div class="rk-hero-label">SF Value</div>' +
            '<div class="rk-hero-val">' + (valsf > 0 ? valsf.toFixed(1) : '—') + '</div>' +
            '<div class="rk-hero-sub">10-team</div>' +
          '</div>' +
        '</div>' +

        // ── Draft (consolidated) ─────────────────────────────────────────
        '<div class="rk-info-row">' +
          '<span style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;white-space:nowrap;">Draft</span>' +
          '<span style="font-size:13px;font-weight:600;color:var(--text);">' + draftStr + '</span>' +
        '</div>' +

        // ── Measurables ──────────────────────────────────────────────────
        '<div class="rk-meas-grid">' +
          '<div class="rk-meas-cell"><div class="rk-meas-label">Height</div><div class="rk-meas-val">' + heightStr + '</div></div>' +
          '<div class="rk-meas-cell"><div class="rk-meas-label">Weight</div><div class="rk-meas-val">' + weightStr + '</div></div>' +
          '<div class="rk-meas-cell"><div class="rk-meas-label">40 Dash</div><div class="rk-meas-val">' + fortyStr + '</div></div>' +
          '<div class="rk-meas-cell"><div class="rk-meas-label">RAS</div><div class="rk-meas-val">' + rasStr + '</div></div>' +
        '</div>' +

        // ── Component scores with bars ───────────────────────────────────
        '<div class="rk-section-divider"></div>' +
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">' +
          '<span style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;">Component Scores</span>' +
          '<span style="font-size:11px;color:var(--text-muted);">Data confidence: <strong style="color:var(--text);">' + conf.toFixed(0) + '</strong></span>' +
        '</div>' +
        '<div class="rk-comp-list">' + compsHtml + '</div>' +

        reasonsHtml +

        // ── Historical Comparables ────────────────────────────────────────
        '<div class="rk-section-divider"></div>' +
        '<div style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:10px;">Historical Comparables</div>' +
        '<div id="rkComparablesBody" style="font-size:13px;color:var(--text-muted);">' +
          '<div style="display:flex;align-items:center;gap:8px;"><div class="loading-spinner" style="width:12px;height:12px;flex-shrink:0;"></div>Loading…</div>' +
        '</div>' +

      '</div>';

    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    // Auto-link to Sleeper ID silently in background
    if (r.player_id) {
      fetch('/api/prospects/auto-link/' + encodeURIComponent(r.player_id)).catch(function(){});
    }

    // Fetch historical comparables
    fetch('/api/prospects/comparables/' + encodeURIComponent(r.player_id))
      .then(function(res){ return res.json(); })
      .then(function(cd) {
        var cb = document.getElementById('rkComparablesBody');
        if (!cb) return;
        var comps = cd.comparables || [];
        if (!comps.length) {
          cb.innerHTML = '<span>No close historical comps found.</span>';
          return;
        }
        var tc_ = ['','#10b981','#3b82f6','#8b5cf6','#f59e0b','#6b7280','#9ca3af'];
        cb.innerHTML = comps.map(function(c) {
          var tc = tc_[c.tier] || '#9ca3af';
          var pickStr = c.actual_pick ? ' · Pick ' + c.actual_pick : '';
          return '<div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--border);">' +
            '<div>' +
              '<span style="font-weight:600;color:var(--text);font-size:13px;">' + c.name + '</span>' +
              '<span style="color:var(--text-muted);font-size:12px;margin-left:6px;">' + c.draft_class_year + pickStr + '</span>' +
              (c.school ? '<span style="color:var(--text-muted);font-size:12px;"> · ' + c.school + '</span>' : '') +
            '</div>' +
            '<div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">' +
              '<span style="font-size:12px;color:var(--text-muted);">' + parseFloat(c.prospect_score).toFixed(1) + '</span>' +
              '<span style="padding:2px 7px;border-radius:5px;font-size:10px;font-weight:700;background:' + tc + '22;color:' + tc + ';border:1px solid ' + tc + '44;">T' + c.tier + '</span>' +
            '</div>' +
          '</div>';
        }).join('');
      })
      .catch(function() {
        var cb = document.getElementById('rkComparablesBody');
        if (cb) cb.innerHTML = '<span>Could not load comparables.</span>';
      });
  }

  function rkCloseModal() {
    document.getElementById('rkModal').style.display = 'none';
    document.body.style.overflow = '';
  }

  // Close on backdrop click
  document.getElementById('rkModal').addEventListener('click', function(e) {
    if (e.target === this) rkCloseModal();
  });

  // ── Load data ──────────────────────────────────────────────────────────────
  (function() {
    var inp = document.getElementById('rookieSearch');
    var clr = document.getElementById('rookieSearchClear');
    inp.addEventListener('input', function() {
      rkSearch = inp.value.trim();
      clr.style.display = rkSearch.length > 0 ? 'block' : 'none';
      rkRender();
    });
  })();

  // Close settings panel when clicking outside
  document.addEventListener('click', function(e) {
    var panel = document.getElementById('rkSettingsPanel');
    var btn = document.getElementById('rkSettingsBtn');

    if (panel && btn && panel.style.display === 'block') {
      if (!panel.contains(e.target) && !btn.contains(e.target)) {
        panel.style.display = 'none';
        btn.classList.remove('active');
      }
    }
  });

  fetch('/api/prospects/active-class')
    .then(function(r){ return r.json(); })
    .then(function(d) {
      rkDraftYear = d.draft_class_year || new Date().getFullYear();
      document.getElementById('rookiesTitle').textContent   = rkDraftYear + ' Prospect Rankings';

      return fetch('/api/prospects/rankings?year=' + rkDraftYear);
    })
    .then(function(r){ return r.json(); })
    .then(function(data) {
      document.getElementById('rkLoading').style.display = 'none';

      // Store size-specific value fields returned from API alongside base fields
      rkAllPlayers = (data.rankings || []);
      rkLoaded = true;
      rkRender();
    })
    .catch(function(err) {
      console.error('[rookies] Load error:', err);
      document.getElementById('rkLoading').innerHTML =
        '<div style="color:#ef4444;">Failed to load rookie data. Please refresh.</div>';
    });

  function _buildLinkSleeperHtml(playerId, existingSleeperIdVal) {
    if (existingSleeperIdVal) {
      return '<div style="font-size:12px;color:var(--text-muted);">Sleeper ID: <strong style="color:var(--text);">' + existingSleeperIdVal + '</strong> <span style="color:#10b981;">✓ linked</span></div>';
    }
    return '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">' +
      '<span style="font-size:12px;color:var(--text-muted);">Sleeper ID:</span>' +
      '<input id="rkSleeperIdInput" type="text" placeholder="e.g. 10229" style="font-size:12px;padding:4px 8px;border:1px solid var(--border);border-radius:6px;background:var(--card-bg);color:var(--text);width:110px;" />' +
      '<button onclick="rkLinkSleeper(\\'' + playerId + '\\')" style="font-size:12px;padding:4px 10px;border-radius:6px;background:var(--accent);color:#fff;border:none;cursor:pointer;font-weight:600;">Link &amp; Promote</button>' +
    '</div>';
  }

  function rkLinkSleeper(playerId) {
    var inp = document.getElementById('rkSleeperIdInput');
    if (!inp) return;
    var sleeperIdVal = inp.value.trim();
    if (!sleeperIdVal) { inp.focus(); return; }
    var btn = inp.nextElementSibling;
    if (!btn) return;
    
    btn.disabled = true;
    btn.textContent = 'Linking...';
    
    fetch('/api/prospects/link-sleeper', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({player_id: playerId, sleeper_id: sleeperIdVal})
    })
    .then(function(r){ return r.json(); })
    .then(function(resp){
      if (resp.success) {
        var sec = document.getElementById('rkLinkSleeperSection');
        if (sec) sec.innerHTML = _buildLinkSleeperHtml(playerId, sleeperIdVal);
        // Update the player in the global list for future renders
        var p = rkAllPlayers.find(function(pl){ return pl.player_id === playerId; });
        if (p) p.sleeper_id = sleeperIdVal;
        rkRender(); // re-render to update the row
      } else {
        alert('Failed to link: ' + (resp.error || 'Unknown error'));
        btn.disabled = false;
        btn.textContent = 'Link & Promote';
      }
    })
    .catch(function(err){
      console.error('Link error:', err);
      alert('Error linking Sleeper ID');
      btn.disabled = false;
      btn.textContent = 'Link & Promote';
    });
  }
</script>
"""
