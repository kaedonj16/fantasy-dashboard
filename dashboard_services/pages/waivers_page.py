"""
Waiver Wire / Start-Sit Advisor page builder.

Renders the HTML shell for /<platform>/<season>/<league_id>/waivers.
Actual data is loaded client-side via /api/waiver-candidates and
/api/start-sit-options.
"""


def build_waivers_body(platform: str, season: int, league_id: str, ctx: dict) -> str:
    """Return the full HTML body for the Waivers page."""

    style = """
<style>
.wv-page { padding: 16px; max-width: 1100px; margin: 0 auto; }
.wv-filters { margin-bottom: 16px; }
.wv-filter-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.wv-pos-pills { display: flex; gap: 6px; flex-wrap: wrap; }
.wv-pos-btn {
  padding: 5px 14px; border-radius: 16px; font-size: 12px; font-weight: 700;
  border: 1px solid var(--border); background: var(--card); color: var(--text-muted); cursor: pointer;
  transition: background .12s, color .12s, border-color .12s;
}
.wv-pos-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }

/* Mobile tab bar */
.wv-tab-bar {
  display: none; border-radius: 10px; overflow: hidden;
  border: 1px solid var(--border); margin-bottom: 16px; background: var(--card);
}
.wv-tab-btn {
  flex: 1; padding: 10px 0; font-size: 13px; font-weight: 700;
  border: none; background: none; color: var(--text-muted); cursor: pointer;
  transition: background .15s, color .15s;
}
.wv-tab-btn.active { background: var(--accent); color: #fff; }
.wv-tab-btn:first-child { border-right: 1px solid var(--border); }

@media(max-width: 768px) {
  .wv-tab-bar { display: flex; }
  .wv-layout { grid-template-columns: 1fr !important; }
  .wv-section { display: none; }
  .wv-section.wv-tab-active { display: block; }
}

.wv-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media(min-width: 769px) { .wv-section { display: block !important; } }
.wv-section-title { font-size: 14px; font-weight: 700; margin-bottom: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; }
.wv-loading { display: flex; justify-content: center; padding: 40px; }

/* Waiver wire rows */
.wv-player-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 12px; border-radius: 8px; background: var(--card);
  border: 1px solid var(--border); margin-bottom: 8px; cursor: pointer;
}
.wv-player-row:hover { border-color: var(--accent); }
.wv-player-name { font-weight: 600; font-size: 14px; color: var(--text); }
.wv-player-sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.wv-right { display: flex; align-items: center; gap: 10px; }
.wv-value { font-size: 13px; font-weight: 700; color: var(--text); }
.wv-signal { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 8px; }
.signal-breakout { background: #10b98120; color: #10b981; }
.signal-rising   { background: #3b82f620; color: #3b82f6; }
.signal-value    { background: #8b5cf620; color: #8b5cf6; }
.signal-aging    { background: #f59e0b20; color: #f59e0b; }
.signal-hold     { background: var(--row); color: var(--text-muted); }
.signal-usage    { background: #ef444420; color: #ef4444; }
.signal-injury      { background: #f43f5e20; color: #f43f5e; }
.signal-injury-soft { background: #fb923c20; color: #fb923c; }
.wv-usage-chip {
  display: inline-block; font-size: 10px; font-weight: 700; color: #10b981;
  margin-left: 6px; white-space: nowrap;
}

/* Start/Sit player cards */
.wv-ss-pos-group { margin-bottom: 16px; }
.wv-ss-pos-label { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
.wv-ss-player {
  padding: 10px 12px; border-radius: 8px; background: var(--card);
  border: 1px solid var(--border); margin-bottom: 6px;
}
.wv-ss-player.wv-ss-start  { border-color: #10b981; background: #10b98108; }
.wv-ss-player.wv-ss-bye    { opacity: .55; }
.wv-ss-player.wv-ss-selected { border-color: var(--accent); background: var(--accent-soft); box-shadow: 0 0 0 2px var(--accent-soft); }

/* Top row: name + badge */
.wv-ss-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.wv-ss-name-block { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 0; cursor: pointer; }
.wv-ss-actions { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }

/* Stats row under the name */
.wv-ss-stats { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px; align-items: stretch; }
.wv-ss-div { width: 1px; align-self: stretch; background: var(--border); margin: 1px 0; flex: 0 0 auto; }
.wv-ss-stat { display: flex; flex-direction: column; gap: 1px; }
.wv-ss-stat-lbl { font-size: 10px; color: var(--text-subtle); text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }
.wv-ss-stat-val { font-size: 13px; font-weight: 700; color: var(--text); }
.wv-ss-stat-val.muted { color: var(--text-muted); }
.wv-ss-env, .wv-ss-total, .wv-ss-cons { font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 6px; align-self: flex-start; }
.wv-ss-cons-steady   { background: rgba(34,197,94,.16); color: #15803d; }
.wv-ss-cons-balanced { background: rgba(148,163,184,.16); color: var(--text-muted); }
.wv-ss-cons-volatile { background: rgba(234,179,8,.16); color: #a16207; }
.wv-ss-cons-boombust { background: rgba(168,85,247,.16); color: #7e22ce; }
[data-theme="dark"] .wv-ss-cons-steady   { background: rgba(34,197,94,.20); color: #4ade80; }
[data-theme="dark"] .wv-ss-cons-volatile { background: rgba(234,179,8,.20); color: #fbbf24; }
[data-theme="dark"] .wv-ss-cons-boombust { background: rgba(168,85,247,.24); color: #c084fc; }
.wv-ss-env-dome { background: rgba(59,130,246,.14); color: #1d4ed8; }
.wv-ss-env-cold { background: rgba(56,189,248,.16); color: #0369a1; }
.wv-ss-env-wind { background: rgba(148,163,184,.20); color: #475569; }
.wv-ss-env-precip { background: rgba(59,130,246,.14); color: #1d4ed8; }
[data-theme="dark"] .wv-ss-env-dome { background: rgba(59,130,246,.20); color: #60a5fa; }
[data-theme="dark"] .wv-ss-env-cold { background: rgba(56,189,248,.20); color: #7dd3fc; }
[data-theme="dark"] .wv-ss-env-wind { background: rgba(148,163,184,.24); color: #cbd5e1; }
[data-theme="dark"] .wv-ss-env-precip { background: rgba(59,130,246,.22); color: #93c5fd; }
/* Vegas implied team total: green when high, muted when low */
.wv-ss-total-high { background: rgba(34,197,94,.16); color: #15803d; }
.wv-ss-total-mid  { background: rgba(148,163,184,.16); color: var(--text-muted); }
.wv-ss-total-low  { background: rgba(239,68,68,.14); color: #b91c1c; }
[data-theme="dark"] .wv-ss-total-high { background: rgba(34,197,94,.20); color: #4ade80; }
[data-theme="dark"] .wv-ss-total-low  { background: rgba(239,68,68,.20); color: #f87171; }

/* Matchup chip */
.wv-mu { font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 6px; }
.wv-mu-easy { background: #22c55e20; color: #22c55e; }
.wv-mu-ok   { background: #84cc1620; color: #84cc16; }
.wv-mu-avg  { background: #f59e0b20; color: #f59e0b; }
.wv-mu-hard { background: #ef444420; color: #ef4444; }

/* Badges */
.wv-ss-start-badge      { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: #10b98120; color: #10b981; flex-shrink: 0; }
.wv-ss-flex-start-badge { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: #10b98120; color: #10b981; border: 1px solid #10b98140; flex-shrink: 0; }
.wv-ss-flex-badge       { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: #3b82f620; color: #3b82f6; flex-shrink: 0; }
.wv-ss-sit-badge        { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: var(--row); color: var(--text-muted); flex-shrink: 0; }
.wv-ss-bye-badge        { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: #f59e0b20; color: #f59e0b; flex-shrink: 0; }
.wv-inj-out { font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 4px; background: #ef444420; color: #ef4444; }
.wv-inj-q   { font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 4px; background: #f59e0b20; color: #f59e0b; }
.wv-inj-d   { font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 4px; background: #f9731620; color: #f97316; }

/* Compare button */
.wv-cmp-btn {
  font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 6px;
  border: 1px solid var(--border); background: transparent; color: var(--text-muted);
  cursor: pointer; transition: background .12s, color .12s, border-color .12s; flex-shrink: 0;
}
.wv-cmp-btn:hover { border-color: var(--accent); color: var(--accent); }
.wv-cmp-btn.selected { background: var(--accent); color: #fff; border-color: var(--accent); }

/* Compare panel */
.wv-compare-panel {
  background: var(--card); border: 1px solid var(--accent); border-radius: 10px;
  margin-bottom: 16px; overflow: hidden;
}
.wv-compare-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
  color: var(--text-muted);
}
.wv-compare-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0;
}
.wv-compare-col { padding: 14px; }
.wv-compare-col:first-child { border-right: 1px solid var(--border); }
.wv-compare-player-name { font-size: 15px; font-weight: 700; margin-bottom: 2px; }
.wv-compare-player-sub  { font-size: 12px; color: var(--text-muted); margin-bottom: 12px; }
.wv-compare-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 13px;
}
.wv-compare-row:last-child { border-bottom: none; }
.wv-compare-lbl { color: var(--text-muted); font-size: 12px; }
.wv-compare-val { font-weight: 700; }
.wv-compare-win { color: #22c55e; }
.wv-compare-lose { color: #ef4444; }
</style>
"""

    html_body = f"""
<div class="wv-page">
  <!-- Position filter pills -->
  <div class="wv-filters">
    <div class="wv-filter-row">
      <div class="wv-pos-pills">
        <button class="wv-pos-btn active" onclick="wvSetPos('ALL')">ALL</button>
        <button class="wv-pos-btn" onclick="wvSetPos('QB')">QB</button>
        <button class="wv-pos-btn" onclick="wvSetPos('RB')">RB</button>
        <button class="wv-pos-btn" onclick="wvSetPos('WR')">WR</button>
        <button class="wv-pos-btn" onclick="wvSetPos('TE')">TE</button>
      </div>
    </div>
  </div>

  <!-- Mobile tab bar -->
  <div class="wv-tab-bar">
    <button class="wv-tab-btn active" id="wvTabWaivers" onclick="wvSetTab('waivers')">Waiver Wire</button>
    <button class="wv-tab-btn" id="wvTabStartSit" onclick="wvSetTab('startsit')">Start/Sit</button>
  </div>

  <!-- Two-column layout -->
  <div class="wv-layout">
    <!-- Left: Waiver Wire -->
    <div class="wv-section wv-tab-active" id="wvSectionWaivers">
      <div class="wv-section-title">Waiver Wire</div>
      <div id="wvWaiverList">
        <div class="wv-loading"><div class="loading-spinner"></div></div>
      </div>
    </div>

    <!-- Right: Start/Sit -->
    <div class="wv-section" id="wvSectionStartSit">
      <div class="wv-section-title">Start/Sit Advisor</div>
      <!-- Compare panel (hidden until 2 players selected) -->
      <div id="wvComparePanel" style="display:none;"></div>
      <div id="wvStartSit">
        <div class="wv-loading"><div class="loading-spinner"></div></div>
      </div>
    </div>
  </div>
</div>
"""

    script = f"""
<script>
const WV_PLATFORM = '{platform}';
const WV_SEASON = {season};
const WV_LEAGUE_ID = '{league_id}';
let wvCurrentPos = 'ALL';
let wvWaiverData = [];
let wvStartSitData = {{}};
let wvCompare = [null, null]; // [playerA, playerB]

if (!window.__brctx) window.__brctx = {{}};
if (!window.__brctx.leagueId) window.__brctx.leagueId = WV_LEAGUE_ID;

function wvSetTab(tab) {{
  const isWaivers = tab === 'waivers';
  document.getElementById('wvSectionWaivers').classList.toggle('wv-tab-active', isWaivers);
  document.getElementById('wvSectionStartSit').classList.toggle('wv-tab-active', !isWaivers);
  document.getElementById('wvTabWaivers').classList.toggle('active', isWaivers);
  document.getElementById('wvTabStartSit').classList.toggle('active', !isWaivers);
}}

function wvSetPos(pos) {{
  wvCurrentPos = pos;
  document.querySelectorAll('.wv-pos-btn').forEach(b => b.classList.toggle('active', b.textContent === pos));
  wvRenderWaivers();
  wvRenderStartSit();
}}

// ── Matchup chip helper ───────────────────────────────────────────────────────
function wvMuChip(rank, total) {{
  if (!rank || !total) return '';
  const pct = rank / total;
  if (pct <= 0.25) return `<span class="wv-mu wv-mu-easy">#${{rank}} easiest</span>`;
  if (pct <= 0.50) return `<span class="wv-mu wv-mu-ok">#${{rank}} favorable</span>`;
  if (pct <= 0.75) return `<span class="wv-mu wv-mu-avg">#${{rank}} tough</span>`;
  return `<span class="wv-mu wv-mu-hard">#${{rank}} hardest</span>`;
}}

function wvMuClass(rank, total) {{
  if (!rank || !total) return '';
  const pct = rank / total;
  if (pct <= 0.25) return 'wv-compare-win';
  if (pct <= 0.75) return '';
  return 'wv-compare-lose';
}}

// ── Injury badge ──────────────────────────────────────────────────────────────
function wvInjBadge(inj) {{
  if (!inj) return '';
  const u = inj.toUpperCase();
  if (['IR','OUT','PUP','SUSP'].includes(u)) return `<span class="wv-inj-out">${{u}}</span>`;
  if (['DOUBTFUL','D'].includes(u))          return `<span class="wv-inj-d">D</span>`;
  if (['QUESTIONABLE','Q','GTD'].includes(u)) return `<span class="wv-inj-q">Q</span>`;
  return `<span class="wv-inj-q">${{inj}}</span>`;
}}

// Venue / weather chip: live weather when available, else the static dome/cold
// venue tag. Rendered separately so it can sit after the matchup chip.
function wvVenueChip(p) {{
  const env = p.weather || p.game_env;
  if (!env) return '';
  const lbl = p.weather ? 'Weather' : 'Venue';
  const note = (env.note || '').replace(/"/g, '&quot;');
  return '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">' + lbl + '</span>' +
         '<span class="wv-ss-env wv-ss-env-' + env.kind + '" title="' + note + '">' +
         env.label + '</span></div>';
}}

// Vegas implied team total chip.
function wvVegasChip(p) {{
  if (p.implied_total == null) return '';
  const it = p.implied_total;
  const k = it >= 26 ? 'high' : (it <= 18 ? 'low' : 'mid');
  return '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Vegas</span>' +
         '<span class="wv-ss-total wv-ss-total-' + k + '" title="Implied team total (Vegas)">' +
         it + ' implied</span></div>';
}}

// Weekly consistency chips: floor-ceiling range + boom/bust profile label.
function wvConsistencyChips(p) {{
  const c = p.consistency;
  if (!c || c.small_sample) return '';
  const k = c.label === 'Steady' ? 'steady'
          : c.label === 'Volatile' ? 'volatile'
          : c.label === 'Boom or bust' ? 'boombust' : 'balanced';
  const yr = c.season ? " '" + String(c.season).slice(-2) : '';
  const tip = 'Consistency ' + c.consistency + '/100 · boom ' +
              Math.round(c.boom_rate * 100) + '% · bust ' +
              Math.round(c.bust_rate * 100) + '% (' + c.games + ' g' +
              (c.season ? ', ' + c.season : '') + ')';
  return '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Floor–Ceil' + yr + '</span>' +
         '<span class="wv-ss-stat-val muted">' + c.floor + '–' + c.ceiling + '</span></div>' +
         '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Profile</span>' +
         '<span class="wv-ss-cons wv-ss-cons-' + k + '" title="' + tip + '">' + c.label + yr + '</span></div>';
}}

// Compose the start/sit stats row in three groups, most-decisive first, with a
// thin divider between groups:
//   1. Output      - what to expect  (Proj PPG, L4 PPG)
//   2. Reliability - can I trust it   (Floor-Ceil, Profile, Usage)
//   3. The spot    - this week's game (Opp, Matchup, Vegas, Venue)
// 'Def vs pos' is folded into the Matchup chip's tooltip to keep the row lean.
function wvStatsRow(p) {{
  const chip = (lbl, val, cls) =>
    '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">' + lbl + '</span>' +
    '<span class="wv-ss-stat-val ' + (cls || '') + '">' + val + '</span></div>';

  const g1 = [];
  if (p.proj_pts > 0)   g1.push(chip('Proj PPG', p.proj_pts));
  if (p.recent_ppg > 0) g1.push(chip('L4 PPG', p.recent_ppg));

  const g2 = [wvConsistencyChips(p)];
  if (p.usage_delta != null && Math.abs(p.usage_delta) >= 1) {{
    const upU = p.usage_delta > 0;
    const lblU = p.usage_stat === 'snap_pct' ? 'snap%' : (p.usage_stat === 'touches' ? 'touches' : 'targets');
    g2.push('<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Usage</span><span class="wv-ss-stat-val" style="color:' +
            (upU ? '#10b981' : '#ef4444') + '" title="Last-3-week ' + lblU + ' vs season avg">' +
            (upU ? '&#9650;' : '&#9660;') + ' ' + (upU ? '+' : '') + p.usage_delta + '</span></div>');
  }}

  const g3 = [];
  if (p.opponent) g3.push(chip('Opp', p.opponent, 'muted'));
  const mu = !p.on_bye ? wvMuChip(p.def_rank, p.def_total) : '';
  if (mu) {{
    const dtip = p.fpts_against > 0
      ? ' · ' + p.fpts_against + ' pts/gm allowed to ' + (p.position || 'the position')
      : '';
    g3.push('<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Matchup</span>' +
            '<span class="wv-ss-stat-val" title="Matchup rank (1 = easiest)' + dtip + '">' + mu + '</span></div>');
  }}
  g3.push(wvVegasChip(p));
  g3.push(wvVenueChip(p));

  const groups = [g1, g2, g3].map(g => g.filter(Boolean).join('')).filter(Boolean);
  return '<div class="wv-ss-stats">' +
         groups.join('<span class="wv-ss-div" aria-hidden="true"></span>') +
         '</div>';
}}

// ── Load ──────────────────────────────────────────────────────────────────────
function wvLoad() {{
  fetch(`/api/waiver-candidates?platform=${{WV_PLATFORM}}&league_id=${{WV_LEAGUE_ID}}&season=${{WV_SEASON}}`)
    .then(r => r.json())
    .then(d => {{ wvWaiverData = d.candidates || []; wvRenderWaivers(); }})
    .catch(() => {{ document.getElementById('wvWaiverList').innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:20px;">Unable to load</div>'; }});

  fetch(`/api/start-sit-options?platform=${{WV_PLATFORM}}&league_id=${{WV_LEAGUE_ID}}&season=${{WV_SEASON}}`)
    .then(r => r.json())
    .then(d => {{
      if (!d.positions || !Object.keys(d.positions).length) {{
        showLoginGate('wvStartSit', {{
          title: 'Sign in to see your lineup',
          description: 'Enter your Sleeper username to get personalized start/sit recommendations for your roster.'
        }});
        return;
      }}
      wvStartSitData = d;
      wvStartSitData._lineup_requirements = d.lineup_requirements || {{}};
      wvRenderStartSit();
    }})
    .catch(() => {{
      document.getElementById('wvStartSit').innerHTML =
        '<div style="color:var(--text-muted);text-align:center;padding:20px;">Unable to load lineup data</div>';
    }});
}}

// ── Waiver list ───────────────────────────────────────────────────────────────
function wvRenderWaivers() {{
  const list = document.getElementById('wvWaiverList');
  let players = wvWaiverData;
  if (wvCurrentPos !== 'ALL') players = players.filter(p => p.position === wvCurrentPos);
  if (!players.length) {{ list.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:20px;">No players found</div>'; return; }}
  list.innerHTML = players.slice(0, 20).map(p => {{
    let usageChip = '';
    if (p.usage_delta != null && p.usage_delta >= 1) {{
      const statLbl = p.usage_stat === 'snap_pct' ? 'snap%' : (p.usage_stat === 'touches' ? 'touches' : 'targets');
      usageChip = `<span class="wv-usage-chip" title="Last-3-week avg vs season avg">&#9650; +${{p.usage_delta}} ${{statLbl}}</span>`;
    }}
    return `
    <div class="wv-player-row" onclick="openPlayerModal('${{p.player_id}}', '${{p.name.replace(/'/g,"\\'")}}')">
      <div>
        <div class="wv-player-name">${{p.name}}</div>
        <div class="wv-player-sub">${{[p.position, p.team, p.pos_rank_label, p.age ? 'Age ' + parseFloat(p.age).toFixed(1) : ''].filter(Boolean).join(' · ')}}${{usageChip}}</div>
      </div>
      <div class="wv-right">
        <span class="wv-signal ${{p.signal_class}}">${{p.signal}}</span>
        <span class="wv-value">${{Math.round(p.value)}}</span>
      </div>
    </div>
  `;
  }}).join('');
}}

// ── Compare slot management ───────────────────────────────────────────────────
function wvToggleCompare(p) {{
  const id = p.player_id;
  const inSlot0 = wvCompare[0] && wvCompare[0].player_id === id;
  const inSlot1 = wvCompare[1] && wvCompare[1].player_id === id;

  if (inSlot0) {{ wvCompare[0] = null; }}
  else if (inSlot1) {{ wvCompare[1] = null; }}
  else if (!wvCompare[0]) {{ wvCompare[0] = p; }}
  else if (!wvCompare[1]) {{ wvCompare[1] = p; }}
  else {{ wvCompare[0] = wvCompare[1]; wvCompare[1] = p; }}

  wvRenderCompare();
  wvRenderStartSit();
}}

function wvIsSelected(id) {{
  return (wvCompare[0] && wvCompare[0].player_id === id) ||
         (wvCompare[1] && wvCompare[1].player_id === id);
}}

function wvRenderCompare() {{
  const panel = document.getElementById('wvComparePanel');
  const a = wvCompare[0], b = wvCompare[1];
  if (!a || !b) {{ panel.style.display = 'none'; return; }}
  panel.style.display = 'block';

  function col(p, other) {{
    const muChip = wvMuChip(p.def_rank, p.def_total);
    const defLabel = p.fpts_against > 0
      ? `${{p.fpts_against}} pts/gm allowed`
      : (p.on_bye ? 'BYE' : '–');
    const projWin  = (other && p.proj_pts > 0 && other.proj_pts > 0)    ? (p.proj_pts   > other.proj_pts   ? 'wv-compare-win' : 'wv-compare-lose') : '';
    const l4ppg    = p.recent_ppg > 0 ? p.recent_ppg : p.season_ppg;
    const l4ppgOth = other && (other.recent_ppg > 0 ? other.recent_ppg : other.season_ppg);
    const ppgWin   = (l4ppg > 0 && l4ppgOth > 0) ? (l4ppg > l4ppgOth ? 'wv-compare-win' : 'wv-compare-lose') : '';
    const muCls   = wvMuClass(p.def_rank, p.def_total);
    return `
      <div class="wv-compare-col">
        <div class="wv-compare-player-name">${{p.name}}</div>
        <div class="wv-compare-player-sub">${{[p.team, p.pos_rank_label, p.opponent || (p.on_bye ? 'BYE' : '')].filter(Boolean).join(' · ')}}</div>
        <div class="wv-compare-row">
          <span class="wv-compare-lbl">Proj PPG</span>
          <span class="wv-compare-val ${{projWin}}">${{p.proj_pts > 0 ? p.proj_pts : '–'}}</span>
        </div>
        <div class="wv-compare-row">
          <span class="wv-compare-lbl">L4 PPG</span>
          <span class="wv-compare-val ${{ppgWin}}">${{p.recent_ppg > 0 ? p.recent_ppg : (p.season_ppg > 0 ? p.season_ppg : '–')}}</span>
        </div>
        <div class="wv-compare-row">
          <span class="wv-compare-lbl">Opponent</span>
          <span class="wv-compare-val">${{p.opponent || (p.on_bye ? 'BYE' : '–')}}</span>
        </div>
        <div class="wv-compare-row">
          <span class="wv-compare-lbl">Def vs pos</span>
          <span class="wv-compare-val ${{muCls}}">${{defLabel}}</span>
        </div>
        <div class="wv-compare-row">
          <span class="wv-compare-lbl">Matchup rank</span>
          <span class="wv-compare-val">${{muChip || (p.on_bye ? '–' : 'No data')}}</span>
        </div>
        <div class="wv-compare-row">
          <span class="wv-compare-lbl">Dynasty rank</span>
          <span class="wv-compare-val">${{p.pos_rank_label || '–'}}</span>
        </div>
      </div>`;
  }}

  panel.innerHTML = `
    <div class="wv-compare-panel">
      <div class="wv-compare-header">
        <span>Compare</span>
        <button onclick="wvClearCompare()" style="font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--text-muted);cursor:pointer;">Clear</button>
      </div>
      <div class="wv-compare-grid">
        ${{col(a, b)}}
        ${{col(b, a)}}
      </div>
    </div>`;
}}

function wvClearCompare() {{
  wvCompare = [null, null];
  wvRenderCompare();
  wvRenderStartSit();
}}

// ── Start/Sit list ────────────────────────────────────────────────────────────
function wvRenderStartSit() {{
  const el = document.getElementById('wvStartSit');
  const positions = wvCurrentPos === 'ALL' ? ['QB','RB','WR','TE'] : [wvCurrentPos];
  const reqs = wvStartSitData._lineup_requirements || {{}};

  const sections = positions.map(pos => {{
    const players = (wvStartSitData.positions || {{}})[pos] || [];
    if (!players.length) return '';

    const rows = players.slice(0, 8).map(p => {{
      const isStart     = p.start === true;
      const isFlexStart = p.flex_start === true;
      const isFlex      = p.flex_eligible === true && !isStart;
      const isBye       = p.on_bye === true;
      const isSelected  = wvIsSelected(p.player_id);

      const badge = isBye
        ? '<span class="wv-ss-bye-badge">BYE</span>'
        : isFlexStart
          ? '<span class="wv-ss-flex-start-badge">FLEX</span>'
          : isStart
            ? '<span class="wv-ss-start-badge">START</span>'
            : isFlex
              ? '<span class="wv-ss-flex-badge">FLEX?</span>'
              : '<span class="wv-ss-sit-badge">SIT</span>';

      const injBadge = wvInjBadge(p.injury_status);
      const cmpCls   = isSelected ? 'selected' : '';
      const statsRow = wvStatsRow(p);

      return `
        <div class="wv-ss-player ${{isStart ? 'wv-ss-start' : ''}} ${{isBye ? 'wv-ss-bye' : ''}} ${{isSelected ? 'wv-ss-selected' : ''}}">
          <div class="wv-ss-top">
            <div class="wv-ss-name-block" onclick="openPlayerModal('${{p.player_id}}', '${{(p.name||'').replace(/'/g,"\\'")}}')">
              <span class="wv-player-name">${{p.name}}</span>
              ${{injBadge}}
              ${{badge}}
            </div>
            <div class="wv-ss-actions">
              <button class="wv-cmp-btn ${{isSelected ? 'selected' : ''}}"
                onclick="event.stopPropagation();wvToggleCompare(${{JSON.stringify(p).replace(/"/g,'&quot;')}})">
                ${{isSelected ? '✓' : '+'}} Compare
              </button>
            </div>
          </div>
          ${{statsRow}}
        </div>`;
    }}).join('');

    const slotCount = reqs[pos] || 1;
    return `<div class="wv-ss-pos-group"><div class="wv-ss-pos-label">${{pos}} <span style="font-size:10px;font-weight:500;color:var(--text-muted);">(${{slotCount}} starter${{slotCount > 1 ? 's' : ''}})</span></div>${{rows}}</div>`;
  }}).join('');

  el.innerHTML = sections || '<div style="color:var(--text-muted);text-align:center;padding:20px;">No roster data found</div>';
}}

document.addEventListener('DOMContentLoaded', wvLoad);

// Deep link: ?tab=startsit opens the Start/Sit Advisor (switches the mobile
// tab and scrolls the section into view on desktop).
document.addEventListener('DOMContentLoaded', function() {{
  try {{
    const params = new URLSearchParams(window.location.search);
    if ((params.get('tab') || '').toLowerCase() === 'startsit') {{
      wvSetTab('startsit');
      const sec = document.getElementById('wvSectionStartSit');
      if (sec) sec.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}
  }} catch (e) {{}}
}});
</script>
"""

    return style + html_body + script
