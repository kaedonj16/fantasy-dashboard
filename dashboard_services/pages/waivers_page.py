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
/* Waiver signals: semantic ones route through tokens (theme-correct for free);
   value stays indigo and injury stays rose so they read as distinct categories.
   Both use color-mix backgrounds and a dark-mode brightening so they adapt too. */
.wv-signal { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 8px; }
.signal-breakout { background: color-mix(in srgb, var(--win) 15%, transparent); color: var(--win); }
.signal-rising   { background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent); }
.signal-value    { background: color-mix(in srgb, var(--indigo) 16%, transparent); color: var(--indigo); }
.signal-aging    { background: color-mix(in srgb, var(--warning) 16%, transparent); color: var(--warning); }
.signal-hold     { background: var(--row); color: var(--text-muted); }
.signal-usage    { background: color-mix(in srgb, var(--loss) 15%, transparent); color: var(--loss); }
.signal-injury      { background: color-mix(in srgb, #f43f5e 15%, transparent); color: #f43f5e; }
[data-theme="dark"] .signal-injury { color: #fb7185; }
.signal-injury-soft { background: color-mix(in srgb, var(--orange) 16%, transparent); color: var(--orange); }
.wv-usage-chip {
  display: inline-block; font-size: 10px; font-weight: 700; color: var(--win);
  margin-left: 6px; white-space: nowrap;
}

/* Start/Sit player cards */
.wv-ss-pos-group { margin-bottom: 16px; }
.wv-ss-pos-label { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
.wv-ss-player {
  padding: 10px 12px; border-radius: 8px; background: var(--card);
  border: 1px solid var(--border); margin-bottom: 6px;
}
.wv-ss-player.wv-ss-start  { border-color: var(--win); background: color-mix(in srgb, var(--win) 6%, transparent); }
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
.wv-ss-cons-steady   { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
.wv-ss-cons-balanced { background: rgba(148,163,184,.16); color: var(--text-muted); }
.wv-ss-cons-volatile { background: color-mix(in srgb, var(--warning) 18%, transparent); color: var(--warning); }
.wv-ss-cons-boombust { background: rgba(168,85,247,.16); color: #7e22ce; }
[data-theme="dark"] .wv-ss-cons-boombust { background: rgba(168,85,247,.24); color: #c084fc; }
/* Subtle "from last season" superscript on blended floor-ceiling */
.wv-yr { font-size: .68em; font-weight: 600; color: var(--text-muted); vertical-align: super; margin-left: 2px; opacity: .85; }
.wv-ss-env-dome { background: rgba(59,130,246,.14); color: #1d4ed8; }
.wv-ss-env-cold { background: rgba(56,189,248,.16); color: #0369a1; }
.wv-ss-env-wind { background: rgba(148,163,184,.20); color: #475569; }
.wv-ss-env-precip { background: rgba(59,130,246,.14); color: #1d4ed8; }
[data-theme="dark"] .wv-ss-env-dome { background: rgba(59,130,246,.20); color: #60a5fa; }
[data-theme="dark"] .wv-ss-env-cold { background: rgba(56,189,248,.20); color: #7dd3fc; }
[data-theme="dark"] .wv-ss-env-wind { background: rgba(148,163,184,.24); color: #cbd5e1; }
[data-theme="dark"] .wv-ss-env-precip { background: rgba(59,130,246,.22); color: #93c5fd; }
/* Vegas implied team total: green when high, muted when low */
.wv-ss-total-high { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
.wv-ss-total-mid  { background: rgba(148,163,184,.16); color: var(--text-muted); }
.wv-ss-total-low  { background: color-mix(in srgb, var(--loss) 15%, transparent); color: var(--loss); }

/* Matchup chip: a 4-step easy→hard scale, so the middle "ok" keeps its own
   lime step; the endpoints use the win/warning/loss tokens. */
.wv-mu { font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 6px; }
.wv-mu-easy { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
.wv-mu-ok   { background: color-mix(in srgb, #84cc16 13%, transparent); color: #65a30d; }
.wv-mu-avg  { background: color-mix(in srgb, var(--warning) 16%, transparent); color: var(--warning); }
.wv-mu-hard { background: color-mix(in srgb, var(--loss) 15%, transparent); color: var(--loss); }
[data-theme="dark"] .wv-mu-ok { color: #a3e635; }

/* Badges */
.wv-ss-start-badge      { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--win) 15%, transparent); color: var(--win); flex-shrink: 0; }
.wv-ss-flex-start-badge { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--win) 15%, transparent); color: var(--win); border: 1px solid color-mix(in srgb, var(--win) 30%, transparent); flex-shrink: 0; }
.wv-ss-flex-badge       { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent); flex-shrink: 0; }
.wv-ss-sit-badge        { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: var(--row); color: var(--text-muted); flex-shrink: 0; }
.wv-ss-bye-badge        { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--warning) 15%, transparent); color: var(--warning); flex-shrink: 0; }
.wv-inj-out { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); }
.wv-inj-q   { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--inj-q) 18%, transparent); color: var(--inj-q); }
.wv-inj-d   { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--orange) 16%, transparent); color: var(--orange); }

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
.wv-compare-win { color: var(--win); }
.wv-compare-lose { color: var(--loss); }

/* Shared-label compare: value(a) · LABEL · value(b), one label per metric */
.wv-cmp { padding: 6px 14px 12px; }
.wv-cmp-row, .wv-cmp-head {
  display: grid; grid-template-columns: 1fr minmax(86px, auto) 1fr;
  align-items: center; gap: 10px; padding: 7px 0;
  border-bottom: 1px solid var(--border);
}
.wv-cmp-row:last-child { border-bottom: none; }
.wv-cmp-head { border-bottom: 2px solid var(--border); padding: 4px 0 10px; align-items: end; }
.wv-cmp-a { text-align: right; font-weight: 700; font-size: 13px; }
.wv-cmp-b { text-align: left;  font-weight: 700; font-size: 13px; }
.wv-cmp-lbl {
  text-align: center; color: var(--text-muted); font-size: 11px;
  text-transform: uppercase; letter-spacing: .03em; font-weight: 600;
}
.wv-cmp-hcol:first-child { text-align: right; }
.wv-cmp-name { font-size: 15px; font-weight: 800; color: var(--text); }
.wv-cmp-sub  { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
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
      <div id="wvComparePanel" style="display:none;scroll-margin-top:16px;"></div>
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
  const yr = c.season ? '<sup class="wv-yr" title="from the ' + c.season + ' season">’' + String(c.season).slice(-2) + '</sup>' : '';
  const tip = 'Consistency ' + c.consistency + '/100 · boom ' +
              Math.round(c.boom_rate * 100) + '% · bust ' +
              Math.round(c.bust_rate * 100) + '% (' + c.games + ' g' +
              (c.season ? ', ' + c.season : '') + ')';
  return '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Floor–Ceil</span>' +
         '<span class="wv-ss-stat-val muted">' + c.floor + '–' + c.ceiling + yr + '</span></div>' +
         '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Profile</span>' +
         '<span class="wv-ss-cons wv-ss-cons-' + k + '" title="' + tip + '">' + c.label + '</span></div>';
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
            (upU ? 'var(--win)' : 'var(--loss)') + '" title="Last-3-week ' + lblU + ' vs season avg">' +
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

  // Once both players are picked the panel appears (often above the fold if you
  // scrolled down to pick) - bring it into view.
  if (wvCompare[0] && wvCompare[1]) {{
    const panel = document.getElementById('wvComparePanel');
    if (panel) panel.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }}
}}

function wvIsSelected(id) {{
  return (wvCompare[0] && wvCompare[0].player_id === id) ||
         (wvCompare[1] && wvCompare[1].player_id === id);
}}

// Derive every comparable field for one player (values + a sortable number
// where a winner makes sense).
function wvCmpDerive(p) {{
  const c = p.consistency, cOk = c && !c.small_sample;
  const yr = cOk && c.season ? '<sup class="wv-yr" title="from the ' + c.season + ' season">’' + String(c.season).slice(-2) + '</sup>' : '';
  const pk = !cOk ? '' : (c.label === 'Steady' ? 'steady'
    : c.label === 'Volatile' ? 'volatile'
    : c.label === 'Boom or bust' ? 'boombust' : 'balanced');
  const env = p.weather || p.game_env;
  return {{
    proj:     p.proj_pts > 0 ? p.proj_pts : null,
    l4:       p.recent_ppg > 0 ? p.recent_ppg : (p.season_ppg > 0 ? p.season_ppg : null),
    floorNum: cOk ? c.floor : null,
    flCeil:   cOk ? (c.floor + '–' + c.ceiling + yr) : '–',
    profile:  cOk ? `<span class="wv-ss-cons wv-ss-cons-${{pk}}">${{c.label}}</span>` : '–',
    boomBust: cOk ? (Math.round(c.boom_rate * 100) + '% / ' + Math.round(c.bust_rate * 100) + '%') : '–',
    opp:      p.opponent || (p.on_bye ? 'BYE' : '–'),
    def:      p.fpts_against > 0 ? `${{p.fpts_against}} pts` : (p.on_bye ? 'BYE' : '–'),
    defCls:   wvMuClass(p.def_rank, p.def_total),
    mu:       (!p.on_bye ? wvMuChip(p.def_rank, p.def_total) : '') || (p.on_bye ? '–' : 'No data'),
    vegasNum: p.implied_total != null ? p.implied_total : null,
    vegas:    p.implied_total != null ? (p.implied_total + ' implied') : '–',
    venue:    env ? `<span class="wv-ss-env wv-ss-env-${{env.kind}}">${{env.label}}</span>` : (p.on_bye ? 'BYE' : '–'),
    dynasty:  p.pos_rank_label || '–',
  }};
}}

// Winner classes for a pair of numeric values (higher is better by default).
function wvWinPair(av, bv, higher) {{
  if (av == null || bv == null || av === bv) return ['', ''];
  const aBetter = (higher === false) ? av < bv : av > bv;
  return aBetter ? ['wv-compare-win', 'wv-compare-lose'] : ['wv-compare-lose', 'wv-compare-win'];
}}

function wvRenderCompare() {{
  const panel = document.getElementById('wvComparePanel');
  const a = wvCompare[0], b = wvCompare[1];
  if (!a || !b) {{ panel.style.display = 'none'; return; }}
  panel.style.display = 'block';

  const da = wvCmpDerive(a), db = wvCmpDerive(b);
  // Shared single label per metric: value(a) · LABEL · value(b).
  function row(label, aHtml, bHtml, aCls, bCls) {{
    return '<div class="wv-cmp-row">' +
      '<span class="wv-cmp-a ' + (aCls || '') + '">' + aHtml + '</span>' +
      '<span class="wv-cmp-lbl">' + label + '</span>' +
      '<span class="wv-cmp-b ' + (bCls || '') + '">' + bHtml + '</span>' +
      '</div>';
  }}
  const dash = (v) => (v == null ? '–' : v);
  const wProj = wvWinPair(da.proj, db.proj, true);
  const wL4   = wvWinPair(da.l4, db.l4, true);
  const wFl   = wvWinPair(da.floorNum, db.floorNum, true);
  const wVeg  = wvWinPair(da.vegasNum, db.vegasNum, true);

  const sub = (p) => [p.team, p.pos_rank_label, p.opponent || (p.on_bye ? 'BYE' : '')].filter(Boolean).join(' · ');

  panel.innerHTML = `
    <div class="wv-compare-panel">
      <div class="wv-compare-header">
        <span>Compare</span>
        <button onclick="wvClearCompare()" style="font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--text-muted);cursor:pointer;">Clear</button>
      </div>
      <div class="wv-cmp">
        <div class="wv-cmp-head">
          <div class="wv-cmp-hcol"><div class="wv-cmp-name">${{a.name}}</div><div class="wv-cmp-sub">${{sub(a)}}</div></div>
          <div class="wv-cmp-lbl"></div>
          <div class="wv-cmp-hcol"><div class="wv-cmp-name">${{b.name}}</div><div class="wv-cmp-sub">${{sub(b)}}</div></div>
        </div>
        ${{row('Proj PPG', dash(da.proj), dash(db.proj), wProj[0], wProj[1])}}
        ${{row('L4 PPG', dash(da.l4), dash(db.l4), wL4[0], wL4[1])}}
        ${{row('Floor–Ceil', da.flCeil, db.flCeil, wFl[0], wFl[1])}}
        ${{row('Profile', da.profile, db.profile)}}
        ${{row('Boom / Bust', da.boomBust, db.boomBust)}}
        ${{row('Opponent', da.opp, db.opp)}}
        ${{row('Def vs pos', da.def, db.def, da.defCls, db.defCls)}}
        ${{row('Matchup', da.mu, db.mu)}}
        ${{row('Vegas total', da.vegas, db.vegas, wVeg[0], wVeg[1])}}
        ${{row('Venue', da.venue, db.venue)}}
        ${{row('Dynasty rank', da.dynasty, db.dynasty)}}
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
