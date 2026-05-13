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
}
.wv-pos-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }

/* Mobile tab bar */
.wv-tab-bar {
  display: none;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--border);
  margin-bottom: 16px;
  background: var(--card);
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
.signal-rising { background: #3b82f620; color: #3b82f6; }
.signal-value { background: #8b5cf620; color: #8b5cf6; }
.signal-aging { background: #f59e0b20; color: #f59e0b; }
.signal-hold { background: var(--row); color: var(--text-muted); }
.wv-ss-pos-group { margin-bottom: 16px; }
.wv-ss-pos-label { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
.wv-ss-player {
  display: flex; align-items: center; justify-content: space-between;
  padding: 9px 12px; border-radius: 8px; background: var(--card);
  border: 1px solid var(--border); margin-bottom: 6px; cursor: pointer;
}
.wv-ss-player.wv-ss-start { border-color: #10b981; background: #10b98108; }
.wv-ss-player.wv-ss-bye { opacity: .55; }
.wv-ss-left { flex: 1; min-width: 0; }
.wv-ss-start-badge      { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: #10b98120; color: #10b981; flex-shrink: 0; }
.wv-ss-flex-start-badge { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: #10b98120; color: #10b981; border: 1px solid #10b98140; flex-shrink: 0; }
.wv-ss-flex-badge       { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: #3b82f620; color: #3b82f6; flex-shrink: 0; }
.wv-ss-sit-badge        { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: var(--row); color: var(--text-muted); flex-shrink: 0; }
.wv-ss-bye-badge        { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: #f59e0b20; color: #f59e0b; flex-shrink: 0; }
.wv-ss-opp  { color: var(--accent, #3b82f6); font-weight: 600; }
.wv-ss-pts  { color: var(--text-muted); }
.wv-ss-slot-count { font-size: 10px; font-weight: 500; color: var(--text-muted); }
.wv-inj-out  { font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 4px; background: #ef444420; color: #ef4444; margin-left: 4px; }
.wv-inj-q    { font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 4px; background: #f59e0b20; color: #f59e0b; margin-left: 4px; }
.wv-inj-d    { font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 4px; background: #f97316a0; color: #f97316; margin-left: 4px; }
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

  <!-- Mobile tab bar (hidden on desktop via CSS) -->
  <div class="wv-tab-bar">
    <button class="wv-tab-btn active" id="wvTabWaivers" onclick="wvSetTab('waivers')">Waiver Wire</button>
    <button class="wv-tab-btn" id="wvTabStartSit" onclick="wvSetTab('startsit')">Start/Sit</button>
  </div>

  <!-- Two-column layout (tabs on mobile) -->
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

function wvLoad() {{
  // Load waiver candidates
  fetch(`/api/waiver-candidates?platform=${{WV_PLATFORM}}&league_id=${{WV_LEAGUE_ID}}&season=${{WV_SEASON}}`)
    .then(r => r.json())
    .then(d => {{ wvWaiverData = d.candidates || []; wvRenderWaivers(); }})
    .catch(() => {{ document.getElementById('wvWaiverList').innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:20px;">Unable to load</div>'; }});

  // Load start-sit options
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
    .catch(() => showLoginGate('wvStartSit', {{
      title: 'Sign in to see your lineup',
      description: 'Enter your Sleeper username to get personalized start/sit recommendations for your roster.'
    }}));
}}

function wvRenderWaivers() {{
  const list = document.getElementById('wvWaiverList');
  let players = wvWaiverData;
  if (wvCurrentPos !== 'ALL') players = players.filter(p => p.position === wvCurrentPos);
  if (!players.length) {{ list.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:20px;">No players found</div>'; return; }}
  list.innerHTML = players.slice(0, 20).map(p => `
    <div class="wv-player-row" onclick="openPlayerModal('${{p.player_id}}', '${{p.name.replace(/'/g,"\\'")}}')">
      <div>
        <div class="wv-player-name">${{p.name}}</div>
        <div class="wv-player-sub">${{[p.position, p.team, p.pos_rank_label, p.age ? 'Age ' + parseFloat(p.age).toFixed(1) : ''].filter(Boolean).join(' · ')}}</div>
      </div>
      <div class="wv-right">
        <span class="wv-signal ${{p.signal_class}}">${{p.signal}}</span>
        <span class="wv-value">${{Math.round(p.value)}}</span>
      </div>
    </div>
  `).join('');
}}

function wvRenderStartSit() {{
  const el = document.getElementById('wvStartSit');
  const positions = wvCurrentPos === 'ALL' ? ['QB','RB','WR','TE'] : [wvCurrentPos];
  const reqs = wvStartSitData._lineup_requirements || {{}};
  const sections = positions.map(pos => {{
    const players = (wvStartSitData.positions || {{}})[pos] || [];
    if (!players.length) return '';
    const rows = players.slice(0, 6).map(p => {{
      const isStart     = p.start === true;
      const isFlexStart = p.flex_start === true;
      const isFlex      = p.flex_eligible === true && !isStart;
      const isBye       = p.on_bye === true;
      const badge = isBye
        ? '<span class="wv-ss-bye-badge">BYE</span>'
        : isFlexStart
          ? '<span class="wv-ss-flex-start-badge">FLEX</span>'
          : isStart
            ? '<span class="wv-ss-start-badge">START</span>'
            : isFlex
              ? '<span class="wv-ss-flex-badge">FLEX?</span>'
              : '<span class="wv-ss-sit-badge">SIT</span>';
      const inj = (p.injury_status || '').toUpperCase();
      const injBadge = !inj ? '' :
        (inj === 'IR' || inj === 'OUT' || inj === 'PUP' || inj === 'SUSP')
          ? `<span class="wv-inj-out">${{inj}}</span>`
          : (inj === 'DOUBTFUL' || inj === 'D')
            ? `<span class="wv-inj-d">D</span>`
            : (inj === 'QUESTIONABLE' || inj === 'Q' || inj === 'GTD')
              ? `<span class="wv-inj-q">Q</span>`
              : `<span class="wv-inj-q">${{inj}}</span>`;
      const matchup = p.opponent ? `<span class="wv-ss-opp">${{p.opponent}}</span>` : '';
      const pts = p.avg_pts > 0 ? `<span class="wv-ss-pts">${{p.avg_pts}} avg</span>` : '';
      return `
        <div class="wv-ss-player ${{isStart ? 'wv-ss-start' : ''}} ${{isBye ? 'wv-ss-bye' : ''}}" onclick="openPlayerModal('${{p.player_id}}', '${{(p.name||'').replace(/'/g,"\\'")}}')">
          <div class="wv-ss-left">
            <div class="wv-player-name">${{p.name}}${{injBadge}}</div>
            <div class="wv-player-sub">${{[p.team, p.pos_rank_label, matchup, pts].filter(Boolean).join(' · ')}}</div>
          </div>
          ${{badge}}
        </div>`;
    }}).join('');
    const slotCount = reqs[pos] || 1;
    return `<div class="wv-ss-pos-group"><div class="wv-ss-pos-label">${{pos}} <span class="wv-ss-slot-count">(${{slotCount}} starter${{slotCount > 1 ? 's' : ''}})</span></div>${{rows}}</div>`;
  }}).join('');
  el.innerHTML = sections || '<div style="color:var(--text-muted);text-align:center;padding:20px;">No roster data found</div>';
}}

document.addEventListener('DOMContentLoaded', wvLoad);
</script>
"""

    return style + html_body + script
