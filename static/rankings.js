// Player Rankings page module — extracted verbatim from the inline <script>
// in page_players (app.py) so the browser caches and minifies it instead of
// re-downloading ~790 lines of inline JS on every rankings navigation.
// Loaded deferred, after app.js and after the inline window.__leagueTePremium
// injection, so both are available when this runs.

// Dock the sticky rankings header just beneath the sticky top-nav. The nav's
// height changes with the viewport and PWA safe-area insets (and shrinks to just
// the logo on phones), so measure it rather than hardcode an offset.
(function prStickyOffset() {
  function setOffset() {
    var nav = document.querySelector('.top-nav');
    var h = nav ? Math.round(nav.getBoundingClientRect().height) : 56;
    document.documentElement.style.setProperty('--pr-sticky-top', h + 'px');
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setOffset);
  } else {
    setOffset();
  }
  window.addEventListener('resize', setOffset);
})();

var prAllPlayers = [];
var prIndicators = {};
var prSparklines = {};
var prLeagueType   = '1qb';
var prLeagueSize   = 10;
var prScoringType  = 'dynasty';  // 'dynasty' | 'redraft'
// Positional rank (1-based, per position) for the ACTIVE view, keyed by player
// id. Recomputed each render from prGetValue() so it tracks the redraft/dynasty
// x 1qb/sf x size toggles instead of the dynasty-only precomputed *_pos_rank
// fields (e.g. an aging back is RB5 in redraft but RB12 in dynasty).
var prPosRankMap   = {};
// Default to the league's TE premium (injected from its scoring settings)
// and fall back to Off for the public/no-league view.
var prTePremium    = (typeof window.__leagueTePremium === 'number') ? window.__leagueTePremium : 0;
// Convert TE-premium points/reception into a value multiplier for TEs.
// ~+20% at full (1.0) PPR-TE premium, scaled linearly.
function prTeBoost(pos) {
  if (!prTePremium || pos !== 'TE') return 1;
  return 1 + prTePremium * 0.20;
}
// Reflect the league-derived TE premium in the settings toggle on load.
(function(){
  try {
    document.querySelectorAll('#prTepSection .settings-toggle').forEach(function(b){
      b.classList.toggle('active', Number(b.getAttribute('data-value')) === prTePremium);
    });
  } catch(e) {}
})();
var prPosFilters = new Set();   // empty = All
var prSearchQuery = '';
var prLoaded = false;
var prPage = 1;
var prPageSize = 50;
var prAdpSourceOptions = {};    // {startup|rookie|redraft: [{value,label}]} from payload
var prAdpSources = {};          // {startup|rookie|redraft: 'Sleeper'|...} label the server used
var prAdpSource = 'auto';       // currently selected ADP source ('auto' = server default)
var prAdpReloading = false;     // guards concurrent source re-fetches
var prAdpColumns = [];          // [{value,label}] per-source ADP columns for the sort-by-ADP view
var prAdpSortSource = '';       // which source column the ADP view is sorted by ('' = default)
var PR_ADP_COL_W = 96;          // px width of each ADP source column (fits "BR FANTASY"/"CONSENSUS" without colliding)
var PR_ADP_COL_W_MOBILE = 58;   // narrower on phones so all sources fit without overflowing the viewport (CSS shrinks the header font to match)

var PR_SPARK_W = 38, PR_SPARK_H = 26;  // logical (CSS) px
// Set true for the one render pass right after sparkline data first loads, so
// the lines draw on left-to-right; false for later sort/filter re-renders.
var _prSparkAnimate = false;

function _prReducedMotion() {
  return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
}

// Stroke the polyline up to a fraction `progress` (0–1) of its length, so the
// caller can sweep it on. Even x-spacing means progress maps cleanly to segments.
function _prPaintSpark(ctx2d, pts, w, h, progress, col) {
  ctx2d.clearRect(0, 0, w, h);
  const n = pts.length, segs = n - 1;
  const t = progress * segs;
  const full = Math.max(0, Math.min(segs, Math.floor(t)));  // never index out of range
  // The polyline visible at this progress (for the reveal animation).
  const line = [pts[0]];
  for (let i = 1; i <= Math.min(full, segs); i++) line.push(pts[i]);
  let tip;
  if (full < segs) {
    const frac = t - full, a = pts[full], b = pts[full + 1];
    tip = { x: a.x + (b.x - a.x) * frac, y: a.y + (b.y - a.y) * frac };
    line.push(tip);
  } else {
    tip = pts[n - 1];
  }
  // Filled area under the line.
  ctx2d.beginPath();
  ctx2d.moveTo(line[0].x, h);
  line.forEach(p => ctx2d.lineTo(p.x, p.y));
  ctx2d.lineTo(line[line.length - 1].x, h);
  ctx2d.closePath();
  ctx2d.globalAlpha = 0.15;
  ctx2d.fillStyle = col;
  ctx2d.fill();
  ctx2d.globalAlpha = 1;
  // The line itself.
  ctx2d.beginPath();
  ctx2d.moveTo(line[0].x, line[0].y);
  for (let i = 1; i < line.length; i++) ctx2d.lineTo(line[i].x, line[i].y);
  ctx2d.strokeStyle = col;
  ctx2d.lineWidth = 1.7;
  ctx2d.lineJoin = 'round';
  ctx2d.lineCap = 'round';
  ctx2d.stroke();
  // Endpoint dot at the current tip.
  ctx2d.beginPath();
  ctx2d.arc(tip.x, tip.y, 2.1, 0, Math.PI * 2);
  ctx2d.fillStyle = col;
  ctx2d.fill();
}

function _prDrawSparkline(canvas, data, animate) {
  if (!canvas || !data || data.length < 2) return;
  const dpr = window.devicePixelRatio || 1;
  const w = PR_SPARK_W, h = PR_SPARK_H;
  // Size the backing store for the display density so the line is crisp.
  canvas.width  = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  const ctx2d = canvas.getContext('2d');
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx2d.clearRect(0, 0, w, h);
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const pad = 3;   // room for the endpoint dot
  const pts = data.map((v, i) => ({
    x: pad + (i / (data.length - 1)) * (w - pad * 2),
    y: (h - pad) - ((v - min) / range) * (h - pad * 2)
  }));
  // Trend color from the app's own win/loss tokens (so it tracks the theme and
  // matches the rank arrows), green when the value is up over the window, red down.
  const cs = getComputedStyle(document.documentElement);
  const green = (cs.getPropertyValue('--win') || '#16a34a').trim();
  const red   = (cs.getPropertyValue('--loss') || '#ef4444').trim();
  const col = (data[data.length - 1] >= data[0]) ? green : red;
  if (animate && !_prReducedMotion()) {
    let start = null; const dur = 650;
    (function frame(now) {
      if (start === null) start = now;
      // Clamp to [0,1]: the first rAF timestamp can be slightly earlier than the
      // performance.now() that seeded `start`, which would make p (and progress)
      // negative → a negative array index → a crash that blanks the sparkline.
      const p = Math.min(1, Math.max(0, (now - start) / dur));
      _prPaintSpark(ctx2d, pts, w, h, 1 - Math.pow(1 - p, 3), col); // easeOutCubic
      if (p < 1) requestAnimationFrame(frame);
    })(performance.now());
  } else {
    _prPaintSpark(ctx2d, pts, w, h, 1, col);
  }
}

// Single source of truth for value column key selection (used by prGetValue + prGetSparkData)
function prValueKey(isSf) {
  if (isSf) return prLeagueSize === 10 ? 'sf_value' : 'sf_value_' + prLeagueSize;
  return prLeagueSize === 10 ? 'value' : 'value_' + prLeagueSize;
}

function prGetSparkData(pid) {
  const entry = prSparklines[pid];
  if (!entry) return null;
  if (Array.isArray(entry)) return entry.length >= 2 ? entry : null;
  if (prLeagueType === 'sf') {
    return entry[prValueKey(true)] || entry['sf_value'] || entry['value'] || null;
  }
  return entry[prValueKey(false)] || entry['value'] || null;
}

// Rescale a sparkline series so its last point equals `target` (the value shown
// in the row). The API scales each series to the calibrated base value, which
// omits the TE premium and size-specific column that prGetValue applies - so
// without this the sparkline's endpoint drifts from the displayed number.
function prAlignSpark(series, target) {
  if (!series || series.length < 2) return series;
  const last = series[series.length - 1];
  if (!last || !isFinite(last) || !isFinite(target) || target <= 0) return series;
  const scale = target / last;
  // Skip a no-op rescale (already aligned) to avoid tiny float churn.
  if (Math.abs(scale - 1) < 1e-6) return series;
  return series.map(v => v * scale);
}

// ---- Fuzzy search (mirrors trade calc logic) ----
function prFuzzyScore(name, query) {
  if (!name || !query) return 0;
  const n = name.toLowerCase(), q = query.toLowerCase();
  if (n.includes(q)) return 100 + (100 - n.indexOf(q));
  const nw = n.split(/[\\s\\-]+/), qw = q.split(/\\s+/).filter(Boolean);
  if (qw.length > 1) {
    if (qw.every((qx, i) => nw.slice(i).some(w => w.startsWith(qx)))) return 70;
  }
  if (nw.some(w => w.startsWith(q))) return 60;
  if (q.length >= 4) {
    for (let i = 0; i < q.length; i++) {
      const del = q.slice(0, i) + q.slice(i + 1);
      if (n.includes(del)) return 40;
      for (const c of 'abcdefghijklmnopqrstuvwxyz') {
        const sub = q.slice(0, i) + c + q.slice(i + 1);
        if (n.includes(sub) && sub !== q) return 30;
      }
    }
  }
  return 0;
}

function prGetValue(p) {
  let base;
  if (prScoringType === 'redraft') {
    base = Number(prLeagueType === 'sf'
      ? (p.redraft_value_sf ?? p.redraft_value_1qb ?? 0)
      : (p.redraft_value_1qb ?? 0));
  } else if (prLeagueType === 'sf') {
    base = Number(p[prValueKey(true)] ?? p.sf_value ?? p.value ?? 0);
  } else {
    base = Number(p[prValueKey(false)] ?? p.value ?? 0);
  }
  base *= prTeBoost(p.position);
  return Math.round(base * 10) / 10;
}

function prFormatValue(v) {
  if (!v || v <= 0) return '-';
  return v.toFixed(1);
}

// POS-cell label for the active view. Uses the per-render prPosRankMap (ranked by
// prGetValue) so it reflects the redraft/dynasty x 1qb/sf x size toggles; falls
// back to the dynasty precomputed labels only before the first render populates it.
function prPosRankLabel(p) {
  const n = prPosRankMap[String(p.id)];
  if (n) return (p.position || '') + n;
  return prLeagueType === 'sf'
    ? (p.sf_pos_rank_label || p.pos_rank_label || p.position)
    : (p.pos_rank_label || p.position);
}

function prIsRookie(id) {
  return prIndicators.rookies && prIndicators.rookies.includes(String(id));
}

function prIsBreakout(id) {
  return prIndicators.breakouts && prIndicators.breakouts.includes(String(id));
}

function prIsElite(id) {
  return prIndicators.elites && prIndicators.elites.includes(String(id));
}

function prIsProspect(id) {
  return prIndicators.prospects && prIndicators.prospects.includes(String(id));
}

// Settings panel toggle
function prToggleSettings() {
  const panel = document.getElementById('prSettingsPanel');
  const btn = document.getElementById('prSettingsBtn');
  if (!panel || !btn) return;

  const isOpen = panel.style.display === 'block';
  if (isOpen) {
    panel.style.display = 'none';
    btn.classList.remove('active');
  } else {
    panel.style.display = 'block';
    btn.classList.add('active');
    if (window.innerWidth <= 768) {
      const rect = btn.getBoundingClientRect();
      panel.style.position = 'fixed';
      panel.style.top = (rect.bottom + 6) + 'px';
      panel.style.left = '12px';
      panel.style.right = '12px';
      panel.style.minWidth = '';
    } else {
      panel.style.position = '';
      panel.style.top = '';
      panel.style.left = '';
      panel.style.right = '';
      panel.style.minWidth = '';
    }
  }
}

// Update active settings indicator tags
function updateSettingsIndicator() {
  const indicator = document.getElementById('prActiveSettings');
  if (!indicator) return;

  const tags = indicator.querySelectorAll('.active-setting-tag');
  if (tags[0]) tags[0].textContent = prLeagueSize + '-Team';
  if (tags[1]) tags[1].textContent = prLeagueType.toUpperCase();
  if (tags[2]) tags[2].textContent = prScoringType === 'redraft' ? 'Redraft' : 'Dynasty';
  const tepTag = document.getElementById('prTepTag');
  if (tepTag) {
    tepTag.style.display = prTePremium ? '' : 'none';
    tepTag.textContent = 'TE+' + prTePremium;
  }
}

function prSetScoringType(type) {
  prScoringType = type;
  // Update panel toggles
  document.querySelectorAll('#prSettingsPanel .settings-toggle[data-value]').forEach(btn => {
    const section = btn.closest('.settings-section');
    if (section && section.querySelector('.settings-section-label').textContent.includes('Scoring')) {
      btn.classList.toggle('active', btn.getAttribute('data-value') === type);
    }
  });
  // Hide league-size in redraft (size doesn't affect redraft values)
  const sizeSection = document.getElementById('prSizeSection');
  if (sizeSection) sizeSection.style.display = type === 'redraft' ? 'none' : '';
  // Hide PICK and ROOKIE filters in redraft
  document.querySelectorAll('.pos-pill[data-pos="PICK"], .pos-pill[data-pos="ROOKIE"]').forEach(btn => {
    btn.style.display = type === 'redraft' ? 'none' : '';
  });
  if (type === 'redraft' && (prPosFilters.has('PICK') || prPosFilters.has('ROOKIE'))) {
    prPosFilters.clear();
  }
  updateSettingsIndicator();
  prPage = 1;
  prRender();
}

function prSetLeagueType(type) {
  prLeagueType = type;

  // Update settings panel toggles
  document.querySelectorAll('#prSettingsPanel .settings-toggle[data-value]').forEach(btn => {
    const section = btn.closest('.settings-section');
    if (section && section.querySelector('.settings-section-label').textContent.includes('Format')) {
      btn.classList.toggle('active', btn.getAttribute('data-value') === type);
    }
  });

  updateSettingsIndicator();
  prPage = 1;
  prRender();
}

function prSetTePremium(pts) {
  prTePremium = Number(pts) || 0;
  // Highlight the active TE-premium button
  const sec = document.getElementById('prTepSection');
  if (sec) {
    sec.querySelectorAll('.settings-toggle[data-value]').forEach(btn => {
      btn.classList.toggle('active', Number(btn.getAttribute('data-value')) === prTePremium);
    });
  }
  updateSettingsIndicator();
  prPage = 1;
  prRender();
}

function prSetSize(size) {
  prLeagueSize = size;

  // Update settings panel toggles
  document.querySelectorAll('#prSettingsPanel .settings-toggle[data-value]').forEach(btn => {
    const section = btn.closest('.settings-section');
    if (section && section.querySelector('.settings-section-label').textContent.includes('Size')) {
      const btnSize = parseInt(btn.getAttribute('data-value'));
      btn.classList.toggle('active', btnSize === size);
    }
  });

  updateSettingsIndicator();
  prPage = 1;
  prRender();
}

// Multi-select position toggle
function prTogglePos(pos) {
  if (pos === 'ALL') {
    prPosFilters.clear();
  } else {
    if (prPosFilters.has(pos)) {
      prPosFilters.delete(pos);
    } else {
      prPosFilters.add(pos);
    }
  }
  // Sync button states
  document.querySelectorAll('.pos-pill').forEach(b => {
    const p = b.getAttribute('data-pos');
    if (p === 'ALL') {
      b.classList.toggle('active', prPosFilters.size === 0);
    } else {
      b.classList.toggle('active', prPosFilters.has(p));
    }
  });
  prPage = 1;
  prRender();
}

function prClearSearch() {
  document.getElementById('prSearch').value = '';
  prSearchQuery = '';
  document.getElementById('prSearchClear').style.display = 'none';
  prPage = 1;
  prRender();
}

// Build overall rank map keyed by player id (ranked by current value)
function prBuildRankMap() {
  const ranked = prAllPlayers
    .filter(p => p.position !== 'PICK')
    .slice()
    .sort((a, b) => prGetValue(b) - prGetValue(a));
  return new Map(ranked.map((p, i) => [String(p.id), i + 1]));
}

// Map sort key → { header label, cell value function }
// ADP for the current scoring mode (redraft vs dynasty/startup) and league type
// (SF vs 1QB). The server attaches both a dynasty (avg_pick/sf_avg_pick) and a
// redraft (redraft_avg_pick/sf_redraft_avg_pick) ADP per player; pick the pair
// matching the Redraft/Dynasty toggle. Null ADP renders as a dash and sorts last.
function prGetAdp(p) {
  if (prScoringType === 'redraft') {
    return prLeagueType === 'sf' ? p.sf_redraft_avg_pick : p.redraft_avg_pick;
  }
  return prLeagueType === 'sf' ? p.sf_avg_pick : p.avg_pick;
}
function prFormatAdp(p) {
  const v = prGetAdp(p);
  return (v != null) ? Number(v).toFixed(1) : '–';
}

// Which ADP source menu applies, given the current scoring mode.
function prAdpMode() {
  return prScoringType === 'redraft' ? 'redraft' : 'startup';
}

// ── Per-source ADP columns (the "sort by ADP" view) ─────────────────────────
// The ADP field on a player object for the current Dynasty/Redraft + 1QB/SF axis.
function prAdpField() {
  if (prScoringType === 'redraft') return prLeagueType === 'sf' ? 'sf_redraft_avg_pick' : 'redraft_avg_pick';
  return prLeagueType === 'sf' ? 'sf_avg_pick' : 'avg_pick';
}
// One source's ADP for a player on the current axis (null when that source has none).
function prAdpSourceVal(p, src) {
  const bs = p.adp_by_source && p.adp_by_source[src];
  if (!bs) return null;
  const v = bs[prAdpField()];
  return (v != null) ? Number(v) : null;
}
// Whether we have per-source columns to show (server sent them for this pool).
function prHasAdpColumns() { return Array.isArray(prAdpColumns) && prAdpColumns.length > 0; }
// The source the ADP view is currently sorted by (defaults to Consensus, else first).
function prActiveAdpSource() {
  if (!prHasAdpColumns()) return '';
  if (prAdpSortSource && prAdpColumns.some(c => c.value === prAdpSortSource)) return prAdpSortSource;
  const cons = prAdpColumns.find(c => c.value === 'consensus');
  return cons ? cons.value : prAdpColumns[0].value;
}
// ADP value used for sorting rows: the active source's when columns exist,
// otherwise the legacy single-source field.
function prGetAdpSortVal(p) {
  if (prHasAdpColumns()) return prAdpSourceVal(p, prActiveAdpSource());
  return prGetAdp(p);
}
// Header click: sort the ADP view by a chosen source column.
function prSortByAdpSource(src) {
  prAdpSortSource = src;
  prPage = 1;
  if (typeof prFlipRender === 'function') prFlipRender(); else prRender();
}
window.prSortByAdpSource = prSortByAdpSource;

// Show the ADP-source dropdown only while sorting by ADP, and populate it with
// the sources valid for the current scoring mode (Yahoo redraft-only, BR
// Fantasy dynasty/rookie only). Preserves the current selection when possible.
function prSyncAdpSourceUI(sortBy) {
  const wrap = document.getElementById('prAdpSrcWrap');
  const sel = document.getElementById('prAdpSource');
  if (!wrap || !sel) return;
  // The per-source columns replace the single-source dropdown entirely; only
  // fall back to the dropdown when the server didn't send columns.
  const show = sortBy === 'adp' && !prHasAdpColumns();
  wrap.style.display = show ? '' : 'none';
  if (!show) return;
  const mode = prAdpMode();
  // "Auto" is the default the page loads with (the server's memoized attach).
  // Label it with whatever source the server actually used, so the dropdown
  // never claims a source the shown ADP didn't come from.
  const usedLabel = prAdpSources[mode];
  const autoLabel = usedLabel && usedLabel !== 'none' ? ('Auto (' + usedLabel + ')') : 'Auto';
  let opts = [{ value: 'auto', label: autoLabel }].concat(prAdpSourceOptions[mode] || []);
  // Yahoo ADP needs a connected Yahoo league (league id + token); hide it
  // anywhere else so the menu never offers a source that can't return data here.
  const yahooAvailable = (typeof window.__platform !== 'undefined'
                          && window.__platform === 'yahoo' && !!window.__leagueId);
  if (!yahooAvailable) opts = opts.filter(o => o.value !== 'yahoo');
  const want = sel.value || prAdpSource;
  sel.innerHTML = opts.map(o =>
    '<option value="' + o.value + '">' + o.label + '</option>').join('');
  const has = opts.some(o => o.value === want);
  sel.value = has ? want : 'auto';
  prAdpSource = sel.value;
}

// Swap the table header between the normal columns and the ADP-source columns.
// Caches the default header once so it can be restored when leaving the ADP view.
function prSetupAdpHeader(adpView, cols, active, gridCols, extra) {
  const header = document.getElementById('prTableHeader');
  if (!header) return;
  if (header._prDefaultHTML == null) {
    header._prDefaultHTML = header.innerHTML;
    header._prDefaultCols = header.style.gridTemplateColumns;
  }
  header.classList.toggle('pr-adp-mode', !!adpView);
  if (adpView) {
    header.style.gridTemplateColumns = gridCols;
    const metaHead = extra
      ? '<span class="pr-adp-meta-h">Pos</span><span class="pr-adp-meta-h">Age</span><span class="pr-adp-meta-h">Team</span>'
      : '';
    header.innerHTML =
      '<span>#</span>' +
      '<span>Player</span>' +
      metaHead +
      cols.map(function (c) {
        const on = c.value === active;
        return '<span class="pr-adp-head' + (on ? ' pr-adp-head-active' : '') + '"' +
          ' role="button" tabindex="0" title="Sort by ' + c.label + ' ADP"' +
          ' onclick="prSortByAdpSource(\'' + c.value + '\')"' +
          ' onkeydown="if(event.key===\'Enter\'||event.key===\' \'){event.preventDefault();prSortByAdpSource(\'' + c.value + '\');}">' +
          c.label + (on ? ' <span class="pr-adp-sort-caret">▲</span>' : '') +
          '</span>';
      }).join('');
    header.dataset.adpMode = '1';
  } else if (header.dataset.adpMode === '1') {
    header.innerHTML = header._prDefaultHTML;
    header.style.gridTemplateColumns = header._prDefaultCols || '';
    header.dataset.adpMode = '0';
  }
}

// Re-fetch the pool with the chosen ADP source and merge only the ADP fields
// back onto the loaded players (preserving all client-side enrichment), then
// re-render. A no-op when nothing changed avoids a needless round trip.
function prReloadAdpSource() {
  const sel = document.getElementById('prAdpSource');
  if (!sel || prAdpReloading) return;
  prAdpSource = sel.value;
  // "Auto" re-fetches the server default (no adp_source override); any real
  // source overlays via the resolver.
  let url = '/api/league-players';
  if (prAdpSource && prAdpSource !== 'auto') {
    url += '?adp_source=' + encodeURIComponent(prAdpSource);
    if (window.__leagueId) url += '&league_id=' + encodeURIComponent(window.__leagueId);
    if (window.__platform) url += '&platform=' + encodeURIComponent(window.__platform);
  }
  prAdpReloading = true;
  fetch(url, { cache: 'no-store' })
    .then(r => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)))
    .then(resp => {
      const pls = Array.isArray(resp) ? resp : (resp.players || []);
      if (!Array.isArray(resp) && resp.adp_sources) prAdpSources = resp.adp_sources;
      const byId = {};
      pls.forEach(p => { byId[String(p.id)] = p; });
      const F = ['avg_pick', 'sf_avg_pick', 'rookie_avg_pick',
                 'sf_rookie_avg_pick', 'redraft_avg_pick', 'sf_redraft_avg_pick'];
      prAllPlayers.forEach(p => {
        const src = byId[String(p.id)];
        if (src) F.forEach(f => { p[f] = (src[f] != null ? src[f] : null); });
      });
      prPage = 1;
      prRender();
    })
    .catch(() => { /* keep current ADP on failure */ })
    .finally(() => { prAdpReloading = false; });
}

// var (not const) so this module can re-execute on a mobile soft-nav without a
// "PR_SORT_META has already been declared" error (const globals can't redeclare).
var PR_SORT_META = {
  rank:      { label: 'Value',    cell: p => prFormatValue(prGetValue(p)) },
  value:     { label: 'Value',    cell: p => prFormatValue(prGetValue(p)) },
  age:       { label: 'Age',      cell: p => p.age != null ? Number(p.age).toFixed(1) : '–' },
  pos_rank:  { label: 'Pos Rank', cell: p => prPosRankLabel(p) },
  ppg:       { label: 'PPG',       cell: p => p.ppg != null ? p.ppg.toFixed(1) : '–' },
  total_pts: { label: 'Total Pts', cell: p => p.total_pts != null ? p.total_pts.toFixed(1) : '–' },
  adp:       { label: 'ADP',       cell: p => prFormatAdp(p) },
};

// Re-render with a FLIP animation so rows glide to their new positions when the
// sort order changes (used by the Sort dropdown). Falls back to a plain render
// when the helper is unavailable or the user prefers reduced motion.
function prFlipRender() {
  var list = document.getElementById('prList');
  if (list && window.brFlipReorder) window.brFlipReorder(list, prRender);
  else prRender();
}

// Sort and filter players, then render rows into the main table
function prRender() {
  if (!prLoaded) return;
  const sortBy = document.getElementById('prSort').value;
  prSyncAdpSourceUI(sortBy);

  // "Sort by ADP" with per-source columns is a distinct view: the right-side
  // columns (Value/Age/Team/Pos) are swapped out for one column per ADP source.
  const isMobile = window.innerWidth <= 768;
  const adpView = (sortBy === 'adp') && prHasAdpColumns();
  const adpCols = adpView ? prAdpColumns : [];
  const adpActive = adpView ? prActiveAdpSource() : '';
  // On desktop keep the Pos / Age / Team columns (compact) alongside the source
  // columns; on mobile drop them so the source columns fit.
  const adpExtra = adpView && !isMobile;
  // Fixed-width source columns (slim, like Pos/Age/Team) instead of 1fr, so they
  // don't stretch across the row -- the Player column absorbs the slack. Wide
  // enough to fit the "BR FANTASY" header without truncating. On phones the
  // columns (and the rank col) shrink so all sources fit instead of overflowing
  // the viewport and clipping the last column / truncating the headers.
  // Only true phones overflow with 96px columns; tablets (481-768) have room.
  const _isPhone = window.innerWidth <= 480;
  const _adpColW = (_isPhone ? PR_ADP_COL_W_MOBILE : PR_ADP_COL_W) + 'px';
  const _adpRankW = _isPhone ? '38px' : '54px';
  const _adpSrcTracks = adpCols.map(() => _adpColW).join(' ');
  const ADP_GRID = adpView
    ? (adpExtra
        ? ('54px minmax(0,1fr) 40px 42px 46px ' + _adpSrcTracks)
        : (_adpRankW + ' minmax(0,1fr) ' + _adpSrcTracks))
    : '';
  prSetupAdpHeader(adpView, adpCols, adpActive, ADP_GRID, adpExtra);
  if (!adpView) {
    // On mobile (≤768px) the Age column is hidden, so switch the sort column
    // to show whatever is being sorted. On desktop all columns are visible.
    const _alwaysShowSort = sortBy === 'ppg' || sortBy === 'total_pts' || sortBy === 'adp';
    const sortMeta0 = (isMobile || _alwaysShowSort) ? (PR_SORT_META[sortBy] || PR_SORT_META.rank) : PR_SORT_META.rank;
    const sortHeaderEl = document.getElementById('prSortHeader');
    if (sortHeaderEl) sortHeaderEl.textContent = sortMeta0.label;
    // Hide age col only on mobile when sort=age (shown in sort col instead)
    const ageHeaderEl = document.getElementById('prAgeHeader');
    if (isMobile && ageHeaderEl) ageHeaderEl.style.visibility = sortBy === 'age' ? 'hidden' : '';
    const ageColEls = document.querySelectorAll('.pr-age');
    if (isMobile) ageColEls.forEach(el => el.style.visibility = sortBy === 'age' ? 'hidden' : '');
    else ageColEls.forEach(el => el.style.visibility = '');
  }
  const _alwaysShowSort = sortBy === 'ppg' || sortBy === 'total_pts' || sortBy === 'adp';
  const sortMeta = (isMobile || _alwaysShowSort) ? (PR_SORT_META[sortBy] || PR_SORT_META.rank) : PR_SORT_META.rank;

  let players = prAllPlayers.slice();

  // Positional rank for the ACTIVE view: rank every real player within his
  // position by the same value the VALUE column shows (prGetValue), so the POS
  // cell agrees with the ordering on screen. Computed over the full pool (not the
  // filtered/searched subset) so ranks are stable regardless of what's visible.
  prPosRankMap = {};
  {
    const _byPos = {};
    prAllPlayers.forEach(p => {
      if (p.position === 'PICK') return;
      if (p.is_rookie && !(p.team && p.team !== 'FA')) return;  // skip undrafted rookies
      if (!(prGetValue(p) > 0)) return;                          // no value in this view
      (_byPos[p.position] = _byPos[p.position] || []).push(p);
    });
    Object.keys(_byPos).forEach(pos => {
      _byPos[pos].sort((a, b) => prGetValue(b) - prGetValue(a));
      _byPos[pos].forEach((p, i) => { prPosRankMap[String(p.id)] = i + 1; });
    });
  }

  // In redraft mode exclude draft picks (not real players). Rookies stay —
  // they play this season, so a rookie with a redraft value (e.g. a first-round
  // RB) belongs on the board. The value check below drops any rookie the model
  // has no redraft value for, so we only show ones that actually rank.
  if (prScoringType === 'redraft') {
    players = players.filter(p => {
      if (p.position === 'PICK') return false;
      const v = prLeagueType === 'sf'
        ? (p.redraft_value_sf ?? p.redraft_value_1qb)
        : p.redraft_value_1qb;
      return v != null && Number(v) > 0;
    });
  }

  const isDrafted = p => p.is_rookie && p.team && p.team !== 'FA';

  // Rank numbers ("#") are OVERALL ranks, computed from the full eligible pool
  // BEFORE the position filter (using the current sort). So filtering to a
  // position keeps each player on their overall rank (e.g. TE McBride shows his
  // overall #, not #1) — which is what the overall movement arrow beside the #
  // measures, so the two line up — and #s stay stable under search too.
  const _rankSort = (a, b) => {
    if (sortBy === 'age')       return (a.age != null ? a.age : 99) - (b.age != null ? b.age : 99);
    if (sortBy === 'adp')       { const aA = prGetAdpSortVal(a); const bA = prGetAdpSortVal(b); return (aA != null ? aA : 99999) - (bA != null ? bA : 99999); }
    if (sortBy === 'pos_rank')  { const rA = prPosRankMap[String(a.id)] || 9999; const rB = prPosRankMap[String(b.id)] || 9999; return rA - rB; }
    if (sortBy === 'ppg')       return (b.ppg != null ? b.ppg : -1) - (a.ppg != null ? a.ppg : -1);
    if (sortBy === 'total_pts') return (b.total_pts != null ? b.total_pts : -1) - (a.total_pts != null ? a.total_pts : -1);
    return prGetValue(b) - prGetValue(a);
  };
  const _rankMap = new Map();
  {
    let _rankIdx = 0;
    players.slice()
      .filter(p => p.position !== 'PICK' && !(p.is_rookie && !isDrafted(p)))
      .sort(_rankSort)
      .forEach(p => { _rankMap.set(String(p.id), ++_rankIdx); });
  }

  // Position filter (multi-select) — narrows which rows show, but each keeps the
  // overall rank number computed above.
  if (prPosFilters.has('ROOKIE')) {
    players = players.filter(p => p.is_rookie);
  } else if (prPosFilters.size > 0) {
    players = players.filter(p => prPosFilters.has(p.position) && (!p.is_rookie || isDrafted(p)));
  } else {
    players = players.filter(p => !p.is_rookie || isDrafted(p));
  }

  // Search filter - fuzzy match, sort by score when query present
  if (prSearchQuery.length > 0) {
    const scored = players
      .map(p => ({
        p,
        score: Math.max(prFuzzyScore(p.name, prSearchQuery), prFuzzyScore(p.search_name, prSearchQuery))
      }))
      .filter(x => x.score > 0)
      .sort((a, b) => b.score !== a.score ? b.score - a.score : prGetValue(b.p) - prGetValue(a.p));
    players = scored.map(x => x.p);
  } else {
    // Normal sort when no search query
    players.sort((a, b) => {
      if (sortBy === 'value') {
        return prGetValue(b) - prGetValue(a);
      } else if (sortBy === 'adp') {
        const aA = prGetAdpSortVal(a); const bA = prGetAdpSortVal(b);
        return (aA != null ? aA : 99999) - (bA != null ? bA : 99999);
      } else if (sortBy === 'age') {
        return (a.age != null ? a.age : 99) - (b.age != null ? b.age : 99);
      } else if (sortBy === 'pos_rank') {
        const rA = prPosRankMap[String(a.id)] || 9999;
        const rB = prPosRankMap[String(b.id)] || 9999;
        return rA - rB;
      } else if (sortBy === 'ppg') {
        return (b.ppg != null ? b.ppg : -1) - (a.ppg != null ? a.ppg : -1);
      } else if (sortBy === 'total_pts') {
        return (b.total_pts != null ? b.total_pts : -1) - (a.total_pts != null ? a.total_pts : -1);
      } else {
        return prGetValue(b) - prGetValue(a);
      }
    });
  }

  const list   = document.getElementById('prList');
  const empty  = document.getElementById('prEmpty');
  const count  = document.getElementById('prCount');
  const header = document.getElementById('prTableHeader');

  if (players.length === 0) {
    list.innerHTML = '';
    empty.style.display = 'block';
    header.style.display = 'none';
    count.style.display = 'none';
    prRenderPagination(0, 0);
    return;
  }

  const total = players.length;
  const totalPages = Math.max(1, Math.ceil(total / prPageSize));
  if (prPage > totalPages) prPage = totalPages;
  const start = (prPage - 1) * prPageSize;
  const end   = Math.min(start + prPageSize, total);
  const pageSlice = players.slice(start, end);
  
  // Store filtered players count for pagination navigation
  window.prFilteredPlayers = players;

  empty.style.display = 'none';
  header.style.display = 'grid';
  count.style.display = 'block';
  count.textContent = 'Showing ' + (start + 1) + '–' + end + ' of ' + total + ' player' + (total !== 1 ? 's' : '');

  const _PR_TIER_COLORS = ['','#10b981','#22d3ee','#3b82f6','#8b5cf6','#a855f7','#f59e0b','#f97316','#94a3b8','#64748b'];
  const _PR_TIER_LABELS = ['','Elite','Star','High-End Starter','Starter','Flex','Bench','Deep Bench','Handcuff','Fringe'];

  list.innerHTML = '';
  let prevTier = null;
  pageSlice.forEach((p, i) => {
    const _tier = (sortBy === 'value' || sortBy === 'rank') ? prGetTier(p) : null;
    if (_tier && _tier !== prevTier) {
      const tc = _PR_TIER_COLORS[_tier] || '#64748b';
      const tl = _PR_TIER_LABELS[_tier] || ('Tier ' + _tier);
      const div = document.createElement('div');
      div.className = 'pr-tier-divider';
      div.innerHTML =
        `<div class="pr-tier-divider-line" style="background:${tc};"></div>` +
        `<span class="pr-tier-divider-label" style="color:${tc};" title="${tl}">T${_tier}</span>` +
        `<div class="pr-tier-divider-line" style="background:${tc};"></div>`;
      list.appendChild(div);
    }
    if (_tier) prevTier = _tier;
    const row = document.createElement('div');
    row.className = 'pr-player-row pr-grid-row';
    row.setAttribute('data-flip-key', 'p' + p.id);
    row.style.cursor = 'pointer';
    row.onclick = function(e) {
      e.stopPropagation();
      const _drafted = p.is_rookie && p.team && p.team !== 'FA';
      if (p.is_rookie && !_drafted) {
        if (typeof rkOpenModal === 'function') {
          rkOpenModal(p);
        } else {
          openProspectModal(p.id, p.name || 'Unknown');
        }
      } else {
        openPlayerModal(p.id, p.name || 'Unknown');
      }
    };

    const _drafted = p.is_rookie && p.team && p.team !== 'FA';
    const displayRank = (p.position === 'PICK' || (p.is_rookie && !_drafted)) ? '' : (_rankMap.get(String(p.id)) ?? (start + i + 1));
    const posRank = prPosRankLabel(p);
    const age = p.age != null ? Number(p.age).toFixed(1) : '–';
    const val = prGetValue(p);

    let badges = '';
    if (prIsRookie(p.id) || p.is_rookie) badges += '<span class="player-badge player-badge-rookie player-badge-collapsible" title="Rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> <span class="player-badge-label">ROOKIE</span></span>';
    else if (prIsProspect(p.id)) badges += '<span class="player-badge player-badge-prospect player-badge-collapsible" title="Prospect"><i class="fa-solid fa-seedling" aria-hidden="true"></i> <span class="player-badge-label">PROSPECT</span></span>';
    if (prIsBreakout(p.id)) badges += '<span class="player-badge player-badge-breakout player-badge-collapsible" title="Breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i> <span class="player-badge-label">BREAKOUT</span></span>';

    // Movement must match the ordering of the current view. We only track
    // dynasty value history, so:
    //   • Redraft has no redraft-ordered movement — hide the arrow rather than
    //     show a mismatched dynasty delta on redraft-sorted rows.
    //   • Dynasty SF uses the SF-ordered delta (QBs move very differently in
    //     Superflex); Dynasty 1QB uses the 1QB delta.
    const rankChange = prScoringType === 'redraft'
      ? null
      : (prLeagueType === 'sf' ? p.sf_rank_change_7d : p.rank_change_7d);
    // Align the sparkline to the value actually displayed in this row. The
    // /api/sparklines series ends at the calibrated base value (no TE premium,
    // base league size), while `val` = prGetValue(p) applies the TE premium and
    // the size-specific column - so rescale the whole series so its last point
    // equals `val`. Preserves the 7-day shape; guarantees the endpoint matches.
    const sparkData = prAlignSpark(prGetSparkData(p.id), val);

    // Arrows column: the value-trend sparkline only. Rank movement now sits
    // beside the rank number, so there's no chevron fallback here.
    let arrowCell = '';
    if (sparkData && sparkData.length >= 2) {
      arrowCell = `<canvas class="pr-sparkline" data-pid="${p.id}" title="7-day value trend"></canvas>`;
    }

    // Overall rank movement, shown next to the rank number. Only meaningful when
    // the rank IS the overall value board (not the ADP / PPG / age / pos views).
    let rankDeltaHTML = '';
    const _rankIsOverall = (sortBy === 'value' || sortBy === 'rank');
    if (displayRank && _rankIsOverall && rankChange != null && rankChange !== 0) {
      const _up = rankChange > 0;
      const _n = Math.abs(rankChange);
      rankDeltaHTML = `<span class="pr-rank-delta ${_up ? 'up' : 'down'}" title="${_n} spot${_n !== 1 ? 's' : ''} overall in 7 days">${_up ? '▲' : '▼'}${_n}</span>`;
    }

    const sortDisplay = p.position === 'PICK' && sortBy === 'age' ? '–' : sortMeta.cell(p);
    let sortDisplayHTML;
    if (p.position !== 'PICK' && sortBy === 'ppg' && p.ppg != null) {
      const pRank = p.ppg_rank ? (p.position + p.ppg_rank) : '-';
      sortDisplayHTML = `<span style="display:flex;flex-direction:column;align-items:flex-end;line-height:1.2;">`
        + `<span>${sortDisplay}</span>`
        + `<span style="font-size:10px;font-weight:600;color:var(--text-muted);">${pRank}</span>`
        + `</span>`;
    } else if (p.position !== 'PICK' && sortBy === 'total_pts' && p.total_pts != null) {
      const tRank = p.total_pts_rank ? (p.position + p.total_pts_rank) : '-';
      sortDisplayHTML = `<span style="display:flex;flex-direction:column;align-items:flex-end;line-height:1.2;">`
        + `<span>${sortDisplay}</span>`
        + `<span style="font-size:10px;font-weight:600;color:var(--text-muted);">${tRank}</span>`
        + `</span>`;
    } else {
      sortDisplayHTML = sortDisplay;
    }

    if (adpView) {
      // ADP view: # | Player | (Pos | Age | Team on desktop) | one ADP cell per
      // source (active source highlighted).
      row.classList.add('pr-adp-mode');
      row.style.gridTemplateColumns = ADP_GRID;
      const metaCells = adpExtra
        ? ('<span class="pr-adp-meta">' + posRank + '</span>' +
           '<span class="pr-adp-meta">' + (p.position === 'PICK' ? '–' : age) + '</span>' +
           '<span class="pr-adp-meta">' + (p.team || '–') + '</span>')
        : '';
      // The ADP the row is sorted by (the highlighted source's), used as the
      // baseline the other sources' arrows compare against for this player.
      const _activeAdp = prAdpSourceVal(p, adpActive);
      row.innerHTML =
        '<span class="pr-rank">'  + (displayRank ? '#' + displayRank : '–') + '</span>' +
        '<span class="pr-name player-clickable">'  + (p.name || 'Unknown') + badges + '</span>' +
        metaCells +
        adpCols.map(function (c) {
          const v = prAdpSourceVal(p, c.value);
          const on = c.value === adpActive;
          // Arrow vs the sorted source: a lower ADP (earlier pick) means this
          // source ranks the player HIGHER than the sorted one → green ▲; a
          // higher ADP (later pick) ranks him LOWER → red ▼. The sorted source
          // itself is the baseline, and equal/missing values get no arrow.
          let arrow = '';
          if (!on && v != null && _activeAdp != null && v !== _activeAdp) {
            const _higher = v < _activeAdp;
            arrow = '<span class="pr-adp-arrow ' + (_higher ? 'up' : 'down') +
              '" aria-hidden="true">' + (_higher ? '▲' : '▼') + '</span>';
          }
          return '<span class="pr-adp-cell' + (on ? ' pr-adp-cell-active' : '') + '">' +
            (v != null ? v.toFixed(1) : '–') + arrow + '</span>';
        }).join('');
    } else {
      row.innerHTML =
        '<span class="pr-rank">'  + (displayRank ? '#' + displayRank : '–') + rankDeltaHTML + '</span>' +
        '<span class="pr-arrows">' + arrowCell + '</span>' +
        '<span class="pr-name player-clickable">'  + (p.name || 'Unknown') + badges + '</span>' +
        '<span class="pr-pos-cell">' + posRank + '</span>' +
        '<span class="pr-age">'   + (p.position === 'PICK' ? '–' : age) + '</span>' +
        '<span class="pr-team">'  + (p.team || '–') + '</span>' +
        '<span class="pr-value">' + sortDisplayHTML + '</span>';

      if (sparkData && sparkData.length >= 2) {
        const cnv = row.querySelector('.pr-sparkline');
        if (cnv) _prDrawSparkline(cnv, sparkData, _prSparkAnimate);
      }
    }

    list.appendChild(row);
  });

  prRenderPagination(prPage, totalPages);
}

function prRenderPagination(page, totalPages) {
  let bar = document.getElementById('prPagination');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'prPagination';
    bar.className = 'pagination';
    document.getElementById('prList').insertAdjacentElement('afterend', bar);
    bar.innerHTML =
      `<div class="pagination-info"><span id="prPaginationText"></span></div>` +
      `<div class="pagination-controls">` +
        `<button class="pagination-btn" id="prPrevBtn" onclick="prGoPage('prev')" disabled>` +
          `<i class="fa-solid fa-chevron-left"></i> Previous` +
        `</button>` +
        `<div class="pagination-pages" id="prPageNumbers"></div>` +
        `<button class="pagination-btn" id="prNextBtn" onclick="prGoPage('next')" disabled>` +
          `Next <i class="fa-solid fa-chevron-right"></i>` +
        `</button>` +
      `</div>`;
  }

  if (totalPages <= 1) {
    bar.style.display = 'none';
    return;
  }

  bar.style.display = 'flex';

  const start = (page - 1) * prPageSize + 1;
  const end = Math.min(page * prPageSize, window.prFilteredPlayers.length);
  const total = window.prFilteredPlayers.length;
  document.getElementById('prPaginationText').textContent = `Showing ${start}-${end} of ${total} players`;

  document.getElementById('prPrevBtn').disabled = page === 1;
  document.getElementById('prNextBtn').disabled = page === totalPages;

  // Page number buttons - show up to 5, centred on current page
  const pageNumbers = document.getElementById('prPageNumbers');
  pageNumbers.innerHTML = '';
  const maxPages = 5;
  let startPage = Math.max(1, page - Math.floor(maxPages / 2));
  let endPage = Math.min(totalPages, startPage + maxPages - 1);
  if (endPage - startPage < maxPages - 1) startPage = Math.max(1, endPage - maxPages + 1);
  for (let i = startPage; i <= endPage; i++) {
    const btn = document.createElement('button');
    btn.className = 'pagination-page' + (i === page ? ' active' : '');
    btn.textContent = i;
    btn.onclick = (function(n){ return function(){ prGoPage(n); }; })(i);
    pageNumbers.appendChild(btn);
  }
}

function prGoPage(p) {
  if (p === 'prev') {
    prPage = Math.max(1, prPage - 1);
  } else if (p === 'next') {
    const totalPages = Math.ceil(window.prFilteredPlayers.length / prPageSize);
    prPage = Math.min(totalPages, prPage + 1);
  } else {
    prPage = p;
  }
  prRender();
  // Scroll to top of player list
  const el = document.getElementById('prTableHeader');
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Wire up search input
(function() {
  const inp   = document.getElementById('prSearch');
  const clear = document.getElementById('prSearchClear');
  if (!inp) return;

  inp.addEventListener('input', function() {
    prSearchQuery = inp.value.trim();
    clear.style.display = prSearchQuery.length > 0 ? 'block' : 'none';
    prPage = 1;
    prRender();
  });
})();

// Export current filtered list as CSV
function prExportCSV() {
  const players = window.prFilteredPlayers && window.prFilteredPlayers.length
    ? window.prFilteredPlayers : prAllPlayers;
  if (!players || !players.length) return;
  const q = (s) => '"' + String(s || '').replace(/"/g, '""') + '"';
  const header = ['Rank','Name','Position','Team','Age','Value','1QB Value','SF Value','7d Rank Change'];
  const rows = players.map((p, i) => {
    const val1qb = Number(p[prValueKey(false)] ?? p.value ?? 0);
    const valsf  = Number(p[prValueKey(true)]  ?? p.sf_value ?? 0);
    return [
      i + 1,
      q(p.name),
      p.position || '',
      p.team || '',
      p.age != null ? Number(p.age).toFixed(1) : '',
      Number(prGetValue(p)).toFixed(1),
      val1qb.toFixed(1),
      valsf.toFixed(1),
      p.rank_change_7d != null ? p.rank_change_7d : ''
    ];
  });
  const csv = [header, ...rows].map(r => r.join(',')).join('\\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'player_rankings.csv'; a.click();
  URL.revokeObjectURL(url);
  if (window.showToast) showToast('Player rankings downloaded!');
}

// Document-level listeners: bound once. This module re-executes on a mobile
// soft-nav (the dock swaps #page-root and re-runs the page script), so guard
// these so they don't stack a new listener on every navigation. They re-query
// their targets at event time, so a single binding stays correct across swaps.
if (!window.__prDocBound) {
  window.__prDocBound = true;

  // '/' focuses search from anywhere on the page
  document.addEventListener('keydown', function(e) {
    if (e.key === '/' && !['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) {
      e.preventDefault();
      const inp = document.getElementById('prSearch');
      if (inp) { inp.focus(); inp.select(); }
    }
  });

  // Close settings panel when clicking outside
  document.addEventListener('click', function(e) {
    const panel = document.getElementById('prSettingsPanel');
    const btn = document.getElementById('prSettingsBtn');

    if (panel && btn && panel.style.display === 'block') {
      if (!panel.contains(e.target) && !btn.contains(e.target)) {
        panel.style.display = 'none';
        btn.classList.remove('active');
      }
    }
  });
}

var prTierThresholds = {};

function prGetTier(p) {
  const lt  = prLeagueType || '1qb';
  const sz  = String(prLeagueSize || 10);
  const tbl = (prTierThresholds[lt] || {})[sz] || (prTierThresholds['1qb'] || {})['10'] || [];
  if (!tbl.length) return null;
  const val = prGetValue(p);
  for (let i = 0; i < tbl.length; i++) {
    if (val >= tbl[i]) return i + 1;
  }
  return tbl.length + 1; // catch-all - matches Python _ps_tier_of
}

// Load data
Promise.all([
  // Fail loud on a non-2xx: the server can return a 500 whose body is still
  // valid JSON (the _api_err shape), which would otherwise parse into an empty
  // players list and render a blank table with no error shown.
  // Pass league context so the server can add a Yahoo ADP column for Yahoo
  // leagues (Yahoo ADP needs a league id + token).
  (function () {
    let _u = '/api/league-players';
    const _q = [];
    if (window.__leagueId) _q.push('league_id=' + encodeURIComponent(window.__leagueId));
    if (window.__platform) _q.push('platform=' + encodeURIComponent(window.__platform));
    if (_q.length) _u += '?' + _q.join('&');
    return fetch(_u, { cache: 'no-store' });
  })().then(r => {
    if (!r.ok) throw new Error('league-players HTTP ' + r.status);
    return r.json();
  }),
  fetch('/api/player-indicators?league_type=1qb&league_size=10', { cache: 'no-store' })
    .then(r => r.json()).catch(() => ({}))
]).then(([resp, indicators]) => {
  prIndicators = indicators || {};
  // Support both old (array) and new (object with players + tier_thresholds) format
  const rawPlayers = Array.isArray(resp) ? resp : (resp.players || []);
  prTierThresholds = (!Array.isArray(resp) && resp.tier_thresholds) ? resp.tier_thresholds : {};
  prAdpSourceOptions = (!Array.isArray(resp) && resp.adp_source_options) ? resp.adp_source_options : {};
  prAdpSources = (!Array.isArray(resp) && resp.adp_sources) ? resp.adp_sources : {};
  prAdpColumns = (!Array.isArray(resp) && Array.isArray(resp.adp_columns)) ? resp.adp_columns : [];

  // Helper function to calculate precise age from birthday
  function calculateAgeFromBirthday(bDay) {
    if (!bDay) return null;
    try {
      const parts = bDay.split('/');
      if (parts.length !== 3) return null;
      const [month, day, year] = parts.map(Number);
      const birthDate = new Date(year, month - 1, day);
      const today = new Date();
      
      // Calculate precise age including partial years
      let age = today.getFullYear() - birthDate.getFullYear();
      const monthDiff = today.getMonth() - birthDate.getMonth();
      const dayDiff = today.getDate() - birthDate.getDate();
      
      if (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0)) {
        age--;
      }
      
      // Calculate partial year as decimal
      const lastBirthday = new Date(
        today.getFullYear() - (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0) ? 1 : 0),
        birthDate.getMonth(),
        birthDate.getDate()
      );
      const daysSinceBirthday = (today - lastBirthday) / (1000 * 60 * 60 * 24);
      const daysInYear = (today.getFullYear() % 4 === 0 && (today.getFullYear() % 100 !== 0 || today.getFullYear() % 400 === 0)) ? 366 : 365;
      
      age += daysSinceBirthday / daysInYear;
      return Math.round(age * 10) / 10; // Round to 1 decimal place
    } catch (e) {
      return null;
    }
  }

  prAllPlayers = rawPlayers
    .filter(p => p && p.id != null)
    .map(p => {
      // Calculate precise age from birthday if available
      const preciseAge = calculateAgeFromBirthday(p.bDay);
      
      return {
        id:               String(p.id),
        name:             p.name || p.full_name || 'Unknown',
        team:             p.team || '',
        position:         String(p.position || '').toUpperCase(),
        age:              preciseAge !== null ? preciseAge : (p.age != null ? Number(p.age) : null),
        value:            Number(p.value    || 0),
        value_8:          Number(p.value_8  || p.value    || 0),
        value_12:         Number(p.value_12 || p.value    || 0),
        value_14:         Number(p.value_14 || p.value    || 0),
        sf_value:         Number(p.sf_value    || p.value || 0),
        sf_value_8:       Number(p.sf_value_8  || p.sf_value || p.value || 0),
        sf_value_12:      Number(p.sf_value_12 || p.sf_value || p.value || 0),
        sf_value_14:      Number(p.sf_value_14 || p.sf_value || p.value || 0),
        redraft_value_1qb: p.redraft_value_1qb != null ? Number(p.redraft_value_1qb) : null,
        redraft_value_sf:  p.redraft_value_sf  != null ? Number(p.redraft_value_sf)  : null,
        avg_pick:            p.avg_pick            != null ? Number(p.avg_pick)            : null,
        sf_avg_pick:         p.sf_avg_pick         != null ? Number(p.sf_avg_pick)         : null,
        redraft_avg_pick:    p.redraft_avg_pick    != null ? Number(p.redraft_avg_pick)    : null,
        sf_redraft_avg_pick: p.sf_redraft_avg_pick != null ? Number(p.sf_redraft_avg_pick) : null,
        rookie_avg_pick:     p.rookie_avg_pick     != null ? Number(p.rookie_avg_pick)     : null,
        sf_rookie_avg_pick:  p.sf_rookie_avg_pick  != null ? Number(p.sf_rookie_avg_pick)  : null,
        // Per-source ADP for the sort-by-ADP columns. This whitelist map drops
        // any field not named here, so it must be carried through explicitly or
        // every source cell renders as a dash.
        adp_by_source:       (p.adp_by_source && typeof p.adp_by_source === 'object') ? p.adp_by_source : null,
        sf_rank_change_7d:   p.sf_rank_change_7d != null ? Number(p.sf_rank_change_7d) : null,
        pos_rank_label:   p.pos_rank_label    || '',
        sf_pos_rank_label:p.sf_pos_rank_label || '',
        pos_rank:         Number(p.pos_rank    || 9999),
        sf_pos_rank:      Number(p.sf_pos_rank || 9999),
        search_name:      p.search_name || '',
        is_rookie:        p.is_rookie === true,
        rank_change_7d:   p.rank_change_7d != null ? Number(p.rank_change_7d) : null,
        ppg:              p.ppg != null ? Number(p.ppg) : null,
        total_pts:        p.total_pts != null ? Number(p.total_pts) : null,
        ppg_rank:         p.ppg_rank != null ? Number(p.ppg_rank) : null,
        total_pts_rank:   p.total_pts_rank != null ? Number(p.total_pts_rank) : null,
        ppg_season:       p.ppg_season || null,
      };
    })
    .filter(p => ['QB','RB','WR','TE','PICK'].includes(p.position) || p.is_rookie)

  // A 200 response with an empty/malformed payload (e.g. the value table failed
  // to load server-side) would otherwise render as a silent blank table.
  if (!prAllPlayers.length) {
    throw new Error('league-players returned no players');
  }

  document.getElementById('prLoading').style.display = 'none';
  prLoaded = true;
  prRender();
  // Lazy-load sparklines - re-render with sparkline data once ready
  fetch('/api/sparklines?v=4').then(r => r.json()).then(function(data) {
    prSparklines = data || {};
    _prSparkAnimate = true;   // sweep the lines on for this first reveal only
    prRender();
    _prSparkAnimate = false;
  }).catch(function() {});
}).catch(err => {
  console.error('Error loading player rankings:', err);
  document.getElementById('prLoading').innerHTML =
    '<div style="color:#ef4444;">Failed to load players. Please refresh.</div>';
});
