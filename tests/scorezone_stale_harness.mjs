// Behavioral harness for the ScoreZone stale badge. Extracts the REAL
// _refresh() + stale-badge functions from static/scorezone.js and drives
// them against a fake clock / fetch / minimal DOM header, covering:
//   * one failed poll while the data is young must NOT raise the badge
//   * data older than _STALE_BADGE_MS (180s, aligned with the server-side
//     store_stale threshold) DOES raise it -- even if polls keep "succeeding"
//   * a latched badge element is removed by the 304 path itself (the old
//     code cleared the flag in memory but returned before any DOM update,
//     so endless healthy 304s left the badge on screen forever)
//   * the first poll carrying a fresh collect clears the badge, and the
//     badge's "updated" time is the payload's collect clock (updated_at),
//     not the poll's completion time and not the newest play's time
//
// Run: node tests/scorezone_stale_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'scorezone.js'), 'utf8');

function extract(name) {
  const marker = `function ${name}(`;
  const at = SRC.indexOf(marker);
  if (at < 0) throw new Error(`function ${name} not found in scorezone.js`);
  const start = SRC.slice(Math.max(0, at - 6), at).endsWith('async ') ? at - 6 : at;
  let depth = 0;
  for (let j = SRC.indexOf('{', at); j < SRC.length; j++) {
    if (SRC[j] === '{') depth++;
    else if (SRC[j] === '}' && --depth === 0) return SRC.slice(start, j + 1);
  }
  throw new Error(`unbalanced braces extracting ${name}`);
}

function constant(name) {
  const m = SRC.match(new RegExp('var\\s+' + name + '\\s*=\\s*(\\d+)'));
  if (!m) throw new Error('constant ' + name + ' not found');
  return parseInt(m[1], 10);
}
const DEADLINE = constant('_RZ_FETCH_DEADLINE_MS');
const STALE_BADGE = constant('_STALE_BADGE_MS');

const REAL = [
  '_rzLog', '_ownsScreen', '_cancelInflight', '_recoverScopeLoad',
  '_refresh', '_hasViewerIdentityFields',
  '_noteCollectAt', '_dataIsStale', '_syncStaleBadge',
].map(extract).join('\n');

// ── Fake infrastructure ─────────────────────────────────────────────────────
function abortError() { const e = new Error('aborted'); e.name = 'AbortError'; return e; }
const microtask = () => new Promise((r) => setImmediate(r));

class FakeClock {
  constructor(start) { this.now = start; this.timers = new Map(); this.seq = 1; }
  setTimeout(fn, ms) { const id = this.seq++; this.timers.set(id, { at: this.now + (ms || 0), fn }); return id; }
  clearTimeout(id) { this.timers.delete(id); }
  async advance(ms) {
    const target = this.now + ms;
    while (true) {
      let next = null;
      for (const [id, t] of this.timers) if (t.at <= target && (!next || t.at < next[1].at)) next = [id, t];
      if (!next) break;
      this.timers.delete(next[0]);
      this.now = next[1].at;
      next[1].fn();
      await microtask(); await microtask();
    }
    this.now = target;
    await microtask(); await microtask();
  }
}

class FakeController {
  constructor() {
    const cbs = [];
    this.signal = { aborted: false, _fire: () => cbs.forEach((c) => c()), addAbort: (c) => cbs.push(c) };
  }
  abort() { if (this.signal.aborted) return; this.signal.aborted = true; this.signal._fire(); }
}

// Minimal DOM: only the surface _syncStaleBadge touches.
class El {
  constructor(tag) { this.tagName = String(tag).toUpperCase(); this.children = []; this.className = ''; this.textContent = ''; this.parentNode = null; }
  _find(cls) {
    for (const c of this.children) {
      if (String(c.className).split(' ').includes(cls)) return c;
      const r = c._find(cls); if (r) return r;
    }
    return null;
  }
  querySelector(sel) { return this._find(sel.slice(1)); }
  insertBefore(node, ref) {
    node.parentNode = this;
    if (ref == null) this.children.push(node);
    else this.children.splice(this.children.indexOf(ref), 0, node);
    return node;
  }
  appendChild(node) { return this.insertBefore(node, null); }
  remove() {
    if (this.parentNode) {
      const i = this.parentNode.children.indexOf(this);
      if (i >= 0) this.parentNode.children.splice(i, 1);
      this.parentNode = null;
    }
  }
}

function makeFetch(plan) {
  return function fetch(url, init) {
    const d = plan(url) || {};
    if (d.network) return Promise.reject(new Error('network down'));
    const headers = { get: () => null };
    return Promise.resolve({
      ok: d.ok !== false, status: d.status || 200, headers,
      json() { return Promise.resolve(d.json || {}); },
    });
  };
}

function leaguePayload(extra) {
  return Object.assign({ scope: 'league', viewer_roster_id: '4', matchups: [], player_info: {},
    games: {}, pbp_by_game: {} }, extra || {});
}

const T0 = 1761430000000; // fixed epoch (ms) so collect clocks are exact

function build(opts) {
  const clock = new FakeClock(T0);
  const headerRight = new El('div'); headerRight.className = 'rz-header-right';
  const timerEl = new El('button'); timerEl.className = 'rz-refresh-timer';
  headerRight.appendChild(timerEl);
  const rootEl = new El('div');
  rootEl.appendChild(headerRight);
  rootEl.isConnected = true;
  rootEl.querySelectorAll = () => [];
  const doc = {
    hidden: false,
    createElement: (t) => new El(t),
    getElementById: (id) => (id === 'rz-timer' ? timerEl : null),
  };
  const win = { location: { pathname: '/sleeper/2026/123/scorezone' } };
  const factory = new Function(
    'D', `
    var setTimeout = D.setTimeout, clearTimeout = D.clearTimeout;
    var fetch = D.fetch, AbortController = D.AbortController;
    var document = D.document, window = D.window, root = D.root;
    var console = D.console;
    var _RZ_FETCH_DEADLINE_MS = ${DEADLINE};
    var _STALE_BADGE_MS = ${STALE_BADGE};
    var _scope = 'league', _streaming = false, _streamGen = 0, _reqSeq = 0, _inflight = null;
    var _scopeJustSwitched = false, _lastDataAt = null, _lastSuccessAt = null;
    var _lastCollectAt = null;
    var _lastPollFailed = false, _scopeLoadError = null;
    var _loadingScope = false, _loadingPlays = false, _countdown = 15;
    var _state = {}, _scopeCache = { league: null, user: null };
    var _rzEtagByScope = { league: '"e1"', user: null };
    var _alertsArmed = false, _flashRids = new Set();
    var _isDemo = false, _demoT = 0;
    function _render() {}
    function _partialUpdate() { _syncStaleBadge(); } // prod _partialUpdate syncs the badge
    function _setState(d) { _state = d || {}; }
    function _detectChanges() {}
    function _seedPrevStats() {} function _saveScopeRuntime() {} function _applyDefaultHero() {}
    function _pollInterval() { return 15; }
    ${REAL}
    return {
      refresh: _refresh, syncBadge: _syncStaleBadge, dataIsStale: _dataIsStale,
      noteCollectAt: _noteCollectAt,
      badge: () => D.headerRight.querySelector('.rz-data-stale'),
      state: () => ({ lastPollFailed: _lastPollFailed, lastCollectAt: _lastCollectAt }),
    };
    `);
  const D = {
    setTimeout: clock.setTimeout.bind(clock), clearTimeout: clock.clearTimeout.bind(clock),
    fetch: makeFetch(opts.plan), AbortController: FakeController,
    document: doc, window: win, root: rootEl, console: { debug: () => {} },
    headerRight,
  };
  const api = factory(D);
  api._headerRight = headerRight;
  return { api, clock };
}

let CURRENT_CLOCK = null;
const realDateNow = Date.now;
Date.now = () => (CURRENT_CLOCK ? CURRENT_CLOCK.now : realDateNow());
const flush = async (n) => { for (let i = 0; i < (n || 12); i++) await microtask(); };
const fmtTime = (ms) => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

async function run() {
  assert.equal(STALE_BADGE, 180000, 'badge threshold stays aligned with the server store_stale 180s');

  // ── §1: young data + one failed poll -> NO badge; old data -> badge ──────
  {
    let fail = false;
    const { api, clock } = build({ plan: () => fail
      ? { network: true }
      : { json: leaguePayload({ updated_at: CURRENT_CLOCK.now / 1000 }) } });
    CURRENT_CLOCK = clock;
    await api.refresh(); await flush();
    assert.equal(api.state().lastCollectAt, T0, 'collect clock seeded from payload updated_at');
    assert.equal(api.badge(), null, 'no badge on fresh data');

    fail = true;
    await clock.advance(60000); // one poll cadence (60s) later
    await api.refresh(); await flush();
    assert.equal(api.state().lastPollFailed, true, 'the poll really did fail');
    assert.equal(api.badge(), null, 'one failed poll with 60s-old data must not raise the badge');

    await clock.advance(121000); // data now 181s old, polls still failing
    await api.refresh(); await flush();
    const badge = api.badge();
    assert.ok(badge, 'genuinely old data (181s) raises the badge');
    assert.ok(badge.textContent.includes('Stale'), 'badge says Stale');
    assert.ok(badge.textContent.includes(fmtTime(T0)), 'badge prints the COLLECT time, got: ' + badge.textContent);
    console.log('ok - §1 young data + failed poll: no badge; 181s-old data: badge with collect time');
  }

  // ── §2: threshold boundaries around the poll cadence ─────────────────────
  {
    const { api, clock } = build({ plan: () => ({ json: leaguePayload() }) });
    CURRENT_CLOCK = clock;
    api.noteCollectAt({ updated_at: T0 / 1000 });
    for (const [ageS, expected] of [[61, false], [120, false], [179, false], [180, false], [181, true], [600, true]]) {
      clock.now = T0 + ageS * 1000;
      assert.equal(api.dataIsStale(), expected, `age ${ageS}s -> stale=${expected}`);
    }
    console.log('ok - §2 threshold: <=180s fresh, >180s stale');
  }

  // ── §3: THE LATCH -- a stuck badge element + healthy 304s ────────────────
  {
    let mode = 'ok';
    const { api, clock } = build({ plan: () => mode === '304'
      ? { status: 304 }
      : { json: leaguePayload({ updated_at: CURRENT_CLOCK.now / 1000 }) } });
    CURRENT_CLOCK = clock;
    await api.refresh(); await flush();
    assert.equal(api.badge(), null);
    // Simulate the legacy latch: a badge element stuck in the DOM (raised by
    // the old _lastPollFailed logic) while the underlying data is FRESH and
    // every poll answers 304. The old 304 path returned before any DOM
    // update, so this element survived forever.
    const stuck = new El('span');
    stuck.className = 'rz-stale-badge rz-data-stale';
    stuck.textContent = 'Stale · updated ' + fmtTime(T0);
    api._headerRight.insertBefore(stuck, api._headerRight.children[0]);
    assert.ok(api.badge(), 'latched badge is in the DOM');
    mode = '304';
    await clock.advance(15000);
    await api.refresh(); await flush();
    assert.equal(api.badge(), null, 'the 304 path itself must clear a latched badge when data is fresh');
    assert.equal(api.state().lastCollectAt, T0, 'a 304 must not advance the collect clock');
    console.log('ok - §3 latched badge cleared by the 304 path; collect clock untouched by 304');
  }

  // ── §4: dead collect -- 304s keep coming, data ages, badge appears, then
  //        the first FRESH payload clears it ────────────────────────────────
  {
    let mode = 'ok';
    const { api, clock } = build({ plan: () => mode === '304'
      ? { status: 304 }
      : { json: leaguePayload({ updated_at: CURRENT_CLOCK.now / 1000 }) } });
    CURRENT_CLOCK = clock;
    await api.refresh(); await flush();
    mode = '304'; // server collect dies: content never changes again
    await clock.advance(120000); // 2 minutes of "successful" 304 polls
    await api.refresh(); await flush();
    assert.equal(api.badge(), null, '120s of 304s: still within cadence headroom, no badge');
    await clock.advance(61000); // data now 181s old
    await api.refresh(); await flush();
    assert.ok(api.badge(), 'dead collect surfaces stale even though polls succeed (304)');
    mode = 'ok'; // fallback collect lands: fresh payload
    await clock.advance(15000);
    await api.refresh(); await flush();
    assert.equal(api.badge(), null, 'first fresh poll clears the badge');
    console.log('ok - §4 dead collect flags via 304 aging; first fresh payload clears');
  }

  console.log('\nALL SCOREZONE STALE BADGE HARNESS CHECKS PASSED');
}

run().catch((e) => { console.error(e); process.exit(1); });
