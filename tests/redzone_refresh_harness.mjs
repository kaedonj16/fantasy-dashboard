// Behavioral harness for the Redzone refresh / polling / streaming lifecycle
// (§1-3, §5). redzone.js is a DOM-guarded IIFE, so we extract the REAL async
// control-flow functions by source and evaluate them against injected fakes:
// a fake clock (mocked setTimeout/clearTimeout/Date.now), a fake fetch, a fake
// AbortController, and a fake NDJSON stream reader. This exercises the shipped
// _refresh() / _refreshUserStream() / _tick() logic -- ownership ordering,
// deadlines, inactivity timeout, manual-refresh recovery -- not a rewrite.
//
// Run: node tests/redzone_refresh_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'redzone.js'), 'utf8');

function extract(name) {
  const marker = `function ${name}(`;
  const at = SRC.indexOf(marker);
  if (at < 0) throw new Error(`function ${name} not found in redzone.js`);
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
const STREAM_DEADLINE = constant('_RZ_STREAM_DEADLINE_MS');
const STREAM_IDLE = constant('_RZ_STREAM_IDLE_MS');

const REAL = [
  '_rzLog', '_ownsScreen', '_cancelInflight', '_recoverScopeLoad', '_manualRefresh',
  '_refresh', '_refreshUserStream', '_tick', '_emptyUserState', '_mergeLeagueSlice',
  '_hasViewerIdentityFields',
].map(extract).join('\n');

// ── Fake infrastructure ─────────────────────────────────────────────────────
function abortError() { const e = new Error('aborted'); e.name = 'AbortError'; return e; }
const microtask = () => new Promise((r) => setImmediate(r));

class FakeClock {
  constructor() { this.now = 0; this.timers = new Map(); this.seq = 1; }
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

// Build a fetch that resolves/rejects per a scripted plan. `plan` is a function
// (url) => descriptor. Descriptors:
//   { json, ok, status, contentType }           immediate JSON response
//   { hangHeaders: true }                        never returns headers (abort → reject)
//   { hangBody: true, ... }                      headers ok, body/json hangs until abort
//   { network: true }                            fetch rejects immediately (network error)
//   { stream: [steps], contentType }             NDJSON stream; steps drive reader.read()
// A stream step is { chunk } | { done:true } | { stall:true } (never resolves until abort).
function makeFetch(plan, clock) {
  return function fetch(url, init) {
    const signal = init && init.signal;
    const d = plan(url) || {};
    if (d.network) return Promise.reject(new Error('network down'));
    if (d.hangHeaders) {
      return new Promise((_, reject) => { if (signal) signal.addAbort(() => reject(abortError())); });
    }
    const headers = { get: (k) => (String(k).toLowerCase() === 'content-type' ? (d.contentType || 'application/json') : null) };
    if (d.stream) {
      let i = 0;
      const body = {
        getReader() {
          return {
            read() {
              const step = d.stream[i++];
              if (!step) return Promise.resolve({ done: true });
              if (step.stall) return new Promise((_, reject) => { if (signal) signal.addAbort(() => reject(abortError())); });
              if (step.done) return Promise.resolve({ done: true });
              return Promise.resolve({ done: false, value: step.chunk });
            },
            cancel() { return Promise.resolve(); },
          };
        },
      };
      return Promise.resolve({ ok: d.ok !== false, status: d.status || 200, headers, body });
    }
    const resp = {
      ok: d.ok !== false, status: d.status || 200, headers,
      json() {
        if (d.hangBody) return new Promise((_, reject) => { if (signal) signal.addAbort(() => reject(abortError())); });
        if (d.parseError) return Promise.reject(new SyntaxError('bad json'));
        if (d.hangBodyUntil != null) {
          // Body arrives after a fake-clock delay; a deadline abort still wins.
          return new Promise((resolve, reject) => {
            if (signal) signal.addAbort(() => reject(abortError()));
            clock.setTimeout(() => resolve(d.json || {}), d.hangBodyUntil);
          });
        }
        return Promise.resolve(d.json || {});
      },
    };
    return Promise.resolve(resp);
  };
}

class FakeTextDecoder { decode(v) { return String(v == null ? '' : v); } }

// League-scope success payload (carries viewer_roster_id identity field).
function leaguePayload(extra) {
  return Object.assign({ scope: 'league', viewer_roster_id: '4', matchups: [], player_info: {},
    games: {}, pbp_by_game: {} }, extra || {});
}
function ndjson(obj) { return JSON.stringify(obj) + '\n'; }

// ── Build one isolated instance of the extracted logic ──────────────────────
function build(opts) {
  opts = opts || {};
  const clock = new FakeClock();
  const logs = [];
  const events = { renders: 0, partials: 0, applied: [], detects: [] };
  const doc = { hidden: !!opts.hidden, getElementById: () => null };
  const win = { location: { pathname: '/sleeper/2025/123/redzone' }, console: null };

  const factory = new Function(
    'D', `
    // ── injected primitives ──
    var setTimeout = D.setTimeout, clearTimeout = D.clearTimeout;
    var fetch = D.fetch, AbortController = D.AbortController, TextDecoder = D.TextDecoder;
    var document = D.document, window = D.window, root = D.root;
    var console = D.console;
    // ── lifecycle constants (kept in sync via the source) ──
    var _RZ_FETCH_DEADLINE_MS = ${DEADLINE};
    var _RZ_STREAM_DEADLINE_MS = ${STREAM_DEADLINE};
    var _RZ_STREAM_IDLE_MS = ${STREAM_IDLE};
    // ── mutable module state ──
    var _scope = D.scope, _streaming = false, _streamGen = 0, _reqSeq = 0, _inflight = null;
    var _scopeJustSwitched = false, _lastDataAt = null, _lastSuccessAt = null;
    var _lastPollFailed = false, _scopeLoadError = null;
    var _loadingScope = D.loadingScope, _loadingPlays = false, _countdown = 15;
    var _state = D.state, _scopeCache = D.scopeCache, _scopeRuntime = { league: null, user: null };
    var _mlFailed = null, _mlNames = [], _mlLoaded = null, _prevMatchupPts = {};
    var _feed = [], _shownFeedIds = new Set(), _isDemo = false, _demoT = 0;
    var _alertsArmed = false, _flashRids = new Set();
    // ── stubbed collaborators (record, never touch a real DOM) ──
    function _render() { D.ev.renders++; }
    function _partialUpdate() { D.ev.partials++; }
    function _setState(d) { _state = d || {}; D.ev.applied.push(d); }
    function _detectChanges(d, mode) { D.ev.detects.push(mode); }
    function _seedPrevStats() {} function _saveScopeRuntime() {} function _applyDefaultHero() {}
    function _seedMilestones() {} function _seedInjuries() {} function _seedLeaders() {}
    function _resetFeedSnapshots() {} function _hydrateFeed() {} function _eid(e) { return e && e.id; }
    function _runtimeIdentity() { return 'x'; }
    function _pollInterval() { return 15; }
    function _anyLive() { return false; }
    function _isGameDay() { return true; }
    function _fmtTimer(n) { return String(n); }
    ${REAL}
    return {
      refresh: _refresh, stream: _refreshUserStream, tick: _tick, manual: _manualRefresh,
      switchScope: function (s) { _saveScopeRuntime(); _streamGen++; _cancelInflight('scope-switch'); _streaming = false; _scope = s; _loadingScope = true; },
      setHidden: function (v) { document.hidden = v; },
      setCountdown: function (v) { _countdown = v; },
      state: function () { return {
        streaming: _streaming, inflight: _inflight, scope: _scope, reqSeq: _reqSeq, streamGen: _streamGen,
        loadingScope: _loadingScope, loadingPlays: _loadingPlays, lastPollFailed: _lastPollFailed,
        lastDataAt: _lastDataAt, lastSuccessAt: _lastSuccessAt, countdown: _countdown,
        scopeCache: _scopeCache, state: _state, scopeLoadError: _scopeLoadError,
      }; },
    };
  `);

  // A console that captures _rzLog output for assertions.
  const captureConsole = { debug: (tag, detail) => logs.push({ tag, detail }) };
  const D = {
    setTimeout: clock.setTimeout.bind(clock), clearTimeout: clock.clearTimeout.bind(clock),
    fetch: makeFetch(opts.plan || (() => ({ json: leaguePayload() })), clock),
    AbortController: FakeController, TextDecoder: FakeTextDecoder,
    document: doc, window: win, root: { querySelectorAll: () => [] }, console: captureConsole,
    scope: opts.scope || 'league', loadingScope: !!opts.loadingScope,
    state: opts.state || {}, scopeCache: opts.scopeCache || { league: null, user: null },
    ev: events,
  };
  const api = factory(D);
  return { api, clock, logs, events };
}

// Patch the global Date.now so the extracted code's Date.now() reads fake time.
let CURRENT_CLOCK = null;
const realDateNow = Date.now;
Date.now = () => (CURRENT_CLOCK ? CURRENT_CLOCK.now : realDateNow());

const flush = async (n) => { for (let i = 0; i < (n || 12); i++) await microtask(); };

async function run() {
  // ── §1: a stream that never returns HEADERS must not pin _streaming ────────
  {
    const { api, clock } = build({ scope: 'user', loadingScope: true, plan: (url) =>
      (url.indexOf('stream=1') >= 0
        ? { hangHeaders: true }                                   // stream never opens
        : { json: { scope: 'user', viewer_roster_ids: ['0:1'], matchups: [{ roster_id: '0:1' }],
            player_info: {}, games: {}, pbp_by_game: {} } }) });   // aggregate fallback works
    CURRENT_CLOCK = clock;
    const p = api.stream();
    await flush();
    assert.equal(api.state().streaming, true, 'stream starts owning the screen');
    await clock.advance(STREAM_IDLE + 1); // inactivity timeout fires -> abort -> fallback
    await p;
    const st = api.state();
    assert.equal(st.streaming, false, 'stalled stream (no headers) must release _streaming');
    assert.equal(st.inflight, null, 'and release the in-flight handle');
    assert.ok(st.state.matchups && st.state.matchups.length === 1, 'aggregate fallback recovered data');
    console.log('ok - §1 stream that never returns headers releases streaming + recovers');
  }

  // ── §1/§6: stream that STALLS BETWEEN CHUNKS, then manual refresh recovers ─
  {
    let phase = 0;
    const okSlice = { type: 'league', index: 0, matchups: [{ roster_id: '0:1', matchup_id: '0:1' }],
      rosters: [], users: [], player_info: {}, viewer_roster_id: '0:1', leagues: [{ league_id: 'l1', name: 'L1' }] };
    const { api, clock } = build({ scope: 'user', loadingScope: true, plan: (url) => {
      if (url.indexOf('stream=1') < 0) return { json: { scope: 'user', viewer_roster_ids: [] } };
      if (phase === 0) return { contentType: 'application/x-ndjson', stream: [
        { chunk: ndjson({ type: 'meta', week: 1, season: 2025, leagues: [{ name: 'L1' }] }) },
        { stall: true } ] };                                       // hangs after meta
      return { contentType: 'application/x-ndjson', stream: [
        { chunk: ndjson({ type: 'meta', week: 1, season: 2025, leagues: [{ name: 'L1' }] }) },
        { chunk: ndjson(okSlice) }, { done: true } ] };
    } });
    CURRENT_CLOCK = clock;
    const p1 = api.stream();
    await clock.advance(1);            // meta chunk consumed
    assert.equal(api.state().streaming, true);
    await clock.advance(STREAM_IDLE + 1); // inter-chunk stall trips inactivity
    await p1;
    assert.equal(api.state().streaming, false, 'inter-chunk stall releases streaming');
    phase = 1;
    const p2 = api.manual();           // user scope -> restarts the stream fresh
    await clock.advance(1);
    await p2;
    const st = api.state();
    assert.equal(st.streaming, false, 'manual recovery finished the stream');
    assert.ok(st.scopeCache.user && st.scopeCache.user.matchups.length === 1, 'recovered stream applied its league');
    console.log('ok - §1/§6 inter-chunk stall released, manual refresh recovered');
  }

  // ── §6: TIMEOUT DURING BODY/JSON read (headers ok, json() hangs to deadline) ─
  {
    const cached = { scope: 'league', viewer_roster_id: '1', matchups: [{ roster_id: '1' }],
      player_info: {}, games: {}, pbp_by_game: {} };
    const { api, clock } = build({ scope: 'league', loadingScope: true,
      scopeCache: { league: cached, user: null }, plan: () => ({ hangBody: true }) });
    CURRENT_CLOCK = clock;
    const p = api.refresh();
    await flush();
    assert.ok(api.state().inflight, 'request in flight while the body hangs');
    await clock.advance(DEADLINE + 1);   // deadline must fire DURING the json() read
    await p;
    const st = api.state();
    assert.equal(st.inflight, null, 'deadline released the in-flight request');
    assert.equal(st.lastPollFailed, true, 'a body-read timeout is a failure');
    assert.ok(st.state.matchups && st.state.matchups.length === 1, 'last-good data preserved after body timeout');
    console.log('ok - §6 timeout stays armed through body/JSON read (not cleared at headers)');
  }

  // ── §2: manual refresh preempts an in-flight poll; OLDER response discarded ─
  {
    let calls = 0;
    const start = { scope: 'league', viewer_roster_id: '1', matchups: [{ roster_id: 'v0' }],
      player_info: {}, games: {}, pbp_by_game: {} };
    const { api, clock } = build({ scope: 'league', state: start,
      scopeCache: { league: start, user: null }, plan: () => {
        calls++;
        if (calls === 1) return { hangBody: true };  // the poll, still reading when preempted
        return { json: { scope: 'league', viewer_roster_id: '1', matchups: [{ roster_id: 'v' + calls }],
          player_info: {}, games: {}, pbp_by_game: {} } };
      } });
    CURRENT_CLOCK = clock;
    const pPoll = api.refresh();
    await flush();
    const seqAfterPoll = api.state().reqSeq;
    const pManual = api.manual();       // cancels the hung poll, starts fresh
    await clock.advance(1);
    await pManual;
    await clock.advance(DEADLINE + 1);
    await pPoll;
    const st = api.state();
    assert.ok(st.reqSeq > seqAfterPoll, 'manual refresh claimed a newer request seq');
    assert.equal(st.state.matchups[0].roster_id, 'v2', 'only the newest (manual) response applied');
    assert.equal(st.inflight, null, 'nothing left in flight');
    assert.equal(st.lastPollFailed, false, 'the discarded older poll did not flag the screen stale');
    console.log('ok - §2 manual refresh preempts poll; older/obsolete response discarded');
  }

  // ── §3: client deadline now exceeds the retired 12s and covers server budget ─
  {
    assert.ok(DEADLINE > 12000, 'fetch deadline must exceed the retired 12s (' + DEADLINE + 'ms)');
    assert.ok(DEADLINE >= 20000, 'and cover the server scoreboard budget');
    // A body that lands at 15s (past the old 12s, under the new deadline) succeeds.
    const { api, clock } = build({ scope: 'league', loadingScope: true, plan: () => ({ hangBodyUntil: 15000,
      json: { scope: 'league', viewer_roster_id: '1', matchups: [{ roster_id: 'slow' }],
        player_info: {}, games: {}, pbp_by_game: {} } }) });
    CURRENT_CLOCK = clock;
    const p = api.refresh();
    await flush();
    await clock.advance(15000);          // body arrives at 15s
    await p;
    const st = api.state();
    assert.equal(st.lastPollFailed, false, '15s latency (was > 12s) now succeeds');
    assert.equal(st.state.matchups[0].roster_id, 'slow', 'the slow-but-in-budget response applied');
    console.log('ok - §3 backend latency past the old 12s deadline still succeeds (' + DEADLINE + 'ms budget)');
  }

  // ── §2: SCOPE SWITCH while a request is active aborts the old request ──────
  {
    const { api, clock } = build({ scope: 'league', loadingScope: true, plan: () => ({ hangBody: true }) });
    CURRENT_CLOCK = clock;
    const p = api.refresh();
    await flush();
    assert.ok(api.state().inflight, 'league request in flight');
    api.switchScope('user');             // bumps gen + aborts the league request
    await clock.advance(1);
    await p;
    const st = api.state();
    assert.equal(st.streamGen, 1, 'scope switch bumped the generation');
    assert.equal(st.state.matchups ? st.state.matchups.length : 0, 0, 'aborted league response never applied under new scope');
    assert.equal(st.inflight, null, 'the old request was cancelled, not left in flight');
    console.log('ok - §2 scope switch aborts the active old request (no cross-scope apply)');
  }

  // ── §5: HTTP 200 ERROR PAYLOAD (portfolio_unavailable) is not fresh data ───
  {
    const good = { scope: 'user', viewer_roster_ids: ['0:1'], matchups: [{ roster_id: 'kept' }],
      player_info: {}, games: {}, pbp_by_game: {} };
    const { api, clock } = build({ scope: 'user', state: good, scopeCache: { league: null, user: good },
      plan: () => ({ json: { scope: 'user', viewer_roster_ids: [], error: 'portfolio_unavailable' } }) });
    CURRENT_CLOCK = clock;
    await api.refresh();
    await flush();
    const st = api.state();
    assert.equal(st.lastPollFailed, true, 'a 200 error payload flags the screen stale');
    assert.equal(st.lastDataAt, null, 'a 200 error payload does NOT advance data freshness');
    assert.equal(st.state.matchups[0].roster_id, 'kept', 'last-good data preserved through the error payload');
    console.log('ok - §5 HTTP 200 error payload preserves last-good, never counted fresh');
  }

  // ── §5/§6: POLLING RESUMES automatically after a failure ───────────────────
  {
    let phase = 0;
    const start = { scope: 'league', viewer_roster_id: '1', matchups: [{ roster_id: 'v0' }],
      player_info: {}, games: {}, pbp_by_game: {} };
    const { api, clock } = build({ scope: 'league', state: start,
      scopeCache: { league: start, user: null },
      plan: () => (phase === 0 ? { network: true }
        : { json: { scope: 'league', viewer_roster_id: '1', matchups: [{ roster_id: 'recovered' }],
            player_info: {}, games: {}, pbp_by_game: {} } }) });
    CURRENT_CLOCK = clock;
    await api.refresh();                  // fails (network)
    await flush();
    let st = api.state();
    assert.equal(st.lastPollFailed, true, 'network failure flagged stale');
    assert.equal(st.inflight, null, 'failed request released (polling not disabled)');
    phase = 1;
    api.setCountdown(1);
    api.tick();                           // countdown 1 -> 0 -> launches the next poll
    await clock.advance(1);
    await flush();
    st = api.state();
    assert.equal(st.lastPollFailed, false, 'polling recovered after the failure');
    assert.equal(st.state.matchups[0].roster_id, 'recovered', 'fresh data applied on recovery');
    console.log('ok - §5/§6 polling resumes automatically after a failure');
  }

  // ── §2: automatic polls DO NOT PILE UP while one request is in flight ──────
  {
    let calls = 0;
    const { api, clock } = build({ scope: 'league', plan: () => { calls++; return { hangBody: true }; } });
    CURRENT_CLOCK = clock;
    const p = api.refresh();
    await flush();
    assert.equal(calls, 1, 'one request opened');
    await api.refresh();                  // automatic: no-op while in flight
    await api.refresh();
    assert.equal(calls, 1, 'automatic polls do not stack onto an in-flight request');
    api.setCountdown(1); api.tick();      // tick must also defer
    assert.equal(calls, 1, 'tick defers while a request is in flight');
    api.switchScope('user');              // release for cleanup
    await clock.advance(DEADLINE + 1);
    await p;
    console.log('ok - §2 automatic polls never pile up (single in-flight request)');
  }

  CURRENT_CLOCK = null;
  console.log('\nALL REDZONE REFRESH HARNESS CHECKS PASSED');
}

run().then(() => { process.exitCode = 0; }).catch((e) => { console.error(e); process.exitCode = 1; });
