// Behavioral harness for the league-players stale-while-revalidate
// refresh notification in static/app.js.
//
// The trade calculator renders from the shared brGetLeaguePlayersData
// pool, which serves a localStorage / in-memory copy immediately and
// revalidates in the background. Before the fix the refreshed data landed
// silently, so the calculator kept showing the stale vintage (old values
// and rank labels) while /api/trade-eval, the Analyst and the player
// modals already showed the current ones. The shipped code now dispatches
// "br:league-players-refreshed" when a background fetch lands with new
// data (ETag changed, or no prior ETag) over a stale copy it served.
//
// Extracts the REAL functions from static/app.js and drives them with a
// fake fetch / fake clock / recording window: cold fetch = no event,
// 304 = no event, new ETag over a served stale copy = exactly one event
// carrying the feed key and the fresh payload.
//
// Run: node tests/lp_refresh_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const APP_SRC = readFileSync(join(here, '..', 'static', 'app.js'), 'utf8');

function extractBlock(src, marker) {
  const at = src.indexOf(marker);
  if (at < 0) throw new Error(`marker not found: ${marker}`);
  let depth = 0;
  for (let j = src.indexOf('{', at); j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}' && --depth === 0) return src.slice(at, j + 1);
  }
  throw new Error(`unbalanced braces extracting ${marker}`);
}

const microtask = () => new Promise((r) => setImmediate(r));

function okJson(body, etag) {
  return {
    ok: true,
    status: 200,
    headers: { get: (n) => (n === 'ETag' ? (etag === undefined ? '"e1"' : etag) : null) },
    json: async () => body,
  };
}
function notModified() {
  return {
    ok: false,
    status: 304,
    headers: { get: () => null },
    json: async () => { throw new Error('304 has no body'); },
  };
}

let passed = 0;
function check(name, cond) {
  if (!cond) throw new Error('FAILED: ' + name);
  passed++;
  console.log('ok - ' + name);
}

const VARS = `
var __brLeaguePlayersPromise = null;
var __brLeaguePlayersAt = 0;
var __brLeaguePlayersKey = "";
var __brLeaguePlayersEtag = "";
var __brLP_TTL_MS = 30 * 60 * 1000;
var __brLP_LS_PREFIX = "br-lp-trade-v1:";
`;

const FNS = [
  extractBlock(APP_SRC, 'function __brLPReadLS(key)'),
  extractBlock(APP_SRC, 'function __brLPWriteLS(key'),
  extractBlock(APP_SRC, 'function __brLPNotifyRefreshed(key'),
  extractBlock(APP_SRC, 'function brGetLeaguePlayersData(forceKey)'),
].join('\n');

function makeWorld() {
  let now = 1_000_000;
  const store = {};
  const localStorage = {
    getItem: (k) => (Object.prototype.hasOwnProperty.call(store, k) ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
  };
  const events = [];
  class FakeCustomEvent {
    constructor(type, init) { this.type = type; this.detail = (init || {}).detail; }
  }
  const fakeWindow = {
    dispatchEvent: (ev) => { events.push(ev); return true; },
  };
  const calls = [];
  let handler = null;
  async function fakeFetch(url, opts) {
    calls.push({ url, opts });
    await microtask();
    if (!handler) throw new Error('no fetch handler set');
    return handler(url, opts);
  }
  const run = new Function(
    'Date', 'fetch', 'localStorage', 'window', 'CustomEvent',
    VARS + '\n' + FNS + '\nreturn brGetLeaguePlayersData;'
  );
  const brGet = run(
    { now: () => now }, fakeFetch, localStorage, fakeWindow, FakeCustomEvent
  );
  return {
    brGet, events, calls,
    setHandler: (h) => { handler = h; },
    advance: (ms) => { now += ms; },
  };
}

const w = makeWorld();

// 1. Cold fetch: the caller awaits the network, so no refresh event.
w.setHandler(() => okJson({ players: ['v1'] }, '"e1"'));
const r1 = await w.brGet();
check('cold fetch returns fresh data', r1.players[0] === 'v1');
await microtask(); await microtask();
check('cold fetch dispatches no refresh event', w.events.length === 0);

// 2. Within the TTL the completed result is reused: no fetch, no event.
const r2 = await w.brGet();
check('within TTL reuses result without refetching', r2 === r1 && w.calls.length === 1);
check('within TTL dispatches no refresh event', w.events.length === 0);

// 3. Past the TTL with a new ETag: stale is served immediately, and the
// background landing dispatches exactly one refresh event with the
// default feed key and the fresh payload.
w.advance(31 * 60 * 1000);
w.setHandler(() => okJson({ players: ['v2'] }, '"e2"'));
const r3 = await w.brGet();
check('past TTL serves the stale copy immediately', r3 === r1);
await microtask(); await microtask(); await microtask();
check('new ETag over a stale copy dispatches one refresh event',
  w.events.length === 1 && w.events[0].type === 'br:league-players-refreshed');
check('refresh event carries the feed key and fresh payload',
  w.events[0].detail.key === '' && w.events[0].detail.data.players[0] === 'v2');
const r4 = await w.brGet();
check('refreshed data is served after the background landing',
  r4.players[0] === 'v2' && w.events.length === 1);

// 4. A 304 revalidation dispatches nothing.
w.advance(31 * 60 * 1000);
w.setHandler(() => notModified());
const r5 = await w.brGet();
await microtask(); await microtask(); await microtask();
check('304 keeps serving the current copy with no event',
  r5 === r4 && w.events.length === 1);

// 5. A keyed feed behaves the same, and its event carries that key (so a
// calculator showing a different feed ignores it).
const KEY = 'league_type=sf&league_size=12';
w.setHandler(() => okJson({ players: ['sf1'] }, '"s1"'));
const k1 = await w.brGet(KEY);
await microtask(); await microtask();
check('keyed cold fetch returns its data with no event',
  k1.players[0] === 'sf1' && w.events.length === 1);
w.advance(31 * 60 * 1000);
w.setHandler(() => okJson({ players: ['sf2'] }, '"s2"'));
const k2 = await w.brGet(KEY);
check('keyed stale copy served immediately', k2 === k1);
await microtask(); await microtask(); await microtask();
check('keyed refresh event carries the keyed feed key',
  w.events.length === 2 && w.events[1].detail.key === KEY &&
  w.events[1].detail.data.players[0] === 'sf2');

console.log(`\nALL LP REFRESH HARNESS CHECKS PASSED (${passed})`);
