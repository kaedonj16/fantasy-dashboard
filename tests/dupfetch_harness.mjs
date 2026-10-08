// Behavioral harness for the duplicate-data-fetch eliminations.
//
// Exercises the SHIPPED client-side request-sharing logic (extracted by source
// from static/app.js and static/player_modal.js, driven by a fake fetch and a
// fake clock -- not string matching):
//
//   * brGetLeaguePlayersData (app.js) -- the nav player-search idle-preload and
//     the trade calculator share one /api/league-players request: in-flight
//     dedup, 60s result reuse, and rejection clears for retry.
//   * _pmSmallFetch (player_modal.js) -- the per-player news/ADP cache:
//     in-flight dedup, 5min TTL reuse, expiry refetch, failures not cached,
//     and the LRU bound.
//
// Run: node tests/dupfetch_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const APP_SRC = readFileSync(join(here, '..', 'static', 'app.js'), 'utf8');
const MODAL_SRC = readFileSync(join(here, '..', 'static', 'player_modal.js'), 'utf8');

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

// ── Fake infrastructure ─────────────────────────────────────────────────────
const microtask = () => new Promise((r) => setImmediate(r));

function makeClock() {
  let now = 1_000_000;
  return { now: () => now, advance: (ms) => { now += ms; } };
}

function makeFetch(clock) {
  const calls = [];
  let handler = null; // (url, opts) => response-ish | throws
  async function fakeFetch(url, opts) {
    calls.push({ url, opts, at: clock.now() });
    await microtask();
    if (!handler) throw new Error('no fetch handler set');
    const res = handler(url, opts);
    if (res && res.then) return res;
    return res;
  }
  fakeFetch.calls = calls;
  fakeFetch.setHandler = (h) => { handler = h; };
  return fakeFetch;
}

function okJson(body) {
  return { ok: true, json: async () => body };
}

let passed = 0;
function check(name, cond) {
  if (!cond) throw new Error('FAILED: ' + name);
  passed++;
  console.log('ok - ' + name);
}

// ═══════════════════════════════════════════════════════════════════════════
// brGetLeaguePlayersData (from static/app.js)
// ═══════════════════════════════════════════════════════════════════════════
{
  const vars = `
var __brLeaguePlayersPromise = null;
var __brLeaguePlayersAt = 0;
var __brLeaguePlayersKey = "";
`;
  const fn = extractBlock(APP_SRC, 'function brGetLeaguePlayersData(forceKey)');
  const clock = makeClock();
  const fakeFetch = makeFetch(clock);
  const scope = { Date: { now: clock.now }, fetch: fakeFetch };
  const run = new Function('Date', 'fetch', vars + '\n' + fn + '\nreturn brGetLeaguePlayersData;');
  const brGetLeaguePlayersData = run.call(scope, scope.Date, scope.fetch);

  // 1. Two concurrent callers share one in-flight request.
  fakeFetch.setHandler(() => okJson({ players: [1, 2, 3] }));
  const p1 = brGetLeaguePlayersData();
  const p2 = brGetLeaguePlayersData();
  const [r1, r2] = await Promise.all([p1, p2]);
  check('league-players: concurrent callers share one in-flight fetch',
    fakeFetch.calls.length === 1 && r1.players.length === 3 && r2 === r1);

  // 2. A later call within 60s reuses the completed result.
  const r3 = await brGetLeaguePlayersData();
  check('league-players: sequential call within 60s reuses result (no refetch)',
    fakeFetch.calls.length === 1 && r3 === r1);

  // 3. After 60s the cache expires and the next call refetches.
  clock.advance(61_000);
  const r4 = await brGetLeaguePlayersData();
  check('league-players: refetches after the 60s window expires',
    fakeFetch.calls.length === 2 && r4 !== r1 && r4.players.length === 3);

  // 4. A rejection clears the shared promise so the next caller retries.
  clock.advance(61_000);
  let attempts = 0;
  fakeFetch.setHandler(() => { attempts++; throw new Error('boom'); });
  await assert.rejects(brGetLeaguePlayersData(), /boom/);
  fakeFetch.setHandler(() => okJson({ players: [] }));
  const r5 = await brGetLeaguePlayersData();
  check('league-players: rejection clears shared state, next caller retries',
    attempts === 1 && r5.players.length === 0 && fakeFetch.calls.length === 4);
}

// ═══════════════════════════════════════════════════════════════════════════
// _pmSmallFetch (from static/player_modal.js) -- news/ADP per-player cache
// ═══════════════════════════════════════════════════════════════════════════
{
  const vars = `
var _pmSmallCache = new Map();
var _PM_SMALL_TTL = 5 * 60 * 1000, _PM_SMALL_MAX = 40;
`;
  const fn = extractBlock(MODAL_SRC, 'function _pmSmallFetch(key, url)');
  const clock = makeClock();
  const fakeFetch = makeFetch(clock);
  const scope = { Date: { now: clock.now }, fetch: fakeFetch };
  const run = new Function(
    'Date', 'fetch', 'Map',
    vars + '\n' + fn + '\nreturn { _pmSmallFetch, _setMax: (n) => { _PM_SMALL_MAX = n; } };'
  );
  const { _pmSmallFetch, _setMax } = run.call(scope, scope.Date, scope.fetch, Map);

  fakeFetch.setHandler((url) => okJson({ url, n: fakeFetch.calls.length }));

  // 1. Two concurrent callers share one in-flight request.
  const a1 = _pmSmallFetch('news:p1', '/api/player-news/p1');
  const a2 = _pmSmallFetch('news:p1', '/api/player-news/p1');
  const [d1, d2] = await Promise.all([a1, a2]);
  check('news/adp cache: concurrent callers share one in-flight fetch',
    fakeFetch.calls.length === 1 && d1 === d2 && d1.url === '/api/player-news/p1');

  // 2. A reopen within the 5min TTL reuses the payload (no refetch).
  clock.advance(60_000);
  const d3 = await _pmSmallFetch('news:p1', '/api/player-news/p1');
  check('news/adp cache: reopen within TTL reuses cached payload',
    fakeFetch.calls.length === 1 && d3 === d1);

  // 3. Distinct keys (player, or same player with a different season) fetch separately.
  const d4 = await _pmSmallFetch('adp:p1:2026', '/api/player-adp/p1?season=2026');
  const d5 = await _pmSmallFetch('adp:p1:2025', '/api/player-adp/p1?season=2025');
  check('news/adp cache: season is part of the key (no cross-season reuse)',
    fakeFetch.calls.length === 3 && d4.url !== d5.url);

  // 4. After TTL expiry the next open refetches.
  clock.advance(5 * 60_000 + 1);
  const d6 = await _pmSmallFetch('news:p1', '/api/player-news/p1');
  check('news/adp cache: refetches after TTL expiry',
    fakeFetch.calls.length === 4 && d6 !== d1);

  // 5. A failed fetch is not cached: the next caller retries.
  fakeFetch.setHandler(() => { throw new Error('net down'); });
  const d7 = await _pmSmallFetch('news:p2', '/api/player-news/p2');
  fakeFetch.setHandler((url) => okJson({ url }));
  const d8 = await _pmSmallFetch('news:p2', '/api/player-news/p2');
  check('news/adp cache: failure is not cached, next caller retries',
    d7 === null && d8 !== null && d8.url === '/api/player-news/p2');

  // 6. The cache is bounded (LRU eviction).
  _setMax(3);
  for (let i = 0; i < 5; i++) {
    await _pmSmallFetch('news:px' + i, '/api/player-news/px' + i);
  }
  const before = fakeFetch.calls.length;
  await _pmSmallFetch('news:px0', '/api/player-news/px0'); // evicted -> refetch
  check('news/adp cache: oldest entry evicted past the bound',
    fakeFetch.calls.length === before + 1);
}

console.log(`\nCHECKS PASSED (${passed})`);
