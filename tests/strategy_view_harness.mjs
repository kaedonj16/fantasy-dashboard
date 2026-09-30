// Behavioral harness for the Trade Suggestions Strategy path loader.
//
// Incident 2026-09-30: the Strategy view "loads the data, then goes back to
// the loading screen and stays there showing the loading skeletons."
// Root causes in the shipped loadStrategyView (static/app.js):
//   1. The cache-hit branch referenced `strategySpinner` BEFORE its const
//      declaration (temporal dead zone) -> every cache hit threw a
//      ReferenceError after aborting the live request, stranding skeletons.
//   2. The in-flight dedupe ran AFTER the call had already bumped the seq
//      token and aborted the previous controller, so a duplicate call for
//      the same key aborted the only live request and then deferred to it.
//   3. `_strategyInflight[key] = true` leaked on the stale and 403 early
//      returns, so that key early-returned forever afterwards.
//
// This harness extracts the SHIPPED function by source and drives it with a
// fake DOM / fake fetch -- not string matching.
//
// Run: node tests/strategy_view_harness.mjs   (exit 0 = pass)
// APP_JS_PATH env var overrides the app.js under test.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const APP_PATH = process.env.APP_JS_PATH || join(here, '..', 'static', 'app.js');
const APP_SRC = readFileSync(APP_PATH, 'utf8');

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

const tick = () => new Promise((r) => setImmediate(r));

function abortError() {
  const e = new Error('The operation was aborted');
  e.name = 'AbortError';
  return e;
}

function makeEl() {
  return { innerHTML: '', textContent: '', style: {}, dataset: {}, value: '' };
}

// Fake fetch. handler(url, opts) returns a response or a promise of one.
// Like real fetch, a pending request rejects with AbortError when its
// signal aborts -- unless honorAbort is set false (emulates a response
// that was already in hand when the abort landed).
function makeFetch() {
  const calls = [];
  let handler = null;
  const fake = async (url, opts = {}) => {
    calls.push({ url, opts });
    const signal = opts.signal;
    if (signal && signal.aborted) throw abortError();
    if (!handler) throw new Error('no fetch handler set');
    const pending = Promise.resolve().then(() => handler(url, opts));
    if (signal && fake.honorAbort !== false) {
      const aborted = new Promise((_, reject) => {
        signal.addEventListener('abort', () => reject(abortError()), { once: true });
      });
      return Promise.race([pending, aborted]);
    }
    return pending;
  };
  fake.calls = calls;
  fake.setHandler = (h) => { handler = h; };
  fake.honorAbort = true;
  return fake;
}

const okRes = (body) => ({ ok: true, status: 200, json: async () => body });
const PAYLOAD = { suggestions: [{ player_id: 'p1' }], current_playoff_pct: 42.5 };

let passed = 0;
function check(name, cond) {
  if (!cond) throw new Error('FAILED: ' + name);
  passed++;
  console.log('ok - ' + name);
}

const unhandled = [];
process.on('unhandledRejection', (e) => unhandled.push(e));

// ── Build the loader scope ────────────────────────────────────────────────
const fnSrc = extractBlock(APP_SRC, 'async function loadStrategyView(archetype)');

function buildLoader() {
  const renders = [];
  const emptyStates = [];
  const errorStates = [];
  const els = {
    '#otcHasPremium': { value: 'true' },
    '#leagueIdInput': { value: 'L1' },
    '#seasonInput': { value: '2026' },
    '#otcStrategySpinner': makeEl(),
    '#otcStrategyImpactHint': makeEl(),
  };
  const strategyImpact = makeEl();
  const strategyCards = makeEl();
  const strategyCardsHead = makeEl();
  const root = {
    querySelector: (sel) => els[sel] || null,
    querySelectorAll: () => [],
  };
  const windowFake = {
    location: { pathname: '/sleeper/2026/L1/trade' },
    brEmptyState: (el, opts) => { emptyStates.push(opts); el.innerHTML = 'EMPTY:' + opts.title; },
    brErrorState: (el, msg) => { errorStates.push(msg); el.innerHTML = 'ERROR:' + msg; },
  };
  const fetchFake = makeFetch();
  const vars = `
var _strategyReqSeq = 0;
var _strategyAbortCtrl = null;
var _strategyCache = {};
var _strategyInflight = {};
var _activeArchetype = '';
`;
  const run = new Function(
    'root', 'window', 'fetch', 'strategyImpact', 'strategyCards', 'strategyCardsHead',
    'getCurrentRosterId', 'getLeagueType', 'getLeagueSize', '_untouchableIds',
    '_renderStrategyResult', 'showPaywall',
    vars + '\n' + fnSrc + '\n' + `
return {
  loadStrategyView,
  setArch: (a) => { _activeArchetype = a; },
  state: () => ({ cache: _strategyCache, inflight: _strategyInflight, seq: _strategyReqSeq }),
};`
  );
  const api = run(
    root, windowFake, fetchFake, strategyImpact, strategyCards, strategyCardsHead,
    () => '7', () => 'dynasty', () => 12, new Set(),
    (data, pct, arch) => {
      renders.push({ data, pct, arch });
      // The real render replaces the skeleton markup wholesale.
      strategyImpact.innerHTML = 'RENDERED:' + arch;
    },
    () => {},
  );
  return { api, renders, emptyStates, errorStates, els, strategyImpact, strategyCards, fetchFake };
}

// ── 1. First load fetches, renders, caches ────────────────────────────────
{
  const { api, renders, els, fetchFake } = buildLoader();
  api.setArch('contending');
  fetchFake.setHandler(() => okRes(PAYLOAD));
  await api.loadStrategyView('contending');
  check('first load renders the fetched suggestions',
    renders.length === 1 && renders[0].data.length === 1 && renders[0].pct === 42.5);
  check('first load hides the spinner', els['#otcStrategySpinner'].style.display === 'none');
  check('first load caches the result', Object.keys(api.state().cache).length === 1);
  check('first load leaves no in-flight flag behind',
    Object.keys(api.state().inflight).length === 0);

  // ── 2. Cache hit renders instantly (TDZ regression) ─────────────────────
  let threw = null;
  try {
    await api.loadStrategyView('contending');
  } catch (e) { threw = e; }
  check('cache hit does not throw (spinner TDZ regression)', threw === null);
  check('cache hit renders the cached result', renders.length === 2 && renders[1].arch === 'contending');
  check('cache hit does not refetch', fetchFake.calls.length === 1);
  check('cache hit hides the spinner', els['#otcStrategySpinner'].style.display === 'none');
}

// ── 3. Duplicate same-key call while in flight must not strand skeletons ─
{
  const { api, renders, strategyImpact, fetchFake } = buildLoader();
  api.setArch('consolidate');
  let release;
  const gate = new Promise((r) => { release = r; });
  fetchFake.setHandler(() => gate.then(() => okRes(PAYLOAD)));
  const pA = api.loadStrategyView('consolidate'); // paints skeletons, fetch pending
  const pB = api.loadStrategyView('consolidate'); // duplicate: aborts A, must own the view
  release();
  await Promise.allSettled([pA, pB]);
  await tick();
  check('duplicate same-key call fetches a live replacement', fetchFake.calls.length === 2);
  check('duplicate same-key call renders exactly once', renders.length === 1);
  check('duplicate same-key call does not leave skeletons on screen',
    !strategyImpact.innerHTML.includes('sk-shimmer') && strategyImpact.innerHTML === 'RENDERED:consolidate');
  check('duplicate same-key call leaves no in-flight flag behind',
    Object.keys(api.state().inflight).length === 0);
}

// ── 4. Stale completion must not poison its key for later visits ──────────
{
  const { api, renders, fetchFake } = buildLoader();
  fetchFake.honorAbort = false; // A's response lands after B superseded it
  let releaseA;
  const gateA = new Promise((r) => { releaseA = r; });
  fetchFake.setHandler((url) => url.includes('archetype=contending')
    ? gateA.then(() => okRes(PAYLOAD))
    : okRes({ suggestions: [{ player_id: 'p2' }], current_playoff_pct: 10 }));
  api.setArch('contending');
  const pA = api.loadStrategyView('contending');
  api.setArch('distribute');
  await api.loadStrategyView('distribute');
  check('newer archetype renders while the stale one is in flight',
    renders.length === 1 && renders[0].arch === 'distribute');
  releaseA();
  await pA; // resolves stale: returns without rendering
  check('stale response never renders', renders.length === 1);
  api.setArch('contending');
  await api.loadStrategyView('contending'); // key must still be loadable
  check('key loadable again after a stale completion (in-flight leak regression)',
    renders.length === 2 && renders[1].arch === 'contending');
  check('no in-flight flags remain', Object.keys(api.state().inflight).length === 0);
}

// ── 5. A 403 renders the paywall state and the key stays retryable ────────
{
  const { api, renders, emptyStates, fetchFake } = buildLoader();
  api.setArch('rebuilding');
  fetchFake.setHandler(() => ({ ok: false, status: 403, json: async () => ({}) }));
  await api.loadStrategyView('rebuilding');
  check('403 renders the PRO empty state', emptyStates.length === 1 && emptyStates[0].title === 'PRO trade tools');
  fetchFake.setHandler(() => okRes(PAYLOAD));
  await api.loadStrategyView('rebuilding');
  check('same key refetches after a 403 (in-flight leak regression)',
    fetchFake.calls.length === 2 && renders.length === 1);
}

// ── 6. Cache hit while another load is in flight replaces the skeletons ──
{
  const { api, renders, strategyImpact, fetchFake } = buildLoader();
  fetchFake.setHandler(() => okRes(PAYLOAD));
  api.setArch('contending');
  await api.loadStrategyView('contending'); // now cached
  let release;
  const gate = new Promise((r) => { release = r; });
  fetchFake.setHandler(() => gate.then(() => okRes(PAYLOAD)));
  api.setArch('distribute');
  const pSlow = api.loadStrategyView('distribute'); // skeletons on screen
  api.setArch('contending');
  await api.loadStrategyView('contending'); // cache hit: aborts slow load, renders
  release();
  await pSlow;
  await tick();
  check('cache hit during an in-flight load renders the cached view',
    renders.length === 2 && renders[1].arch === 'contending');
  check('cache hit during an in-flight load clears the skeletons',
    strategyImpact.innerHTML === 'RENDERED:contending');
}

await tick();
check('no unhandled promise rejections across all scenarios', unhandled.length === 0);

console.log(`\n${passed} CHECKS PASSED`);
