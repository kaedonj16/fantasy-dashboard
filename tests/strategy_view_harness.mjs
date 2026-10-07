// Behavioral harness for the Trade Suggestions Strategy path loader.
//
// Incident 2026-09-30 (#2134): the Strategy view "loads the data, then goes
// back to the loading screen and stays there showing the loading skeletons."
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
// Progressive loading (this update): the loader fetches the analytical
// slate (phase=slate, no Monte Carlo), paints it immediately, then resolves
// each player group's sim numbers with its own request
// (/api/trade-intel/archetype-suggestion-sim). Completions merge in place;
// the final server-ranked order and the memory cache apply only once every
// group settles; a failed group stays retryable and blocks caching.
//
// This harness extracts the SHIPPED functions by source and drives them
// with a fake DOM / fake fetch -- not string matching.
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

async function settleUntil(cond, what) {
  for (let i = 0; i < 200; i++) {
    if (cond()) return;
    await tick();
  }
  throw new Error('TIMEOUT waiting for: ' + what);
}

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
const errRes = (status) => ({ ok: false, status, json: async () => ({}) });

// A slate row as the server sends it in phase=slate: sim fields null,
// rank null, sim_pending true.
const slateRow = (pid) => ({
  player_id: pid, group_key: pid, name: 'Player ' + pid, sim_pending: true,
  rank: null, win_prob_delta: null, playoff_odds_delta: null,
  net_win_prob_delta: null, net_playoff_odds_delta: null,
});
const slatePayload = (pids) => ({
  phase: 'slate', groups: pids, current_playoff_pct: null,
  suggestions: pids.map(slateRow),
});
const simPayload = (pid, rank, pod) => ({
  phase: 'sim', group_key: pid, current_playoff_pct: 42.5,
  suggestions: [{
    player_id: pid, group_key: pid, name: 'Player ' + pid, rank,
    win_prob_delta: 0.01, playoff_odds_delta: pod,
    net_win_prob_delta: 0.02, net_playoff_odds_delta: pod,
  }],
});

let passed = 0;
function check(name, cond) {
  if (!cond) throw new Error('FAILED: ' + name);
  passed++;
  console.log('ok - ' + name);
}

const unhandled = [];
process.on('unhandledRejection', (e) => unhandled.push(e));

// ── Build the loader scope ────────────────────────────────────────────────
const fnSrc = [
  'async function loadStrategyView(archetype)',
  'function _strategyGroupState(gk)',
  'async function _strategySimFanout(job)',
  'async function _strategySimLoadGroup(gk, job)',
  'function _retryStrategyGroup(gk)',
  'function _strategySimFinalize(job)',
].map((m) => extractBlock(APP_SRC, m)).join('\n');

function buildLoader() {
  const renders = [];
  const impactRenders = [];
  const cardRenders = [];
  const emptyStates = [];
  const errorStates = [];
  const els = {
    '#otcHasPremium': { value: 'true' },
    '#leagueIdInput': { value: 'L1' },
    '#seasonInput': { value: '2026' },
    '#otcStrategySpinner': makeEl(),
    '#otcStrategyImpactHint': makeEl(),
    '#otcCurrentPOBadge': makeEl(),
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
var _STRATEGY_CACHE_TTL_MS = 30 * 60 * 1000;
var _strategyInflight = {};
var _strategySimGroups = {};
var _strategySimJob = null;
var _STRATEGY_SIM_CONCURRENCY = 3;
var _strategyData = [];
var _strategyFilter = null;
var _strategyPage = 0;
var _currentPlayoffPct = null;
var _activeArchetype = '';
var _renderStrategyResult = function (data, pct, arch) {
  _strategyData = data; _currentPlayoffPct = pct; _strategyFilter = null; _strategyPage = 0;
  renders.push({ data, pct, arch });
  strategyImpact.innerHTML = 'RENDERED:' + arch;
};
var _renderImpactTable = function (data) { impactRenders.push(data); };
var _renderStrategyCards = function (data, filter) { cardRenders.push({ data, filter }); };
`;
  const run = new Function(
    'root', 'window', 'fetch', 'strategyImpact', 'strategyCards', 'strategyCardsHead',
    'getCurrentRosterId', 'getLeagueType', 'getLeagueSize', '_untouchableIds',
    'showPaywall', 'renders', 'impactRenders', 'cardRenders',
    vars + '\n' + fnSrc + '\n' + `
return {
  loadStrategyView,
  retryGroup: _retryStrategyGroup,
  setArch: (a) => { _activeArchetype = a; },
  state: () => ({
    cache: _strategyCache, inflight: _strategyInflight, seq: _strategyReqSeq,
    data: _strategyData, groups: _strategySimGroups, pct: _currentPlayoffPct,
  }),
};`
  );
  const api = run(
    root, windowFake, fetchFake, strategyImpact, strategyCards, strategyCardsHead,
    () => '7', () => 'dynasty', () => 12, new Set(),
    () => {}, renders, impactRenders, cardRenders,
  );
  return { api, renders, impactRenders, cardRenders, emptyStates, errorStates, els, strategyImpact, strategyCards, fetchFake };
}

const simCallsFor = (fetchFake, gk) =>
  fetchFake.calls.filter((c) => c.url.includes('archetype-suggestion-sim') && c.url.includes('group_key=' + gk));
const slateCalls = (fetchFake) =>
  fetchFake.calls.filter((c) => c.url.includes('phase=slate'));

// ── 1. Progressive happy path: slate paints, sims fill in, then finalize ──
{
  const { api, renders, els, fetchFake } = buildLoader();
  api.setArch('contending');
  fetchFake.setHandler((url) => {
    if (url.includes('archetype-suggestion-sim')) {
      return url.includes('group_key=b') ? okRes(simPayload('b', 0.4, 0.02)) : okRes(simPayload('a', 0.9, 0.05));
    }
    return okRes(slatePayload(['a', 'b']));
  });
  await api.loadStrategyView('contending');
  check('slate renders immediately, before any sim completes',
    renders.length === 1 && renders[0].data.length === 2 && renders[0].pct === null);
  check('slate render hides the spinner', els['#otcStrategySpinner'].style.display === 'none');
  await settleUntil(() => Object.keys(api.state().cache).length === 1, 'finalize caches the completed result');
  check('one sim request per player group', simCallsFor(fetchFake, 'a').length === 1 && simCallsFor(fetchFake, 'b').length === 1);
  const st = api.state();
  check('sim numbers merged into the rendered rows',
    st.data.find((r) => r.player_id === 'a').net_playoff_odds_delta === 0.05);
  check('final order follows the server rank once all sims land',
    st.data.map((r) => r.player_id).join(',') === 'a,b');
  check('playoff pct arrives with the sim phase', st.pct === 42.5);
  check('completed result cached with its playoff pct',
    Object.values(st.cache)[0].playoffPct === 42.5);
  check('no in-flight flag left behind', Object.keys(st.inflight).length === 0);

  // ── 2. Cache hit renders instantly (TDZ regression) ─────────────────────
  let threw = null;
  try {
    await api.loadStrategyView('contending');
  } catch (e) { threw = e; }
  check('cache hit does not throw (spinner TDZ regression)', threw === null);
  check('cache hit renders the cached result', renders.length === 2 && renders[1].pct === 42.5);
  check('cache hit does not refetch', fetchFake.calls.length === 3);
  check('cache hit hides the spinner', els['#otcStrategySpinner'].style.display === 'none');
}

// ── 3. Duplicate same-key call while in flight must not strand skeletons ─
{
  const { api, renders, strategyImpact, fetchFake } = buildLoader();
  api.setArch('consolidate');
  let release;
  const gate = new Promise((r) => { release = r; });
  fetchFake.setHandler((url) => {
    if (url.includes('archetype-suggestion-sim')) return okRes(simPayload('a', 0.9, 0.05));
    return gate.then(() => okRes(slatePayload(['a'])));
  });
  const pA = api.loadStrategyView('consolidate'); // paints skeletons, slate pending
  const pB = api.loadStrategyView('consolidate'); // duplicate: aborts A, must own the view
  release();
  await Promise.allSettled([pA, pB]);
  await settleUntil(() => Object.keys(api.state().cache).length === 1, 'duplicate call finalizes');
  check('duplicate same-key call fetches a live replacement slate', slateCalls(fetchFake).length === 2);
  check('duplicate same-key call renders the slate exactly once', renders.length === 1);
  check('duplicate same-key call sims only for the owning request', simCallsFor(fetchFake, 'a').length === 1);
  check('duplicate same-key call does not leave skeletons on screen',
    !strategyImpact.innerHTML.includes('sk-shimmer') && strategyImpact.innerHTML === 'RENDERED:consolidate');
  check('duplicate same-key call leaves no in-flight flag behind',
    Object.keys(api.state().inflight).length === 0);
}

// ── 4. Stale completion must not poison its key for later visits ──────────
{
  const { api, renders, fetchFake } = buildLoader();
  fetchFake.honorAbort = false; // A's slate lands after B superseded it
  let releaseA;
  const gateA = new Promise((r) => { releaseA = r; });
  fetchFake.setHandler((url) => {
    if (url.includes('archetype-suggestion-sim')) {
      return url.includes('group_key=d1') ? okRes(simPayload('d1', 0.7, 0.03)) : okRes(simPayload('c1', 0.7, 0.03));
    }
    return url.includes('archetype=contending')
      ? gateA.then(() => okRes(slatePayload(['c1'])))
      : okRes(slatePayload(['d1']));
  });
  api.setArch('contending');
  const pA = api.loadStrategyView('contending');
  api.setArch('distribute');
  await api.loadStrategyView('distribute');
  await settleUntil(() => api.state().groups['d1'] && api.state().groups['d1'].state === 'done', 'distribute sim settles');
  check('newer archetype renders while the stale one is in flight',
    renders.length === 1 && renders[0].arch === 'distribute');
  releaseA();
  await pA; // resolves stale: returns without rendering or simming
  check('stale response never renders', renders.length === 1);
  check('stale slate never fans out sim requests', simCallsFor(fetchFake, 'c1').length === 0);
  api.setArch('contending');
  await api.loadStrategyView('contending'); // key must still be loadable
  await settleUntil(() => api.state().groups['c1'] && api.state().groups['c1'].state === 'done', 'contending sim settles');
  check('key loadable again after a stale completion (in-flight leak regression)',
    renders.length === 2 && renders[1].arch === 'contending');
  check('no in-flight flags remain', Object.keys(api.state().inflight).length === 0);
}

// ── 5. A 403 renders the paywall state and the key stays retryable ────────
{
  const { api, renders, emptyStates, fetchFake } = buildLoader();
  api.setArch('rebuilding');
  fetchFake.setHandler(() => errRes(403));
  await api.loadStrategyView('rebuilding');
  check('403 renders the PRO empty state', emptyStates.length === 1 && emptyStates[0].title === 'PRO trade tools');
  fetchFake.setHandler((url) => url.includes('archetype-suggestion-sim')
    ? okRes(simPayload('a', 0.9, 0.05))
    : okRes(slatePayload(['a'])));
  await api.loadStrategyView('rebuilding');
  await settleUntil(() => Object.keys(api.state().cache).length === 1, 'post-403 load finalizes');
  check('same key refetches after a 403 (in-flight leak regression)',
    slateCalls(fetchFake).length === 2 && renders.length === 1);
}

// ── 6. Cache hit while another load is in flight replaces the skeletons ──
{
  const { api, renders, strategyImpact, fetchFake } = buildLoader();
  fetchFake.setHandler((url) => url.includes('archetype-suggestion-sim')
    ? okRes(simPayload('a', 0.9, 0.05))
    : okRes(slatePayload(['a'])));
  api.setArch('contending');
  await api.loadStrategyView('contending'); // slate + sim
  await settleUntil(() => Object.keys(api.state().cache).length === 1, 'contending cached');
  let release;
  const gate = new Promise((r) => { release = r; });
  fetchFake.setHandler((url) => url.includes('archetype-suggestion-sim')
    ? okRes(simPayload('a', 0.9, 0.05))
    : gate.then(() => okRes(slatePayload(['a']))));
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
  check('superseded in-flight load never fans out sims', simCallsFor(fetchFake, 'a').length === 1);
}

// ── 7. A failed group sim stays retryable and blocks the final cache ─────
{
  const { api, fetchFake } = buildLoader();
  api.setArch('contending');
  let bAttempts = 0;
  fetchFake.setHandler((url) => {
    if (url.includes('archetype-suggestion-sim')) {
      if (url.includes('group_key=b')) {
        bAttempts++;
        return bAttempts === 1 ? errRes(500) : okRes(simPayload('b', 0.4, 0.02));
      }
      return okRes(simPayload('a', 0.9, 0.05));
    }
    return okRes(slatePayload(['a', 'b']));
  });
  await api.loadStrategyView('contending');
  await settleUntil(() => {
    const g = api.state().groups;
    return g['a'] && g['a'].state === 'done' && g['b'] && g['b'].state === 'error';
  }, 'group b fails while a completes');
  check('failed group is marked error, not silently zeroed', api.state().groups['b'].state === 'error');
  check('partial results are not cached as final', Object.keys(api.state().cache).length === 0);
  check('completed group merged despite the sibling failure',
    api.state().data.find((r) => r.player_id === 'a').net_playoff_odds_delta === 0.05);
  check('failed group keeps its pending slate row (explicit state, no fake numbers)',
    api.state().data.find((r) => r.player_id === 'b').net_playoff_odds_delta === null);
  api.retryGroup('b');
  await settleUntil(() => Object.keys(api.state().cache).length === 1, 'retry completes and finalizes');
  check('retry resolves the group and the completed result caches',
    api.state().groups['b'].state === 'done' && Object.keys(api.state().cache).length === 1);
  check('final order settles after the retry',
    api.state().data.map((r) => r.player_id).join(',') === 'a,b');
}

// ── 8. A group the sim filters out loses its slate rows ───────────────────
{
  const { api, fetchFake } = buildLoader();
  api.setArch('consolidate');
  fetchFake.setHandler((url) => {
    if (url.includes('archetype-suggestion-sim')) {
      return url.includes('group_key=b')
        ? okRes({ phase: 'sim', group_key: 'b', current_playoff_pct: 42.5, suggestions: [] })
        : okRes(simPayload('a', 0.9, 0.05));
    }
    return okRes(slatePayload(['a', 'b']));
  });
  await api.loadStrategyView('consolidate');
  await settleUntil(() => Object.keys(api.state().cache).length === 1, 'resolved-empty group finalizes');
  check('resolved-empty group drops its slate rows',
    api.state().data.map((r) => r.player_id).join(',') === 'a');
  check('result still caches when a group resolves empty', Object.keys(api.state().cache).length === 1);
}

await tick();
check('no unhandled promise rejections across all scenarios', unhandled.length === 0);

console.log(`\n${passed} CHECKS PASSED`);
