// Pull-to-refresh Tier 1 hook harness (Node).
//
// Extracts the REAL hook registration blocks from their sources and drives
// each with a fake DOM, asserting the shared v1 contract: the hook calls
// the page's existing load/refresh entry point exactly once, the returned
// promise settles when that load settles, and a detached page root makes
// the hook delete its own registration and resolve without loading.
//
// Sources:
//   Portfolio  static/app.js            (owner.ptrRefresh wrapper in bind())
//   Waivers    dashboard_services/pages/waivers_page.py   (embedded JS)
//   Standings  app.py                   (_standings_week_selector script)
//   Rankings   static/rankings.js       (tail registration)
//
// Python-embedded JS is written with doubled braces (f-string); the
// extractor un-doubles them inside the extracted hook block only.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (p) => readFileSync(join(ROOT, p), 'utf-8');

let failures = 0;
function check(name, cond) {
  if (cond) { console.log('ok   -', name); }
  else { failures++; console.log('FAIL -', name); }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function extract(src, startMarker, endMarker) {
  const a = src.indexOf(startMarker);
  if (a < 0) throw new Error('start marker not found: ' + startMarker);
  const b = src.indexOf(endMarker, a);
  if (b < 0) throw new Error('end marker not found: ' + endMarker);
  return src.slice(a, b + endMarker.length);
}
const undouble = (s) => s.replaceAll('{{', '{').replaceAll('}}', '}');

// ── Rankings (Players page board): hook returns prLoadData() directly ──────
{
  const src = read('static/rankings.js');
  const block = extract(src, 'window.brPageRefresh = function () {', '};');
  const window = {};
  let calls = 0;
  const listEl = { isConnected: true };
  const document = { getElementById: (id) => (id === 'prList' ? listEl : null) };
  const prLoadData = () => { calls++; return new Promise((r) => setTimeout(r, 30)); };
  new Function('window', 'document', 'prLoadData', block)(window, document, prLoadData);
  check('rankings: hook registered', typeof window.brPageRefresh === 'function');
  let settled = false;
  const p = window.brPageRefresh();
  check('rankings: returns a promise', !!p && typeof p.then === 'function');
  p.then(() => { settled = true; });
  check('rankings: prLoadData called exactly once', calls === 1);
  await sleep(80);
  check('rankings: promise settles with the load', settled === true);
  listEl.isConnected = false;
  await window.brPageRefresh();
  check('rankings: detached list drops the registration', !('brPageRefresh' in window));
  check('rankings: detached list does not reload', calls === 1);
}

// ── Waivers: hook re-runs wvLoad(); settle polled off the skeleton ─────────
{
  const src = read('dashboard_services/pages/waivers_page.py');
  const block = undouble(extract(
    src,
    'window.brPageRefresh = function() {{',
    '}};',
  ));
  const window = { scrollY: 0, pageYOffset: 0, scrollTo: () => {} };
  let wvLoadCalls = 0;
  let skeletonUntil = 0;
  const listEl = {
    isConnected: true,
    querySelector: (sel) => (sel === '.skeleton' && Date.now() < skeletonUntil ? {} : null),
  };
  const document = {
    documentElement: { scrollTop: 0 },
    body: { scrollTop: 0 },
    getElementById: (id) => (id === 'wvWaiverList' ? listEl : null),
  };
  const wvLoad = () => {
    wvLoadCalls++;
    globalThis.wvCandidateRequestSeq++;
    skeletonUntil = Date.now() + 350;
  };
  // The hook reads the bare page global wvCandidateRequestSeq (its settle
  // poll re-reads it live); leave it out of the Function params so the
  // bare name resolves to globalThis, exactly as in the page.
  globalThis.wvCandidateRequestSeq = 0;
  const hook = new Function(
    'window', 'document', 'wvLoad',
    block + '\nreturn window.brPageRefresh;',
  )(window, document, wvLoad);
  check('waivers: hook registered', typeof hook === 'function');
  let settledAt = 0;
  const t0 = Date.now();
  hook().then(() => { settledAt = Date.now(); });
  check('waivers: wvLoad called exactly once', wvLoadCalls === 1);
  await sleep(200);
  check('waivers: still pending while the skeleton is up', settledAt === 0);
  await sleep(600);
  check('waivers: settles after the skeleton clears', settledAt > 0 && settledAt - t0 >= 300);
  listEl.isConnected = false;
  await hook();
  check('waivers: detached list drops the registration', !('brPageRefresh' in window));
  check('waivers: detached list does not reload', wvLoadCalls === 1);
}

// ── Standings: hook re-dispatches the week change; settle is the class ─────
{
  const src = read('app.py');
  const block = undouble(extract(
    src,
    'window.brPageRefresh = function() {{',
    '}};',
  ));
  const window = { scrollY: 0, pageYOffset: 0, scrollTo: () => {} };
  let dispatches = 0;
  let loadingUntil = 0;
  const sel = {
    isConnected: true,
    dispatchEvent: () => { dispatches++; loadingUntil = Date.now() + 350; return true; },
  };
  const document = {
    documentElement: { scrollTop: 0 },
    body: { scrollTop: 0 },
    querySelector: (selq) => (selq === '.st-week-loading' && Date.now() < loadingUntil ? {} : null),
  };
  class Event { constructor(type) { this.type = type; } }
  const hook = new Function(
    'window', 'document', 'sel', 'Event',
    block + '\nreturn window.brPageRefresh;',
  )(window, document, sel, Event);
  check('standings: hook registered', typeof hook === 'function');
  let settledAt = 0;
  const t0 = Date.now();
  hook().then(() => { settledAt = Date.now(); });
  check('standings: change dispatched exactly once', dispatches === 1);
  await sleep(200);
  check('standings: still pending while panels carry the loading class', settledAt === 0);
  await sleep(600);
  check('standings: settles after the loading class clears', settledAt > 0 && settledAt - t0 >= 300);
  sel.isConnected = false;
  await hook();
  check('standings: detached selector drops the registration', !('brPageRefresh' in window));
  check('standings: detached selector does not dispatch', dispatches === 1);
}

// ── Portfolio: wrapper in bind() delegates to refresh(), alive-gated ───────
{
  const src = read('static/app.js');
  const block = extract(
    src,
    'owner.ptrRefresh = function () {',
    'window.brPageRefresh = owner.ptrRefresh;',
  );
  const window = {};
  let refreshCalls = 0;
  let aliveFlag = true;
  const owner = { alive: () => aliveFlag, ptrRefresh: null, refresh: null };
  const refresh = () => { refreshCalls++; return new Promise((r) => setTimeout(r, 30)); };
  new Function('window', 'owner', 'refresh', block)(window, owner, refresh);
  check('portfolio: hook registered', typeof window.brPageRefresh === 'function');
  let settled = false;
  window.brPageRefresh().then(() => { settled = true; });
  check('portfolio: refresh() called exactly once', refreshCalls === 1);
  await sleep(80);
  check('portfolio: promise settles with refresh()', settled === true);
  aliveFlag = false;
  await window.brPageRefresh();
  check('portfolio: dead owner drops the registration', !('brPageRefresh' in window));
  check('portfolio: dead owner does not refresh', refreshCalls === 1);
}

if (failures) {
  console.log(failures + ' tier-1 harness check(s) failed');
  process.exit(1);
}
console.log('ALL PULL-TO-REFRESH TIER-1 HARNESS CHECKS PASSED');
