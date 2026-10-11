// Behavioral harness for the rivalry-line flicker fix.
//
// Exercises the SHIPPED initRivalryLines logic (extracted by source from
// static/app.js, driven by fake DOM nodes and a fake fetch):
//
//   * first hydrate fetches /api/rivalry and renders the label
//   * after an in-place content swap (soft-nav / auto-revalidate / manual
//     refresh) the replacement node re-renders SYNCHRONOUSLY from the
//     label cache with zero additional fetches (no blink-out window)
//   * concurrent hydrates for the same pair share one in-flight fetch
//   * failed fetches are NOT cached, so the next swap retries
//   * empty labels (no history) are cached, so swaps stay hidden instantly
//
// Run: node tests/rivalry_cache_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const APP_SRC = readFileSync(join(here, '..', 'static', 'app.js'), 'utf8');

function extractIIFE(src, marker) {
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
const flush = async (n = 10) => { for (let i = 0; i < n; i++) await microtask(); };

function makeNode(a, b) {
  return {
    dataset: { rivA: a, rivB: b, rivLuser: 'Lurkzx901', rivRuser: 'CalebsCouch' },
    innerHTML: '',
    hidden: true,
  };
}

// Fake root: querySelectorAll implements the ':not([data-riv-done])' filter
// the shipped selector relies on.
function makeRoot(nodes) {
  return {
    querySelectorAll(sel) {
      assert.match(sel, /\.m-rivalry/);
      return nodes.filter((n) => n.dataset.rivA && !n.dataset.rivDone);
    },
  };
}

function boot(nodes, fetchImpl) {
  const calls = [];
  const fakeFetch = async (url) => {
    calls.push(url);
    await microtask();
    return fetchImpl(url);
  };
  const document = {
    readyState: 'complete',
    addEventListener() { throw new Error('should not bind DOMContentLoaded'); },
    querySelectorAll: () => [],
    getElementById: () => null,
  };
  const window = {
    __brctx: { platform: 'sleeper', season: 2026, leagueId: '123' },
  };
  const iifeSrc = extractIIFE(APP_SRC, '(function initRivalryLines()');
  const bootFn = new Function('window', 'document', 'fetch', 'setTimeout', 'MutationObserver', `${iifeSrc})();`);
  bootFn(window, document, fakeFetch, (fn) => 0 /* never fire the 2s retry in tests */, undefined);
  return { window, calls };
}

const RIVALRY_DATA = {
  wins_a: 1, wins_b: 0, ties: 0,
  games: [{ season: 2025, week: 3, a_pts: 120.5, b_pts: 100.0 }],
};
const okJson = (data) => ({ ok: true, json: async () => data });

// ── 1. first hydrate fetches and renders ─────────────────────────────────────
{
  const nodes = [makeNode('u1', 'u2')];
  const { window, calls } = boot(nodes, () => okJson(RIVALRY_DATA));
  window.brScanRivalryLines(makeRoot(nodes));
  await flush();
  assert.equal(calls.length, 1, 'one /api/rivalry fetch');
  assert.match(calls[0], /\/api\/rivalry\/sleeper\/2026\/123\?a=u1&b=u2/);
  assert.equal(nodes[0].hidden, false);
  assert.match(nodes[0].innerHTML, /All-time H2H:.*Lurkzx901.*1–0/);
  console.log('ok 1: first hydrate fetches and renders');
}

// ── 2. swap: replacement node renders synchronously from cache, no fetch ────
{
  const nodes = [makeNode('u1', 'u2')];
  const { window, calls } = boot(nodes, () => okJson(RIVALRY_DATA));
  window.brScanRivalryLines(makeRoot(nodes));
  await flush();
  assert.equal(calls.length, 1);
  // Simulate the in-place content swap: fresh server HTML, hidden + empty.
  const swapped = [makeNode('u1', 'u2')];
  window.brScanRivalryLines(makeRoot(swapped));
  // Synchronous: no microtask flush needed; fetch count unchanged.
  assert.equal(calls.length, 1, 'no second fetch after swap');
  assert.equal(swapped[0].hidden, false, 'line visible immediately, no blink');
  assert.match(swapped[0].innerHTML, /All-time H2H:.*Lurkzx901.*1–0/);
  console.log('ok 2: post-swap node re-renders from cache with zero fetches');
}

// ── 3. concurrent hydrates share one in-flight fetch ─────────────────────────
{
  const nodes = [makeNode('u1', 'u2'), makeNode('u1', 'u2')];
  let resolvers = [];
  const { window, calls } = boot(nodes, () => new Promise((res) => resolvers.push(res)));
  window.brScanRivalryLines(makeRoot(nodes));
  await flush(3);
  assert.equal(calls.length, 1, 'single in-flight fetch for the pair');
  resolvers.forEach((r) => r(okJson(RIVALRY_DATA)));
  await flush();
  assert.equal(nodes[0].hidden, false);
  assert.equal(nodes[1].hidden, false);
  console.log('ok 3: concurrent hydrates dedupe to one fetch');
}

// ── 4. failed fetch is not cached: next swap retries ─────────────────────────
{
  const nodes = [makeNode('u1', 'u2')];
  let fail = true;
  const { window, calls } = boot(nodes, () => (fail ? { ok: false } : okJson(RIVALRY_DATA)));
  window.brScanRivalryLines(makeRoot(nodes));
  await flush();
  assert.equal(calls.length, 1);
  assert.equal(nodes[0].hidden, true, 'stays hidden on failure');
  fail = false;
  const swapped = [makeNode('u1', 'u2')];
  window.brScanRivalryLines(makeRoot(swapped));
  await flush();
  assert.equal(calls.length, 2, 'retry after swap');
  assert.equal(swapped[0].hidden, false);
  console.log('ok 4: failed fetch is retried on the next swap');
}

// ── 5. empty label cached: pair with no history stays hidden, no refetch ────
{
  const nodes = [makeNode('u1', 'u2')];
  const { window, calls } = boot(nodes, () => okJson({ wins_a: 0, wins_b: 0, ties: 0, games: [] }));
  window.brScanRivalryLines(makeRoot(nodes));
  await flush();
  assert.equal(calls.length, 1);
  assert.equal(nodes[0].hidden, true);
  const swapped = [makeNode('u1', 'u2')];
  window.brScanRivalryLines(makeRoot(swapped));
  assert.equal(calls.length, 1, 'no refetch for known-empty pair');
  assert.equal(swapped[0].hidden, true);
  console.log('ok 5: empty label cached, stays hidden without refetch');
}

console.log('rivalry_cache_harness: all pass');
