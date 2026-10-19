// Behavioral harness for the signed-in recovery fixes (PR C).
//
// Exercises the SHIPPED _pmFetch + _pmSmallFetch from static/player_modal.js
// (extracted by source, declaration lines read from the file so they cannot
// drift) with a fetch that HANGS until its AbortSignal fires -- the failure
// behind the player modal's RECENT NEWS sitting on "Loading..." forever.
// window.brFetchWithTimeout is stubbed with the same contract as the real
// one in app.js (bounded wait, abort on timeout), at a test-sized bound:
//
//   * a hung request settles as null once the bounded wait aborts it
//   * the timeout does not poison the in-flight cache entry: the next call
//     refetches and can succeed (this is what the section Retry relies on)
//   * a normal success is still cached and shared as before
//
// Run: node tests/signedin_recovery_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
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

function declLine(src, marker) {
  const at = src.indexOf(marker);
  if (at < 0) throw new Error(`decl not found: ${marker}`);
  return src.slice(at, src.indexOf('\n', at));
}

const vars = [
  declLine(MODAL_SRC, 'var _pmSmallCache'),
  declLine(MODAL_SRC, 'var _PM_SMALL_TTL'),
].join('\n');
const fn = extractBlock(MODAL_SRC, 'function _pmFetch(url, ms)')
  + '\n' + extractBlock(MODAL_SRC, 'function _pmSmallFetch(key, url)');

let fetchCalls = 0;
let mode = 'hang';
function fakeFetch(url, opts) {
  fetchCalls++;
  if (mode === 'hang') {
    // Never settles on its own; rejects only when the caller aborts.
    return new Promise((_, reject) => {
      const sig = opts && opts.signal;
      if (sig) sig.addEventListener('abort', () => {
        const e = new Error('The operation was aborted');
        e.name = 'AbortError';
        reject(e);
      });
    });
  }
  return Promise.resolve({
    ok: true,
    json: async () => ({ news: [{ headline: 'h' }] }),
  });
}

// Same contract as app.js's brFetchWithTimeout: abort the underlying fetch
// after ms and reject. The bound is shrunk to 40ms so the test runs fast;
// _pmSmallFetch passes its shipped 15000 and the stub's bound wins, exactly
// like the real helper's timeout winning over a hung origin.
const fakeWindow = {
  brFetchWithTimeout: (url, init, ms) => {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 40);
    return fakeFetch(url, { signal: ctl.signal }).finally(() => clearTimeout(t));
  },
};
const run = new Function(
  'Date', 'fetch', 'Map', 'AbortController', 'setTimeout', 'clearTimeout', 'window',
  vars + '\n' + fn +
  '\nreturn { _pmSmallFetch };'
);
const { _pmSmallFetch } = run(
  Date, fakeFetch, Map, AbortController, setTimeout, clearTimeout, fakeWindow
);

let passed = 0;
function check(name, cond) {
  if (!cond) throw new Error('FAILED: ' + name);
  passed++;
  console.log('ok - ' + name);
}

// 1. A hung request settles as null instead of spinning forever.
const watchdog = new Promise((_, reject) =>
  setTimeout(() => reject(new Error('hung fetch never settled')), 2000));
const hung = await Promise.race([_pmSmallFetch('news:p1', '/api/player-news/p1'), watchdog]);
check('hung small fetch settles as null after the bounded wait', hung === null);
check('hung fetch was attempted exactly once', fetchCalls === 1);

// 2. The timeout dropped the in-flight entry: a retry refetches and succeeds.
mode = 'ok';
const retried = await _pmSmallFetch('news:p1', '/api/player-news/p1');
check('retry after timeout refetches', fetchCalls === 2);
check('retry returns the payload', !!(retried && retried.news && retried.news.length === 1));

// 3. Success is still cached/shared within the TTL.
const again = await _pmSmallFetch('news:p1', '/api/player-news/p1');
check('successful payload is cached (no third fetch)', fetchCalls === 2 && again === retried);

console.log(`\nCHECKS PASSED (${passed})`);
