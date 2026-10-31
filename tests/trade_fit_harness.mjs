// Behavioral harness for the Trade Calculator Fit grade (static/app.js).
//
// The Fit tile composites four 0-10 components that arrive from TWO
// responses: the eval (value balance, roster fit, acceptance) and the
// playoff sim (playoff impact). The contracts pinned here:
//
//   1. The composite is the average of the components that EXIST, one
//      decimal; a component the server could not compute (null) is
//      excluded, never defaulted.
//   2. An ungraded trade (graded:false - an asset has no calibrated
//      value and the totals exclude it) yields NO components at all:
//      a score built on partial totals would be fiction.
//   3. The playoff component is only read when the sim's stored
//      signature matches the trade currently on screen, so a stale sim
//      from a previous package never leaks into the grade.
//
// This harness extracts the SHIPPED functions by source and drives them
// with stub state -- not string matching.
//
// Run: node tests/trade_fit_harness.mjs   (exit 0 = pass)
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

function extractConst(src, marker) {
  const at = src.indexOf(marker);
  if (at < 0) throw new Error(`marker not found: ${marker}`);
  const end = src.indexOf('];', at);
  if (end < 0) throw new Error(`const not terminated: ${marker}`);
  return src.slice(at, end + 2);
}

// ── Stubs for the initTradePage scope the functions close over ──────
const state = {
  sideAPlayers: [],
  sideBPlayers: [],
  _lastEvalData: null,
  _lastPiData: null,
  _lastPiSig: null,
};
const root = {
  querySelector(sel) {
    if (sel.includes('viewerSide')) return { value: 'a' };
    return null;
  },
};
const getCurrentRosterId = () => '7';

const sandbox = { state, root, getCurrentRosterId };
const load = new Function(
  'state', 'root', 'getCurrentRosterId',
  [
    extractConst(APP_SRC, 'const _FIT_COMPONENTS'),
    extractBlock(APP_SRC, 'function computeTradeFitScore'),
    extractBlock(APP_SRC, 'function _currentPiSig'),
    extractBlock(APP_SRC, 'function _fitComponents'),
    'return { computeTradeFitScore, _currentPiSig, _fitComponents };',
  ].join('\n'),
);
const { computeTradeFitScore, _currentPiSig, _fitComponents } =
  load(state, root, getCurrentRosterId);

// ── 1. Composite: average of available, one decimal ─────────────────
assert.equal(
  computeTradeFitScore({ value_balance: 8, roster_fit: 6, playoff_impact: 5, acceptance: 9 }),
  7.0);
assert.equal(
  computeTradeFitScore({ value_balance: 8, roster_fit: null, playoff_impact: null, acceptance: null }),
  8.0, 'single present component is the composite');
assert.equal(
  computeTradeFitScore({ value_balance: 7, roster_fit: 8, playoff_impact: null, acceptance: null }),
  7.5);
assert.equal(
  computeTradeFitScore({ value_balance: 7, roster_fit: 7, playoff_impact: 7, acceptance: 8 }),
  7.3, 'rounds to one decimal (7.25 -> 7.3 via Math.round)');
assert.equal(
  computeTradeFitScore({ value_balance: null, roster_fit: null, playoff_impact: null, acceptance: null }),
  null, 'no components -> no score');
assert.equal(computeTradeFitScore(null), null);
assert.equal(computeTradeFitScore({}), null, 'missing keys behave as null');

// ── 2. Ungraded trades expose no components ─────────────────────────
state.sideAPlayers = [{ id: 'p1' }];
state.sideBPlayers = [{ id: 'p2' }];
state._lastEvalData = {
  graded: false,
  fit: { value_balance: 3, roster_fit: 5, acceptance: 5 },
};
assert.equal(_fitComponents(), null, 'ungraded -> null even if fit fields exist');
state._lastEvalData = null;
assert.equal(_fitComponents(), null, 'no eval -> null');

// ── 3. Playoff component gated on the sim signature ─────────────────
state._lastEvalData = {
  graded: true,
  fit: { value_balance: 6, roster_fit: null, acceptance: 4 },
};
let comps = _fitComponents();
assert.deepEqual(comps, {
  value_balance: 6, roster_fit: null, playoff_impact: null, acceptance: 4,
}, 'no sim yet: playoff hidden, eval components pass through');

state._lastPiData = { available: true, fit_playoff_impact: 8 };
state._lastPiSig = 'someone-else|p1|7'; // a previous package's sim
comps = _fitComponents();
assert.equal(comps.playoff_impact, null, 'stale sim signature -> playoff excluded');

state._lastPiSig = _currentPiSig();
comps = _fitComponents();
assert.equal(comps.playoff_impact, 8, 'matching signature -> playoff included');
assert.equal(computeTradeFitScore(comps), 6.0, '(6 + 4 + 8) / 3');

// Signature construction: viewer receives side A, gives side B, picks
// are not players and never appear.
state.sideAPlayers = [{ id: 'p1' }, { id: 'pick_2027_2_mid' }];
state.sideBPlayers = [{ id: 'p2' }, { id: 'p3' }];
assert.equal(_currentPiSig(), 'p2,p3|p1|7');

state._lastPiData = { available: false, fit_playoff_impact: 8 };
comps = _fitComponents();
assert.equal(comps.playoff_impact, null, 'unavailable sim -> playoff excluded');

console.log('CHECKS PASSED');
