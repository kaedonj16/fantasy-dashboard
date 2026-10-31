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
//   4. The eval payload itself is signature-stamped on landing: after
//      an edit, the previous package's payload yields no components and
//      paints nothing. While a recompute is in flight (_fitPending),
//      the Fit tile shows its grading skeleton, the acceptance chip
//      hides, and the sticky bar placeholders its Fit and verdict
//      slots; a terminal failure resolves to the no-Fit state.
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
  sideAPicks: [],
  sideBPicks: [],
  _lastEvalData: null,
  _lastEvalSig: '',
  _fitPending: false,
  _lastPiData: null,
  _lastPiSig: null,
  _lastEffA: 0,
  _lastEffB: 0,
};

function fakeEl() {
  const el = {
    style: {},
    textContent: '',
    innerHTML: '',
    title: '',
    attrs: {},
    classes: new Set(),
    classList: {
      add: c => el.classes.add(c),
      remove: c => el.classes.delete(c),
      contains: c => el.classes.has(c),
    },
    setAttribute(k, v) { el.attrs[k] = v; },
    removeAttribute(k) { delete el.attrs[k]; },
  };
  return el;
}
const els = {};
[
  '#tradeFitTile', '#tradeFitGrid', '#tradeFitScore', '#tradeAcceptChip',
  '#tradeStickyBar', '#tradeStickyTotals', '#tradeStickyVerdict',
].forEach(sel => { els[sel] = fakeEl(); });

const root = {
  querySelector(sel) {
    if (sel.includes('viewerSide')) return { value: 'a' };
    return els[sel] || null;
  },
};
const document = {
  body: { classList: { add() {}, remove() {} } },
};
const getCurrentRosterId = () => '7';

const sandbox = { state, root, document, getCurrentRosterId };
const load = new Function(
  'state', 'root', 'document', 'getCurrentRosterId',
  [
    extractConst(APP_SRC, 'const _FIT_COMPONENTS'),
    extractBlock(APP_SRC, 'function computeTradeFitScore'),
    extractBlock(APP_SRC, 'function _currentPiSig'),
    extractBlock(APP_SRC, 'function _currentEvalSig'),
    extractBlock(APP_SRC, 'function _fitComponents'),
    extractBlock(APP_SRC, 'function renderTradeFit'),
    extractBlock(APP_SRC, 'function renderStickyBar'),
    'return { computeTradeFitScore, _currentPiSig, _currentEvalSig, _fitComponents, renderTradeFit, renderStickyBar };',
  ].join('\n'),
);
const { computeTradeFitScore, _currentPiSig, _currentEvalSig, _fitComponents, renderTradeFit } =
  load(state, root, document, getCurrentRosterId);

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
state._lastEvalSig = _currentEvalSig();
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

state._lastEvalSig = _currentEvalSig();
state._lastPiData = { available: false, fit_playoff_impact: 8 };
comps = _fitComponents();
assert.equal(comps.playoff_impact, null, 'unavailable sim -> playoff excluded');

// ── 4. Signature guard + grading state on the rendered surfaces ─────
// A payload is only ever painted for the exact selection it evaluated:
// recomputeTrade stamps _lastEvalSig when an eval lands, and entering a
// recompute sets _fitPending, which swaps every Fit surface to its
// grading treatment instead of the previous package's numbers.
state.sideAPlayers = [{ id: 'p1' }];
state.sideBPlayers = [{ id: 'p2' }];
state.sideAPicks = [];
state.sideBPicks = [{ id: 'pick_2027_2_mid' }];
state._lastPiData = null;
state._lastPiSig = '';
state._lastEffA = 315.7;
state._lastEffB = 248.9;
assert.equal(_currentEvalSig(), 'p1|p2||pick_2027_2_mid',
  'eval signature covers players and picks on both sides');

state._lastEvalData = {
  graded: true,
  verdict: 'Team 1 is favored by about 66.8 value.',
  fit: { value_balance: 8, roster_fit: null, acceptance: 4, acceptance_label: 'Possible' },
};
state._lastEvalSig = _currentEvalSig();
state._fitPending = false;
comps = _fitComponents();
assert.equal(comps.value_balance, 8, 'fresh-stamped payload: components read');
renderTradeFit();
assert.equal(els['#tradeFitTile'].style.display, 'flex', 'graded tile shows');
assert.equal(els['#tradeFitScore'].textContent, '6.0', '(8 + 4) / 2 paints');
assert.equal(els['#tradeAcceptChip'].textContent, 'Acceptance: Possible');
assert.ok(els['#tradeStickyTotals'].innerHTML.includes('Fit 6.0'), 'sticky bar shows the grade');
assert.equal(els['#tradeStickyVerdict'].textContent, 'Team 1 is favored by about 66.8 value.');

// The user edits: the signature moves on and grading starts. The old
// payload must now paint NOTHING: no components, no grade, no chip,
// no verdict.
state.sideBPlayers = [{ id: 'p2' }, { id: 'p3' }];
state._fitPending = true;
assert.equal(_fitComponents(), null, 'stale-stamped payload -> no components');
renderTradeFit();
assert.equal(els['#tradeFitTile'].style.display, 'flex', 'grading tile stays visible');
assert.ok(els['#tradeFitTile'].classList.contains('otc-fit--grading'), 'grading class on the tile');
assert.equal(els['#tradeFitTile'].attrs['aria-busy'], 'true');
assert.equal(els['#tradeFitScore'].textContent, '…', 'placeholder score, not the old 6.0');
assert.ok(els['#tradeFitGrid'].innerHTML.includes('skeleton'), 'skeleton tracks while grading');
assert.equal(els['#tradeAcceptChip'].style.display, 'none', 'no stale acceptance label');
assert.ok(els['#tradeStickyTotals'].innerHTML.includes('Fit &hellip;'), 'sticky Fit slot placeholders');
assert.ok(!els['#tradeStickyTotals'].innerHTML.includes('Fit 6.0'), 'old grade is gone from the bar');
assert.ok(!els['#tradeStickyTotals'].innerHTML.includes('Acceptance'), 'no stale acceptance in the bar');
assert.equal(els['#tradeStickyVerdict'].textContent, 'Grading…', 'sticky verdict placeholders too');

// Terminal failure: grading resolves to the no-Fit state, and the stale
// payload stays locked out by its signature.
state._fitPending = false;
renderTradeFit();
assert.equal(els['#tradeFitTile'].style.display, 'none', 'failed grading resolves: tile hides');
assert.ok(!els['#tradeFitTile'].classList.contains('otc-fit--grading'), 'grading class cleared');
assert.ok(!els['#tradeStickyTotals'].innerHTML.includes('Fit'), 'no Fit slot after failure');
assert.equal(els['#tradeStickyVerdict'].style.display, 'none', 'stale verdict stays hidden');

// The new eval lands: stamp + payload swap in the real grade.
state._lastEvalData = {
  graded: true,
  verdict: 'Team 2 is favored by about 12.0 value.',
  fit: { value_balance: 4, roster_fit: 6, acceptance: null, acceptance_label: null },
};
state._lastEvalSig = _currentEvalSig();
renderTradeFit();
assert.equal(els['#tradeFitScore'].textContent, '5.0', '(4 + 6) / 2 paints for the new trade');
assert.equal(els['#tradeAcceptChip'].style.display, 'none', 'no acceptance -> chip stays hidden');
assert.equal(els['#tradeStickyVerdict'].textContent, 'Team 2 is favored by about 12.0 value.');

console.log('CHECKS PASSED');
