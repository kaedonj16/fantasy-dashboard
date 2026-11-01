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
//      the Fit tile shows its grading skeleton and the acceptance chip
//      hides; a terminal failure resolves to the no-Fit state.
//   5. The sticky bar is a balance strip only: the two side values and
//      a mini balance band whose geometry comes from _balanceGeometry,
//      the same computation the summary's full bar uses (extracted from
//      the eval-landing handler, which now calls it too). It carries
//      no Fit score, no acceptance, and no verdict text in any state,
//      it hides while the Trade Summary card itself is on screen
//      (state._summaryInView, owned by the IntersectionObserver wired
//      in bindStickyBar), and activating it scrolls to the summary.
//      The scroll targets this app's real scroller: the <body>, not
//      the window (the getScroll/setScroll contract in app.js). The
//      fake DOM below models that faithfully: window.scrollY is 0
//      and only body.scrollTop holds the offset, so a window-only
//      scrollTo (the original #2929 code) computes the wrong target
//      AND writes it to an element that never moves.
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
  _summaryInView: false,
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
  '#tradeStickyBar', '#tradeStickyValueA', '#tradeStickyValueB',
  '#tradeStickyBandFair', '#tradeStickyBandMarker',
].forEach(sel => { els[sel] = fakeEl(); });

// The summary card the sticky bar scrolls to: records its scroll calls.
const summaryCard = fakeEl();
let summaryRectTop = 500;
let summaryScrollIntoViewCalls = 0;
summaryCard.getBoundingClientRect = () => ({ top: summaryRectTop });
summaryCard.scrollIntoView = () => { summaryScrollIntoViewCalls++; };
els['.otc-summary-card'] = summaryCard;

const root = {
  querySelector(sel) {
    if (sel.includes('viewerSide')) return { value: 'a' };
    return els[sel] || null;
  },
};
const navEl = { offsetHeight: 68 };
// This app scrolls the <body>, not the window (app.js getScroll /
// setScroll): model the real shell, where the 200px offset lives on
// body.scrollTop and the window/documentElement offsets stay 0.
const bodyScrollCalls = [];
const docElScrollCalls = [];
const document = {
  body: {
    classList: { add() {}, remove() {} },
    scrollTop: 200,
    scrollTo(opts) { bodyScrollCalls.push(opts); },
  },
  documentElement: {
    scrollTop: 0,
    scrollTo(opts) { docElScrollCalls.push(opts); },
  },
  querySelector(sel) { return sel === '.top-nav' ? navEl : null; },
};
const getCurrentRosterId = () => '7';

// window is a free variable in scrollStickyBarToSummary; stub it on the
// global so the extracted function resolves it like the browser does.
const windowScrollCalls = [];
globalThis.window = {
  scrollY: 0,
  pageYOffset: 0,
  scrollTo(opts) { windowScrollCalls.push(opts); },
};

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
    extractBlock(APP_SRC, 'function _balanceGeometry'),
    extractBlock(APP_SRC, 'function renderStickyBar'),
    extractBlock(APP_SRC, 'function scrollStickyBarToSummary'),
    'return { computeTradeFitScore, _currentPiSig, _currentEvalSig, _fitComponents, renderTradeFit, _balanceGeometry, renderStickyBar, scrollStickyBarToSummary };',
  ].join('\n'),
);
const { computeTradeFitScore, _currentPiSig, _currentEvalSig, _fitComponents, renderTradeFit, _balanceGeometry, scrollStickyBarToSummary } =
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
  diff: 66.8, abs_diff: 66.8, fair_threshold: 25.0, fair_pct: 0.0792,
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
// The sticky bar shows none of the grade: just the two side values and
// the mini band, positioned by the shared _balanceGeometry (diff 66.8
// on a 315.7 max side, 25-point fair floor from the server).
assert.equal(els['#tradeStickyBar'].style.display, 'flex', 'bar shows for a landed trade');
assert.equal(els['#tradeStickyValueA'].textContent, '315.7');
assert.equal(els['#tradeStickyValueB'].textContent, '248.9');
const geo1 = _balanceGeometry(66.8, 315.7, 248.9, 25.0, 0.0792);
assert.equal(els['#tradeStickyBandMarker'].style.left, geo1.leftPct + '%',
  'band marker uses the shared geometry');
assert.equal(els['#tradeStickyBandFair'].style.width, geo1.fairWidthPct + '%',
  'band fair zone uses the shared geometry');
assert.ok(Math.abs(geo1.leftPct - 39.42) < 0.01, 'marker left of center: Team 1 favored');
assert.ok(Math.abs(geo1.fairWidthPct - 7.92) < 0.01, 'fair zone width from fair_threshold');

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
// The bar has no grading placeholders to swap and no slots to stale:
// it keeps showing the last landed values + band, exactly what the
// summary still shows mid-grade.
assert.equal(els['#tradeStickyValueA'].textContent, '315.7', 'bar keeps landed values while grading');
assert.equal(els['#tradeStickyValueB'].textContent, '248.9');
assert.equal(els['#tradeStickyBandMarker'].style.left, geo1.leftPct + '%',
  'band keeps the landed position while grading');

// Terminal failure: grading resolves to the no-Fit state, and the stale
// payload stays locked out by its signature.
state._fitPending = false;
renderTradeFit();
assert.equal(els['#tradeFitTile'].style.display, 'none', 'failed grading resolves: tile hides');
assert.ok(!els['#tradeFitTile'].classList.contains('otc-fit--grading'), 'grading class cleared');
assert.equal(els['#tradeStickyBar'].style.display, 'flex', 'bar stays up on the landed trade');
assert.equal(els['#tradeStickyValueA'].textContent, '315.7', 'bar still mirrors the summary after failure');

// The new eval lands: stamp + payload swap in the real grade.
state._lastEvalData = {
  graded: true,
  verdict: 'Team 2 is favored by about 12.0 value.',
  diff: -12.0, abs_diff: 12.0, fair_threshold: 25.0, fair_pct: 0.0801,
  fit: { value_balance: 4, roster_fit: 6, acceptance: null, acceptance_label: null },
};
state._lastEvalSig = _currentEvalSig();
state._lastEffA = 300.0;
state._lastEffB = 312.0;
renderTradeFit();
assert.equal(els['#tradeFitScore'].textContent, '5.0', '(4 + 6) / 2 paints for the new trade');
assert.equal(els['#tradeAcceptChip'].style.display, 'none', 'no acceptance -> chip stays hidden');
assert.equal(els['#tradeStickyValueA'].textContent, '300.0', 'bar swaps in the new values');
assert.equal(els['#tradeStickyValueB'].textContent, '312.0');
const geo2 = _balanceGeometry(-12.0, 300.0, 312.0, 25.0, 0.0801);
assert.equal(els['#tradeStickyBandMarker'].style.left, geo2.leftPct + '%');
assert.ok(Math.abs(geo2.leftPct - 51.92) < 0.01, 'marker right of center: Team 2 favored');

// ── 5. Sticky bar: hides with no eval; activation scrolls to summary ──
state._lastEvalData = null;
renderTradeFit();
assert.equal(els['#tradeStickyBar'].style.display, 'none', 'no eval payload -> bar hides');

scrollStickyBarToSummary();
// The target is computed from the offset that actually exists (the
// body's 200, not the window's 0) and written to the app's real
// scroller. The #2929 original read window.scrollY (0) and wrote only
// window.scrollTo: wrong target, inert element, dead tap.
assert.deepEqual(bodyScrollCalls, [{ top: 624, behavior: 'smooth' }],
  'activation smooth-scrolls the body: 500 rect + 200 body offset - 68 nav - 8 gap');
assert.deepEqual(docElScrollCalls, [{ top: 624, behavior: 'smooth' }],
  'documentElement is written too (whichever candidate scrolls, the others no-op)');
assert.deepEqual(windowScrollCalls, [{ top: 624, behavior: 'smooth' }],
  'window is written with the same corrected target, never as the only write');
assert.equal(summaryScrollIntoViewCalls, 0, 'explicit offset path, not bare scrollIntoView');

// ── 6. Sticky bar: hidden while the summary itself is on screen ─────
// bindStickyBar's IntersectionObserver owns state._summaryInView; the
// renderer composes it with the armed gate (landed eval + assets).
state._lastEvalData = {
  graded: true,
  diff: 66.8, abs_diff: 66.8, fair_threshold: 25.0, fair_pct: 0.0792,
  fit: { value_balance: 8, roster_fit: null, acceptance: 4, acceptance_label: 'Possible' },
};
state._lastEvalSig = _currentEvalSig();
state._lastEffA = 315.7;
state._lastEffB = 248.9;
state._summaryInView = false;
renderTradeFit();
assert.equal(els['#tradeStickyBar'].style.display, 'flex', 'summary off screen -> bar shows');
state._summaryInView = true;
renderTradeFit();
assert.equal(els['#tradeStickyBar'].style.display, 'none', 'summary on screen -> bar hides');
state._summaryInView = false;
renderTradeFit();
assert.equal(els['#tradeStickyBar'].style.display, 'flex', 'summary leaves -> bar returns');
assert.equal(els['#tradeStickyValueA'].textContent, '315.7', 'values intact across the toggle');

console.log('CHECKS PASSED');
