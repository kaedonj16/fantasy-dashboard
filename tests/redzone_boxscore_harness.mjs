// Behavioral harness for the Redzone game box score sheet.
//
// Exercises the REAL shipped sheet block (extracted by source, not
// reimplemented): open -> fetch -> render, team toggle, pregame + error states,
// and the skill-position filter (K/DEF excluded).
//
// Run: node tests/redzone_boxscore_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'redzone.js'), 'utf8');

const start = SRC.indexOf('// ── Game box score sheet ──');
const end = SRC.indexOf('// ── End game box score sheet ──');
if (start < 0 || end < 0 || end <= start) {
  throw new Error('could not locate box score sheet block in redzone.js');
}
const block = SRC.slice(start, end);

// ── DOM stubs ────────────────────────────────────────────────────────────
const _esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));

function makeEl(tag) {
  const listeners = {};
  const el = {
    tagName: tag, className: '', innerHTML: '', textContent: '',
    style: {}, dataset: {},
    classList: { add() {}, remove() {}, toggle() {} },
    parentNode: null,
    setAttribute() {}, getAttribute() { return ''; },
    addEventListener(type, fn) { listeners[type] = fn; },
    removeEventListener() {},
    querySelector(sel) {
      el._qs = el._qs || {};
      if (!el._qs[sel]) el._qs[sel] = makeEl('div');
      return el._qs[sel];
    },
    _fire(type, e) { if (listeners[type]) listeners[type](e); },
  };
  return el;
}

const appended = [];
const document = {
  body: {
    style: {},
    appendChild(el) { el.parentNode = document.body; appended.push(el); },
    removeChild(el) {
      el.parentNode = null;
      const i = appended.indexOf(el);
      if (i >= 0) appended.splice(i, 1);
    },
  },
  createElement: (tag) => makeEl(tag),
  addEventListener() {},
  removeEventListener() {},
};
const window = {};
let fetchImpl = () => Promise.reject(new Error('no fetch stub'));
const fetch = (...args) => fetchImpl(...args);
const requestAnimationFrame = (fn) => { fn(); return 0; };

// new Function body is sloppy mode: the block's `var`s stay function-scoped,
// and the trailing return exposes the real functions.
const factory = new Function(
  '_esc', 'document', 'window', 'fetch', 'requestAnimationFrame',
  block + '\nreturn { _openBoxScore, _closeBoxScore, _boxStatusLine, _getSheet: function() { return _boxSheet; }, _getEls: function() { return _boxSheetEls; } };',
);
const api = factory(_esc, document, window, fetch, requestAnimationFrame);

// ── Fixture: shaped /api/player-team-boxscore payload ────────────────────
function group(pos, cols, players) {
  return { pos, columns: cols, players };
}
const QB_COLS = [
  { key: 'cmp_att', label: 'C/A' }, { key: 'pass_yds', label: 'Pass Yds' },
  { key: 'pass_td', label: 'Pass TD' }, { key: 'pass_int', label: 'INT' },
];
const RB_COLS = [
  { key: 'rush_att', label: 'Att' }, { key: 'rush_yds', label: 'Ru Yds' },
  { key: 'rush_td', label: 'Ru TD' }, { key: 'rec', label: 'Rec' },
];
const LIVE = {
  available: true, started: true, status: 'live',
  quarter: '3', clock: '4:32', view_team: 'DET',
  home: { team: 'GB', pts: 17 }, away: { team: 'DET', pts: 24 },
  teams: {
    DET: { team: 'DET', groups: [
      group('QB', QB_COLS, [{ id: '1', name: 'J. Goff', cells: { cmp_att: '18/26', pass_yds: 214, pass_td: 2, pass_int: 0 } }]),
      group('RB', RB_COLS, [{ id: '2', name: 'J. Gibbs', cells: { rush_att: 14, rush_yds: 72, rush_td: 1, rec: 3 } }]),
      group('K', [{ key: 'fg', label: 'FG' }], [{ id: '3', name: 'J. Bates', cells: { fg: '1/1' } }]),
      group('DEF', [{ key: 'sacks', label: 'Sacks' }], [{ id: '', name: 'DET Defense', cells: { sacks: 2 } }]),
    ] },
    GB: { team: 'GB', groups: [
      group('QB', QB_COLS, [{ id: '4', name: 'J. Love', cells: { cmp_att: '16/24', pass_yds: 198, pass_td: 1, pass_int: 1 } }]),
      group('WR', [{ key: 'rec', label: 'Rec' }, { key: 'rec_yds', label: 'Yds' }, { key: 'rec_td', label: 'TD' }],
        [{ id: '5', name: 'C. Watson', cells: { rec: 5, rec_yds: 76, rec_td: 1 } }]),
    ] },
  },
};

const ok = (data) => Promise.resolve({ ok: true, json: () => Promise.resolve(data) });
const tick = () => new Promise((r) => setTimeout(r, 10));

// ── 1. Live game: renders, defaults to view_team, skill positions only ───
fetchImpl = () => ok(LIVE);
api._openBoxScore('20240905_DET@GB');
await tick(); await tick();

let sheet = api._getSheet();
assert.ok(sheet && sheet.data, 'sheet should hold fetched data');
assert.equal(sheet.team, 'DET', 'defaults to payload view_team');
let els = api._getEls();
assert.equal(els.title.textContent, 'DET @ GB');
assert.match(els.sub.textContent, /Q3/, 'status line shows quarter');
let html = els.body.innerHTML;
assert.match(html, /J\. Goff/, 'QB row rendered');
assert.match(html, /J\. Gibbs/, 'RB row rendered');
assert.ok(!html.includes('J. Bates'), 'K group excluded (skill players only)');
assert.ok(!html.includes('DET Defense'), 'DEF group excluded');
assert.ok(html.includes('data-box-team="DET"'), 'team toggle has DET');
assert.ok(html.includes('data-box-team="GB"'), 'team toggle has GB');
assert.match(html, /is-on[^>]*data-box-team="DET"|data-box-team="DET"[^>]*is-on/, 'DET toggle active');

// ── 2. Team toggle swaps tables ──────────────────────────────────────────
const sheetEl = appended.find((el) => el.className.includes('rz-bs-sheet'));
assert.ok(sheetEl, 'sheet element mounted on body');
sheetEl._fire('click', {
  target: { closest: (sel) => (sel === '[data-box-team]'
    ? { getAttribute: () => 'GB' } : null) },
});
els = api._getEls();
html = els.body.innerHTML;
assert.equal(api._getSheet().team, 'GB', 'toggle switches selected team');
assert.match(html, /J\. Love/, 'GB players render after toggle');
assert.ok(!html.includes('J. Goff'), 'DET players gone after toggle');

// ── 3. Player rows tap through to the player modal ───────────────────────
let modalArgs = null;
window.openPlayerModal = (pid, name, opts) => { modalArgs = { pid, name, opts }; };
sheetEl._fire('click', {
  target: { closest: (sel) => (sel === '[data-pid]'
    ? { getAttribute: (k) => (k === 'data-pid' ? '4' : 'J. Love') } : null) },
});
assert.deepEqual(modalArgs, { pid: '4', name: 'J. Love', opts: { tab: 'live' } });

// ── 4. Pregame: message, no tables ───────────────────────────────────────
fetchImpl = () => ok({
  available: true, started: false, status: 'scheduled',
  message: 'Box score available once the game begins.',
  home: { team: 'KC', pts: null }, away: { team: 'BUF', pts: null }, teams: {},
});
api._openBoxScore('20240906_BUF@KC');
await tick(); await tick();
els = api._getEls();
assert.match(els.body.innerHTML, /once the game begins/, 'pregame message shown');
assert.match(els.sub.textContent, /Not started/, 'pregame status line');

// ── 5. Error + retry ─────────────────────────────────────────────────────
fetchImpl = () => Promise.reject(new Error('boom'));
api._openBoxScore('20240905_DET@GB');
await tick(); await tick();
els = api._getEls();
assert.match(els.body.innerHTML, /Try again/, 'error state offers retry');
fetchImpl = () => ok(LIVE);
api._getEls().sheet._fire('click', {
  target: { closest: (sel) => (sel === '[data-box-retry]' ? {} : null) },
});
await tick(); await tick();
assert.ok(api._getSheet().data, 'retry refetches and renders');

// ── 6. Close removes the sheet ───────────────────────────────────────────
api._closeBoxScore();
assert.equal(api._getSheet(), null, 'sheet state cleared');
assert.equal(api._getEls(), null, 'sheet elements cleared');

console.log('box score harness: all assertions passed');
