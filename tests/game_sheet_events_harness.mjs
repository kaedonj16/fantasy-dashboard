// Behavioral harness for the Matchups game sheet's play derivation.
//
// The sheet's Plays tab renders one game's raw pbp_by_game rows as
// ScoreZone feed events. window._rzGameEvents (app.js) derives them:
// one event per NFL play, headlined by the primary fantasy contributor,
// scored with that contributor's league scoring, newest first, with the
// score as of the play derived the way ScoreZone's _scoreStrForEvent
// derives it. This exercises the REAL shipped helper (extracted by
// source, not reimplemented).
//
// Run: node tests/game_sheet_events_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'app.js'), 'utf8');

// Extract the contiguous helper block that ends just before the stub IIFE.
const start = SRC.indexOf('window._rzScoringForPid = function');
const end = SRC.indexOf('// Default stub for non-ScoreZone pages');
if (start < 0 || end < 0 || end <= start) {
  throw new Error('could not locate the ScoreZone stub helper block in app.js');
}
const block = SRC.slice(start, end);

const window = {};
// eslint-disable-next-line no-eval
eval(block);
assert.equal(typeof window._rzGameEvents, 'function', '_rzGameEvents exported');

const scoring = {
  rush_yd: 0.1, rush_td: 6, rec: 1, rec_yd: 0.1, rec_td: 6,
  pass_yd: 0.04, pass_td: 4, pass_int: -2, fum_lost: -2, xpm: 1,
};
const state = {
  scoring,
  player_info: {
    '1': { name: 'Bryce Young', team: 'CAR', pos: 'QB' },
    '2': { name: 'Tetairoa McMillan', team: 'CAR', pos: 'WR' },
    '3': { name: 'Chuba Hubbard', team: 'CAR', pos: 'RB' },
    '4': { name: 'Jordan Love', team: 'GB', pos: 'QB' },
    '5': { name: 'Tucker Kraft', team: 'GB', pos: 'TE' },
    '6': { name: 'Ryan Fitzgerald', team: 'CAR', pos: 'K' },
    '7': { name: 'Brandon McManus', team: 'GB', pos: 'K' },
  },
  games: {
    'CAR@GB': { away: 'CAR', home: 'GB', away_pts: 7, home_pts: 7 },
    'KC@BUF': { away: 'KC', home: 'BUF', away_pts: 0, home_pts: 0 },
  },
  pbp_by_game: {
    'CAR@GB': [
      { play_id: 'p1', seq: 1, pid: '3', name: 'Chuba Hubbard', team: 'CAR',
        quarter: 'Q1', clock: '9:24', down: '2', distance: 'Goal', yard_line: 'GB 3',
        play_text: 'Chuba Hubbard run up the middle for 3 yards, touchdown',
        stat_line: { carries: 1, rush_yds: 3, rush_td: 1 }, is_td: true },
      { play_id: 'p2', seq: 2, pid: '6', name: 'Ryan Fitzgerald', team: 'CAR',
        quarter: 'Q1', clock: '9:24',
        play_text: 'Ryan Fitzgerald extra point is good',
        stat_line: { xpm: 1 } },
      // One NFL play, two fantasy contributions (passer + receiver).
      { play_id: 'p9', seq: 9, pid: '4', name: 'Jordan Love', team: 'GB',
        quarter: 'Q3', clock: '12:15', down: '1', distance: '10', yard_line: 'CAR 16',
        play_text: 'Jordan Love pass short middle to Tucker Kraft for 16 yards, touchdown',
        stat_line: { pass_yds: 16, pass_td: 1 }, is_td: true },
      { play_id: 'p9', seq: 9, pid: '5', name: 'Tucker Kraft', team: 'GB',
        quarter: 'Q3', clock: '12:15', down: '1', distance: '10', yard_line: 'CAR 16',
        play_text: 'Jordan Love pass short middle to Tucker Kraft for 16 yards, touchdown',
        stat_line: { rec: 1, rec_yds: 16, rec_td: 1, targets: 1 }, is_td: true },
      { play_id: 'p10', seq: 10, pid: '7', name: 'Brandon McManus', team: 'GB',
        quarter: 'Q3', clock: '12:15',
        play_text: 'Brandon McManus extra point is good',
        stat_line: { xpm: 1 } },
      { play_id: 'p12', seq: 12, pid: '3', name: 'Chuba Hubbard', team: 'CAR',
        quarter: 'Q3', clock: '7:50', down: '1', distance: '10', yard_line: 'CAR 32',
        play_text: 'Chuba Hubbard run up the middle for 4 yards',
        stat_line: { carries: 1, rush_yds: 4 } },
    ],
    'KC@BUF': [
      // Unresolvable player in another game: must never leak into CAR@GB.
      { play_id: 'x1', seq: 1, pid: '99', name: 'Patrick Mahomes', team: 'KC',
        quarter: 'Q1', clock: '5:00',
        play_text: 'Patrick Mahomes pass for 9 yards',
        stat_line: { pass_yds: 9 } },
    ],
  },
};

const game = state.games['CAR@GB'];
const events = window._rzGameEvents(state, 'CAR@GB', game);

// 1. One event per NFL play of the tapped game (the TD pass groups its
//    passer + receiver rows into a single event), nothing from KC@BUF.
assert.equal(events.length, 5, 'expected 5 events, got ' + events.length);
assert.ok(events.every((e) => e.gameId === 'CAR@GB'), 'all events belong to the tapped game');

// 2. Newest first by seq.
assert.deepEqual(events.map((e) => e.seq), [12, 10, 9, 2, 1], 'newest-first order');

const bySeq = Object.fromEntries(events.map((e) => [e.seq, e]));

// 3. The grouped TD pass is headlined by the receiver (primary actor),
//    scored with his line: 1 rec + 16 yds + TD = 8.6.
const td = bySeq[9];
assert.equal(td.pid, '5', 'receiver headlines the TD pass');
assert.equal(td.name, 'Tucker Kraft');
assert.equal(td.kind, 'td');
assert.equal(td.pts, 8.6, 'Kraft pts');
assert.equal(td.down, '1');
assert.equal(td.distance, '10');
assert.equal(td.yardLine, 'CAR 16');

// 4. Score as of the play, derived from the reconciled NFL-points walk:
//    Hubbard TD 6-0, PAT 7-0, Kraft TD 7-6, PAT 7-7, last gain still 7-7.
assert.deepEqual([bySeq[1].scoreAway, bySeq[1].scoreHome], [6, 0], 'TD play itself reads 6-0');
assert.deepEqual([bySeq[2].scoreAway, bySeq[2].scoreHome], [7, 0], 'PAT play reads 7-0');
assert.deepEqual([bySeq[9].scoreAway, bySeq[9].scoreHome], [7, 6], 'Kraft TD reads 7-6');
assert.deepEqual([bySeq[12].scoreAway, bySeq[12].scoreHome], [7, 7], 'latest play reads 7-7');

// 5. A plain gain: kind, points off the rusher's line.
assert.equal(bySeq[12].kind, 'gain');
assert.equal(bySeq[12].pts, 0.4, 'Hubbard gain pts');
assert.equal(bySeq[1].pts, 6.3, 'Hubbard TD pts');

// 6. Other game: only its own (unresolvable) play, so no events; and a
//    game with no pbp at all degrades to an empty list.
assert.deepEqual(window._rzGameEvents(state, 'KC@BUF', state.games['KC@BUF']), []);
assert.deepEqual(window._rzGameEvents(state, 'NOPE', {}), []);
assert.deepEqual(window._rzGameEvents({}, 'CAR@GB', game), []);
assert.deepEqual(window._rzGameEvents(null, 'CAR@GB', game), []);

// 7. When the walk cannot reconcile with the game's current score (a
//    scoring play the stat model cannot see), score-as-of is withheld
//    rather than printed wrong.
const skewed = JSON.parse(JSON.stringify(state));
skewed.games['CAR@GB'] = { away: 'CAR', home: 'GB', away_pts: 9, home_pts: 7 };
const noScore = window._rzGameEvents(skewed, 'CAR@GB', skewed.games['CAR@GB']);
assert.equal(noScore.length, 5, 'events still derive when scores cannot reconcile');
assert.ok(noScore.every((e) => e.scoreAway == null && e.scoreHome == null),
  'unreconciled walk yields no score line');

// ── Plays pull-to-refresh ─────────────────────────────────────────────
// The sheet's Plays tab carries its own pull-to-refresh: the gesture
// helper (/* gs-ptr:start */ block, extracted verbatim) plus the sheet's
// fetch (gsPtrEnabled / gsPlaysRefresh, extracted by source span), wired
// together exactly as openGameSheet wires them, against a stubbed
// __brSzLiveCache and a fake DOM. Cases: a pull at the top past the
// threshold fires exactly one fresh fetch (invalidate, then get) and
// re-renders with the new play on top; a short pull, a pull begun
// scrolled down, a pull the panel scrolls away from, a horizontal drag,
// and any pull on the Stats tab all fetch nothing; a dropped fetch
// (the real cache drops waiters on a network failure) settles through
// the timer with the rendered list byte-identical and the failure note
// shown; a same-reference flush (304 / stale body) counts as current
// and re-renders nothing.

const pStart = SRC.indexOf('/* gs-ptr:start */');
const pEnd = SRC.indexOf('/* gs-ptr:end */');
assert.ok(pStart >= 0 && pEnd > pStart, 'gs-ptr delimiters present in app.js');
const PTR_BLOCK = SRC.slice(pStart, pEnd);

const rStart = SRC.indexOf('function gsPtrEnabled()');
const rEnd = SRC.indexOf('function gsSheetKey', rStart);
assert.ok(rStart >= 0 && rEnd > rStart, 'gs plays refresh functions present in app.js');
const REFRESH_SPAN = SRC.slice(rStart, rEnd);

function makeTimers() {
  const pending = new Map();
  let nextId = 0;
  return {
    setTimeout(fn) { const id = ++nextId; pending.set(id, fn); return id; },
    clearTimeout(id) { pending.delete(id); },
    fireAll() { const fns = [...pending.values()]; pending.clear(); fns.forEach((fn) => fn()); },
    pendingCount() { return pending.size; },
  };
}

class PtrEl {
  constructor() {
    this.style = {};
    this.hidden = false;
    this.textContent = '';
    this.scrollTop = 0;
    this._listeners = {};
    const set = new Set();
    this.classList = {
      add: (...cs) => cs.forEach((c) => set.add(c)),
      remove: (...cs) => cs.forEach((c) => set.delete(c)),
      toggle: (c, force) => {
        const on = force === undefined ? !set.has(c) : !!force;
        if (on) set.add(c); else set.delete(c);
        return on;
      },
      contains: (c) => set.has(c),
    };
  }
  addEventListener(type, fn) { (this._listeners[type] = this._listeners[type] || []).push(fn); }
  removeEventListener(type, fn) {
    const a = this._listeners[type] || [];
    const i = a.indexOf(fn);
    if (i >= 0) a.splice(i, 1);
  }
  fire(type, ev) { (this._listeners[type] || []).slice().forEach((fn) => fn(ev)); }
}

const ptrTouch = (x, y) => ({
  touches: [{ clientX: x, clientY: y }],
  changedTouches: [{ clientX: x, clientY: y }],
  cancelable: true,
  defaultPrevented: false,
  preventDefault() { this.defaultPrevented = true; },
});

// mode 'new': the refetch returns a payload with one newer play.
// mode 'same': the refetch flushes the cached reference (304 / stale).
// mode 'drop': the waiter is silently dropped (network failure).
function makePtrWorld(mode) {
  const ht = makeTimers();
  const st = makeTimers();
  const v1 = state;
  const v2 = JSON.parse(JSON.stringify(state));
  v2.pbp_by_game['CAR@GB'].push({
    play_id: 'p13', seq: 13, pid: '3', name: 'Chuba Hubbard', team: 'CAR',
    quarter: 'Q3', clock: '6:58', down: '1', distance: '10', yard_line: 'CAR 36',
    play_text: 'Chuba Hubbard run up the middle for 4 yards',
    stat_line: { carries: 1, rush_yds: 4 },
  });
  const calls = { invalidate: 0, get: 0, order: [] };
  let held = null;
  const cache = {
    peek() { return v1; },
    invalidate() { calls.invalidate++; calls.order.push('invalidate'); },
    get(variant, staleMs, cb) {
      calls.get++;
      calls.order.push('get');
      assert.equal(variant, 'full', 'refresh fetches the full variant');
      if (mode === 'new') held = () => cb(v2);
      else if (mode === 'same') held = () => cb(v1);
      else held = null;
      return v1;
    },
  };
  const sheet = { game, gid: 'CAR@GB', tab: 'plays', fullStatus: 'ready', full: v1 };
  let renderCount = 0;
  let hostHtml = '';
  const render = () => {
    renderCount++;
    const evs = window._rzGameEvents(sheet.full, sheet.gid, sheet.full.games[sheet.gid]);
    hostHtml = evs.map((e) => e.seq).join(',');
  };
  render(); // the sheet's initial render on open (cache path, no invalidate)
  renderCount = 0;
  const factory = new Function('window', 'gsSheet', 'gsRenderPlays', 'setTimeout', 'clearTimeout',
    REFRESH_SPAN + '\nreturn { enabled: gsPtrEnabled, refresh: gsPlaysRefresh };');
  const api = factory({ __brSzLiveCache: cache }, sheet, render, st.setTimeout, st.clearTimeout);
  const ptrWin = {};
  new Function('window', 'setTimeout', 'clearTimeout', PTR_BLOCK)(
    ptrWin, ht.setTimeout, ht.clearTimeout);
  assert.equal(typeof ptrWin.__brGsAttachPtr, 'function', '__brGsAttachPtr exported');
  const panel = new PtrEl();
  const scroller = new PtrEl();
  const wrap = new PtrEl();
  const chip = new PtrEl();
  const note = new PtrEl();
  note.hidden = true; // as in the sheet markup
  const detach = ptrWin.__brGsAttachPtr({
    panel, scroller, wrap, chip, note,
    enabled: api.enabled, refresh: api.refresh,
  });
  const pull = (dy) => {
    panel.fire('touchstart', ptrTouch(100, 10));
    panel.fire('touchmove', ptrTouch(100, 10 + dy));
    panel.fire('touchend', ptrTouch(100, 10 + dy));
  };
  return {
    ht, st, calls, sheet, panel, scroller, wrap, chip, note, detach, pull,
    release: () => { const h = held; held = null; if (h) h(); },
    host: () => hostHtml,
    renders: () => renderCount,
  };
}

// A. Pull at the top past threshold: exactly one fresh fetch, indicator
//    pins while it runs, no double-fire, new play lands on top.
{
  const w = makePtrWorld('new');
  w.panel.fire('touchstart', ptrTouch(100, 10));
  const mv = ptrTouch(100, 150);
  w.panel.fire('touchmove', mv);
  assert.equal(mv.defaultPrevented, true, 'claimed pull preventDefaults the move');
  assert.equal(w.wrap.style.height, '56px', 'reveal follows pull / 2.5');
  assert.ok(w.chip.classList.contains('is-armed'), 'armed at 140 >= 76');
  w.panel.fire('touchend', ptrTouch(100, 150));
  assert.deepEqual(w.calls.order, ['invalidate', 'get'], 'bypass first, then one fetch');
  assert.equal(w.calls.get, 1, 'exactly one fetch');
  assert.ok(w.chip.classList.contains('is-refreshing'), 'chip spins while refreshing');
  assert.equal(w.wrap.style.height, '52px', 'indicator pinned while refreshing');
  w.pull(140); // a second pull mid-refresh is inert
  assert.equal(w.calls.get, 1, 'no double-fire while a refresh is in flight');
  w.release();
  assert.equal(w.renders(), 1, 'plays re-rendered once');
  assert.ok(w.host().startsWith('13,'), 'new play renders on top: ' + w.host());
  assert.equal(w.wrap.style.height, '0px', 'indicator settled');
  assert.ok(!w.chip.classList.contains('is-refreshing'), 'spin cleared after settle');
  assert.equal(w.note.hidden, true, 'no failure note on success');
  w.detach();
  w.pull(140);
  assert.equal(w.calls.get, 1, 'detached gesture is gone with the sheet');
  console.log('ok - pull at top past threshold = one fresh fetch, new play on top');
}

// B. Short pull: indicator tracks, release fetches nothing.
{
  const w = makePtrWorld('new');
  w.panel.fire('touchstart', ptrTouch(100, 10));
  w.panel.fire('touchmove', ptrTouch(100, 60)); // dy 50 < 76
  assert.equal(w.wrap.style.height, '20px', 'short pull still reveals the chip');
  assert.ok(!w.chip.classList.contains('is-armed'), 'short pull not armed');
  w.panel.fire('touchend', ptrTouch(100, 60));
  assert.equal(w.calls.get, 0, 'short pull: no fetch');
  assert.equal(w.wrap.style.height, '0px', 'short pull: indicator retracted');
  console.log('ok - short pull = no fetch');
}

// C. Pull begun while the sheet body is scrolled down: not a candidate.
{
  const w = makePtrWorld('new');
  w.scroller.scrollTop = 120;
  w.pull(140);
  assert.equal(w.calls.get, 0, 'scrolled: no fetch');
  assert.equal(w.wrap.style.height, undefined, 'scrolled: indicator never sized');
  console.log('ok - pull while scrollTop > 0 = no gesture');
}

// D. The panel scrolls mid-gesture: the pull is handed back, no fetch.
{
  const w = makePtrWorld('new');
  w.panel.fire('touchstart', ptrTouch(100, 10));
  w.panel.fire('touchmove', ptrTouch(100, 50)); // claimed, dy 40
  assert.equal(w.wrap.style.height, '16px', 'pull claimed before the scroll');
  w.scroller.scrollTop = 40;
  w.panel.fire('touchmove', ptrTouch(100, 90));
  w.panel.fire('touchend', ptrTouch(100, 90));
  assert.equal(w.calls.get, 0, 'panel scrolled mid-gesture: no fetch');
  assert.equal(w.wrap.style.height, '0px', 'indicator collapsed on hand-back');
  console.log('ok - panel scrolls mid-gesture = hand back, no fetch');
}

// E. Stats tab active: the gesture is inert.
{
  const w = makePtrWorld('new');
  w.sheet.tab = 'stats';
  w.pull(140);
  assert.equal(w.calls.get, 0, 'stats tab: no fetch');
  assert.equal(w.wrap.style.height, undefined, 'stats tab: indicator never sized');
  console.log('ok - stats tab = gesture inert');
}

// F. Dropped fetch (network failure): the timer settles the pull as
//    failed; the rendered list is byte-identical and the note shows.
{
  const w = makePtrWorld('drop');
  const before = w.host();
  w.pull(140);
  assert.equal(w.calls.get, 1, 'dropped fetch: the fetch was attempted');
  assert.ok(w.chip.classList.contains('is-refreshing'), 'spinning while waiting');
  w.st.fireAll(); // the refresh's own 15s guard, as in production
  assert.equal(w.renders(), 0, 'failed fetch: no re-render');
  assert.equal(w.host(), before, 'failed fetch: list byte-identical');
  assert.equal(w.note.hidden, false, 'failed fetch: quiet note shown');
  assert.ok(w.note.textContent.includes('Could not refresh plays'), 'note copy: ' + w.note.textContent);
  assert.equal(w.wrap.style.height, '0px', 'failed fetch: indicator settled');
  assert.ok(!w.chip.classList.contains('is-refreshing'), 'failed fetch: spin cleared');
  console.log('ok - failed fetch = list unchanged, note shown, indicator settled');
}

// G. Same-reference flush (304 / stale body): counts as current, the
//    list is left exactly as rendered, no note.
{
  const w = makePtrWorld('same');
  const before = w.host();
  w.pull(140);
  w.release();
  assert.equal(w.calls.get, 1, 'same-reference: fetch attempted');
  assert.equal(w.renders(), 0, 'same-reference: no re-render');
  assert.equal(w.host(), before, 'same-reference: list unchanged');
  assert.equal(w.note.hidden, true, 'same-reference: no failure note');
  assert.equal(w.wrap.style.height, '0px', 'same-reference: indicator settled');
  console.log('ok - same-reference flush = current, list untouched');
}

// H. Horizontal drag: never a pull.
{
  const w = makePtrWorld('new');
  w.panel.fire('touchstart', ptrTouch(100, 10));
  w.panel.fire('touchmove', ptrTouch(190, 40)); // dx 90 > dy 30
  w.panel.fire('touchmove', ptrTouch(190, 160)); // same touch, now vertical: too late
  w.panel.fire('touchend', ptrTouch(190, 160));
  assert.equal(w.calls.get, 0, 'horizontal drag: no fetch');
  assert.equal(w.wrap.style.height, undefined, 'horizontal drag: indicator never sized');
  console.log('ok - horizontal drag = no pull');
}

// I. Upward reversal mid-pull: the pull zeroes without preventDefault,
//    and a later partial re-pull below threshold still fetches nothing.
{
  const w = makePtrWorld('new');
  w.panel.fire('touchstart', ptrTouch(100, 100));
  w.panel.fire('touchmove', ptrTouch(100, 200)); // dy 100, claimed
  assert.equal(w.wrap.style.height, '40px', 'pull revealed before reversal');
  const up = ptrTouch(100, 80); // dy -20: reversal
  w.panel.fire('touchmove', up);
  assert.equal(up.defaultPrevented, false, 'reversal move is not preventDefaulted');
  assert.equal(w.wrap.style.height, '0px', 'reversal collapses the indicator');
  w.panel.fire('touchmove', ptrTouch(100, 130)); // dy 30: re-engages below threshold
  assert.equal(w.wrap.style.height, '12px', 're-pull reveals again');
  w.panel.fire('touchend', ptrTouch(100, 130));
  assert.equal(w.calls.get, 0, 'reversed then short: no fetch');
  console.log('ok - upward reversal = hands off, no fetch');
}

// ── Swipe-down-to-dismiss ────────────────────────────────────────────
// The sheet's drag (/* gs-drag:start */ block, extracted verbatim) is
// the assistant sheet's grab-bar drag carried by cfg: the handle and
// head always own the gesture, the board area joins only at scroll
// top, the sheet follows the finger, release past 120px or a fast
// downward flick dismisses, anything less springs back. Cases: a long
// drag dismisses; a short slow drag springs back and cleans its inline
// styles when the glide lands; a flick dismisses below the distance
// threshold; an upward drag never moves the sheet; the board zone is
// inert while scrolled and live at the top; a disabled gate (desktop
// modal widths), a drag starting on a button, a non-primary button,
// and a detached binding are all inert; reduced motion snaps back
// with no spring timer.

const dStart = SRC.indexOf('/* gs-drag:start */');
const dEnd = SRC.indexOf('/* gs-drag:end */');
assert.ok(dStart >= 0 && dEnd > dStart, 'gs-drag delimiters present in app.js');
const DRAG_BLOCK = SRC.slice(dStart, dEnd);

function makeDragWorld(opts = {}) {
  const timers = makeTimers();
  let now = 0;
  const fakeDate = { now: () => now };
  const dragWin = {};
  new Function('window', 'setTimeout', 'clearTimeout', 'Date', DRAG_BLOCK)(
    dragWin, timers.setTimeout, timers.clearTimeout, fakeDate);
  assert.equal(typeof dragWin.__brGsAttachDrag, 'function', '__brGsAttachDrag exported');
  const sheetEl = new PtrEl();
  const scroller = new PtrEl();
  const handle = new PtrEl();
  const head = new PtrEl();
  const board = new PtrEl();
  const state = { enabled: !opts || opts.enabled !== false, reduced: !!(opts && opts.reduced) };
  let dismissals = 0;
  const detach = dragWin.__brGsAttachDrag({
    sheet: sheetEl,
    scroller,
    zones: [{ el: handle }, { el: head }, { el: board, needsTop: true }],
    enabled: () => state.enabled,
    reduceMotion: () => state.reduced,
    dismiss: () => { dismissals++; },
  });
  const evt = (y, o = {}) => ({
    pointerId: 1, clientY: y, button: 0,
    target: { closest: () => null },
    ...o,
  });
  return {
    timers, sheetEl, scroller, handle, head, board, detach, evt,
    advance: (ms) => { now += ms; },
    dismissals: () => dismissals,
  };
}

// J. Long drag on the handle: the sheet follows the finger, release
//    past 120px dismisses exactly once.
{
  const w = makeDragWorld();
  w.handle.fire('pointerdown', w.evt(10));
  assert.equal(w.sheetEl.style.transition, 'none', 'drag kills the sheet transition');
  w.advance(100);
  w.handle.fire('pointermove', w.evt(160));
  assert.equal(w.sheetEl.style.transform, 'translateY(150px)', 'sheet follows the finger');
  w.advance(10);
  w.handle.fire('pointerup', w.evt(160));
  assert.equal(w.dismissals(), 1, 'release past 120px dismisses');
  w.handle.fire('pointerup', w.evt(160));
  assert.equal(w.dismissals(), 1, 'no double dismiss');
  console.log('ok - drag past 120px = dismiss, sheet follows the finger');
}

// K. Short slow drag: springs back on the glide curve, inline styles
//    cleaned when it lands.
{
  const w = makeDragWorld();
  w.head.fire('pointerdown', w.evt(10));
  w.advance(200);
  w.head.fire('pointermove', w.evt(70)); // dy 60 at 0.3 px/ms: no flick
  w.advance(10);
  w.head.fire('pointerup', w.evt(70));
  assert.equal(w.dismissals(), 0, 'short slow drag: no dismiss');
  assert.equal(w.sheetEl.style.transform, 'translateY(0px)', 'spring drives the sheet home');
  assert.ok(w.sheetEl.style.transition.includes('cubic-bezier(0.22, 1, 0.36, 1)'),
    'spring uses the glide curve: ' + w.sheetEl.style.transition);
  w.timers.fireAll();
  assert.equal(w.sheetEl.style.transform, '', 'spring cleanup clears the transform');
  assert.equal(w.sheetEl.style.transition, '', 'spring cleanup clears the transition');
  console.log('ok - short slow drag = spring back, styles cleaned');
}

// L. Fast downward flick below the distance threshold dismisses.
{
  const w = makeDragWorld();
  w.handle.fire('pointerdown', w.evt(10));
  w.advance(50);
  w.handle.fire('pointermove', w.evt(35)); // dy 25
  w.advance(10);
  w.handle.fire('pointermove', w.evt(60)); // dy 50, vel 2.5 px/ms
  w.handle.fire('pointerup', w.evt(60));
  assert.equal(w.dismissals(), 1, 'flick past 40px at velocity dismisses');
  console.log('ok - downward flick = dismiss below the distance threshold');
}

// M. Upward drag: dy clamps at 0, the sheet never moves, release is a
//    plain cleanup with no dismiss and no spring.
{
  const w = makeDragWorld();
  w.handle.fire('pointerdown', w.evt(100));
  w.advance(50);
  w.handle.fire('pointermove', w.evt(40));
  assert.equal(w.sheetEl.style.transform, 'translateY(0px)', 'upward drag clamps at 0');
  w.handle.fire('pointerup', w.evt(40));
  assert.equal(w.dismissals(), 0, 'upward drag: no dismiss');
  assert.equal(w.sheetEl.style.transform, '', 'upward drag: styles cleaned at once');
  assert.equal(w.timers.pendingCount(), 0, 'upward drag: no spring timer');
  console.log('ok - upward drag = sheet stays, no dismiss');
}

// N. Board zone while the body is scrolled: not a drag zone at all.
{
  const w = makeDragWorld();
  w.scroller.scrollTop = 80;
  w.board.fire('pointerdown', w.evt(10));
  w.advance(50);
  w.board.fire('pointermove', w.evt(200));
  w.board.fire('pointerup', w.evt(200));
  assert.equal(w.dismissals(), 0, 'board drag while scrolled: no dismiss');
  assert.equal(w.sheetEl.style.transform, undefined, 'board drag while scrolled: sheet never moves');
  console.log('ok - board zone while scrolled = inert');
}

// O. Board zone at scroll top: a full drag dismisses.
{
  const w = makeDragWorld();
  w.scroller.scrollTop = 0;
  w.board.fire('pointerdown', w.evt(10));
  w.advance(100);
  w.board.fire('pointermove', w.evt(170));
  assert.equal(w.sheetEl.style.transform, 'translateY(160px)', 'board drag moves the sheet at top');
  w.board.fire('pointerup', w.evt(170));
  assert.equal(w.dismissals(), 1, 'board drag at top dismisses');
  console.log('ok - board zone at scroll top = drag dismisses');
}

// P. Disabled gate (desktop modal widths): every zone is inert.
{
  const w = makeDragWorld({ enabled: false });
  w.handle.fire('pointerdown', w.evt(10));
  w.advance(50);
  w.handle.fire('pointermove', w.evt(200));
  w.handle.fire('pointerup', w.evt(200));
  assert.equal(w.dismissals(), 0, 'disabled: no dismiss');
  assert.equal(w.sheetEl.style.transform, undefined, 'disabled: sheet never moves');
  console.log('ok - disabled gate = drag inert');
}

// Q. A drag starting on a button (the head's close button) or with a
//    non-primary button never starts.
{
  const w = makeDragWorld();
  const onButton = w.evt(10, { target: { closest: () => ({}) } });
  w.head.fire('pointerdown', onButton);
  w.advance(50);
  w.head.fire('pointermove', w.evt(200));
  w.head.fire('pointerup', w.evt(200));
  assert.equal(w.dismissals(), 0, 'button start: no dismiss');
  assert.equal(w.sheetEl.style.transform, undefined, 'button start: sheet never moves');
  w.handle.fire('pointerdown', w.evt(10, { button: 2 }));
  w.advance(50);
  w.handle.fire('pointermove', w.evt(200));
  w.handle.fire('pointerup', w.evt(200));
  assert.equal(w.dismissals(), 0, 'right button: no dismiss');
  assert.equal(w.sheetEl.style.transform, undefined, 'right button: sheet never moves');
  console.log('ok - button start and non-primary button = no drag');
}

// R. Reduced motion: a short drag snaps straight back, no spring.
{
  const w = makeDragWorld({ reduced: true });
  w.handle.fire('pointerdown', w.evt(10));
  w.advance(200);
  w.handle.fire('pointermove', w.evt(70));
  w.handle.fire('pointerup', w.evt(70));
  assert.equal(w.dismissals(), 0, 'reduced motion short drag: no dismiss');
  assert.equal(w.sheetEl.style.transform, '', 'reduced motion: snapped back at once');
  assert.equal(w.timers.pendingCount(), 0, 'reduced motion: no spring timer');
  console.log('ok - reduced motion = snap back, no spring');
}

// S. Detached binding (sheet closed): the gesture is gone.
{
  const w = makeDragWorld();
  w.detach();
  w.handle.fire('pointerdown', w.evt(10));
  w.advance(50);
  w.handle.fire('pointermove', w.evt(200));
  w.handle.fire('pointerup', w.evt(200));
  assert.equal(w.dismissals(), 0, 'detached: no dismiss');
  assert.equal(w.sheetEl.style.transform, undefined, 'detached: sheet never moves');
  console.log('ok - detached binding = gesture gone');
}

// ── Stat-row / drive-bar taps open the game sheet ────────────────────
// The matchup board's delegated capture-phase handler (in the "Matchup
// board live widgets" IIFE, extracted whole) resolves data-gs-team to a
// game from the lite games map and opens the REAL openGameSheet. The
// lite map is the full week's games, so a stat-row tap must open the
// sheet for a FINAL game too -- that is the only box-score path once
// the drive bar stops rendering. Driven against a hand-rolled DOM:
// _rzRenderGameBoard is a recording stub (the real board renderer is
// exercised by the ScoreZone harnesses); everything else in the open
// path -- handler, gameForTeam, sheet mount, box-score fetch, Stats /
// Plays renders -- is the shipped code.

const IIFE_START = SRC.indexOf('// ── Matchup board live widgets');
const IIFE_END = SRC.indexOf('/* ── Shared TD clip panels');
assert.ok(IIFE_START >= 0 && IIFE_END > IIFE_START, 'matchup live IIFE present in app.js');
const MB_IIFE = SRC.slice(IIFE_START, IIFE_END);

function tapMatches(el, sel) {
  let rest = sel.trim();
  const tagM = rest.match(/^[a-zA-Z][\w-]*/);
  if (tagM) {
    if (el.tagName !== tagM[0].toUpperCase()) return false;
    rest = rest.slice(tagM[0].length);
  }
  while (rest) {
    let m;
    if ((m = rest.match(/^\.([\w-]+)/))) {
      if (!el._classes.has(m[1])) return false;
    } else if ((m = rest.match(/^\[([\w-]+)(?:="([^"]*)")?\]/))) {
      const v = el.getAttribute(m[1]);
      if (v === null) return false;
      if (m[2] !== undefined && v !== m[2]) return false;
    } else return false;
    rest = rest.slice(m[0].length);
  }
  return true;
}

class TapEl {
  constructor(tag) {
    this.tagName = String(tag || 'div').toUpperCase();
    this._classes = new Set();
    this._attrs = {};
    this._listeners = {};
    this.children = [];
    this.parentNode = null;
    this.style = {};
    this.hidden = false;
    this.scrollTop = 0;
    this._innerHTML = '';
    this._text = '';
    this._qs = {};
  }
  get className() { return [...this._classes].join(' '); }
  set className(v) { this._classes = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get classList() {
    const s = this._classes;
    return {
      add: (...cs) => cs.forEach((c) => s.add(c)),
      remove: (...cs) => cs.forEach((c) => s.delete(c)),
      toggle: (c, force) => {
        const on = force === undefined ? !s.has(c) : !!force;
        if (on) s.add(c); else s.delete(c);
        return on;
      },
      contains: (c) => s.has(c),
    };
  }
  setAttribute(k, v) { this._attrs[k] = String(v); }
  getAttribute(k) { return k in this._attrs ? this._attrs[k] : null; }
  get innerHTML() { return this._innerHTML; }
  set innerHTML(v) { this._innerHTML = String(v); }
  get textContent() { return this._text; }
  set textContent(v) {
    this._text = String(v);
    this._innerHTML = this._text.replace(/[&<>"]/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }
  appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
  removeChild(child) {
    const i = this.children.indexOf(child);
    if (i >= 0) this.children.splice(i, 1);
    child.parentNode = null;
    return child;
  }
  addEventListener(type, fn) { (this._listeners[type] = this._listeners[type] || []).push(fn); }
  removeEventListener(type, fn) {
    const a = this._listeners[type] || [];
    const i = a.indexOf(fn);
    if (i >= 0) a.splice(i, 1);
  }
  querySelector(sel) {
    if (!this._qs[sel]) this._qs[sel] = new TapEl('div');
    return this._qs[sel];
  }
  querySelectorAll() { return []; }
  closest(selList) {
    let el = this;
    const sels = String(selList).split(',').map((s) => s.trim()).filter(Boolean);
    while (el) {
      if (el instanceof TapEl && sels.some((s) => tapMatches(el, s))) return el;
      el = el.parentNode;
    }
    return null;
  }
  contains(other) {
    let el = other;
    while (el) { if (el === this) return true; el = el.parentNode; }
    return false;
  }
}

function makeTapWorld() {
  const lite = {
    games: {
      'g-live': { game_id: 'g-live', away: 'IND', home: 'MIA', status: 'live', away_pts: 14, home_pts: 10, possession: 'IND', down: '3', distance: '15' },
      'g-final': { game_id: 'g-final', away: 'CAR', home: 'GB', status: 'final', away_pts: 17, home_pts: 24 },
    },
  };
  const full = {
    scoring, player_info: state.player_info, games: lite.games,
    pbp_by_game: { 'g-final': state.pbp_by_game['CAR@GB'], 'g-live': [] },
  };
  const boxFor = (gid) => (gid === 'g-final' ? {
    away: { team: 'CAR' }, home: { team: 'GB' }, view_team: 'CAR',
    teams: {
      CAR: { groups: [{ pos: 'RB',
        columns: [{ key: 'rush_yds', label: 'RuYd' }, { key: 'rush_td', label: 'RuTD' }, { key: 'rec', label: 'Rec' }],
        players: [{ id: '3', name: 'Chuba Hubbard', cells: { rush_yds: 96, rush_td: 1, rec: 3, rec_yds: 18 } }] }] },
      GB: { groups: [] },
    },
  } : {
    away: { team: 'IND' }, home: { team: 'MIA' }, view_team: 'IND',
    teams: {
      IND: { groups: [{ pos: 'RB',
        columns: [{ key: 'rush_yds', label: 'RuYd' }],
        players: [{ id: '31', name: 'Jonathan Taylor', cells: { rush_yds: 88, rush_td: 1 } }] }] },
      MIA: { groups: [] },
    },
  });
  const boardCalls = [];
  const fetches = [];
  const doc = {
    _listeners: {},
    hidden: false,
    body: new TapEl('body'),
    addEventListener(type, fn) { (this._listeners[type] = this._listeners[type] || []).push(fn); },
    removeEventListener(type, fn) {
      const a = this._listeners[type] || [];
      const i = a.indexOf(fn);
      if (i >= 0) a.splice(i, 1);
    },
    createElement: (tag) => new TapEl(tag),
    querySelector: () => null,
    fire(type, ev) { (this._listeners[type] || []).slice().forEach((fn) => fn(ev)); },
  };
  const win = {
    _rzGameEvents: window._rzGameEvents,
    _rzRenderGameBoard: (g, opts) => {
      boardCalls.push({ game: g, opts });
      return 'BOARD ' + g.away + ' ' + g.away_pts + ' @ ' + g.home + ' ' + g.home_pts + ' [' + g.status + ']';
    },
    _rzFieldPosition: () => ({ spot: 46, side: 'away', label: 'MIA 46', ltg: 61, goalToGo: false }),
    __brSzLiveCache: {
      get(variant, ms, cb) { const s = variant === 'lite' ? lite : full; if (cb) cb(s); return s; },
      invalidate() {},
      peek() { return full; },
    },
    location: { pathname: '/hub/league-9/matchups' },
  };
  const fetchStub = (url) => {
    fetches.push(String(url));
    const gid = decodeURIComponent(String(url).split('game_id=')[1] || '');
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(boxFor(gid)) });
  };
  new Function('window', 'document', 'fetch', 'requestAnimationFrame', 'setInterval', 'clearInterval',
    MB_IIFE)(win, doc, fetchStub, (fn) => { fn(); return 1; }, () => 1, () => {});
  const slide = new TapEl('div');
  slide.querySelector = (sel) => (sel === '.mb-fld, .mb-moments' ? new TapEl('div') : null);
  slide.querySelectorAll = () => [];
  return { win, doc, boardCalls, fetches, slide };
}

function tapEl(classes, attrs, parent) {
  const el = new TapEl('div');
  (classes || []).forEach((c) => el._classes.add(c));
  Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, v));
  if (parent) parent.appendChild(el);
  return el;
}
function clickEv(target) {
  return { target, _stopped: false, _prevented: false,
    stopPropagation() { this._stopped = true; },
    preventDefault() { this._prevented = true; } };
}
function keyEv(target, key) {
  return { target, key, _stopped: false, _prevented: false,
    stopPropagation() { this._stopped = true; },
    preventDefault() { this._prevented = true; } };
}
const sheetOf = (w) => w.doc.body.children.find((c) => c._classes.has('rz-bs-sheet'));
const flushMicro = async () => {
  await new Promise((r) => setImmediate(r));
  await new Promise((r) => setImmediate(r));
};

async function tapScenarios() {
  // J. Stat-row tap on a LIVE game opens the sheet for that game, and
  //    the tap is claimed (the row's own handlers never also fire).
  {
    const w = makeTapWorld();
    w.win.brInitMatchupLive(w.slide); // paints: lastState = lite payload
    const row = tapEl(['mb-stat-row', 'mb-fld-span-l'],
      { 'data-gs-team': 'IND', role: 'button', tabindex: '0' });
    const inner = tapEl(['meta', 'm-cell-stats', 'mb-stat'], {}, row);
    const ev = clickEv(inner);
    w.doc.fire('click', ev);
    assert.equal(ev._stopped, true, 'live stat tap: propagation stopped');
    assert.equal(ev._prevented, true, 'live stat tap: default prevented');
    assert.equal(w.boardCalls[w.boardCalls.length - 1].game.game_id, 'g-live',
      'live stat tap opens the live game');
    const sheet = sheetOf(w);
    assert.ok(sheet, 'live stat tap: sheet mounted');
    assert.ok(sheet.querySelector('.gs-board-wrap').innerHTML.includes('[live]'),
      'live board rendered in the sheet');
    assert.ok(w.fetches.some((u) => u.includes('game_id=g-live')),
      'box score fetched for the live game');
    await flushMicro();
    assert.ok(sheet.querySelector('[data-gpanel="stats"]').innerHTML.includes('Jonathan Taylor'),
      'live stats tab shows the box score');
    console.log('ok - stat-row tap on a live game opens the sheet + box score');
  }

  // K. Stat-row tap on a FINAL game: the sheet opens in final state and
  //    the Stats tab renders the real (fetched) box score -- the whole
  //    point of the feature, since finals render no drive bar.
  {
    const w = makeTapWorld();
    w.win.brInitMatchupLive(w.slide);
    const row = tapEl(['mb-stat-row', 'mb-fld-span-r', 'mb-cell-r'],
      { 'data-gs-team': 'CAR', role: 'button', tabindex: '0' });
    const inner = tapEl(['meta', 'm-cell-stats', 'mb-stat'], {}, row);
    const ev = clickEv(inner);
    w.doc.fire('click', ev);
    assert.equal(ev._prevented, true, 'final stat tap: default prevented');
    const call = w.boardCalls[w.boardCalls.length - 1];
    assert.equal(call.game.game_id, 'g-final', 'final stat tap resolves the final game');
    assert.equal(call.game.status, 'final', 'resolved game is final');
    const sheet = sheetOf(w);
    assert.ok(sheet, 'final stat tap: sheet mounted');
    const boardHtml = sheet.querySelector('.gs-board-wrap').innerHTML;
    assert.ok(boardHtml.includes('[final]'), 'board renders the final state: ' + boardHtml);
    assert.ok(boardHtml.includes('CAR 17') && boardHtml.includes('GB 24'),
      'board carries the final score: ' + boardHtml);
    await flushMicro();
    const statsHtml = sheet.querySelector('[data-gpanel="stats"]').innerHTML;
    assert.ok(statsHtml.includes('Player stats'), 'stats tab header renders');
    assert.ok(statsHtml.includes('Chuba Hubbard'), 'stats tab shows the real box score rows');
    assert.ok(statsHtml.includes('<td>96</td>'), 'box score stat cells render (96 rush yds)');
    assert.ok(statsHtml.includes('20.4'), 'box score fantasy points render (20.4 FP)');
    const playsHtml = sheet.querySelector('.gs-plays-host').innerHTML;
    assert.ok(playsHtml.includes('gs-plays'), 'plays tab renders the final game plays');
    console.log('ok - stat-row tap on a FINAL game opens the sheet: final board + real box score');
  }

  // L. Inert stat rows (the "Stats unavailable" marker, team-less rows:
  //    the server emits no data-gs-team for either) open nothing and the
  //    event is left completely alone.
  {
    const w = makeTapWorld();
    w.win.brInitMatchupLive(w.slide);
    const row = tapEl(['mb-stat-row', 'mb-fld-span-l'], {});
    const inner = tapEl(['meta', 'm-cell-stats', 'mb-stat', 'm-cell-stats--unavailable'], {}, row);
    const ev = clickEv(inner);
    w.doc.fire('click', ev);
    assert.equal(ev._stopped, false, 'inert stat row: event not stopped');
    assert.equal(ev._prevented, false, 'inert stat row: event not prevented');
    assert.equal(w.boardCalls.length, 0, 'inert stat row: no sheet');
    assert.equal(w.doc.body.children.length, 0, 'inert stat row: nothing mounted');
    assert.equal(w.fetches.length, 0, 'inert stat row: no fetch');
    console.log('ok - unavailable / team-less stat row stays inert');
  }

  // M. Drive-bar regression: the bar's own tap still opens its game.
  {
    const w = makeTapWorld();
    w.win.brInitMatchupLive(w.slide);
    const btn = tapEl(['mb-fld-link'], { 'data-gs-team': 'IND' });
    btn.tagName = 'BUTTON';
    const inner = tapEl(['mb-fld-in'], {}, btn);
    const ev = clickEv(inner);
    w.doc.fire('click', ev);
    assert.equal(ev._prevented, true, 'drive bar tap: default prevented');
    assert.equal(w.boardCalls[w.boardCalls.length - 1].game.game_id, 'g-live',
      'drive bar still opens its game');
    console.log('ok - drive-bar tap still opens the sheet (regression)');
  }

  // N. Keyboard: Enter / Space on a focused stat row opens the sheet;
  //    other keys and inert rows are untouched.
  {
    const w = makeTapWorld();
    w.win.brInitMatchupLive(w.slide);
    const row = tapEl(['mb-stat-row'], { 'data-gs-team': 'CAR', role: 'button', tabindex: '0' });
    let ev = keyEv(row, 'Enter');
    w.doc.fire('keydown', ev);
    assert.equal(ev._prevented, true, 'Enter on stat row: default prevented');
    assert.equal(w.boardCalls[w.boardCalls.length - 1].game.game_id, 'g-final',
      'Enter opens the sheet');
    ev = keyEv(row, ' ');
    w.doc.fire('keydown', ev);
    assert.equal(w.boardCalls[w.boardCalls.length - 1].game.game_id, 'g-final',
      'Space opens the sheet');
    const before = w.boardCalls.length;
    ev = keyEv(row, 'Tab');
    w.doc.fire('keydown', ev);
    assert.equal(ev._prevented, false, 'unrelated key left alone');
    assert.equal(w.boardCalls.length, before, 'unrelated key: no sheet');
    const inert = tapEl(['mb-stat-row'], {});
    ev = keyEv(inert, 'Enter');
    w.doc.fire('keydown', ev);
    assert.equal(ev._prevented, false, 'Enter on inert row left alone');
    assert.equal(w.boardCalls.length, before, 'inert row Enter: no sheet');
    console.log('ok - Enter/Space on a stat row opens the sheet; inert rows and other keys untouched');
  }

  // O. First tap before the lite payload has painted (lastState null):
  //    the handler's fetch fallback still resolves and opens, finals too.
  {
    const w = makeTapWorld();
    const row = tapEl(['mb-stat-row'], { 'data-gs-team': 'CAR' });
    const ev = clickEv(row);
    w.doc.fire('click', ev);
    assert.equal(w.boardCalls[w.boardCalls.length - 1].game.game_id, 'g-final',
      'cold stat tap fetches state, then opens the final game');
    console.log('ok - stat tap before first paint resolves through the fetch fallback');
  }
}

await tapScenarios();

console.log('ALL GAME SHEET EVENT HARNESS CHECKS PASSED');
