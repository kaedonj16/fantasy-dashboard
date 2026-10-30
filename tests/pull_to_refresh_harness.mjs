// Behavioral harness for the pull-to-refresh controller in static/app.js.
// Extracts the REAL controller block (delimited by /* br-ptr:start */ and
// /* br-ptr:end */) and drives it with fake DOM/touch events: no hook = no
// gesture, below threshold = no refresh, above threshold at the top = the
// hook fires exactly once, scrolled down / modal open / inner-scroller
// target = no gesture, and no re-entry while a refresh is in flight.
//
// Run: node tests/pull_to_refresh_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'app.js'), 'utf8');

const start = SRC.indexOf('/* br-ptr:start */');
const end = SRC.indexOf('/* br-ptr:end */');
assert.ok(start >= 0 && end > start, 'br-ptr delimiters present in app.js');
const BLOCK = SRC.slice(start, end);

class El {
  constructor(tag) {
    this.tagName = String(tag || 'div').toUpperCase();
    this.children = [];
    this.className = '';
    this.style = {};
    this.parentElement = null;
    this.parentNode = null;
    this.scrollTop = 0;
    this.scrollLeft = 0;
    this.scrollHeight = 0;
    this.scrollWidth = 0;
    this.clientHeight = 0;
    this.clientWidth = 0;
    this.isContentEditable = false;
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
  setAttribute() {}
  appendChild(child) {
    child.parentElement = this;
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
}

function makeWorld() {
  const listeners = {};
  const doc = {
    body: new El('body'),
    _modal: null,
    addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
    createElement(tag) { return new El(tag); },
    querySelector() { return doc._modal; },
  };
  doc.body.parentElement = null;
  const win = {
    scrollY: 0,
    pageYOffset: 0,
    brPageRefresh: undefined,
    getComputedStyle: (node) => ({ overflowX: node._overflowX || '' }),
  };
  const run = new Function('window', 'document', 'setTimeout', 'clearTimeout', BLOCK);
  run(win, doc, setTimeout, clearTimeout);
  const fire = (type, ev) => (listeners[type] || []).forEach((fn) => fn(ev));
  const touch = (target, x, y) => ({
    target,
    touches: [{ clientX: x, clientY: y }],
    changedTouches: [{ clientX: x, clientY: y }],
    cancelable: true,
    preventDefault() { this.defaultPrevented = true; },
  });
  // One full pull gesture: start at (100, 10), move down dy, release.
  const pull = (target, dy) => {
    fire('touchstart', touch(target, 100, 10));
    fire('touchmove', touch(target, 100, 10 + dy));
    fire('touchend', touch(target, 100, 10 + dy));
  };
  const indicator = () => doc.body.children[0] || null;
  return { doc, win, fire, touch, pull, indicator };
}

const plainTarget = () => new El('div');
const flush = () => new Promise((r) => setImmediate(r));

async function run() {
  // 1. No hook registered: no gesture, no indicator, nothing fires.
  {
    const w = makeWorld();
    w.pull(plainTarget(), 140);
    assert.equal(w.indicator(), null, 'no hook: indicator never created');
    console.log('ok - no hook = no gesture, no indicator');
  }

  // 2. Below threshold: indicator shows mid-pull but release does not refresh.
  {
    const w = makeWorld();
    let calls = 0;
    w.win.brPageRefresh = () => { calls++; return Promise.resolve(); };
    const t = plainTarget();
    w.fire('touchstart', w.touch(t, 100, 10));
    w.fire('touchmove', w.touch(t, 100, 60)); // dy = 50 < 76
    const ind = w.indicator();
    assert.ok(ind, 'below threshold: indicator created mid-pull');
    assert.ok(ind.classList.contains('br-ptr-visible'), 'below threshold: indicator visible mid-pull');
    assert.ok(!ind.classList.contains('br-ptr-armed'), 'below threshold: not armed');
    w.fire('touchend', w.touch(t, 100, 60));
    await flush();
    assert.equal(calls, 0, 'below threshold: hook not called');
    assert.ok(!ind.classList.contains('br-ptr-visible'), 'below threshold: indicator retracted');
    console.log('ok - below threshold = indicator only, no refresh');
  }

  // 3. Above threshold at the top: hook fires exactly once, indicator pins
  //    while the promise is pending, then retracts when it settles.
  {
    const w = makeWorld();
    let calls = 0;
    let release = null;
    w.win.brPageRefresh = () => { calls++; return new Promise((r) => { release = r; }); };
    const t = plainTarget();
    w.fire('touchstart', w.touch(t, 100, 10));
    w.fire('touchmove', w.touch(t, 100, 150)); // dy = 140 >= 76
    const ind = w.indicator();
    assert.ok(ind.classList.contains('br-ptr-armed'), 'above threshold: armed mid-pull');
    w.fire('touchend', w.touch(t, 100, 150));
    assert.equal(calls, 1, 'above threshold: hook called once');
    assert.ok(ind.classList.contains('br-ptr-refreshing'), 'refreshing: indicator pinned');
    w.pull(t, 140); // a second full pull while refreshing must not re-enter
    assert.equal(calls, 1, 'refreshing: no re-entry');
    release();
    await flush();
    assert.ok(!ind.classList.contains('br-ptr-refreshing'), 'settled: refreshing class cleared');
    assert.ok(!ind.classList.contains('br-ptr-visible'), 'settled: indicator retracted');
    console.log('ok - above threshold = one refresh, pins until settle, retracts');
  }

  // 4. Not at the top: no gesture at all.
  {
    const w = makeWorld();
    let calls = 0;
    w.win.brPageRefresh = () => { calls++; return Promise.resolve(); };
    w.win.scrollY = 240;
    w.pull(plainTarget(), 140);
    await flush();
    assert.equal(calls, 0, 'scrolled down: hook not called');
    assert.equal(w.indicator(), null, 'scrolled down: no indicator');
    console.log('ok - not at top = no gesture');
  }

  // 4b. Scrolled in the <body> scroller (window.scrollY stays 0 in that
  //     render): still not the top, no gesture.
  {
    const w = makeWorld();
    let calls = 0;
    w.win.brPageRefresh = () => { calls++; return Promise.resolve(); };
    w.doc.body.scrollTop = 500;
    w.pull(plainTarget(), 140);
    await flush();
    assert.equal(calls, 0, 'body-scrolled: hook not called');
    assert.equal(w.indicator(), null, 'body-scrolled: no indicator');
    console.log('ok - body is the scroller and scrolled = no gesture');
  }

  // 5. Modal open: no gesture.
  {
    const w = makeWorld();
    let calls = 0;
    w.win.brPageRefresh = () => { calls++; return Promise.resolve(); };
    w.doc._modal = new El('div');
    w.pull(plainTarget(), 140);
    await flush();
    assert.equal(calls, 0, 'modal open: hook not called');
    assert.equal(w.indicator(), null, 'modal open: no indicator');
    console.log('ok - modal open = no gesture');
  }

  // 6. Target inside an inner scroller that can still scroll up: no gesture.
  {
    const w = makeWorld();
    let calls = 0;
    w.win.brPageRefresh = () => { calls++; return Promise.resolve(); };
    const scroller = new El('div');
    scroller.scrollTop = 60;
    scroller.scrollHeight = 600;
    scroller.clientHeight = 120;
    const child = new El('span');
    scroller.appendChild(child);
    w.pull(child, 140);
    await flush();
    assert.equal(calls, 0, 'inner scroller: hook not called');
    assert.equal(w.indicator(), null, 'inner scroller: no indicator');
    console.log('ok - inner-scroller target = no gesture');
  }

  // 7. Form field target: no gesture.
  {
    const w = makeWorld();
    let calls = 0;
    w.win.brPageRefresh = () => { calls++; return Promise.resolve(); };
    w.pull(new El('input'), 140);
    await flush();
    assert.equal(calls, 0, 'form field: hook not called');
    console.log('ok - form-field target = no gesture');
  }

  // 8. Horizontally dominant gesture: aborts, no refresh.
  {
    const w = makeWorld();
    let calls = 0;
    w.win.brPageRefresh = () => { calls++; return Promise.resolve(); };
    const t = plainTarget();
    w.fire('touchstart', w.touch(t, 100, 10));
    w.fire('touchmove', w.touch(t, 190, 40)); // dx 90 > dy 30
    w.fire('touchend', w.touch(t, 190, 40));
    await flush();
    assert.equal(calls, 0, 'horizontal gesture: hook not called');
    console.log('ok - horizontal gesture = aborted');
  }

  console.log('\nALL PULL-TO-REFRESH HARNESS CHECKS PASSED');
}

run().then(() => { process.exitCode = 0; }).catch((e) => { console.error(e); process.exitCode = 1; });
