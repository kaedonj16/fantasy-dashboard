"""Site-wide animations: Wrapped count-ups, table sort transitions, tab panel
transitions, and the RedZone LIVE pulse (features 3-6).

Feature 3 (Wrapped number count-ups) was already wired on main: playSlide()
in the Wrapped bootstrap calls (window.brCountUp || wrappedCountUp) on every
slide open, for both Season and Weekly decks and public share pages. These
are regression contract tests, not a re-implementation.

Feature 4 (table row glide on sort): CSS transforms do not animate <tr>/<td>,
so the FLIP helper cannot glide real table rows. window.brTableSortSwap
instead tints climbers (rk-up) / fallers (rk-down) -- background + opacity,
which do apply to <tr> -- and fades brand-new rows in. Wired into the two
live client-side table sorts: Advanced Metrics and NFL Teams.

Feature 5 (tab transitions): window.brAnimateTabPanel slides the incoming
panel in from the travel direction. Wired into every tab switcher:
initCardTabs (standings/history/teams/weekly-hub/power cards), tmSwitchTab,
cmpSwitchTab, cmp3SwitchTab, pmSwitchTab, and the League Scores tabs.

Feature 6 (RedZone LIVE pulse): the scoring-play slide-ins already existed
(rz-live-enter); this adds the breathing dot to the text-only LIVE badges and
kills every infinite live pulse under prefers-reduced-motion (they were not
covered before).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")

_MOTION_START = "// ── Motion layer ──"


def _extract_motion_iife() -> str:
    lines = open(os.path.join(_ROOT, "static", "app.js"), encoding="utf-8").read().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if _MOTION_START in l)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip() == "})();")
    return "".join(lines[start : end + 1])


_NODE_HARNESS = r"""
const fs = require('fs');
const iifeSrc = fs.readFileSync(process.argv[2], 'utf8');
const scenario = JSON.parse(process.argv[3]);

const window = {
  matchMedia: () => ({ matches: !!scenario.reduce }),
  IntersectionObserver: undefined,
};
global.window = window;
global.performance = { now: () => 0 };
global.requestAnimationFrame = (cb) => { cb(); return 1; };
global.document = {
  readyState: 'complete',
  addEventListener() {},
  querySelector() { return null; },
  querySelectorAll() { return []; },
  getElementById() { return null; },
  createElement() { return null; },
};
global.sessionStorage = { getItem: () => null, setItem() {} };

function makeClassList() {
  const s = new Set();
  return {
    add(c) { s.add(c); },
    remove(...cs) { cs.forEach(c => s.delete(c)); },
    contains(c) { return s.has(c); },
    toggle(c, f) { if (f === undefined) f = !s.has(c); f ? s.add(c) : s.delete(c); },
  };
}

function makeTable(keys, attr) {
  const table = {
    _attr: attr,
    _html: '',
    _rows: [],
    querySelectorAll(sel) {
      if (sel === 'tbody tr[' + this._attr + ']') return this._rows.slice();
      return [];
    },
  };
  Object.defineProperty(table, 'innerHTML', {
    set(html) {
      this._html = html;
      const re = new RegExp(this._attr.replace(/-/g, '\\-') + '="([^"]+)"', 'g');
      const found = [];
      let m;
      while ((m = re.exec(html))) found.push(m[1]);
      const tbody = { children: [] };
      const self = this;
      this._rows = found.map(k => ({
        _key: k,
        classList: makeClassList(),
        style: {},
        parentNode: tbody,
        listeners: {},
        getAttribute(n) { return n === self._attr ? k : null; },
        addEventListener(t, cb) { this.listeners[t] = cb; },
      }));
      tbody.children = this._rows;
    },
    get() { return this._html; },
  });
  table.innerHTML = keys.map(k => '<tr ' + attr + '="' + k + '"><td>x</td></tr>').join('');
  return table;
}

function makePanel() {
  return {
    classList: makeClassList(),
    offsetWidth: 100,
    listeners: {},
    addEventListener(t, cb) { this.listeners[t] = cb; },
  };
}

eval(iifeSrc);

const out = {};
if (scenario.case === 'sort') {
  const table = makeTable(scenario.keys, scenario.attr);
  window.brTableSortSwap(table, scenario.html, scenario.attr);
  out.rows = table._rows.map(r => ({
    key: r._key,
    up: r.classList.contains('rk-up'),
    down: r.classList.contains('rk-down'),
    fade: !!r.style.animation,
  }));
  out.htmlSet = table._html === scenario.html;
} else if (scenario.case === 'tab') {
  const panel = makePanel();
  window.brAnimateTabPanel(panel, scenario.dir);
  out.cls = panel.classList.contains('br-tab-in-r') ? 'r'
    : (panel.classList.contains('br-tab-in-l') ? 'l' : 'none');
  if (panel.listeners.animationend) panel.listeners.animationend();
  out.after = (panel.classList.contains('br-tab-in-r') || panel.classList.contains('br-tab-in-l'))
    ? 'kept' : 'cleaned';
}
console.log(JSON.stringify(out));
"""


def _run_scenario(scenario: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(_extract_motion_iife())
        iife_path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(_NODE_HARNESS)
        harness_path = f.name
    try:
        proc = subprocess.run(
            ["node", harness_path, iife_path, json.dumps(scenario)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_ROOT,
        )
    finally:
        os.unlink(iife_path)
        os.unlink(harness_path)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr[-2000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _rows_html(keys, attr="data-rk-key"):
    return "".join(f'<tr {attr}="{k}"><td>{k}</td></tr>' for k in keys)


# ---------------------------------------------------------------------------
# Feature 4: brTableSortSwap
# ---------------------------------------------------------------------------

def test_table_sort_swap_tints_climbers_and_fallers():
    res = _run_scenario({
        "case": "sort",
        "keys": ["a", "b", "c"],
        "html": _rows_html(["b", "c", "a"]),
        "attr": "data-rk-key",
    })
    assert res["htmlSet"] is True
    by_key = {r["key"]: r for r in res["rows"]}
    assert [r["key"] for r in res["rows"]] == ["b", "c", "a"]
    assert by_key["b"]["up"] is True and by_key["b"]["down"] is False
    assert by_key["c"]["up"] is True and by_key["c"]["down"] is False
    assert by_key["a"]["down"] is True and by_key["a"]["up"] is False


def test_table_sort_swap_fades_new_rows_and_skips_stationary():
    res = _run_scenario({
        "case": "sort",
        "keys": ["a", "b"],
        "html": _rows_html(["b", "d", "a"]),
        "attr": "data-rk-key",
    })
    by_key = {r["key"]: r for r in res["rows"]}
    assert by_key["d"]["fade"] is True
    assert by_key["d"]["up"] is False and by_key["d"]["down"] is False
    assert by_key["b"]["up"] is True
    assert by_key["a"]["down"] is True


def test_table_sort_swap_reduced_motion_plain_swap():
    res = _run_scenario({
        "case": "sort",
        "keys": ["a", "b", "c"],
        "html": _rows_html(["c", "b", "a"]),
        "attr": "data-rk-key",
        "reduce": True,
    })
    assert res["htmlSet"] is True
    for r in res["rows"]:
        assert r["up"] is False and r["down"] is False and r["fade"] is False


def test_table_sort_swap_uses_data_abbr_for_nfl_teams():
    res = _run_scenario({
        "case": "sort",
        "keys": ["KC", "BUF"],
        "html": _rows_html(["BUF", "KC"], attr="data-abbr"),
        "attr": "data-abbr",
    })
    by_key = {r["key"]: r for r in res["rows"]}
    assert by_key["BUF"]["up"] is True
    assert by_key["KC"]["down"] is True


# ---------------------------------------------------------------------------
# Feature 5: brAnimateTabPanel
# ---------------------------------------------------------------------------

def test_tab_panel_slides_from_travel_direction():
    fwd = _run_scenario({"case": "tab", "dir": 2})
    back = _run_scenario({"case": "tab", "dir": -1})
    assert fwd["cls"] == "r"
    assert back["cls"] == "l"


def test_tab_panel_class_cleaned_after_animation():
    res = _run_scenario({"case": "tab", "dir": 1})
    assert res["after"] == "cleaned"


def test_tab_panel_reduced_motion_no_animation():
    res = _run_scenario({"case": "tab", "dir": 1, "reduce": True})
    assert res["cls"] == "none"


# ---------------------------------------------------------------------------
# Wiring contracts: every tab switcher animates the incoming panel
# ---------------------------------------------------------------------------

def _app_js() -> str:
    return open(os.path.join(_ROOT, "static", "app.js"), encoding="utf-8").read()


def test_all_tab_switchers_call_br_animate_tab_panel():
    src = _app_js()
    for fn in ("function initCardTabs", "function tmSwitchTab", "function cmpSwitchTab",
               "function cmp3SwitchTab"):
        start = src.find(fn)
        assert start != -1, f"{fn} missing"
        # next function boundary: search a bounded window for the call
        window_src = src[start : start + 4000]
        assert "brAnimateTabPanel" in window_src, f"{fn} does not animate the panel"
    pm = open(os.path.join(_ROOT, "static", "player_modal.js"), encoding="utf-8").read()
    assert "brAnimateTabPanel" in pm, "pmSwitchTab does not animate the panel"
    # League Scores tab delegation (data-ls-tab click handler)
    assert "brAnimateTabPanel(lsIncoming" in src


def test_table_sort_swap_wired_into_both_table_sorts():
    am = open(os.path.join(_ROOT, "dashboard_services", "pages", "advanced_metrics_page.py"),
              encoding="utf-8").read()
    assert "brTableSortSwap(tbody, _rowsHtml, 'data-rk-key')" in am
    assert 'data-rk-key="' in am and "amRowKey(r)" in am
    nt = open(os.path.join(_ROOT, "dashboard_services", "pages", "nfl_teams_page.py"),
              encoding="utf-8").read()
    assert "brTableSortSwap(tbl,h,'data-abbr')" in nt


# ---------------------------------------------------------------------------
# Feature 3: Wrapped count-ups (already wired; regression contracts)
# ---------------------------------------------------------------------------

import ast as _ast


def _history_src() -> str:
    return open(
        os.path.join(_ROOT, "dashboard_services", "pages", "history_page.py"),
        encoding="utf-8",
    ).read()


def _wrapped_bootstrap_const() -> str:
    # history_page has a heavy import chain (openai, espn_api, ...) that is
    # not installed in every CI shard, so read the bootstrap JS via ast
    # instead of importing the module.
    tree = _ast.parse(_history_src())
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name) and t.id == "_WRAPPED_BOOTSTRAP_JS":
                    return _ast.literal_eval(node.value)
    raise AssertionError("_WRAPPED_BOOTSTRAP_JS not found")


def test_wrapped_markup_emits_countup_numbers():
    src = _history_src()
    # The num-slide branch renders the big number with its target value and
    # decimal places as data attributes for the count-up.
    assert "wrapped-big" in src
    assert "data-w-count=" in src
    assert "data-w-dp=" in src


def test_wrapped_bootstrap_counts_up_on_slide_open():
    js = _wrapped_bootstrap_const()
    assert "[data-w-count]" in js
    assert "window.brCountUp || wrappedCountUp" in js
    # playSlide() runs on every slide open, for both deck namespaces (the
    # namespacing in _wrapped_bootstrap_js only rewrites 'wrapped'+id element
    # ids, which the count-up lines do not contain).
    assert "function playSlide" in js
    src = _history_src()
    assert "def _wrapped_bootstrap_js" in src
    assert "def _wrapped_public_bootstrap_js" in src


def test_wrapped_countup_respects_reduced_motion():
    js = _wrapped_bootstrap_const()
    # The public-share fallback (used when app.js is absent) must snap too.
    assert "prefers-reduced-motion" in js


# ---------------------------------------------------------------------------
# Feature 6: LIVE pulse + feed slide-ins (CSS contracts)
# ---------------------------------------------------------------------------

def _css() -> str:
    return open(os.path.join(_ROOT, "static", "dashboard.css"), encoding="utf-8").read()


def test_live_badges_get_breathing_dot():
    css = _css()
    for sel in (".rz-mc-state.live::before", ".rz-mt-live::before", ".rz-lb-live::before"):
        assert sel in css, f"{sel} missing"
    assert "animation: rz-pulse 1.6s ease-in-out infinite" in css


def test_reduced_motion_kills_live_pulses_and_tab_transitions():
    css = _css()
    assert ".br-tab-in-l, .br-tab-in-r" in css
    for sel in (".rz-nav-dot", ".rz-brand-dot.is-live", ".rz-live-dot-sm", ".pm-live-dot",
                ".rz-psb-live-dot", ".rz-mnav-dot", ".rz-mc-state.live::before"):
        assert sel in css
    # The kill list lives inside a prefers-reduced-motion block.
    rm_start = css.find("@media (prefers-reduced-motion: reduce)")
    assert rm_start != -1
    tail = css[rm_start:]
    assert ".rz-nav-dot" in tail and "animation: none !important" in tail


def test_tab_panel_keyframes_exist():
    css = _css()
    assert "@keyframes brTabInR" in css
    assert "@keyframes brTabInL" in css
    assert ".br-tab-in-r" in css and ".br-tab-in-l" in css


def test_feed_slide_in_still_present():
    # Feature 6's scoring-play slide-ins predated this build; guard them.
    css = _css()
    assert "@keyframes rz-live-enter" in css
    assert ".rz-event.is-live-enter" in css
    sz = open(os.path.join(_ROOT, "static", "scorezone.js"), encoding="utf-8").read()
    assert "is-live-enter" in sz
