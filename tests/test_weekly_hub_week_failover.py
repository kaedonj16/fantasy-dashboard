"""Weekly hub week selector: a failed week fetch must not leave the selector
disagreeing with the rendered content.

Regression test for the "selector says Week 4, matchup preview shows Week 3"
bug: the change handler used to leave the native select on the requested week
when its /api/weekly-week fetch failed, so the page silently displayed the
previous week's content under the new week's label.

The test extracts the week-change IIFE from dashboard_services/pages/
weekly_hub_page.py (de-templating the .format() escaping), drives it in Node
with a DOM shim, and simulates successful and failed week fetches.
"""
from __future__ import annotations

import json
import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAGE = os.path.join(_ROOT, "dashboard_services", "pages", "weekly_hub_page.py")


def _extract_week_change_iife() -> str:
    src = open(_PAGE, encoding="utf-8").read()
    start = src.index("<script>\n") + len("<script>\n")
    end = src.index("// \u2500\u2500 Weekly: game-day auto-refresh")
    block = src[start:end]
    iife_start = block.index("(function() {{")
    iife_end = block.index("}})();", iife_start) + len("}})();")
    iife = block[iife_start:iife_end]
    # Substitute the .format() placeholders, then un-double literal braces.
    iife = iife.replace("{league_js}", '"lid123"')
    iife = iife.replace("{platform_js}", '"sleeper"')
    iife = iife.replace("{season_js}", "2026")
    iife = iife.replace("{{", "{").replace("}}", "}")
    assert "{league_js}" not in iife and "{platform_js}" not in iife
    return iife


_NODE_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const scenario = JSON.parse(process.argv[3]); // {startWeek, steps:[{pick, mode}]}

const toasts = [];

function makeEl(id) {
  return {
    id: id || '',
    value: '',
    disabled: false,
    hidden: false,
    innerHTML: '',
    style: {},
    dataset: {},
    attributes: {},
    _listeners: {},
    getAttribute(n) { return this.attributes[n] || ''; },
    setAttribute(n, v) { this.attributes[n] = String(v); },
    addEventListener(t, fn) { this._listeners[t] = fn; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
  };
}

const select = makeEl('hubWeek');
select.value = scenario.startWeek;
const matchupsContainer = makeEl('weeklyMatchupsContainer');
const loadingOverlay = makeEl('weeklyMatchupsLoading');
const mainContainer = makeEl('mainPanels');
const sideContainer = makeEl('sidePanels');

global.window = global;
global.document = {
  getElementById(id) {
    if (id === 'hubWeek') return select;
    if (id === 'weeklyMatchupsContainer') return matchupsContainer;
    if (id === 'weeklyMatchupsLoading') return loadingOverlay;
    return null;
  },
  querySelector(sel) {
    if (sel === '.week-main-panels') return mainContainer;
    if (sel === '.week-side-panels') return sideContainer;
    return null;
  },
};
global.window.showToast = (msg, type) => { toasts.push({ msg: String(msg), type }); };

function responseFor(mode, pick) {
  if (mode === 'network') return Promise.reject(new Error('boom'));
  if (mode === 'http') {
    return Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({}) });
  }
  if (mode === 'api') {
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ ok: false, error: 'nope' }) });
  }
  return Promise.resolve({
    ok: true, status: 200,
    json: () => Promise.resolve({
      ok: true,
      matchups_html: '<div>W' + pick + '</div>',
      top_html: '<div>top' + pick + '</div>',
      highlights_html: '<div>hl' + pick + '</div>',
      league_scores: { week: Number(pick), matchups: [] },
      wrapped_url: '', week_has_scores: false,
    }),
  });
}

let stepIdx = 0;
global.fetch = (url) => {
  const step = scenario.steps[stepIdx];
  return responseFor(step.mode, step.pick);
};

eval(src);

const tick = () => new Promise((r) => setTimeout(r, 20));

(async () => {
  for (const step of scenario.steps) {
    stepIdx = scenario.steps.indexOf(step);
    select.value = step.pick;
    select._listeners['change'].call(select);
    await tick();
    await tick();
  }
  console.log(JSON.stringify({
    selectValue: select.value,
    toasts,
    matchupsHtml: matchupsContainer.innerHTML,
  }));
})();
"""


def _run_harness(scenario):
    iife_path = "/tmp/hub_week_iife.js"
    harness_path = "/tmp/hub_week_harness.js"
    with open(iife_path, "w", encoding="utf-8") as f:
        f.write(_extract_week_change_iife())
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write(_NODE_HARNESS)
    proc = subprocess.run(
        ["node", harness_path, iife_path, json.dumps(scenario)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


def test_failed_week_fetch_rolls_back_selector():
    out = _run_harness({"startWeek": "3", "steps": [{"pick": "4", "mode": "http"}]})
    assert out["selectValue"] == "3"
    assert out["matchupsHtml"] == ""
    assert len(out["toasts"]) == 1
    assert out["toasts"][0]["type"] == "error"
    assert "Week 4" in out["toasts"][0]["msg"]
    assert "Week 3" in out["toasts"][0]["msg"]


def test_network_error_rolls_back_selector():
    out = _run_harness({"startWeek": "3", "steps": [{"pick": "4", "mode": "network"}]})
    assert out["selectValue"] == "3"
    assert len(out["toasts"]) == 1


def test_api_error_payload_rolls_back_selector():
    out = _run_harness({"startWeek": "3", "steps": [{"pick": "4", "mode": "api"}]})
    assert out["selectValue"] == "3"
    assert len(out["toasts"]) == 1


def test_successful_week_change_commits_then_failure_rolls_back_to_it():
    out = _run_harness({"startWeek": "3", "steps": [
        {"pick": "4", "mode": "ok"},
        {"pick": "2", "mode": "http"},
    ]})
    # Week 4 committed; the failed Week 2 change rolls back to Week 4, not Week 3.
    assert out["selectValue"] == "4"
    assert "W4" in out["matchupsHtml"]
    assert len(out["toasts"]) == 1
    assert "Week 2" in out["toasts"][0]["msg"]
    assert "Week 4" in out["toasts"][0]["msg"]


def test_successful_week_change_updates_content_without_toast():
    out = _run_harness({"startWeek": "3", "steps": [{"pick": "4", "mode": "ok"}]})
    assert out["selectValue"] == "4"
    assert "W4" in out["matchupsHtml"]
    assert out["toasts"] == []
