"""Weekly Hub: League Scores tab (My Matchup | League Scores).

Verifies the hub renders the toggle with league context data attributes,
and that the frontend IIFE wires the toggle, fetches from
/api/matchup/league-scores, and renders the Sleeper-style list.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

pytest.importorskip("flask")

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_hub_renders_league_scores_toggle():
    """The hub Matchups tab includes the My Matchup | League Scores toggle."""
    from dashboard_services.pages import weekly_hub_page as hub

    # Minimal ctx to get the matchups panel rendered. We only care that the
    # toggle markup is present with data attributes.
    html = hub.redzone_moments_hub_html("sleeper", "123", 2026)
    assert 'data-rzm-hub' in html  # sanity: helper works

    # The toggle is built inline in build_weekly_hub_body; verify the static
    # marker strings exist in the source.
    src = open(os.path.join(_ROOT, "dashboard_services", "pages", "weekly_hub_page.py")).read()
    assert 'data-ls-tabs' in src
    assert 'data-ls-tab="matchup"' in src
    assert 'data-ls-tab="league"' in src
    assert 'data-ls-view="league"' in src
    assert 'My Matchup' in src
    assert 'League Scores' in src


# ── Client wiring: drive the League Scores IIFE in Node with a DOM shim ──

_HUB_LS_START = "/* Weekly hub: League Scores tab (My Matchup | League Scores)."


def _extract_ls_iife():
    src = open(os.path.join(_ROOT, "static", "app.js"), encoding="utf-8").read()
    start = src.index(_HUB_LS_START)
    end = src.index("})();", start) + len("})();")
    return src[start:end]


_NODE_HARNESS = r"""
const fs = require('fs');
const iifeSrc = fs.readFileSync(process.argv[2], 'utf8');
const scenario = JSON.parse(process.argv[3]);

let fetchCalls = [];

function makeEl(tag, attrs) {
  const el = {
    tagName: (tag || 'div').toUpperCase(),
    children: [],
    attributes: attrs || {},
    hidden: false,
    _lsWired: false,
    _lsLoading: false,
    textContent: '',
    innerHTML: '',
    isConnected: true,
    dataset: {},
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); },
      toggle(c, f) { if (f) this._s.add(c); else this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    getAttribute(name) { return this.attributes[name] || ''; },
    setAttribute(name, v) { this.attributes[name] = String(v); },
    querySelector(sel) {
      if (sel === '#weeklyMatchupsContainer') return global.__matchupView;
      if (sel === '[data-ls-view="league"]') return global.__leagueView;
      return null;
    },
    querySelectorAll(sel) {
      if (sel === '[data-ls-tab]') return [global.__tabMatchup, global.__tabLeague];
      return [];
    },
    closest(sel) {
      if (sel === '.matchups-shell') return global.__shell;
      if (sel === '[data-ls-tab]') return null;
      return null;
    },
    contains(node) { return true; },
    addEventListener(type, fn) { (this._listeners[type] = this._listeners[type] || []).push(fn); },
    _listeners: {},
    _fire(type, ev) { (this._listeners[type] || []).forEach((fn) => fn(ev)); },
  };
  return el;
}

// Build the DOM: shell > tabs + matchupView + leagueView
const shell = makeEl('div');
const tabs = makeEl('div', {
  'data-platform': scenario.platform || 'sleeper',
  'data-league-id': scenario.leagueId || '123',
  'data-season': scenario.season || '2026',
});
const tabMatchup = makeEl('button');
tabMatchup.setAttribute('data-ls-tab', 'matchup');
tabMatchup.classList.add('is-active');
const tabLeague = makeEl('button');
tabLeague.setAttribute('data-ls-tab', 'league');
const matchupView = makeEl('div');
matchupView.hidden = false;
const leagueView = makeEl('div');
leagueView.hidden = true;
global.__shell = shell;
global.__tabs = tabs;
global.__tabMatchup = tabMatchup;
global.__tabLeague = tabLeague;
global.__matchupView = matchupView;
global.__leagueView = leagueView;

const docListeners = {};
global.document = {
  readyState: 'complete',
  querySelector(sel) {
    if (sel === '[data-ls-tabs]') return tabs;
    return null;
  },
  addEventListener(type, fn) { (docListeners[type] = docListeners[type] || []).push(fn); },
  createElement(tag) { return makeEl(tag); },
  body: makeEl('body'),
};
global.window = global;
global.escapeHtml = (s) => String(s == null ? '' : s);
global.fetch = (url) => {
  fetchCalls.push(url);
  return Promise.resolve({ json: () => Promise.resolve(scenario.apiBody) });
};

eval(iifeSrc);

// Simulate clicking the League Scores tab.
const clickEvent = {
  target: {
    closest(sel) {
      if (sel === '[data-ls-tab]') return tabLeague;
      return null;
    },
  },
  preventDefault() {},
};
tabs._fire('click', clickEvent);

setTimeout(() => {
  console.log(JSON.stringify({
    fetchCalls,
    matchupHidden: matchupView.hidden,
    leagueHidden: leagueView.hidden,
    leagueTabActive: tabLeague.classList.contains('is-active'),
    matchupTabActive: tabMatchup.classList.contains('is-active'),
    leagueHtml: leagueView.innerHTML,
    lsLoaded: leagueView.dataset.lsLoaded,
  }));
}, 100);
"""


def _run_harness(scenario):
    iife_path = "/tmp/hub_ls_iife.js"
    harness_path = "/tmp/hub_ls_harness.js"
    with open(iife_path, "w", encoding="utf-8") as f:
        f.write(_extract_ls_iife())
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write(_NODE_HARNESS)
    proc = subprocess.run(
        ["node", harness_path, iife_path, json.dumps(scenario)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


def test_ls_tab_fetches_and_renders():
    out = _run_harness({
        "platform": "sleeper", "leagueId": "123", "season": "2026",
        "apiBody": {
            "matchups": [
                {"left": {"name": "You", "score": 100.5, "proj": 110.2},
                 "right": {"name": "Opp", "score": 95.3, "proj": 105.1},
                 "win_prob": 65.5, "status": "in", "is_you": True},
                {"left": {"name": "Team C", "score": 80.0, "proj": 90.0},
                 "right": {"name": "Team D", "score": 85.0, "proj": 88.0},
                 "win_prob": 40.0, "status": "final", "is_you": False},
            ],
            "week": 3,
        },
    })
    assert len(out["fetchCalls"]) == 1
    assert "platform=sleeper" in out["fetchCalls"][0]
    assert "league_id=123" in out["fetchCalls"][0]
    assert "season=2026" in out["fetchCalls"][0]
    assert out["matchupHidden"] is True
    assert out["leagueHidden"] is False
    assert out["leagueTabActive"] is True
    assert out["matchupTabActive"] is False
    assert out["lsLoaded"] == "true"
    html = out["leagueHtml"]
    assert "Your matchup" in html
    assert "is-you" in html
    assert "100.5" in html
    assert "Live" in html
    assert "Final" in html


def test_ls_tab_empty_state():
    out = _run_harness({
        "platform": "sleeper", "leagueId": "123", "season": "2026",
        "apiBody": {"matchups": [], "week": 3},
    })
    assert "No matchups found" in out["leagueHtml"]


def test_ls_win_prob_uses_styled_win_bar():
    # Regression: the win-probability row must use the styled m-win-bar
    # pattern (shared with My Matchup), not the unstyled pf-live-wp markup
    # that rendered as bare "100%0%" text.
    out = _run_harness({
        "platform": "sleeper", "leagueId": "123", "season": "2026",
        "apiBody": {
            "matchups": [
                {"left": {"name": "You", "score": 100.5, "proj": 110.2},
                 "right": {"name": "Opp", "score": 95.3, "proj": 105.1},
                 "win_prob": 65.5, "status": "in", "is_you": True},
            ],
            "week": 3,
        },
    })
    html = out["leagueHtml"]
    assert "pf-live-wp" not in html
    assert "m-win-bar" in html
    assert "m-wp-track" in html
    assert "linear-gradient" in html
    assert "66%" in html  # rounded left win prob
    assert "34%" in html  # right win prob
