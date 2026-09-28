"""Weekly hub RedZone Moments launcher: markup helper and client wiring."""
from __future__ import annotations

import json
import os
import subprocess

from dashboard_services.pages.weekly_hub_page import redzone_moments_hub_html

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_launcher_markup_has_data_attributes():
    html = redzone_moments_hub_html("sleeper", "123456789", 2026)
    assert 'data-rzm-hub' in html
    assert 'data-platform="sleeper"' in html
    assert 'data-league-id="123456789"' in html
    assert 'data-season="2026"' in html
    assert 'data-rzm-hub-open' in html
    assert 'data-rzm-hub-count' in html
    assert 'RedZone Moments' in html
    # Rendered hidden; the client reveals it only when moments exist.
    assert ' hidden>' in html or ' hidden' in html


def test_launcher_markup_escapes_attributes():
    html = redzone_moments_hub_html('sleeper"x', '1"><script>', 2026)
    assert '<script>' not in html
    assert 'sleeper&quot;x' in html


def _render_slide(rzm_hub_html=""):
    import dashboard_services.matchups as mmod
    matchup = {
        "left": {
            "name": "Team A", "roster_id": "1", "record": "0-0", "username": "a",
            "avatar": "", "pts_total": 100.0,
            "starters": [{"pid": "11564", "name": "Drake Maye", "pos": "QB", "nfl": "NE", "pts": 20.0}],
        },
        "right": {
            "name": "Team B", "roster_id": "2", "record": "0-0", "username": "b",
            "avatar": "", "pts_total": 90.0,
            "starters": [{"pid": "4984", "name": "Josh Allen", "pos": "QB", "nfl": "BUF", "pts": 18.0}],
        },
    }
    return mmod.render_matchup_slide(
        "2026", matchup, w=3, proj_week=3,
        status_by_pid={},
        projections={},
        players={"11564": {"name": "Drake Maye"}, "4984": {"name": "Josh Allen"}},
        teams={},
        team_game_lookup={},
        rzm_hub_html=rzm_hub_html,
    )


def test_slide_includes_rzm_hub_html_under_win_bar():
    launcher = redzone_moments_hub_html("sleeper", "123", 2026)
    html = _render_slide(rzm_hub_html=launcher)
    assert 'data-rzm-hub' in html
    # Launcher sits after the header/win-bar zone and before the starter rows.
    head_end = html.index("m-head")
    body_start = html.index("m-body")
    launcher_pos = html.index("data-rzm-hub")
    assert head_end < launcher_pos < body_start


def test_slide_omits_rzm_hub_html_by_default():
    html = _render_slide()
    assert 'data-rzm-hub' not in html


def test_launcher_uses_shared_rzm_row_classes():
    # Same classes as the portfolio card row so the design stays identical.
    html = redzone_moments_hub_html("sleeper", "1", 2026)
    assert 'class="rzm-row' in html
    assert 'class="rzm-row-btn"' in html


# ── Client wiring: drive the hub launcher IIFE in Node with a DOM shim ──

_HUB_IIFE_START = "/* Weekly hub: RedZone Moments launcher"
_SHARED_IIFE_START = "/* ── Shared RedZone Moments (portfolio cards + weekly hub)"
_SHARED_IIFE_END = "window.brRzmOpenModal = function (payload, ctx) { return window.brRzm.openModal(payload, ctx); };"


def _extract_hub_iife():
    src = open(os.path.join(_ROOT, "static", "app.js"), encoding="utf-8").read()
    start = src.index(_HUB_IIFE_START)
    # The IIFE ends at the first "})();" after its start.
    end = src.index("})();", start) + len("})();")
    return src[start:end]


def _extract_shared_iife():
    src = open(os.path.join(_ROOT, "static", "app.js"), encoding="utf-8").read()
    start = src.index(_SHARED_IIFE_START)
    end = src.index(_SHARED_IIFE_END, start) + len(_SHARED_IIFE_END)
    return src[start:end]


_NODE_HARNESS = r"""
const fs = require('fs');
const iifeSrc = fs.readFileSync(process.argv[2], 'utf8');
const scenario = JSON.parse(process.argv[3]);

let fetchCalls = [];

function makeEl(tag) {
  const el = {
    tagName: (tag || 'div').toUpperCase(),
    children: [],
    attributes: {},
    hidden: false,
    _rzmInit: false,
    textContent: '',
    innerHTML: '',
    isConnected: true,
    style: {},
    dataset: {},
    classList: { add() {}, toggle() {}, contains() { return false; } },
    getAttribute(name) { return this.attributes[name] || ''; },
    setAttribute(name, v) { this.attributes[name] = String(v); },
    querySelector(sel) {
      if (sel === '[data-rzm-hub]') return global.__launcher;
      if (sel === '[data-rzm-hub-count]') return global.__countEl;
      return null;
    },
    querySelectorAll() { return []; },
    closest(sel) {
      if (sel === '[data-rzm-hub]') return global.__launcher;
      return null;
    },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {},
  };
  return el;
}

const launcher = makeEl('div');
launcher.setAttribute('data-platform', scenario.platform || '');
launcher.setAttribute('data-league-id', scenario.leagueId || '');
launcher.setAttribute('data-season', scenario.season || '');
launcher.hidden = true;
const countEl = makeEl('span');
global.__launcher = launcher;
global.__countEl = countEl;

const listeners = {};
global.document = {
  readyState: 'complete',
  querySelector(sel) { return sel === '[data-rzm-hub]' ? launcher : null; },
  querySelectorAll(sel) { return sel === '[data-rzm-hub]' ? [launcher] : []; },
  addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
  createElement(tag) { return makeEl(tag); },
  body: makeEl('body'),
  getElementById() { return null; },
};
global.window = global;
global.escapeHtml = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
// The hub launcher talks to the shared namespace; stub its fetch.
global.brRzm = {
  fetchMoments(platform, leagueId, season) {
    fetchCalls.push({ platform, leagueId, season });
    return Promise.resolve(scenario.apiBody);
  },
};

eval(iifeSrc);

setTimeout(() => {
  console.log(JSON.stringify({
    fetchCalls,
    launcherHidden: launcher.hidden,
    countText: countEl.textContent,
    hasPayload: !!launcher._rzmPayload,
    fetchSkipped: fetchCalls.length === 0,
  }));
}, 50);
"""


def _run_harness(scenario):
    iife_path = os.path.join("/tmp", "rzm_hub_iife.js")
    harness_path = os.path.join("/tmp", "rzm_hub_harness.js")
    with open(iife_path, "w", encoding="utf-8") as f:
        f.write(_extract_hub_iife())
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write(_NODE_HARNESS)
    proc = subprocess.run(
        ["node", harness_path, iife_path, json.dumps(scenario)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


def test_hub_launcher_fetches_moments_for_league():
    out = _run_harness({
        "platform": "sleeper", "leagueId": "12345", "season": "2026",
        "apiBody": {"plays": [], "td_count": 0},
    })
    assert out["fetchCalls"] == [{"platform": "sleeper", "leagueId": "12345", "season": "2026"}]


def test_hub_launcher_shows_row_with_td_count():
    out = _run_harness({
        "platform": "sleeper", "leagueId": "12345", "season": "2026",
        "apiBody": {"plays": [{"kind": "td"}], "td_count": 3, "teams": {"you": "A", "opp": "B"}},
    })
    assert out["launcherHidden"] is False
    assert out["countText"] == "3 touchdowns"
    assert out["hasPayload"] is True


def test_hub_launcher_singular_touchdown():
    out = _run_harness({
        "platform": "sleeper", "leagueId": "12345", "season": "2026",
        "apiBody": {"plays": [{"kind": "td"}], "td_count": 1, "teams": {}},
    })
    assert out["countText"] == "1 touchdown"


def test_hub_launcher_stays_hidden_without_plays():
    out = _run_harness({
        "platform": "sleeper", "leagueId": "12345", "season": "2026",
        "apiBody": {"plays": [], "td_count": 0},
    })
    assert out["launcherHidden"] is True


def test_hub_launcher_reinits_after_node_replacement():
    # Regression: the hub repaints the matchup area after load, replacing the
    # launcher node with a fresh hidden copy. The IIFE must pick up the new
    # node (via MutationObserver or window.brInitHubRzm) and reveal it.
    src = _extract_hub_iife()
    assert "MutationObserver" in src
    assert "window.brInitHubRzm" in src


def test_hub_launcher_skips_fetch_without_league_ref():
    out = _run_harness({
        "platform": "", "leagueId": "", "season": "2026",
        "apiBody": {"plays": [{"kind": "td"}], "td_count": 1},
    })
    assert out["fetchSkipped"] is True
    assert out["launcherHidden"] is True


# ── Shared namespace: fetch URL/cache, modal open via hub click, Escape ──

_SHARED_HARNESS = r"""
const fs = require('fs');
const iifeSrc = fs.readFileSync(process.argv[2], 'utf8');
const scenario = JSON.parse(process.argv[3]);

let fetchCalls = [];
const listeners = {};

function makeEl(tag) {
  const el = {
    tagName: (tag || 'div').toUpperCase(),
    children: [],
    attributes: {},
    hidden: false,
    textContent: '',
    innerHTML: '',
    isConnected: true,
    style: {},
    dataset: {},
    classList: { add() {}, toggle() {}, contains() { return false; } },
    getAttribute(name) { return this.attributes[name] || ''; },
    setAttribute(name, v) { this.attributes[name] = String(v); },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    closest() { return null; },
    appendChild(c) { this.children.push(c); return c; },
    remove() { this._removed = true; },
    addEventListener() {},
  };
  return el;
}

const body = makeEl('body');
let modalOverlay = null;
const origAppend = body.appendChild.bind(body);
body.appendChild = (c) => { if (c.className === 'rzm-modal-overlay') modalOverlay = c; return origAppend(c); };

const launcher = makeEl('div');
launcher.setAttribute('data-platform', 'sleeper');
launcher.setAttribute('data-league-id', '12345');
launcher.setAttribute('data-season', '2026');
launcher._rzmPayload = scenario.payload;

global.document = {
  readyState: 'complete',
  querySelector(sel) { return null; },
  querySelectorAll() { return []; },
  addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
  createElement(tag) { return makeEl(tag); },
  body,
  getElementById(id) { return (id === 'rzmModalOverlay' && modalOverlay && !modalOverlay._removed) ? modalOverlay : null; },
};
global.window = global;
global.escapeHtml = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
global.brFetchWithTimeout = function (url, opts, timeout) {
  fetchCalls.push({ url, timeout, creds: opts && opts.credentials });
  return Promise.resolve({ json() { return Promise.resolve(scenario.apiBody || {}); } });
};

eval(iifeSrc);

function fireClick(targetClosest) {
  const ev = { target: { closest: targetClosest } };
  (listeners['click'] || []).forEach((fn) => fn(ev));
}

(async () => {
  const out = {};
  // fetchMoments: URL + cache (sequential calls; the cache fills on resolve)
  const p1 = window.brRzm.fetchMoments('sleeper', '12345', '2026');
  await p1;
  const p2 = window.brRzm.fetchMoments('sleeper', '12345', '2026');
  await p2;
  out.fetchUrl = fetchCalls.length ? fetchCalls[0].url : null;
  out.fetchCount = fetchCalls.length;
  out.fetchCreds = fetchCalls.length ? fetchCalls[0].creds : null;

  // Hub launcher click opens the modal with payload + league ctx.
  const hubBtn = makeEl('button');
  hubBtn.closest = (sel) => (sel === '[data-rzm-hub]' ? launcher : null);
  fireClick((sel) => (sel === '[data-rzm-hub-open]' ? hubBtn : null));
  out.modalOpened = !!modalOverlay;
  out.modalHasPlays = modalOverlay ? modalOverlay.innerHTML.includes('rzm-play') : false;
  out.modalCtx = modalOverlay ? modalOverlay._rzmCtx : null;
  out.modalTitle = modalOverlay ? modalOverlay.innerHTML.includes('RedZone Moments') : false;
  out.modalHtml = modalOverlay ? modalOverlay.innerHTML : '';

  // Escape closes it.
  (listeners['keydown'] || []).forEach((fn) => fn({ key: 'Escape' }));
  out.modalClosedOnEscape = modalOverlay ? !!modalOverlay._removed : false;

  // openModal with no plays is a no-op.
  modalOverlay = null;
  window.brRzm.openModal({ plays: [] }, {});
  out.emptyPayloadNoop = modalOverlay === null;

  console.log(JSON.stringify(out));
})();
"""


def _run_shared_harness(scenario):
    iife_path = os.path.join("/tmp", "rzm_shared_iife.js")
    harness_path = os.path.join("/tmp", "rzm_shared_harness.js")
    with open(iife_path, "w", encoding="utf-8") as f:
        f.write(_extract_shared_iife())
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write(_SHARED_HARNESS)
    proc = subprocess.run(
        ["node", harness_path, iife_path, json.dumps(scenario)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


def _payload():
    return {
        "plays": [{
            "kind": "td", "name": "J. Cook", "pos": "RB", "side": "you",
            "play_text": "J.Cook 12 yd TD run", "yards": 12,
            "quarter": "2", "clock": "3:14", "game_id": "g1", "play_id": "p1",
        }],
        "td_count": 1,
        "teams": {"you": "Caleb's Couch", "opp": "Rival"},
    }


def test_shared_fetchMoments_url_and_cache():
    out = _run_shared_harness({"apiBody": {"plays": [], "td_count": 0}})
    assert out["fetchUrl"] == "/api/redzone/moments?platform=sleeper&league_id=12345&season=2026"
    assert out["fetchCreds"] == "same-origin"
    # Second call for the same league hits the cache: one network fetch.
    assert out["fetchCount"] == 1


def test_shared_hub_click_opens_modal():
    out = _run_shared_harness({"payload": _payload()})
    assert out["modalOpened"] is True
    assert out["modalTitle"] is True
    assert out["modalHasPlays"] is True
    assert out["modalCtx"] == {"platform": "sleeper", "leagueId": "12345", "season": "2026"}


def test_shared_modal_escapes_team_names():
    payload = _payload()
    payload["teams"] = {"you": "<img src=x>", "opp": "B"}
    out = _run_shared_harness({"payload": payload})
    assert out["modalOpened"] is True
    # The raw tag must not appear unescaped in the modal HTML.
    assert "<img src=x>" not in out["modalHtml"]
    assert "&lt;img src=x&gt;" in out["modalHtml"]


def test_shared_escape_closes_modal():
    out = _run_shared_harness({"payload": _payload()})
    assert out["modalOpened"] is True
    assert out["modalClosedOnEscape"] is True


def test_shared_openModal_empty_payload_noop():
    out = _run_shared_harness({"payload": _payload()})
    assert out["emptyPayloadNoop"] is True


def test_shared_namespace_exposed():
    src = open(os.path.join(_ROOT, "static", "app.js"), encoding="utf-8").read()
    assert "window.brRzm = (function () {" in src
    assert "window.brRzmOpenModal = function (payload, ctx) { return window.brRzm.openModal(payload, ctx); };" in src
    # Portfolio card row fetch delegates to the shared namespace.
    assert "window.brRzm.fetchMoments(platform, leagueId, season)" in src
