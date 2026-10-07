"""Weekly hub ScoreZone Moments launcher: markup helper and client wiring."""
from __future__ import annotations

import json
import os
import subprocess

from dashboard_services.pages.weekly_hub_page import scorezone_moments_hub_html

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_launcher_markup_has_data_attributes():
    html = scorezone_moments_hub_html("sleeper", "123456789", 2026)
    assert 'data-rzm-hub' in html
    assert 'data-platform="sleeper"' in html
    assert 'data-league-id="123456789"' in html
    assert 'data-season="2026"' in html
    assert 'data-rzm-hub-open' in html
    assert 'data-rzm-hub-count' in html
    assert 'ScoreZone Moments' in html
    # Rendered hidden; the client reveals it only when moments exist.
    assert ' hidden>' in html or ' hidden' in html


def test_launcher_markup_escapes_attributes():
    html = scorezone_moments_hub_html('sleeper"x', '1"><script>', 2026)
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
    launcher = scorezone_moments_hub_html("sleeper", "123", 2026)
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
    html = scorezone_moments_hub_html("sleeper", "1", 2026)
    assert 'class="rzm-row' in html
    assert 'class="rzm-row-btn"' in html


# ── Client wiring: drive the hub launcher IIFE in Node with a DOM shim ──

_HUB_IIFE_START = "/* Weekly hub: ScoreZone Moments launcher"
_SHARED_IIFE_START = "/* ── Shared ScoreZone Moments (portfolio cards + weekly hub)"
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

// Fake timers + virtual clock: the launcher schedules retries/polls with
// setTimeout and bounds the watch with Date.now. Stepping the registry by
// hand drives the watch loop deterministically (and leaves no real timers
// behind to hold the process open).
const timers = new Map();
let nextTimerId = 1;
let virtualNow = 1700000000000;
global.setTimeout = function (fn, delay) {
  const id = nextTimerId++;
  timers.set(id, { fn, at: virtualNow + (delay || 0) });
  return id;
};
global.clearTimeout = function (id) { timers.delete(id); };
Date.now = function () { return virtualNow; };

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
// The hub launcher talks to the shared namespace; stub its fetch with a
// scripted sequence (scenario.apiBodies) or a single body (scenario.apiBody).
global.brRzm = {
  fetchMoments(platform, leagueId, season, week) {
    fetchCalls.push({ platform, leagueId, season, week });
    if (scenario.rejectStatus) {
      const err = new Error('Moments request failed (' + scenario.rejectStatus + ')');
      err.status = scenario.rejectStatus;
      return Promise.reject(err);
    }
    const bodies = scenario.apiBodies || [scenario.apiBody];
    return Promise.resolve(bodies[Math.min(fetchCalls.length - 1, bodies.length - 1)]);
  },
};

async function flush() {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

(async () => {
  eval(iifeSrc);
  await flush();
  const steps = scenario.steps || 0;
  for (let s = 0; s < steps; s++) {
    if (!timers.size) break;
    let earliestId = null;
    let earliest = null;
    for (const [id, t] of timers) {
      if (!earliest || t.at < earliest.at) { earliest = t; earliestId = id; }
    }
    timers.delete(earliestId);
    virtualNow = Math.max(virtualNow, earliest.at);
    earliest.fn();
    await flush();
  }
  console.log(JSON.stringify({
    fetchCalls,
    launcherHidden: launcher.hidden,
    countText: countEl.textContent,
    hasPayload: !!launcher._rzmPayload,
    fetchSkipped: fetchCalls.length === 0,
    pendingTimers: timers.size,
  }));
})();
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
    assert out["fetchCalls"] == [{"platform": "sleeper", "leagueId": "12345", "season": "2026", "week": ""}]


def test_hub_launcher_shows_row_with_td_count():
    out = _run_harness({
        "platform": "sleeper", "leagueId": "12345", "season": "2026",
        "apiBody": {"plays": [{"kind": "td"}], "td_count": 3, "teams": {"you": "A", "opp": "B"}},
    })
    assert out["launcherHidden"] is False
    assert out["countText"] == "3 touchdowns from your matchup"
    assert out["hasPayload"] is True


def test_hub_launcher_singular_touchdown():
    out = _run_harness({
        "platform": "sleeper", "leagueId": "12345", "season": "2026",
        "apiBody": {"plays": [{"kind": "td"}], "td_count": 1, "teams": {}},
    })
    assert out["countText"] == "1 touchdown from your matchup"


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


def test_hub_launcher_retries_until_plays_arrive():
    """Regression: the first fetch often lands before the league ctx /
    play store are ready (pending, then empty) and a single fetch left
    the row hidden forever. The launcher must ride the retry ladder and
    reveal the row when plays arrive."""
    out = _run_harness({
        "platform": "sleeper", "leagueId": "12345", "season": "2026",
        "apiBodies": [
            {"plays": [], "td_count": 0, "pending": True},
            {"plays": [], "td_count": 0, "live": True},
            {"plays": [{"kind": "td"}], "td_count": 2, "live": True},
        ],
        "steps": 2,
    })
    assert len(out["fetchCalls"]) == 3
    assert out["launcherHidden"] is False
    assert out["countText"] == "2 touchdowns from your matchup"
    assert out["hasPayload"] is True


def test_hub_launcher_stops_after_final_ladder():
    """All starters' games final and still no plays: the quick ladder
    runs out (final plays can still be landing in the store), then the
    watch stops instead of polling forever."""
    out = _run_harness({
        "platform": "sleeper", "leagueId": "12345", "season": "2026",
        "apiBody": {"plays": [], "td_count": 0, "status": "final", "live": False},
        "steps": 10,
    })
    assert len(out["fetchCalls"]) == 6  # initial fetch + 5 ladder retries
    assert out["pendingTimers"] == 0
    assert out["launcherHidden"] is True


def test_hub_launcher_stops_on_auth_error():
    """A 403 (league not authorized for this account) never heals by
    retrying; the watch stops after the first rejection."""
    out = _run_harness({
        "platform": "sleeper", "leagueId": "12345", "season": "2026",
        "rejectStatus": 403,
        "steps": 5,
    })
    assert len(out["fetchCalls"]) == 1
    assert out["pendingTimers"] == 0
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
// Virtual clock so the fetchMoments cache TTL can be stepped past.
let nowVal = 1700000000000;
Date.now = function () { return nowVal; };
global.brFetchWithTimeout = function (url, opts, timeout) {
  fetchCalls.push({ url, timeout, creds: opts && opts.credentials });
  const status = scenario.httpStatus || 200;
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json() { return Promise.resolve(scenario.apiBody || {}); },
  });
};

eval(iifeSrc);

function fireClick(targetClosest) {
  const ev = { target: { closest: targetClosest } };
  (listeners['click'] || []).forEach((fn) => fn(ev));
}

(async () => {
  const out = {};
  // fetchMoments: URL + cache (sequential calls; the cache fills on resolve)
  async function callFetch() {
    try { await window.brRzm.fetchMoments('sleeper', '12345', '2026'); return null; }
    catch (e) { return (e && e.status) || 'error'; }
  }
  out.firstError = await callFetch();
  out.secondError = await callFetch();
  out.fetchUrl = fetchCalls.length ? fetchCalls[0].url : null;
  out.fetchCount = fetchCalls.length;
  out.fetchCreds = fetchCalls.length ? fetchCalls[0].creds : null;
  // Step past the 45s cache TTL: a third call must hit the network again.
  nowVal += 46000;
  out.thirdError = await callFetch();
  out.fetchCountAfterTtl = fetchCalls.length;

  // Hub launcher click opens the modal with payload + league ctx.
  const hubBtn = makeEl('button');
  hubBtn.closest = (sel) => (sel === '[data-rzm-hub]' ? launcher : null);
  fireClick((sel) => (sel === '[data-rzm-hub-open]' ? hubBtn : null));
  out.modalOpened = !!modalOverlay;
  out.modalHasPlays = modalOverlay ? modalOverlay.innerHTML.includes('rzm-tl-row') : false;
  out.modalCtx = modalOverlay ? modalOverlay._rzmCtx : null;
  out.modalTitle = modalOverlay ? modalOverlay.innerHTML.includes('ScoreZone Moments') : false;
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
    assert out["fetchUrl"] == "/api/scorezone/moments?platform=sleeper&league_id=12345&season=2026"
    assert out["fetchCreds"] == "same-origin"
    # Second call for the same league hits the cache: one network fetch.
    assert out["fetchCount"] == 1


def test_shared_fetchMoments_http_error_rejects_and_is_not_cached():
    """Regression: fetchMoments never checked r.ok, so a 403 body was
    parsed and cached as an (empty) success for the whole session. An
    HTTP error must reject with its status and never enter the cache."""
    out = _run_shared_harness({"apiBody": {}, "httpStatus": 403})
    assert out["firstError"] == 403
    assert out["secondError"] == 403
    # Both calls reached the network: nothing was cached.
    assert out["fetchCount"] == 2


def test_shared_fetchMoments_pending_is_not_cached():
    """A pending body (league ctx still warming) is transient: caching it
    would hide moments behind a 45s stale empty after the ctx warms."""
    out = _run_shared_harness({"apiBody": {"plays": [], "td_count": 0, "pending": True}})
    assert out["firstError"] is None
    assert out["fetchCount"] == 2


def test_shared_fetchMoments_cache_expires():
    """Successful bodies are cached only briefly (45s TTL), so moments
    that land after the first fetch surface on the next watch tick
    instead of freezing at the first body for the whole session."""
    out = _run_shared_harness({"apiBody": {"plays": [], "td_count": 0}})
    assert out["fetchCount"] == 1
    assert out["fetchCountAfterTtl"] == 2


def test_shared_hub_click_opens_modal():
    out = _run_shared_harness({"payload": _payload()})
    assert out["modalOpened"] is True
    assert out["modalTitle"] is True
    assert out["modalHasPlays"] is True
    assert out["modalCtx"] == {"platform": "sleeper", "leagueId": "12345", "season": "2026"}


def test_shared_modal_escapes_team_names():
    payload = _payload()
    payload["teams"] = {"you": "A", "opp": "<img src=x>"}
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
    assert "window.brRzm.fetchMoments(platform, leagueId, season, week)" in src


def test_weekly_week_api_includes_rzm_launcher():
    """Regression: /api/weekly-week must pass rzm_hub_html to
    render_matchup_slide, or the launcher vanishes when she switches weeks."""
    src = open(os.path.join(_ROOT, "app.py"), encoding="utf-8").read()
    # Find the api_weekly_week function body.
    start = src.index("def api_weekly_week():")
    # The next top-level def (or end of file) bounds it.
    nxt = src.find("\n@app.route(", start + 10)
    nxt2 = src.find("\ndef ", start + 10)
    end = min(x for x in (nxt, nxt2) if x != -1)
    body = src[start:end]
    assert "rzm_hub_html=_api_rzm_for_matchup(m)" in body
    assert "scorezone_moments_hub_html" in body


def test_launcher_includes_week_attribute():
    """The launcher must carry the displayed week so the API can fetch
    moments for the viewed week, not just the current NFL week."""
    html = scorezone_moments_hub_html("sleeper", "123456789", 2026, week=3)
    assert 'data-week="3"' in html
    # Week is optional for backward compatibility.
    html2 = scorezone_moments_hub_html("sleeper", "123456789", 2026)
    assert "data-rzm-hub" in html2
