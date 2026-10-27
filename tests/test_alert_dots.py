"""Unread alert dots for Weekly Recap and Breakout Engine.

The dots mirror the changelog bell end to end: the server stamps each nav
dot with the latest entry's id (recap week / breakout set stamp) plus a
localStorage key, app.js compares it with the stored last-seen value, and
landing on the surface marks it seen. These tests pin the server stamps and
gates, the nav wiring, the CSS spec parity with .changelog-dot, and the
client state logic (via node, same harness style as the ScoreZone identity
tests).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


# ── Server stamps: recap ────────────────────────────────────────────────────

def _nfl_state(**over):
    state = {"season": 2026, "season_type": "reg", "week": 8, "previous_week": 7}
    state.update(over)
    return state


@pytest.fixture()
def signed_in(monkeypatch):
    import app as app_module
    from flask import session

    monkeypatch.setattr(app_module, "get_nfl_state", lambda: _nfl_state())
    with app_module.app.test_request_context("/"):
        session["account_id"] = 42
        yield app_module, session


def test_recap_stamp_is_latest_completed_week_keyed_per_league(signed_in):
    app_module, _session = signed_in
    week, key = app_module._recap_dot_stamp("sleeper", 2026, "L1")
    assert week == 7  # week 8 in progress, previous_week 7 is the latest recap
    assert key == "recap_last_seen:sleeper:2026:L1"
    # Per-league isolation: another league gets its own key.
    week_b, key_b = app_module._recap_dot_stamp("sleeper", 2026, "L2")
    assert week_b == 7 and key_b == "recap_last_seen:sleeper:2026:L2"
    assert key_b != key


def test_recap_stamp_gates(signed_in, monkeypatch):
    app_module, session = signed_in
    # Signed out: no dot.
    session.clear()
    assert app_module._recap_dot_stamp("sleeper", 2026, "L1") == (0, "")
    session["viewer_username"] = "kaedon"
    # League season is not the current NFL season.
    assert app_module._recap_dot_stamp("sleeper", 2025, "L1") == (0, "")
    # Preseason: no recap exists yet.
    monkeypatch.setattr(app_module, "get_nfl_state",
                        lambda: _nfl_state(season_type="pre", week=0, previous_week=0))
    assert app_module._recap_dot_stamp("sleeper", 2026, "L1") == (0, "")
    # Week 1 with nothing completed: no recap yet.
    monkeypatch.setattr(app_module, "get_nfl_state",
                        lambda: _nfl_state(week=1, previous_week=0))
    assert app_module._recap_dot_stamp("sleeper", 2026, "L1") == (0, "")
    # No league context.
    assert app_module._recap_dot_stamp("sleeper", 2026, "") == (0, "")


def test_recap_dot_html_carries_stamp_and_starts_hidden(signed_in):
    app_module, _session = signed_in
    html = app_module._recap_dot_html("sleeper", 2026, "L1")
    assert "alert-dot recap-dot" in html
    assert "alert-dot-hidden" in html  # revealed only by app.js when unopened
    assert "data-alert-week='7'" in html
    assert "data-alert-key='recap_last_seen:sleeper:2026:L1'" in html
    inline = app_module._recap_dot_html("sleeper", 2026, "L1", inline=True)
    assert "alert-dot-inline" in inline


# ── Server stamps: breakouts ────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a):
        pass

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _FakeCursor(self._row)


@pytest.fixture()
def breakout_run(monkeypatch):
    import app as app_module
    import dashboard_services.breakout_api as breakout_api
    from flask import session

    row = {"season": 2026, "as_of_week": 7, "scoring_version": "v3"}
    monkeypatch.setattr(breakout_api, "get_conn", lambda: _FakeConn(row))
    monkeypatch.setitem(app_module._breakouts_set_cache, "ts", 0.0)
    monkeypatch.setitem(app_module._breakouts_set_cache, "stamp", "")
    with app_module.app.test_request_context("/"):
        session["account_id"] = 42
        yield app_module, session, row


def test_breakout_stamp_is_latest_completed_set(breakout_run):
    app_module, _session, _row = breakout_run
    assert app_module._breakouts_latest_set() == "2026:7:v3"
    html = app_module._breakout_dot_html()
    assert "alert-dot breakout-dot" in html
    assert "alert-dot-hidden" in html
    assert "data-alert-set='2026:7:v3'" in html
    assert "data-alert-key='breakout_last_seen'" in html


def test_breakout_dot_absent_without_run_or_signin(breakout_run, monkeypatch):
    app_module, session, _row = breakout_run
    import dashboard_services.breakout_api as breakout_api

    monkeypatch.setattr(breakout_api, "get_conn", lambda: _FakeConn({}))
    monkeypatch.setitem(app_module._breakouts_set_cache, "ts", 0.0)
    assert app_module._breakouts_latest_set() == ""
    assert app_module._breakout_dot_html() == ""
    session.clear()
    monkeypatch.setitem(app_module._breakouts_set_cache, "ts", 0.0)
    monkeypatch.setitem(app_module._breakouts_set_cache, "stamp", "2026:7:v3")
    assert app_module._breakout_dot_html() == ""


# ── Nav wiring (source pins) ────────────────────────────────────────────────

def test_nav_wiring_desktop():
    src = _read("app.py")
    # Weekly dropdown: dot inside the Recap item and on the Weekly pill.
    assert 'f"Weekly Recap{_recap_dot_in}"' in src
    assert '"weeklyNavDropdown", btn_extra_cls=_rz_pulse, btn_badge_html=_recap_dot' in src
    # Both Players dropdowns (global + league chrome) carry the breakout dot.
    assert src.count("{_bo_new_badge}{_bo_dot_in}") == 2
    assert src.count("btn_badge_html=_bo_dot") == 2
    # Both dropdown builders accept the badge slot.
    assert src.count("btn_badge_html") >= 6


def test_nav_wiring_mobile():
    src = _read("app.py")
    assert '_sl("recap", "Weekly Recap", badge_html=_recap_dot_in)' in src
    assert '_sl("breakouts", "Breakout Engine", pro=True, badge_html=_bo_dot_in)' in src
    # More tab carries both dots next to the ScoreZone live dot.
    assert "{_more_dot}{_recap_dot}{_bo_dot}" in src
    # Category rows surface the dots for the Weekly and Players sections.
    assert 'if label == "Weekly":\n            alert = _recap_dot_in' in src
    assert 'elif label == "Players":\n            alert = _bo_dot_in' in src


def test_guest_mobile_nav_has_no_dots():
    src = _read("app.py")
    start = src.index("def _mobile_nav_guest")
    end = src.index("\ndef ", start + 1)
    assert "alert-dot" not in src[start:end]


def test_surface_markers_the_js_marks_seen_on():
    # Recap page main element and the Breakout Engine view tabs are what
    # initAlertDots looks for to mark a surface as opened.
    assert 'class="weekly-recap"' in _read("dashboard_services/pages/recap_page.py")
    assert "bo-view-tabs" in _read("app.py")


# ── CSS parity with the bell dot ────────────────────────────────────────────

def _css_block(css: str, selector: str) -> dict:
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert m, f"missing CSS block {selector}"
    decls = {}
    for part in m.group(1).split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            decls[k.strip()] = v.strip()
    return decls


def test_alert_dot_matches_changelog_dot_spec():
    css = _read("static/dashboard.css")
    bell = _css_block(css, ".changelog-dot")
    dot = _css_block(css, ".alert-dot")
    for prop in ("width", "height", "border-radius", "background", "border",
                 "top", "right", "pointer-events"):
        assert dot[prop] == bell[prop], prop
    assert dot["position"] == "absolute"
    hidden = _css_block(css, ".alert-dot-hidden")
    assert hidden["opacity"] == "0"


# ── Client state logic (node) ───────────────────────────────────────────────

def _function_source(source: str, name: str) -> str:
    start = source.index("function %s(" % name)
    brace = source.index("{", start)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start:pos + 1]
    raise AssertionError("unterminated %s" % name)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_alert_dot_state_logic():
    source = _read("static/app.js")
    fn = _function_source(source, "initAlertDots")
    harness = """
function El(attrs) {
  this.attrs = attrs || {};
  this.hidden = true;  // server renders every dot with alert-dot-hidden
  var self = this;
  this.classList = { toggle: function (cls, force) {
    if (cls === 'alert-dot-hidden') self.hidden = !!force;
  } };
}
El.prototype.getAttribute = function (n) {
  return Object.prototype.hasOwnProperty.call(this.attrs, n) ? this.attrs[n] : null;
};
var store = {};
var localStorage = {
  getItem: function (k) { return k in store ? store[k] : null; },
  setItem: function (k, v) { store[k] = String(v); }
};
var page = { recapDots: [], boDots: [], onRecap: false, onBo: false };
var document = {
  querySelectorAll: function (sel) {
    if (sel === '.recap-dot') return page.recapDots;
    if (sel === '.breakout-dot') return page.boDots;
    return [];
  },
  querySelector: function (sel) {
    if (sel === 'main.weekly-recap') return page.onRecap ? {} : null;
    if (sel === '.bo-view-tabs') return page.onBo ? {} : null;
    return null;
  }
};
function ok(v, m) { if (!v) throw new Error(m); }
%s

var KA = 'recap_last_seen:sleeper:2026:A';
var KB = 'recap_last_seen:sleeper:2026:B';
function recapDot(key, week) {
  return new El({ 'data-alert-week': String(week), 'data-alert-key': key });
}
function boDot(set) {
  return new El({ 'data-alert-set': set, 'data-alert-key': 'breakout_last_seen' });
}

// 1. Unopened recap: dot shows.
page.recapDots = [recapDot(KA, 7)]; page.boDots = [];
initAlertDots();
ok(page.recapDots[0].hidden === false, 'unopened recap must show the dot');

// 2. Opened recap (seen == latest week): stays hidden.
store = {}; store[KA] = '7';
page.recapDots = [recapDot(KA, 7)];
initAlertDots();
ok(page.recapDots[0].hidden === true, 'opened recap must hide the dot');

// 3. New week re-arms: seen 6, latest 7.
store = {}; store[KA] = '6';
page.recapDots = [recapDot(KA, 7)];
initAlertDots();
ok(page.recapDots[0].hidden === false, 'new week recap must re-arm the dot');

// 4. Landing on the recap page marks it seen and hides the dot.
store = {}; page.onRecap = true;
page.recapDots = [recapDot(KA, 7)];
initAlertDots();
ok(store[KA] === '7', 'opening the recap must store the latest week');
ok(page.recapDots[0].hidden === true, 'recap dot hides on the recap page');
page.onRecap = false;

// 5. Per-league isolation: league A seen does not clear league B.
store = {}; store[KA] = '7';
page.recapDots = [recapDot(KB, 7)];
initAlertDots();
ok(page.recapDots[0].hidden === false, 'league B dot survives league A being opened');
ok(!(KB in store), 'league B key must stay untouched');

// 6. Breakout set: unopened shows, opened hides, new set re-arms.
store = {}; page.recapDots = [];
page.boDots = [boDot('2026:7:v3')];
initAlertDots();
ok(page.boDots[0].hidden === false, 'unopened breakout set must show the dot');
store = {}; store['breakout_last_seen'] = '2026:7:v3';
page.boDots = [boDot('2026:7:v3')];
initAlertDots();
ok(page.boDots[0].hidden === true, 'opened breakout set must hide the dot');
store = {}; store['breakout_last_seen'] = '2026:6:v3';
page.boDots = [boDot('2026:7:v3')];
initAlertDots();
ok(page.boDots[0].hidden === false, 'new breakout set must re-arm the dot');

// 7. Landing on the Breakout Engine marks the set seen.
store = {}; page.onBo = true;
page.boDots = [boDot('2026:7:v3')];
initAlertDots();
ok(store['breakout_last_seen'] === '2026:7:v3', 'opening breakouts stores the set stamp');
ok(page.boDots[0].hidden === true, 'breakout dot hides on the breakouts page');
page.onBo = false;

// 8. No dots rendered (guest / gated): init is a no-op, never throws.
store = {}; page.recapDots = []; page.boDots = [];
initAlertDots();
console.log('alert-dot state logic ok');
""" % fn
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "alert-dot state logic ok" in result.stdout


# ── Changelog announcement ──────────────────────────────────────────────────

def test_changelog_announces_alert_dots():
    from dashboard_services.changelog import CHANGELOG

    top = CHANGELOG[1]
    assert top["date"] == "2026-10-27"
    assert top["tag"] == "new"
    assert "Weekly Recap" in top["text"] and "Breakout Engine" in top["text"]
    assert "—" not in top["text"]  # no em dashes in UI copy
