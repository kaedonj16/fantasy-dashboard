"""Behavioral regression coverage for RedZone alert gating.

The core bug was a single-event gate (``allEvents.length === 1``) that suppressed
a user's touchdown alert whenever the same poll carried any other event. These
tests drive the real client helpers (extracted from redzone.js and run under
Node) to prove each eligible event is evaluated independently, that alerts dedupe
deterministically, and that two-point conversions / overturned TDs never fire a
TD alert.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RZ = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
APP = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _js_between(start: str, end: str) -> str:
    return RZ[RZ.index(start):RZ.index(end, RZ.index(start))]


def _alert_harness(scenario_js: str) -> str:
    helpers = _js_between("var ALERT_TD = 'TD'", "function _detectChanges(newData, animationIntent) {")
    return f"""
var _notifHistory = [];
var _alertedKeys = new Set();
var localStorage = {{ store: {{}}, setItem: function(k, v) {{ this.store[k] = v; }},
                      getItem: function(k) {{ return this.store[k] || null; }} }};
{helpers}

// Mirrors the caller loop in _detectChanges: independent per-event evaluation,
// dedupe via _pushAlert, my-TDs collected for the local chime path.
function runBatch(events, leagueId) {{
  var myTDs = [];
  _scoringAlertCandidates(events, leagueId).forEach(function(cand) {{
    var ev = cand.ev;
    var pushed = _pushAlert({{ key: cand.key, type: cand.type, name: ev.name,
      team: ev.nflTeam, desc: ev.desc, pts: ev.pts, league: ev.league || '',
      matchup: ev.rosterId || '' }});
    if (pushed && cand.type === ALERT_TD) myTDs.push(ev);
  }});
  return myTDs;
}}
{scenario_js}
"""


def _run(scenario_js: str):
    return json.loads(subprocess.check_output(
        ["node", "-e", _alert_harness(scenario_js)], text=True))


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")


def _td(pid, playId, **kw):
    ev = {"kind": "td", "mine": True, "pid": pid, "playId": playId,
          "name": pid, "desc": "TD", "pts": 6.0, "gameId": "g1"}
    ev.update(kw)
    return json.dumps(ev)


def test_single_td_alerts():
    out = _run(f"console.log(JSON.stringify({{ n: runBatch([{_td('rb','p1')}], 'L1').length }}));")
    assert out["n"] == 1


def test_td_with_unrelated_events_still_alerts():
    # A realistic poll batch: my TD plus other games, an injury, a lead change.
    other = ('{kind:"gain",mine:false,pid:"x",playId:"p2",name:"x",desc:"gain",gameId:"g2"},'
             '{kind:"neg",mine:false,pid:"y",playId:"p3",name:"y",desc:"inj",gameId:"g3"}')
    out = _run(
        f"var td={_td('rb','p1')};"
        f"console.log(JSON.stringify({{ n: runBatch([{other}, td].map(function(e){{return typeof e==='string'?JSON.parse(e):e;}}), 'L1').length }}));"
    )
    assert out["n"] == 1


def test_qb_and_receiver_contributions_produce_one_alert():
    # Two contributions (QB pass_td + WR rec_td) share ONE canonical playId.
    qb = _td("qb", "play9", name="QB")
    wr = _td("wr", "play9", name="WR")
    out = _run(f"console.log(JSON.stringify({{ n: runBatch([{qb},{wr}], 'L1').length }}));")
    assert out["n"] == 1


def test_two_separate_td_plays_produce_two_alerts():
    a = _td("rb", "playA")
    b = _td("wr", "playB")
    out = _run(f"console.log(JSON.stringify({{ n: runBatch([{a},{b}], 'L1').length }}));")
    assert out["n"] == 2


def test_revision_of_existing_td_does_not_duplicate():
    td = _td("rb", "p1")
    upd = _td("rb", "p1", isUpdate=True)
    out = _run(
        f"var first = runBatch([{td}], 'L1').length;"
        f"var second = runBatch([{upd}], 'L1').length;"  # same play re-arrives as update
        "console.log(JSON.stringify({ first: first, second: second, hist: _notifHistory.length }));"
    )
    assert out == {"first": 1, "second": 0, "hist": 1}


def test_overturned_td_does_not_alert():
    # A tombstone is kind 'nullified', never 'td'.
    null_ev = '{kind:"nullified",mine:true,pid:"rb",playId:"p1",name:"RB",desc:"overturned",isNullified:true,gameId:"g1"}'
    out = _run(f"console.log(JSON.stringify({{ n: runBatch([{null_ev}], 'L1').length }}));")
    assert out["n"] == 0


def test_alerts_history_populates_with_structured_entry():
    out = _run(
        f"runBatch([{_td('rb','p1', name='Saquon', pts=12.3)}], 'L1');"
        "console.log(JSON.stringify(_notifHistory[0]));"
    )
    assert out["type"] == "TD"
    assert out["name"] == "Saquon"
    assert out["pts"] == 12.3
    assert out["key"].startswith("td:L1:g1:")


def test_opponent_td_classified_and_not_chimed():
    opp = ('{kind:"td",mine:false,opp:true,pid:"opp",playId:"p1",'
           'name:"OppStar",desc:"TD",pts:6,gameId:"g1"}')
    out = _run(
        f"var myTDs = runBatch([{opp}], 'L1');"
        "console.log(JSON.stringify({ chimed: myTDs.length, type: _notifHistory[0].type }));"
    )
    # Opponent TD records in history but never lands in the local-chime myTDs list.
    assert out == {"chimed": 0, "type": "OPPONENT_TD"}


def test_two_point_conversion_is_never_a_td_alert():
    conv = ('{kind:"two_point",mine:true,pid:"wr",playId:"p1",'
            'name:"Jefferson",desc:"2PT",pts:2,gameId:"g1"}')
    out = _run(
        f"var myTDs = runBatch([{conv}], 'L1');"
        "console.log(JSON.stringify({ chimed: myTDs.length, hist: _notifHistory.length }));"
    )
    assert out == {"chimed": 0, "hist": 0}


# ── Source-level guarantees ──────────────────────────────────────────────────

def test_alert_gating_no_longer_depends_on_single_event_batch():
    block = _js_between("var _liveAlerts = _alertsArmed", "if (myTDs.length) {")
    assert "allEvents.length === 1" not in block
    assert "_scoringAlertCandidates" in block


def test_injury_lead_and_milestone_feed_alert_history():
    changes = _js_between("function _detectChanges(newData, animationIntent) {", "// ── Filters")
    assert "ALERT_INJURY" in changes
    assert "ALERT_LEAD" in changes
    assert "ALERT_MILESTONE" in changes


def test_permission_flow_checks_returned_status_not_just_resolution():
    # A denied/dismissed prompt must NOT fake an enabled state; the handler keys
    # off the real permission + subscription status, never mere resolution.
    handler = RZ[RZ.index("var notifEnable = root.querySelector('#rz-notif-enable')"):]
    handler = handler[:handler.index("var notifDismiss")]
    assert "brEnablePush" in handler
    assert "'granted'" in handler
    # The old blanket "_notifDismissed = true on any resolution" is gone.
    assert "_notifDismissed = true; _render();" not in handler


def test_app_exposes_real_pushsubscription_flow():
    assert "window.brEnablePush" in APP
    assert "pushManager.getSubscription" in APP  # reuse an existing subscription
    assert "/api/push/subscribe" in APP           # persist via the existing endpoint
