"""Regression: focusing a hero matchup must not manufacture false opponents.

The `mine`/`opp` flags built by ``_rosterTags`` in ``static/redzone.js`` drive
two things at once: the feed's MY TEAM / OPP chip *and* the alert path
(``_scoringAlertCandidates`` fires an OPPONENT_TD for any TD whose event carries
``opp``).

The bug: when the viewer focused a hero matchup they were **not** in, the tag
builder added both of that matchup's rosters to ``opp``. Because ``opp`` feeds
the alert path, this stamped "OPP" on players from a matchup the viewer merely
opened to watch and fired opponent-TD alerts for teams the viewer was not
playing against.

These tests drive the real client helpers (extracted from redzone.js and run
under Node) so they guard the source, not a Python mirror of it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RZ = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")


def _between(start: str, end: str) -> str:
    i = RZ.index(start)
    return RZ[i:RZ.index(end, i)]


def _harness(scenario_js: str) -> str:
    roster_tags = _between("function _rosterTags(data) {", "function _isBigPlay(ev) {")
    alert_key = _between("function _alertKey() {", "function _scoringAlertCandidates")
    candidates = _between(
        "function _scoringAlertCandidates(allEvents, leagueId) {",
        "function _detectChanges(newData, animationIntent) {",
    )
    return f"""
var _myRids = new Set(), _heroMid = null, _scope = 'league';
function _isMyRid(rid) {{ return _myRids.has(String(rid)); }}
var ALERT_TD = 'TD', ALERT_OPP_TD = 'OPPONENT_TD';
{alert_key}
{candidates}
{roster_tags}

// A 6-team league. Viewer is roster '1' (matchup '10' vs roster '2').
var DATA = {{ matchups: [
  {{ roster_id: '1', matchup_id: '10', players: ['mahomes', 'kelce'],  starters: ['mahomes', 'kelce'] }},
  {{ roster_id: '2', matchup_id: '10', players: ['allen', 'diggs'],    starters: ['allen', 'diggs'] }},
  {{ roster_id: '3', matchup_id: '20', players: ['hurts', 'brown'],    starters: ['hurts', 'brown'] }},
  {{ roster_id: '4', matchup_id: '20', players: ['lamar', 'andrews'],  starters: ['lamar', 'andrews'] }},
  {{ roster_id: '5', matchup_id: '30', players: ['burrow', 'chase'],   starters: ['burrow', 'chase'] }},
  {{ roster_id: '6', matchup_id: '30', players: ['dak', 'lamb'],       starters: ['dak', 'lamb'] }},
] }};
_myRids = new Set(['1']);

// One TD from every roster's marquee player, tagged via the real _rosterTags.
function tdsFor(tags) {{
  return ['mahomes', 'allen', 'hurts', 'lamar', 'burrow', 'dak'].map(function(pid) {{
    var rid = tags.pidToRoster[pid] || '';
    return {{ kind: 'td', name: pid, pid: pid, gameId: 'g_' + pid, playId: 'p_' + pid,
             mine: tags.my.has(rid), opp: tags.opp.has(rid) }};
  }});
}}
function alertsFor(tags) {{
  return _scoringAlertCandidates(tdsFor(tags), 'L1').map(function(c) {{
    return {{ name: c.ev.name, type: c.type }};
  }});
}}
{scenario_js}
"""


def _run(scenario_js: str):
    return json.loads(subprocess.check_output(
        ["node", "-e", _harness(scenario_js)], text=True))


def test_default_view_tags_only_the_real_opponent():
    out = _run(
        "_heroMid = null;"
        "var t = _rosterTags(DATA);"
        "console.log(JSON.stringify({ my: [...t.my], opp: [...t.opp], alerts: alertsFor(t) }));"
    )
    assert out["my"] == ["1"]
    assert out["opp"] == ["2"]
    # Only my player and my real opponent alert.
    assert out["alerts"] == [
        {"name": "mahomes", "type": "TD"},
        {"name": "allen", "type": "OPPONENT_TD"},
    ]


def test_focusing_foreign_matchup_does_not_create_opponents():
    """Focusing a matchup the viewer is NOT in must not tag it as opponents,
    and must not fire opponent-TD alerts for those teams."""
    out = _run(
        "_heroMid = '20';"  # rosters 3 vs 4 -- viewer is in neither
        "var t = _rosterTags(DATA);"
        "console.log(JSON.stringify({ my: [...t.my], opp: [...t.opp], alerts: alertsFor(t) }));"
    )
    assert out["my"] == ["1"]
    # rosters 3 and 4 must NOT be opponents just because the viewer opened them.
    assert out["opp"] == ["2"]
    assert {"name": "hurts", "type": "OPPONENT_TD"} not in out["alerts"]
    assert {"name": "lamar", "type": "OPPONENT_TD"} not in out["alerts"]
    # The viewer's real matchup is untouched.
    assert out["alerts"] == [
        {"name": "mahomes", "type": "TD"},
        {"name": "allen", "type": "OPPONENT_TD"},
    ]


def test_focusing_own_matchup_is_unchanged():
    out = _run(
        "_heroMid = '10';"  # the viewer's own matchup
        "var t = _rosterTags(DATA);"
        "console.log(JSON.stringify({ my: [...t.my], opp: [...t.opp], alerts: alertsFor(t) }));"
    )
    assert out["my"] == ["1"]
    assert out["opp"] == ["2"]
    assert out["alerts"] == [
        {"name": "mahomes", "type": "TD"},
        {"name": "allen", "type": "OPPONENT_TD"},
    ]


def test_pid_to_roster_still_maps_every_matchup_player():
    """Player->roster mapping is independent of focus, so the focused matchup's
    players still resolve to their owners (for owner/league context labels)."""
    out = _run(
        "_heroMid = '20';"
        "var t = _rosterTags(DATA);"
        "console.log(JSON.stringify(t.pidToRoster));"
    )
    assert out["hurts"] == "3"
    assert out["brown"] == "3"
    assert out["lamar"] == "4"
    assert out["andrews"] == "4"


def test_rostertags_no_longer_injects_hero_rosters_into_opp():
    # Source-level guard: the old hero block that added foreign rosters to opp is gone.
    block = _between("function _rosterTags(data) {", "function _isBigPlay(ev) {")
    assert "oppRosters.add(rid)" not in block
    assert "myRosters.add(rid)" not in block
