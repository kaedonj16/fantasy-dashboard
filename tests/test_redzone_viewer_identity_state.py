"""Regression coverage for Redzone's current-state viewer identity contract."""
from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _function_source(source: str, name: str) -> str:
    """Extract one plain JavaScript function without duplicating its logic."""
    start = source.index("function %s(" % name)
    brace = source.index("{", start)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
    raise AssertionError("unterminated %s" % name)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_viewer_identity_tracks_the_active_state_across_scope_paths():
    """Exercise the exact frontend helpers through all identity-changing paths."""
    source = (_ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
    helpers = "\n".join(
        _function_source(source, name)
        for name in ("_myRidSet", "_setState", "_hasViewerIdentityFields", "_isMyRid",
                     "_myMatchups", "_oppOf", "_focusedPair")
    )
    harness = """
var _scope = 'league', _state = {}, _myRids = new Set(), _heroMid = null;
%s
function ok(value, message) { if (!value) throw new Error(message); }

// Initial This League payload: the user's team is deliberately second.
_setState({scope: 'league', viewer_roster_id: '7', matchups: [
  {roster_id: '2', matchup_id: '10'}, {roster_id: '7', matchup_id: '10'}
]});
ok(_isMyRid('7') && !_isMyRid('2'), 'initial league identity');
ok(_myMatchups()[0].roster_id === '7', 'My Team must not use first roster');
ok(_oppOf(_myMatchups()[0]).roster_id === '2', 'opponent must be opposite side');
_heroMid = '10';
ok(_focusedPair().mine.roster_id === '7' && _focusedPair().opp.roster_id === '2',
   'hero must use the same identity as My Team');

// League -> My Leagues and progressive slices retain namespaced identities.
_scope = 'user'; _heroMid = null;
_setState({scope: 'user', viewer_roster_ids: ['0:1'], matchups: []});
ok(_isMyRid('0:1') && !_isMyRid('1:1'), 'first progressive slice');
_setState({scope: 'user', viewer_roster_ids: ['0:1', '1:1'], matchups: []});
ok(_isMyRid('0:1') && _isMyRid('1:1'), 'second progressive slice');
ok(!_isMyRid('1'), 'namespace must not be stripped');

// Cached user restoration uses its own full portfolio identity.
var cachedUser = {scope: 'user', viewer_roster_ids: ['0:1', '1:1'], matchups: []};
_setState(cachedUser);
ok(_isMyRid('0:1') && _isMyRid('1:1'), 'cached user identity');

// User -> league drops every old namespaced identity.
_scope = 'league';
_setState({scope: 'league', viewer_roster_id: '7', matchups: []});
ok(_isMyRid('7') && !_isMyRid('0:1') && !_isMyRid('1:1'), 'stale user ids removed');

// An incomplete refresh is rejected, while an explicit empty identity is an
// unavailable viewer state rather than a fallback to a prior roster.
var valid = _state;
ok(!_hasViewerIdentityFields({scope: 'league', matchups: []}, 'league'), 'partial refresh rejected');
ok(_state === valid && _isMyRid('7'), 'failed partial keeps last-good identity');
_setState({scope: 'league', viewer_roster_id: '', viewer_roster_ids: [], matchups: []});
ok(!_myRids.size && !_isMyRid('7'), 'explicit unavailable identity has no fallback');
_heroMid = '10';
ok(_focusedPair() === null, 'hero has no first-roster viewer fallback');

// User scope accepts only its canonical plural field.
_scope = 'user';
_setState({scope: 'user', viewer_roster_id: '0:1', matchups: []});
ok(!_myRids.size, 'user scope must not fall back to singular roster id');
""" % helpers
    result = subprocess.run(["node", "-e", harness], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_all_runtime_state_replacements_use_the_identity_synchronizer():
    source = (_ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
    # The initializer and _setState implementation are the only permitted raw
    # assignments. Every cache restore, poll, and stream slice must synchronize.
    assignments = [line.strip() for line in source.splitlines()
                   if re.search(r"\b_state\s*=\s*[^=]", line)]
    assert assignments == [
        "var _state    = window.__rz__ || {};",
        "_state = data || {};",
    ]
    for marker in (
        "_setState(cached);",
        "_setState(newData);",
        "_setState(base);",
        "if (!_hasViewerIdentityFields(newData, myScope))",
        "if (!_myRids.size) return null;",
    ):
        assert marker in source
    assert "_fpair.find(function(m) { return _isMyRid(m.roster_id); }) || _fpair[0]" not in source
