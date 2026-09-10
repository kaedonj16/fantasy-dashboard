"""Redzone This League ↔ My Leagues must not paint the wrong portfolio.

After browsing My Leagues (portfolio / ESPN names) and switching back to
This League, a late ``scope=user`` poll used to overwrite ``_state`` while
the UI chrome was already on ``scope=league``. The header still showed the
URL league (e.g. blackedraw) but matchup cards showed ESPNfan… owners.

These tests lock the client contract in ``static/redzone.js``:

  * ``_refresh`` captures ``_streamGen`` + scope and discards stale responses
  * mismatched ``newData.scope`` is rejected
  * per-scope ``_scopeCache`` restores the last-good league payload on switch
  * failed scope-switch loads do not clear the skeleton onto foreign state
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RZ = (_ROOT / "static" / "redzone.js").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    needle = "async function %s()" % name if name.startswith("_refresh") else "function %s(" % name
    assert needle in _RZ, "missing %s" % name
    start = _RZ.index(needle)
    # Cut at the next top-level function in the IIFE (2-space indent).
    rest = _RZ[start + 1 :]
    nxt = rest.find("\n  async function ")
    nxt2 = rest.find("\n  function ")
    cuts = [c for c in (nxt, nxt2) if c >= 0]
    end = min(cuts) if cuts else len(rest)
    return _RZ[start : start + 1 + end]


def test_refresh_guards_stale_generation_and_scope():
    src = _fn("_refresh")
    assert "var myGen = _streamGen" in src
    assert "var myScope = _scope" in src
    assert "'&scope=' + myScope" in src or '"&scope=" + myScope' in src
    assert "myGen !== _streamGen || myScope !== _scope" in src
    assert "newData.scope !== myScope" in src
    assert "_scopeCache[myScope] = newData" in src
    # Must not clear loading onto whatever foreign payload is in _state.
    assert "_loadingScope = false; _render()" not in src.replace("_recoverScopeLoad", "")


def test_recover_scope_load_uses_cache_not_foreign_state():
    src = _fn("_recoverScopeLoad")
    assert "_scopeCache[myScope]" in src
    assert "keep the skeleton" in src or "do not paint" in src.lower()
    # Clearing loading without a cache restore is the old bug path.
    assert "_loadingScope = false;\n      _render();" not in src or "_scopeCache" in src


def test_scope_switch_restores_cache_before_fetch():
    block = _RZ[_RZ.index("root.querySelectorAll('.rz-scope-btn')") :]
    block = block[: block.index("root.querySelectorAll('.rz-tab-btn')")]
    assert "_streamGen++" in block
    assert "var cached = _scopeCache[_scope]" in block
    assert "_state = cached" in block
    assert "_loadingScope = true" in block


def test_stream_fallbacks_check_generation():
    src = _fn("_refreshUserStream")
    # Every path that falls back to _refresh must first confirm this stream
    # still owns the screen (otherwise it could fetch/apply after a switch).
    assert src.count("if (myGen !== _streamGen) return") >= 2
    assert "_scopeCache.user = base" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_stale_user_poll_discarded_after_league_switch_node():
    """Simulate overlapping My Leagues poll + This League switch."""
    harness = r"""
// Minimal model of the generation/scope guard in _refresh.
var _streamGen = 0;
var _scope = 'user';
var _state = { scope: 'user', users: [{ display_name: 'ESPNfan5010001561' }] };
var _scopeCache = {
  league: { scope: 'league', users: [{ display_name: 'blackedraw-owner' }] },
  user: _state
};
var _loadingScope = false;
var applied = [];

function apply(newData, myGen, myScope) {
  if (myGen !== _streamGen || myScope !== _scope) return false;
  if (newData && newData.scope && newData.scope !== myScope) return false;
  _state = newData;
  _scopeCache[myScope] = newData;
  _loadingScope = false;
  applied.push(newData.scope + ':' + newData.users[0].display_name);
  return true;
}

// In-flight My Leagues poll captures gen/scope at start.
var pollGen = _streamGen;
var pollScope = _scope;

// User switches back to This League (tab handler).
_streamGen++;
_scope = 'league';
var cached = _scopeCache[_scope];
if (cached) { _state = cached; _loadingScope = false; }
else { _loadingScope = true; }

var leagueGen = _streamGen;
var leagueScope = _scope;

// Late user-scope response arrives first — must be discarded.
var stale = apply(
  { scope: 'user', users: [{ display_name: 'ESPNfan5010001561' }] },
  pollGen, pollScope
);

// Fresh league response lands — must apply.
var ok = apply(
  { scope: 'league', users: [{ display_name: 'sleeper-owner' }] },
  leagueGen, leagueScope
);

var out = {
  staleApplied: stale,
  leagueApplied: ok,
  stateScope: _state.scope,
  stateName: _state.users[0].display_name,
  applied: applied
};
process.stdout.write(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(proc.stdout)
    assert out["staleApplied"] is False
    assert out["leagueApplied"] is True
    assert out["stateScope"] == "league"
    assert out["stateName"] == "sleeper-owner"
    assert out["applied"] == ["league:sleeper-owner"]
