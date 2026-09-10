"""Redzone This League ↔ My Leagues scope + Plays feed contracts.

Locks ``static/redzone.js`` so:

  * a late My Leagues poll cannot paint portfolio teams under This League
  * My Leagues stream hydrates Plays at end (seed-only left the feed empty)
  * scope switch resets feed snapshots so the other scope can rehydrate
  * empty feed + hero focus shows the schedule, not "No matching plays"
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _rz() -> str:
    return (_ROOT / "static" / "redzone.js").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    src = _rz()
    needle = (
        "async function %s()" % name
        if name.startswith("_refresh")
        else "function %s(" % name
    )
    assert needle in src, "missing %s" % name
    start = src.index(needle)
    rest = src[start + 1 :]
    nxt = rest.find("\n  async function ")
    nxt2 = rest.find("\n  function ")
    cuts = [c for c in (nxt, nxt2) if c >= 0]
    end = min(cuts) if cuts else len(rest)
    return src[start : start + 1 + end]


def test_refresh_guards_stale_generation_and_scope():
    src = _fn("_refresh")
    assert "var myGen = _streamGen" in src
    assert "var myScope = _scope" in src
    assert "'&scope=' + myScope" in src or '"&scope=" + myScope' in src
    assert "myGen !== _streamGen || myScope !== _scope" in src
    assert "newData.scope !== myScope" in src
    assert "_scopeCache[myScope] = newData" in src
    # State must be applied before detect so owner/league labels are correct.
    assert src.index("_state = newData") < src.index("_detectChanges(newData)")
    assert "_loadingScope = false; _render()" not in src.replace("_recoverScopeLoad", "")


def test_recover_scope_load_uses_cache_not_foreign_state():
    src = _fn("_recoverScopeLoad")
    assert "_scopeCache[myScope]" in src
    assert "keep the skeleton" in src or "do not paint" in src.lower()
    assert "_loadingScope = false;\n      _render();" not in src or "_scopeCache" in src


def test_scope_switch_restores_cache_and_resets_feed_snapshots():
    src = _rz()
    block = src[src.index("root.querySelectorAll('.rz-scope-btn')") :]
    block = block[: block.index("root.querySelectorAll('.rz-tab-btn')")]
    assert "_streamGen++" in block
    assert "var cached = _scopeCache[_scope]" in block
    assert "_state = cached" in block
    assert "_loadingScope = true" in block
    assert "_resetFeedSnapshots()" in block


def test_stream_fallbacks_check_generation():
    src = _fn("_refreshUserStream")
    assert src.count("if (myGen !== _streamGen) return") >= 2
    assert "_scopeCache.user = base" in src


def test_my_leagues_stream_hydrates_plays_at_end():
    src = _fn("_refreshUserStream")
    # Mid-stream must not seed prevStats (that suppressed all Plays).
    mid = src[src.index("obj.type === 'league'") : src.index("_hydrateFeed(base)")]
    assert "_seedPrevStats(base)" not in mid
    assert "_seedMilestones(base)" in mid
    assert "_hydrateFeed(base)" in src
    assert "_resetFeedSnapshots()" in src


def test_hydrate_feed_matches_cold_boot_order():
    src = _fn("_hydrateFeed")
    assert src.index("_seedMilestones") < src.index("_detectChanges")
    assert src.index("_detectChanges") < src.index("_seedPrevStats")


def test_empty_feed_with_hero_shows_schedule_not_no_matching():
    src = _fn("_syncFeed")
    assert "hardFilter" in src
    assert "_heroMid && _feed.length > 0" in src
    assert "_pregameScheduleHtml()" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_stale_user_poll_discarded_after_league_switch_node():
    harness = r"""
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

var pollGen = _streamGen;
var pollScope = _scope;

_streamGen++;
_scope = 'league';
var cached = _scopeCache[_scope];
if (cached) { _state = cached; _loadingScope = false; }
else { _loadingScope = true; }

var leagueGen = _streamGen;
var leagueScope = _scope;

var stale = apply(
  { scope: 'user', users: [{ display_name: 'ESPNfan5010001561' }] },
  pollGen, pollScope
);
var ok = apply(
  { scope: 'league', users: [{ display_name: 'sleeper-owner' }] },
  leagueGen, leagueScope
);

process.stdout.write(JSON.stringify({
  staleApplied: stale,
  leagueApplied: ok,
  stateScope: _state.scope,
  stateName: _state.users[0].display_name,
  applied: applied
}));
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


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_stream_end_hydrate_fills_plays_node():
    """Seed-only mid-stream left Plays empty; end hydrate must fill it."""
    harness = r"""
var _prevStats = { 'OLD': { rush_yds: 1 } };
var _prevPts = {};
var _milestonesSeen = {};
var _blowoutSeen = {};
var _prevInjury = {};
var _prevLeader = {};
var _prevMatchupPts = {};
var _scoreDelta = { me: 0, opp: 0 };
var _flashRids = new Set();
var _feed = [];

function _resetFeedSnapshots() {
  _prevStats = {}; _prevPts = {}; _milestonesSeen = {}; _blowoutSeen = {};
  _prevInjury = {}; _prevLeader = {}; _prevMatchupPts = {};
  _scoreDelta = { me: 0, opp: 0 }; _flashRids = new Set();
}
function _seedMilestones() {}
function _seedInjuries() {}
function _seedLeaders() {}
function _seedPrevStats(data) {
  Object.keys(data.player_info || {}).forEach(function(pid) {
    var sl = data.player_info[pid].stat_line;
    if (sl) _prevStats[pid] = Object.assign({}, sl);
  });
}
function _detectChanges(data) {
  Object.keys(data.player_info || {}).forEach(function(pid) {
    var neu = data.player_info[pid].stat_line || {};
    var old = _prevStats[pid] || {};
    var delta = (neu.rush_yds || 0) - (old.rush_yds || 0);
    if (delta > 0) _feed.push({ pid: pid, desc: delta + ' rush yds' });
  });
}
function _hydrateFeed(data) {
  _seedMilestones(data); _seedInjuries(data); _seedLeaders(data);
  (data.matchups || []).forEach(function(m) {
    _prevMatchupPts[String(m.roster_id)] = parseFloat(m.points || 0);
  });
  _detectChanges(data);
  _seedPrevStats(data);
}

_resetFeedSnapshots();
_feed = [];
var base = {
  matchups: [{ roster_id: '0:1', points: 10 }],
  player_info: { '111': { stat_line: { rush_yds: 55, rush_td: 1 } } }
};
_seedPrevStats(base); // old mid-stream seed
var feedAfterSeedOnly = _feed.slice();

_resetFeedSnapshots();
_feed = [];
_hydrateFeed(base);

process.stdout.write(JSON.stringify({
  seedOnlyFeed: feedAfterSeedOnly.length,
  hydratedFeed: _feed.length,
  desc: _feed[0] && _feed[0].desc,
  seededAfter: !!_prevStats['111']
}));
"""
    proc = subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(proc.stdout)
    assert out["seedOnlyFeed"] == 0
    assert out["hydratedFeed"] == 1
    assert out["desc"] == "55 rush yds"
    assert out["seededAfter"] is True
