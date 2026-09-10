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
    # Cached switch must rehydrate Plays immediately and persist scope in the URL.
    assert "_hydrateFeed(cached)" in block
    assert "_syncScopeUrl()" in block


def test_stream_fallbacks_check_generation():
    src = _fn("_refreshUserStream")
    assert src.count("if (myGen !== _streamGen) return") >= 2
    assert "_scopeCache.user = base" in src


def test_my_leagues_stream_hydrates_plays_at_end():
    src = _fn("_refreshUserStream")
    full = _rz()
    # Mid-stream must not seed prevStats (that suppressed all Plays).
    mid = src[src.index("obj.type === 'league'") : src.index("_hydrateFeed(base)")]
    assert "_seedPrevStats(base)" not in mid
    assert "_hydrateFeed(base)" in src
    assert "_resetFeedSnapshots()" in src
    # Meta must not wipe a last-good portfolio into an empty cache shell.
    assert "last-good portfolio" in src
    # Empty slices surface a failed card, not a silent drop.
    assert "_mlFailed" in src
    assert "function _mlFailedCard(" in full


def test_hydrate_feed_matches_cold_boot_order():
    src = _fn("_hydrateFeed")
    assert src.index("_seedMilestones") < src.index("_detectChanges")
    assert src.index("_detectChanges") < src.index("_seedPrevStats")


def test_empty_feed_with_hero_shows_schedule_not_no_matching():
    src = _fn("_syncFeed")
    assert "hardFilter" in src
    assert "_heroMid && _feed.length > 0" in src
    assert "_pregameScheduleHtml()" in src


def test_focused_pair_drives_scorebar_and_live_chrome():
    src = _rz()
    assert "function _focusedPair(" in src
    assert "function _matchupIsLive(" in src
    assert "_matchupIsLive([_mm, _om])" in src or "_matchupIsLive([myMatchup" in src
    # Non-viewer heroes should not hard-code the "Me" label.
    assert "(pair && pair.isMine) ? 'Me'" in src or "pair.isMine ? 'Me'" in src


def test_filter_polish_league_label_and_my_team_chip():
    src = _rz()
    # Fantasy Team/League filter row is gone; NFL filter is matchups.
    assert "fpRow('Matchup', 'nfl'" in src
    assert "? 'League' : 'Team'" not in src
    assert 'data-clear-myteam="1"' in src
    assert "OPP · " in src


def test_nfl_filter_is_matchups_with_scoreboard():
    """RZ6.3 / RZ6.4 — NFL filter lists games; selected game shows board strip."""
    src = _rz()
    assert "function _nflMatchupOptions(" in src
    assert "function _nflOptions(" not in src
    assert "function _renderNflBoard(" in src
    nfl = _fn("_nflMatchupOptions")
    assert "game_id" in nfl
    assert "away + ' @ ' + home" in nfl or 'away + " @ " + home' in nfl
    # Must not list bare team abbrevs as the primary option value.
    assert "seen[t] = 1" not in nfl
    board = _fn("_renderNflBoard")
    assert "rz-nfl-board" in board
    assert "rz-nfl-ball" in board
    assert "possession" in board
    assert "_nflBoardSitLine" in board or "_downDist" in board
    assert "function _nflBoardSitLine(" in src
    assert "_downDist" in _fn("_nflBoardSitLine")
    assert "_renderNflBoard()" in src
    app = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert "build_games_snapshot as _rz_build_games_snapshot" in app
    assert '"games": games' in app


def test_scoring_prefers_roster_league_over_pid_map():
    src = _fn("_detectChanges")
    assert "mm.league_id" in src
    assert "tags.pidToRoster" in src


def test_user_scope_fetch_fallback_keeps_user_scope():
    app = (_ROOT / "app.py").read_text(encoding="utf-8")
    block = app[app.index("def _redzone_fetch(") : app.index("def _redzone_user_portfolio")]
    assert "portfolio_unavailable" in block
    assert '"scope": "user"' in block
    # Must not fall through to league collect after a user-scope failure.
    assert block.index("portfolio_unavailable") < block.index("_redzone_collect(")


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


def test_default_hero_filter_disabled():
    src = _fn("_applyDefaultHero")
    assert "return;" in src
    assert "_heroMid = def" not in src
    assert _fn("_defaultHeroMid").count("return null") >= 1


def test_demo_header_button_removed():
    src = _rz()
    assert 'class="rz-demo-btn"' not in src
    # Exit Demo stays available when already in demo mode.
    assert "Exit Demo" in src or "rz-demo-exit" in src


def test_play_descriptions_are_play_by_play():
    src = _fn("_describe")
    assert "Throws a " in src and "touchdown pass" in src
    assert "Hauls in a " in src and "touchdown catch" in src
    assert "Completes a pass for " in src
    # Old bulk shorthand should not be returned as live copy (comment examples OK).
    assert "return { desc: ry + ' yd TD catch'" not in src
    assert "return { desc: py + ' yd TD pass'" not in src
    assert "return { desc: uy + ' yd TD run'" not in src


def test_pbp_feed_path_and_soft_rank():
    src = _rz()
    assert "function _eventsFromPbp(" in src
    assert "pbp_by_game" in src
    assert "function _softRank(" in src
    assert "function _isBigPlay(" in src
    assert 'id="rz-bigplays-btn"' in src
    assert "function _loadPrefs(" in src
    assert "function _savePrefs(" in src
    assert "rz-event-impact" in src
    assert "Try the Redzone demo" in src


def test_app_wires_pbp_into_collect_and_demo():
    app = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert "extract_pbp_plays as _rz_extract_pbp_plays" in app
    assert 'play_by_play=want_pbp' in app or "play_by_play=want_pbp" in app
    assert '"pbp_by_game": pbp_by_game' in app
    assert "Try Redzone Demo" in app
    api = (_ROOT / "dashboard_services" / "api.py").read_text(encoding="utf-8")
    assert 'fetch_tank_boxscore' in api
    # Tank01 docs use both spellings; fantasyPoints helps per-play deltas.
    assert 'params["playByPlay"] = "true"' in api
    assert 'params["playByplay"] = "true"' in api
    assert 'params["fantasyPoints"] = "true"' in api
    # Finals must still request PBP — otherwise Plays falls back to bulk
    # "Scored X pts" cards after the game ends.
    assert 'want_pbp = live or final' in app
    assert 'ttl=(None if live else 300.0)' in app
    assert "Finals: plain boxscore is enough" not in app


def test_pbp_coverage_suppresses_bulk_points_without_new_events():
    """Live/final and PBP-covered games never invent Scored / boxscore fiction."""
    src = _fn("_detectChanges")
    assert "Scored ' + delta.toFixed(1) + ' pts'" in src
    # Live/final games never take the bulk Scored fallback or _playsFromDiff.
    assert "code === '1' || code === '2'" in src
    assert "_pbpGames[gid]" in src
    # Boxscore narrative is no longer a live/final substitute for real PBP.
    assert "pbpRows.length" not in src
    assert "_playsFromDiff" in src  # still used for pregame / non-PBP only


def test_app_falls_back_to_plain_boxscore_when_pbp_empty():
    app = (_ROOT / "app.py").read_text(encoding="utf-8")
    # Plain boxscore merge remains for scoreboard totals only.
    assert "play_by_play=False" in app
    assert "pbp_by_game[gid] = plays" in app
    assert "if plays:" not in app[app.index("pbp_by_game[gid] = plays") - 80:
                                    app.index("pbp_by_game[gid] = plays") + 40]
    # Client is responsible for PBP-lines-only; server must not claim narrative diffs.
    assert "narrative diffs" not in app
    # Alternate sources (Sleeper → ESPN) when Tank01 PBP is empty.
    assert "fetch_alt_pbp_plays as _rz_fetch_alt_pbp_plays" in app
    assert "_rz_fetch_alt_pbp_plays" in app


def test_events_from_pbp_resolves_name_when_pid_missing():
    src = _fn("_eventsFromPbp")
    assert "_pidFromPlayName" in (_ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
    assert "fromPbp: true" in src


def test_live_final_empty_feed_is_honest_not_boxscore():
    src = _fn("_syncFeed")
    assert "Play-by-play lines" in src
    assert "not box-score summaries" in src


def test_filter_panel_is_matchups_not_fantasy_teams():
    src = _rz()
    assert "function _nflMatchupOptions(" in src
    assert "fpRow('Matchup', 'nfl'" in src
    assert "fpRow(_scope === 'user' ? 'League' : 'Team', 'team'" not in src
    assert "function _teamOptions(" not in src
    assert "function _nflOptions(" not in src
