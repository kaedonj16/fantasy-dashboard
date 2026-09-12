"""Source contracts for the Redzone play feed controls and score summary."""

from pathlib import Path
import json
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
REDZONE_JS = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_redzone_feed_is_always_latest_without_sort_switch():
    assert "var list = _chronoSort(filtered);" in REDZONE_JS
    assert "_feedSort" not in REDZONE_JS
    assert "function _softRank(" not in REDZONE_JS
    assert "rz-sort-btn" not in REDZONE_JS
    assert "For You</button>" not in REDZONE_JS


def test_filters_are_applied_before_chronological_sort():
    sync = REDZONE_JS.split("function _syncFeed() {", 1)[1].split(
        "function _renderPagination", 1
    )[0]
    assert sync.index("_feed.filter(_eventMatches)") < sync.index(
        "_chronoSort(filtered)"
    )
    assert "_myTeamOnly" in sync
    assert "_bigPlaysOnly" in sync


def test_page_zero_dom_is_canonically_reconciled_before_animation():
    sync = REDZONE_JS.split("function _syncFeed() {", 1)[1].split(
        "function _renderPagination", 1
    )[0]
    assert "function _orderFeedDom(target, orderedItems)" in sync
    assert sync.count("_orderFeedDom(container, page0Items);") == 1
    assert sync.index("_orderFeedDom(container, page0Items);") < sync.index(
        "var liveInsert ="
    )
    assert "setTimeout(function()" not in sync


def test_play_summary_groups_clock_delta_and_labeled_total():
    delta_markup = REDZONE_JS.split(
        "'<div class=\"rz-event-delta ' + deltaCls + '\">'", 1
    )[1][:500]

    assert "rz-event-clock" in delta_markup
    assert "rz-event-score" in delta_markup
    assert delta_markup.index("rz-event-score") < delta_markup.index(
        "rz-event-delta-pts"
    )
    assert "rz-event-delta-pts" in delta_markup
    assert "rz-event-total" in delta_markup
    assert "</span> total" in delta_markup
    assert ".rz-event-delta {" in DASHBOARD_CSS
    assert "border-radius: 10px;" in DASHBOARD_CSS
    assert ".rz-event-delta-game {" in DASHBOARD_CSS


def test_latest_order_is_comparable_across_simultaneous_games():
    """Quarter/clock wall time must win over game-local provider sequence."""
    chrono_key = REDZONE_JS.split("function _chronoKey(ev) {", 1)[1].split(
        "function _chronoSort(list) {", 1
    )[0]

    assert chrono_key.index("_playWallTime(kickoff, ev.gameQuarter, ev.gameClock)") < chrono_key.index(
        "if (ev.gameId && ev.seq != null)"
    )
    assert "return list.slice().sort" in REDZONE_JS
    assert "var ga = _stableString(a.gameId)" in REDZONE_JS


def test_selected_game_board_enriches_situation_and_mirrors_home_logo():
    game_info = REDZONE_JS.split("function _nflGameInfo(gid) {", 1)[1].split(
        "var _POS_LIST", 1
    )[0]
    board = REDZONE_JS.split("function _renderNflBoard() {", 1)[1].split(
        "function _renderFilterChips()", 1
    )[0]

    assert "if (games[gid]) return games[gid]" not in game_info
    assert "plays.slice().sort" in game_info
    assert "return bs - as" in game_info
    assert "if (play.team) return play.team" in game_info
    assert "info[pid] ? (info[pid].team || '')" in game_info
    assert "row.possession = playTeam(best)" in game_info
    assert "? ball + meta + logo(abv)" in board
    assert "Situation pending" not in REDZONE_JS
    assert "Possession pending" not in REDZONE_JS


def test_refresh_uses_fresh_canonical_incremental_update_and_no_browser_cache():
    refresh = REDZONE_JS.split("async function _refresh() {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    assert "fetch(url, { cache: 'no-store', signal:" in refresh
    assert "AbortController" in refresh
    detect_at = refresh.index("_detectChanges(newData, wasContinuouslyActive ? 'live' : 'bulk');")
    assert detect_at < refresh.index("if (wasLoading) _render(); else _partialUpdate();", detect_at)
    assert "savedFeedHtml" not in refresh
    assert "newFeedEl.innerHTML" not in refresh


def test_pbp_chronology_is_stable_and_separate_from_detection_time():
    pbp = REDZONE_JS.split("function _eventsFromPbp(", 1)[1].split(
        "function _detectChanges", 1
    )[0]
    assert "playSortTs: group.playSortTs" in pbp
    assert "detectedAt: group.detectedAt" in pbp
    assert "ts: Date.now() +" not in pbp
    assert "if (group.playSortTs == null && c.playSortTs != null)" in pbp


def test_animation_modes_preserve_nodes_and_gate_live_fx():
    sync = REDZONE_JS.split("function _syncFeed() {", 1)[1].split(
        "function _renderPagination", 1
    )[0]
    assert "_animationMode = 'bulk'" in REDZONE_JS
    assert "mode === 'live' && inserted.length === 1" in sync
    assert "mode === 'bulk'" in sync
    assert "node.innerHTML = fresh.innerHTML" in sync
    assert "container.appendChild(node)" in sync
    assert "oldPositions[node.dataset.eid] - node.getBoundingClientRect().top" in sync
    assert "Math.min(index * 25, 200)" in sync
    assert "prefers-reduced-motion: reduce" in REDZONE_JS
    assert ".rz-event.is-live-enter" in DASHBOARD_CSS
    assert ".rz-event.is-bulk-enter" in DASHBOARD_CSS


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_canonical_sort_handles_cross_game_clock_forms_and_stable_ties():
    helpers = REDZONE_JS[REDZONE_JS.index("function _clockSecs") : REDZONE_JS.index("function _fmtQuarter")]
    script = f"""
var _state = {{games: {{
  early: {{game_time_epoch: 1000}}, late: {{game_time_epoch: 1120}}
}}, player_info: {{}}}};
{helpers}
var rows = [
  {{playId:'late-play', fromPbp:true, gameId:'late', gameQuarter:'Q1', gameClock:'15:00', seq:1}},
  {{playId:'early-newer', fromPbp:true, gameId:'early', gameQuarter:1, gameClock:'9:04', seq:8}},
  {{playId:'early-older', fromPbp:true, gameId:'early', gameQuarter:'1', gameClock:'15:00', seq:2}},
  {{playId:'ot', fromPbp:true, gameId:'early', gameQuarter:'OT', gameClock:'10:00', seq:90}}
];
var forward = _chronoSort(rows).map(function(x){{return x.playId;}});
var reverse = _chronoSort(rows.slice().reverse()).map(function(x){{return x.playId;}});
console.log(JSON.stringify({{forward:forward, reverse:reverse}}));
"""
    result = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert result["forward"] == ["ot", "early-newer", "late-play", "early-older"]
    assert result["reverse"] == result["forward"]


def test_scope_runtime_normalizes_feed_and_render_state_is_scope_local():
    save = REDZONE_JS.split("function _saveScopeRuntime", 1)[1].split(
        "function _restoreScopeRuntime", 1
    )[0]
    restore = REDZONE_JS.split("function _restoreScopeRuntime", 1)[1].split(
        "function _hydrateFeed", 1
    )[0]
    assert "feed: _chronoSort(_feed)" in save
    assert "_feed = _chronoSort(saved.feed)" in restore
    assert "_shownFeedIdsByScope[scope]" in save
    assert "_shownFeedIds = new Set(_feed.map(_eid))" in restore


def test_failed_refresh_recovers_without_rebuilding_mounted_controls():
    refresh = REDZONE_JS.split("async function _refresh() {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    assert refresh.count("_recoverScopeLoad(myGen, myScope);") == 2
    assert refresh.count("if (myGen === _streamGen && myScope === _scope && !_loadingScope) _partialUpdate();") == 2
