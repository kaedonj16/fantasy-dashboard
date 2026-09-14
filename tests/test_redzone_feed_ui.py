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
    # Both teams render as logo + abbr blocks; the home block is mirrored (logo
    # outermost) via CSS row-reverse rather than a JS branch.
    assert "team(away, aPts, awayPoss, 'away')" in board
    assert "team(home, hPts, homePoss, 'home')" in board
    assert "flex-direction: row-reverse" in DASHBOARD_CSS
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


def test_grouped_play_ownership_tracks_headline_actor_not_any_contributor():
    """A grouped NFL play is 'mine' only when the headline (primary) actor is
    mine -- not when any background contributor is. Otherwise your QB completing
    to an opponent's WR headlines the opponent yet reads as MY TEAM and leaks
    into the My Team filter."""
    pbp = REDZONE_JS.split("function _eventsFromPbp(", 1)[1].split(
        "function _detectChanges", 1
    )[0]
    events_block = pbp.split("var event = {", 1)[1][:400]
    assert "mine: !!primary.mine" in events_block
    assert "opp: !!primary.opp" in events_block
    # The discarded "any contributor" ownership must not drive the event flags.
    assert "mine: allMine" not in pbp
    assert "opp: allOpp" not in pbp
    # The My Team filter keys off that per-event flag.
    assert "if (_myTeamOnly && !ev.mine) return false;" in REDZONE_JS


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_primary_actor_is_receiver_so_ownership_follows_the_catch():
    """_selectPrimaryActor headlines the receiver over the passer, so a play
    where only the passer is mine resolves to a not-mine event."""
    helper = REDZONE_JS[
        REDZONE_JS.index("function _selectPrimaryActor")
        : REDZONE_JS.index("function _isPlayNullified")
    ]
    script = f"""
{helper}
var qbMine = {{pid:'qb', mine:true, opp:false, line:{{pass_yds:15}}}};
var wrOpp  = {{pid:'wr', mine:false, opp:true, line:{{rec:1, rec_yds:15}}}};
var primary = _selectPrimaryActor([qbMine, wrOpp]);
console.log(JSON.stringify({{
  pid: primary.pid,
  eventMine: !!primary.mine,
  eventOpp: !!primary.opp
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert out == {"pid": "wr", "eventMine": False, "eventOpp": True}


def test_on_deck_is_a_gridded_section_with_count_and_matches_gutter():
    """On Deck is a header + responsive grid (tidy when many leagues connect),
    aligned to the 14px content gutter like the cards below it."""
    ondeck = REDZONE_JS.split("function _onDeckHtml() {", 1)[1].split(
        "function _pregameScheduleHtml", 1
    )[0]
    assert "rz-ondeck-head" in ondeck
    assert "rz-ondeck-count" in ondeck
    assert "rz-ondeck-grid" in ondeck
    assert "rz-ondeck-item-top" in ondeck
    assert "' game' : ' games'" in ondeck
    assert ".rz-ondeck-bar {\n    margin: 0 14px 10px;" in DASHBOARD_CSS
    assert "grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));" in DASHBOARD_CSS


def test_matchup_strip_and_board_share_the_content_gutter():
    """The two sections that used to run wider than the rest now sit on the same
    14px gutter as the scope toggle and main card."""
    assert ".rz-hero-cards {\n    display: block;\n    /* Match the 14px" in DASHBOARD_CSS
    assert "margin: 0 14px;" in DASHBOARD_CSS
    assert ".rz-nfl-board {" in DASHBOARD_CSS
    board_css = DASHBOARD_CSS.split(".rz-nfl-board {", 1)[1][:300]
    assert "margin: 0 12px 10px;" in board_css


def test_nfl_board_hides_score_until_kickoff():
    """Pregame shows kickoff, not '–' score placeholders; scores render only once
    the game is underway."""
    board = REDZONE_JS.split("function _renderNflBoard()", 1)[1].split(
        "function _gamePillStatus", 1
    )[0]
    assert "var showScore = !isPre;" in board
    assert "showScore ? '<span class=\"rz-nfl-score\">'" in board
    assert "rz-nfl-kick-time" in board
    assert "rz-nfl-live-pill" in board
    # No leftover em-dash score placeholder.
    assert "'–'" not in board


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
    assert refresh.count("_recoverScopeLoad(myGen, myScope);") == 3
    assert refresh.count("if (myGen === _streamGen && myScope === _scope && !_loadingScope) _partialUpdate();") == 3

@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_field_position_handles_sides_aliases_boundaries_and_hidden_states():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    helpers = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("window._rzRenderGameBoard")]
    script = f"""
var window={{}};
{helpers}
function fp(x){{ return window._rzFieldPosition(Object.assign({{status:'live',field_position_reliable:true,away:'MIA',home:'OAK'}},x)); }}
console.log(JSON.stringify([
 fp({{possession:'MIA',yard_line:'MIA 31'}}),
 fp({{possession:'LV',yard_line:'LV 31'}}),
 fp({{possession:'MIA',yard_line:'LV 31'}}),
 fp({{possession:'LV',yard_line:'MIA 31'}}),
 fp({{possession:'MIA',yard_line:'50'}}),
 fp({{possession:'MIA',yard_line:'LV 1'}}),
 fp({{possession:'MIA',yard_line:'MIA 1'}}),
 fp({{status:'halftime',possession:'MIA',yard_line:'MIA 20'}}),
 fp({{field_position_reliable:false,possession:'MIA',yard_line:'MIA 20'}}),
 fp({{possession:'XXX',yard_line:'MIA 20'}})
]));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert out == [
        {"spot": 31, "side": "away"}, {"spot": 69, "side": "home"},
        {"spot": 69, "side": "away"}, {"spot": 31, "side": "home"},
        {"spot": 50, "side": "away"}, {"spot": 99, "side": "away"},
        {"spot": 1, "side": "away"}, None, None, None,
    ]


def test_shared_board_and_modal_refresh_contract():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "window._rzRenderGameBoard(game, { modal:true })" in app_js
    assert "field_position_reliable" in app_js
    assert "requestGeneration === _generation" in app_js
    assert "setTimeout(function tick()" in app_js
    assert "Live box-score stats are temporarily unavailable." in app_js
    assert "Stat breakdown appears once the game is underway." not in app_js
    assert ".rz-field-fill.is-away" in DASHBOARD_CSS
    assert ".rz-field-fill.is-home" in DASHBOARD_CSS

@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_player_modal_log_uses_each_players_latest_canonical_contribution():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    helper = app_js[app_js.index("window._rzPlayerLogEvents") : app_js.index("// Player summary renderer.")]
    script = f"""
var window={{}};
{helper}
var completion={{playId:'p1',gameId:'g',seq:1,playSortTs:1,gameQuarter:'Q1',gameClock:'12:00',pid:'wr',desc:'10-yard reception',playState:'VALID',contributions:[
  {{pid:101,name:'Quarter Back',pos:'QB',line:{{pass_yds:10,pass_td:0}},pts:0.4,kind:'gain'}},
  {{pid:'202',name:'Justin Jefferson',pos:'WR',line:{{rec:1,rec_yds:10}},pts:2,kind:'gain'}}
]}};
var revised=JSON.parse(JSON.stringify(completion)); revised.seq=2; revised.playSortTs=2;
revised.contributions[0].line.pass_yds=12; revised.contributions[0].pts=.48;
revised.contributions[1].line.rec_yds=12; revised.contributions[1].pts=2.2;
var zero={{playId:'p0',seq:3,playSortTs:3,pid:'wr',desc:'Reception',playState:'VALID',contributions:[
  {{pid:'101',name:'Quarter Back',pos:'QB',line:{{pass_yds:0}},pts:0}},
  {{pid:'202',name:'Justin Jefferson',pos:'WR',line:{{rec:1,rec_yds:0}},pts:1}}
]}};
var loss={{playId:'pn',seq:4,playSortTs:4,pid:'wr',desc:'loss',playState:'VALID',contributions:[
  {{pid:'101',name:'Quarter Back',pos:'QB',line:{{pass_yds:-3}},pts:-.12}},
  {{pid:'202',name:'Justin Jefferson',pos:'WR',line:{{rec:1,rec_yds:-3}},pts:.7}}
]}};
var td={{playId:'td',seq:5,playSortTs:5,pid:'wr',desc:'TD reception',playState:'VALID',contributions:[
  {{pid:'101',name:'Quarter Back',pos:'QB',line:{{pass_yds:10,pass_td:1}},pts:4.4}},
  {{pid:'202',name:'Justin Jefferson',pos:'WR',line:{{rec:1,rec_yds:10,rec_td:1}},pts:8,kind:'td'}}
]}};
var nul=JSON.parse(JSON.stringify(td)); nul.playId='nullified'; nul.playState='NULLIFIED'; nul.isNullified=true; nul.desc='Play nullified by penalty'; nul.contributions.forEach(c=>c.isInvalid=true);
var qb=window._rzPlayerLogEvents('101',[completion,revised,zero,loss,td,nul]);
var wr=window._rzPlayerLogEvents(202,[completion,revised,zero,loss]);
var legacy=window._rzPlayerLogEvents('101',[{{pid:101,playId:'legacy',seq:9,desc:'Demo pass',pts:1}}]);
console.log(JSON.stringify({{qb:qb,wr:wr,legacy:legacy}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    qb = {row["playId"]: row for row in out["qb"]}
    wr = {row["playId"]: row for row in out["wr"]}
    assert qb["p1"]["pts"] == 0.48 and qb["p1"]["desc"] == "12-yard completion to Justin Jefferson"
    assert wr["p1"]["pts"] == 2.2 and wr["p1"]["desc"] == "10-yard reception"
    assert qb["p0"]["desc"] == "0-yard completion to Justin Jefferson"
    assert qb["pn"]["desc"] == "-3-yard completion to Justin Jefferson"
    assert qb["td"]["pts"] == 4.4 and qb["td"]["desc"] == "10-yard TD pass"
    assert qb["nullified"]["pts"] == 0 and qb["nullified"]["isNullified"] is True
    assert qb["nullified"]["desc"] == "Play nullified by penalty"
    assert len(out["legacy"]) == 1


def test_modal_history_comes_from_uncapped_latest_play_groups():
    redzone = REDZONE_JS.split("function _modalPlayHistory()", 1)[1].split(
        "window.__rzGetPlayerLive", 1
    )[0]
    assert "Object.keys(_playGroupsByKey)" in redzone
    assert "primaryEvent" in redzone
    assert "_PAGE_SIZE" not in redzone
    assert "_feed.filter(function(ev) { return !ev.fromPbp; })" in redzone
    live = REDZONE_JS.split("window.__rzGetPlayerLive = function(pid)", 1)[1][:180]
    assert "_modalPlayHistory()" in live
