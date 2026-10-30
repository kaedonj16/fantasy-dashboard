"""Source contracts for the ScoreZone play feed controls and score summary."""

from pathlib import Path
import json
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCOREZONE_JS = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_scorezone_feed_is_always_latest_without_sort_switch():
    assert "var list = _chronoSort(filtered);" in SCOREZONE_JS
    assert "_feedSort" not in SCOREZONE_JS
    assert "function _softRank(" not in SCOREZONE_JS
    assert "rz-sort-btn" not in SCOREZONE_JS
    assert "For You</button>" not in SCOREZONE_JS


def test_filters_are_applied_before_chronological_sort():
    sync = SCOREZONE_JS.split("function _syncFeed() {", 1)[1].split(
        "function _renderPagination", 1
    )[0]
    assert sync.index("_feed.filter(_eventMatches)") < sync.index(
        "_chronoSort(filtered)"
    )
    assert "_myTeamOnly" in sync
    assert "_bigPlaysOnly" in sync


def test_page_zero_dom_is_canonically_reconciled_before_animation():
    sync = SCOREZONE_JS.split("function _syncFeed() {", 1)[1].split(
        "function _renderPagination", 1
    )[0]
    assert "function _orderFeedDom(target, orderedItems)" in sync
    assert sync.count("_orderFeedDom(container, page0Items);") == 1
    assert sync.index("_orderFeedDom(container, page0Items);") < sync.index(
        "var liveInsert ="
    )
    assert "setTimeout(function()" not in sync


def test_play_summary_groups_clock_delta_and_labeled_total():
    delta_markup = SCOREZONE_JS.split(
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
    assert "border-radius: var(--radius);" in DASHBOARD_CSS
    assert ".rz-event-delta-game {" in DASHBOARD_CSS


def test_latest_order_is_comparable_across_simultaneous_games():
    """Quarter/clock wall time must win over game-local provider sequence."""
    chrono_key = SCOREZONE_JS.split("function _chronoKey(ev) {", 1)[1].split(
        "function _chronoSort(list) {", 1
    )[0]

    assert chrono_key.index("_playWallTime(kickoff, ev.gameQuarter, ev.gameClock)") < chrono_key.index(
        "if (ev.gameId && ev.seq != null)"
    )
    assert "return list.slice().sort" in SCOREZONE_JS
    assert "var ga = _stableString(a.gameId)" in SCOREZONE_JS


def test_selected_game_board_enriches_situation_and_mirrors_home_logo():
    game_info = SCOREZONE_JS.split("function _nflGameInfo(gid) {", 1)[1].split(
        "var _POS_LIST", 1
    )[0]
    board = SCOREZONE_JS.split("function _renderNflBoard() {", 1)[1].split(
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
    assert "Situation pending" not in SCOREZONE_JS
    assert "Possession pending" not in SCOREZONE_JS


def test_refresh_uses_fresh_canonical_incremental_update_and_no_browser_cache():
    refresh = SCOREZONE_JS.split("async function _refresh(opts) {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    assert "fetch(url, { cache: 'no-store', signal:" in refresh
    assert "AbortController" in refresh
    detect_at = refresh.index(
        "_detectChanges(newData, (!opts.backfill && wasContinuouslyActive) ? 'live' : 'bulk');"
    )
    assert detect_at < refresh.index("if (wasLoading) _render(); else _partialUpdate();", detect_at)
    assert "savedFeedHtml" not in refresh
    assert "newFeedEl.innerHTML" not in refresh


def test_pbp_chronology_is_stable_and_separate_from_detection_time():
    pbp = SCOREZONE_JS.split("function _eventsFromPbp(", 1)[1].split(
        "function _detectChanges", 1
    )[0]
    assert "playSortTs: group.playSortTs" in pbp
    assert "detectedAt: group.detectedAt" in pbp
    assert "ts: Date.now() +" not in pbp
    assert "if (group.playSortTs == null && c.playSortTs != null)" in pbp


def test_animation_modes_preserve_nodes_and_gate_live_fx():
    sync = SCOREZONE_JS.split("function _syncFeed() {", 1)[1].split(
        "function _renderPagination", 1
    )[0]
    assert "_animationMode = 'bulk'" in SCOREZONE_JS
    assert "mode === 'live' && inserted.length === 1" in sync
    assert "mode === 'bulk'" in sync
    assert "node.innerHTML = fresh.innerHTML" in sync
    assert "container.appendChild(node)" in sync
    assert "oldPositions[node.dataset.eid] - node.getBoundingClientRect().top" in sync
    assert "Math.min(index * 25, 200)" in sync
    assert "prefers-reduced-motion: reduce" in SCOREZONE_JS
    assert ".rz-event.is-live-enter" in DASHBOARD_CSS
    assert ".rz-event.is-bulk-enter" in DASHBOARD_CSS


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_canonical_sort_handles_cross_game_clock_forms_and_stable_ties():
    helpers = SCOREZONE_JS[SCOREZONE_JS.index("function _clockSecs") : SCOREZONE_JS.index("function _fmtQuarter")]
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
    pbp = SCOREZONE_JS.split("function _eventsFromPbp(", 1)[1].split(
        "function _detectChanges", 1
    )[0]
    events_block = pbp.split("var event = {", 1)[1][:400]
    assert "mine: !!primary.mine" in events_block
    assert "opp: !!primary.opp" in events_block
    # The discarded "any contributor" ownership must not drive the event flags.
    assert "mine: allMine" not in pbp
    assert "opp: allOpp" not in pbp
    # The My Team filter keys off that per-event flag.
    assert "if (_myTeamOnly && !ev.mine) return false;" in SCOREZONE_JS


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_primary_actor_is_receiver_so_ownership_follows_the_catch():
    """_selectPrimaryActor headlines the receiver over the passer, so a play
    where only the passer is mine resolves to a not-mine event."""
    helper = SCOREZONE_JS[
        SCOREZONE_JS.index("function _selectPrimaryActor")
        : SCOREZONE_JS.index("function _isPlayNullified")
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
    ondeck = SCOREZONE_JS.split("function _onDeckHtml() {", 1)[1].split(
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
    board = SCOREZONE_JS.split("function _renderNflBoard()", 1)[1].split(
        "function _gamePillStatus", 1
    )[0]
    assert "var showScore = !isPre;" in board
    assert "showScore ? '<span class=\"rz-nfl-score'" in board
    # The possessing team's score carries the TD pop hook on a TD arrival.
    assert "rz-fb-score-pop" in board
    assert "rz-nfl-kick-time" in board
    assert "rz-nfl-live-pill" in board
    # No leftover em-dash score placeholder.
    assert "'–'" not in board


def test_scope_runtime_normalizes_feed_and_render_state_is_scope_local():
    save = SCOREZONE_JS.split("function _saveScopeRuntime", 1)[1].split(
        "function _restoreScopeRuntime", 1
    )[0]
    restore = SCOREZONE_JS.split("function _restoreScopeRuntime", 1)[1].split(
        "function _hydrateFeed", 1
    )[0]
    assert "feed: _chronoSort(_feed)" in save
    assert "_feed = _chronoSort(saved.feed)" in restore
    assert "_shownFeedIdsByScope[scope]" in save
    assert "_shownFeedIds = new Set(_feed.map(_eid))" in restore


def test_failed_refresh_recovers_without_rebuilding_mounted_controls():
    refresh = SCOREZONE_JS.split("async function _refresh(opts) {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    # Every non-applying exit (bad HTTP/parse, 200 error payload, missing
    # identity, network/deadline catch) recovers last-good and repaints controls
    # in place -- and only the current request owner may do so.
    assert refresh.count("_recoverScopeLoad(mySeq, myGen, myScope);") == 4
    assert refresh.count("if (_ownsScreen(mySeq, myGen, myScope) && !_loadingScope) _partialUpdate();") == 4

@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_field_position_handles_sides_aliases_boundaries_and_hidden_states():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    helpers = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("window._rzRenderGameBoard")]
    script = f"""
var window={{}};
{helpers}
function fp(x){{ return window._rzFieldPosition(Object.assign({{status:'live',field_position_reliable:true,away:'MIA',home:'OAK'}},x)); }}
console.log(JSON.stringify({{
 awayOwn: fp({{possession:'MIA',yard_line:'MIA 31'}}),
 homeOwn: fp({{possession:'LV',yard_line:'LV 31'}}),
 awayOpp: fp({{possession:'MIA',yard_line:'LV 31'}}),
 homeOpp: fp({{possession:'LV',yard_line:'MIA 31'}}),
 mid: fp({{possession:'MIA',yard_line:'50'}}),
 goalNear: fp({{possession:'MIA',yard_line:'LV 1'}}),
 ownGoal: fp({{possession:'MIA',yard_line:'MIA 1'}}),
 firstDownAway: fp({{possession:'MIA',yard_line:'MIA 31',down:2,distance:6}}),
 firstDownHome: fp({{possession:'LV',yard_line:'MIA 12',down:1,distance:10}}),
 goalToGoNum: fp({{possession:'MIA',yard_line:'LV 3',down:1,distance:3}}),
 goalToGoStr: fp({{possession:'MIA',yard_line:'LV 5',distance:'Goal'}}),
 ltgInside: fp({{possession:'MIA',yard_line:'MIA 48',down:1,distance:10}}),
 halftime: fp({{status:'halftime',possession:'MIA',yard_line:'MIA 20'}}),
 unreliable: fp({{field_position_reliable:false,possession:'MIA',yard_line:'MIA 20'}}),
 badTeam: fp({{possession:'XXX',yard_line:'MIA 20'}})
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    # Away goal on the left (spot 0), home goal on the right (spot 100); OAK aliases to LV.
    assert out["awayOwn"] == {"spot": 31, "side": "away", "ltg": None,
                              "label": "MIA 31", "toGoal": 69, "goalToGo": False}
    assert out["homeOwn"]["spot"] == 69 and out["homeOwn"]["side"] == "home" and out["homeOwn"]["label"] == "LV 31"
    assert out["awayOpp"]["spot"] == 69 and out["awayOpp"]["side"] == "away" and out["awayOpp"]["toGoal"] == 31
    assert out["homeOpp"]["spot"] == 31 and out["homeOpp"]["side"] == "home"
    assert out["mid"]["spot"] == 50 and out["mid"]["label"] == "50"
    assert out["goalNear"]["spot"] == 99 and out["ownGoal"]["spot"] == 1
    # First-down line-to-gain: away drives toward 100 (31+6=37), home toward 0 (12-10=2).
    assert out["firstDownAway"]["ltg"] == 37 and out["firstDownAway"]["goalToGo"] is False
    assert out["firstDownHome"]["ltg"] == 2 and out["firstDownHome"]["side"] == "home"
    assert out["ltgInside"]["ltg"] == 58 and out["ltgInside"]["goalToGo"] is False
    # Goal-to-go (numeric distance reaching the goal, or the literal "Goal") shows no line.
    assert out["goalToGoNum"]["goalToGo"] is True and out["goalToGoNum"]["ltg"] is None
    assert out["goalToGoStr"]["goalToGo"] is True and out["goalToGoStr"]["ltg"] is None
    # Hidden states never invent a value.
    assert out["halftime"] is None and out["unreliable"] is None and out["badTeam"] is None


def test_shared_board_and_modal_refresh_contract():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    # The modal board call is the ScoreZone (live) tab's header: it opts into
    # the field board and passes the game's PBP rows from the payload state.
    assert "window._rzRenderGameBoard(game, { modal:true, fieldBoard: true, pbpPlays: (state.pbp_by_game || {})[info.game_id] || [] })" in app_js
    assert "field_position_reliable" in app_js
    # Board + modal live fetches share one window-level cache keyed by
    # league path, so a response can only land in its own league's entry
    # (replaces the stub's old per-fetch generation guard).
    assert "window.__brSzLiveCache = (function" in app_js
    assert "__brSzLiveCache.get('full', STALE, refresh)" in app_js
    assert "__brSzLiveCache.get('lite', 20000, cb)" in app_js
    assert "setTimeout(function tick()" in app_js
    assert "Live box-score stats are temporarily unavailable." in app_js
    assert "Stat breakdown appears once the game is underway." not in app_js
    # Possession dot rendered by the shared renderer (main only set an unstyled .has-ball).
    assert "rz-nfl-poss" in app_js
    # Goal-to-go relabels the situation line ("1st & 3" from the 3 -> "1st & Goal").
    assert "goalToGo" in app_js
    # Field-position row presentation.
    assert ".rz-field-fill.is-away" in DASHBOARD_CSS
    assert ".rz-field-fill.is-home" in DASHBOARD_CSS
    assert ".rz-field-flag" in DASHBOARD_CSS
    assert ".rz-field-fd" in DASHBOARD_CSS
    assert ".rz-field-tick" in DASHBOARD_CSS
    assert ".rz-field-clip" in DASHBOARD_CSS
    assert ".rz-field-fill.is-dim" in DASHBOARD_CSS
    assert ".rz-nfl-poss" in DASHBOARD_CSS

@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_board_possession_dot_never_sits_in_middle_column():
    """Regression (2026-10-29 screenshot): the possession dot was the last flow
    child past the score, so at phone width it landed in the middle column and
    stacked on the live dot. Now: no dot at all when the field bar renders
    (its fill + flag already show possession); otherwise the dot overlays the
    badge corner inside .rz-nfl-logo-wrap, before any score."""
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    segment = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("// Build a player's modal log")]
    script = f"""
var window={{}};
var document={{createElement:function(){{var el={{_t:''}};Object.defineProperty(el,'textContent',{{set:function(v){{this._t=String(v);}},get:function(){{return this._t;}}}});Object.defineProperty(el,'innerHTML',{{get:function(){{return this._t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}}});return el;}}}};
{segment}
function board(x){{ return window._rzRenderGameBoard(Object.assign({{away:'CAR',home:'GB',away_pts:13,home_pts:27,status:'live',down:1,distance:10,game_quarter:'Q4',game_clock:'14:28'}},x)); }}
console.log(JSON.stringify({{
 withBar: board({{possession:'CAR',yard_line:'CAR 36',field_position_reliable:true}}),
 noBar: board({{possession:'CAR',yard_line:'CAR 36',field_position_reliable:false}}),
 homeBar: board({{possession:'GB',yard_line:'CAR 36',field_position_reliable:true}}),
 homeNoBar: board({{possession:'GB',yard_line:'CAR 36',field_position_reliable:false}})
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    # Field bar present: possession already encoded by fill/flag, dot must go.
    assert "rz-field" in out["withBar"] and "rz-nfl-poss" not in out["withBar"]
    assert "rz-field" in out["homeBar"] and "rz-nfl-poss" not in out["homeBar"]
    # No field bar: exactly one dot, inside the badge wrap, before the last score
    # (i.e. on the possessing team's badge, never past a score toward the middle).
    for html in (out["noBar"], out["homeNoBar"]):
        assert html.count("rz-nfl-poss") == 1
        assert "rz-nfl-logo-wrap" in html
        assert html.index("rz-nfl-poss") < html.rindex("rz-nfl-score")
    assert ".rz-nfl-logo-wrap" in DASHBOARD_CSS
    # Team name appears exactly once per side: badge chip carries the
    # abbreviation, so the board emits no separate abv text element.
    for html in (out["withBar"], out["noBar"], out["homeBar"], out["homeNoBar"]):
        assert "rz-nfl-abv" not in html
    # Yard line appears exactly once too: on the field flag when the bar
    # renders (centre line is down-and-distance only), otherwise in the
    # centre line, which is then the only place it can appear.
    assert '<span class="rz-nfl-sit">1st &amp; 10</span>' in out["withBar"] or \
        '<span class="rz-nfl-sit">1st & 10</span>' in out["withBar"]
    assert "1st & 10 · CAR 36" in out["noBar"]


def test_game_pill_team_row_has_no_duplicate_abv_text():
    sz = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
    row = sz[sz.index("function _gamePillTeamRow") : sz.index("// ESPN-style horizontal NFL game selector")]
    assert "rz-gp-abv" not in row
    board = sz[sz.index("function _renderNflBoard(") : sz.index("function _gamePillStatus")]
    assert "rz-nfl-abv" not in board


def test_game_pills_use_versus_line_with_day_status_for_pregame():
    """V2 versus line (Kaedon's 2026-10-30 pick): one line per pill --
    away chip, away score, @, home score, home chip -- with the status as
    a small line underneath. Pregame status gains the kickoff day from the
    same epoch ("SUN · 1:00 PM") while aria labels keep the time-only
    wording they have always had."""
    sz = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
    row = sz[sz.index("function _gamePillTeamRow") : sz.index("// ESPN-style horizontal NFL game selector")]
    # Home score precedes the home chip; away score follows the away chip.
    assert "side === 'home' ? score + logo : logo + score" in row
    assert "rz-gp-abv" not in row
    status = sz[sz.index("function _gamePillStatus") : sz.index("function _gamePillLogo")]
    assert "function _gamePillStatus(g, norm, includeDay)" in status
    assert "day + ' · ' + time" in status
    for segment in (
        sz[sz.index("function _renderGameStrip()") : sz.index("function _renderFilterChips")],
        sz[sz.index("function _pregameScheduleHtml()") : sz.index("function _syncFeed")],
    ):
        assert "rz-gp-versus" in segment
        assert "_gamePillTeamRow" in segment
        assert '<span class="rz-gp-at" aria-hidden="true">@</span>' in segment
        # Versus order inside the line, status line after it.
        assert segment.index("_gamePillTeamRow") < segment.index("rz-gp-at") < segment.rindex("_gamePillTeamRow")
        assert segment.index("rz-gp-versus") < segment.index("rz-gp-status")
        assert "_gamePillStatus" in segment and ", true)" in segment
    strip = sz[sz.index("function _renderGameStrip()") : sz.index("function _renderFilterChips")]
    # Aria keeps the original time-only status; the visible line adds the day.
    assert "var ariaStatus = _gamePillStatus(g, norm);" in strip
    assert "var status = _gamePillStatus(g, norm, true);" in strip
    assert "', ' + ariaStatus" in strip
    # Scores still obey the existing showScore rule (absent pregame).
    assert "var showScore = norm !== 'pregame' && norm !== 'unknown';" in strip
    # CSS: versus line exists in both .rz-game-pill definitions (ScoreZone
    # page + homepage preview), status is left-aligned, pills size to content.
    assert ".rz-gp-versus {" in DASHBOARD_CSS
    assert ".hp-rz .rz-gp-versus {" in DASHBOARD_CSS
    assert ".rz-gp-at {" in DASHBOARD_CSS and ".hp-rz .rz-gp-at {" in DASHBOARD_CSS
    main_status = DASHBOARD_CSS.split(".rz-gp-status {", 1)[1].split("}", 1)[0]
    hp_status = DASHBOARD_CSS.split(".hp-rz .rz-gp-status {", 1)[1].split("}", 1)[0]
    assert "text-align: left" in main_status and "text-align: left" in hp_status
    assert ".rz-gp-versus .rz-gp-score { margin-left: 0; }" in DASHBOARD_CSS
    assert ".hp-rz .rz-gp-versus .rz-gp-score { margin-left: 0; }" in DASHBOARD_CSS


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_pregame_board_header_is_centered_b1_and_live_final_keep_score_row():
    """B1 pregame header (Kaedon's 2026-10-30 pick): a centered matchup
    unit (badge, code, AT, code, badge) with the game's own status string
    centered underneath, replacing the edge-stranded score row. Live and
    final boards keep the score row. The Box score control compacts only
    next to / inside a pregame board (CSS state selectors)."""
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    seg = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("// Build a player's modal log")]
    script = f"""
var window = {{}};
{_NODE_DOM_STUB}
window.brTeamBadge = function(t) {{ return '<span class="rz-nfl-badge">' + t + '</span>'; }};
{seg}
function board(x){{ return window._rzRenderGameBoard(Object.assign({{away:'ARI',home:'DAL',away_pts:0,home_pts:0,game_status:'11/1 · 1:00 PM EST'}},x)); }}
console.log(JSON.stringify({{
  pre: board({{status:'pregame'}}),
  live: board({{status:'live',away_pts:13,home_pts:27,game_quarter:'Q4',game_clock:'2:41'}}),
  fin: board({{status:'final',away_pts:13,home_pts:27}})
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    pre = out["pre"]
    assert "rz-nfl-board--pregame" in pre
    assert "rz-nfl-pregame-head" in pre and "rz-nfl-pregame-matchup" in pre
    assert "rz-nfl-score-row" not in pre and "rz-nfl-score" not in pre
    # Centered unit order: away badge, away code, AT, home code, home badge,
    # then the game's own status string underneath.
    assert pre.index("rz-nfl-badge") < pre.index("rz-nfl-pregame-code")
    assert pre.index(">ARI<") < pre.index("rz-nfl-pregame-at") < pre.index(">DAL<")
    assert "11/1 · 1:00 PM EST" in pre
    assert pre.index("rz-nfl-pregame-at") < pre.index("rz-nfl-pregame-status")
    for html in (out["live"], out["fin"]):
        assert "rz-nfl-score-row" in html and "rz-nfl-pregame-head" not in html
    # Box score: compact pill only in the pregame state.
    assert ".rz-nfl-board--pregame + .rz-boxscore-btn" in DASHBOARD_CSS
    assert ".rz-nfl-board--pregame .rz-boxscore-btn" in DASHBOARD_CSS
    box = DASHBOARD_CSS.split(".rz-boxscore-btn {", 1)[1].split("}", 1)[0]
    assert "width: 100%" in box  # live/final keep the full-width control
    compact = DASHBOARD_CSS.split(".rz-nfl-board--pregame + .rz-boxscore-btn,", 1)[1].split("}", 1)[0]
    assert "width: max-content" in compact and "padding: 7px 14px" in compact


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


def test_field_board_is_scorezone_only_and_falls_back_to_the_bar():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    # The shared renderer grows an opt-in field board. Only ScoreZone
    # surfaces pass it (the ScoreZone page board and the modal's ScoreZone
    # tab); a plain { modal:true } render keeps the current drive bar.
    assert "window._rzDeriveDrive = function(game, fp, plays)" in app_js
    assert "window._rzFieldBoardHtml = function(game, fp, drive, colors, fbAnim)" in app_js
    assert "if(fp&&options.fieldBoard){" in app_js
    assert "window._rzRenderGameBoard(game, { modal:true, fieldBoard: true, pbpPlays: (state.pbp_by_game || {})[info.game_id] || [] })" in app_js
    # Pass/run cue: pass plays render a hollow dot, in both field-board
    # renderers; the cue class exists only in the field-board CSS block.
    assert "drive.kinds && drive.kinds[i] === 'pass'" in app_js
    assert "drive.kinds && drive.kinds[i] === 'pass'" in SCOREZONE_JS
    assert ".rz-fb-dot.is-pass {" in DASHBOARD_CSS
    assert "border: 2px solid var(--rz-fb-color)" in DASHBOARD_CSS
    assert "box-sizing: border-box" in DASHBOARD_CSS
    # Live-arrival animation gating: the ScoreZone page passes its live
    # single-play signal into the renderer, which keeps a drive-signature
    # registry; the modal never passes it and stays static.
    assert "fbLive: _animationMode === 'live'" in SCOREZONE_JS
    assert "window._rzFbSig" in app_js
    assert "_fbPrevSig" in SCOREZONE_JS
    assert ".rz-fb-td {" in DASHBOARD_CSS and ".rz-fb-tdflash {" in DASHBOARD_CSS
    assert "@keyframes rz-fb-ball-slide" in DASHBOARD_CSS
    assert "@keyframes rz-fb-tdflash" in DASHBOARD_CSS
    # ScoreZone passes the option plus the game's raw PBP rows, on both its
    # shared-renderer call and its app.js-less fallback renderer.
    assert "fieldBoard: true, fbLive: _animationMode === 'live', pbpPlays: (_state.pbp_by_game || {})[_filters.nfl] || []" in SCOREZONE_JS
    assert "function _nflFieldBoardHtml(g, plays)" in SCOREZONE_JS
    assert "fbHtml" in SCOREZONE_JS
    # Presentation contract (approved mock): perspective field, uniform dots,
    # the first-down line is the only yellow.
    assert ".rz-fb-field {" in DASHBOARD_CSS
    assert "rotateX(22deg)" in DASHBOARD_CSS
    assert ".rz-fb-dot {" in DASHBOARD_CSS
    assert "width: 7px; height: 7px" in DASHBOARD_CSS
    assert ".rz-fb-ball {" in DASHBOARD_CSS
    assert "width: 11px; height: 11px" in DASHBOARD_CSS
    assert ".rz-fb-fd {" in DASHBOARD_CSS
    assert ".rz-fb-drive {" in DASHBOARD_CSS
    fb_css = DASHBOARD_CSS.split(".rz-fb {", 1)[1].split(".rz-nfl-board--modal", 1)[0]
    assert fb_css.count("#eab308") == 1  # first-down line only, no other yellow
    # The reduced-motion kill switch lives inside the field-board block.
    assert "@media (prefers-reduced-motion: reduce)" in fb_css


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_field_board_derives_current_drive_and_renders_it():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    helpers = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("// Build a player's modal log")]
    script = f"""
var window={{}};
var document={{createElement:function(){{var o={{_t:''}};
  Object.defineProperty(o,'textContent',{{set:function(v){{this._t=String(v);}}}});
  Object.defineProperty(o,'innerHTML',{{get:function(){{return this._t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}}});
  return o;}}}};
{helpers}
function row(playId, seq, team, yl, sl, extra) {{
  return Object.assign({{play_id: playId, seq: seq, team: team, pid: 'x', name: 'P',
    yard_line: yl, end_yard_line: '', play_text: '', play_state: 'VALID', stat_line: sl}}, extra || {{}});
}}
var game = {{away: 'CAR', home: 'GB', away_pts: 13, home_pts: 27, status: 'live',
  possession: 'CAR', yard_line: 'CAR 36', field_position_reliable: true,
  down: 1, distance: 10, game_quarter: 'Q4', game_clock: '14:28'}};
var fp = window._rzFieldPosition(game);
var plays = [
  row('g1', 1, 'GB', 'GB 25', {{carries: 1, rush_yds: 3}}),
  row('c1', 3, 'CAR', 'CAR 12', {{carries: 1, rush_yds: 8}}),
  row('c2', 4, 'CAR', 'CAR 20', {{pass_att: 1, pass_yds: 12}}),
  row('c2', 4, 'CAR', 'CAR 20', {{targets: 1, rec: 1, rec_yds: 12}}),
  row('c3', 5, 'CAR', 'CAR 32', {{carries: 1, rush_yds: 4}})
];
var drive = window._rzDeriveDrive(game, fp, plays);
var neg = window._rzDeriveDrive(game, fp, [
  row('c1', 3, 'CAR', 'CAR 12', {{carries: 1, rush_yds: 8}}),
  row('c2', 4, 'CAR', 'CAR 20', {{carries: 1, rush_yds: 12}}),
  row('c3', 5, 'CAR', 'CAR 32', {{carries: 1, rush_yds: -3}}),
  row('c4', 6, 'CAR', 'CAR 29', {{carries: 1, rush_yds: 7}})
]);
var sackGame = Object.assign({{}}, game, {{yard_line: 'CAR 29', down: 2, distance: 17}});
var sack = window._rzDeriveDrive(sackGame, window._rzFieldPosition(sackGame), [
  row('s1', 1, 'CAR', 'CAR 25', {{carries: 1, rush_yds: 11}}),
  row('s2', 2, 'CAR', 'CAR 36', {{}}, {{play_text: 'QB sacked at CAR 29 for -7 yards'}}),
  row('s2', 2, 'GB', 'CAR 36', {{sacks: 1}})
]);
var mixed = window._rzDeriveDrive(game, fp, [
  row('m1', 1, 'CAR', 'CAR 12', {{carries: 1, rush_yds: 8}}),
  row('m2', 2, 'CAR', 'CAR 20', {{pass_att: 1, pass_yds: 12}}),
  row('m2', 2, 'CAR', 'CAR 20', {{targets: 1, rec: 1, rec_yds: 12}}),
  row('m3', 3, 'CAR', 'CAR 32', {{}}, {{play_text: 'QB sacked at CAR 29 for -3 yards'}}),
  row('m3', 3, 'GB', 'CAR 32', {{sacks: 1}}),
  row('m4', 4, 'CAR', 'CAR 29', {{targets: 1}})
]);
var none = window._rzDeriveDrive(game, fp, [row('g1', 1, 'GB', 'GB 25', {{carries: 1, rush_yds: 3}})]);
var fbHtml = window._rzRenderGameBoard(game, {{fieldBoard: true, pbpPlays: plays}});
var fbBare = window._rzRenderGameBoard(game, {{fieldBoard: true, pbpPlays: []}});
var modalHtml = window._rzRenderGameBoard(game, {{modal: true}});
var unrelHtml = window._rzRenderGameBoard(Object.assign({{}}, game, {{field_position_reliable: false}}), {{fieldBoard: true, pbpPlays: plays}});
console.log(JSON.stringify({{drive: drive, neg: neg, sack: sack, mixed: mixed, none: none,
  fbHtml: fbHtml, fbBare: fbBare, modalHtml: modalHtml, unrelHtml: unrelHtml}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    # The approved mock drive: start CAR 12, +8, +12, +4 to CAR 36.
    assert out["drive"] == {"count": 3, "points": [20, 32, 36], "kinds": ["run", "pass", "run"],
                            "firstSeq": 3, "lastSeq": 5, "td": False,
                            "startAbs": 12, "startLabel": "CAR 12", "net": 24, "chip": "+4 run"}
    # Negative plays backtrack: the -3 dot lands behind the +12 dot.
    assert out["neg"]["points"] == [20, 32, 29, 36] and out["neg"]["net"] == 24
    assert out["neg"]["kinds"] == ["run", "run", "run", "run"]
    assert out["sack"]["chip"] == "−7 sack" and out["sack"]["points"] == [36, 29]
    # A sack counts as a pass play for the cue.
    assert out["sack"]["kinds"] == ["run", "pass"]
    # Mixed drive: run solid, pass hollow, sack hollow, and a target-only
    # play has no clean run/pass read, so it stays null (solid default dot).
    assert out["mixed"]["points"] == [20, 32, 29, 36]
    assert out["mixed"]["kinds"] == ["run", "pass", "pass", None]
    # No possessing-team scrimmage plays -> no drive (ball + flag only).
    assert out["none"] is None
    fb = out["fbHtml"]
    assert "rz-fb-field" in fb and fb.count("rz-fb-dot") == 3
    # The mock drive's one pass play renders the hollow cue dot.
    assert fb.count("rz-fb-dot is-pass") == 1
    assert "LIVE DRIVE" in fb and "3 plays" in fb and "24 yds" in fb
    assert "started CAR 12" in fb and "+4 run" in fb and ">CAR 36<" in fb
    # The field-board centre line is down & distance only; the flag owns the spot.
    assert ">1st & 10<" in fb and " · CAR 36" not in fb
    bare = out["fbBare"]
    assert "rz-fb-field" in bare and "rz-fb-dot" not in bare and "LIVE DRIVE" not in bare
    # A plain modal render (no fieldBoard opt-in) keeps the current drive
    # bar byte-for-byte in behaviour, cue included: no field board at all.
    modal = out["modalHtml"]
    assert "rz-fb" not in modal and "is-pass" not in modal
    # The bar renders, so PR #2899's rule applies here too: the yard line
    # lives only on the bar's flag, never repeated in the centre line.
    assert "rz-field-fill" in modal and " · CAR 36" not in modal
    # Unreliable field position: no field of either kind, exactly as today.
    assert "rz-fb" not in out["unrelHtml"] and "rz-field" not in out["unrelHtml"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_modal_scorezone_tab_renders_the_field_board():
    """The modal's ScoreZone (live) tab is a ScoreZone surface: its board is
    the field board (cue included) fed by the payload's own PBP rows, while
    the board keeps its modal wrapper. Without PBP rows in scope it degrades
    to ball + flag, never an invented trail."""
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    helpers = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("window._rzSyncTabLive")]
    script = f"""
var window = {{ location: {{ pathname: '/sleeper/2026/123/scorezone' }} }};
var document = {{createElement:function(){{var o={{_t:''}};
  Object.defineProperty(o,'textContent',{{set:function(v){{this._t=String(v);}}}});
  Object.defineProperty(o,'innerHTML',{{get:function(){{return this._t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}}});
  return o;}}, getElementById: function() {{ return {{}}; }} }};
var _rzFgPts = function() {{ return 0; }};
{helpers}
function row(playId, seq, team, yl, sl, extra) {{
  return Object.assign({{play_id: playId, seq: seq, team: team, pid: 'x', name: 'P',
    yard_line: yl, end_yard_line: '', play_text: '', play_state: 'VALID', stat_line: sl}}, extra || {{}});
}}
var plays = [
  row('g1', 1, 'GB', 'GB 25', {{carries: 1, rush_yds: 3}}),
  row('c1', 3, 'CAR', 'CAR 12', {{carries: 1, rush_yds: 8}}),
  row('c2', 4, 'CAR', 'CAR 20', {{pass_att: 1, pass_yds: 12}}),
  row('c2', 4, 'CAR', 'CAR 20', {{targets: 1, rec: 1, rec_yds: 12}}),
  row('c3', 5, 'CAR', 'CAR 32', {{carries: 1, rush_yds: 4}})
];
var game = {{away: 'CAR', home: 'GB', away_pts: 13, home_pts: 27, status: 'live',
  possession: 'CAR', yard_line: 'CAR 36', field_position_reliable: true,
  down: 1, distance: 10, game_quarter: 'Q4', game_clock: '14:28'}};
var state = {{
  player_info: {{ '9': {{ name: 'Wide Receiver', pos: 'WR', team: 'CAR',
    home: 'GB', away: 'CAR', away_pts: '13', home_pts: '27', game_id: 'g1',
    game_code: '1', game_status: 'In Progress',
    stat_line: {{ carries: 0, rush_yds: 0, rush_td: 0, targets: 5, rec: 4, rec_yds: 40, rec_td: 0 }} }} }},
  scoring: {{ rec: 1, rec_yd: 0.1, rec_td: 6, rush_yd: 0.1, rush_td: 6 }},
  matchups: [], games: {{ g1: game }}, pbp_by_game: {{ g1: plays }}
}};
var html = window._rzBuildLiveHtml('9', state, []);
var degraded = window._rzBuildLiveHtml('9', Object.assign({{}}, state, {{pbp_by_game: {{}}}}), []);
console.log(JSON.stringify({{html: html, degraded: degraded}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    html = out["html"]
    # Field board inside the modal wrapper, trail + cue from the real rows.
    assert "rz-nfl-board--modal" in html
    assert "rz-fb-field" in html and "LIVE DRIVE" in html and "started CAR 12" in html
    assert html.count("rz-fb-dot") == 3 and html.count("rz-fb-dot is-pass") == 1
    # No PBP rows in scope: field board degrades to ball + flag, no trail.
    degraded = out["degraded"]
    assert "rz-fb-field" in degraded and ">CAR 36<" in degraded
    assert "rz-fb-dot" not in degraded and "LIVE DRIVE" not in degraded


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_field_board_animation_only_for_single_live_arrivals():
    """Animation gating: the renderer keeps the last drive signature per
    game and emits animation classes only when a live-gated render grows
    the same drive by exactly one play. Bulk jumps, repeat renders and
    non-gated (modal) renders are static; a TD arrival celebrates."""
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    helpers = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("// Build a player's modal log")]
    script = f"""
var window={{}};
var document={{createElement:function(){{var o={{_t:''}};
  Object.defineProperty(o,'textContent',{{set:function(v){{this._t=String(v);}}}});
  Object.defineProperty(o,'innerHTML',{{get:function(){{return this._t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}}});
  return o;}}}};
{helpers}
function row(playId, seq, team, yl, sl, extra) {{
  return Object.assign({{play_id: playId, seq: seq, team: team, pid: 'x', name: 'P',
    yard_line: yl, end_yard_line: '', play_text: '', play_state: 'VALID', stat_line: sl}}, extra || {{}});
}}
function gameAt(gid, yl, down, dist) {{
  return {{game_id: gid, away: 'CAR', home: 'GB', away_pts: 13, home_pts: 27, status: 'live',
    possession: 'CAR', yard_line: yl, field_position_reliable: true,
    down: down, distance: dist, game_quarter: 'Q4', game_clock: '14:28'}};
}}
var a1 = row('a1', 1, 'CAR', 'CAR 12', {{carries: 1, rush_yds: 8}});
var a2q = row('a2', 2, 'CAR', 'CAR 20', {{pass_att: 1, pass_yds: 12}});
var a2w = row('a2', 2, 'CAR', 'CAR 20', {{targets: 1, rec: 1, rec_yds: 12}});
var a3 = row('a3', 3, 'CAR', 'CAR 32', {{carries: 1, rush_yds: 4}});
var a4q = row('a4', 4, 'CAR', 'CAR 36', {{pass_att: 1, pass_yds: 9}});
var a4w = row('a4', 4, 'CAR', 'CAR 36', {{targets: 1, rec: 1, rec_yds: 9}});
var a5 = row('a5', 5, 'CAR', 'CAR 45', {{carries: 1, rush_yds: 5}});
var v1 = [a1, a2q, a2w], v2 = [a1, a2q, a2w, a3], vBulk = [a1, a2q, a2w, a3, a4q, a4w];
function board(gid, yl, plays, opts) {{
  return window._rzRenderGameBoard(gameAt(gid, yl, 1, 10),
    Object.assign({{fieldBoard: true, pbpPlays: plays}}, opts || {{}}));
}}
var out = {{}};
out.first = board('gA', 'CAR 32', v1, {{fbLive: false}});
out.arrival = board('gA', 'CAR 36', v2, {{fbLive: true}});
out.repeat = board('gA', 'CAR 36', v2, {{fbLive: true}});
out.bulkBase = board('gB', 'CAR 32', v1, {{fbLive: false}});
out.bulkJump = board('gB', 'CAR 45', vBulk, {{fbLive: true}});
out.afterBulk = board('gB', 'CAR 50', vBulk.concat([a5]), {{fbLive: true}});
var t1q = row('t1', 1, 'CAR', 'GB 20', {{pass_att: 1, pass_yds: 14}});
var t1w = row('t1', 1, 'CAR', 'GB 20', {{targets: 1, rec: 1, rec_yds: 14}});
var t2 = row('t2', 2, 'CAR', 'GB 6', {{carries: 1, rush_yds: 6, rush_td: 1}});
function tdBoard(plays, opts) {{
  return window._rzRenderGameBoard(gameAt('gC', 'GB 6', 2, 6),
    Object.assign({{fieldBoard: true, pbpPlays: plays}}, opts || {{}}));
}}
out.tdBase = tdBoard([t1q, t1w], {{fbLive: false}});
out.tdArrival = tdBoard([t1q, t1w, t2], {{fbLive: true}});
out.tdRepeat = tdBoard([t1q, t1w, t2], {{fbLive: true}});
console.log(JSON.stringify(out));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    # First render: baseline only, never animated.
    assert "rz-fb-anim" not in out["first"]
    # Exactly one new play on a live render: the arrival animates in.
    arr = out["arrival"]
    assert "rz-fb-anim-play" in arr and "rz-fb-seg-new" in arr
    assert "rz-fb-dot is-new" in arr and "--rz-fb-ball-from:35.6%" in arr
    assert "TOUCHDOWN" not in arr and "rz-fb-score-pop" not in arr
    # The same state re-rendered does not replay.
    assert "rz-fb-anim" not in out["repeat"]
    # A two-play jump never animates, even on a live-gated render...
    assert "rz-fb-anim" not in out["bulkJump"]
    # ...but it moves the baseline, so the next single arrival animates.
    assert "rz-fb-anim-play" in out["afterBulk"]
    # TD arrival: celebration markup, flash + pill + score pop, once.
    td = out["tdArrival"]
    assert "rz-fb-anim-td" in td and "rz-fb-tdflash" in td
    assert ">TOUCHDOWN<" in td and "rz-fb-score-pop" in td
    assert "rz-fb-anim" not in out["tdRepeat"] and "TOUCHDOWN" not in out["tdRepeat"]


def test_modal_history_comes_from_uncapped_latest_play_groups():
    scorezone = SCOREZONE_JS.split("function _modalPlayHistory()", 1)[1].split(
        "window.__rzGetPlayerLive", 1
    )[0]
    assert "Object.keys(_playGroupsByKey)" in scorezone
    assert "primaryEvent" in scorezone
    assert "_PAGE_SIZE" not in scorezone
    assert "_feed.filter(function(ev) { return !ev.fromPbp; })" in scorezone
    live = SCOREZONE_JS.split("window.__rzGetPlayerLive = function(pid)", 1)[1][:180]
    assert "_modalPlayHistory()" in live


def _extract_js_function(source: str, start_marker: str) -> str:
    """Slice one top-level function out of a JS file by brace counting."""
    i = source.index(start_marker)
    j = source.index("{", i)
    depth = 0
    for k in range(j, len(source)):
        if source[k] == "{":
            depth += 1
        elif source[k] == "}":
            depth -= 1
            if depth == 0:
                return source[i : k + 1]
    raise AssertionError("unbalanced braces extracting " + start_marker)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_feed_desc_calls_an_intercepted_target_a_pick_not_an_incompletion():
    fn = _extract_js_function(SCOREZONE_JS, "function _improvePlayDesc(")
    script = f"""
var _improvePlayDesc = {fn};
var wr = {{pid:'9', name:'Jaxon Smith-Njigba', pos:'WR', line:{{targets:1, rec:0, rec_yds:0, rush_yds:0, pass_yds:0}}}};
var qb = {{pid:'1', name:'Sam Darnold', pos:'QB', line:{{int:1, pass_yds:0}}}};
var def = {{pid:'d', name:'KC DEF', pos:'DEF', line:{{def_int:1}}}};
var qbClean = {{pid:'1', name:'Sam Darnold', pos:'QB', line:{{pass_yds:0}}}};
console.log(JSON.stringify({{
  picked: _improvePlayDesc(wr, [wr, qb, def], ''),
  plain: _improvePlayDesc(wr, [wr, qbClean], '')
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert out["picked"] == "Target from Sam Darnold · intercepted"
    assert out["plain"] == "Target from Sam Darnold · incomplete"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_player_modal_timeline_labels_pick_and_reconciles_targets_at_final():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    helpers = app_js[
        app_js.index("window._rzPlayerLogEvents") : app_js.index(
            "window._rzSyncTabLive"
        )
    ]
    script = f"""
var window = {{ location: {{ pathname: '/sleeper/2026/123/scorezone' }} }};
var document = {{ getElementById: function() {{ return {{}}; }} }};
{helpers}
window._rzRenderGameBoard = function() {{ return ''; }};
var QB = '1', WR = '9';
function ev(playId, seq, ts, q, clock, desc, contribs) {{
  return {{ playId: playId, gameId: 'g1', seq: seq, playSortTs: ts,
    gameQuarter: q, gameClock: clock, desc: desc, playState: 'VALID',
    contributions: contribs }};
}}
var events = [
  ev('p1', 1, 100, 'Q1', '10:00', '10-yard reception from Sam Darnold', [
    {{pid: QB, name: 'Sam Darnold', pos: 'QB', line: {{pass_yds: 10}}, pts: 0.4, kind: 'gain'}},
    {{pid: WR, name: 'Jaxon Smith-Njigba', pos: 'WR', line: {{targets: 1, rec: 1, rec_yds: 10}}, pts: 2, kind: 'gain'}}
  ]),
  ev('p2', 2, 200, 'Q1', '9:00', 'Target from Sam Darnold · incomplete', [
    {{pid: QB, name: 'Sam Darnold', pos: 'QB', line: {{pass_yds: 0}}, pts: 0, kind: 'gain'}},
    {{pid: WR, name: 'Jaxon Smith-Njigba', pos: 'WR', line: {{targets: 1}}, pts: 0, kind: 'target'}}
  ]),
  // The bug: PBP names JSN on Darnold's end-zone pick, so the group desc
  // (built from the primary actor) calls it an incomplete target, and the
  // log sums 3 targets against the box score's 2.
  ev('p3', 3, 300, 'Q2', '0:23', 'Target from Sam Darnold · incomplete', [
    {{pid: QB, name: 'Sam Darnold', pos: 'QB', line: {{int: 1}}, pts: -2, kind: 'neg'}},
    {{pid: WR, name: 'Jaxon Smith-Njigba', pos: 'WR', line: {{targets: 1}}, pts: 0, kind: 'target'}},
    {{pid: 'dkc', name: 'KC DEF', pos: 'DEF', line: {{def_int: 1}}, pts: 2, kind: 'gain'}}
  ])
];
var wrRows = window._rzPlayerLogEvents(WR, events);
var qbRows = window._rzPlayerLogEvents(QB, events);
var state = {{
  player_info: {{ '9': {{ name: 'Jaxon Smith-Njigba', pos: 'WR', team: 'SEA',
    home: 'SEA', away: 'KC', away_pts: '3', home_pts: '20', game_id: 'g1',
    game_code: '2', game_status: 'Final',
    stat_line: {{ carries: 0, rush_yds: 0, rush_td: 0, targets: 2, rec: 1, rec_yds: 10, rec_td: 0 }} }} }},
  scoring: {{ rec: 1, rec_yd: 0.1, rec_td: 6, rush_yd: 0.1, rush_td: 6 }},
  matchups: [], games: {{}}
}};
var html = window._rzBuildLiveHtml(WR, state, events);
console.log(JSON.stringify({{ wr: wrRows, qb: qbRows, html: html }}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    wr = {row["playId"]: row for row in out["wr"]}
    qb = {row["playId"]: row for row in out["qb"]}
    # The pick is labeled as a pick in both players' logs, never an incompletion.
    assert wr["p3"]["kind"] == "int" and wr["p3"]["isIntPlay"] is True
    assert "intercepted" in wr["p3"]["desc"] and "incomplete" not in wr["p3"]["desc"]
    assert "intercepted" in qb["p3"]["desc"]
    assert "intended for Jaxon Smith-Njigba" in qb["p3"]["desc"]
    # Ordinary incompletions stay ordinary.
    assert wr["p2"]["kind"] == "target" and wr["p2"]["isIntPlay"] is False
    html = out["html"]
    # Timeline design: stat tiles, build-up chart, chronological timeline.
    assert "rz-pm-tiles" in html and "Fantasy Pts" in html
    # Hero tiles break out receiving and rushing yards separately.
    assert "Receiving Yards" in html and "Rushing Yards" in html
    assert "Points build-up" in html and "<svg" in html
    assert "rz-pm-tl" in html
    # At final the phantom target is demoted so the log matches the box (2).
    assert "No target credited" in html and "not counted" in html
    # Chronological order: Q1 reception renders before the Q2 pick.
    assert html.index("10-yard reception") < html.index("intercepted")


# ── Drive-bar redesign (2026-10-30) ─────────────────────────────────────────
# The classic bar (shared board path via _rzRenderGameBoard, Matchups page
# via driveBarHtml) was restyled in the ScoreZone field board's language.
# Kaedon then found the first redesign too bulky in line and picked the S1
# ultra-slim inline bar: ONE line -- micro end-zone labels outside a 10px
# striped pill track, gradient fill from the goal line to a small solid
# ball, yellow first-down tick, and the spot as plain team-colour text at
# the end of the line. No in-track caps, no floating flag, no arrowhead.
# Geometry includes both end zones (x = 10 + spot * 0.8), as the field
# board. (The player modal's ScoreZone tab renders the field board --
# it passes fieldBoard:true -- so this bar has no modal variant; pinned
# separately below.)

def test_drive_bar_s1_slim_markup_present_in_both_renderers():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    # Board bar (ScoreZone classic path).
    assert "rz-field-line" in app_js
    assert "rz-field-ezl" in app_js
    assert "rz-field-spot" in app_js
    assert "rz-field-ball" in app_js
    assert "rz-field-top" in app_js
    assert "rz-field-live" in app_js
    # The drive summary is caller-supplied only -- never derived or invented
    # inside the renderer.
    assert "options.driveSummary" in app_js
    # Matchups bar.
    assert "mb-fld-bar" in app_js
    assert "mb-fld-ezl" in app_js
    assert "mb-fld-spot" in app_js
    assert "mb-fld-ball" in app_js
    # The bulky first-redesign pieces are gone from both renderers, and the
    # pre-redesign line marker stays gone.
    for gone in ("mb-fld-flag", "mb-fld-twrap", "mb-fld-ez mb-fld-ez-a",
                 "mb-fld-ez-h", "rz-field-ez is-away", "mb-fld-mark"):
        assert gone not in app_js, gone
    # CSS for both bars.
    for cls in (".rz-field-line", ".rz-field-ezl", ".rz-field-spot",
                ".rz-field-ball", ".rz-field-top", ".rz-field-live",
                ".mb-fld-bar", ".mb-fld-ezl", ".mb-fld-spot", ".mb-fld-ball"):
        assert cls in DASHBOARD_CSS, cls
    # ... and the bulky CSS is gone too.
    for gone in (".mb-fld-flag", ".mb-fld-twrap", ".mb-fld-ez {", ".rz-field-ez {"):
        assert gone not in DASHBOARD_CSS, gone


_NODE_DOM_STUB = """
var document = {
  createElement: function() { var v = ''; return {
    set textContent(x) { v = String(x); },
    get innerHTML() { return v.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  }; },
  documentElement: { getAttribute: function() { return null; } }
};
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_board_bar_geometry_and_context_row():
    """The S1 slim bar renders on the shared ScoreZone board path
    (scorezone.js's {id:'rz-nfl-board'} call). Away spot 36 -> ball x 38.8,
    fill exactly 28.8% (goal line to ball, no arrowhead overshoot), FD line
    at the 46 (46.8); home spot (GB 36) mirrors: ball 61.2, fill from the
    right. End labels sit outside the track, the spot is plain text at the
    end of the line (no caps, no flag, no red-zone tint), and the context
    row carries DRIVE + down & distance, plus a summary only when a caller
    passes a real one."""
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    seg = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("// Build a player's modal log")]
    script = f"""
var window = {{}};
{_NODE_DOM_STUB}
window.brTeamBadge = function(t) {{ return '<span class="rz-nfl-badge">' + t + '</span>'; }};
{seg}
var CAR = {{away:'CAR', home:'GB', away_pts:13, home_pts:27, status:'live', possession:'CAR',
  yard_line:'CAR 36', field_position_reliable:true, down:1, distance:10, game_quarter:'Q4', game_clock:'14:28'}};
var GB = {{away:'CAR', home:'GB', away_pts:13, home_pts:27, status:'live', possession:'GB',
  yard_line:'GB 36', field_position_reliable:true, down:2, distance:3, game_quarter:'Q4', game_clock:'9:41'}};
console.log(JSON.stringify({{
  car: window._rzRenderGameBoard(CAR, {{id:'rz-nfl-board'}}),
  carSum: window._rzRenderGameBoard(CAR, {{id:'rz-nfl-board', driveSummary:'3 plays · 24 yds'}}),
  gb: window._rzRenderGameBoard(GB, {{id:'rz-nfl-board'}})
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    car, gb = out["car"], out["gb"]
    # Slim line: micro end labels outside the track (away left, home right),
    # spot as plain text at the end of the line.
    assert 'rz-field-line' in car
    assert 'rz-field-ezl">CAR<' in car and 'rz-field-ezl">GB<' in car
    assert 'rz-field-spot">CAR 36<' in car
    # CAR (away) geometry: ball 38.8, fill from the left goal line ending
    # exactly on the ball, FD at 46.8.
    assert 'rz-field-ball" style="left:38.8%"' in car
    assert 'rz-field-fill is-away" style="width:28.8%"' in car
    assert 'rz-field-fd" style="left:46.8%"' in car
    # None of the bulky pieces: no caps, no flag, no red-zone tint.
    for absent in ("rz-field-ez ", "rz-field-flag", "rz-field-rz"):
        assert absent not in car, absent
    # Context row: DRIVE + situation, no summary unless a caller passes one.
    assert 'rz-field-live"><i></i>DRIVE' in car
    assert 'rz-field-sit">1st &amp; 10<' in car
    assert "rz-field-sum" not in car
    assert 'rz-field-sum">3 plays · 24 yds<' in out["carSum"]
    # GB (home) mirrors: ball 61.2, fill from the right, FD behind the ball at
    # 58.8 (line to gain toward the away goal), spot text follows the ball.
    assert 'rz-field-ball" style="left:61.2%"' in gb
    assert 'rz-field-fill is-home' in gb and 'style="width:28.8%"' in gb
    assert 'rz-field-fd" style="left:58.8%"' in gb
    assert 'rz-field-spot">GB 36<' in gb
    assert 'rz-field-sit">2nd &amp; 3<' in gb


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_matchup_bar_geometry_flag_and_defense_state():
    """driveBarHtml (Matchups page) renders the S1 slim bar with the same
    geometry as the board bar: end labels outside the track, the spot as
    plain text at the end of the line (no flag, no caps), and the compact
    defense label when the viewed team does not have the ball. The league
    payload in scope has no play-by-play, so no plays/yards summary appears."""
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    helpers = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("window._rzRenderGameBoard")]
    fn = _extract_js_function(app_js, "function driveBarHtml(")
    script = f"""
var window = {{}};
{_NODE_DOM_STUB}
{helpers}
var TEAM_COLORS = {{CAR:'#0085CA', GB:'#203731'}};
var norm = function(t) {{ return window._rzNormalizeTeam(t).toUpperCase(); }};
var esc = function(v) {{ var d = document.createElement('div'); d.textContent = String(v == null ? '' : v); return d.innerHTML; }};
var scorezoneGameUrl = function() {{ return ''; }};
var driveBarHtml = {fn};
var CAR = {{away:'CAR', home:'GB', status:'live', possession:'CAR', yard_line:'CAR 36',
  field_position_reliable:true, down:1, distance:10}};
var GB = {{away:'CAR', home:'GB', status:'live', possession:'GB', yard_line:'GB 36',
  field_position_reliable:true, down:2, distance:3}};
console.log(JSON.stringify({{
  car: driveBarHtml(CAR, 'CAR'),
  gb: driveBarHtml(GB, 'GB'),
  def: driveBarHtml(GB, 'CAR')
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    car, gb = out["car"], out["gb"]
    assert 'mb-fld-live"><i></i>DRIVE' in car
    assert 'mb-fld-dd">1st &amp; 10<' in car
    # Slim line: labels outside the track, spot as plain text at the end.
    assert 'mb-fld-bar' in car
    assert 'mb-fld-ezl">CAR<' in car and 'mb-fld-ezl">GB<' in car
    assert 'mb-fld-spot">CAR 36<' in car
    assert 'mb-fld-ball" style="left:38.8%"' in car
    assert 'mb-fld-fill mb-away" style="width:28.8%"' in car
    assert 'mb-fld-fd" style="left:46.8%"' in car
    # None of the bulky pieces: no flag, no wrap, no in-track caps.
    for absent in ("mb-fld-flag", "mb-fld-twrap", "mb-fld-ez "):
        assert absent not in car, absent
    # No invented drive summary on this surface.
    assert "plays" not in car and "yds" not in car
    assert 'mb-fld-ball" style="left:61.2%"' in gb
    assert 'mb-fld-fill mb-home' in gb and 'style="width:28.8%"' in gb
    assert 'mb-fld-spot">GB 36<' in gb
    assert out["def"] == '<div class="mb-fld-def">CAR on Defense</div>'


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_modal_scorezone_tab_has_no_legacy_bar_gate():
    """Reconciled with the field board (#2901): the player modal's ScoreZone
    tab is the ONLY {modal:true} caller of _rzRenderGameBoard (inside
    _rzBuildLiveHtml) and it passes fieldBoard:true, so the tab renders the
    field board. The redesign's original modal gate -- a pre-redesign bar
    kept for the modal under .rz-field--legacy -- is gone: no caller can
    reach it, and a plain {modal:true} render without the fieldBoard
    opt-in falls through to the same S1 slim bar as the shared board path."""
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    # The modal's live-tab call opts into the field board.
    assert "window._rzRenderGameBoard(game, { modal:true, fieldBoard: true, pbpPlays: (state.pbp_by_game || {})[info.game_id] || [] })" in app_js
    # No legacy gate or legacy variant survives in JS or CSS.
    assert "rz-field--legacy" not in app_js
    assert "rz-field--legacy" not in DASHBOARD_CSS
    seg = app_js[app_js.index("window._rzNormalizeTeam") : app_js.index("// Build a player's modal log")]
    assert "!options.modal" not in seg
    script = f"""
var window = {{}};
{_NODE_DOM_STUB}
window.brTeamBadge = function(t) {{ return '<span class="rz-nfl-badge">' + t + '</span>'; }};
{seg}
var CAR = {{away:'CAR', home:'GB', away_pts:13, home_pts:27, status:'live', possession:'CAR',
  yard_line:'CAR 36', field_position_reliable:true, down:1, distance:10, game_quarter:'Q4', game_clock:'14:28'}};
console.log(JSON.stringify({{ modal: window._rzRenderGameBoard(CAR, {{modal:true}}) }}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    m = out["modal"]
    # Plain modal render = the shared path's S1 slim bar (same geometry as
    # the board test above), inside the modal board wrapper -- no field
    # board (no opt-in) and no legacy variant.
    assert "rz-nfl-board--modal" in m
    assert "rz-field-line" in m
    assert 'rz-field-fill is-away" style="width:28.8%"' in m
    assert 'rz-field-ball" style="left:38.8%"' in m
    assert 'rz-field-fd" style="left:46.8%"' in m
    assert 'rz-field-spot">CAR 36<' in m
    for absent in ("rz-field--legacy", "rz-fb", "rz-field-flag"):
        assert absent not in m, absent