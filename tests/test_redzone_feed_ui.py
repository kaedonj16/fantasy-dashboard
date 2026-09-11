"""Source contracts for the Redzone play feed controls and score summary."""

from pathlib import Path


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


def test_page_zero_dom_is_reconciled_without_new_ids_and_after_stagger():
    sync = REDZONE_JS.split("function _syncFeed() {", 1)[1].split(
        "function _renderPagination", 1
    )[0]
    assert "function _orderFeedDom(target, orderedItems)" in sync
    assert sync.count("_orderFeedDom(container, page0Items);") >= 3
    assert sync.index("_orderFeedDom(container, page0Items);\n\n    // Prune") > sync.index(
        "if (toAdd.length)"
    )


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

    assert chrono_key.index("var q = parseInt(ev.gameQuarter, 10);") < chrono_key.index(
        "if (ev.gameId && ev.seq != null)"
    )
    assert "return list.slice().sort" in REDZONE_JS


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


def test_refresh_uses_fresh_canonical_render_and_no_browser_cache():
    refresh = REDZONE_JS.split("async function _refresh() {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    assert "fetch(url, { cache: 'no-store' })" in refresh
    detect_at = refresh.index("_detectChanges(newData);")
    assert detect_at < refresh.index("_render();", detect_at)
    assert "savedFeedHtml" not in refresh
    assert "newFeedEl.innerHTML" not in refresh


def test_failed_refresh_renders_to_clear_manual_spinner():
    refresh = REDZONE_JS.split("async function _refresh() {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    assert refresh.count("_recoverScopeLoad(myGen, myScope);") == 2
    assert refresh.count("if (myGen === _streamGen && myScope === _scope) _render();") == 2
