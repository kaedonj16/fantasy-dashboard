"""Focused rendering guards for the standalone draft cheat sheet."""

import json
import re
from pathlib import Path

from dashboard_services.pages.cheat_sheet_page import build_cheat_sheet_body


def _embedded_config(body: str) -> dict:
    match = re.search(r"window\.__cheatCfg = (.*?);</script>", body)
    assert match, "cheat-sheet config script was not rendered"
    return json.loads(match.group(1))


def test_cheat_sheet_config_round_trips_league_context():
    body = build_cheat_sheet_body(
        "league-123", 2026, "sleeper", num_teams=10,
        is_superflex=True, roster_positions=["QB", "SUPER_FLEX", "RB"],
        mode="dynasty", viewer_user_id="viewer-7", has_premium=True,
    )

    assert _embedded_config(body) == {
        "leagueId": "league-123",
        "season": 2026,
        "platform": "sleeper",
        "numTeams": 10,
        "isSuperflex": True,
        "rosterPositions": ["QB", "SUPER_FLEX", "RB"],
        "mode": "dynasty",
        "viewerUserId": "viewer-7",
        "hasPremium": True,
        "draftUrl": "/sleeper/2026/league-123/draft",
    }


def test_cheat_sheet_config_cannot_break_out_of_script_element():
    hostile_id = "league</script><script>alert(1)</script>&"
    body = build_cheat_sheet_body(hostile_id, 2026, "sleeper")
    config_prefix = body.split("<div class=\"cs-wrap\">", 1)[0]

    assert hostile_id not in config_prefix
    assert "\\u003c/script\\u003e" in config_prefix
    assert _embedded_config(body)["leagueId"] == hostile_id


def test_csv_export_is_free_for_non_premium_viewers():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper", has_premium=False)
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert 'id="csCsvBtn"' in body
    assert "CSV (Pro)" not in script
    assert "if (csvBtn) csvBtn.addEventListener('click', function () {" in script
    assert "exportCsv();" in script
    # Custom-board edits stay gated; CSV is the only draft-cheat-sheet control
    # that used to paywall on click and now always exports.
    assert "if (!cfg.hasPremium) {" in script
    assert "window.showPaywall('draft-cheat-sheet')" in script


def test_live_sync_is_explicit_and_drafted_players_offer_board_reset():
    body = build_cheat_sheet_body(
        "league-123", 2026, "sleeper", has_premium=True,
    )
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert 'id="csConnectLive"' in body
    assert 'id="csResetBoardBtn"' in body
    assert "loadPlayers();" in script
    assert "loadPlayers().then(function ()" not in script
    assert "(hasOverrides() || state.done.size || draftedIds)" in script


def test_desktop_header_keeps_controls_beside_title_without_wrapping():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")

    assert ".cs-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; flex-wrap: nowrap; }" in body
    assert ".cs-controls { display: flex; flex: 0 0 auto;" in body
    assert ".cs-ctrl-row { display: flex; align-items: center; gap: 9px; flex-wrap: nowrap;" in body


def test_mobile_header_has_no_flex_basis_gap_and_controls_wrap():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")

    assert ".cs-top > :first-child { flex: 0 0 auto; min-width: 0; width: 100%; }" in body
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in body
    assert ".cs-ctrl-row:last-child .cs-src, .cs-ctrl-row:last-child .csd-wrap { grid-column: 1 / -1;" in body


def test_market_column_is_conditionally_omitted_from_table_and_export():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert ".cs-wrap table { min-width: 910px; }" in body
    assert ".cs-vor-col, .cs-value-col { display: none; }" not in body
    assert ".cs-market-col { display: none; }" not in body
    assert "sortTh('market', 'Market vs ADP', 'cs-market-col'" in script
    assert 'class="cs-vor-col"' in script
    assert 'class="cs-value-col"' in script
    assert "var SHOW_MARKET_VS_ADP = false" in script
    assert "SHOW_MARKET_VS_ADP = resp.market_vs_adp_available === true" in script
    assert "showMarket(dyn) ? sortTh('market'" in script
    assert "showMarket(dyn) ? ['Market vs ADP'] : []" in script
    assert "showHist(dyn) ? ['Hist P(top-12)'] : []" in script
    assert "Not enough independent market data yet." in script
    assert "marketBasis" in script
    assert "marketConfidenceLabel" in script


def test_cheat_sheet_adds_full_season_schedule_rank_context():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert "Schedule Rank compares each player's position-specific matchups" in body
    assert "week_start=1&week_end=17" in script
    assert "p.sos_rank" in script
    assert "sortTh('scheduleRank', 'Sched Rk'" in script
    assert "'Schedule Rank'" in script


def test_cheat_sheet_adds_projected_ppg_to_board_and_export():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert "Projected PPG is the player's upcoming-season fantasy points per game from Sleeper" in body
    assert "projectedPpg:" in script
    assert "p.proj_ppg" in script
    assert "sortTh('projectedPpg', 'Proj PPG'" in script
    assert "'Proj PPG'" in script
    assert "x.projectedPpg.toFixed(1)" in script
    assert "Last season actual" not in script
    assert "cs-ppg-last" not in script


def test_cheat_sheet_consensus_adp_matches_rankings_to_one_decimal():
    """Cheat sheet ADP must be the rankings Consensus column, to one decimal.

    The dropdown used to label Auto as Consensus while reading the Sleeper
    overlay on avg_pick and Math.round-ing it, so 12.4 on Player Rankings
    showed as 12 (or a different source entirely) on the sheet.
    """
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()
    core = (Path(__file__).parents[1] / "static" / "draft_board_core.js").read_text()
    rankings = (Path(__file__).parents[1] / "static" / "rankings.js").read_text()

    assert "function sourceAdpOf(p, source, mode, sf)" in core
    assert "p.adp_by_source && p.adp_by_source[source]" in core
    assert "function consensusAdpOf(p, mode, sf)" in core
    assert "function sheetAdpOf(p, mode, sf)" in script
    assert "C.sourceAdpOf ? C.sourceAdpOf(p, src, mode, sf)" in script
    assert "adp: sheetAdpOf(p, mode, sf)" in script
    assert "return (state.adpSource && state.adpSource !== 'auto') ? state.adpSource : 'consensus';" in script
    assert "Number(v).toFixed(1)" in script
    assert "x.adp != null ? Math.round(x.adp) : ''" not in script
    assert "adp_source=' + encodeURIComponent(state.adpSource)" not in script
    assert "params = params.concat(leagueParams());" in script
    # Rankings Consensus column is the same 1-decimal adp_by_source field.
    assert "prAdpSourceVal(p, c.value)" in rankings
    assert "(v != null ? v.toFixed(1) : '–')" in rankings


def test_draft_room_only_shares_context_from_a_visible_draft_board():
    script = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()

    assert "(state.mode === 'mock' || state.mode === 'live')" in script
    assert "main && main.style.display !== 'none'" in script
    assert "q.push('live=1')" not in script


def test_draft_room_overlay_stays_in_sync_with_picks():
    """Open overlay is a snapshot first; later picks arrive via postMessage.

    Live Sleeper polling stays opt-in. Cross-off uses the same `drafted` map as
    best-available (keepers included), not only board cells.
    """
    room = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()
    sheet = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert "function cheatDraftedIds()" in room
    assert "Object.keys(drafted).forEach" in room
    assert "type: 'drCheatContext'" in room
    assert "function pushCheatSheetContext()" in room
    assert "pushCheatSheetContext();" in room
    assert "type === 'drCheatReady'" in room
    assert "q.push('live=1')" not in room
    assert "function applyDraftRoomContext(payload)" in sheet
    assert "payload.type !== 'drCheatContext'" in sheet
    assert "type: 'drCheatReady'" in sheet
    assert "e.origin !== window.location.origin" in sheet


def test_draft_room_cheat_sheet_shows_recommendation_context_without_reordering():
    room = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()
    sheet = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")

    assert "function cheatRecommendationOrder()" in room
    assert "rec_order=" in room
    assert "var recommendationOrder = null" in sheet
    assert "x.recRank =" in sheet
    assert "recommendationOrder[x.id]" in sheet
    assert "REC #' + x.recRank" in sheet
    assert "applyOverrides();" in sheet
    assert "scored.sort(function (a, b) {" in sheet
    assert "var aVor = Number(a.vorRaw);" in sheet
    assert "recommendationOrder[a.id]" not in sheet
    assert "Live context without reordering the sheet" in body


def test_in_draft_cheat_sheet_scrolls_once_to_first_available_player():
    sheet = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert "scrollToFirstAvailable = draftedIds.size > 0" in sheet
    assert "#csBoardBody tr.cs-p:not(.drafted):not(.done)" in sheet
    assert "scroller.scrollTop = Math.max(0, row.offsetTop - 4)" in sheet
    assert "scrollToFirstAvailable = false" in sheet
    assert "x.rk = i + 1" in sheet


def test_cheat_sheet_projects_snake_picks_for_a_selected_slot():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    sheet = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()
    room = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()

    assert 'id="csPickSlot"' in body
    assert "Your snake slot on this board" in body
    assert ".cs-projline" in body
    assert ".cs-board.hidedrafted .cs-proj-taken" in body
    assert "function snakePickNum" in sheet
    assert "var inRound = (round % 2 === 1) ? slot : (nTeams - slot + 1);" in sheet
    assert "Proj Pick ' + pk.label" in sheet
    assert "projLineRow(pk, span, x.drafted)" in sheet
    assert "qp.get('slot')" in sheet
    assert "cspickslot:" in sheet
    assert "q.push('slot='" in room
    assert "q.push('teams='" in room
    assert "pickSlot: 0" in sheet


def test_cheat_sheet_proj_pick_uses_custom_select_dropdown():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    sheet = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()
    csd = (Path(__file__).parents[1] / "static" / "custom_selects.js").read_text()
    from dashboard_services.pages.cheat_sheet_page import build_cheat_sheet_embed_document
    embed = build_cheat_sheet_embed_document("league-123", 2026, "sleeper")

    assert "custom_selects.js" in body
    assert "custom_selects.js" in embed
    assert "window.initCustomSelects" in csd
    assert "if (window.initCustomSelects) window.initCustomSelects" in sheet
    assert ".cs-filterbar .cs-src, .cs-filterbar .csd-wrap { flex: 0 0 auto; min-width: 168px; }" in body
    assert ".cs-wrap .csd-trigger { font-size: 12px; font-weight: 700;" in body
    assert "sel.parentNode.querySelector('.csd-value')" in sheet


def test_in_draft_cheat_sheet_is_a_full_screen_mobile_dialog():
    from dashboard_services.pages.draft_room_page import build_draft_room_body

    body = build_draft_room_body(None, None, None)
    script = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()

    assert 'role="dialog" aria-modal="true" aria-labelledby="drCheatTitle"' in body
    assert ".dr-cheat-overlay { padding: 0; align-items: stretch;" in body
    assert "width: 100%; height: 100vh; height: 100dvh;" in body
    assert "body.dr-cheat-open { overflow: hidden; }" in body
    assert "min-width: 0; min-height: 0;" in body
    assert ".dr-cheat-frame { height: 0; }" in body
    assert "document.body.classList.add('dr-cheat-open')" in script
    assert "document.body.classList.remove('dr-cheat-open')" in script


def test_embedded_cheat_sheet_keeps_mobile_content_scrollable():
    from dashboard_services.pages.cheat_sheet_page import build_cheat_sheet_embed_document

    document = build_cheat_sheet_embed_document("league-123", 2026, "sleeper")

    assert "overflow-x:hidden" in document
    assert "-webkit-overflow-scrolling: touch; touch-action: pan-x pan-y;" in document
    assert ".cs-tbl-scroll, .cs-pgrid-scroll { max-height: none; height: auto; }" in document


def test_cheat_sheet_big_board_columns_are_sortable():
    """Big Board headers reorder the table without changing the VOR model.

    ADP / Value / Proj PPG / Sched Rk (and the other data columns) are click-
    to-sort. Default remains VOR descending. By Position, custom-board edits,
    and Rk stay on the model board.
    """
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    for key in ("rk", "name", "pos", "vor", "projectedPpg", "scheduleRank", "market", "hist"):
        assert "sortTh('%s'" % key in script
    assert "col5Key = dyn ? 'age' : 'adp'" in script
    assert "col6Key = dyn ? 'window' : 'value'" in script
    assert "function displayPlayers()" in script
    assert "function setSort(key)" in script
    assert "isDefaultSort()" in script
    assert "thead th[data-sort]" in script
    assert "displayPlayers().forEach" in script
    assert "displayPlayers().map" in script
    assert "scored.sort(function (a, b) {" in script
    assert "var aVor = Number(a.vorRaw);" in script
    assert "boardSort && !recommendationOrder && x.grp !== lastT" in script
    assert "var pickAt = boardSort ? projPickMap() : {}" in script
    assert "th.cs-sort" in body
    assert ".cs-sortbtn" in body
    assert "Click a column header (ADP, Value, Proj PPG, Sched Rk" in body
    # Custom edits snap back to the VOR board so drag-reorder isn't fighting ADP.
    assert "if (editBoard) resetBoardSort()" in script


def test_cheat_sheet_hist_column_is_descriptive_and_lazy():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()
    core = (Path(__file__).parents[1] / "static" / "draft_board_core.js").read_text()
    pick = (Path(__file__).parents[1] / "static" / "pick_score.js").read_text()

    assert "var SHOW_HISTORICAL = false" in script
    assert "SHOW_HISTORICAL = resp.historical_available === true" in script
    assert "sortTh('hist', 'Hist'" in script
    assert "function histCell(x, dyn)" in script
    assert "/api/historical-player/" in script
    assert "openHistPanel" in script
    assert "e.target.closest('.cs-hist-btn')" in script
    assert "scored.sort(function (a, b) {" in script
    assert "var aVor = Number(a.vorRaw);" in script
    assert "histP" not in script.split("scored.sort(function (a, b) {")[1].split("scored.forEach")[0]
    assert "Similar-profile top-12 trend" in script
    assert "historical trends for this profile" in body.lower()
    assert "return !dyn && SHOW_HISTORICAL" in script
    assert "var HIST_STRONG_PCT = 25" in script
    assert "var HIST_TIER_SHORT = { top_5: 'top-5', top_12: 'top-12', top_24: 'top-24' }" in script
    assert "(lead.pct != null ? lead.pct : '-')" in script
    assert "(row.pct != null ? row.pct + '%' : '-')" in script
    assert "—" not in script.split("function renderHistPanel")[1].split("function init()")[0]
    assert "–" not in script.split("function renderHistPanel")[1].split("function init()")[0]
    assert "market_higher" not in script.split("function histCell")[1].split("function smallVal")[0]
    assert "Hist is redraft-only" in body
    assert "copy.trends" in script or "Trends for this player's buckets" in script
    assert "trendsHitRow(row, row && row.polarity, histBaseline, histSpan)" in script
    assert "copy.projection_trends" not in script
    assert "This board's projection" not in script
    assert "cs-hist-hit-top" in script
    assert "margin-top: -2px" not in body
    assert ".cs-wrap, .cs-hist-modal {" in body
    assert "z-index: var(--z-modal, 10000)" in body
    assert ".cs-hist-head" in body
    assert ".cs-hist-dl dt, .cs-hist-dl dd { margin: 0; }" in body
    assert "grid-column: 1" in body
    assert "grid-column: 2" in body
    assert ".cs-hist-close { flex-shrink: 0;" in body
    assert "float: right" not in body
    assert ".cs-hist-modal, .cs-hist-btn { display: none !important; }" in body
    assert ".cs-hist-modal { align-items: flex-end" not in body
    assert "14px 14px 0 0" not in body
    assert ".cs-hist-chip" in body
    assert ".cs-hist-hit-pct" in body
    assert "align-items: center" in body
    assert "data-hist-adp" in script
    assert "liveHistAdp" in script
    assert "mkt_sentence" in script
    assert "redraft_avg_pick=" in script
    assert "Object.keys((resp.preseason)" not in script
    assert "Named comps (this player excluded)" not in script
    assert 'data-tab="trends"' in body
    assert 'id="cs-panel-trends"' in body
    assert "/api/historical-trends" in script
    assert "function loadTrends" in script
    assert "function renderTrends" in script
    assert "cs-panel-trends" in script
    assert "if (!on && currentTab === 'trends') showSheetTab('board')" in script
    assert ".cs-trends-grid" in body
    assert ".cs-trends-bar" in body
    assert ".cs-trends-callouts" in body
    assert ".cs-trends-ages" in body
    assert ".cs-trends-rail" in body
    assert ".cs-trends-lanes" in body
    assert ".cs-trends-conf" in body
    assert "Descriptive — not a ranking input" not in body
    assert "cs-trends-honesty" not in body
    assert "Not a ranking score." in body
    assert ".cs-trends-callout-col" in body
    assert "grid-template-columns: 1fr 1fr" in body
    assert "cs-trends-age-tip" in body
    assert "trendsQualifyLabel" in script
    assert "function trendsQualifyLabel" in script
    assert "data-age-tip" in script
    assert "Open one, or pick a lane." in script
    assert "trendsTopEdges(sections, 10)" in script
    assert "data-trends-lane" in script
    assert "row.vs_label" in script
    assert "The Trends tab shows position-wide rates" in body
    assert "--cs-qb: #3b82f6" in body
    assert "--cs-rb: #22c55e" in body
    assert "--cs-wr: #f59e0b" in body
    assert "--cs-te: #8b5cf6" in body
    assert "--cs-pos:" in body
    assert "cs-hist-modal.cs-hist-wr" in body
    assert "background: var(--cs-pos)" in body
    assert "function applyHistPos" in script
    assert "function histTrendTitle" in script
    assert "function trendsBaselineOf" in script
    assert "trendsRailHtml(row.pct, base, pol, span)" in script
    assert 'id="csHistPos"' in body
    assert "badge(x.pos)" not in script
    assert "function badge(" not in script
    assert "cs-p cs-c-' + x.pos" in script
    assert ".cs-c-QB .cs-pgn" in body
    assert ".cs-c-QB .cs-pname" not in body
    assert "id !== 'adp' && id !== 'adp_positional'" in script
    assert "['career', 'Career']" in script
    assert "['adp', 'ADP']" not in script
    assert "p_hit_pct" not in pick
    assert "historical-player" not in core
    assert "p_hit_pct" not in core


def test_changelog_announces_trends_and_hist_without_em_dashes():
    from dashboard_services.changelog import CHANGELOG

    entry = next(e for e in CHANGELOG if e.get("link") == "/draft/cheat-sheet")
    assert entry["date"] == "2026-08-28"
    assert entry["tag"] == "new"
    assert "Trends" in entry["text"]
    assert "Hist" in entry["text"]
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


def test_changelog_announces_portfolio_positional_percentiles():
    from dashboard_services.changelog import CHANGELOG

    entry = CHANGELOG[0]
    assert entry["date"] == "2026-08-28"
    assert entry["tag"] == "fix"
    assert entry["link"] == "/portfolio"
    assert "percentile" in entry["text"].lower()
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]

