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


def test_mobile_big_board_pins_rank_and_player():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()
    mobile = body.split("@media (max-width: 640px)")[1].split("@media")[0]

    assert "sortTh('rk', 'Rk', 'cs-rk'" in script
    assert "sortTh('name', 'Player', 'l cs-player'" in script
    assert '<td class="cs-player">' in script
    assert "left: 0" in mobile
    assert "left: 42px" in mobile
    assert "position: sticky" in mobile
    assert ".cs-wrap thead th.cs-rk" in mobile
    assert ".cs-wrap thead th.cs-player" in mobile
    assert "border-collapse: separate" in mobile


def test_market_column_is_conditionally_omitted_from_table_and_export():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert "min-width: 910px" in body
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
    assert "params = ['view=board']" in script
    assert "window.__cheatPlayersP" in script
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


def test_in_draft_cheat_sheet_sits_beside_undo_not_in_settings():
    """On the live/mock board, Cheat Sheet is one tap — not buried in Settings.

    The control lives in the always-visible .dr-side-opts cluster (Undo, Cheat,
    Trade, Settings). That wrap relocates to the status bar on desktop and
    beside the side-panel tabs on mobile, so the link stays reachable either
    way. Setup-page hero #drToCheatSheet is unchanged. Unmodified click still
    opens the overlay; Cmd/Ctrl/middle-click still opens a tab.
    """
    from dashboard_services.pages.draft_room_page import build_draft_room_body

    body = build_draft_room_body(None, None, None)
    script = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()

    opts_start = body.index('class="dr-side-opts"')
    panel_start = body.index('id="drOptsPanel"')
    cluster = body[opts_start:panel_start]
    panel = body[panel_start:body.index('id="drBestControls"')]

    assert 'id="drOptsCheatSheet"' in cluster
    assert 'id="drUndo"' in cluster
    assert 'id="drPickTradeBtn"' in cluster
    assert 'id="drOptsBtn"' in cluster
    assert 'class="dr-opts-trigger dr-cs-trigger"' in cluster
    assert 'id="drOptsCheatSheet"' not in panel
    assert 'id="drSummaryBtn"' in panel
    assert 'id="drToCheatSheet"' in body

    assert "var _cs2 = document.getElementById('drOptsCheatSheet');" in script
    assert "if (_cs2) _cs2.addEventListener('click', function(e){" in script
    assert "openCheatSheet();" in script
    assert "if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;" in script


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
    assert "function histPctClass" in script
    assert "HIST_STRONG_PCT" in script
    assert "history_higher" in script.split("function histPctClass")[1].split("function histCell")[0]
    assert "return 'b'" not in script.split("function histPctClass")[1].split("function histCell")[0]
    assert "Never paint market_higher red" in script
    assert "Players like this:" in script
    assert "title=\"This player\\'s historical chance\"" in script
    assert "top-12 chance for this profile" in script
    assert "Historical top-12 chance" in body
    assert "stars like Chase or Bijan" in body
    assert "Hist vs the ADP bucket" not in body
    assert "vs ADP bucket (signed pts)" not in script
    assert "function histEdgeClass" not in script
    assert "function histEdgeBody" not in script
    assert "return !dyn && SHOW_HISTORICAL" in script
    assert "var showTrends = function ()" in script
    assert "return SHOW_HISTORICAL;" in script.split("var showTrends = function ()")[1].split("var currentTab")[0]
    assert "var HIST_STRONG_PCT = 25" in script
    assert "historical chance for this career and situation" in body.lower()
    assert "var HIST_TIER_SHORT = { top_5: 'top-5', top_12: 'top-12', top_24: 'top-24' }" in script
    assert "function histExampleHit" in script
    assert "cs-hist-ex-hit" in script
    assert ".cs-hist-ex-hit" in body
    assert "(lead.pct != null ? lead.pct : '-')" in script
    assert "(row.pct != null ? row.pct + '%' : '-')" in script
    assert "—" not in script.split("function renderHistPanel")[1].split("function init()")[0]
    assert "–" not in script.split("function renderHistPanel")[1].split("function init()")[0]
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
    assert "if (!trendsOn && currentTab === 'trends') showSheetTab('board')" in script
    assert "1QB redraft history" in script
    assert ".cs-trends-grid" in body
    assert ".cs-trends-bar" in body
    assert ".cs-trends-callouts" in body
    assert ".cs-trends-ages" in body
    assert ".cs-trends-rail" in body
    assert ".cs-trends-lanes" in body
    assert ".cs-trends-tiers" in body
    assert ".cs-trends-scout" in body
    assert ".cs-trends-sticky" in body
    assert "top: var(--cs-nav-offset, 0px)" in body
    assert "max-height: min(42vh, 340px)" in body
    assert ".cs-trends-sticky.is-picked" in body
    assert ".cs-trends-sticky.is-collapsed" in body
    assert ".cs-trends-sticky-body" in body
    assert ".cs-trends-sticky-toggle" in body
    assert ".cs-trends-profile-tier.is-on" in body
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in body
    assert "box-shadow: inset 3px 0 0 var(--cs-pos)" in body
    mobile_trends = body.split("@media (max-width: 720px)")[1].split("@media")[0]
    assert "grid-template-columns: minmax(0, 1fr)" in mobile_trends
    peek_css = mobile_trends.split(".cs-trends-card-peek")[1].split("}")[0]
    assert "white-space: normal" in peek_css
    assert "overflow-wrap: anywhere" in peek_css
    assert "display: inline;" not in peek_css
    assert "backdrop-filter: blur(10px)" in body
    assert ".cs-trends-conf" in body
    assert "Descriptive — not a ranking input" not in body
    assert "cs-trends-honesty" not in body
    assert "Historical finish rates by bucket." in body
    assert "Not a ranking score." not in body
    assert ".cs-trends-callout-col" in body
    assert "grid-template-columns: 1fr 1fr" in body
    assert "cs-trends-age-tip" in body
    assert "trendsQualifyLabel" in script
    assert "function trendsQualifyLabel" in script
    assert "data-age-tip" in script
    assert "Open one, or pick a lane." in script
    assert "trendsTopEdges(sections, 10, trendsTier)" in script
    assert "data-trends-tier" in script
    assert "function trendsScoutHtml" in script
    assert "function trendsBoardFeaturesPayload" in script
    assert "board_features" in script
    assert "scout_matches" in script
    assert "function trendsFeatsForPlayer" not in script
    assert "function trendsLivePlayerFeatures" not in script
    assert "sticky.classList.toggle('is-picked'" in script
    assert "sticky.classList.toggle('is-collapsed'" in script
    assert "function setTrendsDockOpen" in script
    assert "var trendsDockOpen" in script
    assert 'data-trends-dock="1"' in script
    assert "cs-trends-sticky-body" in script
    assert "Actual matching seasons." in script
    assert "Actual matching seasons. Not a ranking input." not in script
    assert "draft-trends-scout" in script
    assert "Tap a bucket to list matching players." in script
    assert "Tap historical buckets to build a profile." in script
    assert "function loadTrendsCohort" in script
    assert "/api/historical-cohort" in script
    assert "Historical red flags" in script
    assert "Closest historical examples" in script
    assert "Two groups, not one chance" in script
    assert "Players like this" in script
    assert "anyone taken in that fantasy round" in script
    assert "Need live ADP to show the other group." in script
    assert "Expected at current ADP" not in script
    assert "Historical edge vs market" not in script
    assert ".cs-hist-compare" in body
    assert ".cs-hist-gap" in body
    assert "function trendsRedFlags" in script
    assert "ranking_edge" in script
    assert "'lte'" in script
    assert ".cs-trends-profile" in body
    assert ".cs-hist-market" in body
    assert ".cs-hist-ex-sum" in body
    assert "shrinkage-adjusted lift" in script
    assert "Top 24 is the flex line" not in script
    assert "Board players who match" in script
    grid_at = script.find("html += '<div class=\"cs-trends-grid\">'")
    scout_at = script.find("html += trendsScoutHtml")
    sticky_at = script.find("html += '<div class=\"cs-trends-sticky'")
    assert sticky_at >= 0 and scout_at > sticky_at and grid_at > scout_at
    assert "function paintTrendsSelection" in script
    assert "function bindTrendsDock" in script
    assert "function setTrendsNavOffset" in script
    assert "--cs-nav-offset" in script
    assert "querySelector('.top-nav')" in script
    assert "renderTrends({ keepScroll: true })" not in script
    scout_css = body.split(".cs-trends-scout-list")[1][:280]
    assert "grid-template-columns: 1fr 1fr" in scout_css
    assert "data-trends-lane" in script
    assert "row.vs_label" in script
    assert "stays available in dynasty" in body
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
    assert "ryoe: 'usage'" in script
    assert "touches: 'usage'" in script
    assert "receptions: 'usage'" in script
    assert "pass_attempts: 'usage'" in script
    assert "['adp', 'ADP']" not in script
    assert "p_hit_pct" not in pick
    assert "historical-player" not in core
    assert "p_hit_pct" not in core


def test_changelog_announces_trends_and_hist_without_em_dashes():
    from dashboard_services.changelog import CHANGELOG

    launch = next(
        entry for entry in CHANGELOG
        if entry.get("link") == "/draft/cheat-sheet" and entry.get("tag") == "new"
    )
    assert launch["date"] == "2026-08-28"
    assert "Trends" in launch["text"]
    assert "Hist" in launch["text"]
    assert "—" not in launch["text"]
    assert "–" not in launch["text"]
    scout = next(
        entry for entry in CHANGELOG
        if entry.get("tag") == "update" and "Pro" in entry.get("text", "")
    )
    assert "top-5" in scout["text"]
    assert "top-24" in scout["text"]
    assert "—" not in scout["text"]
    assert "–" not in scout["text"]
    hist_fix = next(
        entry for entry in CHANGELOG
        if "never top-12" in entry.get("text", "") and "Hist" in entry.get("text", "")
    )
    assert hist_fix["tag"] == "fix"
    assert hist_fix["link"] == "/draft/cheat-sheet"
    assert "Hist" in hist_fix["text"]
    assert "—" not in hist_fix["text"]
    assert "–" not in hist_fix["text"]
    example_tags = next(
        entry for entry in CHANGELOG
        if "closest-example tags" in entry.get("text", "").lower()
    )
    assert example_tags["tag"] == "fix"
    assert example_tags["link"] == "/draft/cheat-sheet"
    assert "—" not in example_tags["text"]
    assert "–" not in example_tags["text"]
    mobile_peek = next(
        entry for entry in CHANGELOG
        if "clipped off the right edge" in entry.get("text", "").lower()
    )
    assert mobile_peek["tag"] == "fix"
    assert mobile_peek["link"] == "/draft/cheat-sheet"
    assert "—" not in mobile_peek["text"]
    assert "–" not in mobile_peek["text"]
    example_hits = next(
        entry for entry in CHANGELOG
        if "closest examples now mark" in entry.get("text", "").lower()
    )
    assert example_hits["tag"] == "update"
    assert example_hits["link"] == "/draft/cheat-sheet"
    assert "Top-5" in example_hits["text"]
    assert "—" not in example_hits["text"]
    assert "–" not in example_hits["text"]
    capital_split = next(
        entry for entry in CHANGELOG
        if "Top 10" in entry.get("text", "") and "rest of Round 1" in entry.get("text", "")
    )
    assert capital_split["tag"] == "update"
    assert capital_split["link"] == "/draft/cheat-sheet"
    assert "—" not in capital_split["text"]
    assert "–" not in capital_split["text"]
    chance = next(
        entry for entry in CHANGELOG
        if "historical chance" in entry.get("text", "").lower()
        and "current situation" in entry.get("text", "").lower()
    )
    assert chance["tag"] == "fix"
    assert chance["link"] == "/draft/cheat-sheet"
    assert "Hist" in chance["text"]
    assert "—" not in chance["text"]
    assert "–" not in chance["text"]
    trends_place = next(
        entry for entry in CHANGELOG
        if "two columns" in entry.get("text", "").lower()
        and "RYOE" in entry.get("text", "")
    )
    assert trends_place["tag"] == "fix"
    assert trends_place["link"] == "/draft/cheat-sheet"
    assert "—" not in trends_place["text"]
    assert "–" not in trends_place["text"]
    volume = next(
        entry for entry in CHANGELOG
        if "400+" in entry.get("text", "")
        and "touches" in entry.get("text", "").lower()
    )
    assert volume["tag"] == "update"
    assert volume["link"] == "/draft/cheat-sheet"
    assert "receptions" in volume["text"].lower()
    assert "—" not in volume["text"]
    assert "–" not in volume["text"]
    dock = next(
        entry for entry in CHANGELOG
        if "sticky dock" in entry.get("text", "").lower()
    )
    assert dock["tag"] == "fix"
    assert dock["link"] == "/draft/cheat-sheet"
    assert "Pro" in dock["text"]
    assert "—" not in dock["text"]
    assert "–" not in dock["text"]
    nav_clear = next(
        entry for entry in CHANGELOG
        if "site nav" in entry.get("text", "").lower()
        and "dock" in entry.get("text", "").lower()
    )
    assert nav_clear["tag"] == "fix"
    assert nav_clear["link"] == "/draft/cheat-sheet"
    assert "—" not in nav_clear["text"]
    assert "–" not in nav_clear["text"]
    cohort = next(
        entry for entry in CHANGELOG
        if "combined historical hit rate" in entry.get("text", "").lower()
    )
    assert cohort["tag"] == "update"
    assert cohort["link"] == "/draft/cheat-sheet"
    assert "descriptive-only" in cohort["text"].lower()
    assert "—" not in cohort["text"]
    assert "–" not in cohort["text"]
    compact = next(
        entry for entry in CHANGELOG
        if "selected-bucket dock is compact" in entry.get("text", "").lower()
    )
    assert compact["tag"] == "fix"
    assert compact["link"] == "/draft/cheat-sheet"
    assert "rookies" in compact["text"].lower()
    assert "—" not in compact["text"]
    assert "–" not in compact["text"]
    profile_css = next(
        entry for entry in CHANGELOG
        if "selected profile is a compact verdict card" in entry.get("text", "").lower()
    )
    assert profile_css["tag"] == "update"
    assert profile_css["link"] == "/draft/cheat-sheet"
    assert "—" not in profile_css["text"]
    assert "–" not in profile_css["text"]
    mix = next(
        entry for entry in CHANGELOG
        if "never top-12) no longer fails" in entry.get("text", "").lower()
        or "numeric one" in entry.get("text", "").lower()
    )
    assert mix["tag"] == "fix"
    assert mix["link"] == "/draft/cheat-sheet"
    assert "—" not in mix["text"]
    assert "–" not in mix["text"]
    collapse = next(
        entry for entry in CHANGELOG
        if "collapsed so the tables stay in view" in entry.get("text", "").lower()
    )
    assert collapse["tag"] == "update"
    assert collapse["link"] == "/draft/cheat-sheet"
    assert "Lane chips" in collapse["text"]
    assert "—" not in collapse["text"]
    assert "–" not in collapse["text"]
    example_context = next(
        entry for entry in CHANGELOG
        if "exp: year 4" in entry.get("text", "").lower()
        and "last year: top 5" in entry.get("text", "").lower()
    )
    assert example_context["tag"] == "fix"
    assert example_context["link"] == "/draft/cheat-sheet"
    assert "—" not in example_context["text"]
    assert "–" not in example_context["text"]
    pin_board = next(
        entry for entry in CHANGELOG
        if "rk and player stay pinned" in entry.get("text", "").lower()
    )
    assert pin_board["tag"] == "fix"
    assert pin_board["link"] == "/draft/cheat-sheet"
    assert "—" not in pin_board["text"]
    assert "–" not in pin_board["text"]
    two_groups = next(
        entry for entry in CHANGELOG
        if "two groups (players like this vs that adp round)" in entry.get("text", "").lower()
    )
    assert two_groups["tag"] == "fix"
    assert two_groups["link"] == "/draft/cheat-sheet"
    assert "combined chance" in two_groups["text"].lower()
    assert "—" not in two_groups["text"]
    assert "–" not in two_groups["text"]


def test_changelog_announces_portfolio_positional_percentiles():
    from dashboard_services.changelog import CHANGELOG

    entry = next(e for e in CHANGELOG if "percentile" in e.get("text", "").lower())
    assert entry["date"] == "2026-08-28"
    assert entry["tag"] == "fix"
    assert entry["link"] == "/portfolio"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


def test_changelog_announces_undrafted_draft_countdown():
    from dashboard_services.changelog import CHANGELOG

    entry = next(e for e in CHANGELOG if "countdown" in e.get("text", "").lower())
    assert entry["date"] == "2026-08-28"
    assert entry["tag"] == "fix"
    assert entry["link"] == "/portfolio"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


def test_changelog_announces_predraft_cheat_sheet_sidebar():
    from dashboard_services.changelog import CHANGELOG

    entry = next(e for e in CHANGELOG if "empty roster sidebar" in e.get("text", "").lower())
    assert entry["date"] == "2026-08-28"
    assert entry["tag"] == "fix"
    assert entry["link"] == "/draft/cheat-sheet"
    assert "cheat sheet" in entry["text"].lower()
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


def test_changelog_announces_in_draft_cheat_sheet_control():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        e for e in CHANGELOG
        if "settings dropdown" in e.get("text", "").lower()
        and "cheat sheet" in e.get("text", "").lower()
    )
    assert entry["date"] == "2026-08-28"
    assert entry["tag"] == "update"
    assert entry["link"] == "/draft"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]

