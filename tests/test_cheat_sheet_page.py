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
    assert ".cs-ctrl-row:last-child .cs-src { grid-column: 1 / -1;" in body


def test_market_column_is_conditionally_omitted_from_table_and_export():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert ".cs-wrap table { min-width: 830px; }" in body
    assert ".cs-vor-col, .cs-value-col { display: none; }" not in body
    assert ".cs-market-col { display: none; }" not in body
    assert '<th class="cs-market-col">Market vs ADP</th>' in script
    assert 'class="cs-vor-col"' in script
    assert 'class="cs-value-col"' in script
    assert "var SHOW_MARKET_VS_ADP = false" in script
    assert "SHOW_MARKET_VS_ADP = resp.market_vs_adp_available === true" in script
    assert "showMarket(dyn) ? '<th class=\"cs-market-col\">Market vs ADP</th>' : ''" in script
    assert "showMarket(dyn) ? ['Market vs ADP'] : []" in script
    assert "Not enough independent market data yet." in script
    assert "marketBasis" in script
    assert "marketConfidenceLabel" in script


def test_cheat_sheet_adds_full_season_schedule_rank_context():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert "Schedule Rank compares each player's position-specific matchups" in body
    assert "week_start=1&week_end=17" in script
    assert "p.sos_rank" in script
    assert ">Sched Rk</th>" in script
    assert "'Schedule Rank'" in script


def test_cheat_sheet_adds_projected_ppg_to_board_and_export():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert "Projected PPG is the player's upcoming-season fantasy points per game" in body
    assert "projectedPpg: p.proj_ppg" in script
    assert ">Proj PPG</th>" in script
    assert "'Proj PPG'" in script
    assert "x.projectedPpg.toFixed(1)" in script


def test_draft_room_only_shares_context_from_a_visible_draft_board():
    script = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()

    assert "(state.mode === 'mock' || state.mode === 'live')" in script
    assert "main && main.style.display !== 'none'" in script
    assert "q.push('live=1')" not in script


def test_draft_room_cheat_sheet_shows_recommendation_context_without_reordering():
    room = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()
    sheet = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()
    body = build_cheat_sheet_body("league-123", 2026, "sleeper")

    assert "function cheatRecommendationOrder()" in room
    assert "rec_order=" in room
    assert "var recommendationOrder = null" in sheet
    assert "x.recRank = recommendationOrder" in sheet
    assert "REC #' + x.recRank" in sheet
    assert "applyOverrides();" in sheet
    assert "scored.sort(function (a, b) { return b.vor - a.vor" in sheet
    assert "recommendationOrder[a.id]" not in sheet
    assert "Live context without reordering the sheet" in body


def test_in_draft_cheat_sheet_scrolls_once_to_first_available_player():
    sheet = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text()

    assert "scrollToFirstAvailable = draftedIds.size > 0" in sheet
    assert "#csBoardBody tr.cs-p:not(.drafted):not(.done)" in sheet
    assert "scroller.scrollTop = Math.max(0, row.offsetTop - 4)" in sheet
    assert "scrollToFirstAvailable = false" in sheet
    assert "x.rk = i + 1" in sheet


def test_in_draft_cheat_sheet_is_a_full_screen_mobile_dialog():
    from dashboard_services.pages.draft_room_page import build_draft_room_body

    body = build_draft_room_body(None, None, None)
    script = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()

    assert 'role="dialog" aria-modal="true" aria-labelledby="drCheatTitle"' in body
    assert ".dr-cheat-overlay { padding: 0; align-items: stretch;" in body
    assert "width: 100%; height: 100vh; height: 100dvh;" in body
    assert "body.dr-cheat-open { overflow: hidden; }" in body
    assert "document.body.classList.add('dr-cheat-open')" in script
    assert "document.body.classList.remove('dr-cheat-open')" in script
