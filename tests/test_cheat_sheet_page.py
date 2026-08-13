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


def test_draft_room_only_shares_context_from_a_visible_draft_board():
    script = (Path(__file__).parents[1] / "static" / "draft_room.js").read_text()

    assert "(state.mode === 'mock' || state.mode === 'live')" in script
    assert "main && main.style.display !== 'none'" in script
    assert "q.push('live=1')" not in script
