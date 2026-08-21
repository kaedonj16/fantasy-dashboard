"""Rendering guards for Draft Room scoring controls."""

from dashboard_services.pages.draft_room_page import build_draft_room_body
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_draft_room_offers_four_and_six_point_passing_touchdowns():
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert 'id="drPassTd"' in body
    assert '<option value="4" selected>4 points</option>' in body
    assert '<option value="6">6 points</option>' in body


def test_league_scoring_is_available_to_live_and_mock_drafts():
    body = build_draft_room_body(
        "league", 2026, "sleeper", scoring={"ppr": 0.5, "tep": 0, "passTd": 6},
    )
    match = re.search(r"window\.__draftCfg = (.*?);</script>", body)
    assert match
    assert json.loads(match.group(1))["scoring"] == {"ppr": 0.5, "tep": 0, "passTd": 6}


def test_setup_source_and_draft_pick_pills_match_canonical_chip_styles():
    body = build_draft_room_body(None, None, None, is_guest=True)

    assert ".dr-roster-src-tag, .dr-cap-pill {" in body
    assert "background:var(--row); border:1px solid var(--grid); border-radius:6px; padding:2px 8px;" in body
    assert "color:var(--text-muted); font-size:11px; font-weight:700; line-height:1.45; white-space:nowrap;" in body
    assert ".dr-roster-src-tag { text-transform:none; letter-spacing:normal; }" in body
    assert "rgba(168,85,247,.14)" not in body


def test_player_load_failure_exposes_api_error_and_retry_control():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert "Player API HTTP " in source
    assert "Player API returned non-JSON" in source
    assert "retry.addEventListener('click', loadPlayers)" in source
    assert "console.error('[draft-room] loadPlayers failed', err)" in source


def test_pick_reason_uses_its_own_current_pick_variable():
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
    match = re.search(r"function pickReason\(p, counts\)\{(.*?)\n  \}", source, re.DOTALL)

    assert match
    body = match.group(1)
    assert "var pickNo = (state && state.current) || 1;" in body
    assert "_pn" not in body


def test_board_offers_pick_score_sort_instead_of_steals():
    body = build_draft_room_body(None, None, None, is_guest=True)
    source = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")

    assert 'data-val="pickscore">Pick Score</button>' in body
    assert 'data-val="steals"' not in body
    assert "if (sortBy === 'pickscore'){ return (b._ps || 0) - (a._ps || 0); }" in source
    assert "pickscore: 'Pick Score'" in source
