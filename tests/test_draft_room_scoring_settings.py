"""Rendering guards for Draft Room scoring controls."""

from dashboard_services.pages.draft_room_page import build_draft_room_body
import json
import re


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
