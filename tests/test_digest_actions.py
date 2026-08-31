from __future__ import annotations

from utils.digest_actions import (
    action_section_html,
    lineup_digest_note,
    player_deep_link,
    top_waiver_from_values,
)


def test_action_section_html_requires_title_and_body():
    assert action_section_html("", "body") == ""
    assert action_section_html("T", "") == ""
    html = action_section_html("Waiver wire", "Add Player X", href="https://ex/w", cta="Go →")
    assert "Waiver wire" in html
    assert "Add Player X" in html
    assert "https://ex/w" in html
    assert "Go →" in html


def test_lineup_digest_note_empty():
    assert lineup_digest_note([]) is None


def test_lineup_digest_note_injury():
    note = lineup_digest_note([
        {"kind": "injury", "pid": "1", "name": "A", "detail": "A is listed Out"},
    ])
    assert note is not None
    assert "injured" in note["title"].lower()
    assert "Out" in note["body"]


def test_top_waiver_from_values_skips_owned_and_kickers():
    rows = [
        {"id": "a", "name": "Owned", "pos": "WR", "value": 200},
        {"id": "k", "name": "Kicker", "pos": "K", "value": 500},
        {"id": "b", "name": "Free", "pos": "RB", "value": 120},
        {"id": "c", "name": "Cheap", "pos": "WR", "value": 10},
    ]
    hit = top_waiver_from_values(rows, {"a"})
    assert hit is not None
    assert hit["player_id"] == "b"
    assert hit["name"] == "Free"


def test_player_deep_link_encodes_query():
    url = player_deep_link("https://brfantasy.com", "espn", 2025, "9", "4046", "J. Chase")
    assert url.startswith("https://brfantasy.com/espn/2025/9/dashboard?player=4046")
    assert "player_name=" in url
