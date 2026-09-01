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


def test_recommend_waivers_uses_pickup_score_and_skips_owned():
    from utils.digest_actions import recommend_waivers

    rows = [
        {"id": "owned", "name": "Starter", "pos": "WR", "value": 800},
        {"id": "fa1", "name": "Need RB", "pos": "RB", "value": 120, "age": 24},
        {"id": "k", "name": "Kicker", "pos": "K", "value": 500},
        {"id": "fa2", "name": "Deep WR", "pos": "WR", "value": 40, "age": 28},
    ]
    hits = recommend_waivers(
        rows, {"owned"},
        roster_players=["owned"],
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        pidx={"owned": {"position": "WR"}, "fa1": {"position": "RB"}},
        fmt={"is_dynasty": True, "is_superflex": False},
        limit=3, min_score=1.0,
    )
    ids = [h["player_id"] for h in hits]
    assert "owned" not in ids
    assert "k" not in ids
    assert "fa1" in ids
    assert hits[0]["name"] == "Need RB"


def test_recommend_waivers_skips_duplicate_pos_reason():
    from utils.digest_actions import recommend_waivers

    rows = [
        {"id": "w1", "name": "Parker Washington", "pos": "WR", "value": 90, "age": 23},
        {"id": "w2", "name": "Luther Burden III", "pos": "WR", "value": 88, "age": 22},
        {"id": "w3", "name": "Need RB", "pos": "RB", "value": 70, "age": 24},
    ]
    hits = recommend_waivers(
        rows, set(),
        roster_players=[],
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        pidx={r["id"]: {"position": r["pos"]} for r in rows},
        fmt={"is_keeper": True, "is_dynasty": False, "is_superflex": False},
        limit=3, min_score=1.0,
    )
    names = [h["name"] for h in hits]
    assert not ("Parker Washington" in names and "Luther Burden III" in names)


def test_start_sit_swap_note_reuses_projection_upgrades():
    from unittest import mock
    from utils.digest_actions import start_sit_swap_note

    roster = {"players": ["a", "b"], "starters": ["a"], "reserve": [], "taxi": []}
    pidx = {"a": {"full_name": "Austin Ekeler", "position": "RB"},
            "b": {"full_name": "Kyren Williams", "position": "RB"}}
    with mock.patch(
        "utils.lineup_issues.projection_upgrades",
        return_value=[{"in": "b", "out": "a", "gain": 6.0}],
    ):
        note = start_sit_swap_note(
            starters=["a"], roster=roster, pidx=pidx, nfl_players={},
            proj_map={"a": 8.0, "b": 14.0},
            roster_positions=["RB", "FLEX"],
            min_gain=2.0,
        )
    assert note is not None
    assert "Consider Kyren Williams over Austin Ekeler" in note["body"]
