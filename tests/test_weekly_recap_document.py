"""Tests for the presentation-independent weekly recap document."""

from copy import deepcopy

from dashboard_services.recap.document import (
    apply_ai_narrative,
    build_recap_document,
    recap_document_from_json,
    recap_document_to_json,
)


def _payload() -> dict:
    return {
        "league_name": "Fourth and Long",
        "week": 6,
        "matchups": [
            {
                "team_a": "Gridiron Guild", "team_b": "Fourth & Long",
                "team_a_pts": 127.42, "team_b_pts": 127.50,
                "winner": "Fourth & Long",
            },
            {
                "team_a": "Sunday Scaries", "team_b": "Bye Week Bandits",
                "team_a_pts": 168.7, "team_b_pts": 104.2,
                "winner": "Sunday Scaries",
            },
        ],
        "high_scorer": {"team": "Sunday Scaries", "pts": 168.7},
        "low_scorer": {"team": "Bye Week Bandits", "pts": 104.2},
        "biggest_blowout": {
            "winner": "Sunday Scaries", "loser": "Bye Week Bandits",
            "winner_pts": 168.7, "loser_pts": 104.2, "margin": 64.5,
        },
        "upsets": [],
        "big_movers": [{"team": "Fourth & Long", "from": 6, "to": 3}],
        "playoff_race": {
            "cutline_team": "Gridiron Guild", "cutline_record": "3-3",
            "bubble_team": "Waiver Warriors", "bubble_record": "3-3", "tight": True,
        },
        "next_week_preview": None,
    }


def test_closest_finish_becomes_featured_story():
    document = build_recap_document(_payload())

    assert document["featured_story_id"] == "closest_finish"
    featured = document["stories"][0]
    assert featured["facts"]["winner"] == "Fourth & Long"
    assert round(featured["facts"]["margin"], 2) == 0.08
    assert "0.08 points" in featured["body"]


def test_large_upset_can_outrank_an_ordinary_close_finish():
    payload = _payload()
    payload["matchups"][0]["team_a_pts"] = 121
    payload["matchups"][0]["team_b_pts"] = 112
    payload["upsets"] = [{
        "winner": "Rebuild City", "loser": "Sunday Scaries",
        "winner_rank_before": 10, "loser_rank_before": 1, "margin": 18.0,
    }]

    document = build_recap_document(payload)

    assert document["featured_story_id"] == "biggest_upset"
    assert document["stories"][0]["facts"]["winner_rank_before"] == 10


def test_story_selection_limits_repeated_team_appearances():
    document = build_recap_document(_payload())

    appearances = sum("Sunday Scaries" in story["teams"] for story in document["stories"])
    assert appearances <= 2
    assert len({story["type"] for story in document["stories"]}) == len(document["stories"])


def test_document_signature_is_stable_and_changes_with_facts():
    first = build_recap_document(_payload())
    same = build_recap_document(_payload())
    changed_payload = _payload()
    changed_payload["high_scorer"]["pts"] = 169.0
    changed = build_recap_document(changed_payload)

    assert first["data_signature"] == same["data_signature"]
    assert first["data_signature"] != changed["data_signature"]


def test_ai_narrative_cannot_mutate_verified_facts():
    document = build_recap_document(_payload())
    facts_before = deepcopy(document["facts"])

    updated = apply_ai_narrative(document, {
        "headline": "One yard decided the week",
        "stories": [
            {
                "id": "closest_finish",
                "title": "Fourth & Long escaped",
                "body": "Fourth & Long survived by eight hundredths.",
                "facts": {"winner": "Invented Team"},
            },
            {"id": "invented_story", "title": "Fake", "body": "This must not appear."},
        ],
        "looking_ahead": "Next week is crowded at the cutline.",
        "facts": {"winner": "Invented Team"},
    })

    assert updated["facts"] == facts_before
    assert document["narrative"]["source"] == "deterministic"
    assert updated["narrative"]["source"] == "ai"
    assert updated["narrative"]["headline"] == "One yard decided the week"
    assert updated["stories"][0]["narrative"]["source"] == "ai"
    assert updated["stories"][0]["narrative"]["title"] == "Fourth & Long escaped"
    assert updated["narrative"]["paragraphs"][0] == "Fourth & Long survived by eight hundredths."
    assert all(story["id"] != "invented_story" for story in updated["stories"])
    assert "This must not appear." not in updated["narrative"]["paragraphs"]


def test_missing_ai_story_keeps_deterministic_fallback_in_selected_order():
    document = build_recap_document(_payload())
    updated = apply_ai_narrative(document, {
        "headline": "A close one",
        "stories": [{
            "id": document["stories"][0]["id"],
            "title": "Custom lead",
            "body": "Custom featured story.",
        }],
        "looking_ahead": "",
    })

    assert updated["narrative"]["paragraphs"][0] == "Custom featured story."
    assert updated["narrative"]["paragraphs"][1] == document["stories"][1]["body"]
    assert updated["stories"][1]["narrative"]["source"] == "deterministic"


def test_document_json_round_trip_rejects_unknown_schema():
    document = build_recap_document(_payload())
    loaded = recap_document_from_json(recap_document_to_json(document))

    assert loaded == document
    assert recap_document_from_json('{"schema_version":999}') is None
    assert recap_document_from_json("not json") is None
