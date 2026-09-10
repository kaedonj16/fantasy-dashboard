"""Tests for Sleeper / ESPN alternate Redzone play-by-play helpers."""
from utils.redzone_alt_pbp import (
    build_name_indexes,
    extract_espn_pbp_plays,
    extract_sleeper_pbp_plays,
    parse_tank_game_id,
    pids_mentioned_in_text,
)


def test_parse_tank_game_id():
    assert parse_tank_game_id("20260909_NE@SEA") == ("20260909", "NE", "SEA")
    assert parse_tank_game_id("bad") == ("", "", "")


def test_abbrev_name_matching():
    full, abbrev = build_name_indexes({
        "drake maye": "11564",
        "rhamondre stevenson": "8155",
        "george holani": "12048",
    })
    text = "D.Maye scrambles up the middle to NE 21 for 10 yards (E.Jones)."
    assert pids_mentioned_in_text(text, full_index=full, abbrev_index=abbrev) == ["11564"]
    text2 = "R.Stevenson up the middle to NE 11 for 3 yards (D.Lawrence)."
    assert "8155" in pids_mentioned_in_text(text2, full_index=full, abbrev_index=abbrev)


def test_extract_espn_pbp_maps_abbreviated_names():
    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "1",
                                "text": "R.Stevenson up the middle to NE 11 for 3 yards (D.Lawrence).",
                                "clock": {"displayValue": "11:32"},
                                "period": {"number": 1},
                                "start": {"down": 1, "distance": 10, "possessionText": "NE 8"},
                                "type": {"text": "Rush"},
                                "scoringPlay": False,
                            },
                            {
                                "id": "2",
                                "text": "A.Borregales kicks 63 yards from NE 35 to SEA 2.",
                                "type": {"text": "Kickoff"},
                                "clock": {"displayValue": "15:00"},
                                "period": {"number": 1},
                                "start": {},
                            },
                        ]
                    }
                ]
            }
        }
    }
    plays = extract_espn_pbp_plays(
        payload,
        "20260909_NE@SEA",
        name_to_pid={"rhamondre stevenson": "8155"},
    )
    assert len(plays) == 1
    assert plays[0]["pid"] == "8155"
    assert plays[0]["play_text"].startswith("R.Stevenson")
    assert plays[0]["source"] == "espn"
    assert plays[0]["clock"] == "11:32"
    assert plays[0]["down"] == "1"


def test_extract_sleeper_pbp_when_rows_present():
    raw = [
        {
            "id": "sl1",
            "text": "Drake Maye pass complete to Hunter Henry for 12 yards",
            "quarter": "2",
            "clock": "5:01",
            "down": "2",
            "distance": "8",
        }
    ]
    plays = extract_sleeper_pbp_plays(
        raw,
        "20260909_NE@SEA",
        name_to_pid={"drake maye": "11564", "hunter henry": "4037"},
    )
    assert len(plays) == 2
    assert {p["pid"] for p in plays} == {"11564", "4037"}
    assert plays[0]["source"] == "sleeper"


def test_extract_sleeper_empty_is_safe():
    assert extract_sleeper_pbp_plays([], "g") == []
    assert extract_sleeper_pbp_plays(None, "g") == []  # type: ignore[arg-type]
