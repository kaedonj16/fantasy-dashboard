"""Tests for Sleeper / ESPN alternate Redzone play-by-play helpers."""
from utils.redzone_alt_pbp import (
    build_name_indexes,
    extract_espn_pbp_plays,
    extract_espn_scoreboard_lookup,
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


def _espn_scoreboard_payload():
    return {
        "content": {
            "sbData": {
                "events": [
                    {
                        "id": "401700000",
                        "date": "2026-09-11T00:20Z",
                        "status": {
                            "displayClock": "12:34",
                            "period": 2,
                            "type": {
                                "state": "in",
                                "completed": False,
                                "shortDetail": "12:34 - 2nd",
                            },
                        },
                        "competitions": [
                            {
                                "competitors": [
                                    {
                                        "homeAway": "home",
                                        "team": {"abbreviation": "SEA"},
                                        "score": "10",
                                    },
                                    {
                                        "homeAway": "away",
                                        "team": {"abbreviation": "NE"},
                                        "score": "7",
                                    },
                                ]
                            }
                        ],
                    },
                    {
                        "id": "401700001",
                        "date": "2026-09-14T17:00Z",
                        "status": {
                            "period": 4,
                            "type": {
                                "state": "post",
                                "completed": True,
                                "shortDetail": "Final",
                            },
                        },
                        "competitions": [
                            {
                                "competitors": [
                                    {
                                        "homeAway": "home",
                                        "team": {"abbreviation": "WSH"},
                                        "score": "24",
                                    },
                                    {
                                        "homeAway": "away",
                                        "team": {"abbreviation": "DAL"},
                                        "score": "21",
                                    },
                                ]
                            }
                        ],
                    },
                ]
            }
        }
    }


def test_extract_espn_scoreboard_lookup_shapes_tank01_style():
    lookup = extract_espn_scoreboard_lookup(_espn_scoreboard_payload())
    # Both teams of each game are keyed to the same game dict.
    assert lookup["SEA"] is lookup["NE"]
    sea = lookup["SEA"]
    assert sea["gameID"] == "20260911_NE@SEA"
    assert sea["gameStatusCode"] == "1"  # in-progress
    assert sea["gameClock"] == "12:34"
    assert sea["lineScore"]["period"] == "2"
    assert sea["homePts"] == "10" and sea["awayPts"] == "7"
    assert sea["source"] == "espn"


def test_extract_espn_scoreboard_normalizes_abbrev_and_final():
    lookup = extract_espn_scoreboard_lookup(_espn_scoreboard_payload())
    # ESPN's WSH is normalized to Sleeper/Tank01 WAS so rostered teams match.
    assert "WAS" in lookup and "WSH" not in lookup
    was = lookup["WAS"]
    assert was["gameID"] == "20260914_DAL@WAS"
    assert was["gameStatusCode"] == "2"  # final


def test_extract_espn_scoreboard_empty_is_safe():
    assert extract_espn_scoreboard_lookup({}) == {}
    assert extract_espn_scoreboard_lookup(None) == {}  # type: ignore[arg-type]
    assert extract_espn_scoreboard_lookup({"content": {"sbData": {"events": []}}}) == {}
