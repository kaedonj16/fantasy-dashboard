"""Tests for Sleeper / ESPN alternate Redzone play-by-play helpers."""
from utils.redzone_alt_pbp import (
    attach_cumulative,
    build_name_indexes,
    extract_espn_pbp_plays,
    extract_espn_scoreboard_lookup,
    extract_sleeper_pbp_plays,
    parse_pbp_play_stats,
    parse_tank_game_id,
    pids_mentioned_in_text,
)


def test_attach_cumulative_builds_running_totals_in_order():
    plays = [
        {"pid": "qb", "stat_line": {"pass_yds": 12, "pass_cmp": 1, "pass_att": 1}},
        {"pid": "qb", "stat_line": {"pass_att": 1}},  # incompletion
        {"pid": "qb", "stat_line": {"pass_yds": 1, "pass_cmp": 1, "pass_att": 1}},
        {"pid": "wr", "stat_line": {"rec": 1, "rec_yds": 12, "targets": 1}},
    ]
    out = attach_cumulative(plays)
    assert out[0]["cume"] == {"pass_yds": 12, "pass_cmp": 1, "pass_att": 1}
    # After the incompletion: 1/2 CMP, still 12 yards.
    assert out[1]["cume"] == {"pass_yds": 12, "pass_cmp": 1, "pass_att": 2}
    # After the third pass: 2/3 CMP, 13 yards.
    assert out[2]["cume"] == {"pass_yds": 13, "pass_cmp": 2, "pass_att": 3}
    assert out[3]["cume"] == {"rec": 1, "rec_yds": 12, "targets": 1}


def test_espn_extract_attaches_cumulative_per_player():
    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "1",
                                "text": "D.Maye pass short right to M.Hollins for 12 yards (E.Jones).",
                                "clock": {"displayValue": "2:00"},
                                "period": {"number": 4},
                                "start": {"down": 4, "distance": 9},
                                "type": {"text": "Pass"},
                            },
                            {
                                "id": "2",
                                "text": "D.Maye pass short left to M.Hollins to SEA 36 for 11 yards (E.Jones).",
                                "clock": {"displayValue": "1:40"},
                                "period": {"number": 4},
                                "start": {"down": 1, "distance": 10},
                                "type": {"text": "Pass"},
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
        name_to_pid={"drake maye": "11564", "malik hollins": "200"},
    )
    maye = [p for p in plays if p["pid"] == "11564"]
    assert maye[-1]["cume"] == {"pass_yds": 23, "pass_cmp": 2, "pass_att": 2}
    holl = [p for p in plays if p["pid"] == "200"]
    assert holl[-1]["cume"] == {"rec": 2, "rec_yds": 23, "targets": 2}


def test_parse_pbp_completed_pass_credits_passer_and_receiver():
    sl = parse_pbp_play_stats(
        "(Shotgun) D.Maye pass short right to M.Hollins pushed ob at SEA 16 "
        "for 12 yards (J.Jobe)."
    )
    assert sl["d.maye"] == {"pass_yds": 12, "pass_cmp": 1, "pass_att": 1}
    assert sl["m.hollins"] == {"rec": 1, "rec_yds": 12, "targets": 1}


def test_parse_pbp_td_pass_and_extra_point():
    sl = parse_pbp_play_stats(
        "D.Lock pass short left to J.Smith-Njigba for 45 yards, TOUCHDOWN. "
        "J.Myers extra point is GOOD, Center-C.Stoll, Holder-M.Dickson."
    )
    assert sl["d.lock"] == {"pass_yds": 45, "pass_cmp": 1, "pass_att": 1, "pass_td": 1}
    assert sl["j.smith-njigba"] == {"rec": 1, "rec_yds": 45, "targets": 1, "rec_td": 1}
    assert sl["j.myers"] == {"xpm": 1}


def test_parse_pbp_interception_only_credits_passer_pick():
    sl = parse_pbp_play_stats(
        "(Shotgun) D.Maye pass deep right intended for M.Hollins INTERCEPTED "
        "by J.Jobe [D.Lawrence] at SEA -3. Touchback."
    )
    assert sl == {"d.maye": {"int": 1, "pass_att": 1}}


def test_parse_pbp_sack_is_not_scored_as_a_rush():
    assert parse_pbp_play_stats(
        "(Shotgun) D.Maye sacked at SEA 28 for -7 yards (D.Hall)."
    ) == {}


def test_parse_pbp_rush_with_and_without_td():
    assert parse_pbp_play_stats(
        "R.Stevenson up the middle to NE 11 for 3 yards (D.Lawrence)."
    ) == {"r.stevenson": {"rush_yds": 3, "carries": 1}}
    assert parse_pbp_play_stats(
        "R.Stevenson up the middle for 2 yards, TOUCHDOWN."
    ) == {"r.stevenson": {"rush_yds": 2, "carries": 1, "rush_td": 1}}


def test_parse_pbp_rush_credit_survives_leading_clause():
    # A pre-snap clause must not steal the carry from the actual ball carrier.
    assert parse_pbp_play_stats(
        "G.Van Roten reported in as eligible. D.Maye scrambles left end ran "
        "ob at SEA 21 for 6 yards (J.Jobe)."
    ) == {"d.maye": {"rush_yds": 6, "carries": 1}}


def test_parse_pbp_no_gain_reception_still_counts():
    sl = parse_pbp_play_stats("D.Maye pass complete to H.Henry for no gain.")
    assert sl["h.henry"] == {"rec": 1, "rec_yds": 0, "targets": 1}


def test_parse_pbp_field_goal_keeps_distance():
    # Distance is retained so the client can score fgm_40_49 / fgm_50p buckets.
    assert parse_pbp_play_stats(
        "J.Myers 45 yard field goal is GOOD, Center-C.Stoll."
    ) == {"j.myers": {"fgm": 1, "fg_yds": 45}}
    assert parse_pbp_play_stats(
        "C.Santos 52 yard field goal is GOOD."
    ) == {"c.santos": {"fgm": 1, "fg_yds": 52}}


def test_espn_plays_attach_real_stat_lines():
    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "plays": [
                            {
                                "id": "9",
                                "text": (
                                    "D.Lock pass short left to J.Smith-Njigba "
                                    "for 45 yards, TOUCHDOWN."
                                ),
                                "clock": {"displayValue": "11:28"},
                                "period": {"number": 4},
                                "start": {"down": 4, "distance": 1},
                                "type": {"text": "Passing Touchdown"},
                                "scoringPlay": True,
                            }
                        ]
                    }
                ]
            }
        }
    }
    plays = extract_espn_pbp_plays(
        payload,
        "20260909_NE@SEA",
        name_to_pid={"jaxon smith-njigba": "8155", "drew lock": "99"},
    )
    by_pid = {p["pid"]: p for p in plays}
    assert by_pid["8155"]["stat_line"] == {
        "rec": 1, "rec_yds": 45, "targets": 1, "rec_td": 1,
    }
    assert by_pid["99"]["stat_line"] == {
        "pass_yds": 45, "pass_cmp": 1, "pass_att": 1, "pass_td": 1,
    }


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
