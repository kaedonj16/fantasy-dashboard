"""Unit tests for utils.redzone_pbp (Tank01 play-by-play normalization)."""
from utils.redzone_pbp import demo_play_text, extract_pbp_plays


def test_extract_pbp_maps_players_and_fields():
    box = {
        "allPlayByPlay": [
            {
                "playId": "42",
                "quarter": "2",
                "clock": "8:42",
                "down": "1",
                "distance": "10",
                "yardLine": "KC 35",
                "play": "Patrick Mahomes pass complete to Travis Kelce for 17 yards",
                "playerStats": {
                    "a": {
                        "longName": "Patrick Mahomes",
                        "Passing": {"passYds": "17", "passCompletions": "1", "passAttempts": "1"},
                    },
                    "b": {
                        "longName": "Travis Kelce",
                        "Receiving": {"receptions": "1", "recYds": "17", "targets": "1"},
                    },
                },
            }
        ]
    }
    plays = extract_pbp_plays(
        box,
        "20240905_KC@BAL",
        name_to_pid={"patrick mahomes": "4046", "travis kelce": "2133"},
    )
    assert len(plays) == 2
    by_pid = {p["pid"]: p for p in plays}
    assert by_pid["4046"]["play_text"].startswith("Patrick Mahomes")
    assert by_pid["4046"]["stat_line"]["pass_yds"] == 17.0
    assert by_pid["2133"]["down"] == "1"
    assert by_pid["2133"]["distance"] == "10"
    assert by_pid["2133"]["clock"] == "8:42"


def test_extract_pbp_def_via_team_stats():
    box = {
        "allPlayByPlay": [
            {
                "playId": "99",
                "play": "Chiefs sacked Josh Allen",
                "teamStats": {
                    "home": {
                        "teamAbv": "KC",
                        "Defense": {"sacks": "1"},
                    }
                },
            }
        ]
    }
    plays = extract_pbp_plays(box, "g1", team_to_def_pid={"KC": "KC"})
    assert len(plays) == 1
    assert plays[0]["pid"] == "KC"
    assert plays[0]["stat_line"]["sacks"] == 1.0


def test_extract_pbp_empty_box_is_safe():
    assert extract_pbp_plays({}, "g") == []
    assert extract_pbp_plays(None, "g") == []  # type: ignore[arg-type]


def test_extract_keeps_named_player_when_stats_zero():
    """Tank01 often ships play text + empty deltas — keep the booth line."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "7",
                "play": "Patrick Mahomes pass incomplete intended for Travis Kelce",
                "playerStats": {
                    "a": {"longName": "Patrick Mahomes", "Passing": {}},
                    "b": {"longName": "Travis Kelce", "Receiving": {"targets": "0"}},
                },
            }
        ]
    }
    plays = extract_pbp_plays(
        box, "g",
        name_to_pid={"patrick mahomes": "4046", "travis kelce": "2133"},
    )
    assert len(plays) == 2
    assert all(p["play_text"].startswith("Patrick Mahomes") for p in plays)
    assert {p["pid"] for p in plays} == {"4046", "2133"}


def test_extract_flat_player_stats_and_nested_body():
    box = {
        "body": {
            "allPlayByPlay": [
                {
                    "playId": "1",
                    "play": "Saquon Barkley rush for 12 yards",
                    "playerStats": {
                        "x": {
                            "longName": "Saquon Barkley",
                            "rushYds": "12",
                            "carries": "1",
                        }
                    },
                }
            ]
        }
    }
    plays = extract_pbp_plays(
        box, "g", name_to_pid={"saquon barkley": "4866"}
    )
    assert len(plays) == 1
    assert plays[0]["pid"] == "4866"
    assert plays[0]["stat_line"]["rush_yds"] == 12.0
    assert plays[0]["stat_line"]["carries"] == 1.0


def test_extract_kick_keeps_a_non_overlapping_distance_bucket():
    """Tank01 made-FG rows must remain scoreable after cumulative rebuilds."""
    box = {"allPlayByPlay": [{
        "playId": "fg-1", "play": "Jake Kicker 56 yard field goal",
        "playerStats": {"k": {
            "longName": "Jake Kicker", "Kicking": {"fgMade": 1, "fgLng": 56},
        }},
    }]}
    plays = extract_pbp_plays(box, "g", name_to_pid={"jake kicker": "k1"})
    assert plays[0]["stat_line"]["fgm"] == 1.0
    assert plays[0]["stat_line"]["fgm_50_59"] == 1.0


def test_extract_narrative_only_when_no_player_rows():
    box = {
        "allPlayByPlay": [
            {"playId": "3", "play": "Timeout at the two-minute warning"}
        ]
    }
    plays = extract_pbp_plays(box, "g")
    assert len(plays) == 1
    assert plays[0]["play_text"].startswith("Timeout")
    assert plays[0]["pid"] == ""


def test_demo_play_text_td_shapes():
    assert demo_play_text("pass", 66, 1) == "Throws a 66-yard touchdown pass"
    assert demo_play_text("rec", 26, 1) == "Hauls in a 26-yard touchdown catch"
    assert demo_play_text("target") == "Targeted — pass incomplete"
    assert "field goal" in demo_play_text("fgm", dist=51)


def test_game_situation_from_latest_field_play():
    from utils.redzone_pbp import game_situation_from_plays

    plays = [
        {"seq": 0, "team": "KC", "down": "1", "distance": "10", "yard_line": "KC 25"},
        {"seq": 1, "team": "KC", "down": "2", "distance": "7", "yard_line": "KC 28"},
        {"seq": 2, "team": "BAL", "down": "", "distance": "", "yard_line": ""},  # no field context
    ]
    sit = game_situation_from_plays(plays)
    assert sit["possession"] == "KC"
    assert sit["down"] == "2"
    assert sit["distance"] == "7"
    assert sit["yard_line"] == "KC 28"


def test_field_position_reliable_accepts_snap_yard_line_and_guards_turnovers():
    from utils.redzone_pbp import game_situation_from_plays

    # Snap yard_line only (no end_yard_line) is still a usable current spot --
    # this is what most live feeds send, so the indicator must not be suppressed.
    snap_only = game_situation_from_plays([
        {"seq": 0, "team": "NYG", "down": "2", "distance": "11", "yard_line": "DAL 29"},
    ])
    assert snap_only["yard_line"] == "DAL 29"
    assert snap_only["field_position_reliable"] is True

    # end_yard_line is preferred when present.
    with_end = game_situation_from_plays([
        {"seq": 0, "team": "KC", "down": "1", "distance": "10",
         "yard_line": "KC 25", "end_yard_line": "KC 31"},
    ])
    assert with_end["yard_line"] == "KC 31"
    assert with_end["field_position_reliable"] is True

    # In-progress turnovers stay unreliable -- possession/spot can momentarily lie.
    turnover = game_situation_from_plays([
        {"seq": 0, "team": "DAL", "down": "1", "distance": "10", "yard_line": "DAL 28",
         "play_text": "J.Dart sacked, FUMBLES, recovered by NYG at DAL 34"},
    ])
    assert turnover["field_position_reliable"] is False

    # No usable yard line at all -> not reliable.
    team_only = game_situation_from_plays([{"seq": 0, "team": "CHI"}])
    assert team_only["field_position_reliable"] is False


def test_game_situation_falls_back_to_team_only():
    from utils.redzone_pbp import game_situation_from_plays

    sit = game_situation_from_plays([
        {"seq": 0, "team": "DET"},
        {"seq": 1, "team": "CHI"},
    ])
    assert sit["possession"] == "CHI"
    assert sit["down"] == ""


def test_build_games_snapshot_merges_score_and_pbp():
    from utils.redzone_pbp import build_games_snapshot

    player_info = {
        "1": {
            "game_id": "g1", "away": "BAL", "home": "HOU",
            "away_pts": "21", "home_pts": "14",
            "game_code": "1", "game_clock": "4:32", "game_quarter": "Q3",
            "game_status": "In Progress",
        },
        "2": {
            "game_id": "g1", "away": "BAL", "home": "HOU",
            "away_pts": "21", "home_pts": "14",
            "game_code": "1", "game_clock": "4:32", "game_quarter": "Q3",
        },
    }
    pbp = {
        "g1": [
            {"seq": 0, "team": "HOU", "down": "1", "distance": "10", "yard_line": "HOU 20"},
            {"seq": 1, "team": "BAL", "down": "2", "distance": "7", "yard_line": "HOU 42"},
        ]
    }
    games = build_games_snapshot(player_info, pbp)
    assert "g1" in games
    g = games["g1"]
    assert g["away"] == "BAL" and g["home"] == "HOU"
    assert g["away_pts"] == "21"
    assert g["game_clock"] == "4:32"
    assert g["possession"] == "BAL"
    assert g["down"] == "2"
    assert g["distance"] == "7"
    assert g["yard_line"] == "HOU 42"
