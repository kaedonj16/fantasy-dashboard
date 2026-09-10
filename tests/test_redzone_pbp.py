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


def test_demo_play_text_td_shapes():
    assert demo_play_text("pass", 66, 1) == "Throws a 66-yard touchdown pass"
    assert demo_play_text("rec", 26, 1) == "Hauls in a 26-yard touchdown catch"
    assert demo_play_text("target") == "Targeted — pass incomplete"
    assert "field goal" in demo_play_text("fgm", dist=51)
