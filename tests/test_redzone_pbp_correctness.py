"""Regression tests for Redzone PBP correctness issues.

Tests cover the specific issues identified:
1. Receiver contributions being permanently lost
2. Incomplete target primary players
3. Sacks showing QB instead of DST
4. PBP chronology issues
5. No Play stat contamination
"""
from utils.redzone_pbp import (
    extract_pbp_plays,
    _normalize_name,
    _extract_first_initial_last,
    _resolve_player_name,
    _is_no_play,
    _extract_target_from_text,
    _opponent_team,
)


def test_normalize_name_strips_periods_and_apostrophes():
    """Test that name normalization handles abbreviated names."""
    assert _normalize_name("M.Hollins") == "m hollins"
    assert _normalize_name("Mack Hollins") == "mack hollins"
    assert _normalize_name("J.Smith-Njigba") == "j smith-njigba"
    assert _normalize_name("D'Andre Swift") == "dandre swift"


def test_extract_first_initial_last():
    """Test abbreviated name extraction."""
    assert _extract_first_initial_last("Mack Hollins") == "m hollins"
    assert _extract_first_initial_last("M.Hollins") == "m hollins"
    assert _extract_first_initial_last("Jaxon Smith-Njigba") == "j smith-njigba"
    assert _extract_first_initial_last("Hunter Henry") == "h henry"


def test_resolve_player_name_exact_match():
    """Test exact full name resolution."""
    name_to_pid = {"mack hollins": "1234", "hunter henry": "5678"}
    pid = _resolve_player_name("Mack Hollins", "NE", name_to_pid, {})
    assert pid == "1234"


def test_resolve_player_name_abbreviated():
    """Test abbreviated name resolution."""
    name_to_pid = {"mack hollins": "1234", "m hollins": "1234"}
    pid = _resolve_player_name("M.Hollins", "NE", name_to_pid, {})
    assert pid == "1234"


def test_resolve_player_name_no_ambiguous_guess():
    """Test that ambiguous surnames don't guess."""
    name_to_pid = {"john smith": "111", "jane smith": "222"}
    team_players = {"NE": [("111", "john smith"), ("222", "jane smith")]}
    # Should not match "Smith" ambiguously
    pid = _resolve_player_name("Smith", "NE", name_to_pid, team_players)
    assert pid == ""


def test_is_no_play_detection():
    """Test No Play detection."""
    assert _is_no_play("(Shotgun) D.Maye pass incomplete. PENALTY. No Play.")
    assert _is_no_play("Penalty nullified the play")
    assert not _is_no_play("D.Maye pass complete to M.Hollins for 19 yards")


def test_extract_target_from_text():
    """Test target extraction from play text."""
    assert _extract_target_from_text("pass short right to M.Hollins for 19 yards") == "M.Hollins"
    assert _extract_target_from_text("pass incomplete short left to J.Smith-Njigba") == "J.Smith-Njigba"
    assert _extract_target_from_text("pass short left to H.Henry for 10 yards") == "H.Henry"


def test_opponent_team_resolution():
    """Test opponent team determination."""
    game_context = {"home": "SEA", "away": "NE"}
    assert _opponent_team(game_context, "NE") == "SEA"
    assert _opponent_team(game_context, "SEA") == "NE"
    assert _opponent_team(game_context, "KC") == ""


def test_receiver_contribution_with_abbreviated_name():
    """Test that receiver with abbreviated name creates contribution."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "quarter": "4",
                "clock": "4:41",
                "play": "D.Maye pass short right to M.Hollins for 19 yards",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 19, "passCompletions": 1, "passAttempts": 1}
                    },
                    "WR": {
                        "longName": "M.Hollins",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": 19, "targets": 1}
                    }
                }
            }
        ]
    }
    name_to_pid = {
        "drake maye": "QB1",
        "mack hollins": "WR1",
        "m hollins": "WR1"  # Abbreviated form
    }
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # Should have both QB and WR contributions
    assert len(plays) == 2
    pids = {p["pid"] for p in plays}
    assert "QB1" in pids
    assert "WR1" in pids
    
    # Receiver should have reception
    wr_play = next(p for p in plays if p["pid"] == "WR1")
    assert wr_play["stat_line"]["rec"] == 1
    assert wr_play["stat_line"]["rec_yds"] == 19


def test_incomplete_target_fallback():
    """Test incomplete pass target extraction when playerStats missing."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "quarter": "4",
                "clock": "3:55",
                "play": "D.Maye pass incomplete short left to R.Doubs",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passAttempts": 1, "passCompletions": 0}
                    }
                }
            }
        ]
    }
    name_to_pid = {
        "drake maye": "QB1",
        "romeo doubs": "WR2",
        "r doubs": "WR2"
    }
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # Should have QB and WR target
    assert len(plays) == 2
    wr_play = next((p for p in plays if p["pid"] == "WR2"), None)
    assert wr_play is not None
    assert wr_play["stat_line"]["targets"] == 1
    assert wr_play["stat_line"]["rec"] == 0


def test_sack_creates_dst_contribution():
    """Test that sacks create DST contributions."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "quarter": "4",
                "clock": "5:22",
                "play": "(Shotgun) D.Maye sacked at NE 28 for -7 yards (D.Hall).",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {}
                    }
                }
            }
        ]
    }
    name_to_pid = {"drake maye": "QB1"}
    team_to_def_pid = {"SEA": "DEF_SEA"}
    game_context = {"home": "SEA", "away": "NE"}
    
    plays = extract_pbp_plays(
        box, "g1",
        name_to_pid=name_to_pid,
        team_to_def_pid=team_to_def_pid,
        game_context=game_context
    )
    
    # Should have DST contribution
    dst_play = next((p for p in plays if p["pid"] == "DEF_SEA"), None)
    assert dst_play is not None
    assert dst_play["stat_line"]["sacks"] == 1
    assert dst_play["team"] == "SEA"


def test_no_play_zeroes_fantasy_stats():
    """Test that No Play contributions have zero fantasy stats."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "quarter": "4",
                "clock": "3:03",
                "play": "(Shotgun) D.Maye pass incomplete. PENALTY on NE. No Play.",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passAttempts": 1, "passCompletions": 0}
                    }
                }
            }
        ]
    }
    name_to_pid = {"drake maye": "QB1"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # No Play should not create any contributions
    assert len(plays) == 0


def test_negative_yard_reception():
    """Test negative receiving yards are preserved."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "quarter": "4",
                "clock": "2:59",
                "play": "D.Maye pass short right to L.Larison for -2 yards",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": -2, "passCompletions": 1, "passAttempts": 1}
                    },
                    "RB": {
                        "longName": "L.Larison",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": -2, "targets": 1}
                    }
                }
            }
        ]
    }
    name_to_pid = {"drake maye": "QB1", "l larison": "RB1"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    rb_play = next(p for p in plays if p["pid"] == "RB1")
    assert rb_play["stat_line"]["rec_yds"] == -2


def test_seq_preserved_for_chronology():
    """Test that seq is preserved from provider."""
    box = {
        "allPlayByPlay": [
            {"playId": "1", "quarter": "4", "clock": "9:25", "play": "Play 1"},
            {"playId": "2", "quarter": "4", "clock": "8:11", "play": "Play 2"},
            {"playId": "3", "quarter": "4", "clock": "0:26", "play": "Play 3"},
        ]
    }
    plays = extract_pbp_plays(box, "g1")
    
    # Seq should be 0, 1, 2 (enumerate order)
    assert plays[0]["seq"] == 0
    assert plays[1]["seq"] == 1
    assert plays[2]["seq"] == 2


def test_completed_pass_fallback_from_text():
    """Test completed pass receiver extraction from text when playerStats missing."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "quarter": "4",
                "clock": "2:10",
                "play": "D.Maye pass short left to H.Henry for 10 yards",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 10, "passCompletions": 1, "passAttempts": 1}
                    }
                    # Missing receiver stats
                }
            }
        ]
    }
    name_to_pid = {
        "drake maye": "QB1",
        "hunter henry": "TE1",
        "h henry": "TE1"
    }
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # Should have QB and TE
    assert len(plays) == 2
    te_play = next((p for p in plays if p["pid"] == "TE1"), None)
    assert te_play is not None
    assert te_play["stat_line"]["rec"] == 1
    assert te_play["stat_line"]["rec_yds"] == 10


def test_multiple_plays_preserve_seq_order():
    """Test that multiple plays maintain seq-based chronology."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "p1",
                "quarter": "4",
                "clock": "5:00",
                "play": "Early play",
                "playerStats": {"QB": {"longName": "QB A", "teamAbv": "T1"}}
            },
            {
                "playId": "p2",
                "quarter": "4",
                "clock": "4:00",
                "play": "Middle play",
                "playerStats": {"QB": {"longName": "QB B", "teamAbv": "T2"}}
            },
            {
                "playId": "p3",
                "quarter": "4",
                "clock": "3:00",
                "play": "Late play",
                "playerStats": {"QB": {"longName": "QB C", "teamAbv": "T1"}}
            },
        ]
    }
    name_to_pid = {"qb a": "1", "qb b": "2", "qb c": "3"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # Seq should match input order (0, 1, 2)
    seqs = [p["seq"] for p in plays]
    assert seqs == [0, 1, 2]
