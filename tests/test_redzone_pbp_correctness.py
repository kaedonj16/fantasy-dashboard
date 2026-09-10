"""Regression tests for Redzone PBP correctness issues.

Tests cover the specific issues identified:
1. Receiver contributions being permanently lost
2. Incomplete target primary players
3. Sacks showing QB instead of DST
4. PBP chronology issues
5. No Play stat contamination
6. Text fallback when receiver row exists but pid is empty
"""
from utils.redzone_pbp import (
    extract_pbp_plays,
    _normalize_name,
    _extract_first_initial_last,
    _resolve_player_name,
    _is_no_play,
    _detect_play_state,
    _extract_target_from_text,
    _opponent_team,
    PLAY_STATE_VALID,
    PLAY_STATE_NO_PLAY,
    PLAY_STATE_NULLIFIED,
    PLAY_STATE_OVERTURNED,
    PLAY_STATE_CORRECTED,
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


# ======================================================================
# PLAY REVISION / CALLED-BACK / OVERTURNED TESTS
# ======================================================================

def test_detect_play_state_valid():
    """Test play state detection for valid plays."""
    play = {}
    assert _detect_play_state(play, "D.Maye pass complete to M.Hollins for 19 yards") == PLAY_STATE_VALID
    assert _detect_play_state(play, "J.Taylor rush for 5 yards") == PLAY_STATE_VALID


def test_detect_play_state_no_play_pre_snap():
    """Test No Play detection for pre-snap penalties."""
    play = {}
    assert _detect_play_state(play, "False start. No Play.") == PLAY_STATE_NO_PLAY
    assert _detect_play_state(play, "Delay of game. No Play.") == PLAY_STATE_NO_PLAY
    assert _detect_play_state(play, "Encroachment. No Play.") == PLAY_STATE_NO_PLAY
    assert _detect_play_state(play, "Offsides. No Play.") == PLAY_STATE_NO_PLAY


def test_detect_play_state_nullified():
    """Test nullified play detection (called back by penalty)."""
    play = {}
    text = "D.Maye pass complete to M.Hollins for 19 yards. PENALTY on NE-Offensive Holding. No Play."
    assert _detect_play_state(play, text) == PLAY_STATE_NULLIFIED
    
    text2 = "Play nullified by penalty"
    assert _detect_play_state(play, text2) == PLAY_STATE_NULLIFIED


def test_detect_play_state_overturned():
    """Test overturned play detection (replay review)."""
    play = {}
    assert _detect_play_state(play, "Interception overturned") == PLAY_STATE_OVERTURNED
    assert _detect_play_state(play, "Ruling overturned on replay") == PLAY_STATE_OVERTURNED


def test_detect_play_state_penalty_play_counts():
    """Test that penalties where play counts are marked valid."""
    play = {}
    # Declined penalty
    assert _detect_play_state(play, "Pass complete. PENALTY declined.") == PLAY_STATE_VALID
    # After-the-play penalty
    assert _detect_play_state(play, "TD run. Unnecessary roughness after.") == PLAY_STATE_VALID
    # Defensive penalty where play stands
    assert _detect_play_state(play, "Pass complete. Defensive holding.") == PLAY_STATE_VALID


def test_detect_play_state_structured_fields():
    """Test play state detection from structured provider fields."""
    play = {"playStatus": "no_play"}
    assert _detect_play_state(play, "Some text") == PLAY_STATE_NO_PLAY
    
    play = {"playResult": "nullified"}
    assert _detect_play_state(play, "Some text") == PLAY_STATE_NULLIFIED
    
    play = {"playStatus": "overturned"}
    assert _detect_play_state(play, "Some text") == PLAY_STATE_OVERTURNED


def test_valid_completion_then_nullified():
    """Test valid completion that gets called back on later poll."""
    # Poll 1: Valid completion
    box1 = {
        "allPlayByPlay": [
            {
                "playId": "play123",
                "quarter": "4",
                "clock": "2:00",
                "play": "D.Maye pass complete to M.Hollins for 15 yards",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 15, "passCompletions": 1, "passAttempts": 1}
                    },
                    "WR": {
                        "longName": "Mack Hollins",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": 15, "targets": 1}
                    }
                }
            }
        ]
    }
    name_to_pid = {"drake maye": "QB1", "mack hollins": "WR1"}
    plays1 = extract_pbp_plays(box1, "g1", name_to_pid=name_to_pid)
    
    # Should have valid contributions
    assert len(plays1) == 2
    qb_play = next(p for p in plays1 if p["pid"] == "QB1")
    wr_play = next(p for p in plays1 if p["pid"] == "WR1")
    assert qb_play["stat_line"]["pass_yds"] == 15
    assert wr_play["stat_line"]["rec"] == 1
    assert wr_play["stat_line"]["rec_yds"] == 15
    assert qb_play["play_state"] == PLAY_STATE_VALID
    assert wr_play["play_state"] == PLAY_STATE_VALID
    
    # Poll 2: Same play becomes No Play
    box2 = {
        "allPlayByPlay": [
            {
                "playId": "play123",  # Same play_id
                "quarter": "4",
                "clock": "2:00",
                "play": "D.Maye pass complete to M.Hollins. PENALTY on NE-Offensive Holding. No Play.",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 15, "passCompletions": 1, "passAttempts": 1}
                    },
                    "WR": {
                        "longName": "Mack Hollins",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": 15, "targets": 1}
                    }
                }
            }
        ]
    }
    plays2 = extract_pbp_plays(box2, "g1", name_to_pid=name_to_pid)
    
    # Should have NO contributions (nullified)
    assert len(plays2) == 0


def test_valid_td_then_nullified():
    """Test TD that gets called back."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "td_play",
                "quarter": "4",
                "clock": "1:00",
                "play": "D.Maye pass to M.Hollins for 25 yards. TOUCHDOWN. PENALTY nullified.",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 25, "passTD": 1}
                    },
                    "WR": {
                        "longName": "Mack Hollins",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": 25, "recTD": 1}
                    }
                }
            }
        ]
    }
    name_to_pid = {"drake maye": "QB1", "mack hollins": "WR1"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # Nullified - no contributions
    assert len(plays) == 0


def test_interception_overturned():
    """Test interception overturned to incomplete."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "int_play",
                "quarter": "3",
                "clock": "5:00",
                "play": "D.Maye pass intercepted. Ruling overturned - incomplete.",
                "playStatus": "overturned",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"interceptions": 1}
                    }
                }
            }
        ]
    }
    name_to_pid = {"drake maye": "QB1"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # Overturned - no contributions
    assert len(plays) == 0


def test_sack_nullified_by_penalty():
    """Test sack erased by penalty."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "sack_play",
                "quarter": "2",
                "clock": "10:00",
                "play": "D.Maye sacked. PENALTY - Defensive holding. No Play.",
                "playerStats": {
                    "QB": {"longName": "Drake Maye", "teamAbv": "NE"}
                },
                "teamStats": {
                    "Defense": {"teamAbv": "SEA", "sacks": 1}
                }
            }
        ]
    }
    name_to_pid = {"drake maye": "QB1"}
    team_to_def_pid = {"SEA": "DEF_SEA"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid, team_to_def_pid=team_to_def_pid)
    
    # No Play - no contributions
    assert len(plays) == 0


def test_yardage_correction():
    """Test stat correction (15 yards corrected to 12)."""
    # This would be detected by comparing stat_line changes in frontend
    # Backend just extracts what provider sends
    box = {
        "allPlayByPlay": [
            {
                "playId": "catch_play",
                "quarter": "4",
                "clock": "3:00",
                "play": "D.Maye pass to M.Hollins for 12 yards",  # Corrected from 15
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 12, "passCompletions": 1}
                    },
                    "WR": {
                        "longName": "Mack Hollins",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": 12, "targets": 1}
                    }
                }
            }
        ]
    }
    name_to_pid = {"drake maye": "QB1", "mack hollins": "WR1"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # Should have corrected yardage
    wr_play = next(p for p in plays if p["pid"] == "WR1")
    assert wr_play["stat_line"]["rec_yds"] == 12


def test_false_start_no_fantasy_impact():
    """Test pre-snap penalty has no fantasy impact."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "fs_play",
                "quarter": "1",
                "clock": "12:00",
                "play": "False start on NE. No Play.",
                "playerStats": {}
            }
        ]
    }
    plays = extract_pbp_plays(box, "g1")
    
    # No contributions
    assert len(plays) == 0


def test_penalty_where_stats_count():
    """Test penalty where play still counts (defensive holding, play stands)."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "dh_play",
                "quarter": "3",
                "clock": "8:00",
                "play": "D.Maye pass complete to M.Hollins for 10 yards. Defensive holding.",
                "playerStats": {
                    "QB": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 10, "passCompletions": 1}
                    },
                    "WR": {
                        "longName": "Mack Hollins",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": 10, "targets": 1}
                    }
                }
            }
        ]
    }
    name_to_pid = {"drake maye": "QB1", "mack hollins": "WR1"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # Play counts - should have contributions
    assert len(plays) == 2
    wr_play = next(p for p in plays if p["pid"] == "WR1")
    assert wr_play["stat_line"]["rec"] == 1
    assert wr_play["play_state"] == PLAY_STATE_VALID


def test_declined_penalty_play_stands():
    """Test declined penalty where play stands."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "dec_play",
                "quarter": "2",
                "clock": "6:00",
                "play": "J.Taylor rush for 8 yards. PENALTY declined.",
                "playerStats": {
                    "RB": {
                        "longName": "Jonathan Taylor",
                        "teamAbv": "IND",
                        "Rushing": {"carries": 1, "rushYards": 8}
                    }
                }
            }
        ]
    }
    name_to_pid = {"jonathan taylor": "RB1"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    # Play counts
    assert len(plays) == 1
    assert plays[0]["stat_line"]["carries"] == 1
    assert plays[0]["play_state"] == PLAY_STATE_VALID


def test_play_state_field_added():
    """Test that play_state field is added to all plays."""
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "quarter": "4",
                "clock": "2:00",
                "play": "Valid play",
                "playerStats": {"QB": {"longName": "QB A", "teamAbv": "T1"}}
            }
        ]
    }
    name_to_pid = {"qb a": "1"}
    plays = extract_pbp_plays(box, "g1", name_to_pid=name_to_pid)
    
    assert len(plays) == 1
    assert "play_state" in plays[0]
    assert plays[0]["play_state"] == PLAY_STATE_VALID


def test_abbreviated_receiver_names_all_formats():
    """Test that all abbreviated name formats resolve correctly.
    
    This reproduces the exact live bug where M.Hollins, H.Henry, D.Douglas
    were not resolving, causing only QB to show as primary.
    """
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "quarter": "4",
                "clock": "4:41",
                "play": "D.Maye pass short right to M.Hollins for 12 yards",
                "playerStats": {
                    "QB": {
                        "longName": "D.Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 12, "passCompletions": 1}
                    },
                    "WR1": {
                        "longName": "M.Hollins",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": 12}
                    }
                }
            },
            {
                "playId": "2",
                "quarter": "4",
                "clock": "4:30",
                "play": "D.Maye pass short left to D.Douglas for 1 yard",
                "playerStats": {
                    "QB": {
                        "longName": "D.Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 1, "passCompletions": 1}
                    },
                    "WR2": {
                        "longName": "D.Douglas",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": 1}
                    }
                }
            },
            {
                "playId": "3",
                "quarter": "4",
                "clock": "4:15",
                "play": "D.Maye pass short middle to H.Henry for 10 yards",
                "playerStats": {
                    "QB": {
                        "longName": "D.Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYards": 10, "passCompletions": 1}
                    },
                    "TE": {
                        "longName": "H.Henry",
                        "teamAbv": "NE",
                        "Receiving": {"receptions": 1, "recYards": 10}
                    }
                }
            }
        ]
    }
    
    # Build name_to_pid with explicit aliases like app.py does
    name_to_pid = {
        "drake maye": "QB1",
        "d maye": "QB1",
        "mack hollins": "WR1",
        "m hollins": "WR1",
        "demario douglas": "WR2",
        "d douglas": "WR2",
        "hunter henry": "TE1",
        "h henry": "TE1",
    }
    
    player_meta_by_pid = {
        "QB1": {"name": "Drake Maye", "team": "NE"},
        "WR1": {"name": "Mack Hollins", "team": "NE"},
        "WR2": {"name": "Demario Douglas", "team": "NE"},
        "TE1": {"name": "Hunter Henry", "team": "NE"},
    }
    
    plays = extract_pbp_plays(
        box, "g1",
        name_to_pid=name_to_pid,
        player_meta_by_pid=player_meta_by_pid
    )
    
    # Should have 6 contributions total (QB + receiver for each play)
    assert len(plays) == 6, f"Expected 6 plays, got {len(plays)}"
    
    # Play 1: M.Hollins
    play1_contribs = [p for p in plays if p["play_id"] == "1"]
    assert len(play1_contribs) == 2
    pids1 = {p["pid"] for p in play1_contribs}
    assert "QB1" in pids1, "QB1 missing from play 1"
    assert "WR1" in pids1, "WR1 (M.Hollins) missing from play 1"
    wr1 = next(p for p in play1_contribs if p["pid"] == "WR1")
    assert wr1["stat_line"]["rec"] == 1
    assert wr1["stat_line"]["rec_yds"] == 12
    
    # Play 2: D.Douglas
    play2_contribs = [p for p in plays if p["play_id"] == "2"]
    assert len(play2_contribs) == 2
    pids2 = {p["pid"] for p in play2_contribs}
    assert "QB1" in pids2, "QB1 missing from play 2"
    assert "WR2" in pids2, "WR2 (D.Douglas) missing from play 2"
    wr2 = next(p for p in play2_contribs if p["pid"] == "WR2")
    assert wr2["stat_line"]["rec"] == 1
    assert wr2["stat_line"]["rec_yds"] == 1
    
    # Play 3: H.Henry
    play3_contribs = [p for p in plays if p["play_id"] == "3"]
    assert len(play3_contribs) == 2
    pids3 = {p["pid"] for p in play3_contribs}
    assert "QB1" in pids3, "QB1 missing from play 3"
    assert "TE1" in pids3, "TE1 (H.Henry) missing from play 3"
    te1 = next(p for p in play3_contribs if p["pid"] == "TE1")
    assert te1["stat_line"]["rec"] == 1
    assert te1["stat_line"]["rec_yds"] == 10


def test_receiver_text_fallback_when_pid_empty():
    """Test that text fallback runs when receiver row exists but pid is empty.
    
    This is the core bug: if playerStats has a receiver row but we can't resolve
    the pid, we should still try text extraction to create a resolved contribution.
    """
    box = {
        "allPlayByPlay": [
            {
                "playId": "1",
                "play": "D.Maye pass short right to M.Hollins for 12 yards",
                "playerStats": {
                    "QB1": {"longName": "Drake Maye", "teamAbv": "NE", "Passing": {"passYds": 12}},
                    # Receiver row exists but name doesn't resolve to pid
                    "WR_UNKNOWN": {"longName": "M.Hollins", "teamAbv": "NE", "Receiving": {"receptions": 1, "recYds": 12}}
                }
            }
        ]
    }
    
    # Only QB is in name_to_pid, receiver is NOT
    name_to_pid = {"drake maye": "QB1"}
    
    # But receiver IS in player_meta_by_pid for team resolution
    player_meta_by_pid = {
        "QB1": {"name": "Drake Maye", "team": "NE"},
        "WR1": {"name": "Mack Hollins", "team": "NE"}
    }
    
    plays = extract_pbp_plays(
        box, "g1",
        name_to_pid=name_to_pid,
        player_meta_by_pid=player_meta_by_pid
    )
    
    # Should have QB + receiver (from text fallback)
    assert len(plays) == 2, f"Expected 2 contributions, got {len(plays)}"
    
    pids = {p["pid"] for p in plays}
    assert "QB1" in pids
    assert "WR1" in pids, "Text fallback should have resolved M.Hollins to WR1"
    
    wr = next(p for p in plays if p["pid"] == "WR1")
    assert wr["stat_line"]["rec"] == 1
    assert wr["stat_line"]["rec_yds"] == 12
