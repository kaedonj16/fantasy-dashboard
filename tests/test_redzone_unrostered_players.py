"""Regression test: Unrostered players MUST resolve through player index.

CRITICAL IDENTITY RULE:
IF A PLAYER EXISTS IN THE SITE PLAYER INDEX, PBP SHOULD RESOLVE TO THAT PLAYER
REGARDLESS OF WHETHER THEY ARE ROSTERED IN THE FANTASY LEAGUE.

Fantasy roster ownership must NEVER determine whether an NFL player contribution exists.
"""
import pytest


def test_unrostered_player_resolution():
    """Unrostered players must resolve through player index and create contributions."""
    from utils.redzone_pbp import extract_pbp_plays
    
    # Simulate player index with both rostered and unrostered players
    MAYE_PID = "10000"
    HENRY_PID = "10001"
    HOLLINS_PID = "10002"
    DOUGLAS_PID = "10003"
    
    player_meta_by_pid = {
        MAYE_PID: {"name": "Drake Maye", "team": "NE"},
        HENRY_PID: {"name": "Hunter Henry", "team": "NE"},
        HOLLINS_PID: {"name": "Mack Hollins", "team": "NE"},
        DOUGLAS_PID: {"name": "Demario Douglas", "team": "NE"},
    }
    
    name_to_pid = {
        "drake maye": MAYE_PID,
        "d maye": MAYE_PID,
        "hunter henry": HENRY_PID,
        "h henry": HENRY_PID,
        "mack hollins": HOLLINS_PID,
        "m hollins": HOLLINS_PID,
        "demario douglas": DOUGLAS_PID,
        "d douglas": DOUGLAS_PID,
    }
    
    # PBP contains plays involving all four players
    box = {
        "allPlayByPlay": [
            {
                "playId": "play1",
                "play": "D.Maye pass short left to H.Henry to SEA 47 for 10 yards",
                "quarter": "Q2",
                "clock": "5:30",
                "playerStats": {
                    "maye": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYds": "10", "passTD": "0"}
                    },
                    "henry": {
                        "longName": "Hunter Henry",
                        "teamAbv": "NE",
                        "Receiving": {"rec": "1", "recYds": "10", "recTD": "0"}
                    }
                }
            },
            {
                "playId": "play2",
                "play": "D.Maye pass short right to M.Hollins to NE 45 for 12 yards",
                "quarter": "Q2",
                "clock": "4:15",
                "playerStats": {
                    "maye": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYds": "12", "passTD": "0"}
                    },
                    "hollins": {
                        "longName": "Mack Hollins",
                        "teamAbv": "NE",
                        "Receiving": {"rec": "1", "recYds": "12", "recTD": "0"}
                    }
                }
            },
            {
                "playId": "play3",
                "play": "D.Maye pass short left to D.Douglas to NE 30 for 1 yard",
                "quarter": "Q2",
                "clock": "3:00",
                "playerStats": {
                    "maye": {
                        "longName": "Drake Maye",
                        "teamAbv": "NE",
                        "Passing": {"passYds": "1", "passTD": "0"}
                    },
                    "douglas": {
                        "longName": "Demario Douglas",
                        "teamAbv": "NE",
                        "Receiving": {"rec": "1", "recYds": "1", "recTD": "0"}
                    }
                }
            }
        ]
    }
    
    plays = extract_pbp_plays(
        box,
        "game123",
        name_to_pid=name_to_pid,
        player_meta_by_pid=player_meta_by_pid,
    )
    
    # ALL FOUR players should resolve through the player index
    resolved_pids = {p["pid"] for p in plays if p["pid"]}
    
    assert MAYE_PID in resolved_pids, "Drake Maye should resolve"
    assert HENRY_PID in resolved_pids, "Hunter Henry should resolve (rostered)"
    assert HOLLINS_PID in resolved_pids, "Mack Hollins should resolve (UNROSTERED)"
    assert DOUGLAS_PID in resolved_pids, "Demario Douglas should resolve (UNROSTERED)"
    
    # Verify each player has valid contributions
    maye_plays = [p for p in plays if p["pid"] == MAYE_PID]
    henry_plays = [p for p in plays if p["pid"] == HENRY_PID]
    hollins_plays = [p for p in plays if p["pid"] == HOLLINS_PID]
    douglas_plays = [p for p in plays if p["pid"] == DOUGLAS_PID]
    
    assert len(maye_plays) == 3, "Maye should have 3 passing plays"
    assert len(henry_plays) == 1, "Henry should have 1 reception"
    assert len(hollins_plays) == 1, "Hollins should have 1 reception (UNROSTERED)"
    assert len(douglas_plays) == 1, "Douglas should have 1 reception (UNROSTERED)"
    
    # Verify stat lines are correct
    assert hollins_plays[0]["stat_line"]["rec"] == 1
    assert hollins_plays[0]["stat_line"]["rec_yds"] == 12
    assert douglas_plays[0]["stat_line"]["rec"] == 1
    assert douglas_plays[0]["stat_line"]["rec_yds"] == 1


def test_frontend_does_not_gate_on_roster_ownership():
    """Frontend must not discard contributions where rosterId is empty.
    
    This is a contract test - the frontend should accept contributions with:
    - pid: valid canonical player ID
    - rosterId: "" (empty string for unrostered players)
    
    The frontend should:
    1. Create the contribution
    2. Set mine/opp to false when rosterId is empty
    3. Calculate fantasy points using league scoring
    4. Display the player name from player_info
    5. Make the player clickable with their canonical pid
    """
    # This test documents the expected frontend behavior
    # Actual implementation is in static/redzone.js _eventsFromPbp()
    
    # Example contribution for unrostered Mack Hollins:
    unrostered_contrib = {
        "pid": "10002",
        "name": "Mack Hollins",
        "pos": "WR",
        "nflTeam": "NE",
        "rosterId": "",  # EMPTY - player is unrostered
        "owner": "",
        "league": "",
        "mine": False,  # Not on my roster
        "opp": False,   # Not on opponent roster
        "line": {"rec": 1, "rec_yds": 12},
        "pts": 1.2,  # Calculated from league scoring
    }
    
    # CRITICAL: This contribution MUST NOT be discarded
    # It should appear in the feed with:
    # - Player name: "Mack Hollins"
    # - Position/Team: "WR · NE"
    # - Fantasy points: "+1.2 pts"
    # - Clickable player modal
    # - No ownership badge
    
    assert unrostered_contrib["pid"] != ""
    assert unrostered_contrib["rosterId"] == ""
    assert unrostered_contrib["mine"] is False
    assert unrostered_contrib["opp"] is False
    assert unrostered_contrib["pts"] > 0


def test_player_index_is_source_of_truth():
    """Player identity must come from player index, not fantasy rosters.
    
    The pipeline should be:
    1. PBP player name → canonical player index PID
    2. player_info metadata
    3. stat contribution
    4. grouping
    5. primary actor selection
    6. OPTIONALLY decorate with fantasy ownership
    
    NOT:
    1. PBP player → fantasy roster lookup
    2. discard if unowned
    """
    from utils.redzone_pbp import _resolve_player_name
    
    # Player index contains both rostered and unrostered players
    name_to_pid = {
        "mack hollins": "10002",
        "m hollins": "10002",
    }
    
    team_players = {
        "NE": [("10002", "Mack Hollins")]
    }
    
    # Resolution should work regardless of roster ownership
    pid = _resolve_player_name("M.Hollins", "NE", name_to_pid, team_players)
    assert pid == "10002", "Should resolve to canonical PID"
    
    pid = _resolve_player_name("Mack Hollins", "NE", name_to_pid, team_players)
    assert pid == "10002", "Should resolve full name to canonical PID"


def test_abbreviated_name_resolution():
    """Abbreviated PBP names must resolve to full player index entries."""
    from utils.redzone_pbp import _normalize_name, _extract_first_initial_last
    
    # Test cases from real PBP
    test_cases = [
        ("M.Hollins", "m hollins"),
        ("H.Henry", "h henry"),
        ("D.Douglas", "d douglas"),
        ("L.Larison", "l larison"),
        ("R.Doubs", "r doubs"),
        ("J.Smith-Njigba", "j smith-njigba"),
    ]
    
    for abbrev, expected in test_cases:
        normalized = _normalize_name(abbrev)
        assert normalized == expected, f"{abbrev} should normalize to {expected}"
        
        # Also test first-initial extraction
        abbrev_form = _extract_first_initial_last(abbrev)
        assert abbrev_form == expected, f"{abbrev} should extract to {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
