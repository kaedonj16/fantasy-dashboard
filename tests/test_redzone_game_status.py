"""Authoritative NFL game-status normalization for Redzone.

These lock the fix for the false-FINAL bug: a game is only ``final`` when the
provider game-level code says so. A blank/unknown code must never resolve to
final (bye, provider gap, or a stale player record).
"""

from utils.redzone_pbp import normalize_nfl_game_status, build_games_snapshot


def test_code_2_is_final():
    assert normalize_nfl_game_status("2", "Final") == "final"
    assert normalize_nfl_game_status("2", "") == "final"


def test_code_1_is_live_across_quarters():
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert normalize_nfl_game_status("1", f"In Progress {q}") == "live"


def test_code_1_with_half_text_is_halftime():
    assert normalize_nfl_game_status("1", "Halftime") == "halftime"


def test_code_0_is_pregame():
    assert normalize_nfl_game_status("0", "") == "pregame"
    assert normalize_nfl_game_status("0", "Scheduled") == "pregame"


def test_blank_code_never_final_even_with_stale_final_text_absent():
    # No code and no text -> unknown, never final.
    assert normalize_nfl_game_status("", "") == "unknown"
    assert normalize_nfl_game_status(None, None) == "unknown"


def test_code_wins_over_stale_text():
    # Player record text says Final but the game-level code says in progress:
    # game-level code (1) wins -> live, not final.
    assert normalize_nfl_game_status("1", "Final") == "live"


def test_delayed_preserved_from_text_when_no_code():
    assert normalize_nfl_game_status("", "Postponed") == "delayed"
    assert normalize_nfl_game_status("", "Suspended") == "delayed"


def test_final_from_text_only_when_no_code():
    assert normalize_nfl_game_status("", "Final/OT") == "final"


def test_snapshot_carries_normalized_status_per_game():
    player_info = {
        "p_live": {
            "game_id": "G_LIVE", "away": "NE", "home": "SEA",
            "away_pts": "20", "home_pts": "24",
            "game_code": "1", "game_status": "In Progress",
            "game_clock": "2:18", "game_quarter": "4",
        },
        "p_final": {
            "game_id": "G_FINAL", "away": "SF", "home": "LAR",
            "away_pts": "27", "home_pts": "7",
            "game_code": "2", "game_status": "Final",
        },
        "p_pre": {
            "game_id": "G_PRE", "away": "DAL", "home": "PHI",
            "away_pts": "", "home_pts": "",
            "game_code": "0", "game_status": "Scheduled",
        },
    }
    games = build_games_snapshot(player_info, {})
    assert games["G_LIVE"]["status"] == "live"
    assert games["G_FINAL"]["status"] == "final"
    assert games["G_PRE"]["status"] == "pregame"


def test_two_simultaneous_games_keep_independent_status():
    player_info = {
        "a": {"game_id": "GA", "away": "NE", "home": "SEA",
              "game_code": "1", "game_status": "In Progress"},
        "b": {"game_id": "GB", "away": "SF", "home": "LAR",
              "game_code": "2", "game_status": "Final"},
    }
    games = build_games_snapshot(player_info, {})
    assert games["GA"]["status"] == "live"
    assert games["GB"]["status"] == "final"
