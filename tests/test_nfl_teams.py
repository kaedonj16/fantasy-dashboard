"""Unit tests for utils.nfl_teams.get_team_full_name.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.nfl_teams import TEAM_FULL_NAMES, get_team_full_name


def test_known_abbreviation():
    assert get_team_full_name("KC") == "Kansas City Chiefs"
    assert get_team_full_name("SF") == "San Francisco 49ers"


def test_case_insensitive():
    assert get_team_full_name("kc") == "Kansas City Chiefs"
    assert get_team_full_name("Sf") == "San Francisco 49ers"


def test_both_washington_spellings():
    assert get_team_full_name("WAS") == "Washington Commanders"
    assert get_team_full_name("WSH") == "Washington Commanders"


def test_unknown_passes_through():
    assert get_team_full_name("XXX") == "XXX"
    assert get_team_full_name("FA") == "FA"


def test_all_32_teams_present():
    # 32 franchises + the extra WSH alias = 33 entries.
    assert len(TEAM_FULL_NAMES) == 33
    assert len(set(TEAM_FULL_NAMES.values())) == 32
