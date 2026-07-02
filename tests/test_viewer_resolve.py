"""Unit tests for utils.viewer_resolve.

Pure logic — no app / DB / session import — so these run anywhere pytest does.
"""
from utils.viewer_resolve import normalize_sleeper_username, resolve_viewer_for_league


def test_normalize_username():
    assert normalize_sleeper_username("  KaeDon ") == "kaedon"
    assert normalize_sleeper_username("") == ""
    assert normalize_sleeper_username(None) == ""


USERS = [
    {"user_id": "1", "username": "kaedon", "display_name": "Kae",
     "metadata": {"team_name": "Dream Team"}},
    {"user_id": "2", "username": "rival", "display_name": "Rival",
     "metadata": {"team_name": "The Others"}},
]
ROSTERS = [
    {"roster_id": 10, "owner_id": "1", "metadata": {"team_name": "Dream Team"}},
    {"roster_id": 20, "owner_id": "2"},
]


def test_match_by_user_id_preferred():
    v = resolve_viewer_for_league(USERS, ROSTERS, "anything", user_id="2")
    assert v["viewer_user_id"] == "2"
    assert v["viewer_roster_id"] == "20"


def test_match_by_username():
    v = resolve_viewer_for_league(USERS, ROSTERS, "kaedon")
    assert v["viewer_user_id"] == "1"
    assert v["viewer_roster_id"] == "10"
    assert v["viewer_team_name"] == "Dream Team"


def test_match_by_display_name_case_insensitive():
    v = resolve_viewer_for_league(USERS, ROSTERS, "  RIVAL ")
    assert v["viewer_user_id"] == "2"


def test_match_by_team_name():
    v = resolve_viewer_for_league(USERS, ROSTERS, "the others")
    assert v["viewer_user_id"] == "2"


def test_no_match_returns_none():
    assert resolve_viewer_for_league(USERS, ROSTERS, "ghost") is None


def test_empty_username_returns_none():
    assert resolve_viewer_for_league(USERS, ROSTERS, "") is None


def test_user_without_roster_returns_partial():
    users = [{"user_id": "9", "username": "loner", "display_name": "Loner",
              "metadata": {"team_name": "Solo"}}]
    v = resolve_viewer_for_league(users, [], "loner")
    assert v["viewer_roster_id"] is None
    assert v["viewer_team_name"] == "Solo"


def test_roster_team_name_prefers_roster_metadata():
    users = [{"user_id": "1", "username": "kaedon", "display_name": "Kae",
              "metadata": {"team_name": "Old Name"}}]
    rosters = [{"roster_id": 10, "owner_id": "1",
                "metadata": {"team_name": "New Name"}}]
    v = resolve_viewer_for_league(users, rosters, "kaedon")
    assert v["viewer_team_name"] == "New Name"


def test_fallback_team_name_uses_roster_id():
    users = [{"user_id": "1", "username": "", "display_name": "",
              "metadata": {}}]
    rosters = [{"roster_id": 7, "owner_id": "1", "metadata": {}}]
    v = resolve_viewer_for_league(users, rosters, "anything", user_id="1")
    assert v["viewer_team_name"] == "Roster 7"
