"""Unit tests for utils.league_payload.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.league_payload import (
    build_roster_map,
    format_sleeper_league_option,
    get_most_recent_valid_draft_for_season,
)


# ---- format_sleeper_league_option -----------------------------------------

def test_format_full_league():
    lg = {"league_id": 123, "name": "Dynasty Warriors", "season": 2024,
          "total_rosters": 12, "avatar": "abc"}
    out = format_sleeper_league_option(lg)
    assert out["league_id"] == "123"
    assert out["name"] == "Dynasty Warriors"
    assert out["season"] == "2024"
    assert out["total_rosters"] == 12
    assert out["avatar"] == "abc"
    assert out["label"] == "Dynasty Warriors (2024) • 12 teams"


def test_format_defaults_and_settings_fallback():
    lg = {"settings": {"num_teams": 10}}
    out = format_sleeper_league_option(lg)
    assert out["name"] == "Unnamed League"
    assert out["total_rosters"] == 10
    assert out["label"] == "Unnamed League () • 10 teams"


def test_format_unknown_team_count():
    out = format_sleeper_league_option({"name": "X", "season": 2023})
    assert out["label"] == "X (2023) • ? teams"


# ---- get_most_recent_valid_draft_for_season -------------------------------

def test_empty_or_non_list_returns_none():
    assert get_most_recent_valid_draft_for_season([], 2024) is None
    assert get_most_recent_valid_draft_for_season(None, 2024) is None


def test_picks_newest_by_best_timestamp():
    drafts = [
        {"draft_id": "a", "season": "2024", "start_time": 100},
        {"draft_id": "b", "season": "2024", "created": 200},
    ]
    out = get_most_recent_valid_draft_for_season(drafts, 2024)
    assert out["draft_id"] == "b"


def test_newest_from_older_season_returns_none():
    drafts = [{"draft_id": "old", "season": "2023", "start_time": 999}]
    assert get_most_recent_valid_draft_for_season(drafts, 2024) is None


def test_ignores_non_dict_entries():
    drafts = ["junk", {"draft_id": "a", "season": "2024", "created": 5}]
    out = get_most_recent_valid_draft_for_season(drafts, 2024)
    assert out["draft_id"] == "a"


# ---- build_roster_map -----------------------------------------------------

def test_roster_metadata_team_name_wins():
    users = [{"user_id": "1", "display_name": "Kae"}]
    rosters = [{"roster_id": 10, "owner_id": "1", "metadata": {"team_name": "Champs"}}]
    assert build_roster_map(users, rosters) == {"10": "Champs"}


def test_falls_back_to_user_name_chain():
    users = [{"user_id": "1", "metadata": {}, "username": "kae"}]
    rosters = [{"roster_id": 10, "owner_id": "1", "metadata": {}}]
    assert build_roster_map(users, rosters) == {"10": "kae"}


def test_orphan_roster_uses_roster_id_label():
    users = []
    rosters = [{"roster_id": 7, "owner_id": "99", "metadata": {}}]
    assert build_roster_map(users, rosters) == {"7": "Roster 7"}
