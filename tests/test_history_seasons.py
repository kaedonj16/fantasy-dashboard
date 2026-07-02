"""Unit tests for utils.history_seasons.get_default_history_season.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.history_seasons import get_default_history_season


def test_picks_most_recent_completed_season():
    assert get_default_history_season([2021, 2022, 2023], 2024) == 2023


def test_excludes_current_season():
    # 2024 is current -> pick the newest that is strictly earlier.
    assert get_default_history_season([2022, 2023, 2024], 2024) == 2023


def test_no_prior_season_falls_back_to_newest():
    # Only the current (or future) seasons available -> newest available.
    assert get_default_history_season([2024, 2025], 2024) == 2025


def test_empty_returns_current_season():
    assert get_default_history_season([], 2024) == 2024


def test_falsy_entries_ignored():
    assert get_default_history_season([0, None, 2022], 2024) == 2022


def test_deduplicates_and_orders():
    assert get_default_history_season([2022, 2022, 2021], 2024) == 2022


def test_string_seasons_coerced():
    assert get_default_history_season(["2021", "2023"], "2024") == 2023
