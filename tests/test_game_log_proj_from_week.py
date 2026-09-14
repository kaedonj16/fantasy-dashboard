"""Unit tests for the game-log projection cutoff.

_game_log_proj_from_week decides the first week the player-modal Stats tab
(re)projects. Weeks before it are finished (real stats, or a DNP when the player
missed the game) and must never be overwritten by a projection; weeks at/after
it are still to come and get projected.
"""
import pytest

# app.py imports pandas (and flask) at module load. The fast "lint" CI shard has
# flask but not pandas, so guard on pandas first — otherwise importing app here
# raises at COLLECTION and aborts the whole run. The helper itself is pure; this
# test runs in the full-dependency shard.
pytest.importorskip("pandas")
pytest.importorskip("flask")

from app import _game_log_proj_from_week  # noqa: E402


def test_in_season_regular_projects_current_week_onward():
    # Current season, week 10, regular season -> weeks 1-9 finished, 10+ project.
    assert _game_log_proj_from_week(2025, 2025, 10, "regular") == 10


def test_in_season_post_uses_current_week():
    assert _game_log_proj_from_week(2025, 2025, 16, "post") == 16


def test_completed_past_season_projects_nothing():
    # Viewing a finished prior season: cutoff past the schedule so no week is
    # projected (real stats / DNP stay as-is).
    assert _game_log_proj_from_week(2024, 2025, 3, "regular") == 99


def test_future_season_projects_everything():
    assert _game_log_proj_from_week(2026, 2025, 3, "regular") == 1


def test_preseason_and_offseason_project_everything():
    # Current season not started yet: no week is "finished", project all.
    assert _game_log_proj_from_week(2025, 2025, 1, "pre") == 1
    assert _game_log_proj_from_week(2025, 2025, 18, "off") == 1
    assert _game_log_proj_from_week(2025, 2025, 5, "") == 1


def test_bad_inputs_default_to_projecting_all():
    assert _game_log_proj_from_week(None, 2025, 10, "regular") == 1
    assert _game_log_proj_from_week(2025, "x", 10, "regular") == 1


def test_week_one_regular_is_floored_at_one():
    assert _game_log_proj_from_week(2025, 2025, 0, "regular") == 1
