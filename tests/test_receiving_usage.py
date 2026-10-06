"""Tests for WR/TE receiving usage (target share by situation)."""

import pytest

from utils.rb_usage import (
    _is_receiver_target,
    _is_light_color,
    _build_player_color_map,
    _compute_team_receiving_buckets,
    get_team_receiving_usage,
)


class TestIsReceiverTarget:
    def test_completed_pass_counts(self):
        row = {
            "pass_attempt": 1, "complete_pass": 1,
            "receiver_player_id": "00-0031234",
            "receiver_player_name": "Khalil Shakir",
        }
        assert _is_receiver_target(row) == "00-0031234"

    def test_incomplete_pass_counts_as_target(self):
        # Target share is about opportunities, incompletions still count
        row = {
            "pass_attempt": 1, "complete_pass": 0,
            "receiver_player_id": "00-0031234",
            "receiver_player_name": "Khalil Shakir",
        }
        assert _is_receiver_target(row) == "00-0031234"

    def test_rush_not_a_target(self):
        row = {
            "rush_attempt": 1,
            "rusher_player_id": "00-0031235",
            "pass_attempt": 0,
        }
        assert _is_receiver_target(row) is None

    def test_missing_receiver_id(self):
        row = {"pass_attempt": 1, "complete_pass": 1, "receiver_player_id": None}
        assert _is_receiver_target(row) is None

    def test_nan_receiver_id(self):
        row = {"pass_attempt": 1, "complete_pass": 1, "receiver_player_id": "nan"}
        assert _is_receiver_target(row) is None


class TestLightColorFilter:
    def test_white_is_light(self):
        assert _is_light_color("#FFFFFF") is True

    def test_near_white_is_light(self):
        assert _is_light_color("#F5F5F5") is True

    def test_dark_colors_not_light(self):
        assert _is_light_color("#00338D") is False
        assert _is_light_color("#000000") is False
        assert _is_light_color("#C60C30") is False

    def test_bills_full_palette_used_with_light_flag(self):
        # BUF white is now usable; it must be flagged so the frontend
        # renders a visible border
        cmap = _build_player_color_map("BUF", {"wr1": 30, "wr2": 20, "wr3": 10})
        assert cmap["wr3"]["color"] == "#FFFFFF"
        assert cmap["wr3"]["light"] is True
        assert cmap["wr1"]["light"] is False

    def test_invalid_hex_not_light(self):
        assert _is_light_color("not-a-color") is False
        assert _is_light_color("#FFF") is False


class TestReceivingBuckets:
    def _fake_pbp(self, rows):
        """Build a minimal fake play-by-play DataFrame."""
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(rows)

    def test_targets_counted_per_situation(self):
        pd = pytest.importorskip("pandas")
        rows = [
            {
                "posteam": "BUF", "week": 4, "season_type": "REG",
                "down": 1, "ydstogo": 10, "yardline_100": 75,
                "game_seconds_remaining": 1800,
                "pass_attempt": 1, "complete_pass": 1,
                "receiver_player_id": "wr1", "receiver_player_name": "Khalil Shakir",
                "rush_attempt": 0,
            },
            {
                "posteam": "BUF", "week": 4, "season_type": "REG",
                "down": 3, "ydstogo": 5, "yardline_100": 50,
                "game_seconds_remaining": 1800,
                "pass_attempt": 1, "complete_pass": 0,
                "receiver_player_id": "wr1", "receiver_player_name": "Khalil Shakir",
                "rush_attempt": 0,
            },
            {
                "posteam": "BUF", "week": 4, "season_type": "REG",
                "down": 2, "ydstogo": 8, "yardline_100": 60,
                "game_seconds_remaining": 1800,
                "pass_attempt": 1, "complete_pass": 1,
                "receiver_player_id": "te1", "receiver_player_name": "Dalton Kincaid",
                "rush_attempt": 0,
            },
        ]
        pbp = self._fake_pbp(rows)
        buckets = _compute_team_receiving_buckets(pbp, "BUF", 4)
        # wr1: 2 targets (all, early x1, third x1)
        assert buckets["all"]["wr1"]["targets"] == 2
        assert buckets["early"]["wr1"]["targets"] == 1
        assert buckets["third"]["wr1"]["targets"] == 1
        # te1: 1 target (all, early)
        assert buckets["all"]["te1"]["targets"] == 1
        assert buckets["early"]["te1"]["targets"] == 1

    def test_other_team_excluded(self):
        pd = pytest.importorskip("pandas")
        rows = [
            {
                "posteam": "KC", "week": 4, "season_type": "REG",
                "down": 1, "ydstogo": 10, "yardline_100": 75,
                "game_seconds_remaining": 1800,
                "pass_attempt": 1, "complete_pass": 1,
                "receiver_player_id": "wr9", "receiver_player_name": "Other Guy",
                "rush_attempt": 0,
            },
        ]
        pbp = self._fake_pbp(rows)
        buckets = _compute_team_receiving_buckets(pbp, "BUF", 4)
        assert buckets["all"] == {}

    def test_wrong_week_excluded(self):
        pd = pytest.importorskip("pandas")
        rows = [
            {
                "posteam": "BUF", "week": 3, "season_type": "REG",
                "down": 1, "ydstogo": 10, "yardline_100": 75,
                "game_seconds_remaining": 1800,
                "pass_attempt": 1, "complete_pass": 1,
                "receiver_player_id": "wr1", "receiver_player_name": "Khalil Shakir",
                "rush_attempt": 0,
            },
        ]
        pbp = self._fake_pbp(rows)
        buckets = _compute_team_receiving_buckets(pbp, "BUF", 4)
        assert buckets["all"] == {}


class TestGetTeamReceivingUsage:
    def test_empty_team_returns_empty(self):
        result = get_team_receiving_usage("", 2026, 4)
        assert result["situations"] == []

    def test_response_shape(self):
        # Without play-by-play data (no network in tests), should return
        # either data or a graceful empty/error shape
        result = get_team_receiving_usage("BUF", 2026, 4)
        assert result["team"] == "BUF"
        assert result["season"] == 2026
        assert result["week"] == 4
        assert isinstance(result["situations"], list)
        for sit in result["situations"]:
            assert "key" in sit
            assert "label" in sit
            assert "total" in sit
            assert "segments" in sit
            for seg in sit["segments"]:
                # Light colors (e.g. white) are only allowed when flagged so
                # the frontend renders a visible border around the segment
                if _is_light_color(seg["color"]):
                    assert seg.get("light") is True, (
                        f"Segment {seg['name']} has light color {seg['color']} "
                        "without the light flag"
                    )
