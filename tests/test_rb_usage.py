"""Tests for utils/rb_usage.py - RB situational usage bars."""

import pytest

from utils.rb_usage import (
    TEAM_COLORS,
    SITUATIONS,
    _situation_keys,
    _is_rb_touch,
    _build_player_color_map,
    _is_light_color,
)


class TestTeamColors:
    def test_all_32_teams_have_colors(self):
        assert len(TEAM_COLORS) == 32

    def test_bears_colors(self):
        assert TEAM_COLORS["CHI"][0] == "#0B1628"
        assert TEAM_COLORS["CHI"][1] == "#C83803"

    def test_each_team_has_three_colors(self):
        for team, colors in TEAM_COLORS.items():
            assert len(colors) == 3, f"{team} should have 3 colors"


class TestSituations:
    def test_six_situations_defined(self):
        assert len(SITUATIONS) == 6
        keys = [k for k, _ in SITUATIONS]
        assert keys == ["all", "early", "goalline", "short", "third", "two_min"]


class TestSituationKeys:
    def test_all_plays_always_included(self):
        row = {"down": 1, "ydstogo": 10, "yardline_100": 75, "game_seconds_remaining": 1800}
        assert "all" in _situation_keys(row)

    def test_early_downs(self):
        for down in (1, 2):
            row = {"down": down, "ydstogo": 10, "yardline_100": 75, "game_seconds_remaining": 1800}
            assert "early" in _situation_keys(row)
        row = {"down": 3, "ydstogo": 10, "yardline_100": 75, "game_seconds_remaining": 1800}
        assert "early" not in _situation_keys(row)

    def test_third_downs(self):
        row = {"down": 3, "ydstogo": 5, "yardline_100": 50, "game_seconds_remaining": 1800}
        assert "third" in _situation_keys(row)
        row = {"down": 2, "ydstogo": 5, "yardline_100": 50, "game_seconds_remaining": 1800}
        assert "third" not in _situation_keys(row)

    def test_goalline(self):
        row = {"down": 1, "ydstogo": 10, "yardline_100": 8, "game_seconds_remaining": 1800}
        assert "goalline" in _situation_keys(row)
        row = {"down": 1, "ydstogo": 10, "yardline_100": 25, "game_seconds_remaining": 1800}
        assert "goalline" not in _situation_keys(row)

    def test_short_yardage(self):
        # 3rd and 2
        row = {"down": 3, "ydstogo": 2, "yardline_100": 50, "game_seconds_remaining": 1800}
        assert "short" in _situation_keys(row)
        # 4th and 1
        row = {"down": 4, "ydstogo": 1, "yardline_100": 50, "game_seconds_remaining": 1800}
        assert "short" in _situation_keys(row)
        # 3rd and 5 (not short)
        row = {"down": 3, "ydstogo": 5, "yardline_100": 50, "game_seconds_remaining": 1800}
        assert "short" not in _situation_keys(row)
        # 2nd and 1 (not 3rd/4th)
        row = {"down": 2, "ydstogo": 1, "yardline_100": 50, "game_seconds_remaining": 1800}
        assert "short" not in _situation_keys(row)

    def test_two_minute_drill(self):
        # Last 2 min of 2nd half (0-120 seconds remaining)
        row = {"down": 2, "ydstogo": 10, "yardline_100": 50, "game_seconds_remaining": 90}
        assert "two_min" in _situation_keys(row)
        # Last 2 min of 1st half (1800-1920 seconds remaining)
        row = {"down": 2, "ydstogo": 10, "yardline_100": 50, "game_seconds_remaining": 1850}
        assert "two_min" in _situation_keys(row)
        # Mid-game
        row = {"down": 2, "ydstogo": 10, "yardline_100": 50, "game_seconds_remaining": 1500}
        assert "two_min" not in _situation_keys(row)


class TestIsRbTouch:
    def test_rush_attempt(self):
        row = {"rush_attempt": 1, "rusher_player_id": "00-123", "pass_attempt": 0}
        assert _is_rb_touch(row) == "00-123"

    def test_completed_pass(self):
        row = {"rush_attempt": 0, "pass_attempt": 1, "complete_pass": 1,
               "receiver_player_id": "00-456"}
        assert _is_rb_touch(row) == "00-456"

    def test_incomplete_pass_not_touch(self):
        row = {"rush_attempt": 0, "pass_attempt": 1, "complete_pass": 0,
               "receiver_player_id": "00-456"}
        assert _is_rb_touch(row) is None

    def test_no_touch(self):
        row = {"rush_attempt": 0, "pass_attempt": 0}
        assert _is_rb_touch(row) is None


class TestBuildPlayerColorMap:
    def test_top_player_gets_primary_color(self):
        cmap = _build_player_color_map("BUF", {"rb1": 50, "rb2": 30, "rb3": 10})
        assert cmap["rb1"]["color"] == "#00338D"  # BUF primary
        assert cmap["rb2"]["color"] == "#C60C30"  # BUF secondary
        assert cmap["rb1"]["light"] is False

    def test_white_flagged_as_light(self):
        # BUF tertiary is white; third-ranked player gets it with light flag
        cmap = _build_player_color_map("BUF", {"rb1": 50, "rb2": 30, "rb3": 10})
        assert cmap["rb3"]["color"] == "#FFFFFF"
        assert cmap["rb3"]["light"] is True

    def test_colors_cycle_when_more_players_than_colors(self):
        totals = {f"rb{i}": 50 - i for i in range(5)}
        cmap = _build_player_color_map("BUF", totals)
        # 4th player cycles back to primary
        assert cmap["rb3"]["color"] == "#00338D"
        assert cmap["rb4"]["color"] == "#C60C30"

    def test_same_player_same_color_regardless_of_call_order(self):
        cmap1 = _build_player_color_map("BUF", {"a": 10, "b": 20})
        cmap2 = _build_player_color_map("BUF", {"b": 20, "a": 10})
        assert cmap1 == cmap2


class TestPositionFiltering:
    """Position filtering must not silently fail open on production."""

    def _clear_caches(self):
        from utils import rb_usage
        rb_usage._ROSTER_POS_CACHE.clear()
        rb_usage._POS_GSIS_CACHE.clear()

    def test_roster_lookup_returns_rb_gsis_ids(self, monkeypatch):
        """Primary source: nflverse rosters give positions by GSIS ID."""
        pd = pytest.importorskip("pandas")
        from utils import rb_usage
        self._clear_caches()

        fake_rosters = pd.DataFrame([
            {"player_id": "gsis-rb1", "position": "RB", "team": "BUF"},
            {"player_id": "gsis-rb2", "position": "RB", "team": "BUF"},
            {"player_id": "gsis-wr1", "position": "WR", "team": "BUF"},
            {"player_id": "gsis-qb1", "position": "QB", "team": "BUF"},
            {"player_id": "gsis-rb-other", "position": "RB", "team": "KC"},
        ])
        fake_nfl = type("FakeNfl", (), {
            "import_seasonal_rosters": staticmethod(lambda seasons: fake_rosters),
        })()
        monkeypatch.setitem(__import__("sys").modules, "nfl_data_py", fake_nfl)

        ids = rb_usage._get_position_gsis_ids_from_rosters(2026, "BUF", ["RB"])
        assert ids == {"gsis-rb1", "gsis-rb2"}

    def test_roster_lookup_filters_wr_te(self, monkeypatch):
        pd = pytest.importorskip("pandas")
        from utils import rb_usage
        self._clear_caches()

        fake_rosters = pd.DataFrame([
            {"player_id": "gsis-rb1", "position": "RB", "team": "BUF"},
            {"player_id": "gsis-wr1", "position": "WR", "team": "BUF"},
            {"player_id": "gsis-te1", "position": "TE", "team": "BUF"},
        ])
        fake_nfl = type("FakeNfl", (), {
            "import_seasonal_rosters": staticmethod(lambda seasons: fake_rosters),
        })()
        monkeypatch.setitem(__import__("sys").modules, "nfl_data_py", fake_nfl)

        ids = rb_usage._get_position_gsis_ids("BUF", ["WR", "TE"], season=2026)
        assert ids == {"gsis-wr1", "gsis-te1"}

    def test_falls_back_to_db_when_rosters_unavailable(self, monkeypatch):
        """If nflverse rosters fail, try the crosswalk + DB path."""
        from utils import rb_usage
        self._clear_caches()

        # Rosters raise -> primary source unavailable
        def _boom(seasons):
            raise RuntimeError("no network")
        fake_nfl = type("FakeNfl", (), {
            "import_seasonal_rosters": staticmethod(_boom),
            "import_weekly_rosters": staticmethod(_boom),
            "import_rosters": staticmethod(_boom),
        })()
        monkeypatch.setitem(__import__("sys").modules, "nfl_data_py", fake_nfl)

        # DB fallback returns RBs
        monkeypatch.setattr(
            rb_usage, "_get_position_gsis_ids_from_db",
            lambda team, positions: {"gsis-rb1"},
        )
        ids = rb_usage._get_position_gsis_ids("BUF", ["RB"], season=2026)
        assert ids == {"gsis-rb1"}

    def test_logs_warning_when_both_sources_fail(self, monkeypatch, caplog):
        """Last resort: empty set + warning, never silent."""
        import logging
        from utils import rb_usage
        self._clear_caches()

        def _boom(seasons):
            raise RuntimeError("no network")
        fake_nfl = type("FakeNfl", (), {
            "import_seasonal_rosters": staticmethod(_boom),
            "import_weekly_rosters": staticmethod(_boom),
            "import_rosters": staticmethod(_boom),
        })()
        monkeypatch.setitem(__import__("sys").modules, "nfl_data_py", fake_nfl)
        monkeypatch.setattr(
            rb_usage, "_get_position_gsis_ids_from_db",
            lambda team, positions: set(),
        )
        with caplog.at_level(logging.WARNING, logger="utils.rb_usage"):
            ids = rb_usage._get_position_gsis_ids("BUF", ["RB"], season=2026)
        assert ids == set()
        assert any("position lookup failed" in r.message for r in caplog.records)
