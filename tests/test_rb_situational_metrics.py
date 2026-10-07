"""Tests for situational RB usage metrics in Advanced Metrics."""

import pytest

from data_building.advanced_metrics import (
    LEADERBOARD_METRICS,
    SITUATIONAL_METRICS,
)


class TestSituationalMetricsRegistry:
    def test_all_five_metrics_defined(self):
        assert set(SITUATIONAL_METRICS) == {
            "goalline_share", "short_yardage_share", "third_down_share",
            "early_down_share", "two_minute_share",
        }

    def test_metrics_in_leaderboard_registry(self):
        for key in SITUATIONAL_METRICS:
            assert key in LEADERBOARD_METRICS, f"{key} missing from LEADERBOARD_METRICS"
            spec = LEADERBOARD_METRICS[key]
            assert spec.get("situational_metric") is True
            assert spec["positions"] == ["RB"]
            assert spec["category"] == "Rushing", f"{key} should live in the Rushing category"
            assert spec.get("pct") is True
            assert spec.get("label"), f"{key} needs a label"
            assert spec.get("desc"), f"{key} needs a description"

    def test_metrics_are_free(self):
        """Situational metrics must not be PRO or premium gated."""
        from data_building.advanced_metrics import PRO_METRICS, PREMIUM_METRICS
        for key in SITUATIONAL_METRICS:
            assert key not in PRO_METRICS, f"{key} should not be PRO-gated"
            assert key not in PREMIUM_METRICS, f"{key} should not be premium"

    def test_rb_preset_includes_situational_metrics(self):
        from dashboard_services.pages.advanced_metrics_page import ADVANCED_METRIC_PRESETS
        rb_metrics = ADVANCED_METRIC_PRESETS["rb"]["metrics"]
        for key in SITUATIONAL_METRICS:
            assert key in rb_metrics, f"{key} missing from rb preset"


class TestSituationalLeaderboard:
    def test_unknown_metric_returns_empty(self):
        from data_building.advanced_metrics import get_situational_leaderboard
        assert get_situational_leaderboard("not_a_metric") == []

    def test_leaderboard_with_mocked_data(self, monkeypatch):
        """Mock the pbp layer and verify share computation and sorting."""
        import data_building.advanced_metrics as am

        # Fake per-player shares: {gsis_id: {name, team, metric: value}}
        fake_shares = {
            "00-001": {"name": "Back A", "team": "CHI",
                       "goalline_share": 70.0, "short_yardage_share": 50.0,
                       "third_down_share": 30.0, "early_down_share": 60.0,
                       "two_minute_share": 80.0},
            "00-002": {"name": "Back B", "team": "CHI",
                       "goalline_share": 30.0, "short_yardage_share": 50.0,
                       "third_down_share": 70.0, "early_down_share": 40.0,
                       "two_minute_share": 20.0},
        }
        monkeypatch.setattr(
            "utils.rb_usage.get_all_player_situational_shares",
            lambda season, week=None, **kw: fake_shares,
        )
        # GSIS -> Sleeper crosswalk
        monkeypatch.setattr(
            "data_building.external_data.nflverse_metrics._gsis_to_sleeper",
            lambda: {"00-001": "111", "00-002": "222"},
        )
        # Players index metadata
        monkeypatch.setattr(
            "utils.utils.load_players_index",
            lambda: {
                "111": {"name": "Back A", "team": "CHI", "position": "RB"},
                "222": {"name": "Back B", "team": "CHI", "position": "RB"},
            },
        )

        rows = am.get_situational_leaderboard("goalline_share", season=2026)
        assert len(rows) == 2
        # Sorted desc by value
        assert rows[0]["player_id"] == "111"
        assert rows[0]["value"] == 70.0
        assert rows[0]["name"] == "Back A"
        assert rows[0]["position"] == "RB"
        assert rows[1]["player_id"] == "222"
        assert rows[1]["value"] == 30.0

    def test_position_filter(self, monkeypatch):
        import data_building.advanced_metrics as am

        fake_shares = {
            "00-001": {"name": "Back A", "team": "CHI", "goalline_share": 70.0},
        }
        monkeypatch.setattr(
            "utils.rb_usage.get_all_player_situational_shares",
            lambda season, week=None, **kw: fake_shares,
        )
        monkeypatch.setattr(
            "data_building.external_data.nflverse_metrics._gsis_to_sleeper",
            lambda: {"00-001": "111"},
        )
        monkeypatch.setattr(
            "utils.utils.load_players_index",
            lambda: {"111": {"name": "Back A", "team": "CHI", "position": "RB"}},
        )

        # RB filter passes
        assert len(am.get_situational_leaderboard(
            "goalline_share", position="RB", season=2026)) == 1
        # WR filter excludes
        assert am.get_situational_leaderboard(
            "goalline_share", position="WR", season=2026) == []


class TestAllPlayerShares:
    def test_shares_sum_per_situation(self, monkeypatch):
        """Each player's shares across a team should reflect touch proportions."""
        pd = pytest.importorskip("pandas")
        import utils.rb_usage as rb_usage

        # Build a tiny fake pbp DataFrame: 4 goalline touches, 3 for A, 1 for B
        rows = []
        for i, (pid, name) in enumerate(
                [("00-001", "Back A")] * 3 + [("00-002", "Back B")]):
            rows.append({
                "game_id": "g1", "play_id": i, "week": 4, "season_type": "REG",
                "posteam": "CHI", "defteam": "GB",
                "down": 1, "ydstogo": 5, "yardline_100": 8,
                "qtr": 2, "game_half": "Half1", "game_seconds_remaining": 1800,
                "play_type": "run",
                "rusher_player_id": pid, "rusher_player_name": name,
                "receiver_player_id": None, "receiver_player_name": None,
                "rush_attempt": 1, "pass_attempt": 0, "complete_pass": 0,
            })
        fake_pbp = pd.DataFrame(rows)
        monkeypatch.setattr(rb_usage, "_load_pbp", lambda season: fake_pbp)
        monkeypatch.setattr(
            rb_usage, "_get_rb_gsis_ids_by_team",
            lambda season: {"CHI": {"00-001", "00-002"}},
        )

        shares = rb_usage.get_all_player_situational_shares(2026, 4)
        assert "00-001" in shares
        assert "00-002" in shares
        # 3 of 4 goalline touches for Back A
        assert shares["00-001"]["goalline_share"] == 75.0
        assert shares["00-002"]["goalline_share"] == 25.0
        # Both also count as early-down touches (down=1)
        assert shares["00-001"]["early_down_share"] == 75.0


def _pbp_row(pid, name, week, down=3, rush=True):
    """One fake play-by-play row: a rush or a completed reception."""
    return {
        "game_id": "g1", "play_id": f"{pid}-{week}", "week": week,
        "season_type": "REG", "posteam": "CHI", "defteam": "GB",
        "down": down, "ydstogo": 5, "yardline_100": 50,
        "qtr": 2, "game_half": "Half1", "game_seconds_remaining": 1800,
        "play_type": "run" if rush else "pass",
        "rusher_player_id": pid if rush else None,
        "rusher_player_name": name if rush else None,
        "receiver_player_id": None if rush else pid,
        "receiver_player_name": None if rush else name,
        "rush_attempt": 1 if rush else 0,
        "pass_attempt": 0 if rush else 1,
        "complete_pass": 0 if rush else 1,
    }


class TestSituationalRbFilter:
    def test_non_rb_touches_excluded(self, monkeypatch):
        """WR/TE/QB touches must not enter RB situational shares."""
        pd = pytest.importorskip("pandas")
        import utils.rb_usage as rb_usage

        rows = [
            _pbp_row("00-001", "Back A", week=4),                       # RB rush
            _pbp_row("00-002", "Back B", week=4),                       # RB rush
            _pbp_row("00-009", "Wideout W", week=4, rush=False),         # WR reception
        ]
        monkeypatch.setattr(rb_usage, "_load_pbp", lambda season: pd.DataFrame(rows))
        # Only the two backs are roster RBs.
        monkeypatch.setattr(
            rb_usage, "_get_rb_gsis_ids_by_team",
            lambda season: {"CHI": {"00-001", "00-002"}},
        )

        shares = rb_usage.get_all_player_situational_shares(2026, 4)
        assert set(shares) == {"00-001", "00-002"}
        # 50/50 of the RB third-down touches, not 33/33/33.
        assert shares["00-001"]["third_down_share"] == 50.0
        assert shares["00-002"]["third_down_share"] == 50.0

    def test_fail_open_without_roster_data(self, monkeypatch):
        """No roster data -> unfiltered (existing fail-open behavior)."""
        pd = pytest.importorskip("pandas")
        import utils.rb_usage as rb_usage

        rows = [_pbp_row("00-001", "Back A", week=4)]
        monkeypatch.setattr(rb_usage, "_load_pbp", lambda season: pd.DataFrame(rows))
        monkeypatch.setattr(rb_usage, "_get_rb_gsis_ids_by_team", lambda season: {})

        shares = rb_usage.get_all_player_situational_shares(2026, 4)
        assert "00-001" in shares


class TestSituationalWeekRanges:
    def test_season_aggregates_all_weeks(self, monkeypatch):
        """week=None aggregates the full season, not one week."""
        pd = pytest.importorskip("pandas")
        import utils.rb_usage as rb_usage

        rows = [
            _pbp_row("00-001", "Back A", week=1),
            _pbp_row("00-001", "Back A", week=2),
            _pbp_row("00-002", "Back B", week=2),
        ]
        monkeypatch.setattr(rb_usage, "_load_pbp", lambda season: pd.DataFrame(rows))
        monkeypatch.setattr(
            rb_usage, "_get_rb_gsis_ids_by_team",
            lambda season: {"CHI": {"00-001", "00-002"}},
        )

        shares = rb_usage.get_all_player_situational_shares(2026)
        # 3 third-down touches total: 2 for A, 1 for B.
        assert shares["00-001"]["third_down_share"] == pytest.approx(66.7, abs=0.1)
        assert shares["00-002"]["third_down_share"] == pytest.approx(33.3, abs=0.1)

    def test_week_range_filters(self, monkeypatch):
        """week_start/week_end restrict to that inclusive range."""
        pd = pytest.importorskip("pandas")
        import utils.rb_usage as rb_usage

        rows = [
            _pbp_row("00-001", "Back A", week=1),
            _pbp_row("00-001", "Back A", week=2),
            _pbp_row("00-002", "Back B", week=2),
        ]
        monkeypatch.setattr(rb_usage, "_load_pbp", lambda season: pd.DataFrame(rows))
        monkeypatch.setattr(
            rb_usage, "_get_rb_gsis_ids_by_team",
            lambda season: {"CHI": {"00-001", "00-002"}},
        )

        shares = rb_usage.get_all_player_situational_shares(
            2026, week_start=2, week_end=2)
        assert shares["00-001"]["third_down_share"] == 50.0
        assert shares["00-002"]["third_down_share"] == 50.0

        shares_w1 = rb_usage.get_all_player_situational_shares(
            2026, week_start=1, week_end=1)
        assert set(shares_w1) == {"00-001"}
        assert shares_w1["00-001"]["third_down_share"] == 100.0


class TestSituationalRushOnly:
    def test_rush_only_excludes_receptions(self, monkeypatch):
        """rush_only=True counts rushing attempts, not receptions."""
        pd = pytest.importorskip("pandas")
        import utils.rb_usage as rb_usage

        rows = [
            _pbp_row("00-001", "Back A", week=4),                # rush
            _pbp_row("00-001", "Back A", week=4, rush=False),   # reception
            _pbp_row("00-002", "Back B", week=4),                # rush
        ]
        monkeypatch.setattr(rb_usage, "_load_pbp", lambda season: pd.DataFrame(rows))
        monkeypatch.setattr(
            rb_usage, "_get_rb_gsis_ids_by_team",
            lambda season: {"CHI": {"00-001", "00-002"}},
        )

        # Default: rushes + receptions (3 third-down touches: 2 for A, 1 for B).
        shares = rb_usage.get_all_player_situational_shares(2026, 4)
        assert shares["00-001"]["third_down_share"] == pytest.approx(66.7, abs=0.1)

        # Rush-only: 2 third-down rushes, 1 each.
        shares_ro = rb_usage.get_all_player_situational_shares(2026, 4, rush_only=True)
        assert shares_ro["00-001"]["third_down_share"] == 50.0
        assert shares_ro["00-002"]["third_down_share"] == 50.0
