"""Tests for utils.schedule_ease."""
from utils.schedule_ease import matchup_cell_ease, norm_sched_team, sched_rank_color


class TestNormSchedTeam:
    def test_aliases_map_to_canonical(self):
        assert norm_sched_team("WSH") == "WAS"
        assert norm_sched_team("JAC") == "JAX"
        assert norm_sched_team("OAK") == "LV"
        assert norm_sched_team("SD") == "LAC"

    def test_canonical_passthrough(self):
        assert norm_sched_team("KC") == "KC"

    def test_case_and_whitespace(self):
        assert norm_sched_team(" wsh ") == "WAS"

    def test_none_and_empty(self):
        assert norm_sched_team(None) == ""
        assert norm_sched_team("") == ""


class TestSchedRankColor:
    def test_easiest_quartile_green(self):
        color, bg = sched_rank_color(1, 32)
        assert color == "#22c55e"
        assert bg.startswith("#22c55e")

    def test_hardest_quartile_red(self):
        color, _ = sched_rank_color(32, 32)
        assert color == "#ef4444"

    def test_missing_rank_is_neutral(self):
        assert sched_rank_color(None, 32) == ("#6b7280", "transparent")
        assert sched_rank_color(5, 0) == ("#6b7280", "transparent")

    def test_quartile_boundaries(self):
        assert sched_rank_color(8, 32)[0] == "#22c55e"    # 0.25 exactly
        assert sched_rank_color(16, 32)[0] == "#84cc16"   # 0.50 exactly
        assert sched_rank_color(24, 32)[0] == "#f59e0b"   # 0.75 exactly
        assert sched_rank_color(25, 32)[0] == "#ef4444"


class TestMatchupCellEase:
    def test_prefers_z_derived_ease(self):
        assert matchup_cell_ease(30, 32, {"ease": 87.5}) == 87.5

    def test_falls_back_to_rank_percentile(self):
        assert matchup_cell_ease(1, 33, None) == 100.0
        assert matchup_cell_ease(33, 33, None) == 0.0

    def test_midpoint(self):
        assert matchup_cell_ease(17, 33, {}) == 50.0

    def test_no_data(self):
        assert matchup_cell_ease(None, None, None) == 0.0
        assert matchup_cell_ease(1, 1, None) == 0.0
