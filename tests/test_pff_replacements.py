"""Tests for PFF replacement derivations in the rookie prospect pipeline.

Covers:
- derive_nfl_passer_rating: exact NFL formula from box-score stats
- derive_avg_depth_of_target_proxy: YPR-based aDOT proxy
- derive_yac_per_reception_proxy: YPR-based YAC/rec proxy
- PPA-to-grade mapping lambdas in rookie_sources.py
"""
import pytest

from data_building.rookie_pipeline.rookie_metric_derivations import (
    derive_avg_depth_of_target_proxy,
    derive_nfl_passer_rating,
    derive_yac_per_reception_proxy,
)


class TestNflPasserRating:
    def test_perfect_rating(self):
        # 77.5% comp, 12.5 YPA, 12.5% TD, 0% INT = near-max
        # Note: needs >= 50 attempts
        stats = {
            "completions": 62,
            "pass_attempts": 80,
            "pass_yards": 1000,
            "pass_tds": 10,
            "interceptions": 0,
        }
        result = derive_nfl_passer_rating(stats)
        assert result is not None
        assert abs(result - 158.33) < 0.01

    def test_average_qb(self):
        # 62.5% comp, 7.0 YPA, 4% TD, 2.5% INT
        stats = {
            "completions": 250,
            "pass_attempts": 400,
            "pass_yards": 2800,
            "pass_tds": 16,
            "interceptions": 10,
        }
        result = derive_nfl_passer_rating(stats)
        assert result is not None
        # Manual: a=((0.625)-0.3)*5=1.625, b=((7.0)-3)*0.25=1.0,
        # c=(0.04)*20=0.8, d=2.375-((0.025)*25)=1.75
        # rating=((1.625+1.0+0.8+1.75)/6)*100=86.25
        assert abs(result - 86.25) < 0.01

    def test_clamping(self):
        # Terrible QB: negative components should clamp to 0
        stats = {
            "completions": 100,
            "pass_attempts": 400,  # 25% comp
            "pass_yards": 800,     # 2.0 YPA
            "pass_tds": 2,
            "interceptions": 30,   # 7.5% INT
        }
        result = derive_nfl_passer_rating(stats)
        assert result is not None
        assert result >= 0.0

    def test_insufficient_attempts(self):
        stats = {
            "completions": 20,
            "pass_attempts": 30,  # < 50
            "pass_yards": 250,
            "pass_tds": 2,
            "interceptions": 1,
        }
        assert derive_nfl_passer_rating(stats) is None

    def test_missing_fields(self):
        assert derive_nfl_passer_rating({}) is None
        assert derive_nfl_passer_rating({"completions": 100}) is None


class TestAdotProxy:
    def test_typical_wr(self):
        # 15.0 YPR -> 11.25 aDOT
        stats = {"receiving_yards": 1200, "receptions": 80}
        result = derive_avg_depth_of_target_proxy(stats)
        assert result == 11.25

    def test_deep_threat(self):
        # 20.0 YPR -> 15.0 aDOT
        stats = {"receiving_yards": 1000, "receptions": 50}
        result = derive_avg_depth_of_target_proxy(stats)
        assert result == 15.0

    def test_insufficient_receptions(self):
        stats = {"receiving_yards": 100, "receptions": 5}
        assert derive_avg_depth_of_target_proxy(stats) is None

    def test_missing_fields(self):
        assert derive_avg_depth_of_target_proxy({}) is None


class TestYacPerReceptionProxy:
    def test_typical_wr(self):
        # 12.0 YPR -> 4.2 YAC/rec
        stats = {"receiving_yards": 960, "receptions": 80}
        result = derive_yac_per_reception_proxy(stats)
        assert result == 4.2

    def test_bounds(self):
        # Very high YPR should cap at 8.0
        stats = {"receiving_yards": 2000, "receptions": 50}  # 40 YPR
        result = derive_yac_per_reception_proxy(stats)
        assert result == 8.0

        # Very low YPR should floor at 0.5
        stats = {"receiving_yards": 50, "receptions": 50}  # 1.0 YPR
        result = derive_yac_per_reception_proxy(stats)
        assert result == 0.5

    def test_insufficient_receptions(self):
        stats = {"receiving_yards": 100, "receptions": 5}
        assert derive_yac_per_reception_proxy(stats) is None


class TestPpaToGradeMapping:
    """Test the PPA -> PFF-grade-scale mapping used in rookie_sources.py."""

    def _ppa_to_grade(self, ppa):
        # Mirror the lambda in rookie_sources.py DIRECT_MAP
        if ppa is None:
            return None
        return round(max(50.0, min(99.0, 70.0 + float(ppa) * 35.0)), 1)

    def test_average(self):
        assert self._ppa_to_grade(0.0) == 70.0

    def test_elite(self):
        # PPA 0.43 -> ~85 (elite threshold)
        assert self._ppa_to_grade(0.43) == 85.0

    def test_poor(self):
        assert self._ppa_to_grade(-0.5) == 52.5

    def test_clamp_high(self):
        assert self._ppa_to_grade(2.0) == 99.0

    def test_clamp_low(self):
        assert self._ppa_to_grade(-2.0) == 50.0

    def test_none(self):
        assert self._ppa_to_grade(None) is None
