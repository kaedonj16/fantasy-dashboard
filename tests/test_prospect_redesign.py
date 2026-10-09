"""Tests for the prospect rankings redesign (synthetic data, no DB needed)."""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services.rookie_api import (
    _build_advanced_metrics,
    _clip100,
    _compute_breakout_age,
    _grade_breakout_age,
    _grade_cmp,
    _grade_dominator,
    _grade_market_share,
    _grade_speed_score,
    _grade_td_int,
    _grade_ypa,
    _grade_ypc,
    _grade_ypr,
)
from dashboard_services.pages.rookies_page import build_prospects_body


class TestDisplayGrades:
    def test_clip100_bounds(self):
        assert _clip100(150) == 100.0
        assert _clip100(-5) == 0.0
        assert _clip100(None) is None
        assert _clip100("bad") is None

    def test_dominator_anchors_on_model_elite_mark(self):
        # The model treats >= 0.35 as an elite dominator.
        assert _grade_dominator(0.40) == 100.0
        assert _grade_dominator(0.35) == 87.5
        assert _grade_dominator(None) is None

    def test_breakout_age_uses_model_scale(self):
        assert _grade_breakout_age(19.1, "WR") == round((23.0 - 19.1) / 4.5 * 100, 1)
        assert _grade_breakout_age(19.3, "QB") == round((23.5 - 19.3) / 5.0 * 100, 1)
        assert _grade_breakout_age(None, "WR") is None

    def test_efficiency_grades(self):
        assert _grade_ypr(18.0) == 100.0
        assert _grade_ypc(7.0) == 100.0
        assert _grade_market_share(0.35) == 100.0
        assert _grade_speed_score(120.0) == 100.0
        assert _grade_cmp(70.0) == 100.0
        assert _grade_td_int(4.0) == 100.0
        assert _grade_ypa(9.5) == 100.0
        assert _grade_ypr(None) is None


class TestBreakoutAge:
    def test_first_season_over_threshold(self):
        seasons = [
            {"season": 2023, "dominator_rating": 0.18},
            {"season": 2024, "dominator_rating": 0.31},
            {"season": 2025, "dominator_rating": 0.38},
        ]
        assert _compute_breakout_age(seasons, 20.4, "WR") == 19.4

    def test_position_thresholds(self):
        # TE threshold is 0.12; a 0.15 TE season counts as a breakout.
        seasons = [{"season": 2025, "dominator_rating": 0.15}]
        assert _compute_breakout_age(seasons, 21.5, "TE") == 21.5
        # Same dominator is not a breakout for a WR (threshold 0.25).
        assert _compute_breakout_age(seasons, 20.4, "WR") is None

    def test_qb_uses_pass_share_proxy(self):
        seasons = [
            {"season": 2024, "pass_yards": 2800, "team_total_yards": 5000},
            {"season": 2025, "pass_yards": 3500, "team_total_yards": 5000},
        ]
        assert _compute_breakout_age(seasons, 21.1, "QB") == 21.1

    def test_never_broke_out_returns_none(self):
        seasons = [{"season": 2025, "dominator_rating": 0.10}]
        assert _compute_breakout_age(seasons, 21.0, "WR") is None
        assert _compute_breakout_age([], 21.0, "WR") is None
        assert _compute_breakout_age(None, 21.0, "WR") is None
        assert _compute_breakout_age(seasons, None, "WR") is None


class TestAdvancedMetrics:
    def _row(self, pos="WR"):
        return {"position": pos, "age": 20.4,
                "efficiency_score": 84, "production_score": 90}

    def test_wr_metric_set(self):
        seasons = [{"season": 2025, "dominator_rating": 0.38,
                    "market_share_yards": 0.34, "yds_per_reception": 15.9}]
        adv = _build_advanced_metrics("WR", seasons, {"speed_score": 112}, self._row())
        assert [a["label"] for a in adv] == [
            "Dominator", "Breakout Age", "Yds/Rec",
            "Mkt Share", "Speed Score", "Efficiency",
        ]
        assert adv[0]["raw"] == "38%"
        assert adv[0]["grade"] == 95.0
        assert adv[2]["raw"] == "15.9"
        assert adv[4]["raw"] == "112"

    def test_te_uses_wr_set(self):
        adv = _build_advanced_metrics("TE", [], {}, self._row("TE"))
        assert [a["label"] for a in adv][2] == "Yds/Rec"

    def test_rb_metric_set(self):
        seasons = [{"season": 2025, "dominator_rating": 0.33,
                    "market_share_yards": 0.29, "yds_per_carry": 6.8}]
        adv = _build_advanced_metrics("RB", seasons, {"speed_score": 118},
                                      self._row("RB"))
        assert [a["label"] for a in adv] == [
            "Dominator", "Breakout Age", "Yds/Carry",
            "Mkt Share", "Speed Score", "Efficiency",
        ]
        assert adv[2]["raw"] == "6.8"

    def test_qb_metric_set(self):
        seasons = [{"season": 2025, "completion_pct": 67.2, "td_int_ratio": 3.1,
                    "yds_per_attempt": 8.9, "pass_yards": 3500,
                    "team_total_yards": 5000}]
        adv = _build_advanced_metrics(
            "QB", seasons, {},
            {"position": "QB", "age": 21.1,
             "efficiency_score": 90, "production_score": 90})
        assert [a["label"] for a in adv] == [
            "Cmp%", "TD:INT", "Yds/Att",
            "Breakout Age", "Efficiency", "Production",
        ]
        assert adv[0]["raw"] == "67%"
        assert adv[1]["raw"] == "3.1"

    def test_missing_data_degrades_to_nulls(self):
        adv = _build_advanced_metrics(
            "WR", [], {}, {"position": "WR", "age": None,
                           "efficiency_score": None, "production_score": None})
        assert len(adv) == 6
        assert all(a["raw"] is None and a["grade"] is None for a in adv)


class TestPageBuilder:
    def test_redesign_markers_present(self):
        html = build_prospects_body(False)
        for marker in (
            "otc-tier-divider-line",  # tier breaks reuse player-rankings CSS
            "rk-mock-h",              # mock draft column header
            "rkHeadshot",             # headshot discs
            "rkRadarSVG",             # radar chart
            "rkLetterGrade",          # letter grades
            "brUndoToast",            # shared undo toast for the watchlist
            "m-adv-grid",             # advanced metrics section
            "m-season-table",         # season-by-season table
            "/api/prospects/player/",       # detail bundle endpoint
            "/api/prospects/comparables/",  # historical comparables endpoint
        ):
            assert marker in html, marker

    def test_no_value_or_fantasy_adp(self):
        html = build_prospects_body(False)
        assert '<option value="value">' not in html
        assert '<option value="adp">' not in html
        assert '<option value="mock">' in html
        assert "rkSettingsPanel" not in html
