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
        assert _clip100(150) == 99.9
        assert _clip100(-5) == 0.0
        assert _clip100(None) is None
        assert _clip100("bad") is None

    def test_dominator_anchors_on_model_elite_mark(self):
        # The model treats >= 0.35 as an elite dominator.
        assert _grade_dominator(0.35) == 99.9
        assert _grade_dominator(0.31) == 88.6
        assert _grade_dominator(None) is None

    def test_breakout_age_display_scale(self):
        # Display grading: average (~20.5-21) reads as C, not F.
        assert _grade_breakout_age(19.1, "WR") == 95.6
        assert _grade_breakout_age(20.8, "WR") == 73.2
        assert _grade_breakout_age(20.8, "QB") == 83.6  # shifted scale
        assert _grade_breakout_age(None, "WR") is None

    def test_efficiency_grades(self):
        assert _grade_ypr(18.0) == 99.9
        assert _grade_ypc(7.0) == 99.9
        assert _grade_market_share(0.35) == 99.9
        assert _grade_speed_score(120.0) == 99.9
        assert _grade_cmp(70.0) == 99.9
        assert _grade_td_int(4.0) == 99.9
        assert _grade_ypa(9.5) == 99.9
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
                    "market_share_yards": 0.34, "market_share_tds": 0.28,
                    "yds_per_reception": 15.9}]
        adv = _build_advanced_metrics("WR", seasons, {"speed_score": 112}, self._row())
        assert [a["label"] for a in adv] == [
            "Dominator", "Breakout Age", "Yds/Rec",
            "Mkt Share", "TD Share", "Speed Score", "Efficiency", "Recruiting",
        ]
        assert adv[0]["raw"] == "38%"
        assert adv[0]["grade"] == 99.9  # 0.38 >= 0.35 elite anchor, capped below 100
        assert adv[2]["raw"] == "15.9"
        assert adv[5]["raw"] == "112"
        assert adv[4]["raw"] == "28%"
        assert adv[4]["grade"] == 93.3  # 0.28 / 0.30

    def test_te_uses_wr_set(self):
        adv = _build_advanced_metrics("TE", [], {}, self._row("TE"))
        assert [a["label"] for a in adv][2] == "Yds/Rec"

    def test_rb_metric_set(self):
        seasons = [{"season": 2025, "dominator_rating": 0.33,
                    "market_share_yards": 0.29, "yds_per_carry": 6.8,
                    "rush_yards": 1400, "receiving_yards": 300, "games_played": 12}]
        adv = _build_advanced_metrics("RB", seasons, {"speed_score": 118},
                                      self._row("RB"))
        assert [a["label"] for a in adv] == [
            "Dominator", "Breakout Age", "Yds/Carry", "Scrim Yds/Gm",
            "Mkt Share", "TD Share", "Speed Score", "Efficiency", "Recruiting",
        ]
        assert adv[2]["raw"] == "6.8"
        scrim = next(a for a in adv if a["label"] == "Scrim Yds/Gm")
        assert scrim["raw"] == "141.7"  # 1700 / 12
        assert scrim["grade"] == 99.9  # capped below 100

    def test_qb_metric_set(self):
        seasons = [{"season": 2025, "completion_pct": 67.2, "td_int_ratio": 3.1,
                    "yds_per_attempt": 8.9, "pass_yards": 3500, "pass_tds": 30,
                    "interceptions": 8, "pass_attempts": 400,
                    "team_total_yards": 5000}]
        adv = _build_advanced_metrics(
            "QB", seasons, {},
            {"position": "QB", "age": 21.1,
             "efficiency_score": 90, "production_score": 90})
        assert [a["label"] for a in adv] == [
            "Cmp%", "TD:INT", "Yds/Att", "AY/A",
            "Breakout Age", "Efficiency", "Production", "Recruiting",
        ]
        assert adv[0]["raw"] == "67%"
        assert adv[1]["raw"] == "3.1"
        aya = next(a for a in adv if a["label"] == "AY/A")
        assert aya["raw"] == "9.3"  # (3500 + 600 - 360) / 400
        assert aya["grade"] == 89.0  # 9.35 / 10.5

    def test_missing_data_degrades_to_nulls(self):
        adv = _build_advanced_metrics(
            "WR", [], {}, {"position": "WR", "age": None,
                           "efficiency_score": None, "production_score": None})
        assert len(adv) == 8
        assert all(a["raw"] is None and a["grade"] is None for a in adv)

    def test_speed_score_derived_from_forty_and_weight(self):
        # No official speed_score stored, but forty + weight exist.
        seasons = [{"season": 2025, "dominator_rating": 0.31,
                    "market_share_yards": 0.30, "yds_per_carry": 7.1}]
        adv = _build_advanced_metrics(
            "RB", seasons, {"forty_yard": 4.53},
            {"position": "RB", "age": 21.5, "weight_lbs": 195,
             "efficiency_score": 80, "production_score": 80})
        speed = next(a for a in adv if a["label"] == "Speed Score")
        assert speed["raw"] == "92.6"
        assert speed["grade"] == 77.2

    def test_recruiting_metric_present_when_scored(self):
        adv = _build_advanced_metrics(
            "WR", [], {},
            {"position": "WR", "age": None, "efficiency_score": None,
             "production_score": None, "recruiting_score": 85,
             "recruit_stars": 4})
        rec = next(a for a in adv if a["label"] == "Recruiting")
        assert rec["raw"] == "4-star"
        assert rec["grade"] == 85.0


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


class TestHeadshots:
    def test_espn_headshot_url_format(self):
        from data_building.rookie_pipeline.espn_scraper import _ESPN_HEADSHOT_URL
        url = _ESPN_HEADSHOT_URL.format(id="5079369")
        assert url == ("https://a.espncdn.com/i/headshots/college-football/"
                       "players/full/5079369.png")

    def test_fetch_espn_headshots_uses_cache(self):
        # get_player_age is module-cached: a headshot pass right after the
        # age pass costs zero extra HTTP.
        from data_building.rookie_pipeline import espn_scraper as es
        es._CACHE["jeremiyah love|notre dame|RB"] = {
            "player_name": "Jeremiyah Love", "espn_id": "1234567",
            "age": 21.5, "team": "Notre Dame", "position": "RB",
        }
        try:
            out = es.fetch_espn_headshots(
                ["Jeremiyah Love"], 2027,
                prospects_meta=[{"name": "Jeremiyah Love",
                                 "school": "Notre Dame", "position": "RB"}],
                delay=0,
            )
        finally:
            es._CACHE.pop("jeremiyah love|notre dame|RB", None)
        assert out["jeremiyah love"] == (
            "https://a.espncdn.com/i/headshots/college-football/"
            "players/full/1234567.png")


class TestProspectDraftClass:
    def test_turns_over_around_week_four(self):
        from datetime import date
        from data_building.rookie_pipeline import pipeline as pl
        assert pl.get_active_rookie_class(date(2026, 10, 9)) == 2027
        assert pl.get_active_rookie_class(date(2026, 9, 20)) == 2027
        assert pl.get_active_rookie_class(date(2026, 9, 19)) == 2026
        assert pl.get_active_rookie_class(date(2026, 9, 1)) == 2026
        assert pl.get_active_rookie_class(date(2026, 8, 31)) == 2026
        assert pl.get_active_rookie_class(date(2027, 2, 15)) == 2027
        assert pl.get_active_rookie_class(date(2027, 5, 1)) == 2027
        assert pl.get_active_rookie_class(date(2026, 4, 1)) == 2026
