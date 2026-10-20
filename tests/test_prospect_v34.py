"""Tests for prospect model v3.4: two-mode grading, the durability
role rule, and the WR/TE red-zone proxy recalibration.

In-season mode (round 14 of the iteration log): a player grades
in-season iff the run supplies an active season year, his latest
season is that year, and its stored GP is 1-11. Complete mode is the
v3.3 path. The durability role rule applies in BOTH modes: a sub-12-GP
season counts as missed time only when the player held a real role in
it (usage thresholds in prospect_model._season_has_role), so backup
seasons no longer read as injury seasons (v3.3 priced Sayin's backup
season into a 44.07 durability and Mestemaker's into 6.01).

Panel pins below run the surviving repo fixtures (SMITH / GREEN /
MESTEMAKER from test_prospect_v32) through the real module. Where a
pin differs from the round 14 simulation numbers (Smith 95.16 /
98.09, Green 77.97 / 78.26), the difference is the red-zone
recalibration and is called out per test: the proxy's old thresholds
(TDs per 100 receiving yards of 5.5-10.0) could never fire against a
recorded distribution that tops out at 1.62.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import prospect_model as pm
from data_building.rookie_pipeline.mock_draft_consensus import (
    pick_to_draft_capital_score,
)
from test_prospect_v32 import GREEN, MESTEMAKER, SMITH


def _dc(pos, pick):
    return {
        "projected_pick": pick,
        "projected_round": ((pick - 1) // 32) + 1,
        "projected_draft_capital_score": pick_to_draft_capital_score(pick, pos),
        "num_mocks_used": 5,
        "is_actual_pick": False,
    }


def _score(prospect, dc, **kw):
    return pm.score_prospect(copy.deepcopy(prospect), dc, skip_sagarin=True, **kw)


class TestActiveSeasonYear:
    def test_fall_window(self):
        assert pm.active_season_year_for_run(date(2026, 8, 25)) == 2026
        assert pm.active_season_year_for_run(date(2026, 10, 17)) == 2026
        assert pm.active_season_year_for_run(date(2026, 12, 31)) == 2026

    def test_january_tail_is_prior_season(self):
        assert pm.active_season_year_for_run(date(2027, 1, 15)) == 2026
        assert pm.active_season_year_for_run(date(2027, 1, 20)) == 2026
        assert pm.active_season_year_for_run(date(2027, 1, 21)) is None

    def test_offseason_is_none(self):
        assert pm.active_season_year_for_run(date(2026, 8, 24)) is None
        assert pm.active_season_year_for_run(date(2027, 5, 10)) is None
        assert pm.active_season_year_for_run(date(2027, 2, 1)) is None


class TestModeSwitch:
    def test_latest_partial_season_in_active_year_is_inseason(self):
        out = _score(SMITH, _dc("WR", 2), _active_season_year=2026)
        assert out["in_season"] is True
        assert out["grading_mode"] == "in_season"

    def test_no_active_year_is_complete(self):
        out = _score(SMITH, _dc("WR", 2))
        assert out["in_season"] is False
        assert out["grading_mode"] == "complete"

    def test_switch_is_context_driven_not_player_driven(self):
        # Same profile: an active year that is NOT his latest season
        # grades complete; his latest season's year flips him in-season.
        p = {
            "player_id": "T_flip", "name": "Flip", "position": "QB",
            "draft_class_year": 2026, "age": 21.5,
            "seasons": [
                {"season": 2024, "games_played": 13, "pass_attempts": 380,
                 "completion_pct": 66.0, "pass_yards": 3300, "pass_tds": 28,
                 "td_int_ratio": 3.1, "yds_per_attempt": 8.7,
                 "rush_yards": 120, "rush_tds": 2},
                {"season": 2025, "games_played": 9, "pass_attempts": 300,
                 "completion_pct": 67.0, "pass_yards": 2700, "pass_tds": 22,
                 "td_int_ratio": 3.4, "yds_per_attempt": 9.0,
                 "rush_yards": 90, "rush_tds": 1},
            ],
            "recruiting": None, "athleticism": {},
        }
        complete = _score(p, _dc("QB", 10), _active_season_year=2026)
        inseason = _score(p, _dc("QB", 10), _active_season_year=2025)
        assert complete["in_season"] is False
        assert inseason["in_season"] is True
        # Confidence carries the partial-sample discount: GP 9 costs
        # round(20 * (1 - 9/12)) = 5 points.
        assert complete["confidence_score"] - inseason["confidence_score"] == 5.0

    def test_full_season_gp_grades_complete(self):
        p = copy.deepcopy(SMITH)
        p["seasons"][-1]["games_played"] = 12
        out = _score(p, _dc("WR", 2), _active_season_year=2026)
        assert out["in_season"] is False

    def test_missing_gp_grades_complete(self):
        p = copy.deepcopy(SMITH)
        p["seasons"][-1].pop("games_played", None)
        out = _score(p, _dc("WR", 2), _active_season_year=2026)
        assert out["in_season"] is False

    def test_zero_gp_grades_complete(self):
        p = copy.deepcopy(SMITH)
        p["seasons"][-1]["games_played"] = 0
        out = _score(p, _dc("WR", 2), _active_season_year=2026)
        assert out["in_season"] is False

    def test_score_all_threads_the_mode(self):
        prospects = [copy.deepcopy(SMITH), copy.deepcopy(MESTEMAKER)]
        consensus = {
            "T_smith": _dc("WR", 2),
            "T_mestemaker": _dc("QB", 8),
        }
        scored = pm.score_all_prospects(
            prospects, consensus, skip_sagarin=True, active_season_year=2026)
        assert all(s["in_season"] for s in scored)
        assert [s["overall_rank"] for s in scored] == [1, 2]
        scored = pm.score_all_prospects(
            [copy.deepcopy(SMITH), copy.deepcopy(MESTEMAKER)],
            consensus, skip_sagarin=True)
        assert not any(s["in_season"] for s in scored)


class TestFloorRedistribution:
    def test_complete_mode_keeps_the_projected_floor(self):
        out = _score(SMITH, _dc("WR", 2))
        assert out["athleticism_score"] == 67.0
        assert out["athleticism_projected"] is True
        # Round 14 sim printed 95.16 here; +0.51 is the red-zone
        # recalibration firing on his 2026 season (11 TD / 916 yards =
        # 1.20 per 100, over the new 1.15 low tier).
        assert out["prospect_score"] == 95.67

    def test_inseason_suppresses_floor_and_redistributes(self):
        out = _score(SMITH, _dc("WR", 2), _active_season_year=2026)
        assert out["athleticism_score"] is None
        assert out["athleticism_projected"] is False
        # Round 14 sim printed 98.09; the same +0.56 red-zone delta as
        # complete mode (plus its peak-selection interaction) applies.
        assert out["prospect_score"] == 98.65
        # GP 7 confidence discount: round(20 * (1 - 7/12)) = 8.
        assert out["confidence_score"] == 62.0

    def test_unbound_profile_is_untouched_by_the_toggle(self):
        # Mestemaker was never floor-bound (production under the 90
        # gate): the mode changes his confidence only, not his grade.
        complete = _score(MESTEMAKER, _dc("QB", 8))
        inseason = _score(MESTEMAKER, _dc("QB", 8), _active_season_year=2026)
        assert complete["prospect_score"] == inseason["prospect_score"] == 79.80
        assert complete["confidence_score"] == 60.0
        assert inseason["confidence_score"] == 48.0  # GP 5: discount 12


class TestPanelReproduction:
    # The pins in this class were recorded in college week 8
    # (Oct 13-19, 2026). Durability's in-progress denominator is the
    # calendar's current week, so an unpinned reproduction drifts every
    # week (week 9 priced Mestemaker's durability 70.37, not 75.0, and
    # Green 78.36, not 78.38). Pin the clock to the recording week.
    _AS_OF = date(2026, 10, 19)

    def test_green_pins(self):
        complete = _score(GREEN, _dc("TE", 35), _today=self._AS_OF)
        inseason = _score(GREEN, _dc("TE", 35), _active_season_year=2026,
                          _today=self._AS_OF)
        # Round 14 sim: 77.97 / 78.26. The deltas are the red-zone
        # recalibration: his 2025 season (7 TD / 433 yards = 1.62 per
        # 100) clears the new TE top tier, worth +0.41 complete and
        # +0.18 in-season after the peak blend.
        assert complete["prospect_score"] == 78.38
        assert inseason["prospect_score"] == 78.44
        # All of Green's seasons are role seasons (13+ receptions):
        # the durability role rule does not touch him.
        assert complete["durability_score"] == 69.44

    def test_mestemaker_durability_repair_grade_unchanged(self):
        out = _score(MESTEMAKER, _dc("QB", 8), _today=self._AS_OF)
        assert out["prospect_score"] == 79.80
        # v3.3 priced his 2-game, 41-attempt backup season as missed
        # time: durability 6.01. Excluded as a non-role season, his
        # durability is sane; QB durability weight is 0.00, so the
        # grade does not move.
        assert out["durability_score"] == 75.0


class TestDurabilityRoleRule:
    def test_backup_season_excluded_sayin_shape(self):
        with_backup = [
            {"season": 2023, "games_played": 4, "pass_attempts": 12},
            {"season": 2024, "games_played": 14, "pass_attempts": 391},
            {"season": 2025, "games_played": 6, "pass_attempts": 163},
        ]
        without_backup = with_backup[1:]
        d_with = pm.calc_durability_score(with_backup, position="QB")
        d_without = pm.calc_durability_score(without_backup, position="QB")
        assert d_with == d_without
        # The counted seasons: a full starter season (100) and a
        # 6-game starter season (50, minus the 10.8 missed-games
        # penalty) blend to 55.87. His 2025 here is a completed
        # season, so the penalty legitimately applies; the defect
        # was the backup season, which priced 44.07 in v3.3.
        assert d_with == pytest.approx(55.87, abs=0.01)

    def test_backup_season_excluded_mestemaker_shape(self):
        seasons = [
            {"season": 2023, "games_played": 2, "pass_attempts": 41},
            {"season": 2024, "games_played": 14, "pass_attempts": 463},
            {"season": 2025, "games_played": 5, "pass_attempts": 169},
        ]
        d = pm.calc_durability_score(seasons, position="QB")
        assert d == pm.calc_durability_score(seasons[1:], position="QB")
        assert d == pytest.approx(41.91, abs=0.01)
        assert d > 30.0  # v3.3 printed 6.01 on this shape

    def test_injured_starter_still_scores_low(self):
        # A real starter lost at midseason: both seasons are role
        # seasons (110+ touches), so nothing is forgiven.
        seasons = [
            {"season": 2024, "games_played": 13, "rush_attempts": 240,
             "receptions": 30},
            {"season": 2025, "games_played": 6, "rush_attempts": 105,
             "receptions": 14},
        ]
        d = pm.calc_durability_score(seasons, position="RB")
        assert d == pytest.approx(55.87, abs=0.01)
        assert d < 60.0
        # A mop-up season in the file does not rescue him either: it
        # is excluded, not averaged in as availability.
        with_mopup = [
            {"season": 2023, "games_played": 3, "rush_attempts": 10,
             "receptions": 1},
        ] + seasons
        assert pm.calc_durability_score(with_mopup, position="RB") == d

    def test_backup_only_profile_has_no_durability(self):
        seasons = [
            {"season": 2024, "games_played": 4, "pass_attempts": 12},
            {"season": 2025, "games_played": 5, "pass_attempts": 30},
        ]
        assert pm.calc_durability_score(seasons, position="QB") is None

    def test_full_season_counts_regardless_of_usage(self):
        # A blocking TE who dressed all 16 games was available all 16
        # games: the 12+ GP season counts even with 9 receptions.
        seasons = [
            {"season": 2024, "games_played": 16, "receptions": 9},
            {"season": 2025, "games_played": 6, "receptions": 30,
             "targets": 45},
        ]
        d = pm.calc_durability_score(seasons, position="TE")
        # (100 at w .5 + 50 at w 1.0) / 1.5 - 10.8 penalty = 55.87;
        # had the 16-game season been excluded this would be 39.2.
        assert d == pytest.approx(55.87, abs=0.01)

    def test_role_thresholds(self):
        assert pm._season_has_role({"pass_attempts": 99}, "QB")
        assert not pm._season_has_role({"pass_attempts": 41}, "QB")
        assert pm._season_has_role(
            {"rush_attempts": 100, "receptions": 15}, "RB")
        assert not pm._season_has_role(
            {"rush_attempts": 39, "receptions": 4}, "RB")
        assert pm._season_has_role({"targets": 42, "receptions": 29}, "WR")
        assert not pm._season_has_role({"receptions": 9}, "WR")
        assert pm._season_has_role({"receptions": 13}, "TE")
        assert not pm._season_has_role({"receptions": 9}, "TE")


class TestDurabilityInSeasonVariant:
    def test_nonrole_partial_season_excluded(self):
        # Backup work in the in-progress season: excluded like any
        # non-role season, leaving the full starter season at 100.
        seasons = [
            {"season": 2025, "games_played": 14, "pass_attempts": 391},
            {"season": 2026, "games_played": 5, "pass_attempts": 30},
        ]
        d = pm.calc_durability_score(seasons, position="QB",
                                     _inseason_year=2026)
        assert d == 100.0

    def test_partial_season_never_triggers_missed_games_penalty(self):
        # January context: the partial season is a past calendar year,
        # so it lands in the completed branch; the mode flag must
        # still keep the missed-games penalty off it.
        seasons = [
            {"season": 2024, "games_played": 14, "pass_attempts": 391},
            {"season": 2025, "games_played": 6, "pass_attempts": 163},
        ]
        d = pm.calc_durability_score(seasons, position="QB",
                                     _inseason_year=2025)
        # (100 at w .5 + 50 at w 1.0) / 1.5, no penalty = 66.67
        # (the same seasons in complete mode price 55.87).
        assert d == pytest.approx(66.67, abs=0.01)

    def test_healthy_starter_not_charged_for_byes(self):
        today = date.today()
        season_start = date(today.year, 8, 25)
        if today < season_start or today.month > 12:
            pytest.skip("outside the fall window for the calendar rule")
        current_week = min((today - season_start).days // 7 + 1, 16)
        if current_week < 4:
            pytest.skip("too early in the season for the bye floor")
        year = today.year
        seasons = [
            {"season": year - 1, "games_played": 14, "pass_attempts": 391},
            # Played every team game; the team has had its byes.
            {"season": year, "games_played": current_week - 1,
             "pass_attempts": 240},
        ]
        d = pm.calc_durability_score(seasons, position="QB",
                                     _inseason_year=year)
        assert d == 100.0

    def test_real_absences_in_partial_season_count(self):
        today = date.today()
        season_start = date(today.year, 8, 25)
        if today < season_start or today.month > 12:
            pytest.skip("outside the fall window for the calendar rule")
        current_week = min((today - season_start).days // 7 + 1, 16)
        if current_week < 7:
            pytest.skip("too early in the season to separate byes from absences")
        year = today.year
        gp = current_week - 5  # at least 3 absences beyond the 2 byes
        seasons = [
            {"season": year - 1, "games_played": 14, "pass_attempts": 391},
            {"season": year, "games_played": gp, "pass_attempts": 99},
        ]
        d_in = pm.calc_durability_score(seasons, position="QB",
                                        _inseason_year=year)
        d_complete = pm.calc_durability_score(seasons, position="QB")
        assert d_in is not None and d_in < 100.0
        # Counted, but against the bye-adjusted floor, so it reads
        # strictly better than the raw calendar denominator.
        assert d_in > d_complete


class TestRedZoneRecalibration:
    def _wr(self, tds, yards=1000):
        return {"games_played": 12, "receiving_yards": yards,
                "receiving_tds": tds, "dominator_rating": 0.25}

    def _te(self, tds, yards=800):
        return {"games_played": 12, "receiving_yards": yards,
                "receiving_tds": tds, "receptions": 60,
                "dominator_rating": 0.15}

    def test_wr_low_tier_fires_at_recorded_top_quarter(self):
        # 12 TD / 1000 yards = 1.20 per 100: fires the 1.15 tier.
        # 11 TD = 1.10: does not. The TD-per-game leg saturates for
        # both, so the ratio isolates the proxy multiplier exactly.
        p11 = pm._score_production_season(self._wr(11), "WR", True)
        p12 = pm._score_production_season(self._wr(12), "WR", True)
        assert p12 == pytest.approx(p11 * 1.03, rel=0.002)

    def test_wr_top_tier_fires_at_recorded_elite_edge(self):
        # 15 TD / 1000 yards = 1.50 per 100 (Pitts 2020 scored 1.56).
        p12 = pm._score_production_season(self._wr(12), "WR", True)
        p15 = pm._score_production_season(self._wr(15), "WR", True)
        assert p15 == pytest.approx(p12 / 1.03 * 1.06, rel=0.002)

    def test_wr_ordinary_rate_does_not_fire(self):
        # 0.90 per 100 is a normal scoring season (Smith 2025: 0.97);
        # no proxy credit. It must score below the 1.10 no-fire case
        # only by its TD leg, with no multiplier on either.
        p9 = pm._score_production_season(self._wr(9), "WR", True)
        p11 = pm._score_production_season(self._wr(11), "WR", True)
        assert p9 < p11
        p10 = pm._score_production_season(self._wr(10), "WR", True)
        assert p10 < p11  # still no fire at 1.00

    def test_te_tiers_fire(self):
        # 10 TD / 800 yards = 1.25 per 100 fires the TE low tier
        # (x1.02); 9 TD = 1.125 does not. TD leg saturates for both.
        p9 = pm._score_production_season(self._te(9), "TE", True)
        p10 = pm._score_production_season(self._te(10), "TE", True)
        assert p10 == pytest.approx(p9 * 1.02, rel=0.002)
        # 12 TD = 1.50 clears the TE top tier (x1.05).
        p12 = pm._score_production_season(self._te(12), "TE", True)
        assert p12 == pytest.approx(p9 * 1.05, rel=0.002)

    def test_old_thresholds_would_never_fire(self):
        # The best recorded rate in the fixture + panel set is 1.62
        # (Green 2025). Even it sat far below the old 5.5/7.0 floors.
        green25 = {"games_played": 11, "receiving_yards": 433,
                   "receiving_tds": 7, "receptions": 33,
                   "dominator_rating": 0.11, "team_pass_yards": 3900}
        p = pm._score_production_season(green25, "TE", True)
        base = pm._score_production_season(
            {**green25, "receiving_tds": 4}, "TE", True)  # 0.92: no fire
        assert p > base  # fires now at the recalibrated top tier


class TestStorageAndDisplay:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _read(self, rel):
        with open(os.path.join(self.ROOT, rel), encoding="utf-8") as f:
            return f.read()

    def test_migration_049_adds_in_season(self):
        sql = self._read("migrations/049_rookie_in_season.sql")
        assert "rookie_rankings" in sql
        assert "in_season" in sql

    def test_upsert_writes_in_season(self):
        src = self._read("data_building/rookie_pipeline/pipeline.py")
        assert "%(in_season)s" in src
        assert "in_season                     = EXCLUDED.in_season" in src
        assert "rr.in_season" in src  # board query selects it

    def test_board_and_modal_show_the_tag(self):
        from dashboard_services.pages.rookies_page import build_prospects_body
        body = build_prospects_body()
        # Native board/modal grammar (PR #2693, merged before v3.4): the
        # board row tag uses .m-early and the modal chip uses .state-chip;
        # the pre-rework dense table's .rk-inseason class no longer exists.
        assert "m-early" in body
        assert "state-chip" in body
        assert "In-Season" in body

    def test_detail_carries_in_season(self):
        src = self._read("dashboard_services/rookie_api.py")
        assert 'detail["in_season"]' in src

    def test_breakdown_hides_missing_athleticism(self):
        # The modal breakdown renders the Athleticism row only when
        # the stored score is non-null; in-season redistribution keeps
        # it NULL, so the row (and the projected floor presentation)
        # never appears for an in-season grade.
        from dashboard_services.pages.rookies_page import build_prospects_body
        body = build_prospects_body()
        # Native modal (PR #2693): percentile rows render only for a
        # non-null percentile; the API returns a null athleticism
        # percentile for untested players, so the row is omitted. The
        # pre-rework modal's "if (r.athleticism_score != null)" guard is
        # superseded by this null-percentile skip in rkRenderPercentiles.
        assert "if (p == null) return;" in body
        assert "returns a null percentile otherwise, and the row is omitted" in body
