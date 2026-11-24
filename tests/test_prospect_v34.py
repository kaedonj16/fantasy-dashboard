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
        # v3.6 (round-2 audit F15): the floor is DELETED. Complete mode
        # no longer imputes 67.0 to untested elite producers; the
        # athleticism weight redistributes, as in-season already did.
        out = _score(SMITH, _dc("WR", 2))
        assert out["athleticism_score"] is None
        assert out["athleticism_projected"] is False
        # Round 14 sim printed 95.16 here; +0.51 is the red-zone
        # recalibration firing on his 2026 season (11 TD / 916 yards =
        # 1.20 per 100, over the new 1.15 low tier).
        # v3.6 de-saturation: 95.67 -> 93.26. His pinned legs un-pin:
        # competition 100 -> 89.66 (career Big Ten on the 1.03 anchor),
        # efficiency 100 -> 94.58, production 96.56 -> 91.43 (widened
        # WR anchors). Breakout still saturates at the 1.7x multiple.
        # v3.6 round-2 F5: -> 93.57. WR efficiency drops market share
        # and prices his elite YAC: efficiency 94.58 -> 97.48.
        # v3.6 round-2 F9: -> 92.90. Growth is priced once in breakout
        # (totals switch + ascending bonus deleted): breakout
        # 96.06 -> 88.97.
        # v3.6 round-2 F15: -> 94.95. The deleted floor was DRAGGING
        # him: the imputed 67.0 sat far below his real legs, so
        # redistributing the athleticism weight raises the grade.
        # v3.7: -> 92.48. Capital-free: the grade is the 3.6 talent
        # grade of the same profile. The delta is exactly the removed
        # capital contribution: 3.6 printed capital_delta +2.47 for
        # this call (pick-2 curve value at the redistributed WR
        # capital weight, net of renormalization).
        # v3.7 F12 weights fold: -> 91.93. Weights change only (the
        # gate-validated fitted WR set replaces the design weights;
        # legs unchanged): his two biggest legs ease in effective
        # share (production .2622 -> .2382, efficiency .1311 ->
        # .1162) while breakout, competition and utilization rise;
        # net -0.55 on his leg mix.
        # v3.8 (M2): -> 92.28. Production blend only: the best-season
        # weight rises 85 -> 90, his production leg 91.43 -> 92.55;
        # every other leg is byte-identical to 3.7 on this fixture
        # (the role gate excuses no season here).
        assert out["prospect_score"] == 92.28

    def test_inseason_suppresses_floor_and_redistributes(self):
        out = _score(SMITH, _dc("WR", 2), _active_season_year=2026)
        assert out["athleticism_score"] is None
        assert out["athleticism_projected"] is False
        # Round 14 sim printed 98.09; the same +0.56 red-zone delta as
        # complete mode (plus its peak-selection interaction) applies.
        # v3.6 de-saturation: 98.65 -> 95.99 (same un-pinned legs as
        # complete mode: competition, efficiency, production).
        # v3.6 round-2 F5: -> 96.32 (same efficiency move as complete).
        # v3.6 round-2 F2: -> 92.93. His 2026 partial season (7 games)
        # now shrinks toward his completed-season baseline in
        # production, efficiency and utilization.
        # v3.6 round-2 F9: -> 91.80 (same breakout consolidation).
        # v3.7: -> 87.79 (3.6 capital_delta +4.01 removed, exactly).
        # v3.7 F12 weights fold: -> 87.71 (weights change only; the
        # in-season leg mix sits closer to the fitted shares' centre
        # than complete mode's, so the same re-weighting costs only
        # -0.08 here).
        # v3.8 (M2): -> 87.85 (production 87.07 -> 87.52 on the
        # 90/10 blend; no other leg moves).
        assert out["prospect_score"] == 87.85
        # GP 7 confidence discount: round(20 * (1 - 7/12)) = 8.
        # v3.7: 62.0 -> 47.0. The 15-point drop is the removed
        # capital-presence confidence (+10 for a mock-backed pick,
        # +5 at 5+ mocks); the GP discount itself is unchanged
        # (55 pre-discount - 8 = 47).
        assert out["confidence_score"] == 47.0

    def test_unbound_profile_is_untouched_by_the_toggle(self):
        # Mestemaker was never floor-bound (production under the 90
        # gate): the mode changes his confidence only, not his grade.
        complete = _score(MESTEMAKER, _dc("QB", 8))
        inseason = _score(MESTEMAKER, _dc("QB", 8), _active_season_year=2026)
        # v3.6: 79.80 -> 79.53 in both modes; the only mover is his
        # competition score (69.09 -> 65.52) under the 1.03 top anchor.
        # v3.6 round-2 F3/F6: -> 78.03. His fixture carries 20+ rush
        # yds/game, which translation used to price as a flat +1.5;
        # rushing is now priced once, inside production.
        # v3.6 round-2 F6 (cont.): the modes now differ (78.51 complete
        # / 79.37 in-season). The volume-only production composite
        # re-picks his peak season under the in-season paced key; the
        # old equality was fixture luck under the rate-trio composite.
        # The floor claim this test guards is unaffected: his
        # athleticism is unbound in both modes (see F15 tests).
        # v3.6 round-2 F10/F16: -> 81.60 / 81.15. His fixture has no
        # recruiting data: the 5% weight now redistributes instead of
        # scoring an imputed 50, and the QB experience add-on is
        # priced from the redistributed weights (F16) rather than the
        # raw table.
        # v3.7: -> 80.18 / 79.53. Capital-free: the deltas are exactly
        # 3.6's printed capital_deltas (+1.42 complete / +1.62
        # in-season) removed. Confidences drop the same 15
        # capital-presence points in both modes (60 -> 45, 48 -> 33),
        # so the mode difference the toggle guards is unchanged.
        # v3.7 F12 weights fold: -> 84.69 / 83.96. Weights change
        # only; the driver is QB age's fitted share (effective
        # .1009 -> .2012 - age was the fit's strongest QB leg,
        # r=.463) priced on a young, high-production profile.
        # v3.8 (M2): -> 84.78 / 84.00. The production blend is shared
        # across positions, so the 90/10 best-season lean moves his
        # production leg too (91.32 -> 91.65 / 88.05 -> 88.19); no
        # other leg moves. (Role gate: his 2-game backup season is
        # excused from the gated reads; it moved no leg.)
        assert complete["prospect_score"] == 84.78
        assert inseason["prospect_score"] == 84.00
        assert complete["confidence_score"] == 45.0
        assert inseason["confidence_score"] == 33.0  # GP 5: discount 12


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
        # v3.6 de-saturation: 78.38 / 78.44 -> 77.88 / 77.94.
        # Competition 100 -> 94.83 (career SEC on the 1.03 anchor),
        # breakout 84.80 -> 83.95 (TE saturation 1.8 -> 2.4),
        # production 38.99 -> 38.42 (TE TDs/g anchor 0.5 -> 0.65).
        # v3.6 round-2 F4: 77.88 / 77.94 -> 76.21 / 76.27. His pick 35
        # is round 2: the capital component loses the x1.08 round
        # multiplier (82.80 -> 76.67, the raw curve value) and the
        # grade moves by exactly that x his redistributed capital
        # weight.
        # v3.6 round-2 F3: -> 74.21 / 74.27. The deleted TE pick tiers
        # paid him +2.0 at pick 35; translation is now 0 for TEs.
        # v3.6 round-2 F5: -> 76.51 / 76.57. TE efficiency drops market
        # share; with targets absent his leg is yards per reception
        # alone and his 14.0+ yds/rec sits at the anchor: efficiency
        # 81.52 -> 100.0. (The share term was masking the pin; the TE
        # yds/rec anchor itself is a gate decision, not changed here.)
        # v3.6 round-2 F9: complete -> 75.16. His breakout falls
        # exactly 15.00 (83.95 -> 68.95): the deleted ascending bonus
        # was paying him its full cap.
        # v3.6 round-2 F10: -> 76.09 / 75.31. Green's fixture has no
        # recruiting data; the imputed flat 50 becomes a redistribution
        # over his (stronger) present legs.
        # v3.7: -> 75.88 / 74.80. Decomposition at THIS pinned clock:
        # the capital leg's true price was +0.21 complete / +0.51
        # in-season (76.09 - 75.88, 75.31 - 74.80, measured against
        # the 3.6 no-capital run at the same _today). 3.6's printed
        # capital_deltas (+0.33 / +0.69) overstated it by 0.12 / 0.18
        # because the internal talent recursion dropped the caller's
        # _today pin and graded durability on the run-date calendar -
        # a mixed-clock artifact. 3.7 has a single run, so the pin is
        # the clock-consistent talent grade by construction.
        # v3.7 F12 weights fold: -> 74.81 / 73.74. Weights change
        # only: TE efficiency's effective share eases (.1354 ->
        # .1187) against his pinned 100.0 leg, partly offset by
        # utilization's rise (.0861 -> .1332); net -1.07 / -1.06.
        # v3.8 (M2): -> 74.92 / 73.77 (production 42.55 -> 42.90 /
        # 43.33 -> 43.41 on the 90/10 blend; no other leg moves).
        assert complete["prospect_score"] == 74.92
        assert inseason["prospect_score"] == 73.77
        # All of Green's seasons are role seasons (13+ receptions):
        # the durability role rule does not touch him.
        assert complete["durability_score"] == 69.44

    def test_mestemaker_durability_repair_grade_unchanged(self):
        out = _score(MESTEMAKER, _dc("QB", 8), _today=self._AS_OF)
        # The pin is the current model's grade (v3.7: 84.69 after
        # the F12 weights fold; the durability repair itself still
        # contributes nothing. Path: 79.80 competition anchor ->
        # 78.03 F3/F6 translation cleanup -> 78.51 F6 volume-only
        # production composite (production 83.98 -> 91.32) -> 81.60
        # F10 recruiting redistribution + F16 experience repricing,
        # see the toggle test above) -> 80.18 in v3.7 (the +1.42
        # capital_delta removed exactly) -> 84.69 under the fitted
        # weights (QB age's doubled effective share, same step as
        # the toggle test) -> 84.78 in v3.8 (M2 production blend,
        # same step as the toggle test).
        assert out["prospect_score"] == 84.78
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
        # Report-card modal: the Metric Breakdown renders stored-stat
        # rows only (raw value present, component rows excluded);
        # athleticism is never an advanced row, so a missing or
        # redistributed athleticism score can never surface there. A
        # row whose pool cannot support a percentile shows the empty
        # bar, never a fabricated number.
        from dashboard_services.pages.rookies_page import build_prospects_body
        body = build_prospects_body()
        assert "a.raw != null && !a.component" in body
        assert "rc-mbar empty" in body
