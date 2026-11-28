"""Tests for prospect model v3.8: the under-rating fixes.

Kaedon's order (2026-11-24): "fix all of that" - the M1-M4 mechanisms
from the under-rating audit (~/workspace/prospect-runs/
underrating-audit-2026-11-24.md and m1-m3-followup-2026-11-24.md),
with M2 strengthened on his later 2026-11-24 directive (the grade is
a future outlook, not a career rank: "not performing young in
college shouldn't hurt them"):

- M1: placeholder (all-NULL) seasons are dropped at assembly and in
  score_prospect; season legs with no measurable evidence return
  None (missing-leg redistribution) instead of scoring 0-filled
  dicts; rates derive from counting stats, never from friendly
  defaults; a GP-missing exact 0.0 is missing, not measured; QB
  breakout without team totals is None; competition without any
  schedule data is None; the reference pool carries a data_quality
  flag and 'truncated' rows are quarantined out of percentile pools.
- M2: the production blend leans harder on best-season evidence
  (90/10, was 85/15), AND the role gate: seasons before the player's
  first real role (position opportunity floors in _is_role_season)
  are excused from the production / utilization reads and from
  career aggregates inside them (the M3 share). The excuse is
  directional: once featured, later decline counts in full; a
  never-featured player grades on everything he has.
- M3: bounded RB receiving-share credit inside production
  (career share over role seasons, cap 10).
- M4: the age leg's per-year penalty slope halves beyond 23.0.

Post-gate round (Kaedon 2026-11-24: "fix the TE problem first,
then re-gate"; the first gate read TE Pearson .391 / Spearman .319
vs the 3.7 control's .407 / .384):

- Breakout is NOT role-gated after all: it is a change measure whose
  origin is the pre-role baseline, and gating it deleted exactly
  the rise it prices (elite TEs lost 35.0 points). The gate covers
  the level legs (production, utilization) only.
- The GP-missing production zero gets a verdict, not a blanket
  conversion: the zero stands unless the same evidence prices at or
  above neutral (52) at the GP=4 minimum-sample divisor, or the
  file ends 2+ seasons before the draft class (truncated file).
- The M1 None-exits key on RECORDED fields, not volume > 0: a
  recorded zero is a measurement (a QB-to-TE convert really caught
  zero passes); only never-recorded fields are missing.
- When 3+ of the 4 performance legs are missing, the grade anchors
  them at neutral 52 instead of redistributing their weight onto
  context legs (the Swoopes float).
- M4 was REVERTED before ship: held against the same gate, the
  flattened slope cost correlation at QB/WR/TE. The 3.7 slope
  stands and TestAgeSlope pins it.
- Becker/utilization: a WR/TE season whose targets were never
  recorded reads its receptions on a receptions scale, not the
  targets scale (the RB branch's v3.3 treatment), so the proxy no
  longer manufactures a 0.0 baseline under the in-season shrink.

Synthetic fixtures only; no DB, no CFBD.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import prospect_model as pm
from dashboard_services import rookie_api as ra

_spec = importlib.util.spec_from_file_location(
    "build_reference_population",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "build_reference_population.py"))
refpop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refpop)


PHANTOM = {"season": 2027}  # every stat field NULL, as the eval
# pipeline writes draft-year placeholder rows

WR_REAL = {
    "season": 2026, "games_played": 12, "team": "LSU",
    "conference": "SEC", "receiving_yards": 1100, "receptions": 70,
    "receiving_tds": 9, "targets": 105, "yds_per_reception": 15.7,
    "dominator_rating": 0.32, "team_pass_yards": 3400,
    "team_total_yards": 5200,
}


def _prospect(pos, seasons, age=21.0, **kw):
    p = {"player_id": "T38", "position": pos, "age": age,
         "draft_class_year": 2027, "seasons": seasons}
    p.update(kw)
    return p


class TestVersion:
    def test_model_version_is_current(self):
        assert pm.MODEL_VERSION == "3.14"


class TestPhantomSeasonDrop:
    def test_phantom_does_not_change_the_grade(self):
        with_phantom = pm.score_prospect(
            _prospect("WR", [dict(WR_REAL), dict(PHANTOM)]),
            skip_sagarin=True)
        without = pm.score_prospect(
            _prospect("WR", [dict(WR_REAL)]), skip_sagarin=True)
        assert with_phantom["prospect_score"] == without["prospect_score"]
        assert with_phantom["production_score"] == without["production_score"]
        assert with_phantom["data_quality"] == "ok"

    def test_assembly_drops_phantoms(self):
        # The drop lives at profile assembly in the DB loader (the
        # layer that reads the stored rows), not in score_prospect.
        from data_building.rookie_pipeline.pipeline import (
            _drop_placeholder_seasons)
        kept = _drop_placeholder_seasons([dict(WR_REAL), dict(PHANTOM)])
        assert kept == [WR_REAL]
        assert _drop_placeholder_seasons([dict(PHANTOM)]) == []
        assert _drop_placeholder_seasons([]) == []

    def test_real_season_present_flags_ok(self):
        r = pm.score_prospect(_prospect("WR", [dict(WR_REAL)]),
                              skip_sagarin=True)
        assert r["data_quality"] == "ok"

    def test_phantom_only_prospect_is_flagged_truncated(self):
        r = pm.score_prospect(_prospect("WR", [dict(PHANTOM)]),
                              skip_sagarin=True)
        assert r["data_quality"] == "truncated"
        # The empty-list neutral path still produces a grade; the
        # competition leg has no schedule data at all and is None.
        assert r["competition_score"] is None
        assert r["prospect_score"] is not None

    def test_no_seasons_at_all_is_flagged_truncated(self):
        r = pm.score_prospect(_prospect("RB", []), skip_sagarin=True)
        assert r["data_quality"] == "truncated"

    def test_season_has_counting_stats(self):
        assert not pm.season_has_counting_stats({"season": 2027})
        assert not pm.season_has_counting_stats(
            {"season": 2027, "receiving_yards": None, "targets": None})
        assert pm.season_has_counting_stats(
            {"season": 2026, "rush_attempts": 3})


class TestNoneExits:
    def test_production_none_without_relevant_volume(self):
        # A WR whose only season is a wildcat row: counting stats
        # exist (not a phantom) but no receiving volume at all. He
        # was never featured, so the role gate excuses nothing and
        # the legs see the row - and report it missing, not zero.
        seasons = [{"season": 2025, "games_played": 10,
                    "rush_attempts": 3, "rush_yards": 12,
                    "conference": "SEC", "team": "LSU"}]
        assert pm.calc_production_score(seasons, "WR") is None
        assert pm.calc_efficiency_score(seasons, "WR") is None
        assert pm.calc_breakout_score(seasons, 21.0, "WR") is None
        r = pm.score_prospect(_prospect("WR", seasons), skip_sagarin=True)
        assert r["production_score"] is None
        assert r["efficiency_score"] is None
        assert r["breakout_profile_score"] is None
        # The grade still computes from the surviving legs.
        assert isinstance(r["prospect_score"], float)

    def test_utilization_none_without_opportunity_stats(self):
        seasons = [{"season": 2025, "games_played": 10,
                    "receiving_yards": 300, "conference": "SEC",
                    "team": "LSU"}]
        # receiving yards but no targets and no receptions recorded:
        # production has volume, utilization has nothing to read.
        assert pm.calc_utilization_score(seasons, "WR") is None

    def test_empty_seasons_keep_neutral_paths(self):
        assert pm.calc_production_score([], "WR") == 52.0
        # 3.14 re-pin: the WR efficiency wrapper prices the
        # toll at the latest season; with no seasons the
        # neutral 52.0 reads at the default level
        # (52.0 x 0.9143 = 47.55), the lab-measured form.
        assert pm.calc_efficiency_score([], "WR") == pytest.approx(47.55, abs=0.01)
        assert pm.calc_breakout_score([], 21.0, "WR") == 52.0
        assert pm.calc_utilization_score([], "WR") == 50.0


class TestRatesFromCountingStats:
    def test_qb_efficiency_derives_rates(self):
        season = {"season": 2025, "games_played": 12,
                  "pass_attempts": 200, "completions": 130,
                  "pass_yards": 1700, "pass_tds": 12, "interceptions": 4}
        derived = pm.calc_efficiency_score([season], "QB")
        # YPA 8.5, comp 65.0%, TD:INT 3.0 - the same season with the
        # rates stored must score identically.
        stored = dict(season, yds_per_attempt=8.5, completion_pct=65.0,
                      td_int_ratio=3.0)
        assert derived == pm.calc_efficiency_score([stored], "QB")
        assert derived is not None and derived > 0

    def test_qb_efficiency_partial_evidence_renormalizes(self):
        # Attempts + yards only: YPA alone carries the leg.
        # 3.14 re-pin: the season's value carries its
        # competition toll (conference-less fixture, default
        # level).
        season = {"season": 2025, "games_played": 12,
                  "pass_attempts": 200, "pass_yards": 1900}
        eff = pm.calc_efficiency_score([season], "QB")
        assert eff == pytest.approx(
            pm._scale(9.5, 6.5, 9.5)
            * pm._season_competition_factor(season))

    def test_rb_efficiency_derives_ypc(self):
        # 3.14 re-pin: per-season toll as above.
        season = {"season": 2025, "games_played": 12,
                  "rush_attempts": 100, "rush_yards": 550}
        eff = pm.calc_efficiency_score([season], "RB")
        assert eff == pytest.approx(
            pm._scale(5.5, 3.5, 7.0)
            * pm._season_competition_factor(season))

    def test_no_rate_evidence_is_none_not_defaults(self):
        # Volume exists for production (rushing) but the WR efficiency
        # leg reads receiving rates: none derivable -> None, never
        # the old friendly-default template score.
        season = {"season": 2025, "games_played": 12,
                  "rush_attempts": 40, "rush_yards": 200,
                  "receiving_yards": 0, "receptions": 0}
        assert pm.calc_efficiency_score([season], "WR") is None


class TestGamesPlayedHandling:
    QB_BELOW_FLOORS = {
        "season": 2025, "pass_yards": 1650, "pass_tds": 12,
        "pass_attempts": 210, "completions": 130, "interceptions": 5,
        "team_total_yards": 4100, "conference": "Big Ten",
        "team": "Penn State",
    }

    def test_gp_missing_exact_zero_is_missing(self):
        # 1650 yards / defaulted 12 GP = 137.5/g, under the 150
        # floor; 12 TDs = 1.0/g, under the 1.5 floor: the old code
        # emitted an unflagged exact 0.0 (audit class L2).
        assert pm.calc_production_score(
            [dict(self.QB_BELOW_FLOORS)], "QB") is None

    def test_gp_known_exact_zero_is_measured(self):
        season = dict(self.QB_BELOW_FLOORS, games_played=12)
        assert pm.calc_production_score([season], "QB") == 0.0


class TestQbBreakoutNone:
    def test_missing_team_totals_is_none(self):
        season = {"season": 2025, "games_played": 12,
                  "pass_yards": 2400, "pass_attempts": 300,
                  "conference": "SEC", "team": "LSU"}
        assert pm.calc_breakout_score([season], 22.0, "QB") is None

    def test_team_totals_present_scores(self):
        season = {"season": 2025, "games_played": 12,
                  "pass_yards": 2400, "pass_attempts": 300,
                  "team_total_yards": 4800,
                  "conference": "SEC", "team": "LSU"}
        out = pm.calc_breakout_score([season], 22.0, "QB")
        assert isinstance(out, float)


class TestCompetitionNone:
    def test_no_schedule_data_is_none(self):
        seasons = [{"season": 2025, "games_played": 12,
                    "receiving_yards": 500, "receptions": 40,
                    "team": "Nowhere State"}]
        assert pm.calc_competition_score(seasons) is None
        assert pm.calc_competition_score([]) is None

    def test_conference_present_scores(self):
        seasons = [{"season": 2025, "team": "LSU", "conference": "SEC"}]
        out = pm.calc_competition_score(seasons)
        assert isinstance(out, float) and out > 90  # career SEC ~95

    def test_notre_dame_counts_as_schedule_data(self):
        seasons = [{"season": 2025, "team": "Notre Dame"}]
        assert pm.calc_competition_score(seasons) is not None


class TestReceivingShareCredit:
    def test_bounds(self):
        assert pm._rb_receiving_share_credit(
            [{"rush_yards": 0, "receiving_yards": 500}]) == 10.0
        assert pm._rb_receiving_share_credit(
            [{"rush_yards": 600, "receiving_yards": 400}]) == 10.0
        assert pm._rb_receiving_share_credit(
            [{"rush_yards": 800, "receiving_yards": 200}]) == 0.0
        assert pm._rb_receiving_share_credit(
            [{"rush_yards": 1000, "receiving_yards": 0}]) == -10.0
        assert pm._rb_receiving_share_credit(
            [{"rush_yards": 0, "receiving_yards": 0}]) == 0.0

    def test_credit_is_career_share(self):
        seasons = [{"rush_yards": 900, "receiving_yards": 100},
                   {"rush_yards": 100, "receiving_yards": 300}]
        # Career share = 400 / 1400 = 0.286 -> +4.29, not the mean of
        # the two season credits.
        assert pm._rb_receiving_share_credit(seasons) == pytest.approx(
            10.0 * (0.286 - 0.20) / 0.20, abs=0.1)

    def test_credit_moves_the_leg(self):
        base = {"season": 2025, "games_played": 12, "rush_attempts": 200,
                "rush_yards": 900, "rush_tds": 8, "receptions": 20,
                "receiving_yards": 100, "receiving_tds": 1,
                "dominator_rating": 0.25, "yds_per_carry": 4.5,
                "conference": "SEC", "team": "Georgia"}
        catcher = dict(base, receptions=60, receiving_yards=500)
        low = pm.calc_production_score([base], "RB")
        high = pm.calc_production_score([catcher], "RB")
        assert high > low


class TestAgeSlope:
    # M4 (slope halved beyond 23.0) shipped in the first 3.8 build
    # and was reverted in the post-gate round: the gate arbitrated
    # against it at QB/WR/TE. These tests pin the 3.7 slope.
    def test_below_23_untouched(self):
        assert pm.calc_age_score(21.0, 2027, "WR") == 100.0
        assert pm.calc_age_score(23.0, 2027, "WR") == pytest.approx(55.56,
                                                                   abs=0.1)

    def test_slope_stays_linear_beyond_23(self):
        # Ages kept above the F11 sub-20 tail, where the decline is
        # deliberately 5 points/year in age space instead.
        s22 = pm.calc_age_score(22.0, 2027, "WR")
        s23 = pm.calc_age_score(23.0, 2027, "WR")
        s24 = pm.calc_age_score(24.0, 2027, "WR")
        assert (s23 - s24) == pytest.approx(s22 - s23, abs=0.01)

    def test_tail_still_separates(self):
        assert pm.calc_age_score(26.5, 2027, "WR") > 10.0
        assert (pm.calc_age_score(25.5, 2027, "WR")
                > pm.calc_age_score(26.5, 2027, "WR"))


class TestBestSeasonBlend:
    def test_blend_is_90_10(self):
        older = {"season": 2024, "games_played": 12, "team": "LSU",
                 "conference": "SEC", "receiving_yards": 1300,
                 "receptions": 85, "receiving_tds": 11, "targets": 120,
                 "yds_per_reception": 15.3, "dominator_rating": 0.36,
                 "team_pass_yards": 3600, "team_total_yards": 5400}
        latest = {"season": 2025, "games_played": 12, "team": "LSU",
                  "conference": "SEC", "receiving_yards": 700,
                  "receptions": 50, "receiving_tds": 5, "targets": 80,
                  "yds_per_reception": 14.0, "dominator_rating": 0.22,
                  "team_pass_yards": 3300, "team_total_yards": 5100}
        seasons = [older, latest]
        best = pm._score_production_season(older, "WR")
        lat = pm._score_production_season(latest, "WR")
        expected = max(best, lat) * 0.90 + min(best, lat) * 0.10
        assert pm.calc_production_score(seasons, "WR") == pytest.approx(
            expected)


WR_PRE_ROLE = {
    "season": 2024, "games_played": 12, "team": "LSU",
    "conference": "SEC", "receiving_yards": 70, "receptions": 6,
    "receiving_tds": 0, "targets": 10, "dominator_rating": 0.03,
    "team_pass_yards": 3200, "team_total_yards": 5000,
}


class TestRoleGate:
    def test_role_season_floors(self):
        assert not pm._is_role_season(dict(WR_PRE_ROLE), "WR")
        assert pm._is_role_season(dict(WR_REAL), "WR")
        # QB: mop-up attempts are not a role; a featured wildcat is.
        assert not pm._is_role_season(
            {"pass_attempts": 40, "rush_attempts": 10}, "QB")
        assert pm._is_role_season({"pass_attempts": 120}, "QB")
        assert pm._is_role_season({"rush_attempts": 90}, "QB")
        # RB: carries + catches.
        assert not pm._is_role_season(
            {"rush_attempts": 30, "receptions": 5}, "RB")
        assert pm._is_role_season(
            {"rush_attempts": 45, "receptions": 8}, "RB")
        # TE floor is lower than WR.
        assert pm._is_role_season({"receptions": 11}, "TE")
        assert not pm._is_role_season({"receptions": 11}, "WR")

    def test_dominator_override(self):
        # Raw counts clear the WR floor, but a 0.02 dominator with
        # real team totals on file is garbage-time accumulation.
        s = {"receptions": 20, "receiving_yards": 190, "targets": 34,
             "dominator_rating": 0.02, "team_pass_yards": 3000}
        assert not pm._is_role_season(s, "WR")
        # Without team totals the override cannot fire.
        s2 = dict(s)
        s2.pop("team_pass_yards")
        assert pm._is_role_season(s2, "WR")

    def test_gate_drops_only_pre_featured_seasons(self):
        featured = dict(WR_REAL)  # 2026
        down_year = dict(WR_REAL, season=2027, receiving_yards=420,
                         receptions=34, targets=52,
                         dominator_rating=0.13)
        seasons = [dict(WR_PRE_ROLE), featured, down_year]
        gated = pm._role_gated_seasons(seasons, "WR")
        # The pre-role 2024 season is excused; the post-featured
        # down year (a real role, worse results) counts in full.
        assert [s["season"] for s in gated] == [2026, 2027]

    def test_never_featured_keeps_everything(self):
        seasons = [dict(WR_PRE_ROLE),
                   dict(WR_PRE_ROLE, season=2025, receptions=9,
                        targets=14)]
        assert pm._role_gated_seasons(seasons, "WR") == seasons

    def test_inseason_baseline_ignores_pre_role_seasons(self):
        # The Becker shape: a low-role completed season and an
        # in-progress featured season. Ungated, the in-progress
        # production/utilization shrink toward the pre-role
        # completed baseline; gated, there is no completed role
        # season and the baseline is the leg neutral.
        featured_partial = dict(WR_REAL, season=2026, games_played=6,
                                receiving_yards=620, receptions=38,
                                targets=55)
        seasons = [dict(WR_PRE_ROLE, season=2025), featured_partial]
        gated = pm.score_prospect(_prospect("WR", seasons),
                                  skip_sagarin=True,
                                  _active_season_year=2026)
        orig = pm._role_gated_seasons
        pm._role_gated_seasons = lambda ss, pp: list(ss)
        try:
            ungated = pm.score_prospect(_prospect("WR", seasons),
                                        skip_sagarin=True,
                                        _active_season_year=2026)
        finally:
            pm._role_gated_seasons = orig
        assert gated["production_score"] > ungated["production_score"]
        assert gated["utilization_score"] > ungated["utilization_score"]

    def test_m3_share_computed_over_role_seasons(self):
        pre = {"season": 2024, "games_played": 12, "rush_attempts": 30,
               "rush_yards": 180, "receiving_yards": 0, "receptions": 2,
               "conference": "MAC", "team": "Akron"}
        featured = {"season": 2025, "games_played": 12,
                    "rush_attempts": 200, "rush_yards": 900,
                    "rush_tds": 8, "receptions": 60,
                    "receiving_yards": 500, "receiving_tds": 3,
                    "dominator_rating": 0.30, "yds_per_carry": 4.5,
                    "conference": "MAC", "team": "Akron"}
        seasons = [pre, featured]
        gated = pm._role_gated_seasons(seasons, "RB")
        assert gated == [featured]
        # Career share over role seasons only: 500/1400 = .357,
        # not 500/1580 = .316 with the pre-role season diluting it.
        assert pm._rb_receiving_share_credit(gated) > (
            pm._rb_receiving_share_credit(seasons))
        r = pm.score_prospect(_prospect("RB", seasons),
                              skip_sagarin=True)
        # The share the production leg reads is the gated one (the
        # identity this test pins). v3.9 (competition discount):
        # score_prospect then priced the raw leg against the level
        # it was achieved at. 3.14 re-pin: the toll prices per
        # season INSIDE the leg, so the identity is now direct -
        # the leg calc_production_score returns IS the priced
        # leg score_prospect reports - and the toll binds
        # visibly against the same seasons read at an SEC level
        # (factor 1.0).
        _raw = pm.calc_production_score(gated, "RB", skip_sagarin=True)
        _untolled = pm.calc_production_score(
            [dict(s, conference="SEC") for s in gated], "RB",
            skip_sagarin=True)
        assert _raw < _untolled
        assert r["production_score"] == pytest.approx(
            _raw, abs=0.05)


class TestPoolScreen:
    def test_no_seasons_truncated(self):
        assert refpop._data_quality("WR", []) == "truncated"
        assert refpop._data_quality("WR", [{"season": 2020}]) == "truncated"

    def test_stray_season_truncated(self):
        seasons = [{"season": 2014, "rush_attempts": 2, "rush_yards": 5,
                    "dominator_rating": 0.01}]
        assert refpop._data_quality("RB", seasons) == "truncated"

    def test_rule_f_severe_truncated(self):
        seasons = [
            {"season": 2018, "rush_yards": 184, "receiving_yards": 11,
             "rush_attempts": 40},
            {"season": 2019, "rush_yards": 16, "receiving_yards": 10,
             "rush_attempts": 6},
        ]
        assert refpop._data_quality("RB", seasons) == "truncated"

    def test_real_career_ok(self):
        seasons = [
            {"season": 2021, "rush_yards": 1127, "receiving_yards": 295,
             "rush_attempts": 188, "dominator_rating": 0.31},
            {"season": 2022, "rush_yards": 1578, "receiving_yards": 314,
             "rush_attempts": 258, "dominator_rating": 0.35},
        ]
        assert refpop._data_quality("RB", seasons) == "ok"

    def test_rule_f_with_real_volume_ok(self):
        # Dominator missing everywhere but a full career of volume:
        # the FCS hole only flags fragment-sized careers.
        seasons = [
            {"season": 2021, "rush_yards": 1200, "receiving_yards": 200,
             "rush_attempts": 210},
            {"season": 2022, "rush_yards": 1300, "receiving_yards": 250,
             "rush_attempts": 230},
        ]
        assert refpop._data_quality("RB", seasons) == "ok"

    def test_build_reference_rows_stamps_quality(self):
        prospects = [{
            "player_id": "p1", "name": "Frag Ment", "position": "RB",
            "age": 22.0, "athleticism": {},
            "seasons": [{"season": 2019, "rush_attempts": 40,
                         "rush_yards": 184, "receiving_yards": 11}],
        }]
        scores = [{"player_id": "p1", "position": "RB",
                   "production_score": 30.0, "efficiency_score": 25.0,
                   "breakout_profile_score": 20.0,
                   "competition_score": 46.0, "athleticism_score": None}]
        draft_class = [{"gsis_id": "p1", "name": "Frag Ment",
                        "draft_pick": 200}]
        rows = refpop.build_reference_rows(2020, draft_class, prospects,
                                           scores, "3.8")
        assert len(rows) == 1
        assert rows[0]["data_quality"] == "truncated"

    def test_advanced_marks_floor_drops_fragment_seasons(self):
        real = {"season": 2019, "games_played": 12, "rush_yards": 960,
                "receiving_yards": 240, "dominator_rating": 0.30,
                "yds_per_carry": 5.0}
        fragment = {"season": 2018, "games_played": 1, "rush_yards": 195,
                    "receiving_yards": 0, "dominator_rating": 0.05,
                    "yds_per_carry": 9.0}
        marks_row = {"age": 22.0, "weight_lbs": 210,
                     "efficiency_score": 50.0, "production_score": 50.0}
        marks = refpop._advanced_marks("RB", [real, fragment], {},
                                       marks_row)
        # The 1-game fragment's 195.0 Scrim Yds/Gm must not appear;
        # the real season's 100.0 does.
        assert marks.get("Scrim Yds/Gm") == pytest.approx(100.0)


def _ref_row(i, quality="ok", base=40.0):
    return {
        "player_name": f"Ref {i}", "draft_class_year": 2020,
        "production_score": base + i, "efficiency_score": base + i,
        "breakout_profile_score": base + i,
        "competition_score": base + i, "athleticism_score": None,
        "tested": False, "advanced_marks": {}, "season_rates": [],
        "testing_marks": {}, "data_quality": quality,
    }


class TestQuarantine:
    def test_truncated_rows_excluded_from_pool(self):
        refs = [_ref_row(i) for i in range(35)]
        refs += [_ref_row(100 + i, quality="truncated", base=95.0)
                 for i in range(5)]
        pop = ra.reference_population("WR", ref_rows=refs)
        assert pop["n"] == 35
        assert max(pop["values"]["competition"]) == 40.0 + 34

    def test_unflagged_rows_still_count(self):
        refs = [_ref_row(i) for i in range(35)]
        for r in refs:
            r.pop("data_quality")
        pop = ra.reference_population("WR", ref_rows=refs)
        assert pop["n"] == 35

    def test_truncated_rows_own_percentiles_hide(self):
        refs = [_ref_row(i) for i in range(35)]
        pop = ra.reference_population("WR", ref_rows=refs)
        row = _ref_row(3, quality="truncated")
        row["position"] = "WR"
        out = ra.component_percentiles(row, pop, seasons=[])
        assert out and all(v is None for v in out.values())

    def test_ok_rows_percentiles_still_compute(self):
        refs = [_ref_row(i) for i in range(35)]
        pop = ra.reference_population("WR", ref_rows=refs)
        row = _ref_row(30)
        row["position"] = "WR"
        out = ra.component_percentiles(row, pop, seasons=[])
        assert out["competition"] is not None


class TestPostGateFixes:
    """The TE fix + Becker/utilization round (Kaedon 2026-11-24:
    'fix the TE problem first, then re-gate')."""

    MCBRIDE = [
        {"season": 2023, "games_played": 12, "team": "Colorado State",
         "conference": "MWC", "receptions": 8, "receiving_yards": 82,
         "receiving_tds": 0, "targets": 12, "dominator_rating": 0.03,
         "team_pass_yards": 3000, "team_total_yards": 4600},
        {"season": 2024, "games_played": 12, "team": "Colorado State",
         "conference": "MWC", "receptions": 70, "receiving_yards": 850,
         "receiving_tds": 8, "targets": 95, "dominator_rating": 0.29,
         "team_pass_yards": 3100, "team_total_yards": 4700},
        {"season": 2025, "games_played": 12, "team": "Colorado State",
         "conference": "MWC", "receptions": 62, "receiving_yards": 720,
         "receiving_tds": 6, "targets": 88, "dominator_rating": 0.24,
         "team_pass_yards": 3200, "team_total_yards": 4800},
    ]

    def test_breakout_trajectory_survives_the_role_gate(self):
        # The cameo first season is pre-role (the gate drops it from
        # the level legs), but breakout's trajectory is growth FROM
        # that baseline: score_prospect must read the ungated list,
        # or the leg loses the 35-point trajectory component. The
        # fixture mirrors the real profile that exposed this (peak in
        # the middle season, slight per-game decline after): gated,
        # the trajectory re-bases onto the peak season and collapses
        # toward zero; ungated, the cameo-to-peak growth prices 100.
        seasons = [dict(s) for s in self.MCBRIDE]
        r = pm.score_prospect(
            _prospect("TE", seasons, draft_class_year=2026),
            skip_sagarin=True)
        # v3.10: score_prospect feeds the draft year into the leg
        # (the breakout-age anchor), so the direct calls supply it
        # too - the comparison below isolates the role gate, which
        # is what this test pins.
        ungated = pm.calc_breakout_score(seasons, 21.0, "TE",
                                         _draft_year=2026)
        gated = pm.calc_breakout_score(
            pm._role_gated_seasons(seasons, "TE"), 21.0, "TE",
            _draft_year=2026)
        assert r["breakout_profile_score"] == pytest.approx(ungated,
                                                            abs=0.01)
        # v3.9 (arrival term): the gated read no longer collapses.
        # The leg prices the first FEATURED season's level as well
        # as the rise, so a hypothetical gated read - whose first
        # season is the peak itself - prices the arrival instead of
        # re-basing growth onto the peak. The gap that was ~35
        # (trajectory 100 -> ~0 at its 0.35 weight) is now ~6 (the
        # growth price vs the arrival price of the same season).
        # The mechanism pin: with the arrival term off, the old
        # collapse returns.
        assert ungated - gated < 12.0
        _cfg = pm.BREAKOUT_CFG
        pm.BREAKOUT_CFG = {**_cfg, "arrival": False}
        try:
            ungated_no_arr = pm.calc_breakout_score(
                seasons, 21.0, "TE", _draft_year=2026)
            gated_no_arr = pm.calc_breakout_score(
                pm._role_gated_seasons(seasons, "TE"), 21.0, "TE",
                _draft_year=2026)
        finally:
            pm.BREAKOUT_CFG = _cfg
        assert ungated_no_arr - gated_no_arr > 20.0

    RADER = [
        {"season": 2016, "receptions": 4, "receiving_yards": 73,
         "receiving_tds": 0, "team": "Youngstown State",
         "conference": "MVFC"},
        {"season": 2017, "receptions": 4, "receiving_yards": 53,
         "receiving_tds": 0, "team": "Youngstown State",
         "conference": "MVFC"},
    ]

    def test_thin_resume_zero_stands(self):
        # GP missing, but the recorded evidence is negligible at ANY
        # divisor (126 career yards): the 0.0 is a measurement.
        assert pm.calc_production_score(
            [dict(s) for s in self.RADER], "TE") == 0.0

    def test_truncated_file_zero_is_missing(self):
        # The same thin file, but it ends two seasons before the
        # draft class (the Goedert shape: the deciding seasons were
        # never loaded) - the leg measured nothing, so it is None.
        r = pm.score_prospect(
            _prospect("TE", [dict(s) for s in self.RADER],
                      draft_class_year=2019),
            skip_sagarin=True)
        assert r["production_score"] is None
        # Ending the season before the draft (the normal shape) the
        # zero stands.
        r2 = pm.score_prospect(
            _prospect("TE", [dict(s) for s in self.RADER],
                      draft_class_year=2018),
            skip_sagarin=True)
        assert r2["production_score"] == 0.0

    SWOOPES = [
        {"season": 2015, "games_played": 13, "receptions": 0,
         "receiving_yards": 0, "receiving_tds": 0, "rush_attempts": 74,
         "rush_yards": 451, "pass_attempts": 93, "pass_yards": 537,
         "team": "Texas", "conference": "Big 12"},
        {"season": 2016, "games_played": 12, "receptions": 0,
         "receiving_yards": 0, "receiving_tds": 0, "rush_attempts": 54,
         "rush_yards": 174, "pass_attempts": 9, "pass_yards": 66,
         "team": "Texas", "conference": "Big 12"},
    ]

    def test_recorded_zeros_are_measurements(self):
        # A QB filed at TE: his receiving fields are RECORDED zeros.
        # Production and utilization price at 0.0 - they do not go
        # missing and float the grade on context.
        # v3.13 re-pin (Phase 7, L2): the TE production terms are
        # ALL-PURPOSE now, so his recorded RUSHING prices: 451
        # yards in 13 games (34.7/g) reads 4.96 on the yards term
        # in 2015 and the blend lands at 4.46. Every RECEIVING
        # term still prices exactly 0.0, and utilization is
        # unchanged - the zeros are still measurements, never
        # missing.
        seasons = [dict(s) for s in self.SWOOPES]
        assert pm.calc_production_score(
            seasons, "TE") == pytest.approx(4.46, abs=0.05)
        assert pm.calc_utilization_score(seasons, "TE") == 0.0

    def test_multi_none_anchor(self):
        # A TE profile whose only recorded stats are QB stats: all
        # four performance legs are missing. The grade must anchor
        # them at neutral instead of pricing the player purely on
        # age/competition context (which alone would read ~85).
        seasons = [{"season": 2025, "games_played": 12,
                    "pass_attempts": 300, "pass_yards": 2400,
                    "pass_tds": 18, "rush_attempts": 80,
                    "rush_yards": 400, "team": "LSU",
                    "conference": "SEC"}]
        r = pm.score_prospect(_prospect("TE", seasons, age=21.0),
                              skip_sagarin=True)
        assert r["production_score"] is None
        assert r["utilization_score"] is None
        assert r["efficiency_score"] is None
        assert r["breakout_profile_score"] is None
        assert r["prospect_score"] < 70.0

    def test_wr_proxy_reads_receptions_scale(self):
        # Targets never recorded; 34 rec / 14 GP = 2.43 rec/g read
        # on the receptions scale (1.6-6.5), not the targets scale
        # (2.5-10.0) where it floored at exactly 0.0.
        season = {"season": 2025, "games_played": 14, "receptions": 34,
                  "receiving_yards": 679, "receiving_tds": 4,
                  "team": "Indiana", "conference": "Big Ten"}
        assert pm._utilization_raw([season], "WR") == pytest.approx(
            16.9, abs=0.3)

    def test_te_proxy_reads_receptions_scale(self):
        season = {"season": 2025, "games_played": 12, "receptions": 20,
                  "receiving_yards": 240, "team": "LSU",
                  "conference": "SEC"}
        assert pm._utilization_raw([season], "TE") == pytest.approx(
            pm._scale(20 / 12, 1.0, 5.2), abs=0.01)

    def test_becker_shaped_inseason_utilization(self):
        # The Becker shape: in-progress season with targets (32/6 =
        # 5.33 tpg -> 37.8 raw) shrinking toward a completed season
        # whose targets were never recorded. Pre-fix the proxy read
        # the completed season as 0.0 and the shrink halved the leg
        # to 18.89; on the receptions scale the baseline is ~16.9
        # and the leg lands ~27.3.
        completed = {"season": 2025, "games_played": 14,
                     "receptions": 34, "receiving_yards": 679,
                     "receiving_tds": 4, "team": "Indiana",
                     "conference": "Big Ten"}
        current = {"season": 2026, "games_played": 6, "receptions": 30,
                   "receiving_yards": 529, "receiving_tds": 8,
                   "targets": 32, "team": "Indiana",
                   "conference": "Big Ten"}
        out = pm.calc_utilization_score([completed, current], "WR",
                                        _inseason_year=2026)
        assert out == pytest.approx(27.3, abs=0.6)
        assert out > 22.0
