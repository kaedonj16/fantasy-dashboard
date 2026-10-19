"""Tests for prospect model v3.3.

Covers the v3.3 package (iteration log rounds 8-13):
- F0: the inverted three_cone / short_shuttle _scale calls that scored a
  constant 50.0 for every player are replaced by knot tiers that price
  lower-is-better legs correctly.
- Knot-anchored combine subscales: a p75-tier RB forty (4.45) prices 77 raw.
- The three-zone athleticism transform in score_prospect: identity at raw
  <= 60, slope 0.70 from 60 to 85, slope 0.90 above 85 (raw 100 -> 91.0),
  applied class-wide (the v3.2 tested-QB-only compression is gone).
- V-C: RB utilization without target data reads receptions per game on a
  receptions scale (0.5 to 3.5), not the targets scale (1.0 to 5.0).
- V-A: breakout trajectory reads totals growth when the per-game rate is
  maintained at >= 90%.
- F2: WR efficiency market-share anchor 0.30 (was 0.40).
- F3: WR breakout dominator saturation at 1.4x threshold (was the shared
  linear curve); RB keeps the algebraically identical 1.8x curve.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import prospect_model as pm
from data_building.rookie_pipeline.mock_draft_consensus import (
    pick_to_draft_capital_score,
)


def _dc(pos, pick):
    return {
        "projected_pick": pick,
        "projected_round": ((pick - 1) // 32) + 1,
        "projected_draft_capital_score": pick_to_draft_capital_score(pick, pos),
        "num_mocks_used": 5,
        "is_actual_pick": False,
    }


def _rb_prospect(athleticism):
    return {
        "player_id": "T_v33_rb", "name": "V33 RB", "position": "RB",
        "draft_class_year": 2026, "age": 21.5,
        "seasons": [
            {"season": 2024, "games_played": 12, "team": "State",
             "conference": "SEC", "rush_attempts": 180, "rush_yards": 950,
             "rush_tds": 10, "receptions": 25, "receiving_yards": 210,
             "receiving_tds": 2, "yds_per_carry": 5.28,
             "yds_per_reception": 8.4, "dominator_rating": 0.30,
             "market_share_yards": 0.28, "team_pass_yards": 3200,
             "team_total_yards": 5600},
            {"season": 2025, "games_played": 13, "team": "State",
             "conference": "SEC", "rush_attempts": 210, "rush_yards": 1180,
             "rush_tds": 13, "receptions": 32, "receiving_yards": 280,
             "receiving_tds": 3, "yds_per_carry": 5.62,
             "yds_per_reception": 8.75, "dominator_rating": 0.33,
             "market_share_yards": 0.30, "team_pass_yards": 3400,
             "team_total_yards": 5900},
        ],
        "recruiting": {"composite_rating": 0.95},
        "athleticism": athleticism,
    }


def _qb_prospect(athleticism):
    return {
        "player_id": "T_v33_qb", "name": "V33 QB", "position": "QB",
        "draft_class_year": 2026, "age": 22.0,
        "seasons": [
            {"season": 2025, "games_played": 13, "team": "State",
             "conference": "SEC", "pass_attempts": 420,
             "completion_pct": 67.0, "pass_yards": 3600, "pass_tds": 30,
             "td_int_ratio": 3.0, "yds_per_attempt": 8.57,
             "rush_yards": 300, "rush_tds": 4, "team_total_yards": 6100},
        ],
        "recruiting": {"composite_rating": 0.97},
        "athleticism": athleticism,
    }


class TestVersion:
    def test_model_version_is_33(self):
        assert pm.MODEL_VERSION == "3.4"


class TestKnotPricing:
    def test_p75_rb_forty_prices_77_raw(self):
        # 4.45 is the recorded p75 mark for drafted RBs; round 11 prices
        # the p75 tier at 77 (Bijan's 4.46 prices ~73.5 on the same scale).
        assert pm._knot_score(4.45, "RB", "forty") == pytest.approx(77.0)
        assert pm._knot_score(4.46, "RB", "forty") == pytest.approx(73.5)

    def test_knot_endpoints_and_median(self):
        assert pm._knot_score(4.24, "RB", "forty") == pytest.approx(100.0)
        assert pm._knot_score(4.83, "RB", "forty") == pytest.approx(0.0)
        assert pm._knot_score(4.51, "RB", "forty") == pytest.approx(56.0)
        # Higher-is-better leg prices the same tier structure.
        assert pm._knot_score(37.0, "RB", "vertical") == pytest.approx(77.0)
        assert pm._knot_score(35.0, "RB", "vertical") == pytest.approx(56.0)

    def test_marks_are_position_specific(self):
        # The same 4.45 forty is the p75 mark for an RB (77) but sits
        # between the WR p50 (4.47) and p75 (4.41) marks, pricing 63.
        assert pm._knot_score(4.45, "WR", "forty") == pytest.approx(63.0)
        assert pm._knot_score(4.45, "WR", "forty") < pm._knot_score(4.45, "RB", "forty")


class TestF0AgilityInversion:
    def test_cone_and_shuttle_price_lower_is_better(self):
        assert pm._knot_score(6.50, "RB", "cone") == pytest.approx(100.0)
        assert pm._knot_score(7.68, "RB", "cone") == pytest.approx(0.0)
        assert pm._knot_score(6.89, "RB", "cone") == pytest.approx(77.0)
        assert pm._knot_score(3.93, "RB", "shuttle") == pytest.approx(100.0)
        assert pm._knot_score(4.67, "RB", "shuttle") == pytest.approx(0.0)

    def test_agility_no_longer_constant_50(self):
        # The F0 bug signature: every player scored exactly 50.0 on these
        # legs. Elite vs poor agility must now separate (single metric, so
        # the sparse-data cap of 74 binds the elite raw 100).
        elite = pm.calc_athleticism_score({"three_cone": 6.50}, "RB")
        poor = pm.calc_athleticism_score({"three_cone": 7.68}, "RB")
        assert elite == pytest.approx(74.0)
        assert poor == pytest.approx(0.0)
        assert elite != 50.0 and poor != 50.0
        elite_s = pm.calc_athleticism_score({"short_shuttle": 3.93}, "RB")
        poor_s = pm.calc_athleticism_score({"short_shuttle": 4.67}, "RB")
        assert elite_s == pytest.approx(74.0)
        assert poor_s == pytest.approx(0.0)


class TestTransformZones:
    # Uniform-tier legs (no weight, so no speed leg): five metrics at one
    # tier make the raw component exactly that tier's score.
    def _component(self, ath):
        out = pm.score_prospect(
            copy.deepcopy(_rb_prospect(ath)), _dc("RB", 10), skip_sagarin=True)
        return out["athleticism_score"]

    def test_identity_zone_raw_at_or_below_60(self):
        p25 = {"forty_yard": 4.58, "vertical_inches": 33.0,
               "broad_jump_in": 117, "three_cone": 7.15,
               "short_shuttle": 4.348}
        assert pm.calc_athleticism_score(p25, "RB") == pytest.approx(38.0)
        assert self._component(p25) == pytest.approx(38.0)
        p50 = {"forty_yard": 4.51, "vertical_inches": 35.0,
               "broad_jump_in": 120, "three_cone": 7.01,
               "short_shuttle": 4.235}
        assert pm.calc_athleticism_score(p50, "RB") == pytest.approx(56.0)
        assert self._component(p50) == pytest.approx(56.0)

    def test_confirmation_zone_slope_070(self):
        p75 = {"forty_yard": 4.45, "vertical_inches": 37.0,
               "broad_jump_in": 123.75, "three_cone": 6.89,
               "short_shuttle": 4.143}
        assert pm.calc_athleticism_score(p75, "RB") == pytest.approx(77.0)
        # 60 + (77 - 60) * 0.70 = 71.9
        assert self._component(p75) == pytest.approx(71.9)

    def test_exceptional_toe_raw_100_lands_91(self):
        best = {"forty_yard": 4.24, "vertical_inches": 43.0,
                "broad_jump_in": 137, "three_cone": 6.50,
                "short_shuttle": 3.93}
        assert pm.calc_athleticism_score(best, "RB") == pytest.approx(100.0)
        # 77.5 + (100 - 85) * 0.90 = 91.0
        assert self._component(best) == pytest.approx(91.0)

    def test_qb_no_longer_capped_by_old_compression(self):
        # v3.2 R7-A3 capped a tested QB's component at 62.5 (quarter of the
        # distance from 50). The class-wide transform supersedes it: a
        # record-tier tested QB now lands at the same 91.0 toe as anyone.
        best_qb = {"forty_yard": 4.33, "vertical_inches": 40.5,
                   "broad_jump_in": 129, "three_cone": 6.65,
                   "short_shuttle": 3.87}
        out = pm.score_prospect(
            copy.deepcopy(_qb_prospect(best_qb)), _dc("QB", 1),
            skip_sagarin=True)
        assert out["athleticism_score"] == pytest.approx(91.0)


class TestVCReceptionsScale:
    def _util(self, season):
        return pm.calc_utilization_score([season], "RB")

    def test_missing_targets_reads_receptions_scale(self):
        # 15 carries/g (rush_util 50) and 2.0 receptions/g.
        base = {"season": 2025, "games_played": 10, "rush_attempts": 150,
                "receptions": 20}
        no_tgts = self._util({**base, "targets": 0})
        with_tgts = self._util({**base, "targets": 20})
        # No targets: receptions scale _scale(2.0, 0.5, 3.5) = 50
        #   -> 50 * 0.55 + 50 * 0.45 = 50.0
        assert no_tgts == pytest.approx(50.0)
        # Targets present: targets scale _scale(2.0, 1.0, 5.0) = 25
        #   -> 50 * 0.55 + 25 * 0.45 = 38.75
        assert with_tgts == pytest.approx(38.75)
        # The defect's signature was the missing-data back scoring LOWER.
        assert no_tgts > with_tgts


class TestVATrajectoryTotals:
    def _seasons(self, gp1, yds1, gp2, yds2):
        return [
            {"season": 2024, "games_played": gp1, "rush_yards": yds1,
             "receiving_yards": 0, "pass_yards": 0,
             "dominator_rating": 0.30},
            {"season": 2025, "games_played": gp2, "rush_yards": yds2,
             "receiving_yards": 0, "pass_yards": 0,
             "dominator_rating": 0.30},
        ]

    def test_maintained_rate_reads_totals_growth(self):
        # 100 yds/g over 7 games, then 95 yds/g over 13 games: per-game
        # rate dips 5% but totals grow 76%. V-A scores trajectory on the
        # totals growth (traj 100); the per-game-only reading scores 0
        # when the rate falls 20% (profile B, same dominator/age path).
        a = pm.calc_breakout_score(self._seasons(7, 700, 13, 1235), 22.0, "RB")
        b = pm.calc_breakout_score(self._seasons(7, 700, 13, 1040), 22.0, "RB")
        assert a - b > 30.0  # 0.35 traj weight x (100 - 0)

    def test_maintained_rate_with_shrinking_totals_scores_low(self):
        # Rate maintained (95 vs 100 yds/g) but the workload shrank:
        # totals growth is negative, so V-A must not credit growth.
        # (Draft age 22 keeps both seasons above the underclassman floor's
        # age gate, so only the trajectory term separates the profiles.)
        a = pm.calc_breakout_score(self._seasons(7, 700, 13, 1235), 22.0, "RB")
        c = pm.calc_breakout_score(self._seasons(13, 1300, 7, 665), 22.0, "RB")
        assert a - c > 20.0


class TestF2EfficiencyAnchor:
    def _eff(self, ms):
        seasons = [{"season": 2025, "games_played": 12,
                    "yds_per_reception": 17.0, "market_share_yards": ms}]
        return pm.calc_efficiency_score(seasons, "WR")

    def test_ms_030_saturates_the_leg(self):
        # ypr leg is 100 (17.0); with the 0.30 anchor the MS leg is also
        # 100 at a 0.30 share, so efficiency is 100. The old 0.40 anchor
        # priced the same season's MS leg at 68.75 (eff 84.4).
        assert self._eff(0.30) == pytest.approx(100.0)
        # Half-way share prices half-way on the new anchor.
        assert self._eff(0.30) - self._eff(0.19) == pytest.approx(25.0)


class TestF3BreakoutSaturation:
    def _wr_breakout(self, dom):
        seasons = [
            {"season": 2024, "games_played": 12, "receiving_yards": 900,
             "rush_yards": 0, "pass_yards": 0, "dominator_rating": dom},
            {"season": 2025, "games_played": 12, "receiving_yards": 900,
             "rush_yards": 0, "pass_yards": 0, "dominator_rating": dom},
        ]
        return pm.calc_breakout_score(seasons, 21.0, "WR")

    def test_wr_saturates_at_035_dominator(self):
        # dom_score(0.35) = 100 under F3 (ratio 1.4 = saturation), and
        # dom_score(0.30) = 80; the blend difference also carries the
        # dominance-factor difference on the age term: ~11.3 points total.
        # The old shared curve would give ~6.8.
        diff = self._wr_breakout(0.35) - self._wr_breakout(0.30)
        assert diff == pytest.approx(11.3, abs=0.6)
