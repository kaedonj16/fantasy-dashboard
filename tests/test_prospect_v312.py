"""Tests for prospect model v3.12: the RB-2b reimagination.

Kaedon's directive (2026-11-25): "the rb grading needs
reimagined because all other pos are good." RB gets an
RB-specific architecture off the shared pipeline; QB/WR/TE are
byte-identical by construction. Design + prototype evidence:
prospect-runs/rb-reimagination-design-2026-11-25.md (S1-S7).

The seven build items pinned here:
1. Corroboration-priced release at RB only: the production
   evidence cap releases only when the second-best featured
   season itself prices >= 55 (Abanikanda's 28.50 stops
   counting; Javonte-shape 73.55 still releases). Other
   positions keep the presence release.
2. A direct receiving leg at weight 0.08, ASYMMETRIC under the
   pure-rusher doctrine ("a prospect can be great as just a
   pure rusher though"): below 50 the leg is absent from the
   priced list, so a pure rusher grades exactly as if the leg
   did not exist; a strong receiving profile is paid.
3. Athleticism re-anchored to the frozen RB pool percentile.
4. Efficiency trim-protected at RB (wiring: score_prospect adds
   it to the trim block).
5. A missing RB athleticism leg prices at the pool 40th
   percentile on the re-anchored scale (43.36) instead of
   redistributing.
6. Peak min-weight 0.02 at RB: a token-weight leg (durability
   ~0.006) cannot hold a peak slot.
7. Mixon is a named residual (two labs killed every honest
   mechanism); no peak-composition mechanism ships.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_building.rookie_pipeline import prospect_model as pm
from test_prospect_v32 import SMITH


class TestVersion:
    def test_model_version_is_313(self):
        # 3.13 re-pin (was test_model_version_is_312).
        assert pm.MODEL_VERSION == "3.13"

    def test_rb2b_cfgs(self):
        assert pm.PROD_EVIDENCE_CFG["release_min_price"] == 0.0
        # 3.13: TE joins RB on the corroboration release (L4)
        # and on the peak min-weight (L6).
        assert pm.PROD_EVIDENCE_CFG["release_min_price_by_pos"] == {
            "RB": 55.0, "TE": 55.0}
        assert pm.BLEND_CFG["peak_min_weight_by_pos"] == {
            "RB": 0.02, "TE": 0.02}
        assert pm.RB_RECEIVING_WEIGHT == pytest.approx(0.08)
        assert pm.RB_RECEIVING_MIN_PRICE == pytest.approx(50.0)
        assert pm.RB2B_FUND == {"utilization": -0.04, "age": -0.02,
                                "breakout": -0.02}


def _cap(monkeypatch, second_price, pos):
    """Production 90 with two featured seasons priced 95 / second."""
    if pos == "RB":
        seasons = [
            {"season": 2021, "rush_attempts": 120, "receptions": 5,
             "_price": 95.0},
            {"season": 2022, "rush_attempts": 300, "receptions": 20,
             "_price": second_price},
        ]
    else:
        seasons = [
            {"season": 2021, "targets": 60, "receptions": 40,
             "_price": 95.0},
            {"season": 2022, "targets": 120, "receptions": 80,
             "_price": second_price},
        ]
    monkeypatch.setattr(
        pm, "_score_production_season",
        lambda s, p, skip_sagarin=False: s["_price"])
    return pm._production_evidence_cap(
        90.0, seasons, pos, age=22.0, draft_year=2023,
        skip_sagarin=True)


class TestCorroborationRelease:
    def test_abanikanda_shape_stays_capped_at_rb(self, monkeypatch):
        # Second featured season prices 28.50: presence is not
        # corroboration, so the spike cap binds (75 + 15 * 0.25).
        assert _cap(monkeypatch, 28.50, "RB") == pytest.approx(78.75)

    def test_javonte_shape_releases_at_rb(self, monkeypatch):
        # Second featured season prices 73.55 >= 55: released.
        assert _cap(monkeypatch, 73.55, "RB") == pytest.approx(90.0)

    def test_release_threshold_boundary(self, monkeypatch):
        assert _cap(monkeypatch, 55.0, "RB") == pytest.approx(90.0)
        assert _cap(monkeypatch, 54.9, "RB") == pytest.approx(78.75)

    def test_wr_keeps_presence_release(self, monkeypatch):
        # The priced release is RB-scoped: a WR second season at
        # 28.50 still releases on presence, byte-identical to 3.11.
        assert _cap(monkeypatch, 28.50, "WR") == pytest.approx(90.0)


HENRY_SEASONS = [
    {"season": 2014, "games_played": 12, "rush_attempts": 326,
     "receptions": 13, "rush_yards": 1540, "receiving_yards": 123},
    {"season": 2015, "games_played": 15, "rush_attempts": 395,
     "receptions": 11, "rush_yards": 2219, "receiving_yards": 91},
]
STRONG_REC_SEASONS = [
    {"season": 2022, "games_played": 13, "rush_attempts": 200,
     "receptions": 50, "rush_yards": 1000, "receiving_yards": 500},
    {"season": 2023, "games_played": 13, "rush_attempts": 210,
     "receptions": 55, "rush_yards": 1050, "receiving_yards": 520},
]

BASE_PRICED = [("production", 0.2305, 90.0),
               ("utilization", 0.1367, 70.0),
               ("efficiency", 0.1433, 85.0),
               ("age", 0.1587, 88.0),
               ("breakout", 0.146, 75.0),
               ("athleticism", 0.0936, 80.0),
               ("competition", 0.0472, 65.0)]


class TestReceivingLeg:
    def test_leg_values_bracket_the_asymmetry_threshold(self):
        henry = pm._rb_receiving_leg(HENRY_SEASONS)
        strong = pm._rb_receiving_leg(STRONG_REC_SEASONS)
        assert henry is not None and henry < 50.0
        assert strong is not None and strong >= 50.0

    def test_missing_data_is_absent_never_zero(self):
        assert pm._rb_receiving_leg([]) is None
        assert pm._rb_receiving_leg(None) is None

    def test_weak_receiving_is_absent_from_priced_list(self):
        out = pm._rb2b_priced_list(list(BASE_PRICED), HENRY_SEASONS,
                                   pm.POSITION_WEIGHTS["RB"])
        assert not any(n == "receiving" for n, _, _ in out)

    def test_strong_receiving_is_paid_at_008(self):
        out = pm._rb2b_priced_list(list(BASE_PRICED),
                                   STRONG_REC_SEASONS,
                                   pm.POSITION_WEIGHTS["RB"])
        rec = [t for t in out if t[0] == "receiving"]
        assert len(rec) == 1
        assert rec[0][1] == pytest.approx(0.08)
        assert rec[0][2] >= 50.0

    def test_pure_rusher_never_taxed(self, monkeypatch):
        # The doctrine, priced: for a Henry-shaped profile the
        # asymmetric (shipped) grade is >= the flat-leg grade.
        asym = pm._rb2b_priced_list(list(BASE_PRICED), HENRY_SEASONS,
                                    pm.POSITION_WEIGHTS["RB"])
        monkeypatch.setattr(pm, "RB_RECEIVING_MIN_PRICE", 0.0)
        flat = pm._rb2b_priced_list(list(BASE_PRICED), HENRY_SEASONS,
                                    pm.POSITION_WEIGHTS["RB"])
        assert any(n == "receiving" for n, _, _ in flat)
        g_asym = pm._blend_grade(asym, "RB", trim_block=("efficiency",))
        g_flat = pm._blend_grade(flat, "RB", trim_block=("efficiency",))
        assert g_asym >= g_flat

    def test_funding_deltas_applied(self):
        out = dict((n, w) for n, w, _ in pm._rb2b_priced_list(
            list(BASE_PRICED), [], pm.POSITION_WEIGHTS["RB"]))
        assert out["utilization"] == pytest.approx(0.1367 - 0.04)
        assert out["age"] == pytest.approx(0.1587 - 0.02)
        assert out["breakout"] == pytest.approx(0.146 - 0.02)
        assert out["production"] == pytest.approx(0.2305)


class TestAthleticismReanchorAndFill:
    def test_pool_is_the_frozen_159(self):
        assert len(pm.RB_ATH_POOL) == 159
        assert list(pm.RB_ATH_POOL) == sorted(pm.RB_ATH_POOL)

    def test_fill_constant_is_pool_p40_reanchored(self):
        assert pm.RB_ATH_FILL_P40 == pytest.approx(43.36, abs=0.05)

    def test_reanchor_maps_elite_print_elite(self):
        # Barkley's shipped print (73.49) sat mid-pack on the old
        # scale; on the re-anchored scale it prices ~89.
        assert pm._rb_ath_reanchor(73.49) == pytest.approx(89.25,
                                                           abs=0.6)
        assert pm._rb_ath_reanchor(83.47) == pytest.approx(100.0)
        assert pm._rb_ath_reanchor(10.0) == pytest.approx(5.0)

    def test_missing_athleticism_prices_at_fill(self):
        priced = [t for t in BASE_PRICED if t[0] != "athleticism"]
        out = pm._rb2b_priced_list(priced, [],
                                   pm.POSITION_WEIGHTS["RB"])
        ath = [t for t in out if t[0] == "athleticism"]
        assert len(ath) == 1
        assert ath[0][1] == pytest.approx(0.0936)
        assert ath[0][2] == pytest.approx(pm.RB_ATH_FILL_P40)

    def test_tested_athleticism_gets_no_fill(self):
        out = pm._rb2b_priced_list(list(BASE_PRICED), [],
                                   pm.POSITION_WEIGHTS["RB"])
        ath = [t for t in out if t[0] == "athleticism"]
        assert len(ath) == 1
        assert ath[0][2] == pytest.approx(80.0)


class TestPeakMinWeight:
    PRICED = [("production", 0.2305, 90.0),
              ("efficiency", 0.1433, 80.0),
              ("age", 0.1587, 85.0),
              ("durability", 0.0063, 100.0),
              ("breakout", 0.146, 70.0)]

    def test_durability_cannot_hold_an_rb_peak_slot(self):
        top = pm._peak_set(list(self.PRICED), pos="RB")
        assert "durability" not in [n for n, _, _ in top]
        assert len(top) == 4

    def test_other_positions_peak_sets_untouched(self):
        top = pm._peak_set(list(self.PRICED), pos="WR")
        assert "durability" in [n for n, _, _ in top]


RB_PROSPECT = {
    "player_id": "TEST_RB_TRIM_PROBE",
    "name": "Trim Probe", "position": "RB", "age": 21.5,
    "seasons": [
        {"season": 2022, "games_played": 12, "rush_attempts": 200,
         "rush_yards": 1000, "rush_tds": 10, "receptions": 30,
         "receiving_yards": 250, "targets": 40,
         "dominator_rating": 0.30},
    ],
}


class TestEfficiencyTrimProtectionWiring:
    def test_rb_trim_block_includes_efficiency(self, monkeypatch):
        captured = {}
        orig = pm._blend_grade

        def wrap(priced, pos=None, allow_peak=True, trim_block=()):
            captured["trim_block"] = tuple(trim_block)
            return orig(priced, pos, allow_peak=allow_peak,
                        trim_block=trim_block)

        monkeypatch.setattr(pm, "_blend_grade", wrap)
        pm.score_prospect(copy.deepcopy(RB_PROSPECT), None,
                          skip_sagarin=True)
        assert "efficiency" in captured["trim_block"]


class TestMeasuredBlockStillBinds:
    def test_blocked_utilization_stays_in_the_mean(self):
        # The round-2 machinery under 3.12 trim rules: when the
        # trim fires, a blocked measured leg (utilization 49.2)
        # stays in the mean and the grade reads lower than the
        # unblocked blend, where it is the drop.
        priced = [("production", 0.264, 95.0),
                  ("efficiency", 0.1641, 90.0),
                  ("age", 0.1617, 88.0),
                  ("breakout", 0.1472, 85.0),
                  ("competition", 0.0541, 78.0),
                  ("durability", 0.0072, 94.44),
                  ("utilization", 0.1166, 49.2)]
        blocked = pm._blend_grade(
            list(priced), "RB",
            trim_block=("utilization", "efficiency"))
        open_g = pm._blend_grade(
            list(priced), "RB", trim_block=("efficiency",))
        assert blocked < open_g - 0.5


class TestOtherPositionsByteIdentical:
    def test_smith_wr_grade_matches_311_pin(self):
        # The same call test_prospect_v311 pins at 94.66 under
        # 3.11; WR is untouched by every RB-2b change.
        nov = pm.score_prospect(copy.deepcopy(SMITH), None,
                                skip_sagarin=True,
                                _active_season_year=2026,
                                _today=date(2026, 11, 24))
        assert nov["prospect_score"] == pytest.approx(94.66, abs=0.1)
