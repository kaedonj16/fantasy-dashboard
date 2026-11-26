"""Tests for prospect model v3.10: the last ceiling lever - the
dominator curve's slope just above the breakout threshold, and the
breakout-age anchor.

Two mechanisms, both diagnosed on Ja'Marr Chase's stored profile
(1,780 yards and 20 TDs at LSU in 2019, a 0.3068 dominator = 1.227x
the WR threshold; his 3.9 breakout leg read 74.8):

1. The dom curve (_dom_ratio_score) ran LINEARLY from 60 at the
   threshold to 100 at saturation (WR 1.7x), so the segment where
   real breakouts cluster (1.0-1.4x) was the shallowest part of the
   scale: Chase's generational season priced 73.0. v3.10 bends the
   above-threshold segment with a 0.65 power (DOM_CFG): the slope
   steepens just above the threshold and flattens toward
   saturation. Anchors are unchanged (60 at 1.0x, 100 at the
   saturation multiple, the below-threshold ramp untouched), so it
   is a re-pricing of the same absolute scale, not a new scale.

2. The breakout-age term (and the elite underclassman floor)
   anchored a season's age to the LAST season on file:
   age - (last_year - season_year). Chase opted out of 2020, so
   his last season on file IS 2019 and his 1,780-yard season was
   charged at his draft age (21.15) instead of the 19.6 he actually
   was - and the underclassman floor, which exists for exactly
   that season, never fired. v3.10 anchors to the draft year
   (BREAKOUT_CFG age_anchor_draft, fed by score_prospect's
   draft_class_year): age - (draft_year - season_year) + 0.45,
   the 0.45 being the distance from the mid-November season
   midpoint back from the late-April draft. Callers that cannot
   supply a draft year keep the legacy anchor byte-identically.

The dominance multiplier on the age term was evaluated for
softening in the same pass and deliberately left at its v3.2
shape: on a correctly anchored age it was worth ~1 leg point for
the index case and nothing measurable at the gate. The tax was
the anchor, not the gate.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import prospect_model as pm


# Chase-shaped WR: a cameo 2018, a generational 2019, then a gap
# year (opt-out) before the 2021 draft. Mirrors his stored profile's
# dominators (0.1303 / 0.3068) and draft age (21.15).
CHASE = {
    "player_id": "TEST_CHASE", "name": "Chase Shape", "position": "WR",
    "age": 21.15, "draft_class_year": 2021,
    "seasons": [
        {"season": 2018, "games_played": 13, "receptions": 23,
         "receiving_yards": 313, "receiving_tds": 3,
         "dominator_rating": 0.1303, "team_pass_yards": 2969},
        {"season": 2019, "games_played": 14, "receptions": 84,
         "receiving_yards": 1780, "receiving_tds": 20,
         "dominator_rating": 0.3068, "team_pass_yards": 6024},
    ],
}

# Riser-shaped WR: ordinary debut, strong junior year, no gap years.
RISER = {
    "player_id": "TEST_RISER", "name": "Riser Shape", "position": "WR",
    "age": 21.4, "draft_class_year": 2023,
    "seasons": [
        {"season": 2020, "games_played": 12, "receptions": 30,
         "receiving_yards": 420, "receiving_tds": 3,
         "dominator_rating": 0.16, "team_pass_yards": 3100},
        {"season": 2021, "games_played": 13, "receptions": 65,
         "receiving_yards": 980, "receiving_tds": 9,
         "dominator_rating": 0.27, "team_pass_yards": 3400},
        {"season": 2022, "games_played": 13, "receptions": 88,
         "receiving_yards": 1290, "receiving_tds": 12,
         "dominator_rating": 0.33, "team_pass_yards": 3600},
    ],
}


def _legacy(monkeypatch):
    monkeypatch.setattr(pm, "DOM_CFG", {"curve_power": 1.0})
    monkeypatch.setattr(pm, "BREAKOUT_CFG", {
        "arrival": True, "age_anchor_draft": False,
        "gate_lo": 0.65, "gate_span": 0.4})


class TestDomCurveShape:
    def test_threshold_and_saturation_anchors_unchanged(self):
        assert pm._dom_ratio_score(1.0, "WR") == pytest.approx(60.0)
        assert pm._dom_ratio_score(1.7, "WR") == pytest.approx(100.0)
        assert pm._dom_ratio_score(2.0, "RB") == pytest.approx(100.0)
        assert pm._dom_ratio_score(2.4, "TE") == pytest.approx(100.0)

    def test_below_threshold_ramp_unchanged(self):
        assert pm._dom_ratio_score(0.8, "WR") == pytest.approx(50.0)
        assert pm._dom_ratio_score(0.5, "WR") == pytest.approx(35.0)

    def test_just_above_threshold_prices_higher(self):
        # Chase's peak ratio: 73.0 on the linear segment, 79.2 bent.
        assert pm._dom_ratio_score(1.227, "WR") == pytest.approx(
            79.2, abs=0.15)
        # A 1.5x WR season: 88.6 linear, 92.1 bent.
        assert pm._dom_ratio_score(1.5, "WR") == pytest.approx(
            92.1, abs=0.15)

    def test_linear_when_power_is_one(self, monkeypatch):
        monkeypatch.setattr(pm, "DOM_CFG", {"curve_power": 1.0})
        assert pm._dom_ratio_score(1.227, "WR") == pytest.approx(
            73.0, abs=0.15)


class TestAgeAnchor:
    def test_gap_year_season_is_priced_at_its_real_age(self):
        # Direct leg call with the draft year supplied: the 2019
        # season is priced at 19.6, the underclassman floor fires
        # (1,780 yards at <=20), and the leg lands at the floor.
        leg = pm.calc_breakout_score(
            copy.deepcopy(CHASE["seasons"]), CHASE["age"], "WR",
            _draft_year=2021)
        assert leg == pytest.approx(85.0, abs=0.01)

    def test_no_draft_year_keeps_the_legacy_anchor(self):
        # The identical profile without a draft year keeps the
        # legacy AGE anchor (the 2019 season charged at the draft
        # age, floor silent) - but the bent dom curve is the shipped
        # curve for every caller, so the leg reads 77.6, not 74.8:
        # dom 79.2 x .45 + traj 100 x .35 + gated age 34.9 x .20.
        leg = pm.calc_breakout_score(
            copy.deepcopy(CHASE["seasons"]), CHASE["age"], "WR")
        assert leg == pytest.approx(77.6, abs=0.6)

    def test_full_legacy_reproduces_the_39_read(self, monkeypatch):
        # Both levers reverted (linear curve + legacy anchor) the
        # Chase shape reproduces his stored 3.9 leg exactly: 74.8.
        _legacy(monkeypatch)
        leg = pm.calc_breakout_score(
            copy.deepcopy(CHASE["seasons"]), CHASE["age"], "WR")
        assert leg == pytest.approx(74.8, abs=0.6)

    def test_end_to_end_grade_reaches_the_high_90s_shape(self):
        out = pm.score_prospect(copy.deepcopy(CHASE), None,
                                skip_sagarin=True)
        assert out["breakout_profile_score"] == pytest.approx(
            85.0, abs=0.01)

    def test_riser_is_not_dragged_by_the_new_anchors(self, monkeypatch):
        new = pm.score_prospect(copy.deepcopy(RISER), None,
                                skip_sagarin=True)
        _legacy(monkeypatch)
        old = pm.score_prospect(copy.deepcopy(RISER), None,
                                skip_sagarin=True)
        # Both levers only re-price upward-or-flat for a riser whose
        # peak sits in the bent segment; his grade must not fall.
        assert new["prospect_score"] >= old["prospect_score"]
        assert new["breakout_profile_score"] >= \
            old["breakout_profile_score"]


class TestVersion:
    def test_model_version_is_310(self):
        assert pm.MODEL_VERSION == "3.13"

    def test_shipped_cfg(self):
        assert pm.DOM_CFG["curve_power"] == pytest.approx(0.65)
        assert pm.BREAKOUT_CFG["age_anchor_draft"] is True
        # The dominance gate was evaluated and deliberately kept.
        assert pm.BREAKOUT_CFG["gate_lo"] == pytest.approx(0.65)
        assert pm.BREAKOUT_CFG["gate_span"] == pytest.approx(0.4)
