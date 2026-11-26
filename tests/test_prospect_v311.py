"""Tests for prospect model v3.11: ceiling attainability.

Kaedon's ruling (2026-11-24): "it's too difficult because no one
can get a 99 right now", with the steer that "a prospect can be
generational while checking some boxes and leaving some
unchecked" - greatness is spiky, not uniform.

The 3.10 audit (ceiling-attainability-2026-11-24.md) priced every
leg across the 2,265-profile recorded population: production,
utilization, efficiency, age and competition all print 100
somewhere; the arithmetic (chimera) ceiling was already 95.0 /
98.9 / 99.5 / 99.2 by position. Leg ceilings were not the bind -
co-occurrence was: no real profile holds the maxima at once, and
the blend's 40% mean share charged a generational profile full
price for its one mediocre box.

v3.11 changes the blend only:
1. Peak share 0.60 -> 0.74 (QB keeps its 0.30 guard share; raising
   it was measured to re-open the QB Poor-tier wound, +9.5 points
   of Poor QBs over 70, for +0.3 on the best stored QB).
2. A gated trimmed mean: the single weakest substantive box is
   dropped from the MEAN term when the profile has earned it -
   >= 5 priced legs, best leg >= 90, the production-anchored
   peak set >= 88, the dropped leg <= 75 and carrying >= 0.02
   weight. Production is never droppable, and the trim never
   applies under the multi-None anchor. Ungated, the same trim
   floated fragments (+0.55 mean) and forgave the Anderson case;
   gated, fragments move only with the pool-wide alpha lift and
   Anderson is byte-identical at 71.62 - his best leg (84.8)
   never reaches the 90 gate, so the competition discount keeps
   biting exactly as in 3.10.
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
from test_prospect_v39 import TestCompetitionDiscount

LEGACY_BLEND = {"peak_alpha": 0.60, "trim_lowest": False,
                # Round 6 (step A) made utilization peak-ineligible;
                # the legacy blend predates the exclusion.
                "peak_exclude": ()}

WR_W = pm.POSITION_WEIGHTS["WR"]
QB_W = pm.POSITION_WEIGHTS["QB"]


def _priced(weights, **legs):
    return [(name, weights[name], float(v)) for name, v in legs.items()]


def _legacy(monkeypatch):
    monkeypatch.setattr(pm, "BLEND_CFG",
                        {**pm.BLEND_CFG, **LEGACY_BLEND})


# The spiky WR shape from the audit: four legs at their honest
# observed maxima, competition mediocre (a small-school star),
# environment and recruiting mid.
SPIKY_WR = dict(production=100, utilization=100, efficiency=100,
                breakout=98.8, age=85, competition=60,
                environment=70, recruiting=73)


class TestTrimmedMean:
    def test_spiky_wr_forgives_the_one_weak_box(self):
        # Trimmed mean drops competition (60): 94.61 - unchanged
        # in round 6. The peak term moved: utilization (100) is
        # peak-ineligible since round 6 step A, so the peak set
        # is {production, efficiency, breakout, age} at 96.39
        # (was 99.73 with utilization resident);
        # grade .26 x 94.61 + .74 x 96.39 = 95.92.
        g = pm._blend_grade(_priced(WR_W, **SPIKY_WR), "WR")
        assert g == pytest.approx(95.92, abs=0.05)

    def test_spiky_wr_under_the_310_blend(self, monkeypatch):
        # Same legs, 3.10 blend (alpha .60, no trim): the mean
        # term taxes the weak box at full price - 96.12.
        _legacy(monkeypatch)
        g = pm._blend_grade(_priced(WR_W, **SPIKY_WR), "WR")
        assert g == pytest.approx(96.12, abs=0.05)

    def test_production_is_never_the_dropped_leg(self):
        # Production (70) is the second-weakest leg; competition
        # (65) is the weakest eligible. The trim must drop
        # competition, not production: the trimmed mean is 90.37
        # (competition's 6.74 weighted points leave; production's
        # stay). Round 6: utilization (100) is peak-ineligible,
        # so production is top-4 on value and the peak set
        # {efficiency, breakout, age, production} prices 88.71;
        # grade .26 x 90.37 + .74 x 88.71 = 89.14.
        legs = _priced(WR_W, production=70, utilization=100,
                       efficiency=100, breakout=100, age=100,
                       competition=65)
        g = pm._blend_grade(legs, "WR")
        assert g == pytest.approx(89.14, abs=0.1)

    def test_sub_two_percent_legs_are_not_boxes(self):
        # QB environment carries 0.0126 weight: it is a context
        # adjustment, not a box, so the trim drops utilization
        # (65, the weakest SUBSTANTIVE leg) and leaves the 40
        # environment leg in the mean: grade 97.55 at the QB
        # alpha. (Dropping environment instead reads 96.65.)
        legs = _priced(QB_W, production=100, utilization=65,
                       efficiency=100, age=100, competition=95,
                       environment=40, experience=90, recruiting=73)
        g = pm._blend_grade(legs, "QB")
        assert g == pytest.approx(97.55, abs=0.1)

    def test_no_elite_box_no_forgiveness(self):
        # Best leg 89 < 90: nothing is trimmed even with a 50 leg
        # present - the plain blend prices every box (85.94;
        # a trim would read higher). Round 6: utilization (88)
        # is peak-ineligible, so the peak set is {production,
        # efficiency, breakout, age} at 87.09.
        legs = _priced(WR_W, production=89, utilization=88,
                       efficiency=87, breakout=86, age=85,
                       competition=50)
        g = pm._blend_grade(legs, "WR")
        assert g == pytest.approx(85.94, abs=0.1)

    def test_one_spike_leg_does_not_earn_forgiveness(self):
        # Production 100 but the production-anchored peak set
        # prices 75.89 < 88: a lone spike in a mediocre profile
        # earns nothing (74.47). Round 6: utilization (60) was
        # the fourth peak leg by value; ineligible now, so
        # breakout (58) takes the slot and the peak prices
        # lower - the gate still refuses the trim.
        legs = _priced(WR_W, production=100, utilization=60,
                       efficiency=62, breakout=58, age=64,
                       competition=55, recruiting=50)
        g = pm._blend_grade(legs, "WR")
        assert g == pytest.approx(74.47, abs=0.08)

    def test_merely_least_strong_is_not_a_weak_box(self):
        # Weakest leg 80 > 75: no trim - forgiveness is for a
        # genuinely weak box, not for rounding down a strong
        # profile's softest strength (95.99). Round 6: the peak
        # set is {production, efficiency, breakout, age} at
        # 96.39, utilization (100) ineligible.
        legs = _priced(WR_W, production=100, utilization=100,
                       efficiency=100, breakout=98.8, age=85,
                       competition=80)
        g = pm._blend_grade(legs, "WR")
        assert g == pytest.approx(95.99, abs=0.08)

    def test_trim_needs_five_priced_legs(self):
        # Four priced legs: no trim whatever the values. Round
        # 6: only three legs are peak-eligible (utilization is
        # not), so the peak prices {production, efficiency,
        # competition} at 88.68 and the grade reads 89.25.
        legs = _priced(WR_W, production=100, utilization=100,
                       efficiency=100, competition=50)
        g = pm._blend_grade(legs, "WR")
        assert g == pytest.approx(89.25, abs=0.05)

    def test_anchor_profile_is_never_trimmed(self):
        # allow_peak=False (the multi-None anchor) prices the
        # plain mean of the full leg set: 90.71 on the spiky
        # shape, no peak, no trim.
        g = pm._blend_grade(_priced(WR_W, **SPIKY_WR), "WR",
                            allow_peak=False)
        assert g == pytest.approx(90.71, abs=0.05)


class TestEndToEnd:
    def test_anderson_moves_only_by_step_a(self, monkeypatch):
        # Through round 5 Anderson was byte-identical to 3.10
        # (71.62): the competition discount bites and nothing
        # opened his trim gate. Round 6 step A moves him -0.21
        # and nothing else in the round touches him: his
        # utilization was peak-resident, so excluding it from
        # the peak set reprices his peak term; steps B and C1
        # net to zero on his profile (staged lab, S14).
        anderson = TestCompetitionDiscount.ANDERSON
        new = pm.score_prospect(copy.deepcopy(anderson), None,
                                skip_sagarin=True,
                                _active_season_year=2026)
        assert new["prospect_score"] == pytest.approx(71.41,
                                                      abs=0.05)
        monkeypatch.setattr(
            pm, "BLEND_CFG",
            {**pm.BLEND_CFG, "peak_exclude": ()})
        old = pm.score_prospect(copy.deepcopy(anderson), None,
                                skip_sagarin=True,
                                _active_season_year=2026)
        assert old["prospect_score"] == pytest.approx(71.62,
                                                      abs=0.05)

    def test_smith_fixture_moves_only_a_whisker(self):
        # Smith's profile is nearly uniform-elite: the ceiling
        # changes are worth +0.11 complete / +0.14 in-season to
        # him (94.24 / 94.52 under 3.10). The lift belongs to
        # spiky profiles, not to him. Round 6: steps A and B
        # leave him at the round-5 pin (94.35); step C1's rate
        # key reprices his complete-mode production anchor -
        # his best per-game season now anchors the blend - for
        # +0.31, landing the complete read on the in-season
        # read (94.66), which does not move.
        complete = pm.score_prospect(copy.deepcopy(SMITH), None,
                                     skip_sagarin=True)
        assert complete["prospect_score"] == pytest.approx(
            94.66, abs=0.1)
        nov = pm.score_prospect(copy.deepcopy(SMITH), None,
                                skip_sagarin=True,
                                _active_season_year=2026,
                                _today=date(2026, 11, 24))
        assert nov["prospect_score"] == pytest.approx(94.66, abs=0.1)


class TestVersion:
    def test_model_version_is_311(self):
        assert pm.MODEL_VERSION == "3.12"

    def test_shipped_revision_cfgs(self):
        # The 3.11 revision (Kaedon's three calibration flags):
        # boundary-band discount bend + production evidence cap.
        cc = pm.COMPETITION_CFG
        assert cc["prod_discount"] == pytest.approx(0.35)
        assert cc["discount_floor"] == pytest.approx(65.0)
        assert cc["discount_band"] == pytest.approx(50.0)
        assert cc["discount_power"] == pytest.approx(0.3)
        ec = pm.PROD_EVIDENCE_CFG
        assert ec["enabled"] is True
        assert ec["spike_level"] == pytest.approx(75.0)
        assert ec["spike_keep"] == pytest.approx(0.25)
        assert ec["release_age"] == pytest.approx(20.0)


# ── 3.11 revision: the boundary-band discount bend ─────────────
# The linear depth spent 2.7% at competition 60 (the Fannin
# case: a record MAC season priced nearly whole). The bend
# re-prices the [50, 65] band on a concave curve continuous
# with the linear depth at 50: below the band the discount is
# byte-identical (Anderson, Kupp), above the floor it is zero,
# and no level is ever refunded.

# A Kupp-shaped small-school line (the v3.9 fixture, copied so
# this file stands alone): three huge FCS years. At Eastern
# Washington his competition reads 49.1, just below the band;
# the same line at a MAC school reads in the band.
KUPP_LINE = {
    "player_id": "T_kuppline", "name": "Small-school WR",
    "position": "WR", "draft_class_year": 2017, "age": 23.8,
    "athletic": {"forty": 4.62, "height_in": 74, "weight": 204},
    "seasons": [
        {"season": 2014, "team": "Eastern Washington",
         "conference": "Big Sky", "games": 13, "receptions": 104,
         "receiving_yards": 1431, "receiving_tds": 16},
        {"season": 2015, "team": "Eastern Washington",
         "conference": "Big Sky", "games": 11, "receptions": 114,
         "receiving_yards": 1642, "receiving_tds": 19},
        {"season": 2016, "team": "Eastern Washington",
         "conference": "Big Sky", "games": 12, "receptions": 117,
         "receiving_yards": 1700, "receiving_tds": 17},
    ],
}

MAC_KUPP = copy.deepcopy(KUPP_LINE)
for _s in MAC_KUPP["seasons"]:
    _s["team"] = "Bowling Green"
    # The stored data spells it "Mid-American" (conference
    # quality 0.78 -> competition 60.0, the Fannin reading);
    # the "MAC" alias prices lower.
    _s["conference"] = "Mid-American"


def _prod_with_cfg(prospect, **comp_over):
    saved = pm.COMPETITION_CFG
    pm.COMPETITION_CFG = {**saved, **comp_over}
    try:
        return pm.score_prospect(copy.deepcopy(prospect), None,
                                 skip_sagarin=True)
    finally:
        pm.COMPETITION_CFG = saved


class TestDiscountBandBend:
    def test_below_the_band_is_byte_identical(self):
        # Kupp's competition reads 49.1 - below the band, so the
        # bent shape must price him exactly as the linear one.
        bent = _prod_with_cfg(KUPP_LINE)
        linear = _prod_with_cfg(KUPP_LINE, discount_power=1.0)
        assert bent["competition_score"] < 50.0
        assert bent["production_score"] == pytest.approx(
            linear["production_score"], abs=0.01)
        assert bent["prospect_score"] == pytest.approx(
            linear["prospect_score"], abs=0.01)

    def test_in_band_pays_a_material_toll(self):
        on = _prod_with_cfg(MAC_KUPP)
        comp = on["competition_score"]
        assert 50.0 <= comp < 65.0
        raw = _prod_with_cfg(MAC_KUPP, prod_discount=0.0)
        linear = _prod_with_cfg(MAC_KUPP, discount_power=1.0)
        # The bent factor at this competition, from the cfg.
        gap = (65.0 - comp) / 65.0
        bgap = (65.0 - 50.0) / 65.0
        bent_gap = bgap * (gap / bgap) ** 0.3
        factor = 1.0 - 0.35 * bent_gap
        assert on["production_score"] == pytest.approx(
            raw["production_score"] * factor, abs=0.75)
        # And it is materially deeper than the linear price.
        assert on["production_score"] < \
            linear["production_score"] - 1.0


# ── 3.11 revision: the production evidence base ────────────────
# One featured season cannot carry the leg's full spike
# (Jameson Williams: two buried Ohio State years, one monster
# Alabama year, priced as the decade's WR2). A second featured
# season - or a single one played at an elite-underclassman
# age - releases the cap. Buried seasons are never scored
# against the player: the role gate excused them before the
# leg ever read them.

JAMESON = {
    "player_id": "T_jameson", "name": "One-season WR",
    "position": "WR", "draft_class_year": 2022, "age": 21.9,
    "seasons": [
        {"season": 2019, "team": "Ohio State", "conference": "Big Ten",
         "receptions": 6, "receiving_yards": 112, "receiving_tds": 1,
         "dominator_rating": 0.027},
        {"season": 2020, "team": "Ohio State", "conference": "Big Ten",
         "receptions": 9, "receiving_yards": 154, "receiving_tds": 2,
         "dominator_rating": 0.080},
        {"season": 2021, "team": "Alabama", "conference": "SEC",
         "receptions": 79, "receiving_yards": 1572, "receiving_tds": 15,
         "dominator_rating": 0.311},
    ],
}


def _prod_leg(prospect, **ev_over):
    saved = pm.PROD_EVIDENCE_CFG
    pm.PROD_EVIDENCE_CFG = {**saved, **ev_over}
    try:
        gated = pm._role_gated_seasons(
            [s for s in prospect["seasons"]
             if pm.season_has_counting_stats(s)], "WR")
        return pm.calc_production_score(
            copy.deepcopy(gated), "WR", skip_sagarin=True,
            _draft_year=prospect["draft_class_year"],
            _age=prospect["age"])
    finally:
        pm.PROD_EVIDENCE_CFG = saved


class TestProductionEvidenceCap:
    def test_one_featured_season_cannot_carry_the_spike(self):
        off = _prod_leg(JAMESON, enabled=False)
        on = _prod_leg(JAMESON)
        assert off > 75.0
        assert on == pytest.approx(75.0 + (off - 75.0) * 0.25,
                                   abs=0.05)

    def test_second_featured_season_releases_the_cap(self):
        two = copy.deepcopy(JAMESON)
        two["seasons"][1] = {
            "season": 2020, "team": "Ohio State",
            "conference": "Big Ten", "receptions": 61,
            "receiving_yards": 940, "receiving_tds": 8,
            "dominator_rating": 0.27}
        off = _prod_leg(two, enabled=False)
        on = _prod_leg(two)
        assert on == pytest.approx(off, abs=0.05)

    def test_youth_release_for_a_teenage_breakout(self):
        young = copy.deepcopy(JAMESON)
        # Featured season 2021, draft 2022: anchored age
        # 20.4 - 1 + 0.45 = 19.85 <= 20 - released in full.
        young["age"] = 20.4
        off = _prod_leg(young, enabled=False)
        on = _prod_leg(young)
        assert on == pytest.approx(off, abs=0.05)
        # At 21.4 the anchored age is 20.85 - capped.
        young["age"] = 21.4
        on_old = _prod_leg(young)
        assert on_old == pytest.approx(75.0 + (off - 75.0) * 0.25,
                                       abs=0.05)

    def test_buried_seasons_are_not_scored_against_him(self):
        # The same monster season with NO buried years on file
        # prices the identical capped leg: the buried seasons
        # were excused by the role gate, not averaged in.
        bare = copy.deepcopy(JAMESON)
        bare["seasons"] = bare["seasons"][2:]
        assert _prod_leg(bare) == pytest.approx(
            _prod_leg(JAMESON), abs=0.05)

    def test_shipped_blend_cfg(self):
        cfg = pm.BLEND_CFG
        assert cfg["peak_alpha"] == pytest.approx(0.74)
        # The QB peak-share guard survives: it is what keeps the
        # Poor-QB tier from floating at high peak weight.
        assert cfg["peak_alpha_by_pos"] == {"QB": 0.30}
        assert cfg["peak_k"] == 4
        assert cfg["trim_lowest"] is True
        assert cfg["trim_exclude"] == ("production",)
        assert cfg["trim_min_legs"] == 5
        assert cfg["trim_min_top"] == pytest.approx(90.0)
        assert cfg["trim_min_weight"] == pytest.approx(0.02)
        assert cfg["trim_drop_ceiling"] == pytest.approx(75.0)
        # Round 7: forgiveness has a depth limit - a leg under
        # 40 is a failed trait, not a weak box, and cannot be
        # the dropped leg.
        assert cfg["trim_drop_floor"] == pytest.approx(40.0)
        assert cfg["trim_min_peak"] == pytest.approx(88.0)


# ── 3.11 round 2: the RB fix round ─────────────────────────────
# Kaedon (2026-11-24): "my wr grading seems to be good and te
# pretty good too but rb needs fixed big time." Three general
# mechanisms, bite scoped to RB, WR/TE calibration a
# do-not-disturb guardrail (zero WR/TE grade movers pool-wide):
# 1. The trim forgives absence, not measured weakness: with a
#    featured season on file, utilization - the one leg that
#    aggregates the whole featured body of work - is blocked
#    from the trim drop (the Javonte split backfield).
# 2. RB efficiency volume shrinkage: above 85 the leg credits
#    only the evidence-backed share of the excess, n/(n+100)
#    on the priced season's opportunities (Javonte's 100.0 off
#    157 carries; Barkley's 76.1 and Bijan's 81.5 sit below
#    the level and never move).
# 3. The production level ceiling (RB only): below competition
#    60 the production leg's printable value is capped, 82 at
#    competition <= 45 rising to 100 at 60, before the discount
#    prices it (the Gillman case: Montana, competition 49.09).

RB_W = pm.POSITION_WEIGHTS["RB"]

# Javonte Williams' leg shape from the audit (section 7.1):
# production 98.2, efficiency 100, and the split backfield in
# utilization (43.7) as the trim drop.
JAVONTE_LEGS = dict(production=98.2, utilization=43.7,
                    efficiency=100.0, age=88.9, breakout=67.7,
                    athleticism=64.7, competition=78.2,
                    environment=66.8)

SPLIT_RB = {
    "player_id": "T_splitrb", "name": "Split RB", "position": "RB",
    "draft_class_year": 2021, "age": 21.6,
    "seasons": [
        {"season": 2018, "team": "North Carolina",
         "conference": "ACC", "games_played": 11,
         "rush_attempts": 110, "rush_yards": 680, "rush_tds": 6,
         "receptions": 12, "receiving_yards": 110, "receiving_tds": 1,
         "dominator_rating": 0.16, "team_total_yards": 5600,
         "team_pass_yards": 3100, "team_total_tds": 52,
         "team_rush_attempts": 480},
        {"season": 2019, "team": "North Carolina",
         "conference": "ACC", "games_played": 13,
         "rush_attempts": 166, "rush_yards": 933, "rush_tds": 8,
         "receptions": 17, "receiving_yards": 155, "receiving_tds": 1,
         "dominator_rating": 0.21, "team_total_yards": 6100,
         "team_pass_yards": 3400, "team_total_tds": 58,
         "team_rush_attempts": 510},
        {"season": 2020, "team": "North Carolina",
         "conference": "ACC", "games_played": 11,
         "rush_attempts": 157, "rush_yards": 1140, "rush_tds": 13,
         "receptions": 25, "receiving_yards": 305, "receiving_tds": 3,
         "dominator_rating": 0.30, "team_total_yards": 5900,
         "team_pass_yards": 3200, "team_total_tds": 55,
         "team_rush_attempts": 470},
    ],
}

FCS_RB = {
    "player_id": "T_fcsrb", "name": "FCS RB", "position": "RB",
    "draft_class_year": 2027, "age": 22.4,
    "seasons": [
        {"season": 2024, "team": "Montana", "conference": "Big Sky",
         "games_played": 13, "rush_attempts": 190, "rush_yards": 1150,
         "rush_tds": 14, "receptions": 28, "receiving_yards": 300,
         "receiving_tds": 2, "dominator_rating": 0.34,
         "team_total_yards": 5400, "team_pass_yards": 2900,
         "team_total_tds": 50, "team_rush_attempts": 430},
        {"season": 2025, "team": "Montana", "conference": "Big Sky",
         "games_played": 13, "rush_attempts": 205, "rush_yards": 1330,
         "rush_tds": 16, "receptions": 33, "receiving_yards": 350,
         "receiving_tds": 3, "dominator_rating": 0.38,
         "team_total_yards": 5600, "team_pass_yards": 3000,
         "team_total_tds": 54, "team_rush_attempts": 445},
        {"season": 2026, "team": "Montana", "conference": "Big Sky",
         "games_played": 12, "rush_attempts": 211, "rush_yards": 1477,
         "rush_tds": 17, "receptions": 37, "receiving_yards": 377,
         "receiving_tds": 3, "dominator_rating": 0.41,
         "team_total_yards": 5500, "team_pass_yards": 2950,
         "team_total_tds": 52, "team_rush_attempts": 440},
    ],
}


def _score_with(prospect, attr, **over):
    saved = getattr(pm, attr)
    setattr(pm, attr, {**saved, **over})
    try:
        return pm.score_prospect(copy.deepcopy(prospect), None,
                                 skip_sagarin=True)
    finally:
        setattr(pm, attr, saved)


class TestTrimForgivesAbsenceNotWeakness:
    def test_block_redirects_the_drop_off_utilization(self):
        # Unblocked, the trim drops utilization (43.7) and the
        # shape prices 92.32 - the decade-RB-max failure. With
        # utilization blocked the drop redirects to athleticism
        # (64.7) and the same legs price 91.18.
        legs = _priced(RB_W, **JAVONTE_LEGS)
        open_g = pm._blend_grade(legs, "RB")
        blocked_g = pm._blend_grade(legs, "RB",
                                    trim_block=("utilization",))
        assert open_g == pytest.approx(92.32, abs=0.1)
        assert blocked_g == pytest.approx(91.18, abs=0.1)

    def test_measured_split_backfield_is_not_forgiven(self):
        # Three featured seasons, elite rates, a split share:
        # utilization is a measurement, so the shipped block
        # keeps it in the mean. Under 3.11 that cost the grade
        # ~1 against the block-off arm. Under 3.12 the fixture's
        # production is corroboration-capped (79.4) and its
        # production-anchored peak (79.9) no longer clears the
        # 88 trim gate, so the trim never fires in either arm
        # and the grades coincide; the block's blend-level pin
        # (blocked < unblocked when the trim fires) lives in
        # test_prospect_v312.
        on = pm.score_prospect(copy.deepcopy(SPLIT_RB), None,
                               skip_sagarin=True)
        off = _score_with(SPLIT_RB, "TRIM_MEASURED_CFG",
                          enabled=False)
        assert on["utilization_score"] == pytest.approx(
            off["utilization_score"], abs=0.01)
        assert on["utilization_score"] < 60.0
        assert on["prospect_score"] == pytest.approx(
            off["prospect_score"], abs=0.01)

    def test_no_featured_season_block_is_inert(self):
        # Never featured (every season under the role floors):
        # nothing was measured in a real role, so the block
        # never engages and the grade is identical either way.
        buried = copy.deepcopy(SPLIT_RB)
        for s in buried["seasons"]:
            s["rush_attempts"] = 30
            s["receptions"] = 8
        on = pm.score_prospect(copy.deepcopy(buried), None,
                               skip_sagarin=True)
        off = _score_with(buried, "TRIM_MEASURED_CFG",
                          enabled=False)
        assert on["prospect_score"] == pytest.approx(
            off["prospect_score"], abs=0.01)

    def test_shipped_trim_measured_cfg(self):
        cfg = pm.TRIM_MEASURED_CFG
        assert cfg["enabled"] is True
        # Utilization only: blocking efficiency/breakout too
        # disturbed WR (Jefferson -1.29 in the round-2 sweep).
        assert tuple(cfg["legs"]) == ("utilization",)


def _eff(seasons, **over):
    saved = pm.EFF_VOLUME_SHRINK_CFG
    pm.EFF_VOLUME_SHRINK_CFG = {**saved, **over}
    try:
        return pm.calc_efficiency_score(
            copy.deepcopy(seasons), "RB")
    finally:
        pm.EFF_VOLUME_SHRINK_CFG = saved


_EFF_BASE = {"season": 2019, "yds_per_carry": 5.6,
             "rush_attempts": 150, "rush_yards": 840,
             "receptions": 15, "receiving_yards": 120}
_EFF_SPIKE = {"season": 2020, "yds_per_carry": 7.26,
              "rush_attempts": 157, "rush_yards": 1140,
              "receptions": 25, "receiving_yards": 305}


class TestEfficiencyVolumeShrink:
    def test_small_sample_spike_is_shrunk(self):
        seasons = [_EFF_BASE, _EFF_SPIKE]
        raw = _eff(seasons, enabled=False)
        shrunk = _eff(seasons)
        # Round 7 layers the pair-form parity discount under
        # this pin: the spike's pair value 100.0 now reads
        # 86.0 (35% of the excess over 60 discounted - the
        # pooled pair-vs-composite gap), so the unshrunk leg
        # is 86.0 + the +1.29 consistency term (latest YPC
        # 7.26 vs the two-season average 6.43) = 87.29, where
        # round 2 read a clipped 100.0.
        assert raw == pytest.approx(87.29, abs=0.1)
        # n = 157 + 25 = 182 opportunities behind the print:
        # 85 + 2.29 x 182/282 = 86.48 (round 2: 94.68, when
        # the pair form still printed its raw 100.0).
        assert shrunk == pytest.approx(86.48, abs=0.15)

    def test_volume_backed_rate_stands(self):
        big = dict(_EFF_SPIKE, rush_attempts=400, rush_yards=2904,
                   receptions=60, receiving_yards=732)
        seasons = [_EFF_BASE, big]
        shrunk = _eff(seasons)
        # Same parity-priced 87.29 print, more evidence behind
        # it: n = 460, so 85 + 2.29 x 460/560 = 86.88 - the
        # proven workload still keeps more of the excess than
        # the 182-opportunity spike above (86.48).
        assert shrunk == pytest.approx(86.88, abs=0.15)
        assert shrunk > _eff([_EFF_BASE, _EFF_SPIKE])

    def test_ordinary_print_is_untouched(self):
        # A Barkley-shaped rate (5.86 YPC) prices ~76, below
        # the level: the shrink never fires and the leg is
        # byte-identical on and off.
        seasons = [
            {"season": 2016, "yds_per_carry": 5.5,
             "rush_attempts": 238, "rush_yards": 1309,
             "receptions": 28, "receiving_yards": 195},
            {"season": 2017, "yds_per_carry": 5.86,
             "rush_attempts": 217, "rush_yards": 1271,
             "receptions": 54, "receiving_yards": 476},
        ]
        assert _eff(seasons) == pytest.approx(
            _eff(seasons, enabled=False), abs=0.01)

    def test_rate_without_recorded_volume_is_not_shrunk(self):
        # Stored rates with no counting stats behind them are
        # evidence the leg already accepts (v3.8 M1): absence
        # of volume data is not a small sample.
        seasons = [
            {"season": 2019, "yds_per_carry": 5.6,
             "yds_per_reception": 8.0},
            {"season": 2020, "yds_per_carry": 7.26,
             "yds_per_reception": 12.2},
        ]
        assert _eff(seasons) == pytest.approx(
            _eff(seasons, enabled=False), abs=0.01)

    def test_shipped_eff_shrink_cfg(self):
        cfg = pm.EFF_VOLUME_SHRINK_CFG
        assert cfg["enabled"] is True
        assert cfg["level"] == pytest.approx(85.0)
        assert cfg["k"] == pytest.approx(100.0)


class TestProductionLevelCeiling:
    def test_cap_curve(self):
        cap = pm._production_level_cap
        assert cap(40.0) == pytest.approx(82.0)
        assert cap(45.0) == pytest.approx(82.0)
        assert cap(52.5) == pytest.approx(91.0)
        assert cap(60.0) is None
        assert cap(80.0) is None
        assert cap(None) is None

    def test_fcs_spike_is_capped_before_the_discount(self):
        # The Gillman shape: competition 49.09, raw production
        # ~100. The cap (86.9 at his level) lands first, then
        # the ~9% discount prices the capped value: 79.46.
        on = pm.score_prospect(copy.deepcopy(FCS_RB), None,
                               skip_sagarin=True)
        off = _score_with(FCS_RB, "PROD_CEILING_CFG",
                          enabled=False)
        assert on["competition_score"] == pytest.approx(
            49.09, abs=0.5)
        assert off["production_score"] > 90.0
        assert on["production_score"] == pytest.approx(
            79.46, abs=0.75)
        assert on["prospect_score"] < off["prospect_score"] - 3.0

    def test_wr_at_the_same_level_is_untouched(self):
        # The Kupp line reads competition 49.1 - Gillman's
        # exact level - but the ceiling is RB-scoped: small-
        # school WR production is the model's own evidence
        # that the level translates at receiver.
        on = pm.score_prospect(copy.deepcopy(KUPP_LINE), None,
                               skip_sagarin=True)
        off = _score_with(KUPP_LINE, "PROD_CEILING_CFG",
                          enabled=False)
        assert on["production_score"] == pytest.approx(
            off["production_score"], abs=0.01)
        assert on["prospect_score"] == pytest.approx(
            off["prospect_score"], abs=0.01)

    def test_shipped_ceiling_cfg(self):
        cfg = pm.PROD_CEILING_CFG
        assert cfg["enabled"] is True
        assert tuple(cfg["positions"]) == ("RB",)
        assert cfg["cap_floor"] == pytest.approx(82.0)
        assert cfg["comp_lo"] == pytest.approx(45.0)
        assert cfg["comp_hi"] == pytest.approx(60.0)


def _te_eff(seasons, **kw):
    return pm.calc_efficiency_score(copy.deepcopy(seasons), "TE",
                                    **kw)


# The Bowers shape: three seasons of YPR evidence, the last a
# career-low injury year (56/714 at 12.75 after 15.75 / 14.95).
_BOWERS_TE = [
    {"season": 2021, "receptions": 56, "receiving_yards": 882,
     "yds_per_reception": 15.75},
    {"season": 2022, "receptions": 63, "receiving_yards": 942,
     "yds_per_reception": 14.95},
    {"season": 2023, "receptions": 56, "receiving_yards": 714,
     "yds_per_reception": 12.75},
]


class TestTEEfficiencyBestSeason:
    def test_final_season_dip_no_longer_prices_the_leg(self):
        # Latest-only priced this shape at 79.57 (the 12.75
        # season at face value, less the consistency drag).
        # Best-season pricing: peak (942 yards, YPR scale 100)
        # at 0.90 + latest (80.77) at 0.10 = 98.08, consistency
        # -1.20 -> 96.88.
        eff = _te_eff(_BOWERS_TE)
        assert eff == pytest.approx(96.88, abs=0.25)
        assert eff > 90.0

    def test_latest_peak_blend_is_symmetric(self):
        # When the latest season IS the peak, the blend of the
        # value with itself returns the single-season read.
        seasons = [
            {"season": 2022, "receptions": 40,
             "receiving_yards": 440, "yds_per_reception": 11.0},
            {"season": 2023, "receptions": 60,
             "receiving_yards": 840, "yds_per_reception": 14.0},
        ]
        blended = _te_eff(seasons)
        single = _te_eff([seasons[1]])
        assert blended == pytest.approx(single, abs=0.01)

    def test_catch_rate_prices_per_season(self):
        # Peak season: YPR scale 100, catch 50/60 -> 93.33,
        # per-season 97.50. Latest: YPR 53.85, catch 40/80 ->
        # 0, per-season 33.65. Blend 91.12, consistency
        # (latest YPR 11 vs avg 12.5) -1.20 -> 89.92.
        seasons = [
            {"season": 2022, "receptions": 50, "targets": 60,
             "receiving_yards": 700, "yds_per_reception": 14.0},
            {"season": 2023, "receptions": 40, "targets": 80,
             "receiving_yards": 440, "yds_per_reception": 11.0},
        ]
        assert _te_eff(seasons) == pytest.approx(89.92, abs=0.3)

    def test_peak_evidence_rescues_a_missing_latest(self):
        # The latest season carries no rate evidence at all;
        # the leg prices from the peak season instead of
        # reading missing as the player (M1 form, as QB's
        # branch has always done).
        seasons = [
            {"season": 2022, "receptions": 50,
             "receiving_yards": 700, "yds_per_reception": 14.0},
            {"season": 2023, "receptions": 0, "receiving_yards": 0},
        ]
        eff = _te_eff(seasons)
        assert eff is not None and eff > 90.0

    def test_no_rate_evidence_anywhere_is_none(self):
        seasons = [
            {"season": 2022, "receptions": 0, "receiving_yards": 0},
            {"season": 2023, "receptions": 0, "receiving_yards": 0},
        ]
        assert _te_eff(seasons) is None

    def test_inseason_latest_shrinks_toward_completed(self):
        # Completed 2024 at YPR 14 (scale 100); in-progress
        # 2025 at YPR 8 (7.69) over 6 games shrinks halfway
        # toward the completed read (53.85) BEFORE the blend:
        # 100 x 0.90 + 53.85 x 0.10, consistency -2.73 -> 92.66.
        # Untreated, the same seasons price 88.04.
        seasons = [
            {"season": 2024, "receptions": 50,
             "receiving_yards": 700, "yds_per_reception": 14.0,
             "games_played": 13},
            {"season": 2025, "receptions": 30,
             "receiving_yards": 240, "yds_per_reception": 8.0,
             "games_played": 6},
        ]
        raw = _te_eff(seasons)
        shrunk = _te_eff(seasons, _inseason_year=2025)
        assert raw == pytest.approx(88.04, abs=0.4)
        assert shrunk == pytest.approx(92.66, abs=0.4)
        assert shrunk > raw

    def test_lone_inseason_season_shrinks_toward_neutral(self):
        seasons = [
            {"season": 2025, "receptions": 30,
             "receiving_yards": 420, "yds_per_reception": 14.0,
             "games_played": 6},
        ]
        # 52 + 0.5 x (100 - 52) = 76.
        assert _te_eff(seasons, _inseason_year=2025) == \
            pytest.approx(76.0, abs=0.3)

    def test_wr_and_rb_reads_are_unchanged(self):
        # The best-season treatment is TE-scoped: WR still
        # prices the latest season alone (YPR 11 -> 22.73 on
        # the 8.5-19.5 scale, consistency -2.41), and RB the
        # latest season's YPC/YPR pair.
        wr = [
            {"season": 2022, "receptions": 50,
             "receiving_yards": 900, "yds_per_reception": 18.0},
            {"season": 2023, "receptions": 40,
             "receiving_yards": 440, "yds_per_reception": 11.0},
        ]
        assert pm.calc_efficiency_score(
            copy.deepcopy(wr), "WR") == pytest.approx(
                22.73 - 2.41, abs=0.3)
        rb = [
            {"season": 2022, "rush_attempts": 200,
             "rush_yards": 1300, "yds_per_carry": 6.5,
             "receptions": 20, "receiving_yards": 160},
            {"season": 2023, "rush_attempts": 150,
             "rush_yards": 750, "yds_per_carry": 5.0,
             "receptions": 15, "receiving_yards": 120},
        ]
        eff = pm.calc_efficiency_score(copy.deepcopy(rb), "RB")
        # Round 4 superseded the latest-only read this pin was
        # written under (A1's scope guard), blending the pair
        # values 0.90 x higher + 0.10 x lower; round 7 prices
        # the 2022 pair value at parity (72.86 - 12.86 x 0.35
        # = 68.36; the 2023 value 42.86 sits under the parity
        # floor and is unchanged), so the blend reads
        # 0.90 x 68.36 + 0.10 x 42.86 = 65.81, consistency
        # (5.0 vs avg 5.75) -1.30 -> 64.50. The WR assertion
        # above still guards A1's TE-only scope.
        assert eff == pytest.approx(64.50, abs=0.4)


class TestQBExperienceMissingGP:
    """Round 3 A3 (M1 doctrine): a QB with no games-started /
    games-played evidence on any season has an unmeasured
    experience leg (None), not a measured 0.0. Recorded evidence
    of any kind still prices exactly as before."""

    def test_no_games_evidence_returns_none(self):
        seasons = [
            {"season": 2022, "pass_attempts": 300,
             "pass_yards": 2800},
            {"season": 2023, "pass_attempts": 420,
             "pass_yards": 3900},
        ]
        assert pm.calc_experience_score(seasons, "QB") is None

    def test_recorded_zero_starts_with_gp_prices_from_gp(self):
        # Appeared but never started: starts recorded 0, GP
        # recorded - the fallback prices GP (25 -> 62.5).
        seasons = [
            {"season": 2022, "games_started": 0,
             "games_played": 12},
            {"season": 2023, "games_started": 0,
             "games_played": 13},
        ]
        assert pm.calc_experience_score(seasons, "QB") == \
            pytest.approx(62.5)

    def test_starts_without_gp_price_from_starts(self):
        seasons = [
            {"season": 2022, "games_started": 13},
            {"season": 2023, "games_started": 13},
        ]
        assert pm.calc_experience_score(seasons, "QB") == \
            pytest.approx(65.0)

    def test_recorded_zero_everywhere_is_a_measured_zero(self):
        seasons = [
            {"season": 2023, "games_started": 0,
             "games_played": 0},
        ]
        assert pm.calc_experience_score(seasons, "QB") == 0.0

    def test_non_qb_unaffected(self):
        seasons = [{"season": 2023, "rush_attempts": 200}]
        assert pm.calc_experience_score(seasons, "RB") == 0.0

    def test_gp_missing_qb_scores_with_null_experience(self):
        # End to end: the payload carries None (the UI hides it)
        # and the blend runs on the remaining legs - no crash
        # through the _priced append or the payload round().
        prospect = {
            "player_id": "TEST_NOGP_QB", "name": "No GP QB",
            "position": "QB",
            "seasons": [
                {"season": 2023, "team": "State",
                 "pass_attempts": 420, "completions": 280,
                 "pass_yards": 3600, "pass_td": 30,
                 "interceptions": 8, "rush_attempts": 60,
                 "rush_yards": 300},
            ],
        }
        sc = pm.score_prospect(copy.deepcopy(prospect), None,
                               skip_sagarin=True)
        assert sc["experience_score"] is None
        assert sc["prospect_score"] is not None


def _rb_season(season, att, yds, rec=0, rec_yds=0, **kw):
    s = {"season": season, "rush_attempts": att, "rush_yards": yds,
         "yds_per_carry": (yds / att) if att else None,
         "receptions": rec, "receiving_yards": rec_yds}
    if rec:
        s["yds_per_reception"] = rec_yds / rec
    s.update(kw)
    return s


class TestRBEfficiencyBlend:
    """Round 4 (Phase B, part 1): the RB efficiency leg prices
    per featured season - latest + peak by rush yards - and
    blends 0.90 x higher + 0.10 x lower, production's treatment,
    instead of reading the latest season alone."""

    def test_baugh_shape_prices_off_the_best_season(self):
        # Two featured seasons: peak (190 att, 5.32 YPC) and a
        # latest dip (200 att, 4.98 YPC). Latest-only priced the
        # dip's pair value 42.46; the blend prices
        # 0.9 x 49.26 + 0.1 x 42.46 = 48.58 (pair values:
        # YPC scales 52.0 / 42.29 at 0.70, YPR scale 42.86 at
        # 0.30 both seasons).
        seasons = [
            _rb_season(2025, 190, 1011, rec=20, rec_yds=160),
            _rb_season(2026, 200, 996, rec=25, rec_yds=200),
        ]
        val, form = pm._rb_efficiency_of(seasons[1])
        assert form == "ypc"
        assert val == pytest.approx(42.46, abs=0.3)
        eff = pm.calc_efficiency_score(copy.deepcopy(seasons), "RB")
        # 48.58 less the consistency drag (latest YPC 4.98 vs
        # two-season avg 5.15 -> -0.33) and no volume shrink
        # (48.25 < 85): the leg lands in the high 40s, off the
        # best season's body of work - not the 33-42 crater the
        # latest-only read produced for this shape.
        assert eff == pytest.approx(48.25, abs=0.6)

    def test_latest_is_peak_blends_to_the_single_read(self):
        seasons = [_rb_season(2026, 200, 1240, rec=30, rec_yds=270)]
        single = pm.calc_efficiency_score(
            copy.deepcopy(seasons), "RB")
        val, _ = pm._rb_efficiency_of(seasons[0])
        assert single == pytest.approx(val, abs=0.01)


class TestRBEfficiencyComposite:
    """Round 4 (Phase B, part 2): the rush-quality composite
    behind the coverage gate (RB_EFF_CFG weights 0.40 YPC /
    0.20 exp20 / 0.10 exp12 / 0.30 inverted stuff_ctx)."""

    _COVERED = dict(pbp_rush_att=217, pbp_team_games=13,
                    pbp_exp12_rate=0.132, pbp_exp20_rate=0.065,
                    pbp_stuff_ctx=0.0)

    def test_composite_prices_a_covered_season(self):
        # YPC 6.0 -> 71.43; exp20 .065 -> 76.92; exp12 .132 ->
        # 68.57; ctx 0.0 -> inverted 36.47. rush_q = 61.75;
        # YPR 11.7 -> 95.71; value = 61.75 x .70 + 95.71 x .30
        # = 71.94.
        s = _rb_season(2017, 217, 1302, rec=50, rec_yds=585,
                       **self._COVERED)
        val, form = pm._rb_efficiency_of(s)
        assert form == "composite"
        assert val == pytest.approx(71.94, abs=0.4)

    def test_gate_blocks_on_team_games(self):
        s = _rb_season(2017, 217, 1302, rec=50, rec_yds=585,
                       **{**self._COVERED, "pbp_team_games": 10})
        assert pm._rb_efficiency_of(s) == (None, "blocked")

    def test_gate_blocks_on_attempt_ratio(self):
        # A whole missing game: reconstructed attempts reach
        # only 80% of the stored attempts.
        s = _rb_season(2017, 217, 1302, rec=50, rec_yds=585,
                       **{**self._COVERED, "pbp_rush_att": 174})
        assert pm._rb_efficiency_of(s) == (None, "blocked")

    def test_gate_blocks_below_the_calibration_sample(self):
        # 23 reconstructed carries pass coverage but sit far
        # below the >=100-carry calibration population.
        s = _rb_season(2015, 23, 145, rec=1, rec_yds=11,
                       pbp_rush_att=23, pbp_team_games=13,
                       pbp_exp12_rate=0.174, pbp_exp20_rate=0.043,
                       pbp_stuff_ctx=-0.122)
        assert pm._rb_efficiency_of(s) == (None, "blocked")

    def test_missing_stuff_ctx_renormalizes_out(self):
        s = _rb_season(2017, 217, 1302, rec=50, rec_yds=585,
                       **{**self._COVERED, "pbp_stuff_ctx": None})
        val, form = pm._rb_efficiency_of(s)
        assert form == "composite"
        # rush_q over the three remaining terms renormalized:
        # (71.43 x .40 + 76.92 x .20 + 68.57 x .10) / .70 = 72.59
        # value = 72.59 x .70 + 95.71 x .30 = 79.52
        assert val == pytest.approx(79.52, abs=0.4)

    def test_all_candidates_blocked_falls_back_to_pair_read(self):
        seasons = [
            _rb_season(2016, 272, 1496, rec=40, rec_yds=360,
                       **{**self._COVERED, "pbp_team_games": 9}),
            _rb_season(2017, 217, 1271, rec=50, rec_yds=585,
                       **{**self._COVERED, "pbp_team_games": 9}),
        ]
        eff = pm.calc_efficiency_score(copy.deepcopy(seasons), "RB")
        pair, _ = pm._rb_efficiency_of(
            {k: v for k, v in seasons[1].items()
             if not k.startswith("pbp_")})
        # The pre-rebuild read of the latest season, plus the
        # leg-level consistency adjustment (latest YPC 5.86 vs
        # avg 5.68 -> +0.31): the fallback is today's path.
        assert eff == pytest.approx(pair + 0.31, abs=0.5)

    def test_no_rates_anywhere_is_the_pair_blend(self):
        s = _rb_season(2017, 217, 1302, rec=50, rec_yds=585)
        val, form = pm._rb_efficiency_of(s)
        assert form == "ypc"
        # The raw pair value is 71.43 x 0.70 + 95.71 x 0.30 =
        # 78.71; round 7's parity discount takes 35% of the
        # excess over 60: 78.71 - 18.71 x 0.35 = 72.16.
        assert val == pytest.approx(72.16, abs=0.4)


# ── 3.11 round 5: the availability drop (double injury count) ──
# Kaedon 2026-11-24: "fix the double injury count." A featured
# season cut short by injury entered the production blend at
# the latest-season share while the durability leg priced the
# same missed games again - one absence, two charges. The
# blend now drops the latest-season share when the latest
# featured season is an availability season (recorded GP at or
# below the cfg line) AND is not the player's best priced
# season: production prices what he did when he played,
# durability alone prices whether he played. A short season
# that IS the peak (Pitts' 8-game 2020) is his best evidence
# and is never deleted.

def _prod_leg_avail(prospect, pos="TE", **av_over):
    saved = pm.PROD_AVAILABILITY_CFG
    pm.PROD_AVAILABILITY_CFG = {**saved, **av_over}
    try:
        gated = pm._role_gated_seasons(
            [s for s in prospect["seasons"]
             if pm.season_has_counting_stats(s)], pos)
        return pm.calc_production_score(
            copy.deepcopy(gated), pos, skip_sagarin=True,
            _draft_year=prospect["draft_class_year"],
            _age=prospect["age"])
    finally:
        pm.PROD_AVAILABILITY_CFG = saved


# Best season 2022 (900 yds), a strong 2021, and a 6-game 2023
# that prices well below both - the Bowers shape, miniaturized.
AVAIL_DROP = {
    "player_id": "T_availdrop", "name": "Short latest TE",
    "position": "TE", "draft_class_year": 2024, "age": 22.0,
    "seasons": [
        {"season": 2021, "team": "Georgia", "conference": "SEC",
         "games_played": 14, "receptions": 50,
         "receiving_yards": 800, "receiving_tds": 8},
        {"season": 2022, "team": "Georgia", "conference": "SEC",
         "games_played": 13, "receptions": 60,
         "receiving_yards": 900, "receiving_tds": 9},
        {"season": 2023, "team": "Georgia", "conference": "SEC",
         "games_played": 6, "receptions": 24,
         "receiving_yards": 240, "receiving_tds": 1},
    ],
}

# The Pitts shape: the short season (8 GP) is the latest AND
# the best priced AND the volume peak. It must be retained.
AVAIL_BEST = {
    "player_id": "T_availbest", "name": "Short best TE",
    "position": "TE", "draft_class_year": 2021, "age": 21.5,
    "seasons": [
        {"season": 2019, "team": "Florida", "conference": "SEC",
         "games_played": 13, "receptions": 45,
         "receiving_yards": 600, "receiving_tds": 5},
        {"season": 2020, "team": "Florida", "conference": "SEC",
         "games_played": 8, "receptions": 43,
         "receiving_yards": 770, "receiving_tds": 12},
    ],
}

# A full 13-game latest season priced below the best: the
# blend must be byte-identical to the pre-round-5 behavior.
AVAIL_FULL = {
    "player_id": "T_availfull", "name": "Full latest TE",
    "position": "TE", "draft_class_year": 2024, "age": 22.0,
    "seasons": [
        {"season": 2022, "team": "Georgia", "conference": "SEC",
         "games_played": 13, "receptions": 60,
         "receiving_yards": 900, "receiving_tds": 9},
        {"season": 2023, "team": "Georgia", "conference": "SEC",
         "games_played": 13, "receptions": 55,
         "receiving_yards": 780, "receiving_tds": 7},
    ],
}


class TestProductionAvailabilityDrop:
    def test_short_not_best_latest_leaves_the_blend(self):
        off = _prod_leg_avail(AVAIL_DROP, enabled=False)
        on = _prod_leg_avail(AVAIL_DROP)
        # The drag is removed: production rises toward the best
        # season alone.
        assert on > off
        # And the short season is OUT of the blend, not merely
        # discounted: weakening it further (still a role season,
        # still priced below the best) cannot move the leg.
        weaker = copy.deepcopy(AVAIL_DROP)
        weaker["seasons"][2].update(
            {"receptions": 20, "receiving_yards": 180,
             "receiving_tds": 0})
        on_weaker = _prod_leg_avail(weaker)
        off_weaker = _prod_leg_avail(weaker, enabled=False)
        assert on_weaker == pytest.approx(on, abs=0.01)
        assert off_weaker < off

    def test_short_best_season_is_retained_in_full(self):
        off = _prod_leg_avail(AVAIL_BEST, enabled=False)
        on = _prod_leg_avail(AVAIL_BEST)
        assert on == pytest.approx(off, abs=0.01)

    def test_full_latest_season_is_byte_identical(self):
        off = _prod_leg_avail(AVAIL_FULL, enabled=False)
        on = _prod_leg_avail(AVAIL_FULL)
        assert on == pytest.approx(off, abs=0.01)
        # A missing GP is not evidence of absence: no drop.
        nogp = copy.deepcopy(AVAIL_FULL)
        del nogp["seasons"][1]["games_played"]
        assert _prod_leg_avail(nogp) == pytest.approx(
            _prod_leg_avail(nogp, enabled=False), abs=0.01)


# ─────────────────────────────────────────────────────────────
# Round 6: peak-set eligibility (step A), utilization
# best+latest (step B), the production anchor key (step C1),
# and the production forced slot swept and kept (step C2).
# Kaedon's ruling (2026-11-25): "bowers should be higher and
# rb is still bad." Evidence: S14 of the ceiling-attainability
# note. MODEL_VERSION stays "3.11".
# ─────────────────────────────────────────────────────────────


def _with_blend_cfg(fn, **over):
    saved = pm.BLEND_CFG
    pm.BLEND_CFG = {**saved, **over}
    try:
        return fn()
    finally:
        pm.BLEND_CFG = saved


class TestPeakEligibility:
    """Step A: utilization keeps its full mean weight but
    cannot occupy a peak slot (BLEND_CFG peak_exclude)."""

    def test_utilization_cannot_occupy_a_peak_slot(self):
        legs = _priced(WR_W, production=90, utilization=100,
                       efficiency=95, breakout=92, age=88,
                       competition=70)
        names = [t[0] for t in pm._peak_set(legs)]
        assert "utilization" not in names
        # The next eligible leg takes the fourth slot.
        assert names == ["efficiency", "breakout", "production",
                         "age"]

    def test_exclusion_is_inert_when_utilization_is_not_top4(self):
        # Utilization fifth by value: the peak set is the same
        # with and without the exclusion, so the grade is
        # byte-identical.
        legs = _priced(WR_W, production=95, utilization=60,
                       efficiency=94, breakout=93, age=92,
                       competition=91)
        on = pm._blend_grade(legs, "WR")
        off = _with_blend_cfg(
            lambda: pm._blend_grade(legs, "WR"), peak_exclude=())
        assert on == pytest.approx(off, abs=1e-9)

    def test_utilization_still_counts_in_the_mean(self):
        # A peak-resident utilization no longer double-pays:
        # the spiky shape grades lower with the exclusion than
        # without it, but utilization's value still moves the
        # grade through the mean term.
        legs_hi = _priced(WR_W, **SPIKY_WR)
        legs_lo = _priced(WR_W, **{**SPIKY_WR, "utilization": 40})
        assert pm._blend_grade(legs_lo, "WR") < \
            pm._blend_grade(legs_hi, "WR")
        off = _with_blend_cfg(
            lambda: pm._blend_grade(legs_hi, "WR"),
            peak_exclude=())
        assert pm._blend_grade(legs_hi, "WR") < off

    def test_production_is_still_forced_into_the_peak_set(self):
        # Step C2 status quo: production fifth by value is
        # admitted anyway, evicting the fourth eligible leg -
        # the anchor the round-6 sweep kept (S14).
        legs = _priced(WR_W, production=40, utilization=70,
                       efficiency=95, breakout=94, age=93,
                       competition=92)
        names = [t[0] for t in pm._peak_set(legs)]
        assert "production" in names
        assert "competition" not in names


def _util_leg(seasons, pos="RB", **over):
    saved = pm.UTIL_BLEND_CFG
    pm.UTIL_BLEND_CFG = {**saved, **over}
    try:
        return pm.calc_utilization_score(
            copy.deepcopy(seasons), pos)
    finally:
        pm.UTIL_BLEND_CFG = saved


_UTIL_WORKHORSE = {"season": 2023, "games_played": 12,
                   "rush_attempts": 300, "rush_yards": 1500,
                   "targets": 40, "receptions": 30}
_UTIL_SPLIT = {"season": 2024, "games_played": 12,
               "rush_attempts": 96, "rush_yards": 380,
               "targets": 12, "receptions": 9}


class TestUtilizationBestLatest:
    """Step B: utilization prices 0.9 x higher + 0.1 x lower of
    the latest-season read and the best-season read (best by
    the leg's own volume key), production's form."""

    def test_hurt_final_season_is_repaired(self):
        seasons = [_UTIL_WORKHORSE, _UTIL_SPLIT]
        r_best = pm._utilization_of_season(_UTIL_WORKHORSE, "RB")
        r_latest = pm._utilization_of_season(_UTIL_SPLIT, "RB")
        assert r_latest < r_best  # fixture sanity
        on = _util_leg(seasons)
        assert on == pytest.approx(
            0.9 * r_best + 0.1 * r_latest, abs=0.01)
        # The pre-round-6 leg read the latest season alone.
        off = _util_leg(seasons, enabled=False)
        assert off == pytest.approx(r_latest, abs=0.01)
        assert on > off

    def test_latest_workload_peak_is_byte_identical(self):
        # The compiler shape: the latest season IS the volume
        # peak, so the blend returns the single read untouched.
        seasons = [_UTIL_SPLIT, {**_UTIL_WORKHORSE,
                                 "season": 2024},
                   ]
        seasons[0] = {**_UTIL_SPLIT, "season": 2023}
        on = _util_leg(seasons)
        off = _util_leg(seasons, enabled=False)
        assert on == pytest.approx(off, abs=1e-9)

    def test_inseason_raw_is_not_blended(self):
        # The F2 composition: in-season, the raw stays the
        # current season's own read, shrunk toward the
        # completed-seasons read; step B prices that baseline
        # (here a single completed season, so the baseline is
        # its plain read) and never the raw - the completed
        # season cannot be counted twice. The Becker shape
        # (pinned since v3.8 in test_prospect_v38).
        completed = {"season": 2025, "games_played": 14,
                     "receptions": 34, "receiving_yards": 679,
                     "receiving_tds": 4, "team": "Indiana",
                     "conference": "Big Ten"}
        current = {"season": 2026, "games_played": 6,
                   "receptions": 30, "receiving_yards": 529,
                   "receiving_tds": 8, "targets": 32,
                   "team": "Indiana", "conference": "Big Ten"}
        out = pm.calc_utilization_score([completed, current],
                                        "WR", _inseason_year=2026)
        assert out == pytest.approx(27.3, abs=0.6)

    def test_missing_latest_evidence_stays_none(self):
        # The latest with-data season records counting stats
        # but no opportunity fields at all: the leg is missing
        # (v3.8 M1), however the earlier seasons read - the
        # blend never manufactures a read from another season
        # when the latest carries no measurement.
        seasons = [_UTIL_WORKHORSE,
                   {"season": 2024, "games_played": 12,
                    "receiving_yards": 140, "receiving_tds": 1}]
        assert _util_leg(seasons) is None


def _prod_leg_key(seasons, pos, **over):
    saved = pm.PROD_PEAK_CFG
    pm.PROD_PEAK_CFG = {**saved, **over}
    try:
        return pm.calc_production_score(
            copy.deepcopy(seasons), pos, skip_sagarin=True)
    finally:
        pm.PROD_PEAK_CFG = saved


# Three WR seasons whose yards peak (2022: 1,140 in 15 GP) and
# rate peak (2021: 1,020 in 12 GP) are different seasons; the
# latest (2023) is neither.
ANCHOR_WR = [
    {"season": 2021, "team": "Texas", "conference": "Big 12",
     "games_played": 12, "receptions": 80, "receiving_yards": 1020,
     "receiving_tds": 9, "targets": 110, "dominator_rating": 0.30},
    {"season": 2022, "team": "Texas", "conference": "Big 12",
     "games_played": 15, "receptions": 95, "receiving_yards": 1140,
     "receiving_tds": 7, "targets": 140, "dominator_rating": 0.27},
    {"season": 2023, "team": "Texas", "conference": "Big 12",
     "games_played": 13, "receptions": 70, "receiving_yards": 840,
     "receiving_tds": 5, "targets": 100, "dominator_rating": 0.22},
]

# Three QB seasons: 2021 and 2022 tie on the rate key (within
# the 1% band) but price far apart (TD:INT), 2023 is a weak
# latest. The tie must break by price.
ANCHOR_QB = [
    {"season": 2021, "team": "State", "conference": "SEC",
     "games_played": 13, "pass_attempts": 470, "completions": 310,
     "pass_yards": 3580, "pass_tds": 38, "interceptions": 8,
     "rush_attempts": 60, "rush_yards": 250, "rush_tds": 4},
    {"season": 2022, "team": "State", "conference": "SEC",
     "games_played": 13, "pass_attempts": 480, "completions": 300,
     "pass_yards": 3600, "pass_tds": 20, "interceptions": 12,
     "rush_attempts": 55, "rush_yards": 180, "rush_tds": 2},
    {"season": 2023, "team": "State", "conference": "SEC",
     "games_played": 12, "pass_attempts": 380, "completions": 230,
     "pass_yards": 2600, "pass_tds": 15, "interceptions": 10,
     "rush_attempts": 40, "rush_yards": 120, "rush_tds": 1},
]


class TestProductionAnchorKey:
    """Step C1: the production blend's anchor season is chosen
    by the per-game rate key with a price tie-break
    (PROD_PEAK_CFG rate_price), not by raw yardage."""

    def test_anchor_is_the_best_rate_season_not_the_most_yards(self):
        p21 = pm._score_production_season(ANCHOR_WR[0], "WR", True)
        p22 = pm._score_production_season(ANCHOR_WR[1], "WR", True)
        p23 = pm._score_production_season(ANCHOR_WR[2], "WR", True)
        assert p21 > p22 > p23  # fixture sanity
        shipped = _prod_leg_key(ANCHOR_WR, "WR")
        assert shipped == pytest.approx(
            0.9 * p21 + 0.1 * p23, abs=0.05)
        yards = _prod_leg_key(ANCHOR_WR, "WR", by="yards")
        assert yards == pytest.approx(
            0.9 * p22 + 0.1 * p23, abs=0.05)
        assert shipped > yards

    def test_rate_tie_breaks_by_price(self):
        p21 = pm._score_production_season(ANCHOR_QB[0], "QB", True)
        p22 = pm._score_production_season(ANCHOR_QB[1], "QB", True)
        p23 = pm._score_production_season(ANCHOR_QB[2], "QB", True)
        assert p21 > p22 > p23  # fixture sanity
        shipped = _prod_leg_key(ANCHOR_QB, "QB")
        # The 2021 season anchors on the price tie-break even
        # though its rate key is a hair below 2022's.
        assert shipped == pytest.approx(
            0.9 * p21 + 0.1 * p23, abs=0.05)
        pure_rate = _prod_leg_key(ANCHOR_QB, "QB", by="rate")
        assert pure_rate == pytest.approx(
            0.9 * p22 + 0.1 * p23, abs=0.05)
        assert shipped > pure_rate


class TestRound6ShippedCfgs:
    def test_shipped_round6_cfgs(self):
        assert pm.BLEND_CFG["peak_exclude"] == ("utilization",)
        # Step C2 swept and kept: production's forced peak
        # admission stays "always" (S14).
        assert pm.BLEND_CFG["peak_force"] == "always"
        assert pm.UTIL_BLEND_CFG == {"enabled": True,
                                     "hi_share": 0.9}
        assert pm.PROD_PEAK_CFG["by"] == "rate_price"
        assert pm.PROD_PEAK_CFG["tie_band"] == 0.01


# ── 3.11 round 7: pair-form parity + the trim floor ────────
# Kaedon (2026-11-25): Javonte, Harvey and Mixon are too high;
# Barkley and Bijan should hold the top-5 spots instead. The
# diagnostic (S15) found two general defects, neither fitted
# to the names:
# 1. Efficiency form asymmetry. The RB leg prices on the
#    play-by-play composite where coverage passes, on the
#    pre-rebuild pair (YPC/YPR only) where a season has no
#    rates at all - and the PBP build ends at 2023, so the
#    seasons pricing the top of the board can be pair-only
#    (Javonte's 2020, Harvey's 2024). On the 502 recorded
#    seasons carrying both forms the pair overstates the
#    composite by a pooled margin growing with the print
#    (median +9.6 at 85+). EFF_FORM_PARITY_CFG discounts the
#    pair form's excess over 60 by that pooled share.
# 2. The trim deleted a failed trait. Harvey's age leg
#    (19.4 - a 24-year-old draftee) was his trim drop: the
#    oldest prospect on the board paid no age price in the
#    mean term at all. BLEND_CFG's trim_drop_floor makes a
#    leg under 40 ineligible for the drop - forgiveness is
#    for a weak box (40-75), not a crater.

RB_W = pm.POSITION_WEIGHTS["RB"]


class TestPairFormParity:
    def test_shipped_cfg(self):
        cfg = pm.EFF_FORM_PARITY_CFG
        assert cfg["enabled"] is True
        assert cfg["floor"] == pytest.approx(60.0)
        assert cfg["discount"] == pytest.approx(0.35)

    def test_elite_pair_print_is_discounted(self):
        # Javonte's 2020 shape: 7.26 YPC and 12.2 YPR price
        # the raw pair at the 100 clip; parity credits
        # 100 - 40 x 0.35 = 86.0.
        s = _rb_season(2020, 157, 1140, rec=25, rec_yds=305)
        val, form = pm._rb_efficiency_of(s)
        assert form == "ypc"
        assert val == pytest.approx(86.0, abs=0.05)

    def test_mid_pair_print_is_untouched(self):
        # 54.86 sits under the parity floor: byte-identical.
        s = _rb_season(2019, 150, 840, rec=15, rec_yds=120)
        val, form = pm._rb_efficiency_of(s)
        assert form == "ypc"
        assert val == pytest.approx(54.86, abs=0.05)

    def test_composite_print_is_untouched(self):
        # A gate-passing season prices on the composite
        # exactly as in round 4 (71.94); parity is pair-only.
        s = _rb_season(2017, 217, 1302, rec=50, rec_yds=585,
                       **TestRBEfficiencyComposite._COVERED)
        val, form = pm._rb_efficiency_of(s)
        assert form == "composite"
        assert val == pytest.approx(71.94, abs=0.4)

    def test_disabled_is_byte_identical(self, monkeypatch):
        monkeypatch.setattr(
            pm, "EFF_FORM_PARITY_CFG",
            {**pm.EFF_FORM_PARITY_CFG, "enabled": False})
        s = _rb_season(2020, 157, 1140, rec=25, rec_yds=305)
        val, _ = pm._rb_efficiency_of(s)
        assert val == pytest.approx(100.0, abs=0.05)


class TestTrimDropFloor:
    def test_sub_floor_age_is_not_the_drop(self):
        # Harvey's leg shape: age 19.4 is the lowest leg by
        # far. Under the floor the drop goes to the weakest
        # eligible box (athleticism 72.82, utilization blocked
        # the way score_prospect blocks it) and age stays in
        # the mean at full price: the grade reads ~2.7 lower
        # than the same legs with the floor switched off.
        legs = _priced(RB_W, production=97.98, utilization=62.02,
                       efficiency=93.08, breakout=74.76,
                       age=19.40, athleticism=72.82,
                       competition=76.36, environment=51.19,
                       durability=100.0)
        floored = pm._blend_grade(legs, "RB",
                                  trim_block=("utilization",))
        saved = pm.BLEND_CFG
        pm.BLEND_CFG = {**saved, "trim_drop_floor": 0.0}
        try:
            unfloored = pm._blend_grade(
                legs, "RB", trim_block=("utilization",))
        finally:
            pm.BLEND_CFG = saved
        # 3.12 re-pin: the peak min-weight (0.02) now keeps the
        # durability 100 (weight 0.0063) out of the peak set in
        # BOTH arms, so both read lower than the 3.11 pins
        # (88.27 / 90.97); the floor gap is what this test pins.
        assert floored == pytest.approx(84.55, abs=0.1)
        assert unfloored == pytest.approx(87.26, abs=0.1)
        assert floored < unfloored - 2.0

    def test_weak_box_above_the_floor_still_drops(self):
        # The spiky WR's competition (60) is a weak box, not
        # a failed trait: the round-6 pin is byte-identical
        # under the floor.
        g = pm._blend_grade(_priced(WR_W, **SPIKY_WR), "WR")
        assert g == pytest.approx(95.92, abs=0.05)

    def test_sub_floor_performance_leg_is_protected_too(self):
        # The floor is a value line, not a leg list: a 35
        # efficiency (a cratered rate season) cannot be the
        # drop either - the 60 competition box takes it, and
        # the grade reads lower than the unfloored blend.
        legs = _priced(WR_W, production=100, utilization=100,
                       efficiency=35, breakout=98.8, age=85,
                       competition=60, environment=70,
                       recruiting=73)
        floored = pm._blend_grade(legs, "WR")
        saved = pm.BLEND_CFG
        pm.BLEND_CFG = {**saved, "trim_drop_floor": 0.0}
        try:
            unfloored = pm._blend_grade(legs, "WR")
        finally:
            pm.BLEND_CFG = saved
        assert floored < unfloored


# ── 3.11 round 8: the QB rushing price + peak residency ───
# Discrepancy audit P3 (S16): QB production priced passing
# volume with a 25% rushing garnish inside a max(), so a
# dual-threat season's rushing output could never stand as
# production in its own right (Allen production 2.41, Lamar
# 42.87 on the frame, against outcome ranks #1 / #7). The
# shipped dose raises the blend share to 0.55 - the last
# dose with the QB Poor tier and pool median byte-identical
# (0.65 crosses Joshua Dobbs into the Poor tier). The
# total-offense third read (offense_max) posted the round's
# best raw gate but crossed Dobbs at every dose: held
# behind its flag as Kaedon's ruling. The peak-residency
# exclusion for QB context legs was swept and not shipped
# (it taxes the dual-threat family); the machinery stays
# behind BLEND_CFG peak_exclude_by_pos, default empty.

def _qb_season(pass_yds, pass_tds, rush_yds, rush_tds, gp=12):
    return {"season": 2020, "team": "State", "conference": "SEC",
            "games_played": gp, "pass_attempts": 360,
            "completions": 240, "pass_yards": pass_yds,
            "pass_tds": pass_tds, "interceptions": 8,
            "completion_pct": 66.7, "yds_per_attempt": 8.0,
            "td_int_ratio": pass_tds / 8.0,
            "rush_attempts": 150, "rush_yards": rush_yds,
            "rush_tds": rush_tds, "receptions": 0,
            "receiving_yards": 0, "receiving_tds": 0,
            "targets": None, "dominator_rating": 0.10}


def _qb_prod(season, **cfg_over):
    saved = pm.QB_PROD_CFG
    pm.QB_PROD_CFG = {**saved, **cfg_over}
    try:
        return pm._score_production_season(season, "QB", True)
    finally:
        pm.QB_PROD_CFG = saved


# Lamar-shaped: ordinary passing volume, generational legs.
DUAL_THREAT = _qb_season(2400, 22, 1080, 13)
# Burrow-shaped: a perfect passing season, no rushing.
PURE_PASSER = _qb_season(4200, 36, 60, 1)


class TestQBRushingPrice:
    def test_shipped_cfg(self):
        cfg = pm.QB_PROD_CFG
        assert cfg["rush_share"] == pytest.approx(0.55)
        assert cfg["offense_max"] is False
        assert cfg["rush_td_weight"] == 0.0

    def test_dual_threat_prices_his_rushing(self):
        shipped = _qb_prod(DUAL_THREAT)
        legacy = _qb_prod(DUAL_THREAT, rush_share=0.25)
        # Analytic: pass_comp from the V5 anchors, rush_comp
        # capped at 100, the 1.06 elite-rush multiplier.
        pass_comp = (pm._scale(200.0, 150, 285) * 0.5
                     + pm._scale(22 / 12, 1.5, 2.8) * 0.5)
        expect = (pass_comp * 0.45 + 100.0 * 0.55) * 1.06
        assert shipped == pytest.approx(expect, abs=0.1)
        assert shipped > legacy + 15.0

    def test_pure_passer_is_untouched_by_the_share(self):
        # The max() form: a perfect passing season is a 100
        # at any rush share (the R7-A1 idiom).
        assert _qb_prod(PURE_PASSER) == pytest.approx(
            100.0, abs=0.05)
        assert _qb_prod(PURE_PASSER, rush_share=0.25) == \
            pytest.approx(100.0, abs=0.05)

    def test_rushing_alone_cannot_carry_the_price(self):
        # A run-only season (weak arm, elite legs) prices
        # through the blend, well below the pure passer: the
        # share is a partnership, not a replacement.
        run_only = _qb_season(1440, 10, 1140, 12)
        dual = _qb_prod(DUAL_THREAT)
        assert _qb_prod(run_only) < dual
        assert _qb_prod(run_only) < 75.0

    def test_offense_max_is_held_behind_its_flag(self):
        # The total-offense read exists and lifts the
        # dual-threat season over the LEGACY form (at the
        # shipped 0.55 share the blend already prices him
        # past it), but ships OFF (S16: it crossed Dobbs
        # into the Poor tier at every dose).
        off = _qb_prod(DUAL_THREAT, offense_max=True,
                       rush_share=0.25)
        legacy = _qb_prod(DUAL_THREAT, rush_share=0.25)
        assert off > legacy + 15.0
        # And it leaves the pure passer at his passing price.
        assert _qb_prod(PURE_PASSER, offense_max=True) == \
            pytest.approx(100.0, abs=0.05)


class TestPeakExcludeByPos:
    def test_default_empty_is_inert(self):
        legs = _priced(QB_W, production=90, efficiency=80,
                       age=70, competition=100, experience=100,
                       utilization=60)
        assert pm.BLEND_CFG["peak_exclude_by_pos"] == {}
        assert pm._blend_grade(legs, "QB") == pytest.approx(
            pm._blend_grade(legs, "QB"), abs=1e-9)

    def test_excluded_leg_keeps_mean_but_leaves_peak(self):
        legs = _priced(QB_W, production=90, efficiency=80,
                       age=70, competition=100, experience=100,
                       utilization=60)
        plain = pm._blend_grade(legs, "QB")
        saved = pm.BLEND_CFG
        pm.BLEND_CFG = {**saved, "peak_exclude_by_pos":
                        {"QB": ("experience", "competition")}}
        try:
            excluded = pm._blend_grade(legs, "QB")
            # Another position is untouched by the QB entry.
            wr = _priced(WR_W, **SPIKY_WR)
            assert pm._blend_grade(wr, "WR") == pytest.approx(
                95.92, abs=0.05)
        finally:
            pm.BLEND_CFG = saved
        # Two 100 context legs leave the QB peak set: the
        # peak term falls, the mean term still prices them.
        assert excluded < plain


class TestRound9Machinery:
    """Round 9 (audit P2/P4): the utilization de-saturation
    and evidence-cap corroboration machinery ships INERT
    (legacy defaults) - the gate frame cannot price WR/TE
    utilization at all (v3.2 V2/V7 pops its weight whenever
    the latest season lacks targets, and no frame season
    carries targets), and the corroboration release failed
    its sweep on elite collateral. These pins hold the
    defaults and the machinery's math for the day either is
    ruled on."""

    def test_util_cfg_defaults_are_the_legacy_anchors(self):
        assert pm.UTIL_CFG["wr_rec_hi"] == 6.5
        assert pm.UTIL_CFG["te_rec_hi"] == 5.2
        assert pm.UTIL_CFG["sample_denom"] == 0.0
        s = {"games_played": 10, "targets": None, "receptions": 65}
        assert pm._utilization_of_season(s, "WR") == 100.0
        s = {"games_played": 10, "targets": None, "receptions": 52}
        assert pm._utilization_of_season(s, "TE") == 100.0

    def test_util_sample_shrink_math_when_enabled(self):
        season = {"games_played": 5}
        assert pm._util_sample_shrink(100.0, season) == 100.0
        saved = dict(pm.UTIL_CFG)
        try:
            pm.UTIL_CFG["sample_denom"] = 10.0
            assert pm._util_sample_shrink(100.0, season) == 75.0
            assert pm._util_sample_shrink(100.0,
                                          {"games_played": 12}) == 100.0
            assert pm._util_sample_shrink(None, season) is None
        finally:
            pm.UTIL_CFG.clear()
            pm.UTIL_CFG.update(saved)

    def test_release_min_price_inert_by_default(self):
        assert pm.PROD_EVIDENCE_CFG["release_min_price"] == 0.0

    def test_release_min_price_requires_a_real_second_season(self):
        big = {"season": 2022, "games_played": 12,
               "rush_attempts": 300, "rush_yards": 1500,
               "rush_tds": 15, "receptions": 20,
               "receiving_yards": 160, "targets": 25}
        small = {"season": 2021, "games_played": 12,
                 "rush_attempts": 60, "rush_yards": 240,
                 "rush_tds": 2, "receptions": 5,
                 "receiving_yards": 40, "targets": 8}
        seasons = [small, big]
        # 3.11's default was presence release (two featured
        # seasons release the spike untouched). v3.12 ships the
        # corroboration-priced release at RB by default
        # (release_min_price_by_pos RB = 55), and this small
        # season prices below it, so the cap binds at the
        # shipped dose: 75 + (95 - 75) * 0.25 = 80. Presence
        # release remains the default at the other positions
        # (pinned in test_prospect_v312).
        assert pm._production_evidence_cap(
            95.0, seasons, "RB", 21.0, 2023, True) == 80.0
        saved = dict(pm.PROD_EVIDENCE_CFG)
        try:
            pm.PROD_EVIDENCE_CFG["release_min_price"] = 90.0
            capped = pm._production_evidence_cap(
                95.0, seasons, "RB", 21.0, 2023, True)
            assert capped == 75.0 + (95.0 - 75.0) * 0.25
        finally:
            pm.PROD_EVIDENCE_CFG.clear()
            pm.PROD_EVIDENCE_CFG.update(saved)


class TestRound10Machinery:
    """Round 10 (audit P1 + pattern 5): the per-season room
    placement and the discount-once sub-item ship INERT -
    every swept variant failed the gate standard (D1 taxed
    Chase -3.59 and moved the roommate family ~0; D2 doubled
    RB Poor; discount-once worsened RB/WR Poor). Pins hold
    the legacy defaults and the helper's equivalence with
    the legacy adjustment."""

    def test_depth_cfg_default_is_legacy_placement(self):
        assert pm.DEPTH_CFG["per_season"] is False
        assert pm.COMPETITION_CFG["discount_efficiency"] is True

    def test_helper_matches_legacy_adjustment_on_latest(self):
        # Round 12 (R2-replace) retired the adjustment: the
        # helper still computes the legacy tier table
        # (machinery on record), but the value that reaches a
        # grade through calc_depth_chart_adjustment is 1.0
        # while ROOM_CFG replaces it. With the gate off, the
        # legacy equivalence returns exactly.
        seasons = [
            {"season": 2019, "games_played": 12, "depth_rank": 1,
             "position_group_size": 5, "receiving_yards": 900},
            {"season": 2020, "games_played": 12, "depth_rank": 2,
             "position_group_size": 5, "receiving_yards": 840},
        ]
        assert pm._depth_multiplier_for_season(seasons[-1], "WR") > 1.0
        assert pm.calc_depth_chart_adjustment(seasons, "WR") == 1.0
        # Rank 1 or missing data: no credit.
        assert pm._depth_multiplier_for_season(seasons[0], "WR") == 1.0
        assert pm._depth_multiplier_for_season(
            {"games_played": 12, "receiving_yards": 900}, "WR") == 1.0
        saved = dict(pm.ROOM_CFG)
        try:
            pm.ROOM_CFG["replace_depth"] = False
            assert pm.calc_depth_chart_adjustment(seasons, "WR") == \
                pm._depth_multiplier_for_season(seasons[-1], "WR")
        finally:
            pm.ROOM_CFG.clear()
            pm.ROOM_CFG.update(saved)

    def test_per_season_mode_prices_the_room_inside_the_season(self):
        s = {"season": 2020, "games_played": 12, "depth_rank": 2,
             "position_group_size": 5, "receptions": 60,
             "receiving_yards": 840, "receiving_tds": 6,
             "targets": 90}
        base_price = pm._score_production_season(s, "WR", True)
        saved = dict(pm.DEPTH_CFG)
        try:
            pm.DEPTH_CFG["per_season"] = True
            priced = pm._score_production_season(s, "WR", True)
        finally:
            pm.DEPTH_CFG.clear()
            pm.DEPTH_CFG.update(saved)
        mult = pm._depth_multiplier_for_season(s, "WR")
        assert mult > 1.0
        assert priced == min(100.0, base_price * mult)


class TestRound12RoomPrice:
    """Round 12 (Phase B / audit P1, Kaedon's ruling
    2026-11-25): R2-replace ships - the measured room price
    (ROOM_CFG cap .15) is the only room price, and the
    depth-chart adjustment is retired. Swept in round12_lab
    on gate_rows_v8 (control RB .4374/.4393 -> .4415/.4430,
    WR .4029/.4737 -> .4084/.4766, TE .3982/.3674 ->
    .3970/.3719, QB identical; Poor WR 12.9 -> 11.8, median
    68.40 -> 68.87). The two named bar exceptions are
    ACCEPTED outcomes of the ruling, pinned below at the
    mechanism level: Jefferson +3.44 (his LSU 2019 roommate
    price) and the TE perfect fixture 98.67 -> 99.28 (a
    full-strength TE room price)."""

    def test_room_cfg_defaults(self):
        assert pm.ROOM_CFG["cap"] == 0.15
        assert pm.ROOM_CFG["replace_depth"] is True

    def test_room_multiplier_prices_a_suppressed_wr_season(self):
        # Waddle shape: the roommate out-produces him per game.
        s = {"season": 2020, "games_played": 8,
             "receiving_yards": 800, "room_games": 12,
             "room_mate_rec_yds": 1713}   # DeVonta 142.8/g
        mult = pm._room_multiplier_for_season(s, "WR")
        assert mult == 1.0 + 0.15 * pm._scale(1713 / 12, 32, 120) / 100.0
        assert mult > 1.14

    def test_room_multiplier_fails_closed_without_aggregates(self):
        s = {"season": 2020, "games_played": 12,
             "receiving_yards": 840}
        assert pm._room_multiplier_for_season(s, "WR") == 1.0
        s2 = dict(s, room_games=0, room_mate_rec_yds=1500)
        assert pm._room_multiplier_for_season(s2, "WR") == 1.0
        # And the season price is untouched by absent data.
        assert pm._score_production_season(s, "WR", True) == \
            pm._score_production_season(s2, "WR", True)

    def test_room_multiplier_zero_when_player_leads_the_room(self):
        s = {"season": 2022, "games_played": 13,
             "rush_yards": 1500, "receiving_yards": 400,
             "room_games": 13, "room_mate_all_yds": 900}
        assert pm._room_multiplier_for_season(s, "RB") == 1.0

    def test_room_multiplier_does_not_price_qb(self):
        s = {"season": 2020, "games_played": 12,
             "receiving_yards": 100, "room_games": 12,
             "room_mate_rec_yds": 1500}
        assert pm._room_multiplier_for_season(s, "QB") == 1.0

    def test_room_price_applies_inside_the_season_price(self):
        base = {"season": 2020, "games_played": 8, "receptions": 45,
                "receiving_yards": 640, "receiving_tds": 5,
                "targets": 65}
        roomed = dict(base, room_games=12, room_mate_rec_yds=1713)
        plain = pm._score_production_season(base, "WR", True)
        priced = pm._score_production_season(roomed, "WR", True)
        mult = pm._room_multiplier_for_season(roomed, "WR")
        assert priced == min(100.0, plain * mult)
        assert priced > plain

    def test_room_price_exempts_partial_seasons(self):
        s = {"season": 2020, "games_played": 8, "receptions": 50,
             "receiving_yards": 800, "receiving_tds": 6,
             "targets": 70, "room_games": 12,
             "room_mate_rec_yds": 1713}
        bare = {k: v for k, v in s.items()
                if not k.startswith("room_")}
        assert pm._score_production_season(
            s, "WR", True, _partial=True) == \
            pm._score_production_season(bare, "WR", True, _partial=True)

    def test_accepted_exception_jefferson_roommate_price(self):
        # LSU 2019: Chase's 118.7 rec yds/game is the frame's
        # strongest roommate line; Jefferson produced 102.7/g
        # behind him. The price this season earns is what
        # moves Jefferson 92.19 -> 95.63 on gate_rows_v8 -
        # accepted by Kaedon's round-12 ruling.
        s = {"season": 2019, "games_played": 14,
             "receiving_yards": 1438, "room_games": 14,
             "room_mate_rec_yds": 118.7 * 14}
        mult = pm._room_multiplier_for_season(s, "WR")
        assert mult == 1.0 + 0.15 * pm._scale(118.7, 32, 120) / 100.0
        assert mult > 1.14

    def test_accepted_exception_te_full_strength_room_price(self):
        # A TE roommate at the top of the TE anchors prices at
        # full strength (mult 1.15). On gate_rows_v8 this is
        # what moves the TE perfect fixture 98.67 -> 99.28 -
        # accepted by Kaedon's round-12 ruling.
        s = {"season": 2020, "games_played": 12,
             "receiving_yards": 600, "room_games": 12,
             "room_mate_rec_yds": 1200}
        assert pm._room_multiplier_for_season(s, "TE") == 1.15

    def test_depth_bonus_gone_from_a_profile_that_received_it(self):
        # The round-10 fixture's latest season (WR2, 840 yds)
        # earned the 1.12 depth tier under the legacy
        # adjustment; under round 12 it earns nothing there.
        seasons = [
            {"season": 2019, "games_played": 12, "depth_rank": 1,
             "position_group_size": 5, "receiving_yards": 900},
            {"season": 2020, "games_played": 12, "depth_rank": 2,
             "position_group_size": 5, "receiving_yards": 840},
        ]
        assert pm.calc_depth_chart_adjustment(seasons, "WR") == 1.0
