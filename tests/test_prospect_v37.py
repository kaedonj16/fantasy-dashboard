"""Tests for prospect model v3.7: capital-free grades.

Kaedon's directive (2026-11-23): "I want something that doesn't
handle the draft capital, because this is meant to be my own
rankings." The grade is a pure talent grade - draft capital (actual,
projected, consensus, seed) plays NO role in it, and consensus /
mock-draft data goes with capital. Every other leg stays exactly as
3.6 built it, at full strength, sharing the redistributed capital
weight through the model's existing missing-leg redistribution.

THE defining invariant of this version: a prospect's grade is
byte-identical whether capital inputs are present, absent, or wildly
different (pick 1 vs pick 250 vs none). Mechanically, at the
version's first build each 3.7 grade was exactly the 3.6 TALENT grade
of the same profile (the run 3.6 computed with the capital leg
excluded), because 3.7 makes that exclusion unconditional. Panel
decompositions against the pristine 3.6 module (fixtures from
test_prospect_v32):

    SMITH      complete   94.95 -> 92.48   (3.6 capital_delta +2.47)
    SMITH      in-season  91.80 -> 87.79   (3.6 capital_delta +4.01)
    MESTEMAKER complete   81.60 -> 80.18   (3.6 capital_delta +1.42)
    MESTEMAKER in-season  81.15 -> 79.53   (3.6 capital_delta +1.62)
    GREEN      complete   76.09 -> 75.88   (pinned clock, see below)
    GREEN      in-season  75.31 -> 74.80   (pinned clock, see below)

F12 weights fold (2026-11-24): the gate-validated fitted weight set
replaces the design weights in POSITION_WEIGHTS (legs unchanged;
provenance above the tables in prospect_model.py). The panel moves
on weights alone:

    SMITH      complete   92.48 -> 91.93   (production/efficiency shares ease)
    SMITH      in-season  87.79 -> 87.71
    MESTEMAKER complete   80.18 -> 84.69   (QB age eff. share .1009 -> .2012)
    MESTEMAKER in-season  79.53 -> 83.96
    GREEN      complete   75.88 -> 74.81   (pinned clock; pinned-100 efficiency leg eases)
    GREEN      in-season  74.80 -> 73.74   (pinned clock)

GREEN is the one case where the delta is NOT 3.6's printed
capital_delta (+0.33 / +0.69): the 3.6 talent recursion dropped the
caller's _today pin, so its talent grade graded durability on the
run-date calendar while the headline grade used the pinned one. At a
fixed clock the leg's true price was +0.21 / +0.51; the printed
deltas carried 0.12 / 0.18 of that mixed-clock artifact. 3.7 runs a
single pass, so its grade is the clock-consistent talent grade.

Confidences drop by exactly the removed capital-presence points
(+10 for a mock-backed pick, +5 more at 5+ mocks) wherever they
applied: SMITH 70 -> 55 complete / 62 -> 47 in-season, MESTEMAKER
60 -> 45 / 48 -> 33, GREEN 70 -> 55 / 55 -> 40 (pinned clock).

Gate note: like every model change, 3.7 ships only if the backtest
gate clears its bar - which for a capital-free model is 3.5's
TALENT-ONLY (no-capital slice) correlation, not full 3.5.

3.8 (under-rating fixes; the capital invariance this file guards is
untouched - the full payload still matches across all 11 capital
variants): the panel moves on the M2 production blend alone
(best-season weight 85 -> 90; leg decompositions against the
pristine 3.7 module show production as the ONLY moving leg for all
three panelists):

    SMITH      complete   91.93 -> 92.28   (production 91.43 -> 92.55)
    SMITH      in-season  87.71 -> 87.85   (production 87.07 -> 87.52)
    MESTEMAKER complete   84.69 -> 84.78   (production 91.32 -> 91.65)
    MESTEMAKER in-season  83.96 -> 84.00   (production 88.05 -> 88.19)
    GREEN      complete   74.81 -> 74.92   (production 42.55 -> 42.90)
    GREEN      in-season  73.74 -> 73.77   (production 43.33 -> 43.41)

3.9 (ceiling redesign; capital invariance untouched): the final
blend prices a peak term over the weighted mean and the competition
anchor re-bases 1.03 -> 1.00. Panel: SMITH 92.28 -> 93.92 complete
(89.80 in-season), MESTEMAKER 84.78 -> 86.09 (85.27), GREEN
74.92 -> 75.59 (74.66 at the pinned clock, after the shrink step).

3.10 (dom curve bent at 0.65 + draft-anchored breakout age):
SMITH 93.92 -> 94.24 complete (90.48 in-season; breakout leg
88.97 -> 93.41), GREEN 75.59 -> 75.87 complete (74.93 at the
pinned clock; breakout 68.95 -> 72.58), MESTEMAKER unchanged
(the QB breakout weight is popped, so neither lever reaches
his grade).

The M2 role gate (seasons before the first real role excused from
the production / utilization / breakout reads) changes no panel
grade: Smith and Green carry no pre-role seasons, and Mestemaker's
2-game backup season is excused but was moving no leg.

The panel's RB fixture (HARRIS, used by the invariance tests, not
pinned here) moves more, and it is the M1 showcase: his fixture
carries no stored rates, so 3.7 scored his efficiency off the
friendly defaults (the 23.57 template value) and his production
with a 0.0 yds/carry term; 3.8 derives both rates from his counting
stats (efficiency 23.57 -> 61.70, production 53.53 -> 70.64 with
the blend and the M3 share credit inside it; grade 57.47 -> 67.90).
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

_AS_OF = date(2026, 10, 19)  # the v34 panel clock

# A two-season RB panel fixture (the v32 module carries WR/QB/TE).
HARRIS = {
    "player_id": "T_harris", "name": "Test Harris", "position": "RB",
    "draft_class_year": 2027, "age": 21.8,
    "seasons": [
        {"season": 2025, "games_played": 13, "team": "Georgia",
         "conference": "SEC", "rush_attempts": 210, "rush_yards": 1180,
         "rush_tds": 12, "receptions": 28, "receiving_yards": 240,
         "receiving_tds": 2, "dominator_rating": 0.31,
         "team_rush_yards": 2600, "team_total_yards": 6100},
        {"season": 2026, "games_played": 11, "team": "Georgia",
         "conference": "SEC", "rush_attempts": 188, "rush_yards": 1095,
         "rush_tds": 11, "receptions": 31, "receiving_yards": 265,
         "receiving_tds": 1, "dominator_rating": 0.33,
         "team_rush_yards": 2450, "team_total_yards": 5900},
    ],
    "recruiting": {"composite": 0.94, "stars": 4},
}

PANEL = [("WR", SMITH), ("QB", MESTEMAKER), ("TE", GREEN), ("RB", HARRIS)]


def _dc(pos, pick, mocks=5, actual=False):
    return {
        "projected_pick": pick,
        "projected_round": ((pick - 1) // 32) + 1,
        "projected_draft_capital_score": pick_to_draft_capital_score(pick, pos),
        "num_mocks_used": mocks,
        "is_actual_pick": actual,
        "consensus_confidence": 80.0,
    }


def _capital_variants(pos):
    """Every flavour of capital input the pipeline/backtest can pass."""
    seed = _dc(pos, 4, mocks=0)
    seed["mock_sources"] = ["seed_consensus"]
    return {
        "none": None,
        "empty_dict": {},
        "pick_1": _dc(pos, 1),
        "pick_32": _dc(pos, 32),
        "pick_100": _dc(pos, 100),
        "pick_250": _dc(pos, 250),
        "actual_pick_1": _dc(pos, 1, actual=True),
        "actual_pick_200": _dc(pos, 200, actual=True),
        "seed_pick_4": seed,
        "no_pick_but_mocks": {"num_mocks_used": 7, "mock_sources": ["x"]},
        "score_but_no_pick": {"projected_draft_capital_score": 88.0,
                              "num_mocks_used": 3},
    }


def _score(prospect, dc, **kw):
    return pm.score_prospect(copy.deepcopy(prospect), dc,
                             skip_sagarin=True, **kw)


class TestVersion:
    def test_model_version_is_current(self):
        assert pm.MODEL_VERSION == "3.14"


class TestCapitalInvariance:
    """The defining invariant: the FULL payload (grade, every
    component, confidence, archetype, reasons) is identical across
    every capital input - present, absent, or wildly different."""

    @pytest.mark.parametrize("pos,prospect", PANEL,
                             ids=[p for p, _ in PANEL])
    def test_payload_identical_across_capital_variants(self, pos, prospect):
        results = {
            name: _score(prospect, dc)
            for name, dc in _capital_variants(pos).items()
        }
        baseline = results["none"]
        for name, r in results.items():
            assert r == baseline, f"capital variant {name!r} changed the payload"

    @pytest.mark.parametrize("pos,prospect", PANEL,
                             ids=[p for p, _ in PANEL])
    def test_payload_identical_in_season_mode_too(self, pos, prospect):
        results = {
            name: _score(prospect, dc, _active_season_year=2026)
            for name, dc in _capital_variants(pos).items()
        }
        baseline = results["none"]
        for name, r in results.items():
            assert r == baseline, f"capital variant {name!r} changed the payload"

    def test_pick_1_vs_pick_250_vs_none_grade(self):
        # The invariant stated the way Kaedon stated the requirement.
        grades = {
            _score(SMITH, _dc("WR", 1))["prospect_score"],
            _score(SMITH, _dc("WR", 250))["prospect_score"],
            _score(SMITH, None)["prospect_score"],
        }
        assert len(grades) == 1


class TestPayloadContract:
    @pytest.mark.parametrize("pos,prospect", PANEL,
                             ids=[p for p, _ in PANEL])
    def test_capital_fields_are_none_and_talent_is_the_grade(self, pos, prospect):
        r = _score(prospect, _dc(pos, 3))
        assert r["projected_draft_capital_score"] is None
        assert r["capital_delta"] is None
        assert r["talent_score"] == r["prospect_score"]
        assert r["model_version"] == "3.14"

    def test_capital_derived_payload_diagnostics_are_zero(self):
        # late_round_upside's only input was the projected round and
        # the athleticism x capital interaction's was the capital
        # score; both are payload-only diagnostics (neither ever fed
        # the grade) and both are 0.0 now.
        r = _score(GREEN, _dc("TE", 200))
        assert r["late_round_upside"] == 0.0
        assert r["athleticism_draft_capital_interaction"] == 0.0


class TestConfidenceHasNoCapitalPoints:
    def test_mock_count_never_moves_confidence(self):
        confs = {
            _score(SMITH, _dc("WR", 2, mocks=m))["confidence_score"]
            for m in (0, 1, 5, 25)
        }
        confs.add(_score(SMITH, None)["confidence_score"])
        assert len(confs) == 1

    def test_panel_confidence_decomposition(self):
        # 3.6 complete-mode confidence 70.0 included +15 of capital
        # points (10 for a mock-backed pick, 5 at 5+ mocks). Without
        # them the same profile certifies 55.0.
        assert _score(SMITH, _dc("WR", 2))["confidence_score"] == 55.0
        assert _score(MESTEMAKER, _dc("QB", 8))["confidence_score"] == 45.0


class TestPanelGrades:
    """Panel pins under the shipped weights (decompositions in the
    module docstring: 3.6 -> capital-free design weights -> F12
    fitted weights -> 3.8 M2 blend). These pins duplicate the
    updated v34 pins on purpose: they are this version's contract."""

    def test_smith(self):
        # v3.9 ceiling redesign: 92.28 -> 93.92 complete, 87.85 ->
        # 89.80 in-season (peak blend + competition re-anchor; see
        # test_prospect_v34 for the decomposition).
        # v3.10: -> 94.24 complete / 90.48 in-season (breakout leg
        # 88.97 -> 93.41 on the bent dom curve + draft anchor).
        # v3.11: -> 94.35 complete / 90.76 in-season (peak share
        # 0.74; no leg weak enough to trim - see v34).
        # v3.11 round 6: -> 94.66 complete (step C1's rate key
        # reprices his production anchor; steps A/B do not touch
        # him - S14).
        assert _score(SMITH, _dc("WR", 2))["prospect_score"] == 94.66
        inseason = _score(SMITH, _dc("WR", 2), _active_season_year=2026)
        assert inseason["prospect_score"] == 90.76

    def test_mestemaker(self):
        # v3.9 ceiling redesign: 84.78 -> 86.09 complete, 84.00 ->
        # 85.27 in-season (QB peak share 0.30).
        # v3.11: -> 87.41 complete (the gated trim forgives his
        # competition box; QB peak share unchanged), in-season
        # unchanged at 85.27 (the trim gate stays shut).
        # v3.11 round 6: -> 85.90 complete. The whole -1.51 is
        # step A: his utilization (73.86) was peak-resident, so
        # excluding it from the peak set reprices his peak term
        # (steps B/C1 net -0.01/0.00 on his profile - S14).
        # v3.14 re-pin: -> 84.60 complete, 82.97 in-season
        # under per-season discount keying (his North Texas
        # seasons toll at the American's own level instead of
        # his blended profile level).
        assert _score(MESTEMAKER, _dc("QB", 8))["prospect_score"] == 84.60
        inseason = _score(MESTEMAKER, _dc("QB", 8), _active_season_year=2026)
        # Round 6 step A also prices his in-season peak set:
        # 85.27 -> 85.13 (3.14: 82.97, as above).
        assert inseason["prospect_score"] == 82.97

    def test_green_pinned_clock(self):
        complete = _score(GREEN, _dc("TE", 35), _today=_AS_OF)
        inseason = _score(GREEN, _dc("TE", 35), _active_season_year=2026,
                          _today=_AS_OF)
        # v3.9 ceiling redesign: 74.92 -> 75.59 complete.
        # v3.10: -> 75.87 complete (breakout 68.95 -> 72.58).
        # v3.11: -> 75.86 complete (peak prices a hair under his
        # mean at the 0.74 share; no trimmable leg).
        # v3.11 round 5: complete -> 76.12 (production 42.90 ->
        # 43.59: the 3-GP 2026 leaves the blend in complete
        # mode; see test_prospect_v34 for the split).
        # v3.13 (Phase 7, L1): -> 77.14; his targets-absent
        # utilization leg prices again (see test_prospect_v34).
        assert complete["prospect_score"] == 77.14
        # v3.9: 73.77 -> 74.17. At this pinned clock (Oct 19) the
        # season is 6 team games old, above his 3 GP, so the shrink
        # factor doubles (3/12 -> 6/12); see test_prospect_v34 for
        # the leg decomposition.
        # v3.9 ceiling redesign: 74.17 -> 74.66 (SEC competition
        # re-anchor + peak term, production-anchored).
        # v3.10: -> 74.93 (the same breakout-leg move as complete).
        # v3.11: -> 74.88 (the same peak-share cents as complete).
        # v3.11 round 3: -> 75.08. TE efficiency's best-season
        # pricing moves only his in-season read (efficiency
        # 94.51 -> 95.50; see test_prospect_v34 for the split).
        # v3.13 (Phase 7, L1): -> 75.46 (same mechanism).
        assert inseason["prospect_score"] == 75.46

    def test_panel_ordering_unchanged_by_the_fold(self):
        # The round-3 audit panel order (Smith > Mestemaker > Green
        # > Harris) survives the re-weighting; Smith remains the
        # panel's clear top grade.
        grades = {
            "smith": _score(SMITH, None)["prospect_score"],
            "mestemaker": _score(MESTEMAKER, None)["prospect_score"],
            "green": _score(GREEN, None, _today=_AS_OF)["prospect_score"],
            "harris": _score(HARRIS, None)["prospect_score"],
        }
        assert grades["smith"] > grades["mestemaker"] > grades["green"] \
            > grades["harris"]


# The F12 fitted set, exactly as validated by the weights gate
# (~/workspace/prospect-runs/gate_v37_weights.log; the artifact the
# gate consumed is ~/workspace/prospect-runs/fitted_weights.json).
FITTED_WEIGHTS = {
    "QB": {"production": 0.1869, "utilization": 0.0537,
           "efficiency": 0.1891, "age": 0.2012, "breakout": 0.0705,
           "athleticism": 0.1473, "competition": 0.0613,
           "environment": 0.0126, "durability": 0.0,
           "experience": 0.0442, "recruiting": 0.0332},
    "RB": {"production": 0.2305, "utilization": 0.1367,
           "efficiency": 0.1433, "age": 0.1587, "breakout": 0.146,
           "athleticism": 0.0936, "competition": 0.0472,
           "environment": 0.0044, "durability": 0.0063,
           "recruiting": 0.0332},
    "WR": {"production": 0.2382, "utilization": 0.1094,
           "efficiency": 0.1162, "age": 0.1415, "breakout": 0.1369,
           "athleticism": 0.0802, "competition": 0.1037,
           "environment": 0.0393, "durability": 0.0,
           "recruiting": 0.0345},
    "TE": {"production": 0.2189, "utilization": 0.1332,
           "efficiency": 0.1187, "age": 0.1637, "breakout": 0.1363,
           "athleticism": 0.1311, "competition": 0.0432,
           "environment": 0.0164, "durability": 0.0062,
           "recruiting": 0.0324},
}


def _effective_weights(pos):
    """The weights score_prospect prices from the shipped table:
    its own derivation - copy the table, pop draft_capital, spread
    the popped share over the remaining legs. (The tables are stored
    in effective form since the F12 fold, so the pop is a no-op and
    this returns the table; the derivation is spelled out so the pin
    still means 'post-pop effective weights' if a capital entry ever
    reappears in a table.)"""
    w = dict(pm.POSITION_WEIGHTS[pos])
    dc_w = w.pop("draft_capital", 0.0)
    remaining = sum(w.values())
    if remaining > 0 and dc_w > 0:
        for k in w:
            w[k] *= (1.0 + dc_w / remaining)
    return w


class TestFittedWeights:
    """The shipped tables ARE the gated weights: the fold is only
    correct if table mode reproduces the override-mode gate run."""

    def test_tables_carry_no_capital_entry(self):
        for pos, weights in pm.POSITION_WEIGHTS.items():
            assert "draft_capital" not in weights, pos

    def test_effective_weights_equal_the_gated_set(self):
        for pos, fitted in FITTED_WEIGHTS.items():
            eff = _effective_weights(pos)
            assert set(eff) == set(fitted), pos
            for leg, v in fitted.items():
                assert eff[leg] == pytest.approx(v, abs=1e-9), (pos, leg)

    def test_refined_mirror_matches_the_gated_set(self):
        # historical_calibration's REFINED_POSITION_WEIGHTS is the
        # baseline its recommendations blend from; it mirrors the
        # shipped set so calibration tooling can never silently
        # substitute a stale pre-fit blend for the model's weights.
        from data_building.rookie_pipeline.historical_calibration import (
            REFINED_POSITION_WEIGHTS,
        )
        for pos, fitted in FITTED_WEIGHTS.items():
            mirror = REFINED_POSITION_WEIGHTS[pos]
            assert set(mirror) == set(fitted), pos
            for leg, v in fitted.items():
                assert mirror[leg] == pytest.approx(v, abs=1e-9), (pos, leg)

    @pytest.mark.parametrize("pos,prospect", PANEL,
                             ids=[p for p, _ in PANEL])
    def test_table_mode_reproduces_override_mode(self, pos, prospect):
        # The gated configuration (fitted set injected as an
        # override over the old tables) vs the shipped configuration
        # (no override; the tables carry the set): grades must agree
        # to rounding. Verified exactly (max diff 0.00) over the
        # panel plus all 1,945 dumped gate-population profiles at
        # fold time; this pins the panel slice of that equivalence.
        via_table = _score(prospect, None)
        via_override = pm.score_prospect(
            copy.deepcopy(prospect), None, skip_sagarin=True,
            position_weights_override=FITTED_WEIGHTS)
        assert abs(via_table["prospect_score"]
                   - via_override["prospect_score"]) <= 0.05

    def test_table_mode_reproduces_override_mode_pinned_clock(self):
        via_table = _score(GREEN, None, _today=_AS_OF)
        via_override = pm.score_prospect(
            copy.deepcopy(GREEN), None, skip_sagarin=True, _today=_AS_OF,
            position_weights_override=FITTED_WEIGHTS)
        assert abs(via_table["prospect_score"]
                   - via_override["prospect_score"]) <= 0.05


class TestReasonsHaveNoCapitalBullet:
    def test_no_pick_language_even_with_an_actual_pick(self):
        r = _score(SMITH, _dc("WR", 5, actual=True))
        assert "Projected pick" not in r["key_reasons"]
        assert "Drafted:" not in r["key_reasons"]
        assert "mock draft" not in r["key_reasons"]


class TestBatchScorerIgnoresConsensus:
    def test_score_all_prospects_consensus_map_changes_nothing(self):
        prospects = [copy.deepcopy(SMITH), copy.deepcopy(GREEN)]
        consensus = {SMITH["player_id"]: _dc("WR", 1),
                     GREEN["player_id"]: _dc("TE", 250)}
        with_map = pm.score_all_prospects(
            copy.deepcopy(prospects), consensus, skip_sagarin=True)
        without = pm.score_all_prospects(
            copy.deepcopy(prospects), {}, skip_sagarin=True)
        assert with_map == without


class TestLateRoundUpsideRetired:
    """Round-3 audit fix: calc_late_round_upside still READ
    projected_round from any dict it was handed (75-100 for a
    round-4+ projection with an elite dominator/YPRR). It never fed
    the grade and score_prospect already passed None, but a
    capital-free module cannot carry a live capital read. The
    function is retired: 0.0 whatever it is handed."""

    def test_direct_call_with_late_round_dict_is_zero(self):
        seasons = [{"season": 2026, "games_played": 12,
                    "dominator_rating": 0.41, "yards_per_route_run": 3.1}]
        dc = {"projected_round": 6, "projected_pick": 190,
              "num_mocks_used": 9}
        # Pre-fix this returned 95.0 for a WR (90 base + 5 WR bonus).
        assert pm.calc_late_round_upside(dc, seasons, "WR") == 0.0
        assert pm.calc_late_round_upside(dc, seasons, "RB") == 0.0
        assert pm.calc_late_round_upside(dc, seasons, "TE") == 0.0
        assert pm.calc_late_round_upside(None, seasons, "WR") == 0.0
