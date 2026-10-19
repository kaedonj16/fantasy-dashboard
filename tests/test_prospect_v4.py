"""
Tests for the v4 prospect model FOUNDATION (prospect_model_v4.py).

The v4 module is a skeleton whose legs price against recorded
distributions loaded from a reference artifact. There is no recorded
artifact yet (it is produced on Render by
scripts/derive_v4_reference.py), so these tests run against the
committed FIXTURE SAMPLE (v4_reference_fixture_sample.json, derived
from the three surviving fixtures and marked as fixture data in its
provenance) plus small synthetic anchor tables built inline. They
verify the architecture: the knot helper, the ONE season-selection
rule, missing-anchor redistribution, lifecycle states, capital
separation, cast pricing, and end-to-end fixture runs.
"""

import copy
import json
import os
import sys

import pytest

pytest.importorskip("pandas")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_building.rookie_pipeline import prospect_model_v4 as v4  # noqa: E402
from test_prospect_v32 import GREEN, MESTEMAKER, SMITH, _dc  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PATH = os.path.join(
    _REPO, "data_building", "rookie_pipeline",
    "v4_reference_fixture_sample.json")


@pytest.fixture(scope="module")
def sample_ref():
    return v4.load_reference(SAMPLE_PATH)


# A small synthetic (NOT recorded) QB/WR anchor set with mid-range
# marks, so cast and coherence tests can land mid-scale.
SYNTHETIC_REF = {
    "provenance": {"fixture_sample": True, "source": "test synthetic"},
    "anchors": {
        "QB": {
            "pass_yds_pg": [100, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420],
            "pass_tds_pg": [0.5, 1.0, 1.4, 1.7, 2.0, 2.3, 2.6, 3.0, 3.4, 3.8, 4.2],
            "completion_pct": [50, 55, 58, 60, 62, 64, 66, 68, 70, 72, 75],
            "ypa": [5.5, 6.2, 6.8, 7.2, 7.6, 8.0, 8.5, 9.0, 9.6, 10.2, 11.0],
            "td_int": [0.5, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.6, 4.2, 5.0, 6.0],
            "rush_yds_pg": [-5, 0, 5, 10, 15, 22, 30, 40, 55, 70, 90],
            "qb_share": [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
            "breakout_dom": [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
            "traj_growth": [-0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.8, 3.6, 5.0],
            "age_years": [25.5, 24.8, 24.2, 23.7, 23.2, 22.7, 22.2, 21.7, 21.2, 20.7, 20.0],
            "competition_index": [0.50, 0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90, 0.94, 1.00],
            "recruiting_composite": [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.975, 0.99, 0.995, 1.00],
        },
        "WR": {
            "rec_yds_pg": [10, 25, 35, 45, 55, 65, 78, 92, 105, 118, 135],
            "rec_tds_pg": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.78, 0.90, 1.05, 1.25],
            "pass_share": [0.02, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.25, 0.29, 0.33, 0.38],
            "dominator_rating": [0.02, 0.07, 0.11, 0.15, 0.19, 0.23, 0.27, 0.31, 0.36, 0.41, 0.48],
            "market_share_yards": [0.02, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.25, 0.29, 0.33, 0.38],
            "ypr": [8.0, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 19.0],
            "breakout_dom": [0.02, 0.07, 0.11, 0.15, 0.19, 0.23, 0.27, 0.31, 0.36, 0.41, 0.48],
            "traj_growth": [-0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.8, 3.6, 5.0],
            "age_years": [25.5, 24.8, 24.2, 23.7, 23.2, 22.7, 22.2, 21.7, 21.2, 20.7, 20.0],
            "competition_index": [0.50, 0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90, 0.94, 1.00],
            "recruiting_composite": [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.975, 0.99, 0.995, 1.00],
        },
    },
    "percentiles": {},
    "athleticism": {},
    "populations": {},
}


class TestKnotHelper:
    MARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_interpolation_midpoint(self):
        # Between p50 (mark 5 -> 56) and p75 (mark 6 -> 74).
        assert v4.knot_score(5.5, self.MARKS) == pytest.approx(65.0)

    def test_edges_and_beyond(self):
        assert v4.knot_score(0, self.MARKS) == 0.0
        assert v4.knot_score(-3, self.MARKS) == 0.0
        assert v4.knot_score(10, self.MARKS) == 100.0
        assert v4.knot_score(42, self.MARKS) == 100.0

    def test_lower_is_better(self):
        marks = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # performance order
        assert v4.knot_score(0, marks) == 100.0
        assert v4.knot_score(10, marks) == 0.0
        assert v4.knot_score(5, marks) == pytest.approx(56.0)

    def test_tied_marks_earn_highest_tier(self):
        # A value on a fully tied mark reaches every tier it spans.
        assert v4.knot_score(5, [5] * 11) == 100.0
        # Tied at the top: the recorded best earns 100 by definition.
        marks = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]
        assert v4.knot_score(5, marks) == 100.0

    def test_missing_inputs(self):
        assert v4.knot_score(None, self.MARKS) is None
        assert v4.knot_score(5, None) is None
        assert v4.knot_score(5, []) is None
        assert v4.knot_score(5, [1, 2, 3]) is None  # wrong table shape


class TestPercentileOf:
    TABLE = [[0, 10.0], [50, 20.0], [100, 30.0]]

    def test_lookup_and_interpolation(self):
        assert v4.percentile_of(20.0, self.TABLE) == 50.0
        assert v4.percentile_of(25.0, self.TABLE) == 75.0

    def test_outside_range_clamps(self):
        assert v4.percentile_of(5.0, self.TABLE) == 0.0
        assert v4.percentile_of(35.0, self.TABLE) == 100.0

    def test_tied_mark_earns_highest_percentile(self):
        table = [[0, 10.0], [50, 10.0], [100, 30.0]]
        assert v4.percentile_of(10.0, table) == 50.0

    def test_missing(self):
        assert v4.percentile_of(None, self.TABLE) is None
        assert v4.percentile_of(20.0, None) is None


# A Mensah-shaped profile: a high-volume 2025 at Duke and a short,
# hyper-efficient 2026 at Miami. v3.3 priced production on the Miami
# season and efficiency on the Duke season; v4 must price ONE pair.
MENSAH_SHAPED = {
    "player_id": "TEST_MENSAH", "name": "Mensah-Shaped QB",
    "position": "QB", "school": "Miami", "draft_class_year": 2027,
    "age": 22.5, "recruiting": {"composite_rating": 0.89},
    "athleticism": {},
    "seasons": [
        {"season": 2025, "team": "Duke", "conference": "ACC",
         "games_played": 14, "pass_attempts": 500, "completions": 340,
         "pass_yards": 3973, "pass_tds": 34, "interceptions": 12,
         "rush_yards": 120, "rush_tds": 2, "team_total_yards": 6100},
        {"season": 2026, "team": "Miami", "conference": "ACC",
         "games_played": 5, "pass_attempts": 133, "completions": 96,
         "pass_yards": 1438, "pass_tds": 11, "interceptions": 2,
         "rush_yards": 40, "rush_tds": 1, "team_total_yards": 2300},
    ],
}


class TestSeasonSelectionCoherence:
    def test_peak_is_the_highest_rate_season(self):
        sel = v4.select_priced_seasons(
            copy.deepcopy(MENSAH_SHAPED["seasons"]), "QB")
        # 2026: 287.6 pass yds/g > 2025: 283.8 pass yds/g.
        assert sel["peak"]["season"] == 2026
        assert sel["latest"]["season"] == 2026

    def test_partial_season_never_peaks_in_season(self):
        sel = v4.select_priced_seasons(
            copy.deepcopy(MENSAH_SHAPED["seasons"]), "QB",
            partial_year=2026)
        assert sel["partial"] is True
        assert sel["peak"]["season"] == 2025
        assert sel["latest"]["season"] == 2026

    def test_production_and_efficiency_price_the_same_season(self):
        result = v4.score_prospect_v4(
            copy.deepcopy(MENSAH_SHAPED), None, reference=SYNTHETIC_REF)
        assert result["seasons_priced"] == {"peak": 2026, "latest": 2026}
        # Efficiency priced on the 2026 season (YPA 10.8) is elite;
        # priced on 2025 (YPA 7.9) it would land near 60.
        assert result["efficiency_score"] is not None
        assert result["efficiency_score"] > 85
        assert result["production_score"] is not None


class TestMissingAnchorRedistribution:
    def test_only_production_anchors(self):
        ref = {"provenance": {"fixture_sample": True},
               "anchors": {"WR": {"rec_yds_pg": [10, 25, 35, 45, 55, 65,
                                                 78, 92, 105, 118, 135],
                                   "rec_tds_pg": [0.05, 0.15, 0.25, 0.35,
                                                  0.45, 0.55, 0.65, 0.78,
                                                  0.90, 1.05, 1.25],
                                   "pass_share": [0.02, 0.06, 0.09, 0.12,
                                                  0.15, 0.18, 0.21, 0.25,
                                                  0.29, 0.33, 0.38],
                                   "dominator_rating": [0.02, 0.07, 0.11,
                                                        0.15, 0.19, 0.23,
                                                        0.27, 0.31, 0.36,
                                                        0.41, 0.48]}},
               "percentiles": {}, "athleticism": {}}
        result = v4.score_prospect_v4(copy.deepcopy(SMITH), None,
                                      reference=ref)
        assert result["production_score"] is not None
        assert result["durability_score"] is not None
        for leg in ("efficiency", "breakout", "age", "competition",
                    "recruiting"):
            assert leg in result["legs_missing"]
        assert set(result["weights_used"]) == {"production", "durability"}
        assert sum(result["weights_used"].values()) == pytest.approx(1.0)

    def test_empty_reference_leaves_only_anchor_free_legs(self):
        result = v4.score_prospect_v4(copy.deepcopy(SMITH), None,
                                      reference={})
        # Durability needs no anchor table; everything else is missing.
        assert result["talent_score"] == result["durability_score"]
        assert set(result["weights_used"]) == {"durability"}
        assert result["reference_kind"] == "none"


class TestLifecycleStates:
    def test_in_season(self, sample_ref):
        result = v4.score_prospect_v4(copy.deepcopy(SMITH), _dc("WR", 2),
                                      reference=sample_ref,
                                      active_season_year=2026)
        assert result["lifecycle_state"] == "in_season"
        assert result["in_season"] is True
        # Untested athleticism redistributes; it is NOT floored.
        assert result["athleticism_score"] is None
        assert "athleticism" in result["legs_missing"]

    def test_pre_combine_default(self, sample_ref):
        result = v4.score_prospect_v4(copy.deepcopy(SMITH), _dc("WR", 2),
                                      reference=sample_ref)
        assert result["lifecycle_state"] == "pre_combine"
        assert result["in_season"] is False
        assert result["athleticism_score"] is None

    def test_post_combine_with_testing(self, sample_ref):
        prospect = copy.deepcopy(SMITH)
        prospect["athleticism"] = {"forty_yard": 4.40, "weight_lbs": 223,
                                   "vertical_inches": 38.0}
        result = v4.score_prospect_v4(prospect, _dc("WR", 2),
                                      reference=sample_ref)
        assert result["lifecycle_state"] == "post_combine"
        assert result["athleticism_score"] is not None
        assert result["athleticism_receipt"]["source"] == "tested"

    def test_settled_absence_keeps_floor(self, sample_ref):
        result = v4.score_prospect_v4(copy.deepcopy(SMITH), _dc("WR", 2),
                                      reference=sample_ref,
                                      cycle_complete=True)
        assert result["lifecycle_state"] == "post_combine"
        assert result["athleticism_score"] == v4.UNTESTED_FLOOR
        assert "athleticism" not in result["legs_missing"]


class TestCapitalSeparation:
    def test_talent_invariant_to_capital(self, sample_ref):
        talents = []
        deltas = []
        for dc in (_dc("WR", 2), _dc("WR", 200), None):
            result = v4.score_prospect_v4(copy.deepcopy(SMITH), dc,
                                          reference=sample_ref)
            talents.append(result["talent_score"])
            deltas.append(result["capital_delta"])
        assert talents[0] == talents[1] == talents[2]
        assert deltas[0] > deltas[1]
        no_cap = v4.score_prospect_v4(copy.deepcopy(SMITH), None,
                                      reference=sample_ref)
        assert no_cap["prospect_score"] == no_cap["talent_score"]
        assert no_cap["capital_delta"] == 0.0
        assert no_cap["projected_draft_capital_score"] is None


class TestCastAdjustment:
    def test_qb_charged_for_elite_cast(self):
        base = v4.score_prospect_v4(copy.deepcopy(MESTEMAKER), None,
                                    reference=SYNTHETIC_REF)
        elite = v4.score_prospect_v4(
            copy.deepcopy(MESTEMAKER), None, reference=SYNTHETIC_REF,
            cast_context={"qb_cast_quality": 90.0})
        weak = v4.score_prospect_v4(
            copy.deepcopy(MESTEMAKER), None, reference=SYNTHETIC_REF,
            cast_context={"qb_cast_quality": 10.0})
        assert base["efficiency_score"] is not None
        assert (elite["efficiency_score"] < base["efficiency_score"]
                < weak["efficiency_score"])
        assert elite["cast"]["efficiency_adj"] == pytest.approx(-3.2)

    def test_receiver_credited_for_weak_qb(self, sample_ref):
        strong_qb = v4.score_prospect_v4(
            copy.deepcopy(SMITH), None, reference=sample_ref,
            cast_context={"receiver_qb_quality": 90.0})
        weak_qb = v4.score_prospect_v4(
            copy.deepcopy(SMITH), None, reference=sample_ref,
            cast_context={"receiver_qb_quality": 10.0})
        assert (strong_qb["production_score"]
                < weak_qb["production_score"])

    def test_cast_from_same_run_teammates(self):
        qb = {"player_id": "TQB", "name": "Test QB", "position": "QB",
              "age": 22.0, "recruiting": {"composite_rating": 0.95},
              "athleticism": {},
              "seasons": [{"season": 2025, "team": "Test U",
                           "conference": "SEC", "games_played": 12,
                           "pass_attempts": 400, "completions": 280,
                           "pass_yards": 3400, "pass_tds": 30,
                           "interceptions": 8, "rush_yards": 200,
                           "team_total_yards": 5200}]}
        wr = {"player_id": "TWR", "name": "Test WR", "position": "WR",
              "age": 21.0, "recruiting": {"composite_rating": 0.97},
              "athleticism": {},
              "seasons": [{"season": 2025, "team": "Test U",
                           "conference": "SEC", "games_played": 12,
                           "receptions": 70, "receiving_yards": 1050,
                           "receiving_tds": 9, "targets": 100,
                           "team_pass_yards": 3600,
                           "dominator_rating": 0.30,
                           "market_share_yards": 0.28}]}
        results = v4.score_all_v4([copy.deepcopy(qb), copy.deepcopy(wr)],
                                  {}, reference=SYNTHETIC_REF)
        by_id = {r["player_id"]: r for r in results}
        assert "qb_cast_quality" in by_id["TQB"]["cast"]
        assert "receiver_qb_quality" in by_id["TWR"]["cast"]
        assert by_id["TWR"]["overall_rank"] in (1, 2)
        assert by_id["TWR"]["position_rank"] == 1


class TestDurabilityRoleAware:
    def test_backup_season_is_not_an_injury_season(self):
        # Mestemaker's 2-game 2024 (41 attempts) is backup work: the
        # role test excludes it, so durability prices his two real
        # seasons only.
        score = v4.calc_durability_v4(
            copy.deepcopy(MESTEMAKER["seasons"]), "QB")
        assert score == pytest.approx(75.0)

    def test_injured_starter_still_pays(self):
        seasons = [
            {"season": 2024, "games_played": 13, "pass_attempts": 410},
            {"season": 2025, "games_played": 6, "pass_attempts": 260},
        ]
        score = v4.calc_durability_v4(seasons, "QB")
        assert score is not None and score < 65


class TestFixturesEndToEnd:
    @pytest.mark.parametrize("fixture,dc", [
        (SMITH, _dc("WR", 2)), (GREEN, _dc("TE", 20)),
        (MESTEMAKER, _dc("QB", 8))])
    def test_fixture_runs(self, sample_ref, fixture, dc):
        result = v4.score_prospect_v4(copy.deepcopy(fixture), dc,
                                      reference=sample_ref)
        assert result["model_version"] == "4.0-dev"
        assert isinstance(result["prospect_score"], float)
        assert result["band"] in ("elite", "great", "average", "day3")
        assert result["reference_kind"] == "fixture_sample"
        for key in ("talent_score", "capital_delta", "weights_used",
                    "legs_missing", "seasons_priced", "reasons",
                    "confidence_score", "display_percentiles"):
            assert key in result

    def test_mestemaker_recruiting_missing_not_faked(self, sample_ref):
        result = v4.score_prospect_v4(copy.deepcopy(MESTEMAKER),
                                      _dc("QB", 8), reference=sample_ref)
        assert result["recruiting_score"] is None
        assert "recruiting" in result["legs_missing"]

    def test_smith_athleticism_missing_pre_combine(self, sample_ref):
        result = v4.score_prospect_v4(copy.deepcopy(SMITH), _dc("WR", 2),
                                      reference=sample_ref)
        assert result["athleticism_score"] is None
        assert "athleticism" in result["legs_missing"]


class TestBands:
    def test_boundaries(self):
        assert v4.band_for(92.0)[0] == "elite"
        assert v4.band_for(91.99)[0] == "great"
        assert v4.band_for(80.0)[0] == "great"
        assert v4.band_for(79.99)[0] == "average"
        assert v4.band_for(70.0)[0] == "average"
        assert v4.band_for(69.9)[0] == "day3"
        assert v4.band_for(None) == (None, None)


class TestReferenceSample:
    def test_sample_is_marked_fixture_data(self):
        with open(SAMPLE_PATH, "r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        assert artifact["provenance"]["fixture_sample"] is True
        assert "FIXTURE SAMPLE" in artifact["provenance"]["source"]
        # The QB recruiting pool is empty in the fixtures (Mestemaker
        # has no recruiting data): no table may be invented for it.
        assert "recruiting_composite" not in artifact["anchors"]["QB"]

    def test_sample_carries_advanced_tables(self):
        with open(SAMPLE_PATH, "r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        adv = artifact["advanced"]["WR"]
        # Percentile anchor tables: 12 [percentile, mark] rows each.
        assert len(adv["Dominator"]) == 12
        assert adv["Dominator"][0][0] == 0
        assert adv["Dominator"][-1][0] == 100
        for label in ("Breakout Age", "Mkt Share", "Yds/Rec",
                      "Best Dominator", "Career Yds/Rec", "Rec Yds/Gm"):
            assert label in adv
        assert artifact["populations"]["advanced"]["WR"]["Dominator"] == 1
        assert artifact["advanced_lower_better"]["WR"] == ["Breakout Age"]


def _load_derive():
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import derive_v4_reference as drv
    return drv


def _synthetic_wr(pid, dom):
    return {
        "player_id": pid, "position": "WR", "age": 21.0,
        "weight_lbs": 200, "athleticism": {}, "recruiting": {},
        "seasons": [{
            "season": 2022, "games_played": 12, "team": "State",
            "dominator_rating": dom, "market_share_yards": 0.30,
            "market_share_tds": 0.28, "receptions": 60, "targets": 90,
            "receiving_yards": 900, "receiving_tds": 8,
            "yds_per_reception": 15.0,
        }],
    }


class TestDeriveAdvancedDistributions:
    """The reference run also emits Advanced-tab metric distributions,
    valued by the API's own builder (build_reference section a2)."""

    def test_build_reference_emits_advanced_tables(self):
        drv = _load_derive()
        population = [_synthetic_wr("A", 0.20), _synthetic_wr("B", 0.32),
                      _synthetic_wr("C", 0.41)]
        artifact = drv.build_reference(
            population, {}, {}, {"fixture_sample": True})
        adv = artifact["advanced"]["WR"]
        for label in ("Dominator", "Breakout Age", "Mkt Share",
                      "TD Share", "Tgt/Gm", "Catch%", "Yds/Rec",
                      "Best Dominator", "Career Yds/Rec", "Rec Yds/Gm"):
            assert label in adv, label
            assert len(adv[label]) == 12
        pops = artifact["populations"]["advanced"]["WR"]
        assert pops["Dominator"] == 3
        assert pops["Tgt/Gm"] == 3
        # Component-score rows never enter the advanced pools.
        assert "Efficiency" not in adv
        assert "Production" not in adv
        assert artifact["advanced_lower_better"]["WR"] == ["Breakout Age"]
        # Marks are the recorded values in ascending order.
        dom_marks = [m for _p, m in adv["Dominator"]]
        assert dom_marks == sorted(dom_marks)
        assert dom_marks[0] == 0.20 and dom_marks[-1] == 0.41

    def test_dry_run_covers_advanced_section(self, tmp_path):
        drv = _load_derive()
        out = str(tmp_path / "v4_reference_dry.json")
        artifact = drv.run_dry(out)
        assert artifact["provenance"]["fixture_sample"] is True
        assert "Dominator" in artifact["advanced"]["WR"]
        assert "Breakout Age" in artifact["advanced"]["QB"]
        with open(out, "r", encoding="utf-8") as fh:
            written = json.load(fh)
        assert written["advanced"] == artifact["advanced"]
