"""Tests for prospect model v3.2 (pipeline improvement package).

Covers: missing draft capital treated as missing data (no pick-150 tax),
transparent capital (talent_score / capital_delta), missing/invalid
targets treatment, missing-age weight redistribution, the rule-based
archetype module's reference cases, the v3.2 seed capital map, and the
rookie_api advanced-metrics additions (WR/TE WEPA append, QB rushing
rows, per-season rows and best/career marks).

Panel fixtures below are the iteration log's panel inputs
(~/workspace/prospect-v32-iteration-log.md), where v3.1 scored Green
64.44 and Mestemaker 60.89 on the same inputs, and printed a forbidden
100.00 for Smith.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import prospect_model as pm
from data_building.rookie_pipeline.mock_draft_consensus import (
    pick_to_draft_capital_score,
)
from data_building.rookie_pipeline.prospect_archetypes import classify_archetype


def _dc(pos, pick, actual=False):
    return {
        "projected_pick": pick,
        "projected_round": ((pick - 1) // 32) + 1,
        "projected_draft_capital_score": pick_to_draft_capital_score(pick, pos),
        "num_mocks_used": 5,
        "is_actual_pick": actual,
    }


SMITH = {
    "player_id": "T_smith", "name": "Jeremiah Smith", "position": "WR",
    "draft_class_year": 2027, "age": 21.4,
    "seasons": [
        {"season": 2024, "games_played": 16, "team": "Ohio State",
         "conference": "Big Ten", "receptions": 76, "receiving_yards": 1315,
         "receiving_tds": 15, "yds_per_reception": 17.3,
         "dominator_rating": 0.30, "market_share_yards": 0.29,
         "team_pass_yards": 4530},
        {"season": 2025, "games_played": 13, "team": "Ohio State",
         "conference": "Big Ten", "receptions": 87, "receiving_yards": 1243,
         "receiving_tds": 12, "yds_per_reception": 14.29,
         "dominator_rating": 0.32, "market_share_yards": 0.30,
         "team_pass_yards": 4150},
        {"season": 2026, "games_played": 7, "team": "Ohio State",
         "conference": "Big Ten", "receptions": 48, "receiving_yards": 916,
         "receiving_tds": 11, "yds_per_reception": 19.08,
         "dominator_rating": 0.36, "market_share_yards": 0.315,
         "team_pass_yards": 2910},
    ],
    "recruiting": {"composite_rating": 0.997},
    "athleticism": {},
}

GREEN = {
    "player_id": "T_green", "name": "Trey'Dez Green", "position": "TE",
    "draft_class_year": 2027, "age": 21.2,
    "seasons": [
        {"season": 2024, "games_played": 13, "team": "LSU",
         "conference": "SEC", "receptions": 13, "receiving_yards": 101,
         "receiving_tds": 4, "yds_per_reception": 7.77,
         "dominator_rating": 0.04, "market_share_yards": 0.03,
         "team_pass_yards": 3300},
        {"season": 2025, "games_played": 11, "team": "LSU",
         "conference": "SEC", "receptions": 33, "receiving_yards": 433,
         "receiving_tds": 7, "yds_per_reception": 13.12,
         "dominator_rating": 0.11, "market_share_yards": 0.11,
         "team_pass_yards": 3900},
        {"season": 2026, "games_played": 3, "team": "LSU",
         "conference": "SEC", "receptions": 14, "receiving_yards": 222,
         "receiving_tds": 1, "yds_per_reception": 15.86,
         "dominator_rating": 0.13, "market_share_yards": 0.13,
         "team_pass_yards": 1700},
    ],
    "recruiting": {"composite_rating": 0.98},
    "athleticism": {},
}

MESTEMAKER = {
    "player_id": "T_mestemaker", "name": "Drew Mestemaker", "position": "QB",
    "draft_class_year": 2027, "age": 21.0,
    "seasons": [
        {"season": 2024, "games_played": 2, "team": "North Texas",
         "conference": "American Athletic", "pass_attempts": 41,
         "completion_pct": 63.4, "pass_yards": 393, "pass_tds": 2,
         "td_int_ratio": 2.0, "yds_per_attempt": 9.59,
         "rush_yards": 20, "rush_tds": 0, "team_total_yards": 900},
        {"season": 2025, "games_played": 14, "team": "North Texas",
         "conference": "American Athletic", "pass_attempts": 463,
         "completion_pct": 68.9, "pass_yards": 4379, "pass_tds": 34,
         "td_int_ratio": 3.78, "yds_per_attempt": 9.46,
         "rush_yards": 89, "rush_tds": 5, "team_total_yards": 6700},
        {"season": 2026, "games_played": 5, "team": "Oklahoma State",
         "conference": "Big 12", "pass_attempts": 169,
         "completion_pct": 67.5, "pass_yards": 1573, "pass_tds": 13,
         "td_int_ratio": 2.6, "yds_per_attempt": 9.31,
         "rush_yards": 108, "rush_tds": 2, "team_total_yards": 2600},
    ],
    "recruiting": None,
    "athleticism": {},
}


def _score(prospect, dc):
    return pm.score_prospect(copy.deepcopy(prospect), dc, skip_sagarin=True)


class TestVersion:
    def test_model_version_bumped(self):
        assert pm.MODEL_VERSION == "3.4"


class TestMissingCapital:
    def test_no_pick_means_no_capital_tax(self):
        # Same Green profile three ways: consensus pick, pick-150 dict
        # (the old default's value), and no capital at all.
        with_pick = _score(GREEN, _dc("TE", 35))
        at_150 = _score(GREEN, _dc("TE", 150))
        no_pick = _score(GREEN, None)
        # Missing capital is the talent grade itself: strictly above the
        # pick-150 outcome the old default produced for uncovered players.
        assert no_pick["prospect_score"] > at_150["prospect_score"]
        # And it equals the talent_score of the covered run: the covered
        # grade is talent plus exactly capital_delta.
        assert no_pick["prospect_score"] == pytest.approx(
            with_pick["talent_score"], abs=0.011)
        assert no_pick["projected_draft_capital_score"] is None

    def test_missing_capital_contract(self):
        no_pick = _score(GREEN, None)
        assert no_pick["capital_delta"] is None
        assert no_pick["talent_score"] == no_pick["prospect_score"]

    def test_consensus_dict_without_pick_is_missing(self):
        # A consensus row that carries no projected pick must not invent one.
        r = _score(GREEN, {"num_mocks_used": 0, "mock_sources": []})
        assert r["capital_delta"] is None
        assert r["projected_draft_capital_score"] is None


class TestTransparentCapital:
    def test_green_consensus_pick_delta_positive_and_modest(self):
        r = _score(GREEN, _dc("TE", 35))
        assert r["capital_delta"] is not None
        assert 0 < r["capital_delta"] < 10
        assert r["talent_score"] == pytest.approx(
            r["prospect_score"] - r["capital_delta"], abs=0.011)

    def test_talent_score_is_pick_invariant(self):
        # Talent is a property of the profile, not of the pick assigned.
        t35 = _score(GREEN, _dc("TE", 35))["talent_score"]
        t8 = _score(GREEN, _dc("TE", 8))["talent_score"]
        t150 = _score(GREEN, _dc("TE", 150))["talent_score"]
        assert t35 == pytest.approx(t8, abs=0.011)
        assert t35 == pytest.approx(t150, abs=0.011)


class TestTargetsTreatment:
    def test_invalid_targets_equal_missing_targets(self):
        bad = copy.deepcopy(SMITH)
        bad["seasons"][-1]["targets"] = 18  # stored garbage: 18 tgt vs 48 rec
        missing = copy.deepcopy(SMITH)  # no targets key at all
        r_bad = _score(bad, _dc("WR", 2))
        r_missing = _score(missing, _dc("WR", 2))
        assert r_bad["prospect_score"] == pytest.approx(
            r_missing["prospect_score"], abs=0.011)

    def test_valid_targets_change_the_grade(self):
        # Sanity: the missing treatment is not "targets never matter".
        valid = copy.deepcopy(SMITH)
        valid["seasons"][-1]["targets"] = 62
        missing = copy.deepcopy(SMITH)
        assert _score(valid, _dc("WR", 2))["prospect_score"] != pytest.approx(
            _score(missing, _dc("WR", 2))["prospect_score"], abs=0.011)

    def test_te_efficiency_invalid_targets_renormalize(self):
        seasons = [{
            "season": 2026, "games_played": 10, "receptions": 40,
            "receiving_yards": 600, "yds_per_reception": 15.0,
            "market_share_yards": 0.20, "targets": 25,  # invalid: < rec
        }]
        eff_bad = pm.calc_efficiency_score(seasons, "TE")
        seasons[0].pop("targets")
        eff_missing = pm.calc_efficiency_score(seasons, "TE")
        assert eff_bad == pytest.approx(eff_missing)


class TestMissingAge:
    def test_missing_age_redistributes_instead_of_neutral_50(self):
        # Find the age whose age_score is ~50 (the old neutral imputation's
        # effective value) for a WR, then show the missing-age grade is NOT
        # that grade: the weight moves to the profile's real components,
        # which for Smith's profile are far above 50.
        candidates = [20.0 + 0.1 * i for i in range(70)]
        neutral_age = min(
            candidates,
            key=lambda a: abs(pm.calc_age_score(a, 2027, "WR") - 50.0))
        assert abs(pm.calc_age_score(neutral_age, 2027, "WR") - 50.0) < 2.0

        no_age = copy.deepcopy(SMITH)
        no_age["age"] = None
        r_missing = _score(no_age, _dc("WR", 2))
        aged = copy.deepcopy(SMITH)
        aged["age"] = neutral_age
        r_neutral = _score(aged, _dc("WR", 2))
        assert r_missing["prospect_score"] > r_neutral["prospect_score"] + 1.0


class TestRegressionVsV31:
    def test_green_materially_above_v31(self):
        # v3.1 on identical inputs: 64.44 (iteration log round 3 table).
        assert _score(GREEN, _dc("TE", 35))["prospect_score"] >= 75.0

    def test_mestemaker_materially_above_v31(self):
        # v3.1 on identical inputs: 60.89.
        assert _score(MESTEMAKER, _dc("QB", 8))["prospect_score"] >= 75.0

    def test_smith_elite_but_no_100(self):
        # v3.1 printed a forbidden 100.00; v3.2 must be elite and < 100.
        s = _score(SMITH, _dc("WR", 2))["prospect_score"]
        assert 90.0 <= s < 100.0


class TestArchetypeReferenceCases:
    """Archetype v2 (prospect-archetypes-v2.md): scored NFL-outlook fits.

    Fixtures are the validation panel's profiles; classify is called with
    the full profile (age, projected pick, size/speed when measured), the
    way score_prospect calls it in production.
    """

    def test_smith_alpha_x_high(self):
        r = classify_archetype(
            "WR", SMITH["seasons"],
            profile={"age": 21.4, "projected_pick": 2})
        assert r["archetype"] == "Alpha X"
        assert r["archetype_confidence"] == "High"
        assert "dominator" in r["archetype_receipt"]

    def test_green_red_zone_target_seam_stretcher(self):
        # v2 headline flip: the 21.2% TD rate on his 2025 basis season is
        # the loudest trait; the seam fit sits within half a point.
        r = classify_archetype(
            "TE", GREEN["seasons"],
            profile={"age": 21.2, "projected_pick": 35})
        assert r["archetype"] == "Red-Zone Target · Seam Stretcher"
        assert r["archetype_confidence"] == "Medium"

    def test_mestemaker_pocket_passer_distributor(self):
        # v1 read 9.46 YPA and stopped at Gunslinger; v2 reads the whole
        # shape: dropback volume, no rushing role, 68.9% completions.
        r = classify_archetype(
            "QB", MESTEMAKER["seasons"],
            profile={"age": 21.0, "projected_pick": 8})
        assert r["archetype"] == "Pocket Passer · Distributor"
        assert r["archetype_confidence"] == "Medium"
        assert r["fits"]["Gunslinger"] < r["fits"]["Distributor"]

    def test_hardy_power_back_high(self):
        seasons = [
            {"season": 2024, "games_played": 12, "rush_attempts": 237,
             "rush_yards": 1351, "rush_tds": 13, "yds_per_carry": 5.7,
             "receptions": 8, "receiving_yards": 72,
             "dominator_rating": 0.38, "market_share_yards": 0.30},
            {"season": 2025, "games_played": 12, "rush_attempts": 253,
             "rush_yards": 1645, "rush_tds": 16, "yds_per_carry": 6.5,
             "receptions": 8, "receiving_yards": 54,
             "dominator_rating": 0.36, "market_share_yards": 0.30},
        ]
        r = classify_archetype(
            "RB", seasons, profile={"age": 21.9, "projected_pick": 17})
        assert r["archetype"] == "Power Back"
        assert r["archetype_confidence"] == "High"

    def test_moore_distributor_high(self):
        # v2 label change: 71.8% completions with a short-area efficiency
        # shape is a distributor's profile, not a generic pocket passer.
        seasons = [
            {"season": 2024, "games_played": 11, "pass_attempts": 192,
             "pass_yards": 1257, "pass_tds": 4, "completion_pct": 59.4,
             "yds_per_attempt": 6.55, "td_int_ratio": 0.4,
             "rush_yards": 60, "rush_tds": 1},
            {"season": 2025, "games_played": 14, "pass_attempts": 412,
             "pass_yards": 3565, "pass_tds": 30, "completion_pct": 71.8,
             "yds_per_attempt": 8.65, "td_int_ratio": 3.0,
             "rush_yards": 146, "rush_tds": 2},
        ]
        r = classify_archetype(
            "QB", seasons, profile={"age": 21.7, "projected_pick": 14})
        assert r["archetype"] == "Distributor"
        assert r["archetype_confidence"] == "High"

    def test_coleman_alpha_x_developmental(self):
        # The canonical v2 case: v1 printed Auburn's slot usage as the
        # label (Big Slot, deleted in v2). The projection is outside;
        # the unproven proof rides as the Developmental secondary trait.
        seasons = [
            {"season": 2024, "games_played": 11, "receptions": 37,
             "receiving_yards": 598, "receiving_tds": 8,
             "yds_per_reception": 16.16, "dominator_rating": 0.24,
             "market_share_yards": 0.23},
            {"season": 2025, "games_played": 12, "receptions": 56,
             "receiving_yards": 708, "receiving_tds": 5,
             "yds_per_reception": 12.64, "dominator_rating": 0.25,
             "market_share_yards": 0.25},
        ]
        r = classify_archetype(
            "WR", seasons, profile={"age": 20.8, "projected_pick": 6})
        assert r["archetype"] == "Alpha X · Developmental"
        assert r["archetype_confidence"] == "Medium"

    def test_no_seasons_no_label(self):
        r = classify_archetype("WR", [])
        assert r["archetype"] is None
        assert r["archetype_confidence"] is None

    def test_score_prospect_carries_archetype(self):
        r = _score(SMITH, _dc("WR", 2))
        assert r["archetype"] == "Alpha X"
        assert r["archetype_confidence"] == "High"
        assert r["archetype_receipt"]


class TestArchetypeAnchorsV2:
    """The v2 spec's sanity anchors (prospect-archetypes-v2.md 8.2)."""

    def test_gibbs_is_a_pass_catching_back_not_a_power_back(self):
        seasons = [
            {"season": 2021, "games_played": 12, "rush_attempts": 143,
             "rush_yards": 746, "rush_tds": 4, "yds_per_carry": 5.22,
             "receptions": 36, "receiving_yards": 354,
             "dominator_rating": 0.24, "market_share_yards": 0.18},
            {"season": 2022, "games_played": 12, "rush_attempts": 151,
             "rush_yards": 926, "rush_tds": 7, "yds_per_carry": 6.13,
             "receptions": 44, "receiving_yards": 444,
             "dominator_rating": 0.27, "market_share_yards": 0.20},
        ]
        r = classify_archetype(
            "RB", seasons,
            profile={"weight_lbs": 200, "forty_yard": 4.36,
                     "projected_pick": 12, "age": 21.1})
        assert r["archetype"] == "Pass-Catching Back"
        assert r["archetype_confidence"] == "High"
        assert r["fits"]["Power Back"] < 55.0

    def test_allen_dual_threat_developmental_gated(self):
        seasons = [
            {"season": 2016, "games_played": 14, "pass_attempts": 373,
             "pass_yards": 3203, "pass_tds": 28, "completion_pct": 56.0,
             "yds_per_attempt": 8.59, "td_int_ratio": 1.87,
             "rush_yards": 523, "rush_tds": 7},
            {"season": 2017, "games_played": 11, "pass_attempts": 270,
             "pass_yards": 1812, "pass_tds": 16, "completion_pct": 56.3,
             "yds_per_attempt": 6.71, "td_int_ratio": 2.67,
             "rush_yards": 204, "rush_tds": 5},
        ]
        r = classify_archetype(
            "QB", seasons,
            profile={"weight_lbs": 237, "forty_yard": 4.75,
                     "projected_pick": 7, "age": 21.9})
        assert r["archetype"].startswith("Dual-Threat")
        # A defined rushing role caps the tools label at 55 (v2.1 gate).
        assert r["fits"]["Developmental"] <= 55.0

    def test_burrow_pocket_passer_gunslinger_capped(self):
        seasons = [
            {"season": 2018, "games_played": 13, "pass_attempts": 379,
             "pass_yards": 2894, "pass_tds": 16, "completion_pct": 57.8,
             "yds_per_attempt": 7.64, "td_int_ratio": 3.2,
             "rush_yards": 399, "rush_tds": 7},
            {"season": 2019, "games_played": 15, "pass_attempts": 527,
             "pass_yards": 5671, "pass_tds": 60, "completion_pct": 76.3,
             "yds_per_attempt": 10.76, "td_int_ratio": 10.0,
             "rush_yards": 368, "rush_tds": 5},
        ]
        r = classify_archetype(
            "QB", seasons, profile={"projected_pick": 1, "age": 23.3})
        assert r["archetype"] == "Pocket Passer"
        assert r["archetype_confidence"] == "High"
        # 76.3% completions trip the precision gate: capped at 62.
        assert r["fits"]["Gunslinger"] <= 62.0

    def test_devonta_playmaker_alpha_capped_by_size(self):
        seasons = [
            {"season": 2019, "games_played": 13, "receptions": 68,
             "receiving_yards": 1256, "receiving_tds": 14,
             "yds_per_reception": 18.47, "dominator_rating": 0.28,
             "market_share_yards": 0.27},
            {"season": 2020, "games_played": 13, "receptions": 117,
             "receiving_yards": 1856, "receiving_tds": 23,
             "yds_per_reception": 15.86, "dominator_rating": 0.42,
             "market_share_yards": 0.40},
        ]
        r = classify_archetype(
            "WR", seasons,
            profile={"weight_lbs": 170, "projected_pick": 10, "age": 22.4})
        assert r["archetype"] == "Playmaker"
        # Alpha production, not an X frame: the size gate caps Alpha X.
        assert r["fits"]["Alpha X"] == pytest.approx(55.0, abs=0.6)

    def test_fournette_never_a_pass_catcher(self):
        seasons = [
            {"season": 2014, "games_played": 13, "rush_attempts": 187,
             "rush_yards": 1034, "rush_tds": 10, "yds_per_carry": 5.53,
             "receptions": 7, "receiving_yards": 127,
             "dominator_rating": 0.24, "market_share_yards": 0.22},
            {"season": 2015, "games_played": 12, "rush_attempts": 300,
             "rush_yards": 1953, "rush_tds": 22, "yds_per_carry": 6.51,
             "receptions": 19, "receiving_yards": 253,
             "dominator_rating": 0.40, "market_share_yards": 0.37},
            {"season": 2016, "games_played": 7, "rush_attempts": 129,
             "rush_yards": 843, "rush_tds": 8, "yds_per_carry": 6.53,
             "receptions": 15, "receiving_yards": 146,
             "dominator_rating": 0.20, "market_share_yards": 0.18},
        ]
        r = classify_archetype(
            "RB", seasons,
            profile={"weight_lbs": 240, "forty_yard": 4.51,
                     "projected_pick": 4, "age": 22.3})
        assert not r["archetype"].startswith("Pass-Catching")
        # 163 rush yds/g makes receiving a trait, not the role (gate: 72).
        assert r["fits"]["Pass-Catching Back"] <= 72.0
        assert r["archetype"].startswith("Workhorse")


class TestSeedCapitalMap:
    def test_v1_picks_in_seed(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_building", "rookie_pipeline", "seed_data",
            "prospects_2027.json")
        with open(path) as f:
            profiles = {p["name"]: p for p in json.load(f)["profiles"]}
        expected = {
            "Drew Mestemaker": 8, "Trey'Dez Green": 35, "Julian Sayin": 12,
            "Dante Moore": 14, "Arch Manning": 24, "Jadan Baugh": 16,
            "Charlie Becker": 15, "Terrance Carter Jr.": 54,
            "Peter Clarke": 62,
        }
        for name, pick in expected.items():
            assert profiles[name].get("projected_pick") == pick, name
        # Uncovered names keep NO invented pick.
        assert profiles["LaNorris Sellers"].get("projected_pick") is None
        assert profiles["Lawson Luckie"].get("projected_pick") is None

    def test_seed_consensus_flows_through_existing_fallback(self):
        from data_building.rookie_pipeline.pipeline import build_seed_consensus
        from data_building.rookie_pipeline.ingestion import get_seed_prospects
        consensus = build_seed_consensus(get_seed_prospects(2027))
        by_name = {v["player_name"]: v for v in consensus.values()}
        assert by_name["Drew Mestemaker"]["projected_pick"] == 8
        assert by_name["Trey'Dez Green"]["projected_pick"] == 35
        assert "LaNorris Sellers" not in by_name


class TestAdvancedMetricsApi:
    def _api(self):
        from dashboard_services import rookie_api
        return rookie_api

    def test_wepa_row_appended_for_wr(self):
        api = self._api()
        rows = api._build_advanced_metrics(
            "WR", SMITH["seasons"], {}, {"efficiency_score": 80.0},
            wepa_score=88.0)
        labels = [m["label"] for m in rows]
        assert "WEPA" in labels
        rows_no = api._build_advanced_metrics(
            "WR", SMITH["seasons"], {}, {"efficiency_score": 80.0})
        assert "WEPA" not in [m["label"] for m in rows_no]

    def test_wepa_row_appended_for_te(self):
        api = self._api()
        rows = api._build_advanced_metrics(
            "TE", GREEN["seasons"], {}, {"efficiency_score": 70.0},
            wepa_score=75.0)
        assert "WEPA" in [m["label"] for m in rows]

    def test_qb_rushing_rows(self):
        api = self._api()
        seasons = [{
            "season": 2025, "games_played": 13, "pass_attempts": 400,
            "pass_yards": 3400, "pass_tds": 30, "completions": 280,
            "interceptions": 8, "completion_pct": 70.0,
            "yds_per_attempt": 8.5, "td_int_ratio": 3.75,
            "rush_attempts": 90, "rush_yards": 520, "rush_tds": 9,
            "yds_per_carry": 5.8,
        }]
        rows = {m["label"]: m for m in api._build_advanced_metrics(
            "QB", seasons, {}, {"efficiency_score": 80.0})}
        assert rows["Rush Yds/Gm"]["raw"] == "40"
        assert rows["Yds/Carry"]["grade"] is not None
        assert rows["Rush TDs"]["raw"] == "9"
        assert rows["Rush TDs"]["grade"] is None  # raw-only, no helper

    def test_invalid_targets_hide_tgt_and_catch(self):
        api = self._api()
        seasons = [{
            "season": 2026, "games_played": 7, "receptions": 48,
            "targets": 18, "receiving_yards": 916, "receiving_tds": 11,
            "yds_per_reception": 19.08, "dominator_rating": 0.36,
            "market_share_yards": 0.315, "market_share_tds": 0.333,
        }]
        rows = {m["label"]: m for m in api._build_advanced_metrics(
            "WR", seasons, {}, {})}
        assert rows["Tgt/Gm"]["raw"] is None
        assert rows["Catch%"]["raw"] is None

    def test_marks_definitions(self):
        api = self._api()
        seasons = [
            {"season": 2024, "games_played": 10, "receptions": 50,
             "receiving_yards": 800, "receiving_tds": 5,
             "market_share_yards": 0.20, "market_share_tds": 0.22},
            {"season": 2025, "games_played": 10, "receptions": 70,
             "receiving_yards": 1050, "receiving_tds": 9,
             "market_share_yards": 0.30, "market_share_tds": 0.34},
        ]
        marks = api._advanced_marks(seasons)
        # Best dominator from shares: 0.65*0.30 + 0.35*0.34 = 0.314
        assert marks["best_dominator"] == pytest.approx(0.314, abs=0.001)
        # Career: mean shares 0.25 / 0.28 -> 0.65*0.25 + 0.35*0.28 = 0.2605
        assert marks["career_dominator"] == pytest.approx(0.2605, abs=0.001)
        # Career YPR: 1850 yards / 120 rec
        assert marks["career_yds_per_rec"] == pytest.approx(1850 / 120, abs=0.01)
        assert marks["latest_rec_yards_per_game"] == pytest.approx(105.0)
        assert marks["latest_rec_tds_per_game"] == pytest.approx(0.9)
        assert marks["td_share_over_yard_share_pts"] == pytest.approx(4.0)

    def test_season_rows_shape(self):
        api = self._api()
        rows = api._advanced_season_rows(SMITH["seasons"])
        assert [r["season"] for r in rows] == [2024, 2025, 2026]
        assert rows[-1]["rec_yards_per_game"] == pytest.approx(916 / 7, abs=0.01)
        assert rows[-1]["dominator"] == pytest.approx(0.36)

    def test_mark_rows_graded_with_same_stat_formulas(self):
        api = self._api()
        rows = {m["label"]: m for m in api._build_advanced_metrics(
            "WR", SMITH["seasons"], {}, {})}
        marks = api._advanced_marks(SMITH["seasons"])
        for label in ("Best Dominator", "Career Yds/Rec", "Rec Yds/Gm"):
            assert label in rows
            assert rows[label]["raw"] is not None
            assert rows[label]["grade"] is not None
        # Each mark is graded by the display formula of the same stat:
        # dominator, Yds/Rec, and the yards-per-game (Scrim Yds/Gm) formula.
        assert rows["Best Dominator"]["grade"] == api._grade_dominator(
            marks["best_dominator"])
        assert rows["Best Dominator"]["grade"] == rows["Dominator"]["grade"]
        assert rows["Career Yds/Rec"]["grade"] == api._grade_ypr(
            marks["career_yds_per_rec"])
        assert rows["Rec Yds/Gm"]["grade"] == api._grade_scrim(
            marks["latest_rec_yards_per_game"])


class TestReasonsWithNoDurabilityData:
    """Regression: _build_reasons crashed with TypeError when durability
    was None (prospect with no games_played anywhere, e.g. the backtest's
    college-stats fetch, which skips the games-played lookup). The weighted
    sum already treats a None durability as missing and redistributes; the
    reasons builder must do the same instead of comparing None <= 40.
    """

    def _no_gp_prospect(self):
        p = copy.deepcopy(SMITH)
        for s in p["seasons"]:
            s.pop("games_played", None)
        return p

    def test_durability_is_none_without_any_gp(self):
        p = self._no_gp_prospect()
        assert pm.calc_durability_score(p["seasons"]) is None

    def test_score_prospect_without_gp_returns_score_and_reasons(self):
        r = _score(self._no_gp_prospect(), _dc("WR", 2))
        assert r["durability_score"] is None
        assert isinstance(r["prospect_score"], (int, float))
        assert isinstance(r["key_reasons"], str) and r["key_reasons"]
        assert "Durability concern" not in r["key_reasons"]

    def test_build_reasons_with_none_durability(self):
        p = self._no_gp_prospect()
        reasons = pm._build_reasons(
            p, "WR", p["seasons"],
            80.0, 75.0, 80.0, 50.0, 50.0, None, 80.0,
            None, None, 90.0, None,
        )
        assert isinstance(reasons, str)
        assert "Durability concern" not in reasons

    def test_build_reasons_low_durability_still_flags(self):
        # Control: a real low durability score still emits the bullet.
        p = self._no_gp_prospect()
        reasons = pm._build_reasons(
            p, "WR", p["seasons"],
            80.0, 75.0, 80.0, 50.0, 50.0, None, 80.0,
            None, 30.0, 90.0, None,
        )
        assert "Durability concern" in reasons
