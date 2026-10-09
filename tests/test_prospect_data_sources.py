"""Tests for the v2.0 prospect data sources (synthetic data, no DB needed)."""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline.prospect_model import (
    MODEL_VERSION,
    POSITION_WEIGHTS,
    _RECRUITING_WEIGHT,
    calc_competition_score,
    calc_efficiency_score,
    calc_recruiting_score,
    score_prospect,
)
from data_building.rookie_pipeline.ingestion import (
    _percentile_rank,
    _wepa_metric_value,
    fetch_cfbd_recruiting,
    fetch_cfbd_team_context,
    fetch_cfbd_wepa,
)


class TestModelVersion:
    def test_version_bumped_for_grading_change(self):
        assert MODEL_VERSION == "2.0"

    def test_recruiting_weight_present_and_sums_to_one(self):
        for pos, weights in POSITION_WEIGHTS.items():
            assert weights["recruiting"] == pytest.approx(_RECRUITING_WEIGHT)
            assert abs(sum(weights.values()) - 1.0) < 0.001


class TestRecruitingScore:
    def test_five_star_composite_scores_elite(self):
        s = calc_recruiting_score({"composite_rating": 0.995, "stars": 5})
        assert s > 90

    def test_four_star_scores_above_average(self):
        s = calc_recruiting_score({"composite_rating": 0.90, "stars": 4})
        assert 55 < s < 75

    def test_average_three_star_is_neutral(self):
        # Median draft prospect (~0.87 composite) maps to neutral 50,
        # so known-average pedigree never scores below unknown.
        assert calc_recruiting_score({"composite_rating": 0.87, "stars": 3}) == 50.0

    def test_stars_fallback_when_no_rating(self):
        assert calc_recruiting_score({"stars": 5}) == 94.0
        assert calc_recruiting_score({"stars": 3}) == 50.0

    def test_missing_data_is_neutral_never_punitive(self):
        assert calc_recruiting_score(None) == 50.0
        assert calc_recruiting_score({}) == 50.0
        assert calc_recruiting_score({"stars": None, "composite_rating": None}) == 50.0

    def test_invalid_data_is_neutral(self):
        assert calc_recruiting_score({"composite_rating": "n/a"}) == 50.0


class TestWepaHelpers:
    def test_extracts_dict_ppa(self):
        raw = {"name": "X", "averagePPA": {"passing": 0.25, "rushing": 0.05}}
        assert _wepa_metric_value(raw) == 0.25

    def test_extracts_flat_number(self):
        assert _wepa_metric_value({"epa": 0.31}) == 0.31

    def test_unrecognized_shape_returns_none(self):
        assert _wepa_metric_value({"name": "X", "team": "Y"}) is None
        assert _wepa_metric_value({}) is None

    def test_percentile_rank(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile_rank(vals, 50.0) == 80.0
        assert _percentile_rank(vals, 10.0) == 0.0
        assert _percentile_rank([], 5.0) == 50.0


class TestGracefulDegradation:
    def test_fetches_skip_without_api_key(self, monkeypatch):
        monkeypatch.delenv("CFBD_API_KEY", raising=False)
        # ingestion reads the key at import time into CFBD_KEY; patch the module attr
        import data_building.rookie_pipeline.ingestion as ing
        monkeypatch.setattr(ing, "CFBD_KEY", "")
        assert fetch_cfbd_recruiting(2027) == []
        assert fetch_cfbd_wepa(2027) == {}
        assert fetch_cfbd_team_context(2027) == {}


class TestEfficiencyWepaBlend:
    def _qb_seasons(self):
        return [{
            "season": 2026, "team": "Texas", "conference": "SEC",
            "yds_per_attempt": 8.5, "completion_pct": 66.0, "td_int_ratio": 3.0,
            "games_played": 12,
        }]

    def test_no_wepa_is_noop(self):
        base = calc_efficiency_score(self._qb_seasons(), "QB")
        assert calc_efficiency_score(self._qb_seasons(), "QB", wepa=None) == base
        assert calc_efficiency_score(self._qb_seasons(), "QB", wepa={}) == base

    def test_wepa_blends_conservatively(self):
        base = calc_efficiency_score(self._qb_seasons(), "QB")
        wepa = {(2026, "passing"): {"adj_efficiency_score": 95.0}}
        blended = calc_efficiency_score(self._qb_seasons(), "QB", wepa=wepa)
        # 25% blend toward 95: moves up but stays anchored to the base
        assert blended > base
        assert blended == pytest.approx(base * 0.75 + 95.0 * 0.25)

    def test_wepa_wrong_type_ignored_for_position(self):
        base = calc_efficiency_score(self._qb_seasons(), "QB")
        wepa = {(2026, "rushing"): {"adj_efficiency_score": 99.0}}
        assert calc_efficiency_score(self._qb_seasons(), "QB", wepa=wepa) == base


class TestCompetitionTeamContext:
    def _seasons(self):
        # Sun Belt team: conference quality leaves headroom for the SOS blend
        return [{"season": 2026, "team": "Coastal Carolina", "conference": "Sun Belt"}]

    def test_no_context_is_noop(self):
        base = calc_competition_score(self._seasons())
        assert calc_competition_score(self._seasons(), team_context=None) == base
        assert calc_competition_score(self._seasons(), team_context={}) == base

    def test_brutal_sos_raises_score(self):
        base = calc_competition_score(self._seasons())
        ctx = {("coastal carolina", 2026): {"sp_sos": 10.0}}
        assert calc_competition_score(self._seasons(), team_context=ctx) > base

    def test_cupcake_sos_lowers_score(self):
        base = calc_competition_score(self._seasons())
        ctx = {("coastal carolina", 2026): {"sp_sos": -10.0}}
        assert calc_competition_score(self._seasons(), team_context=ctx) < base


class TestScoreProspectV2:
    def _prospect(self):
        return {
            "player_id": "ROOKIE_2027_TEST_WR",
            "name": "Test Receiver",
            "position": "WR",
            "school": "Ohio State",
            "age": 21.0,
            "draft_class_year": 2027,
            "seasons": [{
                "season": 2026, "team": "Ohio State", "conference": "Big Ten",
                "games_played": 13, "receptions": 80, "targets": 120,
                "receiving_yards": 1200, "receiving_tds": 10,
                "yds_per_reception": 15.0, "market_share_yards": 0.30,
                "dominator_rating": 0.32,
            }],
            "athleticism": {"forty_yard": 4.4, "speed_score": 105.0},
            "recruiting": {"stars": 5, "composite_rating": 0.99,
                           "national_rank": 12},
            "wepa": {},
            "team_context": {("ohio state", 2026): {"sp_sos": 4.0}},
        }

    def test_v2_fields_in_output(self):
        out = score_prospect(self._prospect())
        assert out["model_version"] == "2.0"
        assert out["recruiting_score"] > 90

    def test_recruiting_moves_final_score(self):
        five_star = score_prospect(self._prospect())["prospect_score"]
        p = self._prospect()
        p["recruiting"] = {"stars": 2, "composite_rating": 0.80}
        two_star = score_prospect(p)["prospect_score"]
        assert five_star > two_star

    def test_missing_recruiting_is_neutral(self):
        p = self._prospect()
        del p["recruiting"]
        out = score_prospect(p)
        assert out["recruiting_score"] == 50.0
        assert out["model_version"] == "2.0"
