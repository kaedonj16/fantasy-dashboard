"""Tests for prospect model v5 (learned outcome model).

The wiring and the contract, not a trained model's quality: the
feature extractor on fixture prospects (values + missing staying
missing), abstention with no college seasons, the grade-anchor
map's shape, the capital-blind receipt, the artifact round-trip,
CV fold hygiene (anchors fit without the held-out class), and the
pipeline selector (accepts v5, fails closed without an artifact).

Synthetic data only; no DB, no network. Quality gates live in
scripts/train_prospect_model_v5.py --cv, run by hand.
"""

import os
import random
import sys

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_building.rookie_pipeline import pipeline  # noqa: E402
from data_building.rookie_pipeline import prospect_model_v5 as v5  # noqa: E402
from test_prospect_v32 import MESTEMAKER, SMITH, _dc  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Synthetic training rows: outcomes driven by primary production,
# efficiency and capital, so a fit has something real to learn.
# ─────────────────────────────────────────────────────────────────────


def _synthetic_rows(pos, n, seed, outcome_scale=1.0, class_year=2020):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        primary = rng.uniform(15.0, 125.0)
        eff = rng.uniform(4.0, 19.0)
        capital = rng.uniform(2.0, 100.0)
        feats = {name: None
                 for name in v5.FEATURES_BY_POSITION[pos]}
        feats.update({
            "primary_pg_peak": primary,
            "primary_pg_latest": primary * rng.uniform(0.7, 1.05),
            "primary_pg_trend": rng.uniform(-25.0, 25.0),
            "efficiency_peak": eff,
            "efficiency_latest": eff * rng.uniform(0.8, 1.05),
            "td_pg_peak": rng.uniform(0.0, 1.4),
            "age": rng.uniform(20.5, 24.5),
            "breakout_age": rng.uniform(18.5, 23.5),
            "competition_index": rng.uniform(0.3, 1.0),
            "cast_quality": rng.uniform(20.0, 110.0),
            "forty": rng.uniform(4.35, 4.85),
            "peak_games": rng.uniform(8.0, 14.0),
            "career_games": rng.uniform(20.0, 50.0),
            "capital_score": capital,
            "tested": 1.0,
        })
        if pos == "QB":
            feats.update({
                "completion_pct_peak": rng.uniform(55.0, 72.0),
                "td_int_peak": rng.uniform(1.0, 5.0),
                "qb_share_peak": rng.uniform(0.4, 0.85),
                "rush_pg_peak": rng.uniform(0.0, 45.0),
                "attempts_peak": rng.uniform(250.0, 550.0),
                "vertical": rng.uniform(28.0, 38.0),
                "broad": rng.uniform(105.0, 130.0),
                "speed_score": rng.uniform(80.0, 115.0),
                "weight": rng.uniform(195.0, 235.0),
            })
        else:
            feats.update({
                "rec_pg_peak": rng.uniform(1.0, 8.0),
                "dominator_peak": rng.uniform(0.02, 0.45),
                "market_share_peak": rng.uniform(0.02, 0.4),
                "touches_peak": rng.uniform(40.0, 320.0),
                "n_seasons": rng.uniform(2.0, 4.0),
                "cone": rng.uniform(6.6, 7.6),
                "shuttle": rng.uniform(4.0, 4.6),
                "vertical": rng.uniform(28.0, 42.0),
                "broad": rng.uniform(105.0, 135.0),
                "ras": rng.uniform(2.0, 10.0),
                "speed_score": rng.uniform(75.0, 120.0),
                "weight": rng.uniform(175.0, 250.0),
            })
            if pos in ("WR", "TE"):
                feats["pass_share_peak"] = rng.uniform(0.02, 0.4)
            if pos == "WR":
                feats["height"] = rng.uniform(68.0, 77.0)
            else:
                feats["bmi"] = rng.uniform(24.0, 34.0)
        outcome = max(0.0, outcome_scale * (
            (primary - 15.0) * 1.9 + eff * 4.0 + capital * 0.9
            + rng.uniform(-12.0, 12.0)))
        rows.append({
            "player_id": f"SYN_{pos}_{i}",
            "position": pos,
            "draft_class_year": class_year,
            "features": feats,
            "outcome": outcome,
        })
    return rows


@pytest.fixture(scope="module")
def artifact():
    rows = (_synthetic_rows("WR", 90, seed=11)
            + _synthetic_rows("QB", 45, seed=12)
            + _synthetic_rows("RB", 70, seed=13)
            + _synthetic_rows("TE", 45, seed=14))
    return v5.fit_artifact(rows)


WEAK_WR = {
    "player_id": "T_weak", "name": "Weak Sister", "position": "WR",
    "draft_class_year": 2027, "age": 23.9,
    "seasons": [
        {"season": 2025, "games_played": 12, "team": "Nowhere State",
         "conference": "MAC", "receptions": 22, "receiving_yards": 240,
         "receiving_tds": 1, "yds_per_reception": 10.9,
         "dominator_rating": 0.07, "market_share_yards": 0.08,
         "team_pass_yards": 2900},
        {"season": 2026, "games_played": 12, "team": "Nowhere State",
         "conference": "MAC", "receptions": 26, "receiving_yards": 300,
         "receiving_tds": 2, "yds_per_reception": 11.5,
         "dominator_rating": 0.08, "market_share_yards": 0.09,
         "team_pass_yards": 3100},
    ],
    "recruiting": None,
    "athleticism": {},
}


# ─────────────────────────────────────────────────────────────────────
# Feature extractor
# ─────────────────────────────────────────────────────────────────────


class TestFeatureExtractor:
    def test_smith_values(self):
        feats = v5.extract_features(SMITH, _dc("WR", 1))
        # Peak season is 2026 (916 yards / 7 games), by the shared
        # season-selection rule.
        assert feats["primary_pg_peak"] == pytest.approx(916 / 7, abs=0.5)
        assert feats["efficiency_peak"] == pytest.approx(19.08)
        assert feats["dominator_peak"] == pytest.approx(0.36)
        assert feats["capital_score"] is not None
        assert feats["capital_score"] > 90.0
        assert feats["age"] == pytest.approx(21.4)
        assert feats["breakout_age"] is not None
        assert feats["competition_index"] is not None

    def test_missing_stays_missing(self):
        feats = v5.extract_features(MESTEMAKER, _dc("QB", 12))
        # Untested: no invented floor, no imputed mark.
        assert feats["forty"] is None
        assert feats["vertical"] is None
        assert feats["speed_score"] is None
        assert feats["tested"] == 0.0
        # But his measured college production is real and present.
        # Peak season by primary rate is 2026 (314.6 pass yds/gm).
        assert feats["primary_pg_peak"] == pytest.approx(1573 / 5, abs=0.5)
        assert feats["completion_pct_peak"] == pytest.approx(67.5)

    def test_no_capital_is_missing_not_zero(self):
        feats = v5.extract_features(SMITH, None)
        assert feats["capital_score"] is None

    def test_has_college_data(self):
        assert v5.has_college_data(SMITH)
        ghost = dict(SMITH, seasons=[])
        assert not v5.has_college_data(ghost)


# ─────────────────────────────────────────────────────────────────────
# Grade anchors
# ─────────────────────────────────────────────────────────────────────


class TestGradeAnchors:
    def test_monotone_and_bounded(self):
        grades = [v5.grade_for_percentile(p) for p in range(0, 101)]
        assert all(b >= a for a, b in zip(grades, grades[1:]))
        assert grades[-1] == pytest.approx(99.5)
        assert grades[-1] < 100.0

    def test_scale_semantics(self):
        assert v5.grade_for_percentile(50) == pytest.approx(70.0)
        assert v5.grade_for_percentile(97.5) == pytest.approx(92.0)
        assert v5.grade_for_percentile(75) == pytest.approx(80.0)

    def test_99_only_at_the_extreme(self):
        assert v5.grade_for_percentile(99.0) < 99.0
        assert v5.grade_for_percentile(99.9) < 99.0
        assert v5.grade_for_percentile(100.0) >= 99.0


# ─────────────────────────────────────────────────────────────────────
# Scoring: abstention, receipts, ranks
# ─────────────────────────────────────────────────────────────────────


class TestScoring:
    def test_abstains_without_seasons(self, artifact):
        ghost = {"player_id": "T_ghost", "name": "No Stats",
                 "position": "WR", "draft_class_year": 2027,
                 "age": 22.0, "seasons": [], "athleticism": {}}
        (row,) = v5.score_all_v5([ghost], {"T_ghost": _dc("WR", 5)},
                                 model_artifact=artifact)
        assert row["prospect_score"] is None
        assert row["talent_score"] is None
        assert row["band"] is None
        assert "abstention" in row["reasons"][0]

    def test_abstains_for_untrained_position(self):
        tiny = v5.fit_artifact(_synthetic_rows("WR", 40, seed=21))
        (row,) = v5.score_all_v5([MESTEMAKER], {"T_mestemaker": _dc("QB", 3)},
                                 model_artifact=tiny)
        assert row["prospect_score"] is None

    def test_elite_fixture_beats_weak_fixture(self, artifact):
        rows = v5.score_all_v5(
            [WEAK_WR, SMITH],
            {"T_weak": _dc("WR", 180), "T_smith": _dc("WR", 1)},
            model_artifact=artifact)
        by_id = {r["player_id"]: r for r in rows}
        assert by_id["T_smith"]["prospect_score"] > \
            by_id["T_weak"]["prospect_score"]
        assert by_id["T_smith"]["overall_rank"] == 1
        assert by_id["T_smith"]["model_version"] == "5.0"
        assert by_id["T_smith"]["band"] in (
            "elite", "great", "average", "day3")

    def test_receipts_carry_top_features(self, artifact):
        (row,) = v5.score_all_v5([SMITH], {"T_smith": _dc("WR", 1)},
                                 model_artifact=artifact)
        assert row["prospect_score"] is not None
        assert 1 <= len(row["top_features"]) <= 3
        for t in row["top_features"]:
            assert t["feature"] in v5.FEATURES_BY_POSITION["WR"]
            assert isinstance(t["delta"], float)
        assert row["predicted_ppr_season"] is not None
        assert row["reasons"]

    def test_capital_blind_receipt_differs(self, artifact):
        (row,) = v5.score_all_v5([SMITH], {"T_smith": _dc("WR", 1)},
                                 model_artifact=artifact)
        assert row["projected_draft_capital_score"] is not None
        # Capital carries real weight in the synthetic fit, so the
        # blind grade must move and the delta must reconcile.
        assert row["talent_score"] != row["prospect_score"]
        assert row["capital_delta"] == pytest.approx(
            row["prospect_score"] - row["talent_score"], abs=0.02)

    def test_no_capital_no_delta(self, artifact):
        (row,) = v5.score_all_v5([SMITH], {}, model_artifact=artifact)
        assert row["projected_draft_capital_score"] is None
        assert row["capital_delta"] == 0.0
        assert row["talent_score"] == row["prospect_score"]


# ─────────────────────────────────────────────────────────────────────
# Artifact: round-trip, fingerprint, fold hygiene
# ─────────────────────────────────────────────────────────────────────


class TestArtifact:
    def test_round_trip_in_memory(self, artifact):
        blob = v5.artifact_to_bytes(artifact)
        back = v5.artifact_from_bytes(blob)
        assert back["fingerprint"] == artifact["fingerprint"]
        assert set(back["models"]) == set(artifact["models"])
        (row,) = v5.score_all_v5([SMITH], {"T_smith": _dc("WR", 1)},
                                 model_artifact=blob)
        assert row["prospect_score"] is not None

    def test_fingerprint_records_population(self, artifact):
        fp = artifact["fingerprint"]
        assert fp["population_rows"] == 250
        assert fp["classes"] == [2020]
        assert fp["random_seed"] == v5.RANDOM_SEED
        assert fp["feature_lists"]["WR"] == v5.FEATURES_BY_POSITION["WR"]
        assert len(fp["population_hash"]) == 64

    def test_fold_hygiene_anchors_exclude_held_out(self):
        # Class A outcomes live in a low range; class B's are huge.
        # An artifact fit on A only must carry A's ECDF only: B's
        # outcomes never touched the anchors.
        rows_a = _synthetic_rows("WR", 40, seed=31, class_year=2019)
        rows_b = _synthetic_rows("WR", 40, seed=32, outcome_scale=6.0,
                                 class_year=2020)
        art_a = v5.fit_artifact(rows_a)
        ecdf_a = art_a["outcome_ecdf"]["WR"]
        assert max(ecdf_a) == max(r["outcome"] for r in rows_a)
        assert max(ecdf_a) < min(r["outcome"] for r in rows_b) \
            or max(ecdf_a) < max(r["outcome"] for r in rows_b)
        assert art_a["fingerprint"]["classes"] == [2019]
        # A B-scale prediction against A's anchors extrapolates
        # past the recorded max instead of being capped by it.
        pct = v5.outcome_percentile(ecdf_a, max(r["outcome"]
                                                for r in rows_b))
        assert pct > 99.0


# ─────────────────────────────────────────────────────────────────────
# Pipeline selector
# ─────────────────────────────────────────────────────────────────────


class TestPipelineSelector:
    def test_selector_accepts_v5(self, monkeypatch):
        for raw, want in (("v5", "v5"), ("5", "v5"), ("5.0", "v5"),
                          ("V5", "v5")):
            monkeypatch.setenv("PROSPECT_MODEL", raw)
            assert pipeline.selected_prospect_model() == want

    def test_default_unchanged(self, monkeypatch):
        monkeypatch.delenv("PROSPECT_MODEL", raising=False)
        assert pipeline.DEFAULT_PROSPECT_MODEL == "v4"
        assert pipeline.selected_prospect_model() == "v4"

    def test_v5_fails_closed_without_artifact(self, monkeypatch):
        monkeypatch.setenv("PROSPECT_MODEL", "v5")
        monkeypatch.setattr(pipeline, "_V5_ARTIFACT_CACHE", None)

        def _no_artifact():
            raise v5.V5ArtifactError("no v5 model artifact is stored yet")

        monkeypatch.setattr(v5, "load_artifact_from_db", _no_artifact)
        with pytest.raises(v5.V5ArtifactError, match="no v5 model artifact"):
            pipeline.assert_scorer_ready()

    def test_v5_ready_with_artifact(self, monkeypatch, artifact):
        monkeypatch.setenv("PROSPECT_MODEL", "v5")
        monkeypatch.setattr(pipeline, "_V5_ARTIFACT_CACHE", artifact)
        assert pipeline.assert_scorer_ready() == "v5"
