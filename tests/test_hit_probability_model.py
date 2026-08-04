"""The breakout hit-probability inference: fitted model when present, curve otherwise.

Guards the two paths in calculate_hit_probability:
  - no committed model file -> falls back to the empirical curve (unchanged behavior)
  - a fitted model present   -> uses the per-position logistic (z-score + sigmoid)
the pure logistic math in _hit_prob_from_model, and the per-component contribution
breakdown the modal renders.

The model is fit on the six RAW component scores (_HIT_MODEL_FEATURES), so the test
model blocks carry length-6 mean/std/coef arrays.
"""
import math

import pytest

mp = pytest.importorskip("data_building.breakout_engine.multitask_predictions")

N = len(mp._HIT_MODEL_FEATURES)  # 6 raw components


def _block(coef, intercept, mean=0.0, std=1.0):
    return {"mean": [mean] * N, "std": [std] * N, "coef": list(coef), "intercept": intercept}


def test_curve_fallback_when_no_model(monkeypatch):
    # Force "no model" and confirm we still get a sane curve probability.
    monkeypatch.setattr(mp._load_hit_model, "cache_clear", lambda: None, raising=False)
    monkeypatch.setattr(mp, "_load_hit_model", lambda: None)
    p = mp.calculate_hit_probability(80, 70, 70, "WR", opportunity_score=70, role_trajectory_score=75)
    assert 0.01 <= p <= 0.95
    # A high-score WR should beat a low-score one on the curve.
    lo = mp.calculate_hit_probability(30, 30, 30, "WR")
    assert p > lo


def test_hit_prob_from_model_matches_logistic():
    # z-scored logistic: mean 0, std 1, so logit = intercept + coef·features.
    # First feature is opportunity_opened_score; give it a coef and drive it.
    model = {
        "features": list(mp._HIT_MODEL_FEATURES),
        "positions": {"WR": _block([0.04] + [0.0] * (N - 1), -3.0)},
    }
    feats = {f: 0.0 for f in mp._HIT_MODEL_FEATURES}
    feats["opportunity_opened_score"] = 100
    got = mp._hit_prob_from_model(model, "WR", feats)
    expected = 1.0 / (1.0 + math.exp(-(-3.0 + 0.04 * 100)))  # sigmoid(1.0)
    assert got == pytest.approx(expected, abs=1e-9)


def test_model_used_when_present(monkeypatch):
    # Inject a model that returns a fixed-ish high prob and confirm the fitted path
    # (not the curve) drives the output.
    model = {
        "features": list(mp._HIT_MODEL_FEATURES),
        "positions": {"_global": _block([0.0] * N, 3.0)},  # sigmoid(3)=~0.953 -> clamps to 0.95
    }
    monkeypatch.setattr(mp, "_load_hit_model", lambda: model)
    p = mp.calculate_hit_probability(10, 10, 10, "QB")
    assert p == 0.95  # clamped high end, proving the model path ran


def test_curve_position_skips_model(monkeypatch):
    # QB is flagged as a curve position -> the fitted model is bypassed and the
    # empirical curve drives the output, even though a _global block exists.
    model = {
        "features": list(mp._HIT_MODEL_FEATURES),
        "curve_positions": ["QB"],
        "positions": {"_global": _block([0.0] * N, 3.0)},  # would clamp to 0.95
    }
    monkeypatch.setattr(mp, "_load_hit_model", lambda: model)
    p = mp.calculate_hit_probability(10, 10, 10, "QB")
    assert p != 0.95  # model path skipped for QB; low score -> low curve prob
    assert 0.01 <= p <= 0.95
    # A non-curve position still uses the model (clamped high).
    assert mp.calculate_hit_probability(10, 10, 10, "WR") == 0.95


def test_global_block_fallback_for_unknown_position(monkeypatch):
    model = {
        "features": list(mp._HIT_MODEL_FEATURES),
        "positions": {"_global": _block([0.0] * N, -3.0)},
    }
    monkeypatch.setattr(mp, "_load_hit_model", lambda: model)
    # No WR block -> uses _global.
    p = mp.calculate_hit_probability(50, 50, 50, "WR")
    assert p == pytest.approx(round(1.0 / (1.0 + math.exp(3.0)), 3), abs=1e-3)


def test_contributions_signed_and_ordered(monkeypatch):
    # A positive coef on an above-mean feature contributes positively; a negative
    # coef (or below-mean feature) contributes negatively. Bars = coef * z.
    coef = [0.05, -0.03] + [0.0] * (N - 2)  # opportunity +, competition -
    model = {
        "features": list(mp._HIT_MODEL_FEATURES),
        "positions": {"WR": _block(coef, -2.0, mean=50.0, std=10.0)},
    }
    monkeypatch.setattr(mp, "_load_hit_model", lambda: model)
    feats = {f: 50.0 for f in mp._HIT_MODEL_FEATURES}
    feats["opportunity_opened_score"] = 70.0   # +2 sd -> +0.05*2 = +0.10
    feats["competition_removed_score"] = 70.0  # +2 sd, neg coef -> -0.03*2 = -0.06
    out = mp.hit_probability_contributions("WR", feats)
    assert out is not None and len(out) == N
    by = {r["feature"]: r for r in out}
    assert by["opportunity_opened_score"]["contribution"] == pytest.approx(0.10, abs=1e-6)
    assert by["competition_removed_score"]["contribution"] == pytest.approx(-0.06, abs=1e-6)
    # Feature sitting at the mean contributes ~0.
    assert by["team_environment_score"]["contribution"] == pytest.approx(0.0, abs=1e-9)
    assert by["opportunity_opened_score"]["label"] == "Opportunity"


def test_contributions_none_for_curve_position(monkeypatch):
    model = {
        "features": list(mp._HIT_MODEL_FEATURES),
        "curve_positions": ["QB"],
        "positions": {"_global": _block([0.05] * N, -2.0)},
    }
    monkeypatch.setattr(mp, "_load_hit_model", lambda: model)
    assert mp.hit_probability_contributions("QB", {f: 50.0 for f in mp._HIT_MODEL_FEATURES}) is None


def test_contributions_none_when_no_model(monkeypatch):
    monkeypatch.setattr(mp, "_load_hit_model", lambda: None)
    assert mp.hit_probability_contributions("WR", {f: 50.0 for f in mp._HIT_MODEL_FEATURES}) is None
