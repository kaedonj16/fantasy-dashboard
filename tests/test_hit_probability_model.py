"""The breakout hit-probability inference: fitted model when present, curve otherwise.

Guards the two paths in calculate_hit_probability:
  - no committed model file -> falls back to the empirical curve (unchanged behavior)
  - a fitted model present   -> uses the per-position logistic (z-score + sigmoid)
and the pure logistic math in _hit_prob_from_model.
"""
import math

import pytest

mp = pytest.importorskip("data_building.breakout_engine.multitask_predictions")


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
    model = {
        "features": list(mp._HIT_MODEL_FEATURES),
        "positions": {
            "WR": {
                "mean": [0, 0, 0, 0, 0],
                "std": [1, 1, 1, 1, 1],
                "coef": [0.04, 0.0, 0.0, 0.0, 0.0],
                "intercept": -3.0,
            }
        },
    }
    feats = {"breakout_score": 100, "readiness_score": 0, "confidence_score": 0,
             "opportunity_score": 0, "role_trajectory_score": 0}
    got = mp._hit_prob_from_model(model, "WR", feats)
    expected = 1.0 / (1.0 + math.exp(-(-3.0 + 0.04 * 100)))  # sigmoid(1.0)
    assert got == pytest.approx(expected, abs=1e-9)


def test_model_used_when_present(monkeypatch):
    # Inject a model that returns a fixed-ish high prob for QB and confirm the
    # fitted path (not the curve) drives the output.
    model = {
        "features": list(mp._HIT_MODEL_FEATURES),
        "positions": {"_global": {"mean": [0]*5, "std": [1]*5,
                                  "coef": [0]*5, "intercept": 3.0}},  # sigmoid(3)=~0.953 -> clamps to 0.95
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
        "positions": {"_global": {"mean": [0]*5, "std": [1]*5,
                                  "coef": [0]*5, "intercept": 3.0}},  # would clamp to 0.95
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
        "positions": {"_global": {"mean": [0]*5, "std": [1]*5,
                                  "coef": [0]*5, "intercept": -3.0}},
    }
    monkeypatch.setattr(mp, "_load_hit_model", lambda: model)
    # No WR block -> uses _global.
    p = mp.calculate_hit_probability(50, 50, 50, "WR")
    assert p == pytest.approx(round(1.0 / (1.0 + math.exp(3.0)), 3), abs=1e-3)
