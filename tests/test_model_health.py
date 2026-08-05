"""model_health metrics + the champion/challenger gate metric.

The pure ranking helpers (Precision@K, gate metric, applying a model to a row)
must work without sklearn (CI has no sklearn); the AUC-based pooled metrics are
guarded with importorskip.
"""
import math

import pytest

mh = pytest.importorskip("data_building.breakout_engine.model_health")


def _row(pos, score, prob, hit, feats=None):
    return {
        "pid": f"{pos}{score}", "pos": pos,
        "feats": feats or {f: 50.0 for f in mh.FEATURES},
        "score": score, "prob": prob, "hit": hit,
        "prev_ppg": 8.0, "actual_ppg": 12.0 if hit else 6.0,
    }


def test_precision_at_k_counts_top_k_hits():
    rows = [_row("WR", 0.9, 0.9, 1), _row("WR", 0.8, 0.8, 1),
            _row("WR", 0.7, 0.7, 0), _row("WR", 0.1, 0.1, 1)]
    scores = [r["score"] for r in rows]
    hits = [r["hit"] for r in rows]
    assert mh._precision_at_k(scores, hits, 2) == pytest.approx(1.0)   # top-2 both hit
    assert mh._precision_at_k(scores, hits, 3) == pytest.approx(2 / 3)  # 3rd is a miss


def test_precision_at_k_handles_k_larger_than_pool():
    assert mh._precision_at_k([0.5, 0.6], [1, 0], 10) == pytest.approx(0.5)


def test_blend_metric_fn_weights_score_vs_prob():
    r = _row("WR", 0.8, 0.2, 0)
    assert mh.blend_metric_fn(0.0)(r) == pytest.approx(0.2)   # all prob
    assert mh.blend_metric_fn(1.0)(r) == pytest.approx(0.8)   # all score
    assert mh.blend_metric_fn(0.5)(r) == pytest.approx(0.5)


def test_prob_under_model_none_returns_shipped_prob():
    r = _row("WR", 0.8, 0.42, 1)
    assert mh.prob_under_model(r, None) == pytest.approx(0.42)


def test_prob_under_model_curve_position_falls_back_to_shipped():
    model = {"positions": {"WR": {"mean": [0]*6, "std": [1]*6, "coef": [0]*6, "intercept": 0.0}},
             "features": list(mh.FEATURES), "curve_positions": ["TE"]}
    te = _row("TE", 0.8, 0.33, 0)
    assert mh.prob_under_model(te, model) == pytest.approx(0.33)  # TE is a curve position


def test_prob_under_model_applies_logistic_for_fitted_position():
    # intercept 0, all coef 0 -> logit 0 -> prob 0.5, regardless of features
    model = {"positions": {"WR": {"mean": [0]*6, "std": [1]*6, "coef": [0]*6, "intercept": 0.0}},
             "features": list(mh.FEATURES), "curve_positions": []}
    assert mh.prob_under_model(_row("WR", 0.8, 0.1, 0), model) == pytest.approx(0.5, abs=1e-6)

    # positive intercept -> prob > 0.5
    model["positions"]["WR"]["intercept"] = 2.0
    p = mh.prob_under_model(_row("WR", 0.8, 0.1, 0), model)
    assert p == pytest.approx(1 / (1 + math.exp(-2.0)), abs=1e-6)


def test_gate_metric_rewards_auc_and_precision():
    lo = {"auc": 0.55, "mean_p_at_10": 0.2}
    hi = {"auc": 0.70, "mean_p_at_10": 0.5}
    assert mh.gate_metric(hi) > mh.gate_metric(lo)
    # None AUC is treated as chance (0.5), not a crash
    assert mh.gate_metric({"auc": None, "mean_p_at_10": 0.0}) == pytest.approx(0.3)


def test_pooled_metrics_precision_and_labels():
    pytest.importorskip("sklearn")  # AUC path needs sklearn
    rows_by_season = {
        2023: [_row("WR", 0.9, 0.9, 1), _row("WR", 0.8, 0.8, 1),
               _row("RB", 0.7, 0.7, 0), _row("TE", 0.1, 0.1, 0)],
    }
    m = mh.pooled_metrics(rows_by_season, lambda r: r["score"])
    assert m["n"] == 4 and m["hits"] == 2
    # top-2 by score are the two hits
    assert m["per_season"][2023]["p_at_10"] == pytest.approx(0.5)  # 2 hits / min(10, 4)=4
    assert 0.0 <= m["auc"] <= 1.0
