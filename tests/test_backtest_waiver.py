"""Tests for the pure evaluation layer of scripts/backtest_waiver_targets.py:
metrics, the added baselines, coverage reporting, and leakage-safe recommendation
snapshots. No DB / app import."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_waiver_targets import (  # noqa: E402
    WeekSnapshot,
    evaluate,
    persist_recommendation_snapshot,
    precision_at_k,
    spearman,
)
from utils.waiver_score import WEIGHTS  # noqa: E402


def test_spearman_and_precision_basics():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert precision_at_k([3, 2, 1], [3, 2, 1], 2) == pytest.approx(1.0)


def _snap():
    cands = [{"player_id": str(i), "position": "RB", "value": v, "rank_change_7d": 0}
             for i, v in enumerate([900, 700, 500, 300, 200, 150, 120, 100, 80, 60,
                                    50, 40, 30, 25, 20, 15])]
    realized = {c["player_id"]: c["value"] / 10.0 for c in cands}  # value ~ outcome
    last_week = {c["player_id"]: c["value"] / 12.0 for c in cands}
    projection = {c["player_id"]: c["value"] / 11.0 for c in cands}
    return WeekSnapshot(candidates=cands, breakout={}, realized_points=realized,
                        last_week_points=last_week, projection=projection)


def test_evaluate_reports_all_baselines_and_coverage():
    m = evaluate([_snap()], WEIGHTS, k=5)
    for key in ("spearman_model", "spearman_value_only", "spearman_projection_only",
                "spearman_last_week", "precision_at_k_model",
                "precision_at_k_projection_only", "precision_at_k_last_week"):
        assert key in m
    assert m["coverage_projection_weeks"] == 1
    assert m["coverage_last_week_weeks"] == 1


def test_evaluate_uncovered_baseline_not_faked():
    # No projection / last-week data -> those baselines are uncovered (0 weeks),
    # never silently scored.
    s = _snap()
    s.projection = {}
    s.last_week_points = {}
    m = evaluate([s], WEIGHTS, k=5)
    assert m["coverage_projection_weeks"] == 0
    assert m["coverage_last_week_weeks"] == 0


def test_persist_recommendation_snapshot_is_leakage_safe(tmp_path):
    path = persist_recommendation_snapshot(2024, 6, _snap(), WEIGHTS, str(tmp_path), k=5)
    data = json.loads(open(path).read())
    assert data["leakage_safe"] is True
    assert data["season"] == 2024 and data["week"] == 6
    rows = data["recommendations"]
    assert len(rows) == 5
    # Only as-of-week features + the graded outcome are stored — no current
    # ownership / injury / revised-projection fields.
    allowed = {"player_id", "position", "score", "value", "rank_change_7d",
               "breakout", "projection", "last_week_points", "realized_points"}
    for r in rows:
        assert set(r) <= allowed
    # Ranked by model score, descending.
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
