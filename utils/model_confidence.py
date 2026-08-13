"""Shared, explainable confidence and rank-stability helpers."""
from __future__ import annotations

import math


def confidence_from_inputs(available: int, expected: int, sample_size: int = 0) -> dict:
    """Confidence derived from completeness plus optional historical sample size."""
    completeness = min(1.0, max(0.0, available / max(expected, 1)))
    sample = 1.0 - math.exp(-max(sample_size, 0) / 100.0) if sample_size else completeness
    score = round(100 * (0.7 * completeness + 0.3 * sample))
    label = "High" if score >= 80 else "Medium" if score >= 55 else "Low"
    return {"score": score, "label": label, "completeness": round(completeness, 3)}


def rank_interval(rank: int, confidence_score: float, field_size: int) -> tuple[int, int]:
    """Explainable likely-rank range that narrows as confidence increases."""
    uncertainty = max(0.0, min(1.0, 1.0 - float(confidence_score) / 100.0))
    spread = max(1, round(max(field_size, 1) * uncertainty * 0.25))
    return max(1, rank - spread), min(max(field_size, 1), rank + spread)
