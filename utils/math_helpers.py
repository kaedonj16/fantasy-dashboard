"""
Shared numeric utility functions used across the data_building pipeline.

Centralising these prevents the same micro-helpers from being copy-pasted into
every module (previously duplicated in components.py, player_value.py,
prospect_model.py, and others).

All functions are pure (no I/O, no side effects) and deliberately small so
they remain easy to inline-test or replace.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Safe coercion
# ---------------------------------------------------------------------------

def safe_float(value, default: float = 0.0) -> float:
    """Coerce *value* to float, returning *default* on failure or None/''."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    """Coerce *value* to int, returning *default* on failure or None/''."""
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_str(value, default: str = "") -> str:
    """Coerce *value* to a stripped string, returning *default* on None."""
    if value is None:
        return default
    return str(value).strip()


# ---------------------------------------------------------------------------
# Clamp / clip
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to [low, high]."""
    return max(low, min(high, value))


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Alias for clamp with [0, 1] defaults (common normalisation range)."""
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_to_one(value: float, full_value: float) -> float:
    """Normalize *value* to [0, 1] against a max / full-confidence ceiling."""
    if full_value <= 0:
        return 0.0
    return clamp(value / full_value, 0.0, 1.0)


def normalize_range(value: float, low: float, high: float) -> float:
    """Linear map *value* from [low, high] → [0, 1], clamped."""
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def scale(raw: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] → [0, 100], clipped. Convenience for 0-100 scores."""
    if hi <= lo:
        return 50.0
    return clip((raw - lo) / (hi - lo) * 100.0, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Curve functions
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid: 1 / (1 + exp(-x))."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_score(x: float, midpoint: float, steepness: float = 1.0) -> float:
    """Logistic curve scaled to 0-100, centred at *midpoint*."""
    z = steepness * (x - midpoint)
    return round(100.0 / (1.0 + math.exp(-z)), 2)


def smoothstep01(x: float) -> float:
    """
    Smooth Hermite interpolation in [0, 1].

    Derivative is zero at both endpoints so the curve has no abrupt corners
    at x=0 or x=1. Useful as a confidence ramp-up function.
    """
    x = clip(x)
    return x * x * (3.0 - 2.0 * x)


def pct_change(curr: float, prev: float, neutral_when_prev_zero: Optional[float] = None) -> Optional[float]:
    """Percentage change from *prev* to *curr*. Returns *neutral_when_prev_zero* when prev==0."""
    if prev == 0:
        return neutral_when_prev_zero
    return ((curr - prev) / prev) * 100.0


# ---------------------------------------------------------------------------
# Weighted aggregation
# ---------------------------------------------------------------------------

def weighted_average(pairs: List[Tuple[float, float]]) -> float:
    """
    Weighted mean of (value, weight) pairs.

    Zero-weight pairs are excluded from both numerator and denominator so
    they don't dilute the result when weights are structurally absent.
    """
    total_weight = sum(w for _, w in pairs if w > 0)
    if total_weight <= 0:
        return 0.0
    return sum(v * w for v, w in pairs if w > 0) / total_weight


def sample_confidence(
        observed: float,
        full_confidence: float,
        min_confidence: float = 0.35,
) -> float:
    """
    Bayesian-style reliability multiplier in [min_confidence, 1.0].

    As *observed* approaches *full_confidence*, the factor asymptotically
    approaches 1.0. Used to shrink estimates from small sample sizes.
    """
    if full_confidence <= 0:
        return 1.0
    ratio = clamp(observed / full_confidence, 0.0, 1.0)
    return min_confidence + (1.0 - min_confidence) * ratio
