"""Guards the cohort-relative breakout tiering in breakout_api.

Fixed absolute cutoffs (55/45/35 on the overall score) drift year to year: a
strong season over-crowns "elite" breakouts and a weak one under-labels them.
compute_tier_thresholds blends each anchor 50/50 with a cohort percentile so the
tiers stay calibrated to the field while staying tethered to the absolute
meaning. These tests pin that behavior and the small-cohort fallback.
"""
from dashboard_services.breakout_api import (
    _ABS_TIERS,
    classify_breakout_type,
    compute_tier_thresholds,
)


def _label(overall, thresholds=None):
    return classify_breakout_type(60.0, 40.0, overall, tier_thresholds=thresholds)["profile_label"]


def test_thresholds_are_ordered():
    th = compute_tier_thresholds([80, 75, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54])
    assert th["elite"] > th["strong"] > th["moderate"]


def test_small_cohort_falls_back_to_absolute():
    # Fewer than 8 scores: not enough to trust percentiles.
    assert compute_tier_thresholds([50, 40, 30]) is None
    # With no thresholds, the absolute anchors apply: 52 < 55 -> not elite.
    assert "Elite" not in _label(52.0)


def test_weak_season_promotes_the_field_topper():
    weak = [52, 50, 49, 47, 45, 44, 42, 40, 38, 36, 34, 30]
    th = compute_tier_thresholds(weak)
    # Absolutely 52 is only "strong", but it tops a weak field, so cohort-relative
    # it reads as an elite breakout.
    assert "Elite" not in _label(52.0)              # absolute
    assert "Elite" in _label(52.0, th)              # cohort-relative


def test_hot_season_does_not_auto_crown_a_mid_scorer():
    hot = [80, 75, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54]
    th = compute_tier_thresholds(hot)
    # Absolutely 56 clears 55 and would be "elite", but it's mid-pack in a hot
    # field, so cohort-relative it drops to "strong".
    assert "Elite" in _label(56.0)                  # absolute
    assert "Elite" not in _label(56.0, th)          # cohort-relative


def test_blend_prevents_mediocre_from_being_crowned():
    # A uniformly weak field shouldn't manufacture an "elite" from a low score:
    # the 50/50 blend with the absolute anchor keeps a floor.
    weak = [30, 28, 27, 25, 24, 22, 20, 18, 16, 15]
    th = compute_tier_thresholds(weak)
    assert th["elite"] >= _ABS_TIERS["elite"] * 0.5  # anchored, not free-floating
    assert "Elite" not in _label(32.0, th)
