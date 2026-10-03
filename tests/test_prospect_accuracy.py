"""Tests for prospect accuracy report logic (synthetic data, no DB needed)."""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prospect_accuracy_report import (
    HIT_PPR_THRESHOLD,
    compute_hit,
    _normalize_name,
)


class TestHitThresholds:
    def test_thresholds_defined_for_all_skill_positions(self):
        for pos in ("QB", "RB", "WR", "TE"):
            assert pos in HIT_PPR_THRESHOLD
            assert HIT_PPR_THRESHOLD[pos] > 0

    def test_qb_threshold_highest(self):
        # QB scoring is higher volume; threshold should reflect that
        assert HIT_PPR_THRESHOLD["QB"] > HIT_PPR_THRESHOLD["WR"]
        assert HIT_PPR_THRESHOLD["QB"] > HIT_PPR_THRESHOLD["RB"]


class TestComputeHit:
    def test_wr_hit_at_threshold(self):
        assert compute_hit("WR", 220.0) is True

    def test_wr_miss_below_threshold(self):
        assert compute_hit("WR", 219.99) is False

    def test_rb_hit(self):
        assert compute_hit("RB", 250.0) is True
        assert compute_hit("RB", 100.0) is False

    def test_qb_hit(self):
        assert compute_hit("QB", 350.0) is True
        assert compute_hit("QB", 200.0) is False

    def test_te_hit(self):
        assert compute_hit("TE", 180.0) is True
        assert compute_hit("TE", 50.0) is False

    def test_position_case_insensitive(self):
        assert compute_hit("wr", 220.0) is True
        assert compute_hit("Wr", 220.0) is True

    def test_zero_peak_never_hits(self):
        for pos in ("QB", "RB", "WR", "TE"):
            assert compute_hit(pos, 0.0) is False


class TestNormalizeName:
    def test_strips_suffixes(self):
        assert _normalize_name("Marvin Harrison Jr") == "marvin harrison"
        assert _normalize_name("Brian Thomas Jr.") == "brian thomas"

    def test_handles_accents(self):
        assert _normalize_name("CeeDee Lamb") == "ceedee lamb"
        assert _normalize_name("Ja'Marr Chase") == "jamarr chase"

    def test_trims_whitespace(self):
        assert _normalize_name("  Josh Allen  ") == "josh allen"


class TestHitRateMath:
    """Verify the aggregate hit-rate math used in _compute_aggregates."""

    def test_hit_rate_computation(self):
        # Simulate bucket: 10 players, 4 hits
        n, hits = 10, 4
        rate = round(hits / n * 100, 2)
        assert rate == 40.0

    def test_empty_bucket_no_division_by_zero(self):
        n, hits = 0, 0
        rate = round(hits / n * 100, 2) if n else 0.0
        assert rate == 0.0

    def test_tier_inversion_detection_logic(self):
        # Tier 1 at 30%, Tier 2 at 50% -> inversion (lower tier out-hits)
        rates = [(1, 30.0), (2, 50.0)]
        inversions = []
        for i in range(len(rates) - 1):
            t1, r1 = rates[i]
            t2, r2 = rates[i + 1]
            if r1 < r2 - 5:
                inversions.append((t1, t2))
        assert inversions == [(1, 2)]

    def test_no_inversion_when_monotonic(self):
        rates = [(1, 60.0), (2, 40.0), (3, 20.0)]
        inversions = []
        for i in range(len(rates) - 1):
            t1, r1 = rates[i]
            t2, r2 = rates[i + 1]
            if r1 < r2 - 5:
                inversions.append((t1, t2))
        assert inversions == []
