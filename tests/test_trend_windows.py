"""Unit tests for the shared trend-window helper (data_building/trend_windows).

The window rule is canonical for the Advanced Metrics trend columns and the
weekly usage-trends payload: 2-3 samples compare the latest value against
the average of the priors; 4+ samples compare the last 3 against the whole
series (season) average. Missing samples are skipped, never zero-filled.
"""
import pytest

from data_building.trend_windows import (
    recent_vs_baseline_delta,
    recent_vs_baseline_ratio,
    trend_window,
)


# --------------------------------------------------------------------------- #
# trend_window: sample-count shapes
# --------------------------------------------------------------------------- #

def test_window_none_below_two_samples():
    assert trend_window([]) == (None, None)
    assert trend_window([4.0]) == (None, None)
    assert trend_window([None]) == (None, None)
    assert trend_window([None, None]) == (None, None)


def test_window_two_samples_latest_vs_prior():
    recent, baseline = trend_window([4.0, 7.0])
    assert recent == 7.0
    assert baseline == 4.0


def test_window_three_samples_latest_vs_prior_average():
    recent, baseline = trend_window([4.0, 7.0, 5.0])
    assert recent == 5.0
    assert baseline == pytest.approx(5.5)


def test_window_four_samples_last_three_vs_season_average():
    recent, baseline = trend_window([2.0, 2.0, 2.0, 8.0])
    assert recent == pytest.approx(4.0)          # (2+2+8)/3
    assert baseline == pytest.approx(3.5)        # whole-series average


def test_window_five_plus_samples_last_three_vs_season_average():
    recent, baseline = trend_window([5.0, 5.0, 5.0, 5.0, 15.0])
    assert recent == pytest.approx(25.0 / 3)     # (5+5+15)/3
    assert baseline == pytest.approx(7.0)        # 35/5


def test_window_custom_recent_n():
    recent, baseline = trend_window([1.0, 2.0, 3.0, 4.0, 5.0], recent_n=2)
    assert recent == pytest.approx(4.5)
    assert baseline == pytest.approx(3.0)
    # len == recent_n still takes the early-season branch.
    recent, baseline = trend_window([2.0, 6.0], recent_n=2)
    assert recent == 6.0
    assert baseline == 2.0


# --------------------------------------------------------------------------- #
# missing values: skipped, never coerced to 0
# --------------------------------------------------------------------------- #

def test_window_missing_values_are_skipped_not_zeroed():
    # The None week contributes no sample: this is the 3-sample case
    # [4, 7, 5], NOT a 4-sample case with a fabricated 0.
    recent, baseline = trend_window([4.0, None, 7.0, 5.0])
    assert recent == 5.0
    assert baseline == pytest.approx(5.5)
    # A series of only-missing plus one real value has no comparison.
    assert trend_window([None, 9.0]) == (None, None)


# --------------------------------------------------------------------------- #
# ratio
# --------------------------------------------------------------------------- #

def test_ratio_none_without_comparison_or_with_zero_baseline():
    assert recent_vs_baseline_ratio([]) is None
    assert recent_vs_baseline_ratio([5.0]) is None
    assert recent_vs_baseline_ratio([0.0, 0.0, 5.0]) is None   # zero baseline
    assert recent_vs_baseline_ratio([0.0, 0.0, 0.0, 0.0]) is None


def test_ratio_early_window():
    assert recent_vs_baseline_ratio([1.0, 2.0]) == pytest.approx(1.0)
    assert recent_vs_baseline_ratio([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_ratio_full_window():
    # last-3 avg 3.0 vs season avg 2.25 -> +1/3
    assert recent_vs_baseline_ratio([0.0, 2.0, 3.0, 4.0]) == pytest.approx(1 / 3)
    # declining: last-3 avg 6.0 vs season avg 6.5 -> -1/13
    assert recent_vs_baseline_ratio([8.0, 8.0, 8.0, 2.0]) == pytest.approx(-1 / 13)


# --------------------------------------------------------------------------- #
# delta
# --------------------------------------------------------------------------- #

def test_delta_none_below_two_samples():
    assert recent_vs_baseline_delta([]) is None
    assert recent_vs_baseline_delta([4.0]) is None


def test_delta_early_window():
    assert recent_vs_baseline_delta([4.0, 7.0]) == pytest.approx(3.0)
    assert recent_vs_baseline_delta([4.0, 7.0, 5.0]) == pytest.approx(-0.5)


def test_delta_full_window():
    assert recent_vs_baseline_delta([2.0, 2.0, 2.0, 8.0]) == pytest.approx(0.5)
    assert recent_vs_baseline_delta([8.0, 8.0, 8.0, 2.0]) == pytest.approx(-0.5)


def test_delta_zero_baseline_is_a_real_delta():
    # Unlike the ratio, a delta against a zero baseline is meaningful.
    assert recent_vs_baseline_delta([0.0, 0.0, 5.0]) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# parity with the surfaces that adopted the helper
# --------------------------------------------------------------------------- #

def test_am_ratio_wrapper_matches_helper():
    from data_building.advanced_metrics import _recent_vs_season_ratio
    for vals in ([], [5.0], [0.0, 0.0, 5.0], [1.0, 2.0], [1.0, 2.0, 3.0],
                 [0.0, 2.0, 3.0, 4.0], [3.0, 3.0, 9.0, 9.0, 9.0, 1.0]):
        expected = recent_vs_baseline_ratio(list(vals))
        got = _recent_vs_season_ratio(list(vals))
        assert (got is None and expected is None) or got == pytest.approx(expected)


def test_weekly_metrics_delta_wrapper_matches_helper():
    from data_building.weekly_metrics import _recent_vs_season_delta
    for vals in ([], [4.0], [4.0, 7.0], [4.0, 7.0, 5.0], [2.0, 2.0, 2.0, 8.0]):
        expected = recent_vs_baseline_delta(list(vals))
        got = _recent_vs_season_delta(list(vals))
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(round(expected, 1))
