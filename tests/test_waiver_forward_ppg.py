"""Unit tests for the waiver forward-projection resolver.

_resolve_forward_ppg pins the None-vs-0.0 contract behind the waiver card's
PROJ number: None means the projection feed itself is unavailable for the
window (unknown -- callers may fall back to backward production); 0.0 means a
live feed carries no projection for the player (benched/cut -- the honest
forward read is zero, and substituting their old scoring average as "PROJ"
would mislead).
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from routes.waiver_bp import _resolve_forward_ppg  # noqa: E402


def test_mean_of_weekly_values():
    assert _resolve_forward_ppg([12.0, 14.0, 16.0], True) == pytest.approx(14.0)


def test_single_value():
    assert _resolve_forward_ppg([9.5], True) == pytest.approx(9.5)


def test_absent_from_live_feed_is_zero_not_none():
    # The Drew Lock case: feed is live (other players projected) but he is in
    # none of the upcoming weeks -> 0.0, so ros_ppg never falls back to his
    # 17.1 backward average.
    assert _resolve_forward_ppg([], True) == 0.0


def test_unpublished_feed_is_unknown():
    # Feed not yet published for the window -> None, preserving the
    # backward-ppg fallback for genuinely unknown (not benched) players.
    assert _resolve_forward_ppg([], False) is None
    assert _resolve_forward_ppg([12.0], False) is None


def test_explicit_zeros_average_to_zero():
    assert _resolve_forward_ppg([0.0, 0.0], True) == 0.0
