"""Unit tests for streak_edge_class (standings streak edge signal)."""
import pytest

pytest.importorskip("pandas")

from utils.utils import streak_edge_class


@pytest.mark.parametrize(
    "streak,expected",
    [
        ("W1", "streak-w1"),
        ("W2", "streak-w2"),
        ("W3", "streak-w3"),
        ("W4", "streak-w4plus"),
        ("W7", "streak-w4plus"),
        ("L1", "streak-l1"),
        ("L2", "streak-l2"),
        ("L3", "streak-l3"),
        ("L5", "streak-l4plus"),
        ("w3", "streak-w3"),      # lowercase accepted
        (" L2 ", "streak-l2"),    # surrounding whitespace
        ("", ""),
        (None, ""),
        ("-", ""),
        ("W0", ""),
    ],
)
def test_streak_edge_class(streak, expected):
    assert streak_edge_class(streak) == expected
