"""Covers the pure rank-correlation helpers of the opp-plays-faced backtest.

The backtest's data pulls need nfl_data_py, but its Spearman implementation is
plain Python and worth guarding, since a wrong rho would silently mislead the
"is this stat worth trusting?" decision.
"""
import importlib.util
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "backtest_opp_plays_faced.py")
_spec = importlib.util.spec_from_file_location("backtest_opp_plays_faced", _PATH)
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def test_spearman_monotonic():
    assert bt.spearman([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]) == 1.0


def test_spearman_inverse():
    assert bt.spearman([1, 2, 3, 4, 5, 6], [60, 50, 40, 30, 20, 10]) == -1.0


def test_spearman_handles_ties():
    rho = bt.spearman([1, 1, 2, 3, 3, 4], [2, 1, 3, 5, 4, 6])
    assert 0.9 < rho < 1.0


def test_spearman_small_sample_none():
    assert bt.spearman([1, 2, 3], [3, 2, 1]) is None


def test_parse_seasons():
    assert bt._parse_seasons("2022,2023,2024") == [2022, 2023, 2024]
