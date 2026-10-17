"""Daily move cap (Option A, Kaedon 2026-10-16): both served value layers
clamp a player's day-over-day move to ±10% of the previous value by default.

Layer 1 (model board): data_building/value_model_training.py
    _daily_move_cap() + _clamp_board_value(), applied in
    rewrite_value_table_with_model() against yesterday's stored board.
Layer 2 (calibrated track): data_building/trade_intel/trade_value_model.py
    DAILY_MOVE_CAP + _clamp_to_prev(), applied to WLS calibrated values
    (the track that displays whenever it exists).

Both layers read the same VALUE_MOVE_CAP_OVERRIDE env var; the code default
is 0.10 (raised from 0.02 so the board converges to the fresh market blend
in ~a week after a shock instead of ~7 weeks). The clamp mechanism itself
is unchanged: a value 30% off target moves exactly 10% in one run.
"""
import os

import pytest

pytest.importorskip("numpy")  # both value modules import numpy at load
pytest.importorskip("pandas")  # value_model_training imports pandas at load
pytest.importorskip("flask")  # utils.utils (via value_model_training) needs flask
pytest.importorskip("bs4")  # utils.utils also imports BeautifulSoup

from data_building import value_model_training as vmt
from data_building.trade_intel import trade_value_model as tvm

NEW_DEFAULT = 0.10


# ── Layer 1: model board (value_model_training) ─────────────────────────────

def test_board_cap_default_is_10pct(monkeypatch):
    monkeypatch.delenv("VALUE_MOVE_CAP_OVERRIDE", raising=False)
    assert vmt._daily_move_cap() == pytest.approx(NEW_DEFAULT)


def test_board_cap_override_still_wins(monkeypatch):
    # The one-off reset path ("50" ≈ unclamped) must keep working.
    monkeypatch.setenv("VALUE_MOVE_CAP_OVERRIDE", "50")
    assert vmt._daily_move_cap() == pytest.approx(50.0)
    monkeypatch.setenv("VALUE_MOVE_CAP_OVERRIDE", "0.02")
    assert vmt._daily_move_cap() == pytest.approx(0.02)


def test_board_clamp_moves_exactly_10pct_up():
    # 30% above yesterday's value -> clamped to exactly +10%.
    assert vmt._clamp_board_value(130.0, 100.0, NEW_DEFAULT) == pytest.approx(110.0)


def test_board_clamp_moves_exactly_10pct_down():
    # 30% below yesterday's value -> clamped to exactly -10%.
    assert vmt._clamp_board_value(70.0, 100.0, NEW_DEFAULT) == pytest.approx(90.0)


def test_board_clamp_leaves_in_band_moves_alone():
    assert vmt._clamp_board_value(105.0, 100.0, NEW_DEFAULT) == pytest.approx(105.0)
    assert vmt._clamp_board_value(95.0, 100.0, NEW_DEFAULT) == pytest.approx(95.0)


# ── Layer 2: calibrated track (trade_value_model) ───────────────────────────

def test_calibrated_cap_default_is_10pct():
    # DAILY_MOVE_CAP binds at import from VALUE_MOVE_CAP_OVERRIDE; in any
    # environment without the override set it must be the new 10% default.
    if os.environ.get("VALUE_MOVE_CAP_OVERRIDE") is None:
        assert tvm.DAILY_MOVE_CAP == pytest.approx(NEW_DEFAULT)


def test_calibrated_clamp_moves_exactly_10pct_up(monkeypatch):
    monkeypatch.setattr(tvm, "DAILY_MOVE_CAP", NEW_DEFAULT)
    assert tvm._clamp_to_prev(130.0, 100.0) == pytest.approx(110.0)


def test_calibrated_clamp_moves_exactly_10pct_down(monkeypatch):
    monkeypatch.setattr(tvm, "DAILY_MOVE_CAP", NEW_DEFAULT)
    assert tvm._clamp_to_prev(60.0, 100.0) == pytest.approx(90.0)


def test_calibrated_clamp_leaves_in_band_moves_alone(monkeypatch):
    monkeypatch.setattr(tvm, "DAILY_MOVE_CAP", NEW_DEFAULT)
    assert tvm._clamp_to_prev(105.0, 100.0) == pytest.approx(105.0)
    assert tvm._clamp_to_prev(95.0, 100.0) == pytest.approx(95.0)


def test_calibrated_clamp_no_history_establishes_freely(monkeypatch):
    # First run / new player: no usable previous value, no clamp.
    monkeypatch.setattr(tvm, "DAILY_MOVE_CAP", NEW_DEFAULT)
    assert tvm._clamp_to_prev(500.0, None) == 500.0
    assert tvm._clamp_to_prev(500.0, 0.0) == 500.0
    assert tvm._clamp_to_prev(None, 100.0) is None
