"""Unit tests for the TE-premium value helpers (pure, no DB/pandas deps).

These encode the logic behind several bugs fixed in the wild: the player modal
not applying the premium, and a ``Decimal * float`` 500 when scaling DB values.
"""
from decimal import Decimal

import pytest

from utils.value_helpers import apply_te_premium, te_premium_from_settings


# ── te_premium_from_settings ────────────────────────────────────────────────
def test_tier_none():
    assert te_premium_from_settings({"bonus_rec_te": 0}) == 0.0
    assert te_premium_from_settings({}) == 0.0


def test_tier_half():
    assert te_premium_from_settings({"bonus_rec_te": 0.5}) == 0.5


def test_tier_full():
    assert te_premium_from_settings({"bonus_rec_te": 1.0}) == 1.0


def test_tier_boundaries():
    # Snap points: >=0.25 -> half, >=0.75 -> full.
    assert te_premium_from_settings({"bonus_rec_te": 0.24}) == 0.0
    assert te_premium_from_settings({"bonus_rec_te": 0.25}) == 0.5
    assert te_premium_from_settings({"bonus_rec_te": 0.74}) == 0.5
    assert te_premium_from_settings({"bonus_rec_te": 0.75}) == 1.0


def test_malformed_settings_are_safe():
    assert te_premium_from_settings(None) == 0.0
    assert te_premium_from_settings({"bonus_rec_te": "not a number"}) == 0.0
    assert te_premium_from_settings({"bonus_rec_te": None}) == 0.0
    assert te_premium_from_settings("nonsense") == 0.0  # .get on a str -> handled


# ── apply_te_premium ────────────────────────────────────────────────────────
def test_full_premium_scales_te_by_20pct():
    assert apply_te_premium(100.0, "TE", 1.0) == pytest.approx(120.0)


def test_half_premium_scales_te_by_10pct():
    assert apply_te_premium(100.0, "TE", 0.5) == pytest.approx(110.0)


def test_no_premium_is_passthrough():
    assert apply_te_premium(100.0, "TE", 0.0) == 100.0


def test_non_te_never_scaled():
    for pos in ("QB", "RB", "WR", "K", "DEF"):
        assert apply_te_premium(100.0, pos, 1.0) == 100.0


def test_position_case_insensitive():
    assert apply_te_premium(100.0, "te", 1.0) == 120.0


def test_decimal_value_does_not_raise():
    # DB numerics arrive as Decimal; Decimal * float used to 500 the modal.
    out = apply_te_premium(Decimal("100"), "TE", 1.0)
    assert out == pytest.approx(120.0)
    assert isinstance(out, float)


def test_none_and_malformed_value_safe():
    assert apply_te_premium(None, "TE", 1.0) == 0.0
    assert apply_te_premium("nope", "TE", 1.0) == 0.0
