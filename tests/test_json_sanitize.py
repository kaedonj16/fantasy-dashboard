"""Tests for utils.json_sanitize."""
import json
import math

from utils.json_sanitize import sanitize_for_json


def test_nan_becomes_none():
    assert sanitize_for_json(float("nan")) is None


def test_infinities_become_none():
    assert sanitize_for_json(float("inf")) is None
    assert sanitize_for_json(float("-inf")) is None


def test_normal_values_untouched():
    payload = {"a": 1, "b": 2.5, "c": "x", "d": None, "e": True}
    assert sanitize_for_json(payload) == payload


def test_recurses_nested_structures():
    payload = {"rows": [{"v": float("nan")}, {"v": 3.0}], "meta": {"m": float("inf")}}
    out = sanitize_for_json(payload)
    assert out == {"rows": [{"v": None}, {"v": 3.0}], "meta": {"m": None}}


def test_result_is_strict_json_serializable():
    payload = {"a": float("nan"), "b": [float("inf"), 1.0]}
    text = json.dumps(sanitize_for_json(payload), allow_nan=False)
    assert json.loads(text) == {"a": None, "b": [None, 1.0]}


def test_integers_never_touched():
    # ints can't be NaN; make sure they don't fall into the float branch
    assert sanitize_for_json({"n": 10**18}) == {"n": 10**18}


def test_zero_and_negative_zero_survive():
    assert sanitize_for_json(0.0) == 0.0
    assert not math.isnan(sanitize_for_json(-0.0))
