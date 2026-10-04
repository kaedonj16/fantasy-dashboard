"""Tests for _short_td_desc in utils/push_notifications.py.

Verifies the TD push body extractor handles pass TDs, rush/scramble/run
TDs, and never leaks a text fragment like "(Shotgun) T" (regression from
the Trevor Lawrence scramble TD on 2026-10-04).
"""
import importlib.util

import pytest


def _load_fn():
    src = open("utils/push_notifications.py").read()
    start = src.index("def _short_td_desc(play):")
    end = src.index("def _scorezone_roster_owner")
    ns = {}
    exec(src[start:end], {"__name__": "td_desc_test"}, ns)
    return ns["_short_td_desc"]


@pytest.fixture(scope="module")
def short_td_desc():
    return _load_fn()


def test_pass_td_with_yards(short_td_desc):
    assert (
        short_td_desc(
            {"play_text": "(Shotgun) T.Lawrence pass short left to B.Thomas for 12 yards, TOUCHDOWN."}
        )
        == "12-yd pass to B.Thomas"
    )


def test_pass_td_drops_extra_point_detail(short_td_desc):
    assert (
        short_td_desc(
            {
                "play_text": "J.Brissett pass short left to M.Harrison for 12 yards, "
                "TOUCHDOWN. C.Ryland extra point is GOOD, Center-C.Kreiter."
            }
        )
        == "12-yd pass to M.Harrison"
    )


def test_scramble_td(short_td_desc):
    # Regression: Lawrence's scramble TD produced "(Shotgun) T".
    assert (
        short_td_desc(
            {"play_text": "(Shotgun) T.Lawrence scrambles up the middle for 4 yards, TOUCHDOWN."}
        )
        == "4-yd rush TD"
    )


def test_scrambles_plural_td(short_td_desc):
    assert (
        short_td_desc(
            {"play_text": "(Shotgun) T.Lawrence scrambles right end for 12 yards, TOUCHDOWN."}
        )
        == "12-yd rush TD"
    )


def test_runs_td(short_td_desc):
    assert (
        short_td_desc({"play_text": "T.Lawrence runs up the middle for 5 yards, TOUCHDOWN."})
        == "5-yd rush TD"
    )


def test_ran_td(short_td_desc):
    assert (
        short_td_desc({"play_text": "T.Lawrence ran for 3 yards, TOUCHDOWN."}) == "3-yd rush TD"
    )


def test_rushes_td(short_td_desc):
    assert (
        short_td_desc({"play_text": "D.Henry rushes up the middle for 8 yards, TOUCHDOWN."})
        == "8-yd rush TD"
    )


def test_rush_td_no_yards(short_td_desc):
    assert (
        short_td_desc({"play_text": "D.Henry rushes for a TOUCHDOWN."}) == "Rush TD"
    )


def test_unparseable_returns_empty_never_fragment(short_td_desc):
    # Full-name receiver (no "X." initial) can't be parsed: return "" so the
    # caller falls back to "Touchdown!", never a fragment like "(Shotgun) T".
    got = short_td_desc(
        {"play_text": "(Shotgun) T.Lawrence pass to Brian Thomas for a TOUCHDOWN."}
    )
    assert got == ""
    assert not got.startswith("(")


def test_empty_play_text(short_td_desc):
    assert short_td_desc({"play_text": ""}) == ""
    assert short_td_desc({}) == ""


def test_no_fragment_leak_property(short_td_desc):
    # Fuzz-ish guard: across representative inputs, output never starts with
    # a parenthetical fragment.
    samples = [
        "(Shotgun) T.Lawrence scrambles for 4 yards, TOUCHDOWN.",
        "(No Huddle, Shotgun) J.Allen pass deep right to K.Shakir for 30 yards, TOUCHDOWN.",
        "J.Cook rush left tackle for 2 yards, TOUCHDOWN.",
        "(Shotgun) T.Lawrence sacked, FUMBLES, TOUCHDOWN.",
    ]
    for s in samples:
        got = short_td_desc({"play_text": s})
        assert not got.startswith("("), f"fragment leaked for {s!r}: {got!r}"
