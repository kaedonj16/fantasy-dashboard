"""Waiver-wire injury flags: a hurt player must never read as a clean add.

Covers the shared helpers in utils.waiver_score (no app import needed).
"""
import pytest

pytest.importorskip("pandas")  # repo convention for the test suite

from utils.waiver_score import (
    drop_seriously_hurt,
    is_seriously_hurt,
    pick_waiver_push_candidate,
    self_injury_multiplier,
    waiver_injury_note,
    waiver_signal,
)


def test_is_seriously_hurt_covers_all_designations():
    for s in ["IR", "PUP", "NFI", "SUSP", "SUS", "OUT", "DOUBTFUL", "NA",
              "ir", "Out", "doubtful", "sus"]:
        assert is_seriously_hurt(s), s
    for s in ["", None, "Questionable", "Q", "Active", "Healthy"]:
        assert not is_seriously_hurt(s), s


def test_self_injury_multiplier_zeroes_serious():
    assert self_injury_multiplier("OUT") == 0.0
    assert self_injury_multiplier("IR") == 0.0
    assert self_injury_multiplier("SUS") == 0.0
    assert self_injury_multiplier("NA") == 0.0
    assert self_injury_multiplier("DOUBTFUL") == 0.35
    assert self_injury_multiplier("QUESTIONABLE") == 0.85
    assert self_injury_multiplier("") == 1.0
    assert self_injury_multiplier(None) == 1.0


def test_waiver_injury_note_formats():
    note = waiver_injury_note("Out", "Knee", 3)
    assert "OUT" in note and "Knee" in note and "3 wks" in note
    assert waiver_injury_note("Questionable", "Ankle", None) == "Game-time call (Ankle)"
    assert waiver_injury_note("Questionable") == "Game-time call"
    assert waiver_injury_note("IR", None, None) == "IR"
    assert waiver_injury_note("", None, None) is None
    assert waiver_injury_note(None) is None


def _vtbl(*rows):
    return [
        {"id": pid, "name": name, "position": pos, "team": "KC", "value": val}
        for pid, name, pos, val in rows
    ]


def test_push_candidate_skips_seriously_hurt():
    tbl = _vtbl(
        ("1", "Hurt Star", "WR", 9000),
        ("2", "Healthy Guy", "WR", 8000),
    )
    players = {
        "1": {"injury_status": "Out", "injury_body_part": "Knee"},
        "2": {"injury_status": "", "injury_body_part": ""},
    }
    top = pick_waiver_push_candidate(tbl, set(), players=players)
    assert top["player_id"] == "2"


def test_push_candidate_none_when_all_hurt():
    tbl = _vtbl(("1", "Hurt Star", "WR", 9000))
    players = {"1": {"injury_status": "IR"}}
    assert pick_waiver_push_candidate(tbl, set(), players=players) is None


def test_push_candidate_questionable_still_eligible():
    tbl = _vtbl(("1", "GameTime", "WR", 9000))
    players = {"1": {"injury_status": "Questionable"}}
    top = pick_waiver_push_candidate(tbl, set(), players=players)
    assert top["player_id"] == "1"


def test_push_candidate_no_feed_keeps_old_behavior():
    tbl = _vtbl(("1", "Anyone", "WR", 9000))
    top = pick_waiver_push_candidate(tbl, set())
    assert top["player_id"] == "1"


def test_drop_seriously_hurt_redraft_vs_dynasty():
    cands = [
        {"player_id": "1", "self_status": "Out"},
        {"player_id": "2", "self_status": "Questionable"},
        {"player_id": "3", "self_status": ""},
        {"player_id": "4", "self_status": "IR"},
    ]
    redraft = drop_seriously_hurt(cands, dynasty=False)
    assert [c["player_id"] for c in redraft] == ["2", "3"]
    dynasty = drop_seriously_hurt(cands, dynasty=True)
    assert [c["player_id"] for c in dynasty] == ["1", "2", "3", "4"]


def test_waiver_signal_labels_hurt_candidate_injured():
    cls, label = waiver_signal(
        {"player_id": "1", "self_status": "OUT", "position": "WR",
         "value": 5000, "age": 25, "rank_change_7d": 50},
        {},
    )
    assert label == "Injured"
    assert cls == "signal-aging"
