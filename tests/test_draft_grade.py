"""Unit tests for utils.draft_grade.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
import pytest

from utils.draft_grade import (
    clamp01,
    dr_apply_field_curve,
    dr_avg_top_n,
    dr_grade_letter,
    dr_letter_to_score,
    dr_lineup_score,
    dr_optimal_lineup,
    dr_slot_eligible,
    dr_team_grade_score,
)


# ---- clamp01 --------------------------------------------------------------

def test_clamp01_bounds():
    assert clamp01(-0.5) == 0.0
    assert clamp01(1.5) == 1.0
    assert clamp01(0.3) == 0.3


# ---- dr_grade_letter ------------------------------------------------------

@pytest.mark.parametrize("score,letter", [
    (95, "A+"), (86, "A"), (80, "A-"), (75, "B+"), (70, "B"),
    (65, "B-"), (60, "C+"), (55, "C"), (50, "C-"), (42, "D"), (10, "F"),
])
def test_grade_letter_bands(score, letter):
    assert dr_grade_letter(score) == letter


# ---- dr_letter_to_score ---------------------------------------------------

def test_letter_to_score_known_and_default():
    assert dr_letter_to_score("A+") == 92
    assert dr_letter_to_score("F") == 20
    assert dr_letter_to_score("???") == 55


# ---- dr_slot_eligible -----------------------------------------------------

def test_slot_eligibility():
    assert dr_slot_eligible("FLEX", "rb") is True
    assert dr_slot_eligible("FLEX", "QB") is False
    assert dr_slot_eligible("SF", "QB") is True
    assert dr_slot_eligible("QB", "QB") is True
    assert dr_slot_eligible("QB", "RB") is False


# ---- dr_lineup_score ------------------------------------------------------

def test_lineup_score_prefers_ppg():
    assert dr_lineup_score({"ppg": 18.5, "val": 9000}) == 18.5


def test_lineup_score_falls_back_to_scaled_value():
    assert dr_lineup_score({"val": 5000}) == 5.0
    assert dr_lineup_score({}) == 0.0


# ---- dr_optimal_lineup ----------------------------------------------------

def test_optimal_lineup_fills_restrictive_slots_first():
    players = [
        {"id": "qb1", "pos": "QB", "ppg": 20},
        {"id": "rb1", "pos": "RB", "ppg": 15},
        {"id": "rb2", "pos": "RB", "ppg": 12},
        {"id": "wr1", "pos": "WR", "ppg": 14},
    ]
    slots = ["QB", "RB", "FLEX"]
    starters = dr_optimal_lineup(players, slots)
    assert "qb1" in starters       # QB slot
    assert "rb1" in starters       # RB slot -> best RB
    # FLEX -> best remaining RB/WR/TE (wr1 14 > rb2 12)
    assert "wr1" in starters
    assert len(starters) == 3


# ---- dr_avg_top_n ---------------------------------------------------------

def test_avg_top_n():
    assert dr_avg_top_n([1, 5, 3, 9], 2) == 7.0   # (9+5)/2
    assert dr_avg_top_n([], 3) == 0.0
    assert dr_avg_top_n([4, 2], 0) == 0.0


# ---- dr_apply_field_curve -------------------------------------------------

def test_field_curve_passthrough_under_three():
    assert dr_apply_field_curve([50, 60]) == [50, 60]


def test_field_curve_centers_on_anchor():
    # Zero spread would center every team on the anchor (68), but the raw cap
    # keeps a mediocre field from being inflated: a 60-composite team tops out
    # at raw + 8 = 68 no matter how it compares to the field.
    assert dr_apply_field_curve([60, 60, 60]) == [68, 68, 68]
    # A strong tied field lands on the anchor (a B-), not capped down.
    assert dr_apply_field_curve([90, 90, 90]) == [68, 68, 68]


def test_field_curve_orders_preserved_and_bounded():
    out = dr_apply_field_curve([10, 50, 90])
    assert out[0] < out[1] < out[2]
    assert all(0.0 <= v <= 100.0 for v in out)


def test_field_curve_compressed_a_needs_more_than_one_sd():
    # Recalibrated (anchor 68, PTS 9): the average draft is a B- and an A is
    # reserved for clearly-above-average drafts. A +1 SD team lands in B+, so an
    # A requires well over one SD of real separation.
    curved = dr_apply_field_curve([66, 74, 82])   # top team is exactly +1 SD
    assert curved == [59, 68, 77]
    assert dr_grade_letter(curved[2]) == "B+"     # +1 SD -> B+, not A
    assert dr_grade_letter(curved[1]) == "B-"     # field average is a B-


# ---- dr_team_grade_score --------------------------------------------------

def test_team_grade_none_for_empty_picks():
    assert dr_team_grade_score(
        [], slots=["QB"], targets={}, num_teams=12,
        draft_type="startup", league_ppg_list=[], league_val_list=[],
    ) is None


def test_team_grade_returns_bounded_number():
    picks = [
        {"id": "1", "pos": "QB", "ps": 80, "pn": 1, "val": 6000, "ppg": 20},
        {"id": "2", "pos": "RB", "ps": 70, "pn": 13, "val": 5000, "ppg": 15},
        {"id": "3", "pos": "WR", "ps": 65, "pn": 25, "val": 4500, "ppg": 13},
    ]
    score = dr_team_grade_score(
        picks, slots=["QB", "RB", "WR"], targets={"QB": 1, "RB": 1, "WR": 1},
        num_teams=12, draft_type="startup",
        league_ppg_list=[15, 14, 13, 12], league_val_list=[5000, 4500, 4000, 3500],
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0
