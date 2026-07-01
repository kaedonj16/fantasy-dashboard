"""Unit tests for projection-variant selection (pure, no heavy deps).

pick_proj_variant maps a league's raw Sleeper scoring to the projection set it
should use. TE premium and 6pt-passing-TD only layer onto the full-PPR base.
"""
from utils.proj_variant import pick_proj_variant


def test_defaults_to_ppr():
    # Empty settings -> rec 1.0, pass_td 4.0, no TE bonus.
    assert pick_proj_variant({}) == "ppr"
    assert pick_proj_variant(None) == "ppr"


def test_reception_base_tiers():
    assert pick_proj_variant({"rec": 1.0}) == "ppr"
    assert pick_proj_variant({"rec": 0.5}) == "half_ppr"
    assert pick_proj_variant({"rec": 0.4}) == "half_ppr"   # >=0.4 is half
    assert pick_proj_variant({"rec": 0.39}) == "std"
    assert pick_proj_variant({"rec": 0.0}) == "std"


def test_te_premium_layers_onto_ppr():
    assert pick_proj_variant({"rec": 1.0, "bonus_rec_te": 0.5}) == "tep"
    # threshold is >= 0.25
    assert pick_proj_variant({"rec": 1.0, "bonus_rec_te": 0.25}) == "tep"
    assert pick_proj_variant({"rec": 1.0, "bonus_rec_te": 0.24}) == "ppr"


def test_te_premium_ignored_off_full_ppr():
    # TEP only combines with the ppr base — half/std leagues ignore it.
    assert pick_proj_variant({"rec": 0.5, "bonus_rec_te": 1.0}) == "half_ppr"
    assert pick_proj_variant({"rec": 0.0, "bonus_rec_te": 1.0}) == "std"


def test_six_point_passing_td():
    assert pick_proj_variant({"rec": 1.0, "pass_td": 6}) == "6pt_ppr"
    assert pick_proj_variant({"rec": 0.5, "pass_td": 6}) == "6pt_half"
    # threshold is >= 5.5
    assert pick_proj_variant({"rec": 1.0, "pass_td": 5.5}) == "6pt_ppr"
    assert pick_proj_variant({"rec": 1.0, "pass_td": 5.4}) == "ppr"


def test_six_point_ignored_off_ppr_half():
    # six only layers onto ppr/half; a std 6pt league is just "std".
    assert pick_proj_variant({"rec": 0.0, "pass_td": 6}) == "std"


def test_six_point_te_premium_combo():
    assert pick_proj_variant({"rec": 1.0, "pass_td": 6, "bonus_rec_te": 1.0}) == "6pt_tep"
    # 6pt + TEP on a half-PPR base -> no 6pt_tep variant, falls to 6pt_half
    assert pick_proj_variant({"rec": 0.5, "pass_td": 6, "bonus_rec_te": 1.0}) == "6pt_half"
