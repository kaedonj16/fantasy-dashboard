"""Unit tests for projection-variant selection (pure, no heavy deps).

pick_proj_variant maps a league's raw Sleeper scoring to the projection set it
should use. TE premium and 6pt-passing-TD layer onto EVERY reception base
(ppr / half_ppr / std) — the variant set is the full 3 x 2 x 2 cross.
"""
from utils.proj_variant import pick_proj_variant, pick_proj_variant_from_draft_scoring


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


def test_te_premium_layers_onto_half_and_std():
    # TEP combines with every base — half/std TEP leagues keep the premium.
    assert pick_proj_variant({"rec": 0.5, "bonus_rec_te": 1.0}) == "tep_half"
    assert pick_proj_variant({"rec": 0.5, "bonus_rec_te": 0.5}) == "tep_half"
    assert pick_proj_variant({"rec": 0.0, "bonus_rec_te": 1.0}) == "tep_std"
    assert pick_proj_variant({"rec": 0.0, "bonus_rec_te": 0.25}) == "tep_std"


def test_six_point_passing_td():
    assert pick_proj_variant({"rec": 1.0, "pass_td": 6}) == "6pt_ppr"
    assert pick_proj_variant({"rec": 0.5, "pass_td": 6}) == "6pt_half"
    # threshold is >= 5.5
    assert pick_proj_variant({"rec": 1.0, "pass_td": 5.5}) == "6pt_ppr"
    assert pick_proj_variant({"rec": 1.0, "pass_td": 5.4}) == "ppr"


def test_six_point_layers_onto_std():
    # six layers onto std too; a std 6pt league keeps the 6pt passing-TD rate.
    assert pick_proj_variant({"rec": 0.0, "pass_td": 6}) == "6pt_std"


def test_six_point_te_premium_combo():
    assert pick_proj_variant({"rec": 1.0, "pass_td": 6, "bonus_rec_te": 1.0}) == "6pt_tep"
    # 6pt + TEP on half/std bases has its own variants now.
    assert pick_proj_variant({"rec": 0.5, "pass_td": 6, "bonus_rec_te": 1.0}) == "6pt_tep_half"
    assert pick_proj_variant({"rec": 0.0, "pass_td": 6, "bonus_rec_te": 0.5}) == "6pt_tep_std"


def test_missing_rec_defaults_to_ppr_everywhere():
    # Missing rec = PPR (1.0), the unified default; explicit 0.0 stays std.
    assert pick_proj_variant({"pass_td": 6}) == "6pt_ppr"
    assert pick_proj_variant({"bonus_rec_te": 0.5}) == "tep"
    assert pick_proj_variant({"rec": None, "pointsPerReception": 0.5}) == "half_ppr"
    assert pick_proj_variant({"rec": 0.0, "pass_td": 4}) == "std"


def test_draft_scoring_maps_ppr_tep_and_pass_td():
    assert pick_proj_variant_from_draft_scoring(None) == "ppr"
    assert pick_proj_variant_from_draft_scoring({"ppr": 1, "tep": 0, "passTd": 4}) == "ppr"
    assert pick_proj_variant_from_draft_scoring({"ppr": 0.5, "tep": 0, "passTd": 4}) == "half_ppr"
    assert pick_proj_variant_from_draft_scoring({"ppr": 0, "tep": 0, "passTd": 4}) == "std"
    assert pick_proj_variant_from_draft_scoring({"ppr": 1, "tep": 0.5, "passTd": 4}) == "tep"
    assert pick_proj_variant_from_draft_scoring({"ppr": 1, "tep": 0, "passTd": 6}) == "6pt_ppr"
    assert pick_proj_variant_from_draft_scoring({"ppr": 0.5, "tep": 0, "passTd": 6}) == "6pt_half"
    assert pick_proj_variant_from_draft_scoring({"ppr": 1, "tep": 1, "passTd": 6}) == "6pt_tep"
    assert pick_proj_variant_from_draft_scoring({"ppr": 0.5, "tep": 0.5, "passTd": 4}) == "tep_half"
    assert pick_proj_variant_from_draft_scoring({"ppr": 0, "tep": 0.5, "passTd": 4}) == "tep_std"
    assert pick_proj_variant_from_draft_scoring({"ppr": 0, "tep": 0, "passTd": 6}) == "6pt_std"
    assert pick_proj_variant_from_draft_scoring({"ppr": 0.5, "tep": 1, "passTd": 6}) == "6pt_tep_half"
    assert pick_proj_variant_from_draft_scoring({"ppr": 0, "tep": 1, "passTd": 6}) == "6pt_tep_std"
