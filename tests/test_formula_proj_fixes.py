"""Formula-audit fixes F1 + F4 (2026-10-21 audit, domain 02).

F1: projection variants must honor the league's scoring on every base —
TE premium layers onto half/std (not just PPR) and 6pt passing TD layers
onto std (not just PPR/half). The variant set is the full 3 x 2 x 2 cross.

F4: a missing ``rec`` (reception) rate defaults to 1.0 (PPR) in EVERY
scoring path — score_stats, normalize_league_scoring,
scoring_format_from_settings, pick_proj_variant. An explicit 0.0 in
settings is still respected as standard.
"""
import pytest

from data_building.fetch_projections import (
    PROJ_VARIANTS,
    _raw_to_pts,
    season_ppg_from_weekly,
)
from data_building.start_score_bundle import variant_for_exact_scoring
from utils.fantasy_scoring import projection_points, score_stats
from utils.league_scoring import normalize_league_scoring
from utils.proj_variant import pick_proj_variant
from utils.projections import _sleeper_stats_to_variants
from utils.trade import scoring_format_from_settings


WR_LINE = {"rec": 8, "rec_yd": 80, "rec_td": 1}
TE_LINE = {"rec": 5, "rec_yd": 50, "rec_td": 0}
QB_LINE = {"pass_yd": 300, "pass_td": 3, "pass_int": 1}


# ---------------------------------------------------------------------------
# F4 — one missing-rec default (PPR) everywhere
# ---------------------------------------------------------------------------

def test_score_stats_missing_rec_scores_ppr():
    # The audit's worked example: 8/80/1 with a partial settings dict that
    # omits rec must score 22.0 (PPR), not 14.0 (standard).
    assert score_stats(WR_LINE, {"pass_td": 6}) == pytest.approx(22.0)
    assert score_stats(WR_LINE, {}) == pytest.approx(22.0)


def test_score_stats_explicit_zero_rec_stays_standard():
    assert score_stats(WR_LINE, {"rec": 0.0}) == pytest.approx(14.0)
    assert score_stats(WR_LINE, {"rec": 0.5}) == pytest.approx(18.0)


def test_projection_points_recompute_missing_rec_scores_ppr():
    entry = {"raw_stats": dict(WR_LINE)}
    assert projection_points(entry, {"pass_td": 6}, "WR") == pytest.approx(22.0)
    assert projection_points(entry, {"rec": 0.0, "pass_td": 6}, "WR") == pytest.approx(14.0)


def test_normalize_missing_rec_defaults_ppr_and_keeps_explicit_zero():
    assert normalize_league_scoring("sleeper", {"pass_td": 4})["rec"] == 1.0
    assert normalize_league_scoring("sleeper", {})["rec"] == 1.0
    assert normalize_league_scoring("sleeper", {"rec": 0})["rec"] == 0.0
    assert normalize_league_scoring("espn", {"pointsPerReception": 0.5})["rec"] == 0.5


def test_format_and_variant_missing_rec_agree_on_ppr():
    assert scoring_format_from_settings({}) == "ppr"
    assert scoring_format_from_settings({"rec": 0.0}) == "std"
    assert pick_proj_variant({}) == "ppr"


# ---------------------------------------------------------------------------
# F1 — variants honor TEP / 6pt on every reception base
# ---------------------------------------------------------------------------

def test_variant_table_is_the_full_cross():
    assert set(PROJ_VARIANTS) == {
        "ppr", "half_ppr", "std",
        "tep", "tep_half", "tep_std",
        "6pt_ppr", "6pt_half", "6pt_std",
        "6pt_tep", "6pt_tep_half", "6pt_tep_std",
    }
    for variant in PROJ_VARIANTS:
        from data_building.start_score_bundle import VARIANT_SCORING
        assert pick_proj_variant(VARIANT_SCORING[variant]) == variant


def test_te_keeps_full_premium_on_half_ppr_variant_path():
    # TE at 5 rec/game, half-PPR + 0.5 TEP: the variant path must carry the
    # full +2.5 PPG premium over the plain half-PPR variant.
    variants = _sleeper_stats_to_variants(TE_LINE, "TE", None)
    assert variants["half_ppr"] == pytest.approx(7.5)
    assert variants["tep_half"] == pytest.approx(10.0)
    assert variants["tep_half"] - variants["half_ppr"] == pytest.approx(2.5)
    assert variants["tep_std"] == pytest.approx(7.5)

    entry = {k: v for k, v in variants.items() if k != "raw_stats"}
    settings = {"rec": 0.5, "bonus_rec_te": 0.5}
    assert pick_proj_variant(settings) == "tep_half"
    assert projection_points(entry, settings, "TE") == pytest.approx(10.0)


def test_qb_keeps_6pt_rate_on_std_variant_path():
    variants = _sleeper_stats_to_variants(QB_LINE, "QB", None)
    assert variants["std"] == pytest.approx(22.0)
    assert variants["6pt_std"] == pytest.approx(28.0)

    entry = {k: v for k, v in variants.items() if k != "raw_stats"}
    settings = {"rec": 0.0, "pass_td": 6}
    assert pick_proj_variant(settings) == "6pt_std"
    assert projection_points(entry, settings, "QB") == pytest.approx(28.0)


def test_raw_to_pts_new_variants_and_actual_tep():
    # Bare variant calls price the TEP layer at the canonical 0.5.
    assert _raw_to_pts(TE_LINE, "TE", "tep_half") == pytest.approx(10.0)
    assert _raw_to_pts(TE_LINE, "TE", "tep_std") == pytest.approx(7.5)
    assert _raw_to_pts(QB_LINE, "QB", "6pt_std") == pytest.approx(28.0)
    assert _raw_to_pts(QB_LINE, "QB", "std") == pytest.approx(22.0)
    # With the league's settings, the ACTUAL bonus_rec_te is used, not 0.5:
    # a 1.0-TEP league's TE gains 5 x 1.0 = 5.0 of premium, not 2.5.
    assert _raw_to_pts(TE_LINE, "TE", "tep_half",
                       scoring_settings={"bonus_rec_te": 1.0}) == pytest.approx(12.5)
    assert _raw_to_pts(TE_LINE, "TE", "tep_half",
                       tep_bonus=1.0) == pytest.approx(12.5)
    # Non-TEs never get the premium.
    assert _raw_to_pts(TE_LINE, "WR", "tep_half",
                       scoring_settings={"bonus_rec_te": 1.0}) == pytest.approx(7.5)


def test_season_ppg_variants_carry_new_keys():
    week = {"p1": {"half_ppr": 7.5, "tep_half": 10.0, "6pt_std": 28.0}}
    out = season_ppg_from_weekly([week])
    assert out["p1"]["tep_half"] == pytest.approx(10.0)
    assert out["p1"]["6pt_std"] == pytest.approx(28.0)


def test_exact_scoring_gate_covers_new_variants():
    assert variant_for_exact_scoring({"rec": 0.5, "bonus_rec_te": 0.5}) == "tep_half"
    assert variant_for_exact_scoring({"rec": 0.0, "bonus_rec_te": 0.5}) == "tep_std"
    assert variant_for_exact_scoring({"rec": 0.0, "pass_td": 6}) == "6pt_std"
    assert variant_for_exact_scoring(
        {"rec": 0.5, "pass_td": 6, "bonus_rec_te": 0.5}) == "6pt_tep_half"
    assert variant_for_exact_scoring(
        {"rec": 0.0, "pass_td": 6, "bonus_rec_te": 0.5}) == "6pt_tep_std"
    # A non-canonical TE premium is still exotic: no silent approximation.
    assert variant_for_exact_scoring({"rec": 0.5, "bonus_rec_te": 1.0}) is None
