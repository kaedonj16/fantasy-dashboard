"""Pure unit tests for the shared scoring-rate contract.

Yahoo, ESPN, and Fleaflicker all publish a per-yard rate and a 300-yard extra
under the same id. ``assign_scoring_rate`` keeps the per-unit fraction.
``stamp_scoring_aliases`` keeps leftover ESPN PPR aliases in lockstep with
canonical ``rec`` so a standard league cannot keep ``pointsPerReception=1``.
"""
from utils.league_scoring import (
    assign_scoring_rate,
    normalize_league_scoring,
    stamp_scoring_aliases,
)


def test_assign_scoring_rate_keeps_per_yard_fraction_over_milestone():
    out = {}
    assign_scoring_rate(out, "pass_yd", 0.04)
    assign_scoring_rate(out, "pass_yd", 3)
    assert out["pass_yd"] == 0.04


def test_assign_scoring_rate_replaces_milestone_when_fraction_arrives_second():
    out = {}
    assign_scoring_rate(out, "rec_yd", 3)
    assign_scoring_rate(out, "rec_yd", 0.1)
    assert out["rec_yd"] == 0.1


def test_assign_scoring_rate_does_not_let_later_ppr_overwrite_explicit_zero():
    out = {}
    assign_scoring_rate(out, "rec", 0)
    assign_scoring_rate(out, "rec", 1)
    assert out["rec"] == 0.0


def test_assign_scoring_rate_keeps_first_ppr_value():
    out = {}
    assign_scoring_rate(out, "rec", 1)
    assign_scoring_rate(out, "rec", 0)
    assert out["rec"] == 1.0


def test_stamp_aliases_explicit_zero_rec_overwrites_leftover_ppr():
    stamped = stamp_scoring_aliases({"rec": 0.0, "pointsPerReception": 1})
    assert stamped["rec"] == 0.0
    assert stamped["pointsPerReception"] == 0.0


def test_normalize_stamps_zero_rec_over_espn_ppr_alias():
    out = normalize_league_scoring("espn", {"rec": 0, "pointsPerReception": 1})
    assert out["rec"] == 0.0
    assert out["pointsPerReception"] == 0.0


def test_normalize_preserves_yahoo_ppr():
    out = normalize_league_scoring("yahoo", {"rec": 1.0, "pass_yd": 0.04})
    assert out["rec"] == 1.0
    assert out["pointsPerReception"] == 1.0
    assert out["passYards"] == 0.04
