"""Unit tests for utils.consistency (pure boom-bust profiles)."""
from utils.consistency import consistency_profile


def test_none_for_no_games():
    assert consistency_profile([], "WR") is None
    assert consistency_profile([None, None], "WR") is None


def test_steady_player_scores_high_consistency():
    # A metronome WR: same score every week -> zero variance -> 100 consistency.
    prof = consistency_profile([14, 14, 14, 14, 14], "WR")
    assert prof["games"] == 5
    assert prof["mean"] == 14.0
    assert prof["cv"] == 0.0
    assert prof["consistency"] == 100
    assert prof["floor"] == 14.0 and prof["ceiling"] == 14.0
    assert prof["label"] == "Steady"


def test_boom_bust_player_flagged():
    # Alternating dud/smash weeks: high boom AND bust rate -> "Boom or bust".
    prof = consistency_profile([2, 28, 3, 30, 1, 26], "WR")
    assert prof["boom_rate"] >= 0.30
    assert prof["bust_rate"] >= 0.30
    assert prof["label"] == "Boom or bust"
    assert prof["consistency"] < 62
    assert prof["ceiling"] > prof["floor"]


def test_small_sample_flagged():
    prof = consistency_profile([20, 5], "RB")
    assert prof["small_sample"] is True
    assert prof["label"] == "Small sample"


def test_floor_ceiling_are_percentiles_not_extremes():
    # Floor (p20) / ceiling (p80) sit inside the min/max so one outlier week
    # doesn't define them.
    prof = consistency_profile([10, 12, 13, 14, 40], "RB")  # 40 is an outlier smash
    assert prof["ceiling"] < 40
    assert prof["floor"] > 10


def test_position_thresholds_differ():
    # 16 points is a boom for a TE (>=15) but not for an RB (>=20).
    scores = [16, 16, 16, 16]
    te = consistency_profile(scores, "TE")
    rb = consistency_profile(scores, "RB")
    assert te["boom_rate"] == 1.0
    assert rb["boom_rate"] == 0.0


def test_zero_weeks_count_as_bad_games():
    # Active-but-blanked weeks are real; they drag the floor and bust rate.
    prof = consistency_profile([0, 0, 22, 24], "WR")
    assert prof["bust_rate"] == 0.5
    assert prof["floor"] is not None and prof["floor"] < 5


def test_unknown_position_uses_default_thresholds():
    prof = consistency_profile([19, 19, 19], "FLEX")
    # default boom threshold is 18, so 19s all boom
    assert prof["boom_rate"] == 1.0
