"""Unit tests for the unified start/sit formula (no Flask)."""
from utils.start_sit_score import compute_start_score


def test_bye_zeros_score():
    score, factors, demotion = compute_start_score(20.0, on_bye=True)
    assert score == 0.0
    assert demotion == "bye"
    assert factors["proj"] == 20.0


def test_out_zeros_score():
    score, factors, demotion = compute_start_score(20.0, injury_status="OUT")
    assert score == 0.0
    assert factors["avail"] == 0.0
    assert demotion == "out"


def test_form_is_capped_at_1_10():
    score, factors, _ = compute_start_score(10.0, recent_ppg=20.0, season_ppg=10.0)
    assert factors["form"] == 1.10
    assert abs(score - 11.0) < 1e-9


def test_matchup_rank_1_is_easier_than_32():
    _, easy, _ = compute_start_score(10.0, def_rank=1, def_total=32)
    _, hard, _ = compute_start_score(10.0, def_rank=32, def_total=32)
    assert easy["matchup"] == 1.10
    assert hard["matchup"] == 0.90
    assert easy["matchup"] > hard["matchup"]


def test_questionable_availability():
    score, factors, demotion = compute_start_score(10.0, injury_status="Q")
    assert factors["avail"] == 0.85
    assert demotion == "questionable"
    assert abs(score - 8.5) < 1e-9


def test_low_implied_total():
    score, factors, demotion = compute_start_score(10.0, implied_total=16)
    assert factors["vegas"] == 0.94
    assert demotion == "low_total"
    assert abs(score - 9.4) < 1e-9


def test_bust_rate_floor_is_capped():
    _, high_bust, _ = compute_start_score(10.0, bust_rate=1.0)
    _, low_bust, _ = compute_start_score(10.0, bust_rate=0.0)
    assert high_bust["floor"] == 0.90
    assert low_bust["floor"] == 1.10
