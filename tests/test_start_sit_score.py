"""Unit tests for the unified start/sit formula (no Flask)."""
from utils.start_sit_score import compute_start_score


def test_bye_zeros_score():
    score, factors, demotion = compute_start_score(20.0, on_bye=True)
    assert score == 0.0
    assert demotion == "bye"
    assert factors["proj"] == 20.0
    assert factors["weather"] == 1.0


def test_out_zeros_score():
    score, factors, demotion = compute_start_score(20.0, injury_status="OUT")
    assert score == 0.0
    assert factors["avail"] == 0.0
    assert demotion == "out"


def test_form_is_capped():
    score, factors, _ = compute_start_score(10.0, recent_ppg=20.0, season_ppg=10.0)
    assert factors["form"] == 1.08
    assert abs(score - 10.8) < 1e-9


def test_matchup_does_not_double_count_by_default():
    """Weekly projections already include the opponent — do not re-apply it."""
    s_easy, easy, _ = compute_start_score(10.0, def_rank=1, def_total=32)
    s_hard, hard, _ = compute_start_score(10.0, def_rank=32, def_total=32)
    assert s_easy == s_hard == 10.0
    assert easy["matchup"] == hard["matchup"] == 1.0


def test_matchup_optional_residual_when_explicitly_enabled():
    s_easy, easy, _ = compute_start_score(
        10.0, def_rank=1, def_total=32, apply_matchup=True,
    )
    s_hard, hard, _ = compute_start_score(
        10.0, def_rank=32, def_total=32, apply_matchup=True,
    )
    assert easy["matchup"] == 1.03
    assert hard["matchup"] == 0.97
    assert s_easy > s_hard


def test_questionable_availability():
    score, factors, demotion = compute_start_score(10.0, injury_status="Q")
    assert factors["avail"] == 0.85
    assert demotion == "questionable"
    assert abs(score - 8.5) < 1e-9


def test_low_implied_total_default_position():
    score, factors, demotion = compute_start_score(10.0, implied_total=16)
    assert factors["vegas"] == 0.94
    assert demotion == "low_total"
    assert abs(score - 9.4) < 1e-9


def test_vegas_hurts_pass_catchers_more_in_low_totals():
    _, wr, _ = compute_start_score(10.0, implied_total=15, position="WR")
    _, rb, _ = compute_start_score(10.0, implied_total=15, position="RB")
    assert wr["vegas"] < rb["vegas"]
    assert wr["vegas"] == 0.92
    assert rb["vegas"] == 0.96


def test_vegas_helps_pass_catchers_more_in_shootouts():
    _, qb, _ = compute_start_score(10.0, implied_total=28, position="QB")
    _, rb, _ = compute_start_score(10.0, implied_total=28, position="RB")
    assert qb["vegas"] > rb["vegas"]
    assert qb["vegas"] == 1.05
    assert rb["vegas"] == 1.02


def test_bust_rate_floor_is_capped():
    _, high_bust, _ = compute_start_score(10.0, bust_rate=1.0)
    _, low_bust, _ = compute_start_score(10.0, bust_rate=0.0)
    assert high_bust["floor"] == 0.90
    assert low_bust["floor"] == 1.10


def test_wind_hurts_qb_and_kicker_more_than_rb():
    s_qb, qb, dem = compute_start_score(10.0, weather_kind="wind", position="QB")
    s_k, k, _ = compute_start_score(10.0, weather_kind="wind", position="K")
    s_rb, rb, _ = compute_start_score(10.0, weather_kind="wind", position="RB")
    assert dem == "weather"
    assert qb["weather"] == 0.92
    assert k["weather"] == 0.90
    assert rb["weather"] == 0.99
    assert s_qb < s_rb
    assert s_k < s_qb


def test_weather_and_vegas_stack():
    score, factors, demotion = compute_start_score(
        10.0,
        implied_total=15,
        weather_kind="precip",
        position="WR",
    )
    assert factors["vegas"] == 0.92
    assert factors["weather"] == 0.95
    assert abs(score - 10.0 * 0.92 * 0.95) < 1e-9
    assert demotion == "low_total"  # injury-style demotions keep priority
