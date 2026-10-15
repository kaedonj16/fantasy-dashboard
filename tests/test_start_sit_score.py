"""Unit tests for the unified start/sit formula (no Flask)."""
from utils.start_sit_score import bottom_teams_by_implied_total, compute_start_score, likely_range


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


def test_form_is_absolute_delta_not_ratio():
    # +2 PPG on a 20 PPG base and on a 10 PPG base are the same form signal
    # under the absolute-delta formula (the old ratio form rewarded the
    # small base more).
    _, big, _ = compute_start_score(10.0, recent_ppg=22.0, season_ppg=20.0,
                                    position="RB")
    _, small, _ = compute_start_score(10.0, recent_ppg=12.0, season_ppg=10.0,
                                      position="RB")
    assert big["form"] == small["form"] == 1.04  # +2 / 4.0 scale * 8%


def test_form_scale_is_position_specific():
    # QB deltas scale by 5.5, so a +4.4 PPG delta is worth less than an RB's.
    _, qb, _ = compute_start_score(10.0, recent_ppg=24.4, season_ppg=20.0,
                                   position="QB")
    _, rb, _ = compute_start_score(10.0, recent_ppg=14.4, season_ppg=10.0,
                                   position="RB")
    assert qb["form"] == round(1 + (4.4 / 5.5) * 0.08, 3)
    assert rb["form"] == 1.08  # 4.4 / 4.0 clamps to the full +8%


def test_form_shrinks_with_small_samples():
    # played=4 -> shrink 4/8 = 0.5, so the +8% cap becomes +4%.
    score, factors, _ = compute_start_score(
        10.0, recent_ppg=20.0, season_ppg=10.0, position="RB", played_games=4)
    assert factors["form"] == 1.04
    assert abs(score - 10.4) < 1e-9
    # No played games at all: factors fully neutral.
    _, none_f, _ = compute_start_score(
        10.0, recent_ppg=20.0, season_ppg=10.0, position="RB", played_games=0)
    assert none_f["form"] == 1.0


def test_usage_dead_zone_and_scales():
    # |delta| < 1.0 per game is noise: neutral even for a big relative move.
    _, dead, _ = compute_start_score(10.0, usage_delta=0.9,
                                     usage_season_avg=2.0, position="WR")
    assert dead["usage"] == 1.0
    # RB opportunities: +1.5 on a 3.0 scale = +2.5%.
    _, rb, _ = compute_start_score(10.0, usage_delta=1.5,
                                   usage_season_avg=12.0, position="RB")
    assert rb["usage"] == 1.025
    # WR targets: -2.0 on a 2.0 scale = the full -5%.
    _, wr, _ = compute_start_score(10.0, usage_delta=-2.0,
                                   usage_season_avg=6.0, position="WR")
    assert wr["usage"] == 0.95


def test_usage_dropped_for_qb():
    _, qb, _ = compute_start_score(10.0, usage_delta=6.0,
                                   usage_season_avg=30.0, position="QB")
    assert qb["usage"] == 1.0


def test_form_usage_product_is_capped():
    # Form +8% and usage +5% compound to 1.134 raw; the product caps at 1.10.
    score, factors, _ = compute_start_score(
        10.0, recent_ppg=20.0, season_ppg=10.0,
        usage_delta=3.0, usage_season_avg=15.0, position="RB")
    assert factors["form"] == 1.08
    assert factors["usage"] == 1.05
    assert factors["form_usage"] == 1.10
    assert abs(score - 11.0) < 1e-9


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
    # Questionable is display-only since the 2026-10 rework: the tag is kept
    # (demotion label) but there is no score haircut.
    score, factors, demotion = compute_start_score(10.0, injury_status="Q")
    assert factors["avail"] == 1.0
    assert demotion == "questionable"
    assert abs(score - 10.0) < 1e-9


def test_questionable_never_penalizes():
    """Q must not penalize at all, unlike the Vegas/weather residuals."""
    _, q, _ = compute_start_score(10.0, injury_status="QUESTIONABLE")
    _, vegas, _ = compute_start_score(10.0, implied_total=15, position="WR")
    _, wind, _ = compute_start_score(10.0, weather_kind="wind", position="QB")
    assert q["avail"] == 1.0
    assert q["avail"] >= vegas["vegas"]
    assert q["avail"] >= wind["weather"]


def test_doubtful_and_suspended_no_longer_zero():
    # Unified availability (2026-10): Doubtful stays on the board with its
    # tag and no zeroing; Suspended no longer zeroes either. Only the
    # not-playing statuses (OUT/IR/PUP/NFI) zero, as a backstop for
    # surfaces that score a single player directly.
    for status in ("Doubtful", "Suspended"):
        score, factors, demotion = compute_start_score(10.0, injury_status=status)
        assert factors["avail"] == 1.0
        assert score == 10.0
        assert demotion is None
    for status in ("IR", "PUP", "NFI"):
        score, factors, demotion = compute_start_score(10.0, injury_status=status)
        assert factors["avail"] == 0.0
        assert score == 0.0
        assert demotion == "out"


def test_pool_exclusion_statuses():
    from utils.start_sit_score import start_sit_pool_excluded
    for status in ("IR", "Out", "PUP", "NFI"):
        assert start_sit_pool_excluded(status)
    for status in ("Questionable", "Doubtful", "Suspended", "", None):
        assert not start_sit_pool_excluded(status)


def test_form_usage_inputs_played_weeks():
    from utils.start_sit_score import form_usage_inputs
    # Points/usage lists contain PLAYED weeks only (callers filter phantom
    # rows); recent is the last 3, season is all of them.
    out = form_usage_inputs([10.0, 20.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0])
    assert out["played_games"] == 4
    assert out["season_ppg"] == 25.0
    assert out["recent_ppg"] == 30.0
    assert out["usage_season_avg"] == 6.5
    assert out["usage_delta"] == 0.5
    empty = form_usage_inputs([], [])
    assert empty["played_games"] == 0
    assert empty["usage_delta"] is None


def test_usage_value_from_stats():
    from utils.start_sit_score import usage_value_from_stats
    row = {"rush_att": 12, "rec_tgt": 4, "rec": 3, "pass_att": 30}
    assert usage_value_from_stats(row, "RB") == 16.0  # opportunities, not touches
    assert usage_value_from_stats(row, "WR") == 4.0
    assert usage_value_from_stats(row, "QB") is None  # QB usage dropped
    assert usage_value_from_stats(None, "RB") is None


def test_low_implied_total_default_position():
    score, factors, demotion = compute_start_score(10.0, implied_total=16, low_total_team=True)
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


def test_oline_residual_is_small_and_directional():
    _, best, _ = compute_start_score(10.0, oline_index=100)
    _, avg, _ = compute_start_score(10.0, oline_index=50)
    _, worst, dem = compute_start_score(10.0, oline_index=0)
    assert best["oline"] == 1.04          # best line, +4%
    assert avg["oline"] == 1.0            # league-average is neutral
    assert worst["oline"] == 0.96         # worst line, -4%
    assert dem == "oline"                 # a below-average line flags a demotion
    # None leaves the factor neutral and out of the way.
    _, none_f, _ = compute_start_score(10.0)
    assert none_f["oline"] == 1.0


def test_oline_out_of_range_clamps():
    _, hi, _ = compute_start_score(10.0, oline_index=250)
    _, lo, _ = compute_start_score(10.0, oline_index=-40)
    assert hi["oline"] == 1.04
    assert lo["oline"] == 0.96


def test_weather_and_vegas_stack():
    score, factors, demotion = compute_start_score(
        10.0,
        implied_total=15,
        weather_kind="precip",
        position="WR",
        low_total_team=True,
    )
    assert factors["vegas"] == 0.92
    assert factors["weather"] == 0.95
    assert abs(score - 10.0 * 0.92 * 0.95) < 1e-9
    assert demotion == "low_total"  # injury-style demotions keep priority


def test_expected_plays_is_bounded_and_directional():
    slow, sf, dem = compute_start_score(10, expected_team_plays=56, league_average_plays=64)
    fast, ff, _ = compute_start_score(10, expected_team_plays=72, league_average_plays=64)
    assert sf["expected_plays"] == 0.95
    assert ff["expected_plays"] == 1.05
    assert slow < fast
    assert dem == "low_play_volume"


def test_role_confidence_changes_mean_conservatively_and_range_materially():
    stable, stable_f, _ = compute_start_score(10, role_confidence=1.0)
    volatile, volatile_f, dem = compute_start_score(10, role_confidence=0.0)
    assert stable_f["role"] == 1.0
    assert volatile_f["role"] == 0.97
    assert stable > volatile
    assert dem == "volatile_role"
    stable_range = likely_range(stable, 1.0)
    volatile_range = likely_range(volatile, 0.0)
    assert stable_range[1] - stable_range[0] < volatile_range[1] - volatile_range[0]


def test_defensive_injury_is_small_quality_weighted_residual():
    base, _, _ = compute_start_score(10)
    impacted, factors, _ = compute_start_score(10, defensive_injury_impact=1.0)
    assert factors["def_injuries"] == 1.03
    assert impacted == base * 1.03


def test_low_total_label_is_rank_based_not_threshold():
    # Same low absolute total, but the team is NOT in the week's bottom 8:
    # no "Low team total" label, while the vegas score multiplier still applies.
    score, factors, demotion = compute_start_score(10.0, implied_total=16, low_total_team=False)
    assert factors["vegas"] == 0.94
    assert demotion != "low_total"
    assert abs(score - 9.4) < 1e-9

    # Omitted flag defaults to False: same behavior.
    _, _, demotion2 = compute_start_score(10.0, implied_total=16)
    assert demotion2 != "low_total"

    # A bottom-8 team with a total ABOVE the old 20-point threshold still gets
    # the label: rank, not absolute total.
    _, _, demotion3 = compute_start_score(10.0, implied_total=21.5, low_total_team=True)
    assert demotion3 == "low_total"


def test_low_total_label_respects_first_dock_wins():
    # Injury demotions keep priority over the rank-based low-total label.
    _, _, demotion = compute_start_score(
        10.0, implied_total=16, low_total_team=True, injury_status="OUT"
    )
    assert demotion == "out"
    _, _, demotion_q = compute_start_score(
        10.0, implied_total=16, low_total_team=True, injury_status="Q"
    )
    assert demotion_q == "questionable"


def test_bottom_teams_by_implied_total():
    conds = {
        "KC": {"implied_total": 27.5},
        "BUF": {"implied_total": 26.0},
        "PHI": {"implied_total": 25.0},
        "SF": {"implied_total": 24.0},
        "DAL": {"implied_total": 23.0},
        "BAL": {"implied_total": 22.0},
        "DET": {"implied_total": 21.0},
        "MIA": {"implied_total": 20.0},
        "CIN": {"implied_total": 19.0},
        "NYJ": {"implied_total": 17.5},
        "CAR": {"implied_total": None},  # missing total: excluded
        # Bye-week team: no entry at all, never ranked.
    }
    bottom = bottom_teams_by_implied_total(conds)
    assert bottom == {"NYJ", "CIN", "MIA", "DET", "BAL", "DAL", "SF", "PHI"}
    assert "CAR" not in bottom


def test_bottom_teams_by_implied_total_fewer_than_n():
    conds = {"KC": {"implied_total": 27.5}, "NYJ": {"implied_total": 17.5}}
    assert bottom_teams_by_implied_total(conds) == {"KC", "NYJ"}
    assert bottom_teams_by_implied_total({}) == set()
    assert bottom_teams_by_implied_total(None) == set()
    # Custom n is honored.
    assert bottom_teams_by_implied_total(conds, n=1) == {"NYJ"}
