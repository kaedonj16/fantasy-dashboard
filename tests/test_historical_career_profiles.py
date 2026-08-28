"""Career-stage, breakout vs first-time elite, draft-capital rates (slim CI)."""
from pathlib import Path

from dashboard_services.historical.career_profiles import (
    BREAKOUT_RANK_THRESHOLD,
    PRIOR_NON_STARTER_RANK,
    assemble_profile_aggregates,
    build_career_path_overlay,
    build_draft_capital_rates,
    build_repeat_and_breakout_rates,
    build_stage_rates,
    is_engine_breakout,
    is_first_time_elite,
    is_league_winner,
    is_league_winner_smash,
    was_engine_non_starter,
)
from dashboard_services.historical.definitions import DEFAULT_BAYES_PRIOR_N, empirical_bayes
from dashboard_services.historical.finish_rates import cohort_hit_rate, make_rate


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "data_building" / "breakout_engine" / "backtest_breakout_model.py"


def _row(**kwargs):
    base = {
        "sleeper_id": kwargs.get("sleeper_id", "p"),
        "season": kwargs.get("season", 2020),
        "position": kwargs.get("position", "RB"),
        "games": 16,
    }
    base.update(kwargs)
    return base


def test_engine_breakout_constants_stay_in_sync_with_source():
    src = ENGINE.read_text(encoding="utf-8")
    assert "BREAKOUT_RANK_THRESHOLD = 12" in src
    assert "PRIOR_NON_STARTER_RANK = 13" in src
    assert "was_non_starter = (prior_rank is None) or (prior_rank > prior_non_starter)" in src
    assert BREAKOUT_RANK_THRESHOLD == 12
    assert PRIOR_NON_STARTER_RANK == 13
    # Rank 13 is NOT a non-starter under the engine (`>` not `>=`).
    assert was_engine_non_starter(None) is True
    assert was_engine_non_starter(14) is True
    assert was_engine_non_starter(13) is False
    assert was_engine_non_starter(12) is False


def test_engine_breakout_is_not_first_time_elite():
    # Prior rank 13, this season RB8: first-time elite, not engine breakout.
    assert is_engine_breakout(13, 8) is False
    assert is_first_time_elite(False, 8) is True
    # Prior rank 20 but already had a career top-12: engine breakout, not first-time.
    assert is_engine_breakout(20, 8) is True
    assert is_first_time_elite(True, 8) is False
    # Rookie top-12 is both.
    assert is_engine_breakout(None, 5) is True
    assert is_first_time_elite(False, 5) is True
    # Prior RB1 repeating is neither.
    assert is_engine_breakout(4, 4) is False
    assert is_first_time_elite(True, 4) is False


def test_league_winner_reuses_top_5_and_smash_is_not_engine_cut():
    # Finish 5 is inside the existing top_5 cutoff; 6 is not.
    assert is_league_winner(5) is True
    assert is_league_winner(6) is False
    assert is_league_winner(None) is False
    # Rank 13 last year finishing top-5 is a smash, not an engine breakout.
    assert is_league_winner_smash(13, 3) is True
    assert is_engine_breakout(13, 3) is False
    # Rank 12 last year is already elite — league winner but not smash.
    assert is_league_winner(2) is True
    assert is_league_winner_smash(12, 2) is False
    # No prior + top-5 is a smash (and an engine breakout, which uses top-12).
    assert is_league_winner_smash(None, 4) is True
    assert is_engine_breakout(None, 4) is True


def test_repeat_rates_use_previous_season_not_missing_as_zero():
    rows = [
        _row(
            sleeper_id="rep",
            season=2021,
            previous_season_finish=4,
            ppr_positional_finish=2,
            prior_top12_count=1,
            previously_top12=True,
        ),
        _row(
            sleeper_id="rep-miss",
            season=2021,
            previous_season_finish=8,
            ppr_positional_finish=20,
            prior_top12_count=1,
            previously_top12=True,
        ),
        # Missing prior finish: excluded from the prev-top12 cohort, not a 0.
        _row(
            sleeper_id="no-prior",
            season=2021,
            previous_season_finish=None,
            ppr_positional_finish=1,
            prior_top12_count=0,
            previously_top12=False,
            first_time_top12_candidate=True,
        ),
    ]
    rates = build_repeat_and_breakout_rates(rows)["RB"]
    assert rates["prev_top12_to_top12"]["sample_size"] == 2
    assert rates["prev_top12_to_top12"]["successes"] == 1
    assert abs(rates["prev_top12_to_top12"]["raw_rate"] - 0.5) < 1e-9
    # The no-prior row is a first-time candidate (and an engine non-starter).
    assert rates["n_first_time_candidates"] == 1
    assert rates["first_time_elite_among_candidates"]["successes"] == 1
    assert rates["n_engine_non_starters"] == 1
    assert rates["engine_breakout_among_non_starters"]["successes"] == 1
    assert rates["league_winner"]["successes"] == 2  # finish 2 and finish 1
    assert rates["n_league_winner_smash_candidates"] == 1
    assert rates["league_winner_smash_among_non_top12"]["successes"] == 1


def test_prev_top12_to_top5_uses_top_5_cutoff():
    rows = [
        _row(sleeper_id="a", previous_season_finish=10, ppr_positional_finish=4),
        _row(sleeper_id="b", previous_season_finish=11, ppr_positional_finish=6),
        _row(sleeper_id="c", previous_season_finish=3, ppr_positional_finish=1),
    ]
    rates = build_repeat_and_breakout_rates(rows)["RB"]
    assert rates["prev_top12_to_top5"]["sample_size"] == 3
    assert rates["prev_top12_to_top5"]["successes"] == 2  # finishes 4 and 1
    assert rates["two_plus_prior_top12_to_top12"]["sample_size"] == 0
    assert rates["two_plus_prior_top12_to_top12"]["raw_rate"] is None

    two = [
        _row(
            sleeper_id="vet",
            prior_top12_count=2,
            previously_top12=True,
            previous_season_finish=7,
            ppr_positional_finish=9,
        ),
        _row(
            sleeper_id="vet2",
            prior_top12_count=3,
            previously_top12=True,
            previous_season_finish=2,
            ppr_positional_finish=20,
        ),
    ]
    rates2 = build_repeat_and_breakout_rates(two)["RB"]
    assert rates2["two_plus_prior_top12_to_top12"]["sample_size"] == 2
    assert rates2["two_plus_prior_top12_to_top12"]["successes"] == 1


def test_bounce_back_rates_need_prior_elite_and_a_down_year():
    rows = [
        _row(
            sleeper_id="down",
            years_experience=2,
            draft_capital_bucket="round_1",
            previous_season_finish=42,
            ppr_positional_finish=8,
            prior_top12_count=1,
        ),
        _row(
            sleeper_id="down-miss",
            years_experience=2,
            draft_capital_bucket="round_1",
            previous_season_finish=50,
            ppr_positional_finish=30,
            prior_top12_count=1,
        ),
        _row(
            sleeper_id="still-elite",
            years_experience=2,
            previous_season_finish=4,
            ppr_positional_finish=3,
            prior_top12_count=1,
        ),
        _row(
            sleeper_id="never",
            years_experience=2,
            previous_season_finish=42,
            ppr_positional_finish=8,
            prior_top12_count=0,
        ),
    ]
    rates = build_repeat_and_breakout_rates(rows)["RB"]
    assert rates["n_bounce_back"] == 2
    assert rates["bounce_back"]["top_12"]["successes"] == 1
    assert rates["bounce_back"]["top_12"]["sample_size"] == 2
    assert rates["bounce_back_by_stage"]["year_3"]["top_12"]["sample_size"] == 2
    assert rates["bounce_back_by_capital"]["round_1"]["top_12"]["sample_size"] == 2
    overlay = build_career_path_overlay(rows)
    assert overlay["bounce_back"]["RB"]["n_bounce_back"] == 2
    assert overlay["prior_top12_count"]["down"] >= 1


def test_missing_exp_is_not_rookie():
    rows = [
        _row(sleeper_id="rook", years_experience=0, ppr_positional_finish=8),
        _row(sleeper_id="vet", years_experience=6, ppr_positional_finish=8),
        _row(sleeper_id="unknown", years_experience=None, ppr_positional_finish=1),
    ]
    stages = build_stage_rates(rows)["RB"]
    assert stages["n_missing_exp_excluded"] == 1
    assert stages["by_stage"]["rookie"]["sample_size"] == 1
    assert stages["by_stage"]["year_6_plus"]["sample_size"] == 1
    assert stages["by_stage"]["rookie"]["successes"] == 1
    assert stages["n_known_stage"] == 2


def test_missing_draft_capital_is_not_undrafted():
    rows = [
        _row(
            sleeper_id="r1",
            position="WR",
            draft_capital_bucket="round_1",
            draft_year=2018,
            years_experience=3,
            ppr_positional_finish=5,
            season=2021,
        ),
        _row(
            sleeper_id="unknown",
            position="WR",
            draft_capital_bucket=None,
            draft_year=2018,
            years_experience=3,
            ppr_positional_finish=1,
            season=2021,
        ),
        _row(
            sleeper_id="udfa",
            position="WR",
            draft_capital_bucket="undrafted",
            draft_year=2018,
            years_experience=3,
            ppr_positional_finish=40,
            season=2021,
        ),
    ]
    caps = build_draft_capital_rates(rows)["WR"]
    assert caps["n_missing_capital_excluded"] == 1
    r1 = caps["season_level_by_capital"]["round_1"]["top_12"]
    assert r1["sample_size"] == 1
    assert r1["successes"] == 1
    udfa = caps["season_level_by_capital"]["undrafted"]["top_12"]
    assert udfa["sample_size"] == 1
    assert udfa["successes"] == 0
    assert r1["sample_size"] + udfa["sample_size"] == 2


def test_cumulative_by_year_2_requires_closed_window():
    rows = [
        _row(
            sleeper_id="hit",
            position="WR",
            draft_capital_bucket="round_1",
            draft_year=2018,
            years_experience=0,
            ppr_positional_finish=40,
            season=2018,
        ),
        _row(
            sleeper_id="hit",
            position="WR",
            draft_capital_bucket="round_1",
            draft_year=2018,
            years_experience=1,
            ppr_positional_finish=9,
            season=2019,
        ),
        # 2024 draftee: year 2 is 2025; warehouse max here is 2024 → excluded.
        _row(
            sleeper_id="too-new",
            position="WR",
            draft_capital_bucket="round_1",
            draft_year=2024,
            years_experience=0,
            ppr_positional_finish=3,
            season=2024,
        ),
        _row(
            sleeper_id="no-cap",
            position="WR",
            draft_capital_bucket=None,
            draft_year=2018,
            years_experience=1,
            ppr_positional_finish=2,
            season=2019,
        ),
    ]
    caps = build_draft_capital_rates(rows)["WR"]
    by2 = caps["cumulative"]["top12_by_year_2"]["by_capital"]["round_1"]
    assert by2["n_players"] == 1
    assert by2["successes"] == 1
    assert abs(by2["raw_rate"] - 1.0) < 1e-9


def test_smoothing_and_confidence_on_small_samples():
    rows = [_row(sleeper_id=f"h{i}", ppr_positional_finish=1) for i in range(3)]
    rows += [_row(sleeper_id=f"m{i}", ppr_positional_finish=40) for i in range(7)]
    rate = cohort_hit_rate(rows, prior_rate=0.20)
    assert rate["sample_size"] == 10
    assert rate["successes"] == 3
    assert abs(rate["raw_rate"] - 0.30) < 1e-9
    expected = empirical_bayes(3, 10, 0.20 * DEFAULT_BAYES_PRIOR_N, DEFAULT_BAYES_PRIOR_N)
    assert abs(rate["smoothed_rate"] - expected) < 1e-9
    assert abs(rate["smoothed_rate"] - 0.25) < 1e-9
    assert rate["confidence"] == "low"
    empty = make_rate(0, 0)
    assert empty["raw_rate"] is None
    assert empty["smoothed_rate"] is None
    assert empty["display_pct"] is None


def test_assemble_has_no_adp_or_projections_and_keeps_both_breakout_defs():
    rows = [
        _row(
            sleeper_id="a",
            season=2020,
            age=24.0,
            years_experience=2,
            draft_year=2018,
            draft_capital_bucket="round_1",
            ppr_positional_finish=4,
            previous_season_finish=20,
            previously_top12=False,
            first_time_top12_candidate=True,
            prior_top12_count=0,
        ),
        _row(
            sleeper_id="b",
            season=2015,
            age=24.0,
            years_experience=2,
            draft_year=2013,
            draft_capital_bucket="round_1",
            ppr_positional_finish=1,
            previous_season_finish=1,
            previously_top12=True,
            prior_top12_count=2,
        ),
    ]
    payload = assemble_profile_aggregates(rows)
    assert payload["phase"] == 9
    assert "prior_usage" in payload
    assert "comps" in payload
    assert "adp" in payload
    assert "signals" in payload
    assert "walkforward" in payload
    assert payload["era_floor"] == 2016
    assert payload["n_player_seasons"] == 1
    assert payload["season_range"] == [2020, 2020]
    assert payload["descriptive_only"] is True
    defs = payload["definitions"]
    assert defs["no_adp"] is False
    assert defs["adp_in_comps"] is False
    assert defs["adp_in_ranking"] is False
    assert defs["projections_in_comps"] is False
    assert defs["projections_in_ranking"] is False
    assert defs["no_projections"] is True
    assert defs["pick_score_validated"] is False
    assert defs["pick_score_in_live_ranking"] is False
    assert "league_winner" in defs and "league_winner_smash" in defs
    assert payload["signals"]["no_blended_score"] is True
    assert payload["signals"]["warehouse_has_projections"] is False
    assert payload["board"]["not_in_ranking"] is True
    assert payload["board"]["not_in_pick_score"] is True
    assert payload["walkforward"]["not_a_second_engine"] is True
    assert payload["walkforward"]["pick_score"]["validated"] is False
    assert payload["walkforward"]["pick_score"]["in_live_ranking"] is False
    assert "preseason_profiles" in payload
    assert "age_curves_by_tier" in payload
    assert "career_stages_by_tier" in payload
    assert "prior_usage_by_tier" in payload
    assert "engine_breakout" in defs and "first_time_elite" in defs

    def _keys(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                yield str(key)
                yield from _keys(val)
        elif isinstance(obj, list):
            for item in obj:
                yield from _keys(item)

    keys = set(_keys(payload))
    forbidden = [k for k in keys if k.startswith("projected_")]
    assert forbidden == []
    for leaf in payload["comps"]["by_position"]["RB"]["leaves"]:
        assert "adp" not in (leaf.get("key") or {})
    rb_age = payload["age_curves"]["RB"]["by_integer_age"]["24"]
    assert "distribution" in rb_age and "conditional" in rb_age
