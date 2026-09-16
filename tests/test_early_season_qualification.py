from datetime import date, timedelta

from utils.fantasy_scoring import completed_points_summary, score_stats
from utils.season_qualification import (
    completed_regular_season_rounds,
    qualification_policy,
    scaled_minimum,
)


def _schedule(final_weeks, partial_weeks=()):
    def load(_season, week):
        if week in final_weeks:
            return [{"seasonType": "Regular", "gameStatus": "Final"}] * 2
        if week in partial_weeks:
            return [
                {"seasonType": "Regular", "gameStatus": "Final"},
                {"seasonType": "Regular", "gameStatus": "Scheduled"},
            ]
        return []
    return load


def test_one_complete_round_uses_one_game_and_scaled_volume_minimums():
    policy = qualification_policy(2026, load_week=_schedule({1}, {2}))
    assert policy.completed_weeks == (1,)
    assert policy.games_min == 1
    assert policy.minimum("total_pass_att", 50) == 13
    assert policy.minimum("total_carries", 20) == 5
    assert policy.note() == "Small sample · 1 game"


def test_staggered_completion_and_byes_do_not_advance_round():
    assert completed_regular_season_rounds(
        2026, load_week=_schedule({1}, {2})) == [1]


def test_stale_scheduled_status_qualifies_after_entire_round_date_passes():
    """Preseason schedule caches must not blank every current-season modal."""
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y%m%d")

    def load(_season, week):
        dates = {1: [yesterday, yesterday], 2: [yesterday, tomorrow]}.get(week, [])
        return [{"seasonType": "Regular", "gameStatus": "Scheduled",
                 "gameStatusCode": "0", "gameDate": value} for value in dates]

    # Week 1's games are all on prior calendar dates despite stale provider
    # statuses. Week 2 remains excluded until its complete slate has passed.
    assert completed_regular_season_rounds(2026, load_week=load) == [1]


def test_date_fallback_rejects_postponed_and_invalid_schedule_rows():
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")

    def load(_season, week):
        if week == 1:
            return [{"seasonType": "Regular", "gameStatus": "Postponed",
                     "gameStatusCode": "0", "gameDate": yesterday}]
        if week == 2:
            return [{"seasonType": "Regular", "gameStatus": "Scheduled",
                     "gameStatusCode": "0", "gameDate": "20261399"}]
        return []

    assert completed_regular_season_rounds(2026, load_week=load) == []


def test_qualification_increases_to_existing_four_game_gate():
    assert [scaled_minimum(4, n) for n in range(1, 6)] == [1, 2, 3, 4, 4]
    assert [scaled_minimum(20, n) for n in range(1, 5)] == [5, 10, 15, 20]


def test_selected_week_range_only_counts_fully_completed_rounds():
    policy = qualification_policy(
        2026, week_start=2, week_end=3, load_week=_schedule({1, 2, 3}))
    assert policy.completed_weeks == (2, 3)
    assert policy.games_min == 2


def test_custom_scoring_and_te_premium_share_scoring_engine():
    line = {"rec": 4, "rec_yd": 50, "rec_td": 1}
    settings = {"rec": 0.5, "rec_yd": 0.2, "rec_td": 7, "bonus_rec_te": 1.0}
    assert score_stats(line, settings, "WR") == 19
    assert score_stats(line, settings, "TE") == 23


def test_zero_denominator_is_not_manufactured():
    # The production SQL uses NULLIF(volume, 0); document that qualification
    # scaling itself never turns zero opportunities into one.
    assert scaled_minimum(15, 1) > 0


def test_actual_points_do_not_multiply_rounded_ppg_and_zero_is_real():
    summary = completed_points_summary([10.04, 10.04, 10.04])
    assert summary == {"games": 3, "total": 30.119999999999997, "ppg": 10.04}
    assert round(summary["total"], 1) == 30.1
    assert completed_points_summary([0.0]) == {"games": 1, "total": 0.0, "ppg": 0.0}
    assert completed_points_summary([]) is None


def test_non_league_modal_keeps_selected_season():
    source = open("static/player_modal.js", encoding="utf-8").read()
    assert ": `/api/player-details/${playerId}?season=${season}&${leagueParams}`" in source
    assert "ppgVal != null ? Number(ppgVal).toFixed(1) : 'N/A'" in source
    assert "totalPts != null ? fmtPts(totalPts) : 'N/A'" in source
