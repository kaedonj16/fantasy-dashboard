"""Regression coverage for standings Luck and expected-seed calculations."""
import math

import pytest

pd = pytest.importorskip("pandas")


def _frame(rows):
    defaults = {"finalized": True, "matchup_id": 1, "avatar": ""}
    framed = [{**defaults, **row} for row in rows]
    owner_ids = {owner: str(i + 1) for i, owner in enumerate(dict.fromkeys(
        row.get("owner") for row in framed
    ))}
    for row in framed:
        row.setdefault("roster_id", owner_ids[row["owner"]])
    return pd.DataFrame(framed)


def test_canonical_rows_without_win_support_one_week_results_and_zero_scores():
    from app import _all_play_from_df_weekly

    df = _frame([
        {"week": 1, "owner": "Zero", "points": 0, "points_against": 10},
        {"week": 1, "owner": "Winner", "points": 10, "points_against": 0},
        {"week": 1, "owner": "Tie A", "points": 5, "points_against": 5},
        {"week": 1, "owner": "Tie B", "points": 5, "points_against": 5},
    ])

    result = _all_play_from_df_weekly(df)

    assert set(result) == {"Zero", "Winner", "Tie A", "Tie B"}
    assert result["Zero"]["actual_wins"] == 0.0
    assert result["Winner"]["actual_wins"] == 1.0
    assert result["Tie A"]["actual_wins"] == 0.5
    # Hand calculation: Tie A beats Zero, ties Tie B, and loses to Winner.
    assert result["Tie A"]["all_play_pct"] == 0.5
    assert result["Tie A"]["expected_wins"] == 0.5
    assert result["Tie A"]["luck_delta"] == 0.0
    assert result["Winner"]["expected_seed"] == 1
    assert result["Zero"]["expected_seed"] == 4


def test_invalid_scores_are_skipped_without_becoming_zero_or_erasing_valid_rows():
    from app import _all_play_from_df_weekly

    df = _frame([
        {"week": 1, "owner": "A", "points": 100, "points_against": 90},
        {"week": 1, "owner": "B", "points": 90, "points_against": 100},
        {"week": 1, "owner": "Missing", "points": None, "points_against": 80},
        {"week": 1, "owner": "NaN", "points": float("nan"), "points_against": 75},
        {"week": 1, "owner": "Pandas NA", "points": pd.NA, "points_against": 72},
        {"week": 1, "owner": "Infinite", "points": math.inf, "points_against": 70},
        {"week": 1, "owner": "Text", "points": "not-a-score", "points_against": 60},
    ])

    result = _all_play_from_df_weekly(df)

    assert set(result) == {"A", "B"}
    assert result["A"]["expected_wins"] == 1.0
    assert result["A"]["luck_delta"] == 0.0
    assert result["B"]["expected_wins"] == 0.0
    assert result["B"]["luck_delta"] == 0.0


def test_live_future_cutoff_and_playoff_weeks_are_excluded():
    from app import _all_play_from_df_weekly

    df = _frame([
        {"week": 1, "owner": "A", "points": 100, "points_against": 90},
        {"week": 1, "owner": "B", "points": 90, "points_against": 100},
        {"week": 2, "owner": "A", "points": 0, "points_against": 10},
        {"week": 2, "owner": "B", "points": 10, "points_against": 0},
        {"week": 3, "owner": "A", "points": 0, "points_against": 10, "finalized": False},
        {"week": 3, "owner": "B", "points": 10, "points_against": 0, "finalized": False},
        {"week": 15, "owner": "A", "points": 0, "points_against": 10},
        {"week": 15, "owner": "B", "points": 10, "points_against": 0},
    ])

    through_one = _all_play_from_df_weekly(df, max_week=1, regular_season_weeks=14)
    assert through_one["A"]["actual_wins"] == 1.0
    assert through_one["A"]["expected_wins"] == 1.0

    regular = _all_play_from_df_weekly(df, regular_season_weeks=14)
    assert regular["A"]["actual_wins"] == 1.0
    assert regular["A"]["expected_wins"] == 1.0
    assert regular["A"]["games"] == 2


def test_zero_luck_renders_as_number_and_no_completed_games_stays_unavailable():
    from app import _all_play_from_df_weekly, render_standings

    stats = pd.DataFrame([{
        "owner": "A", "Wins": 1, "Losses": 0, "Ties": 0, "Win%": 1.0,
        "PF": 100.0, "PA": 90.0, "Streak": "W1", "avatar": "",
    }, {
        "owner": "B", "Wins": 0, "Losses": 1, "Ties": 0, "Win%": 0.0,
        "PF": 90.0, "PA": 100.0, "Streak": "L1", "avatar": "",
    }])
    completed = _frame([
        {"week": 1, "owner": "A", "points": 100, "points_against": 90},
        {"week": 1, "owner": "B", "points": 90, "points_against": 100},
    ])
    analysis = _all_play_from_df_weekly(completed)
    html = render_standings(stats, 2, all_play=analysis)
    assert "luck-chip luck-neu" in html
    assert ">0.0</span>" in html
    assert "1st" in html

    live = completed.assign(finalized=False)
    assert _all_play_from_df_weekly(live) == {}
    unavailable_html = render_standings(stats, 2, all_play={})
    assert unavailable_html.count("&ndash;") >= 4


def test_initial_and_latest_week_selector_calculations_render_identically():
    from app import _all_play_from_df_weekly, build_standings_as_of_week, render_standings
    from dashboard_services.service import finalize_team_stats

    df = _frame([
        {"week": 1, "owner": "A", "points": 100, "points_against": 90},
        {"week": 1, "owner": "B", "points": 90, "points_against": 100},
        {"week": 2, "owner": "A", "points": 80, "points_against": 110, "matchup_id": 2},
        {"week": 2, "owner": "B", "points": 110, "points_against": 80, "matchup_id": 2},
    ])
    avatars = {"A": "", "B": ""}
    stats = finalize_team_stats(df, avatars, {}, [], 2, regular_season_weeks=14)
    ctx = {"df_weekly": df, "team_stats": stats, "matchups_by_week": {},
           "users": [], "league_settings": {"playoff_week_start": 15}}
    selected = build_standings_as_of_week(ctx, 2)

    initial_html = render_standings(stats, 2, all_play=_all_play_from_df_weekly(df))
    selected_html = render_standings(
        selected["team_stats"], 2,
        all_play=_all_play_from_df_weekly(selected["df_weekly"]),
    )
    assert initial_html == selected_html
