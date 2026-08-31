"""Usage-table validation must accept the (correct) all-zero-games table
until regular-season games have actually been played.

Sleeper reports season_type "pre" through August, then often flips to
"regular" / week 1 days (sometimes a week+) before Thursday kickoff. Every
player still has 0 regular-season games in that window. The validator used
to treat only "off"/"pre" as a no-games-expected state, so the week-1
pre-kickoff snapshot fell into the strict in-season branch and raised on
the 100%-zero-games table. That stopped usage_table.json from being
written, which in turn made rewrite_value_table_with_model fail
(FileNotFoundError) — freezing all player values on the last good
model_values.json until games start.
"""
import pytest

from data_building.external_data.usage_table_validation import validate_usage_table


def _rows(n=500, games=0, ppg=0.0):
    return [{"id": str(i), "usage": {"games": games, "ppr_ppg": ppg}} for i in range(n)]


def _mixed_rows(n=500, with_games=250, ppg=12.0):
    rows = []
    for i in range(n):
        if i < with_games:
            rows.append({"id": str(i), "usage": {"games": 1, "ppr_ppg": ppg}})
        else:
            rows.append({"id": str(i), "usage": {"games": 0, "ppr_ppg": 0.0}})
    return rows


def test_preseason_all_zero_games_is_accepted():
    # 100% zero games is correct in preseason and must NOT raise.
    validate_usage_table(_rows(games=0, ppg=0.0), {}, 2026, {"season_type": "pre"})


def test_offseason_all_zero_games_is_accepted():
    validate_usage_table(_rows(games=0, ppg=0.0), {}, 2026, {"season_type": "off"})


def test_regular_week1_all_zero_games_is_accepted():
    # Sleeper has flipped to regular/week 1 but no games have been played.
    validate_usage_table(
        _rows(games=0, ppg=0.0), {}, 2026, {"season_type": "regular", "week": 1}
    )


def test_regular_week1_with_real_stats_is_accepted():
    # Once Thursday/Sunday games land, week 1 is a normal in-season table.
    validate_usage_table(_mixed_rows(), {}, 2026, {"season_type": "regular", "week": 1})


def test_in_season_all_zero_games_is_rejected():
    # Mid-season, 100% zero games really does mean a broken fetch — keep raising.
    with pytest.raises(ValueError, match="Too many players with 0 games"):
        validate_usage_table(
            _rows(games=0, ppg=0.0), {}, 2026, {"season_type": "regular", "week": 9}
        )


def test_too_few_players_is_always_rejected():
    with pytest.raises(ValueError, match="Usage table too small"):
        validate_usage_table(_rows(n=10, games=0, ppg=0.0), {}, 2026, {"season_type": "pre"})
