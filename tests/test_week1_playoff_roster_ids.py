"""Week-1 seeded 0-0 standings must not mint fake playoff-odds roster_ids.

The Season Hub seed for empty finalized weeks left ``team_stats`` non-empty at
0-0. The playoff sim treated that as ``has_games`` and built teams from the
DataFrame RangeIndex (0..N-1), so Front Office / sellers-to-call looked up the
wrong row while the odds *table* (by team name) still looked fine.
"""
import pytest

pd = pytest.importorskip("pandas")

from data_building.simulate_playoff_odds import (
    _build_teams,
    team_stats_have_played_games,
)


def _seeded_zero(owners):
    rows = []
    for owner in owners:
        rows.append({
            "owner": owner,
            "Wins": 0, "Losses": 0, "Ties": 0, "G": 0,
            "PF": 0.0, "AVG": 0.0, "STD": 0.0,
        })
    return pd.DataFrame(rows)


def test_seeded_zero_standings_are_not_played_games():
    seeded = _seeded_zero(["Alpha", "Bravo"])
    assert not seeded.empty
    assert not team_stats_have_played_games(seeded)
    assert not team_stats_have_played_games(None)
    assert not team_stats_have_played_games(pd.DataFrame())


def test_standings_with_wins_count_as_played():
    df = pd.DataFrame([
        {"owner": "A", "Wins": 1, "Losses": 0, "Ties": 0, "G": 1, "PF": 100.0, "AVG": 100.0, "STD": 10.0},
        {"owner": "B", "Wins": 0, "Losses": 1, "Ties": 0, "G": 1, "PF": 80.0, "AVG": 80.0, "STD": 10.0},
    ])
    assert team_stats_have_played_games(df)


def test_build_teams_resolves_roster_id_from_owner_not_range_index():
    # RangeIndex 0..1 would previously become roster_id 0 and 1 — wrong when
    # real ESPN/Sleeper ids are 5 and 12.
    stats = _seeded_zero(["Puka Nacua Matata", "You Win"])
    roster_map = {"5": "Puka Nacua Matata", "12": "You Win"}
    teams = _build_teams(stats, roster_map)
    by_name = {t["name"]: t["roster_id"] for t in teams}
    assert by_name["Puka Nacua Matata"] == 5
    assert by_name["You Win"] == 12
    assert {t["roster_id"] for t in teams} == {5, 12}


def test_build_teams_skips_range_index_without_roster_map_match():
    stats = pd.DataFrame([
        {"owner": "Solo", "Wins": 2, "Losses": 1, "Ties": 0,
         "PF": 300.0, "AVG": 100.0, "STD": 12.0},
    ])
    # Owner not in map and index 0 is not a roster key → drop the row rather
    # than invent roster_id=0.
    assert _build_teams(stats, {"9": "Other Team"}) == []
