"""Derived / projected playoff brackets for hosts without a native bracket."""
from utils.playoff_bracket import (
    derive_bracket_from_matchups,
    derive_or_project_bracket,
    pair_matchup_sides,
    project_bracket_from_seeds,
)


def test_pair_matchup_sides_groups_two_team_games():
    rows = [
        {"roster_id": 1, "matchup_id": 1, "points": 100},
        {"roster_id": 2, "matchup_id": 1, "points": 90},
        {"roster_id": 3, "matchup_id": 2, "points": 80},
    ]
    pairs = pair_matchup_sides(rows)
    assert len(pairs) == 1
    assert pairs[0][0] == 1


def test_derive_bracket_from_scored_playoff_week():
    by_week = {
        15: [
            {"roster_id": 3, "matchup_id": 1, "points": 110},
            {"roster_id": 6, "matchup_id": 1, "points": 90},
            {"roster_id": 4, "matchup_id": 2, "points": 80},
            {"roster_id": 5, "matchup_id": 2, "points": 95},
        ]
    }
    games = derive_bracket_from_matchups(by_week, 15)
    assert len(games) == 2
    winners = {g["w"] for g in games}
    assert winners == {3, 5}
    assert all(g.get("derived") for g in games)


def test_derive_skips_consolation_and_unplayed_zero_zero():
    assert derive_bracket_from_matchups({15: []}, 15, kind="losers") == []
    by_week = {15: [
        {"roster_id": 1, "matchup_id": 1, "points": 0},
        {"roster_id": 2, "matchup_id": 1, "points": 0},
    ]}
    games = derive_bracket_from_matchups(by_week, 15)
    assert games[0]["w"] is None and games[0]["l"] is None


def test_project_six_team_field_gives_byes_to_top_two():
    games = project_bracket_from_seeds([1, 2, 3, 4, 5, 6], playoff_teams=6)
    assert len(games) == 2
    ids = {(g["t1"], g["t2"]) for g in games}
    assert (3, 6) in ids
    assert (4, 5) in ids
    assert all(g.get("projected") for g in games)


def test_derive_or_project_prefers_real_games():
    actual = derive_or_project_bracket(
        matchups_by_week={15: [
            {"roster_id": 1, "matchup_id": 1, "points": 10},
            {"roster_id": 8, "matchup_id": 1, "points": 3},
        ]},
        playoff_week_start=15,
        seed_roster_ids=[1, 2, 3, 4],
        playoff_teams=4,
    )
    assert actual[0]["w"] == 1
    projected = derive_or_project_bracket(
        matchups_by_week={},
        playoff_week_start=15,
        seed_roster_ids=[1, 2, 3, 4],
        playoff_teams=4,
    )
    assert len(projected) == 2
    assert all(g.get("projected") for g in projected)
