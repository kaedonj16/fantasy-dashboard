"""Winners-bracket-only derivation (frontier leftover).

Fleaflicker schedules every team during playoff weeks, so the raw
derivation counted consolation games as bracket games and the
prior-season rules deriver stamped a 12-team field on an 8-team
league (92916). derive_bracket_from_matchups now keeps only the
winners bracket, identified structurally from the completed final.
"""
from utils.playoff_bracket import derive_bracket_from_matchups


def _game(mid, a, b, pa, pb):
    return [
        {"roster_id": a, "matchup_id": mid, "points": pa},
        {"roster_id": b, "matchup_id": mid, "points": pb},
    ]


def _full_league_playoffs():
    """12 teams play Weeks 15-17; only seeds 1-8 are the real field."""
    return {
        15: (_game(1, 1, 8, 110, 90) + _game(2, 4, 5, 80, 95)
             + _game(3, 2, 7, 105, 88) + _game(4, 3, 6, 92, 99)
             + _game(5, 9, 12, 100, 70) + _game(6, 10, 11, 85, 91)),
        16: (_game(1, 1, 5, 112, 90) + _game(2, 2, 6, 95, 101)
             + _game(3, 8, 4, 97, 89) + _game(4, 7, 3, 90, 93)
             + _game(5, 9, 11, 102, 84) + _game(6, 12, 10, 88, 94)),
        17: (_game(1, 1, 6, 118, 100) + _game(2, 5, 2, 96, 90)
             + _game(3, 8, 3, 99, 91) + _game(4, 4, 7, 87, 92)
             + _game(5, 9, 10, 103, 95) + _game(6, 11, 12, 90, 93)),
    }


def test_consolation_games_are_not_the_winners_bracket():
    games = derive_bracket_from_matchups(_full_league_playoffs(), 15)
    field = {g[s] for g in games for s in ("t1", "t2")}
    assert field == set(range(1, 9))
    by_round = {}
    for g in games:
        by_round.setdefault(g["r"], []).append(g)
    assert len(by_round[1]) == 4 and len(by_round[2]) == 2
    assert len(by_round[3]) == 1
    final = by_round[3][0]
    assert {final["t1"], final["t2"]} == {1, 6}
    assert final["w"] == 1


def test_derived_games_carry_their_week():
    games = derive_bracket_from_matchups(_full_league_playoffs(), 15)
    assert {g["week"] for g in games} == {15, 16, 17}
    assert min(g["week"] for g in games) == 15


def test_in_progress_playoffs_pass_through_unchanged():
    by_week = _full_league_playoffs()
    games = derive_bracket_from_matchups({15: by_week[15]}, 15)
    # No completed final yet: nothing is dropped on a guess.
    assert len(games) == 6


def test_clean_field_without_consolation_is_untouched():
    by_week = {
        15: _game(1, 1, 4, 100, 90) + _game(2, 2, 3, 95, 99),
        16: _game(1, 1, 3, 105, 98),
    }
    games = derive_bracket_from_matchups(by_week, 15)
    assert len(games) == 3
    assert games[-1]["w"] == 1


def test_six_team_field_with_byes_and_consolation():
    # Seeds 1-2 bye; 3v6 and 4v5 in round 1; teams 7-12 play
    # consolation games all three weeks.
    by_week = {
        15: (_game(1, 3, 6, 100, 90) + _game(2, 4, 5, 88, 95)
             + _game(3, 7, 12, 99, 80) + _game(4, 8, 11, 92, 90)
             + _game(5, 9, 10, 85, 87)),
        16: (_game(1, 1, 5, 110, 95) + _game(2, 2, 3, 100, 102)
             + _game(3, 6, 4, 90, 93) + _game(4, 7, 10, 98, 90)
             + _game(5, 12, 8, 85, 91) + _game(6, 11, 9, 94, 89)),
        17: (_game(1, 1, 3, 115, 105) + _game(2, 5, 2, 97, 94)
             + _game(3, 4, 6, 92, 90) + _game(4, 7, 8, 100, 96)
             + _game(5, 10, 11, 90, 92) + _game(6, 12, 9, 88, 90)),
    }
    # The consolation bracket also produces an undefeated final-round
    # pair (7 and 8), so the standings order breaks the tie toward
    # the real finalists.
    games = derive_bracket_from_matchups(
        by_week, 15, seed_order=list(range(1, 13)))
    field = {g[s] for g in games for s in ("t1", "t2")}
    assert field == {1, 2, 3, 4, 5, 6}
    assert len(games) == 5   # 2 quarters + 2 semis + final
