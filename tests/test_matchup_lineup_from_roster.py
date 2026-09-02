"""Matchup preview should not treat a full roster as starters."""
from utils.matchup_schedule import lineup_from_roster


def test_lineup_from_roster_uses_reserve():
    roster = {
        "players": ["a", "b", "c"],
        "starters": ["a", "b", "c"],
        "reserve": ["c"],
    }
    starters, bench = lineup_from_roster(roster)
    assert starters == ["a", "b"]
    assert bench == ["c"]


def test_lineup_from_roster_rejects_full_starter_list():
    players = [str(i) for i in range(15)]
    roster = {"players": players, "starters": players, "reserve": []}
    starters, bench = lineup_from_roster(roster)
    assert starters == []
    assert bench == players
