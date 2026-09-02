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
    assert len(starters) == 9
    assert len(bench) == 6


def test_lineup_from_roster_all_bench_does_not_blank():
    players = [str(i) for i in range(15)]
    roster = {"players": players, "starters": [], "reserve": players}
    starters, bench = lineup_from_roster(roster)
    assert len(starters) == 9
    assert len(bench) == 6


def test_lineup_from_roster_prefers_stored_starters_when_reserve_is_ir():
    """Yahoo/ESPN reserve is IR-only; bench must not become matchup starters."""
    players = [str(i) for i in range(12)]
    roster = {
        "players": players,
        "starters": players[:8],
        "reserve": [players[-1]],
    }
    starters, bench = lineup_from_roster(roster)
    assert starters == players[:8]
    assert bench == players[8:]
