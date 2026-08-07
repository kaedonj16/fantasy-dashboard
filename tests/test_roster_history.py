"""Unit tests for reconstruct_rosters_as_of (dashboard_services.roster_history).

Pure logic — reversing timestamped add/drop/trade transactions against the
current rosters to recover an earlier roster state.
"""
from dashboard_services.roster_history import reconstruct_rosters_as_of


def _roster(rid, players):
    return {"roster_id": rid, "players": players}


CUTOFF = 1000   # reverse anything with ts > 1000


def test_no_transactions_after_cutoff_returns_current():
    cur = [_roster(1, ["a", "b"]), _roster(2, ["c"])]
    txns = [{"adds": {"z": 1}, "status_updated": 500}]   # before cutoff — kept
    out = reconstruct_rosters_as_of(cur, txns, CUTOFF)
    assert out == {"1": {"a", "b"}, "2": {"c"}}


def test_reverse_a_waiver_add():
    # Team 1 currently has a, b; b was added after the cutoff -> wasn't there.
    cur = [_roster(1, ["a", "b"])]
    txns = [{"adds": {"b": 1}, "status_updated": 2000}]
    out = reconstruct_rosters_as_of(cur, txns, CUTOFF)
    assert out == {"1": {"a"}}


def test_reverse_a_drop_restores_player():
    # Team 1 currently has a; it dropped c after the cutoff -> c was on it before.
    cur = [_roster(1, ["a"])]
    txns = [{"drops": {"c": 1}, "status_updated": 2000}]
    out = reconstruct_rosters_as_of(cur, txns, CUTOFF)
    assert out == {"1": {"a", "c"}}


def test_reverse_a_trade():
    # A traded x to B for y. Now A has y, B has x. Before: A had x, B had y.
    cur = [_roster("A", ["y"]), _roster("B", ["x"])]
    txns = [{"adds": {"x": "B", "y": "A"}, "drops": {"x": "A", "y": "B"},
             "status_updated": 3000}]
    out = reconstruct_rosters_as_of(cur, txns, CUTOFF)
    assert out == {"A": {"x"}, "B": {"y"}}


def test_add_then_trade_resolves_newest_first():
    # Window: (older) waiver-add p to A, then (newer) trade p from A to B.
    # Now p is on B. As of before both, p was a free agent (on neither).
    cur = [_roster("A", []), _roster("B", ["p"])]
    txns = [
        {"adds": {"p": "A"}, "status_updated": 2000},                     # older
        {"adds": {"p": "B"}, "drops": {"p": "A"}, "status_updated": 3000},  # newer
    ]
    out = reconstruct_rosters_as_of(cur, txns, CUTOFF)
    assert out == {"A": set(), "B": set()}


def test_unknown_team_or_missing_players_are_safe():
    cur = [_roster(1, ["a"])]
    # add references a roster we don't have; drop references an unknown player id
    txns = [{"adds": {"a": 1, "q": 99}, "drops": {"z": 1}, "status_updated": 2000}]
    out = reconstruct_rosters_as_of(cur, txns, CUTOFF)
    assert out == {"1": {"z"}}   # 'a' removed (add reversed), 'z' restored, 99 skipped
