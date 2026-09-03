"""Synthetic Week-1 matchups when the platform schedule feed is empty."""
from utils.matchup_schedule import resolve_matchup_week, synthetic_week_matchups


def test_synthetic_week_matchups_pairs_every_team():
    rosters = [{"roster_id": i, "players": [f"p{i}"], "starters": []} for i in range(1, 11)]
    rows = synthetic_week_matchups(rosters, week=1)
    assert len(rows) == 10  # 5 matchups × 2 sides
    mids = {r["matchup_id"] for r in rows}
    assert mids == {1, 2, 3, 4, 5}
    rids = sorted(r["roster_id"] for r in rows)
    assert rids == list(range(1, 11))


def test_synthetic_week_matchups_deterministic_across_calls():
    rosters = [{"roster_id": i, "players": [], "starters": []} for i in (3, 1, 2, 4)]
    a = synthetic_week_matchups(rosters, week=2)
    b = synthetic_week_matchups(list(reversed(rosters)), week=2)
    assert [(r["matchup_id"], r["roster_id"]) for r in a] == [
        (r["matchup_id"], r["roster_id"]) for r in b
    ]


def test_synthetic_week_matchups_needs_two_teams():
    assert synthetic_week_matchups([{"roster_id": 1}], week=1) == []
    assert synthetic_week_matchups([], week=1) == []


def test_resolve_matchup_week_skips_nfl_week_zero():
    preview = [{"matchup_id": 1, "left": {}, "right": {}}]
    assert resolve_matchup_week(0, {1: preview}) == 1
    assert resolve_matchup_week(None, {1: preview}) == 1
    assert resolve_matchup_week(1, {1: preview}) == 1
    assert resolve_matchup_week(0, {}) == 1
    assert resolve_matchup_week(2, {2: preview}) == 2
