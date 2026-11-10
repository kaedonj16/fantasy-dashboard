"""Playoff Path weekly impact: the if-win / if-lose conditionals.

The Path tab's Now / If Win / If Lose numbers must come from ONE Monte
Carlo run (the same run the Playoff Odds tab renders), partitioned on the
first remaining week's head-to-head results. These tests pin the
partition math in ``_pack_mc_results`` with hand-built sim arrays (no RNG:
the winner masks and final standings per sim are spelled out), and the
capture wiring in ``_standings_from_score_map`` with a deterministic
score map, mirroring tests/test_playoff_odds_median_match.py.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from data_building.simulate_playoff_odds import (
    _pack_mc_results,
    _standings_from_score_map,
)


def _team(rid, name, wins, losses, pf):
    return {"roster_id": rid, "name": name, "wins": wins, "losses": losses,
            "ties": 0, "avg": 100.0, "std": 10.0, "pf": float(pf)}


def _four_team_pack():
    """4 teams, 3 playoff spots (2 byes), 4 sims.

    First week: A vs B, C vs D. Winners per sim: A,C / B,D / A,D / B,C.
    Final wins / PF per sim are built so C and D are true bubble teams:
    each makes the playoffs exactly in the sims where it won its
    first-week game, and in no others.
    """
    teams = [_team(1, "A", 6, 4, 1000), _team(2, "B", 5, 5, 950),
             _team(3, "C", 5, 5, 900), _team(4, "D", 4, 6, 850)]
    # rows = sims, cols = A, B, C, D
    wins = np.array([
        [7, 5, 6, 4],
        [6, 6, 5, 6],
        [7, 5, 5, 6],
        [6, 7, 6, 4],
    ], dtype=np.float32)
    pf = np.array([
        [750, 540, 630, 420],
        [650, 640, 530, 620],
        [750, 540, 530, 620],
        [650, 740, 630, 420],
    ], dtype=np.float32)
    won = np.array([
        [True, False, True, False],
        [False, True, False, True],
        [True, False, False, True],
        [False, True, True, False],
    ])
    rows = _pack_mc_results(
        teams, wins, pf, np.zeros_like(wins), np.full(4, 12.0),
        playoff_teams=3, n_sims=4,
        first_week=11, first_games=[(0, 1), (2, 3)],
        first_won=won, first_lost=~won,
    )
    return {r["roster_id"]: r for r in rows}


def test_bubble_team_conditionals_bracket_the_headline():
    by_id = _four_team_pack()
    c = by_id[3]
    # C is in exactly half the sims, and in precisely the ones it won.
    assert c["playoff_pct"] == 50.0
    assert c["playoff_if_win"] == 99.9
    assert c["playoff_if_lose"] == 0.1
    assert c["playoff_if_win"] > c["playoff_pct"] > c["playoff_if_lose"]
    assert c["swing"] == 99.8
    assert c["impact_week"] == 11
    d = by_id[4]
    assert d["playoff_pct"] == 50.0
    assert d["playoff_if_win"] == 99.9
    assert d["playoff_if_lose"] == 0.1
    assert d["swing"] == 99.8


def test_bye_conditionals_use_the_same_partitions():
    by_id = _four_team_pack()
    a = by_id[1]
    # 3 playoff spots means a single bye (top seed). A is top seed in
    # sims 0, 1, 2: both its win sims (0, 2) and one of its loss sims (1).
    assert a["bye_pct"] == 75.0
    assert a["bye_if_win"] == 99.9
    assert a["bye_if_lose"] == 50.0
    assert a["bye_if_win"] > a["bye_pct"] > a["bye_if_lose"]


def test_root_for_conditions_on_other_games_only():
    by_id = _four_team_pack()
    c = by_id[3]
    # C's own game (vs D) is excluded; only A vs B remains.
    assert len(c["root_for"]) == 1
    entry = c["root_for"][0]
    assert (entry["a"], entry["b"]) == (1, 2)
    # C makes it in one of the two A-won sims and one of the two B-won.
    assert entry["playoff_if_a"] == 50.0
    assert entry["playoff_if_b"] == 50.0
    # C is never the top seed, so its bye partitions are empty of byes.
    assert entry["bye_if_a"] == 0.1
    assert entry["bye_if_b"] == 0.1
    a = by_id[1]
    assert len(a["root_for"]) == 1
    assert (a["root_for"][0]["a"], a["root_for"][0]["b"]) == (3, 4)
    # A's bye (top seed) came in one of the two C-won sims and both of
    # the two D-won sims.
    assert a["root_for"][0]["bye_if_a"] == 50.0
    assert a["root_for"][0]["bye_if_b"] == 99.9


def test_bye_week_team_gets_nulls_but_keeps_root_for():
    teams = [_team(1, "A", 5, 5, 1000), _team(2, "B", 5, 5, 950),
             _team(3, "C", 5, 5, 900), _team(4, "D", 5, 5, 850),
             _team(5, "E", 5, 5, 800)]
    wins = np.array([[6, 5, 5, 4, 5], [5, 6, 4, 5, 5]], dtype=np.float32)
    pf = np.array([[650, 540, 530, 420, 510],
                   [550, 640, 430, 520, 510]], dtype=np.float32)
    won = np.array([[True, False, True, False, False],
                    [False, True, False, True, False]])
    lost = np.array([[False, True, False, True, False],
                     [True, False, True, False, False]])
    rows = _pack_mc_results(
        teams, wins, pf, np.zeros_like(wins), np.full(5, 12.0),
        playoff_teams=3, n_sims=2,
        first_week=11, first_games=[(0, 1), (2, 3)],
        first_won=won, first_lost=lost,
    )
    e = {r["roster_id"]: r for r in rows}[5]
    # No first-week game: no honest split exists, so nulls, not zeros.
    assert e["playoff_if_win"] is None
    assert e["playoff_if_lose"] is None
    assert e["bye_if_win"] is None
    assert e["swing"] is None
    # But E still gets root-for partitions for both games played.
    assert len(e["root_for"]) == 2
    ab = next(g for g in e["root_for"] if (g["a"], g["b"]) == (1, 2))
    assert ab["playoff_if_a"] == 0.1  # E missed in both sims; clamped
    assert ab["playoff_if_b"] == 0.1
    assert ab["bye_if_a"] is not None


def test_no_bye_format_nulls_bye_conditionals():
    # 4 playoff spots means no byes: the bye splits must be null, never a
    # clamped 0.1 dressed up as a real partition.
    teams = [_team(1, "A", 6, 4, 1000), _team(2, "B", 5, 5, 950),
             _team(3, "C", 5, 5, 900), _team(4, "D", 4, 6, 850)]
    wins = np.array([[7, 5, 6, 4], [6, 6, 5, 6]], dtype=np.float32)
    pf = np.array([[750, 540, 630, 420], [650, 640, 530, 620]],
                  dtype=np.float32)
    won = np.array([[True, False, True, False],
                    [False, True, False, True]])
    rows = _pack_mc_results(
        teams, wins, pf, np.zeros_like(wins), np.full(4, 12.0),
        playoff_teams=4, n_sims=2,
        first_week=11, first_games=[(0, 1), (2, 3)],
        first_won=won, first_lost=~won,
    )
    for r in rows:
        assert r["bye_if_win"] is None
        assert r["bye_if_lose"] is None
        assert r["playoff_if_win"] is not None
        for g in r["root_for"]:
            assert g["bye_if_a"] is None
            assert g["bye_if_b"] is None


def test_score_map_capture_threads_first_week_results():
    # Week 11 is fixed across sims (T2 always beats T1, T3 always beats
    # T4), so each team's first-week partition is either every sim or
    # none, and the empty side must come back null. Week 12 alternates
    # winners by sim parity so the run is not the pack's fully
    # deterministic shortcut and the headline clamps like production.
    teams = [
        {"roster_id": 1, "name": "T1", "wins": 5, "losses": 5, "ties": 0,
         "avg": 80.0, "std": 1.0, "pf": 900.0},
        {"roster_id": 2, "name": "T2", "wins": 5, "losses": 5, "ties": 0,
         "avg": 120.0, "std": 1.0, "pf": 950.0},
        {"roster_id": 3, "name": "T3", "wins": 6, "losses": 4, "ties": 0,
         "avg": 110.0, "std": 1.0, "pf": 1000.0},
        {"roster_id": 4, "name": "T4", "wins": 4, "losses": 6, "ties": 0,
         "avg": 90.0, "std": 1.0, "pf": 850.0},
    ]
    matchups = {11: [(1, 2), (3, 4)], 12: [(1, 2), (3, 4)]}
    even = np.arange(200) % 2 == 0
    mirror = {1: 120.0, 2: 80.0, 3: 90.0, 4: 110.0}
    score_map = {
        11: {t["roster_id"]: np.full(200, t["avg"], dtype=np.float32)
             for t in teams},
        12: {t["roster_id"]: np.where(even, t["avg"], mirror[t["roster_id"]]).astype(np.float32)
             for t in teams},
    }
    rows = _standings_from_score_map(
        teams, matchups, score_map, 3, 200,
        np.full(4, 2.0, dtype=np.float32),  # remaining games per team
    )
    by_id = {r["roster_id"]: r for r in rows}
    assert all(r["impact_week"] == 11 for r in rows)
    t2 = by_id[2]  # won its first-week game in every sim
    assert t2["playoff_if_win"] == t2["playoff_pct"]
    assert t2["playoff_if_lose"] is None
    assert t2["swing"] is None
    t1 = by_id[1]  # lost it in every sim
    assert t1["playoff_if_win"] is None
    assert t1["playoff_if_lose"] == t1["playoff_pct"]
    # Root-for excludes the team's own game and keeps the other one;
    # conditioning on a fixed-result game reproduces the headline, and
    # the never-occurring winner's partition is honestly null.
    assert len(t1["root_for"]) == 1
    entry = t1["root_for"][0]
    assert (entry["a"], entry["b"]) == (3, 4)
    assert entry["playoff_if_a"] == t1["playoff_pct"]
    assert entry["playoff_if_b"] is None
