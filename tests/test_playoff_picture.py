"""Seeded, hand-verified scenarios for the playoff-picture engine."""
from utils.playoff_picture import (
    compute_playoff_picture, bye_count,
    BYE, CLINCHED, IN, BUBBLE, ELIMINATED,
)


def _by_name(rows):
    return {r["name"]: r for r in rows}


def test_bye_count_standard_brackets():
    assert bye_count(4) == 0
    assert bye_count(6) == 2
    assert bye_count(8) == 0
    assert bye_count(7) == 1
    assert bye_count(3) == 1
    assert bye_count(2) == 0


def test_clinch_eliminate_bubble_4team():
    # 14-game regular season, 3 weeks left (each team has played 11).
    teams = [
        {"id": 1, "name": "A", "wins": 11, "losses": 0, "pf": 1400},
        {"id": 2, "name": "B", "wins": 9,  "losses": 2, "pf": 1300},
        {"id": 3, "name": "C", "wins": 7,  "losses": 4, "pf": 1200},
        {"id": 4, "name": "D", "wins": 6,  "losses": 5, "pf": 1150},
        {"id": 5, "name": "E", "wins": 4,  "losses": 7, "pf": 1050},
        {"id": 6, "name": "F", "wins": 2,  "losses": 9, "pf": 900},
    ]
    res = compute_playoff_picture(teams, playoff_spots=4, total_regular_weeks=14)
    r = _by_name(res)

    # Seeding follows wins.
    assert [x["name"] for x in res] == ["A", "B", "C", "D", "E", "F"]

    assert r["A"]["status"] == CLINCHED          # only B can reach 11 wins
    assert r["B"]["status"] == CLINCHED          # only A/C/D can reach 9 → top-4 locked
    assert r["C"]["status"] == IN                # holds a spot, not math-clinched
    assert r["D"]["status"] == IN
    assert r["D"]["controls_own_fate"] is True   # winning out guarantees D a berth
    assert r["D"]["scenario"] == "Win out and you're in."
    assert r["E"]["status"] == BUBBLE            # 5th, still alive
    assert r["E"]["controls_own_fate"] is False
    assert r["F"]["status"] == ELIMINATED        # 4 teams already above its ceiling
    assert r["F"]["scenario"] is None


def test_byes_and_elimination_6team():
    # 6 playoff spots, top 2 get byes. 14-game season, 2 weeks left.
    teams = [
        {"id": 1, "name": "T1", "wins": 12, "losses": 0, "pf": 1600},
        {"id": 2, "name": "T2", "wins": 11, "losses": 1, "pf": 1550},
        {"id": 3, "name": "T3", "wins": 7,  "losses": 5, "pf": 1300},
        {"id": 4, "name": "T4", "wins": 7,  "losses": 5, "pf": 1280},
        {"id": 5, "name": "T5", "wins": 6,  "losses": 6, "pf": 1250},
        {"id": 6, "name": "T6", "wins": 6,  "losses": 6, "pf": 1240},
        {"id": 7, "name": "T7", "wins": 5,  "losses": 7, "pf": 1200},
        {"id": 8, "name": "T8", "wins": 3,  "losses": 9, "pf": 1100},
    ]
    res = compute_playoff_picture(teams, playoff_spots=6, total_regular_weeks=14)
    r = _by_name(res)

    assert bye_count(6) == 2
    assert r["T1"]["status"] == BYE
    assert r["T2"]["status"] == BYE
    assert r["T8"]["status"] == ELIMINATED       # 6 teams above its ceiling of 5
    # T7 can still reach 7 wins; only T1/T2 are locked above it → alive.
    assert r["T7"]["status"] == BUBBLE
    # Nobody outside the top 6 is wrongly marked clinched.
    assert all(x["status"] != BYE for x in res if x["seed"] > 2)


def test_season_over_resolves_to_in_or_out():
    # No games left: the top-N are simply in, the rest out.
    teams = [
        {"id": i, "name": f"T{i}", "wins": 10 - i, "losses": 4 + i, "pf": 1000 - 10 * i}
        for i in range(6)
    ]
    res = compute_playoff_picture(teams, playoff_spots=3, total_regular_weeks=14)
    for x in res:
        assert x["games_left"] == 0
        if x["seed"] <= 3:
            assert x["status"] in (CLINCHED, BYE)
        else:
            assert x["status"] == ELIMINATED
        assert x["scenario"] is None


def test_fewer_teams_than_spots():
    teams = [
        {"id": 1, "name": "A", "wins": 3, "losses": 1, "pf": 400},
        {"id": 2, "name": "B", "wins": 1, "losses": 3, "pf": 300},
    ]
    res = compute_playoff_picture(teams, playoff_spots=6, total_regular_weeks=14)
    # Everyone is in when there are fewer teams than spots; nothing crashes.
    assert all(x["status"] != ELIMINATED for x in res)


def test_pf_breaks_seed_ties():
    teams = [
        {"id": 1, "name": "Low", "wins": 5, "losses": 5, "pf": 1000},
        {"id": 2, "name": "High", "wins": 5, "losses": 5, "pf": 1200},
    ]
    res = compute_playoff_picture(teams, playoff_spots=1, total_regular_weeks=14)
    assert res[0]["name"] == "High"   # higher PF seeds first on equal records
