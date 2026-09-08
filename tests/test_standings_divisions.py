"""Division-aware standings: grouping, seeding, and render hooks."""
import pandas as pd
import pytest

from utils.standings_divisions import (
    active_divisions,
    assign_playoff_seeds,
    division_name_map,
    playoff_seed_order,
    resolve_divisions,
    roster_division_map,
)
from utils.playoff_picture import compute_playoff_picture, CLINCHED


def test_roster_division_map_and_names():
    rosters = [
        {"roster_id": 1, "settings": {"division": 1}, "metadata": {"division_name": "East"}},
        {"roster_id": 2, "settings": {"division": 1}},
        {"roster_id": "3", "settings": {"division": 2}, "metadata": {"division_name": "West"}},
        {"roster_id": 4, "settings": {}},
    ]
    assert roster_division_map(rosters) == {1: 1, 2: 1, 3: 2}
    names = division_name_map(
        {"metadata": {"division_1": "Atlantic"}},
        [1, 2],
        rosters,
    )
    # League metadata wins over roster labels.
    assert names[1] == "Atlantic"
    assert names[2] == "West"


def test_active_divisions_requires_two_assignments():
    rosters = [
        {"roster_id": 1, "settings": {"division": 1}},
        {"roster_id": 2, "settings": {"division": 1}},
    ]
    assert active_divisions({"divisions": 2}, rosters) is None
    rosters[1]["settings"]["division"] = 2
    info = active_divisions({"divisions": 2}, rosters)
    assert info and info["ids"] == [1, 2]
    # Explicit 0/1 in settings keeps standings flat.
    assert active_divisions({"divisions": 0}, rosters) is None


def test_playoff_seeds_division_winners_before_wild_cards():
    # Weak division champion (8 wins) still seeds ahead of a 9-win runner-up.
    teams = [
        {"wins": 10, "pf": 1200, "division": 1},  # A — div1 winner
        {"wins": 9,  "pf": 1100, "division": 1},  # B — wild card by record
        {"wins": 8,  "pf": 1000, "division": 2},  # C — div2 winner
        {"wins": 2,  "pf": 600,  "division": 2},  # D
    ]
    seeds = assign_playoff_seeds(teams)
    # Order: A (div winner, best), C (div winner), B (WC), D
    assert seeds == [1, 3, 2, 4]
    order = playoff_seed_order(teams)
    assert [teams[i]["wins"] for i in order] == [10, 8, 9, 2]


def test_resolve_divisions_from_ctx():
    ctx = {
        "league_settings": {"divisions": 2},
        "league": {"metadata": {"division_1": "North", "division_2": "South"}},
        "rosters": [
            {"roster_id": 1, "settings": {"division": 1}},
            {"roster_id": 2, "settings": {"division": 2}},
        ],
    }
    info = resolve_divisions(ctx)
    assert info["names"] == {1: "North", 2: "South"}


def test_render_standings_splits_by_division():
    pytest.importorskip("flask")
    import app as appmod

    rows = [
        {"owner": "Alpha", "Wins": 10, "Losses": 2, "Ties": 0, "PF": 1400, "PA": 1100,
         "Streak": "W3", "avatar": "", "past_sos": 100.0, "ros_sos": 100.0, "Win%": 0.83},
        {"owner": "Bravo", "Wins": 7, "Losses": 5, "Ties": 0, "PF": 1200, "PA": 1150,
         "Streak": "L1", "avatar": "", "past_sos": 100.0, "ros_sos": 100.0, "Win%": 0.58},
        {"owner": "Charlie", "Wins": 9, "Losses": 3, "Ties": 0, "PF": 1300, "PA": 1120,
         "Streak": "W2", "avatar": "", "past_sos": 100.0, "ros_sos": 100.0, "Win%": 0.75},
        {"owner": "Delta", "Wins": 4, "Losses": 8, "Ties": 0, "PF": 1000, "PA": 1300,
         "Streak": "L2", "avatar": "", "past_sos": 100.0, "ros_sos": 100.0, "Win%": 0.33},
    ]
    df = pd.DataFrame(rows)
    o2r = {"Alpha": "1", "Bravo": "2", "Charlie": "3", "Delta": "4"}
    divisions = {
        "by_rid": {1: 1, 2: 1, 3: 2, 4: 2},
        "names": {1: "East", 2: "West"},
        "ids": [1, 2],
        "count": 2,
    }
    html = appmod.render_standings(
        df, length=4, owner_to_rid=o2r, divisions=divisions,
    )
    assert 'data-divisions="1"' in html
    assert "st-div-row" in html
    assert "st-div-head" in html
    assert "East" in html and "West" in html
    assert "3 teams" not in html  # 2 teams each in this fixture
    assert "2 teams" in html
    assert "st-div-lead" in html
    # East block should appear before West (sorted by division id).
    assert html.index("East") < html.index("West")
    # Flat leagues stay flat.
    flat = appmod.render_standings(df, length=4, owner_to_rid=o2r, divisions=None)
    assert "st-div-row" not in flat
    assert 'data-divisions="1"' not in flat
    assert "st-div-lead" not in flat


def test_build_standings_map_uses_division_seeds():
    pytest.importorskip("flask")
    from dashboard_services.service import build_standings_map

    df = pd.DataFrame([
        {"owner": "A", "Wins": 10, "PF": 1200, "PA": 1000, "Ties": 0},
        {"owner": "B", "Wins": 9,  "PF": 1100, "PA": 1000, "Ties": 0},
        {"owner": "C", "Wins": 8,  "PF": 1000, "PA": 1000, "Ties": 0},
        {"owner": "D", "Wins": 2,  "PF": 600,  "PA": 1000, "Ties": 0},
    ])
    roster_map = {1: "A", 2: "B", 3: "C", 4: "D"}
    # Without divisions: overall record order.
    assert build_standings_map(df, roster_map) == {1: 1, 2: 2, 3: 3, 4: 4}
    # With divisions: C (div2 winner at 8-?) seeds ahead of B (9 wins, runner-up).
    seeds = build_standings_map(df, roster_map, division_by_rid={1: 1, 2: 1, 3: 2, 4: 2})
    assert seeds == {1: 1, 3: 2, 2: 3, 4: 4}


def test_playoff_picture_division_seeding():
    teams = [
        {"id": 1, "name": "A", "wins": 10, "losses": 2, "pf": 1200, "division": 1},
        {"id": 2, "name": "B", "wins": 9,  "losses": 3, "pf": 1100, "division": 1},
        {"id": 3, "name": "C", "wins": 8,  "losses": 4, "pf": 1000, "division": 2},
        {"id": 4, "name": "D", "wins": 2,  "losses": 10, "pf": 600, "division": 2},
    ]
    res = compute_playoff_picture(teams, playoff_spots=2, total_regular_weeks=14)
    assert [r["name"] for r in res] == ["A", "C", "B", "D"]
    # Season nearly over with clear floors — A and C have clinched their
    # division berths (2 spots = both winners).
    by_name = {r["name"]: r for r in res}
    assert by_name["A"]["seed"] == 1
    assert by_name["C"]["seed"] == 2
    assert by_name["D"]["status"] != CLINCHED
