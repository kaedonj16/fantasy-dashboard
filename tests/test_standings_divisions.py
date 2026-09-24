"""Division-aware standings: grouping, seeding, and render hooks.

Pure helper tests run in lightweight CI (pytest only). Flask/pandas-backed
render + standings_map checks importorskip so the collection path stays clean.
"""
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
    pd = pytest.importorskip("pandas")
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
    pd = pytest.importorskip("pandas")
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


def test_format_record_with_division():
    from utils.standings_divisions import format_record
    assert format_record(2, 1) == "2-1"
    assert format_record(2, 1, 0, (2, 0, 0)) == "2-1 (2-0)"
    assert format_record(2, 1, 1, (1, 0, 1)) == "2-1-1 (1-0-1)"
    # No parenthetical when divisions are off.
    assert format_record(2, 1, 0, None) == "2-1"


def _div_weekly_frame(pd):
    # 4 teams, 2 divisions; week 1: 1v2 (div game), 3v4 (div game);
    # week 2: 1v3 (cross-div), 2v4 (cross-div).
    rows = [
        # week, matchup_id, roster_id, points, points_against
        (1, 101, 1, 100.0, 90.0), (1, 101, 2, 90.0, 100.0),
        (1, 102, 3, 80.0, 80.0),  (1, 102, 4, 80.0, 80.0),
        (2, 201, 1, 70.0, 110.0), (2, 201, 3, 110.0, 70.0),
        (2, 202, 2, 95.0, 85.0),  (2, 202, 4, 85.0, 95.0),
    ]
    df = pd.DataFrame(rows, columns=["week", "matchup_id", "roster_id", "points", "points_against"])
    df["finalized"] = True
    df["owner"] = df["roster_id"].map({1: "A", 2: "B", 3: "C", 4: "D"})
    return df


def test_division_records_counts_only_division_games():
    pd = pytest.importorskip("pandas")
    from utils.standings_divisions import division_records
    by_rid = {1: 1, 2: 1, 3: 2, 4: 2}
    recs = division_records(_div_weekly_frame(pd), by_rid)
    # Team 1: beat 2 (div) in wk1, lost to 3 (cross-div) in wk2 -> 1-0 div.
    assert recs[1] == (1, 0, 0)
    assert recs[2] == (0, 1, 0)
    # Team 3: tied 4 (div) in wk1 -> 0-0-1 div (cross-div win over 1 excluded).
    assert recs[3] == (0, 0, 1)
    assert recs[4] == (0, 0, 1)
    # Empty / missing columns are safe.
    assert division_records(pd.DataFrame(), by_rid) == {}


def test_render_standings_shows_division_record():
    pytest.importorskip("flask")
    pd = pytest.importorskip("pandas")
    import app as appmod

    rows = [
        {"owner": "A", "Wins": 1, "Losses": 1, "Ties": 0, "PF": 170, "PA": 200,
         "Streak": "", "avatar": "", "Win%": 0.5},
        {"owner": "B", "Wins": 1, "Losses": 1, "Ties": 0, "PF": 185, "PA": 185,
         "Streak": "", "avatar": "", "Win%": 0.5},
        {"owner": "C", "Wins": 1, "Losses": 0, "Ties": 1, "PF": 190, "PA": 150,
         "Streak": "", "avatar": "", "Win%": 0.75},
        {"owner": "D", "Wins": 0, "Losses": 1, "Ties": 1, "PF": 165, "PA": 175,
         "Streak": "", "avatar": "", "Win%": 0.25},
    ]
    df = pd.DataFrame(rows)
    o2r = {"A": "1", "B": "2", "C": "3", "D": "4"}
    divisions = {"by_rid": {1: 1, 2: 1, 3: 2, 4: 2}, "names": {1: "East", 2: "West"},
                 "ids": [1, 2], "count": 2}
    html = appmod.render_standings(
        df, length=4, owner_to_rid=o2r, divisions=divisions,
        detailed_df=_div_weekly_frame(pd),
    )
    assert "1-1 (1-0)" in html  # A: 1-1 overall, 1-0 in division
    assert "1-0-1 (0-0-1)" in html  # C: tie was a division game
    # Flat leagues keep the plain record.
    flat = appmod.render_standings(df, length=4, owner_to_rid=o2r, divisions=None)
    assert "(1-0)" not in flat
    assert "1-1<" in flat or ">1-1<" in flat


def test_render_standings_compact_shows_division_record():
    pytest.importorskip("flask")
    pd = pytest.importorskip("pandas")
    import app as appmod

    rows = [
        {"owner": "A", "Wins": 1, "Losses": 1, "Ties": 0, "PF": 170, "PA": 200, "Rank": 1},
        {"owner": "B", "Wins": 1, "Losses": 1, "Ties": 0, "PF": 185, "PA": 185, "Rank": 2},
    ]
    df = pd.DataFrame(rows)
    o2r = {"A": "1", "B": "2"}
    divisions = {"by_rid": {1: 1, 2: 1, 3: 2}, "names": {1: "East", 2: "West"},
                 "ids": [1, 2], "count": 2}
    html = appmod.render_standings_compact(
        df, owner_to_rid=o2r, divisions=divisions,
        div_records={1: (1, 0, 0), 2: (0, 1, 0)},
    )
    assert "1-1 (1-0)" in html
    assert "1-1 (0-1)" in html


def test_division_records_for_ctx_none_without_divisions():
    from utils.standings_divisions import division_records_for_ctx
    pd = pytest.importorskip("pandas")
    # No division info -> None (renderers keep the plain record).
    assert division_records_for_ctx({}) is None
    assert division_records_for_ctx({"df_weekly": pd.DataFrame()}) is None
    # Divisions active -> map, even with an empty frame.
    ctx = {
        "league_settings": {"divisions": 2},
        "rosters": [
            {"roster_id": 1, "settings": {"division": 1}},
            {"roster_id": 2, "settings": {"division": 2}},
        ],
        "df_weekly": pd.DataFrame(),
    }
    assert division_records_for_ctx(ctx) == {}


def test_render_matchup_slide_shows_division_record(monkeypatch):
    pytest.importorskip("flask")
    pytest.importorskip("requests")
    from dashboard_services import matchups as mmod
    monkeypatch.setattr(mmod, "load_teams_index", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "build_offense_rankings", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "load_week_schedule", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "build_team_schedule_lookup", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "_allow_live_game_indicators", lambda *_a, **_k: False)

    def _team(name, rid, record):
        return {
            "name": name, "roster_id": rid, "record": record, "username": "u",
            "avatar": "", "pts_total": None, "starters": [],
        }

    matchup = {"left": _team("Team A", "1", "2-1"), "right": _team("Team B", "2", "1-2")}
    kw = dict(
        status_by_pid={}, projections={}, players={}, teams={},
        team_game_lookup={}, scoring_settings={},
    )
    html = mmod.render_matchup_slide(
        "2026", matchup, w=1, proj_week=0,
        div_records={1: (2, 0, 0), 2: (0, 1, 0)}, **kw)
    assert "2-1 (2-0)" in html
    assert "1-2 (0-1)" in html
    # No div_records -> plain records.
    plain = mmod.render_matchup_slide("2026", matchup, w=1, proj_week=0, **kw)
    assert "2-1 (2-0)" not in plain
    assert "2-1" in plain
