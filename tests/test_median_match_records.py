"""Median-match (Sleeper league_average_match) records.

When a league plays a weekly game against the league median, the official
record includes that game. These tests cover the detection flag and the
median game folded into computed records (standings + recap snapshot).
"""
import pytest

pd = pytest.importorskip("pandas")

from utils.standings import median_match_enabled, median_game_outcomes
from dashboard_services.service import _compute_team_records
from dashboard_services.pages.recap_page import _recap_standings_rows


def _fixture_df():
    # 4 teams, 2 weeks.
    # Week 1: A 100 vs B 90, C 80 vs D 70 -> median 85: A W, B W, C L, D L.
    # Week 2: A 60 vs B 110, C 95 vs D 95 -> median 95: A L, B W, C T, D T.
    rows = [
        {"week": 1, "matchup_id": 1, "roster_id": "1", "owner": "A", "points": 100.0},
        {"week": 1, "matchup_id": 1, "roster_id": "2", "owner": "B", "points": 90.0},
        {"week": 1, "matchup_id": 2, "roster_id": "3", "owner": "C", "points": 80.0},
        {"week": 1, "matchup_id": 2, "roster_id": "4", "owner": "D", "points": 70.0},
        {"week": 2, "matchup_id": 3, "roster_id": "1", "owner": "A", "points": 60.0},
        {"week": 2, "matchup_id": 3, "roster_id": "2", "owner": "B", "points": 110.0},
        {"week": 2, "matchup_id": 4, "roster_id": "3", "owner": "C", "points": 95.0},
        {"week": 2, "matchup_id": 4, "roster_id": "4", "owner": "D", "points": 95.0},
    ]
    return pd.DataFrame(rows)


def test_median_match_enabled_truthy_forms():
    assert median_match_enabled({"league_average_match": 1})
    assert median_match_enabled({"league_average_match": "1"})
    assert median_match_enabled({"league_average_match": True})


def test_median_match_enabled_falsy_and_garbage():
    assert not median_match_enabled({"league_average_match": 0})
    assert not median_match_enabled({"league_average_match": "0"})
    assert not median_match_enabled({"league_average_match": None})
    assert not median_match_enabled({})
    assert not median_match_enabled(None)
    assert not median_match_enabled("garbage")
    assert not median_match_enabled({"league_average_match": "bogus"})


def test_median_game_outcomes_per_week():
    out = median_game_outcomes(_fixture_df())
    assert len(out) == 8
    by_owner_week = {(o["owner"], o["week"]): o["outcome"] for o in out}
    assert by_owner_week[("A", 1)] == "W"
    assert by_owner_week[("B", 1)] == "W"
    assert by_owner_week[("C", 1)] == "L"
    assert by_owner_week[("D", 1)] == "L"
    assert by_owner_week[("A", 2)] == "L"
    assert by_owner_week[("B", 2)] == "W"
    assert by_owner_week[("C", 2)] == "T"
    assert by_owner_week[("D", 2)] == "T"
    assert all("roster_id" in o and "week" in o for o in out)


def _rec_by_owner(df):
    return {r["owner"]: r for r in df.to_dict("records")}


def test_compute_team_records_flag_off_unchanged():
    recs = _rec_by_owner(_compute_team_records(_fixture_df()))
    # Head-to-head only: A 1-1, B 1-1, C 1-0-1, D 0-1-1.
    assert (recs["A"]["Wins"], recs["A"]["Losses"], recs["A"]["Ties"], recs["A"]["G"]) == (1, 1, 0, 2)
    assert (recs["B"]["Wins"], recs["B"]["Losses"], recs["B"]["Ties"], recs["B"]["G"]) == (1, 1, 0, 2)
    assert (recs["C"]["Wins"], recs["C"]["Losses"], recs["C"]["Ties"], recs["C"]["G"]) == (1, 0, 1, 2)
    assert (recs["D"]["Wins"], recs["D"]["Losses"], recs["D"]["Ties"], recs["D"]["G"]) == (0, 1, 1, 2)


def test_compute_team_records_flag_on_adds_median_games():
    recs = _rec_by_owner(_compute_team_records(_fixture_df(), median_match=True))
    # A: 1-1 + W,L -> 2-2 (G 4, .500). B: 1-1 + W,W -> 3-1 (G 4, .750).
    # C: 1-0-1 + L,T -> 1-1-2 (G 4, .500). D: 0-1-1 + L,T -> 0-2-2 (G 4, .250).
    assert (recs["A"]["Wins"], recs["A"]["Losses"], recs["A"]["Ties"], recs["A"]["G"]) == (2, 2, 0, 4)
    assert (recs["B"]["Wins"], recs["B"]["Losses"], recs["B"]["Ties"], recs["B"]["G"]) == (3, 1, 0, 4)
    assert (recs["C"]["Wins"], recs["C"]["Losses"], recs["C"]["Ties"], recs["C"]["G"]) == (1, 1, 2, 4)
    assert (recs["D"]["Wins"], recs["D"]["Losses"], recs["D"]["Ties"], recs["D"]["G"]) == (0, 2, 2, 4)
    assert recs["A"]["Win%"] == pytest.approx(0.5)
    assert recs["B"]["Win%"] == pytest.approx(0.75)
    assert recs["C"]["Win%"] == pytest.approx(0.5)
    assert recs["D"]["Win%"] == pytest.approx(0.25)


def _recap_df():
    df = _fixture_df()
    df["finalized"] = True
    # Head-to-head opponent points for the recap snapshot's win/tie columns.
    opp = {
        ("1", 1): 90.0, ("2", 1): 100.0, ("3", 1): 70.0, ("4", 1): 80.0,
        ("1", 2): 110.0, ("2", 2): 60.0, ("3", 2): 95.0, ("4", 2): 95.0,
    }
    df["points_against"] = [opp[(r["roster_id"], r["week"])] for r in df.to_dict("records")]
    return df


def test_recap_standings_rows_flag_off():
    rows = {r["rid"]: r for r in _recap_standings_rows(_recap_df(), {}, False)}
    assert (rows["1"]["wins"], rows["1"]["losses"], rows["1"]["ties"]) == (1, 1, 0)
    assert rows["1"]["record"] == "1-1"


def test_recap_standings_rows_flag_on_adds_median_wins():
    rows = {r["rid"]: r for r in _recap_standings_rows(_recap_df(), {}, False, median_match=True)}
    assert (rows["1"]["wins"], rows["1"]["losses"], rows["1"]["ties"]) == (2, 2, 0)
    assert rows["1"]["record"] == "2-2"
    assert (rows["2"]["wins"], rows["2"]["losses"], rows["2"]["ties"]) == (3, 1, 0)
    assert (rows["3"]["wins"], rows["3"]["losses"], rows["3"]["ties"]) == (1, 1, 2)
    assert rows["3"]["record"] == "1-1-2"
    # PF is scoring only; the median game adds no points.
    assert rows["1"]["pf"] == pytest.approx(160.0)
