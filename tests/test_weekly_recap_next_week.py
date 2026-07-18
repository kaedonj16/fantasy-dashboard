"""Guards the deterministic 'game of the week' pick in the weekly recap.

The AI only narrates the pick; Python chooses which game leads based on
rankings, closeness, playoff stakes, streaks, a rematch angle, and missing
starters. These tests pin that scoring so the wrong game can't quietly win.

weekly_recap imports pandas at module load, so the whole module is skipped in
the base suite when pandas isn't installed.
"""
import pytest

pd = pytest.importorskip("pandas")

from dashboard_services.ai.weekly_recap import (  # noqa: E402
    _build_next_week_preview,
    _injured_starters,
    _prior_series,
    _render_next_week_html,
)


def _fin_row(week, mid, rid, pts, opp):
    return {"week": week, "matchup_id": mid, "roster_id": rid,
            "points": pts, "points_against": opp, "finalized": True}


def _storyline(rid, team, rank, record="4-2", streak="W1"):
    return {"rid": str(rid), "team": team, "rank_after": rank,
            "record_after": record, "streak": streak}


def _matchup(rid_a, rid_b, starters_a=None, starters_b=None):
    return {
        "left": {"roster_id": str(rid_a), "starters": starters_a or []},
        "right": {"roster_id": str(rid_b), "starters": starters_b or []},
    }


def test_prior_series_counts_only_meetings_between_the_two():
    df = pd.DataFrame([
        _fin_row(1, 10, "1", 120, 100), _fin_row(1, 10, "2", 100, 120),  # 1 beat 2
        _fin_row(2, 11, "1", 90, 110), _fin_row(2, 11, "3", 110, 90),    # unrelated
        _fin_row(3, 12, "1", 105, 130), _fin_row(3, 12, "2", 130, 105),  # 2 beat 1
    ])
    meetings, a_wins, b_wins = _prior_series(df, "1", "2", upto_week=5)
    assert (meetings, a_wins, b_wins) == (2, 1, 1)


def test_injured_starters_flags_out_players_biggest_first():
    starters = [
        {"pid": "p1", "name": "Star RB", "pos": "RB"},
        {"pid": "p2", "name": "Depth WR", "pos": "WR"},
        {"pid": "p3", "name": "Healthy QB", "pos": "QB"},
    ]
    player_index = {
        "p1": {"injury_status": "OUT"},
        "p2": {"injury_status": "DOUBTFUL"},
        "p3": {"injury_status": "ACTIVE"},
    }
    values = {"p1": 8000.0, "p2": 1200.0}
    out = _injured_starters({"starters": starters}, player_index, values)
    assert [p["name"] for p in out] == ["Star RB", "Depth WR"]  # healthy QB excluded, sorted by value
    assert out[0]["status"] == "OUT"


def test_questionable_is_not_treated_as_missing():
    starters = [{"pid": "p1", "name": "Maybe WR", "pos": "WR"}]
    out = _injured_starters({"starters": starters}, {"p1": {"injury_status": "QUESTIONABLE"}}, {})
    assert out == []


def test_marquee_top_ranked_game_beats_a_lopsided_one():
    # Two title contenders (#1 vs #2) should outrank a #1-vs-#10 mismatch.
    df = pd.DataFrame([_fin_row(1, 1, "1", 100, 90), _fin_row(1, 1, "2", 90, 100)])
    storylines = {
        "1": _storyline(1, "Alpha", 1),
        "2": _storyline(2, "Bravo", 2),
        "9": _storyline(9, "India", 10),
    }
    matchups = [_matchup("1", "2"), _matchup("9", "1")]
    preview = _build_next_week_preview(
        df, matchups, storylines, {}, {}, selected_week=6,
        playoff_start=14, playoff_teams=6, num_teams=10,
    )
    assert preview is not None
    assert {preview["game_of_the_week"]["team_a"], preview["game_of_the_week"]["team_b"]} == {"Alpha", "Bravo"}


def test_missing_star_is_surfaced_as_a_reason_and_rendered():
    df = pd.DataFrame([_fin_row(1, 1, "1", 100, 90), _fin_row(1, 1, "2", 90, 100)])
    storylines = {"1": _storyline(1, "Alpha", 1), "2": _storyline(2, "Bravo", 3)}
    starters_b = [{"pid": "b1", "name": "Bijan", "pos": "RB"}]
    matchups = [_matchup("1", "2", starters_b=starters_b)]
    preview = _build_next_week_preview(
        df, matchups, storylines, {"b1": {"injury_status": "OUT"}}, {"b1": 8000.0},
        selected_week=6, playoff_start=14, playoff_teams=6, num_teams=10,
    )
    g = preview["game_of_the_week"]
    assert any("Bijan" in r for r in g["reasons"])
    assert g["injured_b"] and g["injured_b"][0]["status"] == "OUT"

    html_out = _render_next_week_html(preview, "Bravo is down Bijan and it shows.")
    assert "GAME OF THE WEEK" in html_out
    assert "Bijan" in html_out and "Bravo is down Bijan" in html_out


def test_no_matchups_yields_no_preview():
    df = pd.DataFrame([_fin_row(1, 1, "1", 100, 90)])
    assert _build_next_week_preview(df, [], {}, {}, {}, 6, 14, 6, 10) is None
    assert _render_next_week_html(None, "") == ""
