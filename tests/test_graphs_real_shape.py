"""Graphs page regression tests shaped EXACTLY like the real service output.

dashboard_services/service.py builds df_weekly with columns
week / matchup_id / roster_id / owner / points / points_against / avatar /
finalized and NO 'win' column. The older graphs fixture invented a 'win'
column, which masked the empty Standings Bump Chart (seed_series_for
requires 'win') and the silently suppressed Luck scatter (actual wins are
read from 'win'). These fixtures omit 'win' on purpose.
"""
from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("plotly")

OWNERS = [f"Team {i}" for i in range(10)]


def _real_shape_weekly(weeks=8):
    """Paired matchups, points_against filled by the pairing, no 'win'."""
    rows = []
    for wk in range(1, weeks + 1):
        for m in range(5):
            a, b = OWNERS[2 * m], OWNERS[2 * m + 1]
            pa = 95.0 + wk * 2 + m * 7 + (wk % 3)
            pb = 90.0 + wk * 3 + m * 5
            rows.append({"week": wk, "matchup_id": m + 1, "roster_id": 2 * m + 1,
                         "owner": a, "points": pa, "points_against": pb,
                         "avatar": None, "finalized": True})
            rows.append({"week": wk, "matchup_id": m + 1, "roster_id": 2 * m + 2,
                         "owner": b, "points": pb, "points_against": pa,
                         "avatar": None, "finalized": True})
    return pd.DataFrame(rows, columns=["week", "matchup_id", "roster_id",
                                       "owner", "points", "points_against",
                                       "avatar", "finalized"])


def _colors():
    from dashboard_services.pages.graphs_page import owner_color_map
    return owner_color_map(OWNERS)


def test_real_shape_fixture_has_no_win_column():
    df = _real_shape_weekly()
    assert "win" not in df.columns
    assert list(df.columns) == ["week", "matchup_id", "roster_id", "owner",
                                "points", "points_against", "avatar", "finalized"]


def test_bump_chart_has_traces_without_win_column():
    from dashboard_services.pages.graphs_page import _bump_card
    df = _real_shape_weekly()
    figs = {}
    html = _bump_card(df, _colors(), OWNERS[0], figs)
    assert "Standings Bump Chart" in html
    assert "chart-bump" in figs
    fig = json.loads(figs["chart-bump"])
    assert len(fig["data"]) == 10
    for trace in fig["data"]:
        assert trace["x"], "bump trace has no weeks"
        assert trace["y"], "bump trace has no seeds"
    # The caller's frame must not be mutated.
    assert "win" not in df.columns


def test_luck_card_renders_without_win_column():
    from dashboard_services.pages.graphs_page import _luck_card
    df = _real_shape_weekly()
    html = _luck_card(df, OWNERS[0], _colors())
    assert "Performance vs Luck" in html
    assert "luck-quadrant" in html
    assert "win" not in df.columns


def test_luck_delta_card_renders_without_win_column():
    from dashboard_services.pages.graphs_page import _luck_delta_card
    df = _real_shape_weekly()
    html = _luck_delta_card(df, OWNERS[0], _colors())
    assert "Luck Delta" in html
    assert "win" not in df.columns


# ── Scope I: division-aware bump chart ───────────────────────────────────────

DIV_OWNERS = ["E1", "E2", "W1", "W2"]
DIV_MAP = {"E1": 1, "E2": 1, "W1": 2, "W2": 2}
DIV_RIDS = {"E1": 1, "E2": 2, "W1": 3, "W2": 4}


def _division_weekly():
    """2 divisions x 2 teams, 2 weeks, no 'win' column.

    Week 1: W1 beats E1 100-90; W2 beats E2 150-80.
    Week 2: E1 beats W2 95-94; W1 beats E2 160-70.
    Records: W1 2-0 (PF 260), W2 1-1 (PF 244), E1 1-1 (PF 185), E2 0-2.
    Plain wins/PF order after week 2: W1, W2, E1, E2.
    Division-aware (winners first):   W1, E1, W2, E2.
    """
    games = [
        (1, 1, "W1", 100.0, "E1", 90.0),
        (1, 2, "W2", 150.0, "E2", 80.0),
        (2, 1, "E1", 95.0, "W2", 94.0),
        (2, 2, "W1", 160.0, "E2", 70.0),
    ]
    rows = []
    for wk, mid, o1, p1, o2, p2 in games:
        rows.append({"week": wk, "matchup_id": mid, "roster_id": DIV_RIDS[o1],
                     "owner": o1, "points": p1, "points_against": p2,
                     "avatar": None, "finalized": True})
        rows.append({"week": wk, "matchup_id": mid, "roster_id": DIV_RIDS[o2],
                     "owner": o2, "points": p2, "points_against": p1,
                     "avatar": None, "finalized": True})
    return pd.DataFrame(rows, columns=["week", "matchup_id", "roster_id",
                                       "owner", "points", "points_against",
                                       "avatar", "finalized"])


def _bump_y_by_owner(figs):
    fig = json.loads(figs["chart-bump"])
    return {t["name"]: list(t["y"]) for t in fig["data"]}


def test_bump_chart_division_winner_seeds_ahead_of_better_wild_card():
    from dashboard_services.pages.graphs_page import _bump_card, owner_color_map
    df = _division_weekly()
    figs = {}
    html = _bump_card(df, owner_color_map(DIV_OWNERS), "", figs,
                      div_by_owner=DIV_MAP)
    y = _bump_y_by_owner(figs)
    # After week 2 the East winner E1 (1-1, PF 185) must seed above the West
    # wild card W2 (1-1, PF 244).
    assert y["E1"][-1] == 2
    assert y["W2"][-1] == 3
    assert y["W1"][-1] == 1
    assert "division winners first" in html
    assert "win" not in df.columns


def test_bump_chart_without_divisions_keeps_wins_then_pf_order():
    from dashboard_services.pages.graphs_page import _bump_card, owner_color_map
    df = _division_weekly()
    figs = {}
    html = _bump_card(df, owner_color_map(DIV_OWNERS), "", figs)
    y = _bump_y_by_owner(figs)
    assert y["W1"][-1] == 1
    assert y["W2"][-1] == 2  # same wins as E1, more PF
    assert y["E1"][-1] == 3
    assert y["E2"][-1] == 4
    assert "division winners first" not in html


def test_bump_chart_tie_game_counts_half_win_before_pf():
    from dashboard_services.pages.graphs_page import _bump_card, owner_color_map
    # W beats X 120-105; Y ties Z 100-100. Y/Z (0.5 wins, PF 100) must rank
    # above X (0 wins, PF 105): the half win breaks before PF.
    games = [(1, 1, "W", 120.0, "X", 105.0), (1, 2, "Y", 100.0, "Z", 100.0)]
    rows = []
    rids = {"W": 1, "X": 2, "Y": 3, "Z": 4}
    for wk, mid, o1, p1, o2, p2 in games:
        rows.append({"week": wk, "matchup_id": mid, "roster_id": rids[o1],
                     "owner": o1, "points": p1, "points_against": p2,
                     "avatar": None, "finalized": True})
        rows.append({"week": wk, "matchup_id": mid, "roster_id": rids[o2],
                     "owner": o2, "points": p2, "points_against": p1,
                     "avatar": None, "finalized": True})
    df = pd.DataFrame(rows, columns=["week", "matchup_id", "roster_id",
                                     "owner", "points", "points_against",
                                     "avatar", "finalized"])
    figs = {}
    _bump_card(df, owner_color_map(["W", "X", "Y", "Z"]), "", figs)
    y = _bump_y_by_owner(figs)
    assert y["W"][-1] == 1
    assert y["Y"][-1] < y["X"][-1]
    assert y["Z"][-1] < y["X"][-1]


# ── New charts + hover ────────────────────────────────────────────────────────

def _team_stats():
    avgs = [130.0 - 2 * i for i in range(10)]
    return pd.DataFrame({
        "owner": OWNERS, "PF": [a * 8 for a in avgs], "PA": [900.0] * 10,
        "MAX": [a + 30 for a in avgs], "MIN": [a - 35 for a in avgs],
        "AVG": avgs, "STD": [18.0] * 10,
        "Wins": [8 - (i % 5) for i in range(10)],
        "Losses": [i % 5 for i in range(10)], "Ties": [0] * 10,
    })


def test_floor_ceiling_card_uses_one_shared_scale():
    from dashboard_services.pages.graphs_page import _floor_ceiling_card
    html = _floor_ceiling_card(_team_stats(), _colors(), OWNERS[0])
    assert "Floor / Ceiling" in html
    assert OWNERS[0] in html


def test_heatmap_and_env_figs_registered_with_hover_bg():
    from dashboard_services.pages.graphs_page import _env_card, _heatmap_card
    df = _real_shape_weekly()
    figs = {}
    html = _heatmap_card(df, _colors(), figs)
    assert "Scoring Heatmap" in html
    fig = json.loads(figs["chart-heatmap"])
    assert fig["data"][0]["type"] == "heatmap"
    assert len(fig["data"][0]["z"]) == 10
    assert fig["layout"]["hoverlabel"]["bgcolor"] == "#ffffff"
    figs2 = {}
    html2 = _env_card(df, figs2)
    assert "League Scoring by Week" in html2
    fig2 = json.loads(figs2["chart-env"])
    assert len(fig2["data"]) == 3
    assert fig2["layout"]["hoverlabel"]["bgcolor"] == "#ffffff"


def test_form_card_needs_three_weeks_and_renders():
    from dashboard_services.pages.graphs_page import _form_card
    assert _form_card(_real_shape_weekly(weeks=2), "", _colors()) == ""
    html = _form_card(_real_shape_weekly(), OWNERS[0], _colors())
    assert "Form: Last 3 Weeks" in html


def test_career_wins_card_lists_all_owners():
    from dashboard_services.pages.graphs_page import _career_wins_card
    html = _career_wins_card(_team_stats(), OWNERS[0], _colors())
    assert "Career Wins by Team" in html
    for o in OWNERS:
        assert o in html


def test_plotly_theme_hoverlabel_is_opaque():
    import plotly.graph_objs as go
    from dashboard_services.plotly_theme import apply_brand_layout
    fig = apply_brand_layout(go.Figure(go.Scatter(x=[1], y=[2])))
    hl = fig.to_dict()["layout"]["hoverlabel"]
    assert hl["bgcolor"] == "#ffffff"
    assert hl["bordercolor"] == "#e5e7eb"
    assert hl["font"]["color"] == "#334155"
