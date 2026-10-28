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


def test_heatmap_fig_registered_with_hover_bg_and_env_insight_merged():
    from dashboard_services.pages.graphs_page import _heatmap_card
    df = _real_shape_weekly()
    figs = {}
    html = _heatmap_card(df, _colors(), figs)
    assert "Scoring Heatmap" in html
    fig = json.loads(figs["chart-heatmap"])
    assert fig["data"][0]["type"] == "heatmap"
    assert len(fig["data"][0]["z"]) == 10
    assert fig["layout"]["hoverlabel"]["bgcolor"] == "#ffffff"
    # The removed League Scoring by Week card's one unique fact (the
    # league's biggest scoring week) is merged into this insight.
    assert "biggest scoring week" in html


def test_form_card_needs_three_weeks_and_renders():
    from dashboard_services.pages.graphs_page import _form_card, owner_color_map
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


def test_luck_analysis_computed_once_per_season_build(monkeypatch):
    import dashboard_services.pages.graphs_page as gp

    counts = {"n": 0}
    real = gp._luck_analysis

    def counting(df):
        counts["n"] += 1
        return real(df)

    monkeypatch.setattr(gp, "_luck_analysis", counting)
    df = _real_shape_weekly()
    team_stats = pd.DataFrame({
        "owner": OWNERS,
        "PF": [900.0 + i * 10 for i in range(10)],
        "PA": [850.0 + i * 5 for i in range(10)],
        "AVG": [100.0 + i for i in range(10)],
        "STD": [15.0] * 10,
        "MAX": [150.0] * 10,
        "MIN": [60.0] * 10,
    })
    ctx = {"team_stats": team_stats, "df_weekly": df, "league_name": "L",
           "rosters": [], "players_index": {}, "model_value_table": []}
    html = gp.build_graphs_body(ctx)
    # The scatter and the delta bars share one all-play analysis per build.
    assert counts["n"] == 1
    assert "Performance vs Luck" in html
    assert "Luck Delta" in html


# ── Fill + duplicate-removal + data correctness audit (2026-10-28) ──────────

def _paired_weekly(games):
    """games: list of (week, owner_a, pts_a, owner_b, pts_b)."""
    rows = []
    for wk, a, pa, b, pb in games:
        rows.append({"week": wk, "matchup_id": wk, "roster_id": 1, "owner": a,
                     "points": pa, "points_against": pb, "avatar": None, "finalized": True})
        rows.append({"week": wk, "matchup_id": wk, "roster_id": 2, "owner": b,
                     "points": pb, "points_against": pa, "avatar": None, "finalized": True})
    return pd.DataFrame(rows, columns=["week", "matchup_id", "roster_id", "owner",
                                       "points", "points_against", "avatar", "finalized"])


def test_no_spacer_gap_mechanism_and_fill_classes_present():
    from dashboard_services.pages.graphs_page import build_graphs_body
    ctx = {"team_stats": _team_stats(), "df_weekly": _real_shape_weekly(),
           "viewer": {"viewer_team_name": OWNERS[0]}, "league": {"name": "L"},
           "model_value_table": [], "rosters": []}
    html = build_graphs_body(ctx)
    # The old spacer that pinned insights to the bottom (leaving the hole
    # in Kaedon's screenshots) is gone; content fills the card instead.
    assert "gs-flexsp" not in html
    assert ".gs-rows{" in html
    assert 'class="gs-rows"' in html
    assert 'class="gs-plot" id="chart-pfpa"' in html
    assert "ResizeObserver" in html


def test_every_season_card_title_renders_exactly_once():
    from dashboard_services.pages.graphs_page import build_graphs_body
    ctx = {"team_stats": _team_stats(), "df_weekly": _real_shape_weekly(),
           "viewer": {"viewer_team_name": OWNERS[0]}, "league": {"name": "L"},
           "model_value_table": [], "rosters": []}
    html = build_graphs_body(ctx)
    titles = ["Performance vs Luck", "Luck Delta", "Consistency Ranking",
              "Floor / Ceiling", "PF vs PA", "Margin of Victory / Defeat",
              "Head-to-Head Matrix", "Radar Comparison",
              "Weekly Scoring: All Teams", "Scoring Heatmap",
              "Standings Bump Chart", "Form: Last 3 Weeks",
              "Strength of Schedule"]
    for t in titles:
        assert html.count(f"<h2>{t}</h2>") == 1, t
    # Removed duplicates render nowhere.
    assert "<h2>Weekly Scoring Trend</h2>" not in html
    assert "League Scoring by Week" not in html
    assert "Show score distributions" not in html
    assert "chart-boxdist" not in html and "chart-env" not in html and "chart-trend" not in html


def test_form_delta_matches_hand_computed_screenshot_case():
    from dashboard_services.pages.graphs_page import _form_card, owner_color_map
    caleb = "Caleb's Casting Couch"
    games = []
    # Caleb: season mean 164.5, last-3 mean 143.8 -> delta -20.7 (the exact
    # numbers in Kaedon's 2026-10-28 screenshot).
    caleb_pts = [195.55, 195.55, 143.8, 143.8, 143.8]
    for i, pts in enumerate(caleb_pts, start=1):
        games.append((i, caleb, pts, "Other Team", 100.0))
    df = _paired_weekly(games)
    html = _form_card(df, caleb, owner_color_map([caleb, "Other Team"]))
    assert "Form: Last 3 Weeks" in html
    assert "-20.7" in html
    assert "at 143.8 per week, -20.7 vs your season average of 164.5" in html
    # Full name is reachable on the (possibly truncated) name cell.
    assert 'title="Caleb&#x27;s Casting Couch"' in html


def test_luck_delta_equals_actual_minus_expected_wins():
    from dashboard_services.pages.graphs_page import _luck_analysis, _luck_delta_card
    df = _real_shape_weekly()
    analysis = _luck_analysis(df)
    html = _luck_delta_card(df, OWNERS[0], _colors(), analysis=analysis)
    for owner, a in analysis.items():
        if a.get("games"):
            assert f"{float(a['luck_delta']):+.1f}" in html
            assert abs(float(a["luck_delta"]) - (float(a["actual_wins"]) - float(a["expected_wins"]))) < 0.11


def test_floor_ceiling_range_matches_team_stats():
    from dashboard_services.pages.graphs_page import _floor_ceiling_card
    ts = _team_stats()
    html = _floor_ceiling_card(ts, _colors(), OWNERS[0])
    r0 = ts.iloc[0]
    assert f"{float(r0['MIN']):.0f}-{float(r0['MAX']):.0f}" in html


def test_margin_card_win_loss_means_and_tie_excluded():
    from dashboard_services.pages.graphs_page import _margin_card, owner_color_map
    games = [(1, "A", 150.0, "B", 100.0), (2, "B", 130.0, "A", 90.0),
             (3, "A", 110.0, "B", 110.0)]
    df = _paired_weekly(games)
    html = _margin_card(df, owner_color_map(["A", "B"]), "A")
    # A: win margin +50.0, loss margin -40.0, the tie counts as neither.
    assert "+50.0" in html and "-40.0" in html and "1W-1L" in html


def test_heatmap_cells_equal_owner_week_points():
    from dashboard_services.pages.graphs_page import _heatmap_card
    df = _real_shape_weekly()
    figs = {}
    _heatmap_card(df, _colors(), figs)
    fig = json.loads(figs["chart-heatmap"])
    owners_y = fig["data"][0]["y"]
    z = fig["data"][0]["z"]
    for oi, owner in enumerate(owners_y):
        for wi, wk in enumerate(fig["data"][0]["x"]):
            expected = float(df[(df["owner"] == owner) & (df["week"] == wk)]["points"].iloc[0])
            assert z[oi][wi] == expected


def test_pfpa_yaxis_padded_like_xaxis():
    from dashboard_services.pages.graphs_page import _pf_pa_card
    ts = _team_stats()
    figs = {}
    _pf_pa_card(ts, _colors(), OWNERS[0], figs)
    fig = json.loads(figs["chart-pfpa"])
    yr = fig["layout"]["yaxis"]["range"]
    pfs = [float(v) for v in ts["PF"]]
    assert yr[0] < min(pfs) and yr[1] > max(pfs)


def test_all_teams_viewer_line_emphasized_and_y_padded():
    from dashboard_services.pages.graphs_page import _all_teams_trend_card
    df = _real_shape_weekly()
    figs = {}
    html = _all_teams_trend_card(df, _colors(), figs, OWNERS[0])
    fig = json.loads(figs["chart-allteams"])
    mine = [t for t in fig["data"] if t.get("name") == OWNERS[0]][0]
    assert mine["line"]["width"] == 3.5
    yr = fig["layout"]["yaxis"]["range"]
    pts = [float(v) for v in df["points"]]
    assert yr[0] < min(pts) and yr[1] > max(pts)
    assert "trending" in html  # the removed Trend card's viewer insight


def test_sos_viewer_chart_uses_own_points_against():
    from dashboard_services.pages.graphs_page import _sos_card
    df = _real_shape_weekly()
    figs = {}
    _sos_card(df, OWNERS[0], _colors(), figs)
    fig = json.loads(figs["chart-sos"])
    g = df[df["owner"] == OWNERS[0]].sort_values("week")
    assert [float(v) for v in fig["data"][0]["y"]] == [float(v) for v in g["points_against"]]


def test_career_wins_falls_back_to_points_without_wins():
    from dashboard_services.pages.graphs_page import _career_wins_card
    ts = _team_stats().drop(columns=["Wins", "Losses", "Ties"])
    html = _career_wins_card(ts, OWNERS[0], _colors())
    assert "Career Points by Team" in html


def test_radar_zmap_flips_pa_and_std_signs():
    import re
    from dashboard_services.pages.graphs_page import _radar_card
    ts = _team_stats()
    ts["PA"] = [800.0 + 20 * i for i in range(10)]  # Team 0 lowest PA
    figs = {}
    html = _radar_card(ts, _colors(), figs)
    m = re.search(r"var ZMAP=(\{.*?\});", html)
    assert m, "radar ZMAP not found"
    zmap = json.loads(m.group(1))
    # Metrics order PF, PA, MAX, MIN, AVG, STD: PA is sign-flipped so the
    # lowest-PA team scores highest (better outward) on that axis.
    assert zmap[OWNERS[0]][1] > zmap[OWNERS[-1]][1]
    assert zmap[OWNERS[0]][0] > zmap[OWNERS[-1]][0]  # PF not flipped


def test_career_card_titles_render_exactly_once():
    from dashboard_services.pages.graphs_page import build_career_graphs_body
    rec = pd.DataFrame({"season": [2023, 2024, 2025] * 2,
                        "owner": [OWNERS[0]] * 3 + [OWNERS[1]] * 3,
                        "wins": [5, 7, 9, 8, 6, 6], "losses": [8, 6, 4, 5, 7, 7],
                        "ties": [0] * 6})
    pf = pd.DataFrame({"season": [2023, 2024, 2025] * 2,
                       "owner": [OWNERS[0]] * 3 + [OWNERS[1]] * 3,
                       "pf": [1400.0, 1500.0, 1600.0, 1300.0, 1350.0, 1380.0]})
    ctx = {"team_stats": _team_stats(), "season_record_df": rec, "season_pf_df": pf}
    html = build_career_graphs_body(ctx, viewer_owner=OWNERS[0], league_name="L")
    for t in ("Career Win % by Season", "Career Points For", "Career Wins by Team"):
        assert html.count(f"<h2>{t}</h2>") == 1, t
