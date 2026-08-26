"""Graphs page must render real chart markup from weekly data, not the empty card.

Constructs DataFrames by hand (do not use ``build_tour_mock_graphs_ctx`` — that
pulls history_page, which imports Flask). Auto-marked integration via
importorskip("pandas").
"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("plotly")
pytest.importorskip("bs4")


def _load_builder():
    from dashboard_services.pages.graphs_page import build_graphs_body
    return build_graphs_body


def _ctx():
    team_stats = pd.DataFrame(
        {
            "owner": ["Gridiron", "Haunted"],
            "PF": [1240.4, 1112.8],
            "PA": [1088.2, 1190.6],
            "MAX": [162.3, 151.0],
            "MIN": [82.1, 71.4],
            "AVG": [124.0, 111.3],
            "STD": [18.2, 22.7],
            "PowerScore": [1.15, 0.72],
        }
    )
    df_weekly = pd.DataFrame(
        {
            "week": [1, 1, 2, 2, 3, 3],
            "owner": ["Gridiron", "Haunted", "Gridiron", "Haunted", "Gridiron", "Haunted"],
            "points": [130.0, 110.0, 118.0, 99.0, 141.0, 125.0],
            "win": [1, 0, 1, 0, 1, 0],
            "finalized": [True, True, True, True, True, True],
        }
    )
    return {
        "team_stats": team_stats,
        "df_weekly": df_weekly,
        "viewer": {"viewer_team_name": "Gridiron"},
        "model_value_table": [],
        "rosters": [],
    }


def test_graphs_empty_ctx_is_the_static_card():
    html = _load_builder()({"team_stats": pd.DataFrame(), "df_weekly": pd.DataFrame()})
    assert "graphs-empty" in html
    assert "No weekly data" in html
    assert "graphs-page" not in html


def test_graphs_renders_plotly_charts_from_weekly_data():
    html = _load_builder()(_ctx())
    assert "graphs-empty" not in html
    assert "No weekly data" not in html
    assert "graphs-page" in html
    assert "PF vs PA Scatter" in html
    assert 'id="chart-pfpa"' in html
    assert "chart-pfpa" in html
    assert "ensurePlotly" in html
    assert "Weekly Scores by Team" in html
