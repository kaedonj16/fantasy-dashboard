"""Graphs page ("League Stats") renders the tabbed chart layout from real data.

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
            "owner": ["Gridiron", "Haunted", "Blitz"],
            "PF": [1240.4, 1112.8, 1180.0],
            "PA": [1088.2, 1190.6, 1150.0],
            "MAX": [162.3, 151.0, 155.0],
            "MIN": [82.1, 71.4, 80.0],
            "AVG": [124.0, 111.3, 118.0],
            "STD": [18.2, 22.7, 20.0],
            "PowerScore": [1.15, 0.72, 0.9],
        }
    )
    df_weekly = pd.DataFrame(
        {
            "week": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "owner": ["Gridiron", "Haunted", "Blitz"] * 3,
            "points": [130.0, 110.0, 120.0, 118.0, 99.0, 125.0, 141.0, 125.0, 130.0],
            "points_against": [110.0, 130.0, 120.0, 99.0, 118.0, 125.0, 125.0, 141.0, 130.0],
            "win": [1, 0, 0, 0, 0, 1, 1, 0, 0],
            "finalized": [True] * 9,
        }
    )
    return {
        "team_stats": team_stats,
        "df_weekly": df_weekly,
        "viewer": {"viewer_team_name": "Gridiron"},
        "league": {"name": "Test League"},
        "model_value_table": [],
        "rosters": [],
    }


def test_graphs_empty_ctx_is_the_static_card():
    html = _load_builder()({"team_stats": pd.DataFrame(), "df_weekly": pd.DataFrame()})
    assert "graphs-empty" in html
    assert "No weekly data" in html
    assert "graphs-page" not in html


def test_graphs_renders_tabbed_shell_with_title():
    html = _load_builder()(_ctx())
    assert "graphs-empty" not in html
    assert "League Stats" in html
    assert "Test League" in html
    for tab in ("Performance", "Value", "Trends", "Career"):
        assert tab in html
    for pane in ("gs-pane-perf", "gs-pane-value", "gs-pane-trends", "gs-pane-career"):
        assert pane in html
    # Performance tab is active by default.
    assert 'data-tab="perf"' in html


def test_graphs_tab_param_selects_initial_tab():
    html = _load_builder()(_ctx(), tab="trends")
    assert '<button class="gs-tab active" data-tab="trends"' in html
    assert '<div class="gs-pane active" id="gs-pane-trends"' in html


def test_graphs_performance_tab_has_luck_and_consistency_with_real_insights():
    html = _load_builder()(_ctx())
    # Luck scatter needs 3+ teams; the ctx has 3.
    assert "Performance vs Luck" in html
    assert "luck-quadrant" in html
    # Real luck insight: Gridiron is 2-1, all-play says 2.5 wins -> neutral.
    assert "You&#x27;re 2-1" in html or "You're 2-1" in html
    assert "all-play record says 2.5 wins" in html
    # Consistency ranking with the viewer marked and a real rank insight.
    assert "Consistency Ranking" in html
    assert "Gridiron (you)" in html
    assert "in steadiness" in html


def test_graphs_trends_tab_has_trend_and_sos_with_real_insights():
    html = _load_builder()(_ctx())
    assert "Weekly Scoring Trend" in html
    assert 'id="chart-trend"' in html
    assert "Strength of Schedule" in html
    assert 'id="chart-sos"' in html
    # Both charts are registered for deferred Plotly rendering.
    assert '"chart-trend":' in html
    assert '"chart-sos":' in html
    # Real trend insight from Gridiron's last 3 weeks (130, 118, 141).
    assert "trending up" in html
    assert "130 → 118 → 141" in html
    # Real SOS insight from points_against.
    assert "Opponents have averaged" in html


def test_graphs_sos_card_omitted_without_points_against():
    ctx = _ctx()
    ctx["df_weekly"] = ctx["df_weekly"].drop(columns=["points_against"])
    html = _load_builder()(ctx)
    assert 'id="chart-sos"' not in html
    # The trend chart does not depend on points_against, so it still renders.
    assert "Weekly Scoring Trend" in html


def test_graphs_value_tab_redraft_shows_honest_empty_state():
    ctx = _ctx()
    ctx["platform"] = "espn"  # ESPN is always redraft
    html = _load_builder()(ctx)
    assert "dynasty and keeper leagues" in html
    assert "This league is redraft" in html


def _value_rows():
    return [
        {"owner": "Gridiron", "total_value": 4820.0, "avg_age": 24.2, "n": 15},
        {"owner": "Haunted", "total_value": 4410.0, "avg_age": 25.1, "n": 15},
        {"owner": "Blitz", "total_value": 3950.0, "avg_age": 27.8, "n": 15},
    ]


def test_roster_value_card_ranks_viewer_with_real_gap():
    from dashboard_services.pages.graphs_page import _roster_value_card, owner_color_map
    colors = owner_color_map(["Gridiron", "Haunted", "Blitz"])
    html = _roster_value_card(_value_rows(), "Haunted", colors)
    assert "Roster Value by Team" in html
    assert "Haunted (you)" in html
    assert "gs-bar-row you" in html
    # Haunted is 2nd, 410 behind Gridiron.
    assert "2nd of 3" in html
    assert "410 behind Gridiron" in html


def test_value_age_card_computes_real_window_insight():
    from dashboard_services.pages.graphs_page import _value_age_card, owner_color_map
    colors = owner_color_map(["Gridiron", "Haunted", "Blitz"])
    html = _value_age_card(_value_rows(), "Gridiron", colors)
    assert "Dynasty Value vs Age" in html
    assert "age 24.2" in html
    assert "younger than" in html
    assert "ranks 1st of 3" in html


def _career_ctx():
    rec = pd.DataFrame(
        {
            "season": [2023, 2024, 2025, 2023, 2024, 2025],
            "owner_key": ["a", "a", "a", "b", "b", "b"],
            "owner": ["Gridiron"] * 3 + ["Haunted"] * 3,
            "wins": [5, 7, 9, 8, 6, 6],
            "losses": [8, 6, 4, 5, 7, 7],
            "ties": [0] * 6,
        }
    )
    pf = pd.DataFrame(
        {
            "season": [2023, 2024, 2025, 2023, 2024, 2025],
            "owner_key": ["a", "a", "a", "b", "b", "b"],
            "owner": ["Gridiron"] * 3 + ["Haunted"] * 3,
            "pf": [1420.0, 1598.0, 1847.0, 1500.0, 1511.0, 1490.0],
        }
    )
    ts = pd.DataFrame(
        {
            "owner": ["Gridiron", "Haunted"],
            "owner_key": ["a", "b"],
            "Wins": [21, 20], "Losses": [18, 19], "Ties": [0, 0],
            "PF": [4865.0, 4501.0], "PA": [4200.0, 4300.0],
            "AVG": [120.0, 118.0], "Win%": [0.53, 0.51],
            "MAX": [160.0, 155.0], "MIN": [80.0, 85.0], "STD": [18.0, 20.0],
        }
    )
    return {"team_stats": ts, "season_pf_df": pf, "season_record_df": rec}


def test_career_body_renders_viewer_charts_with_real_insights():
    from dashboard_services.pages.graphs_page import build_career_graphs_body
    html = build_career_graphs_body(
        _career_ctx(), viewer_owner="Gridiron", league_name="Test League"
    )
    assert "League Stats" in html
    assert "Career (all seasons)" in html
    assert "Career Win % by Season" in html
    assert 'id="chart-career-wpct"' in html
    assert '"chart-career-wpct":' in html
    assert "Career Points For" in html
    # Gridiron win% climbs 2023->2024->2025 (two straight climbs).
    assert "has climbed 2 straight seasons" in html
    # 2025 is the best scoring season.
    assert "2025 was your highest-scoring season ever (1,847 points)" in html
    # Career tab is the active one.
    assert '<button class="gs-tab active" data-tab="career"' in html


def test_career_winpct_streak_lines():
    from dashboard_services.pages.graphs_page import _winpct_streak_line
    assert "climbed 2 straight seasons" in _winpct_streak_line([2023, 2024, 2025], [0.42, 0.50, 0.55])
    assert "dipped in 2025 after climbing 1 straight season" in _winpct_streak_line(
        [2023, 2024, 2025], [0.42, 0.55, 0.50]
    )
    assert "held steady" in _winpct_streak_line([2023, 2024, 2025], [0.42, 0.50, 0.50])
    assert "One season on record" in _winpct_streak_line([2025], [0.60])


def test_seed_series_for_ranks_by_cumulative_wins_then_pf():
    from utils.seed_series import seed_series_for

    df = pd.DataFrame(
        {
            "week": [1, 1, 2, 2],
            "owner": ["Gridiron", "Haunted", "Gridiron", "Haunted"],
            "points": [100.0, 120.0, 150.0, 110.0],
            "win": [0, 1, 1, 0],
            "finalized": [True, True, True, True],
        }
    )
    # Week 1: Haunted 1-0 leads. Week 2: both 1-1, Gridiron leads on PF (250 vs 230).
    assert seed_series_for(df, "Gridiron") == [(1, 2), (2, 1)]
    assert seed_series_for(df, "Haunted") == [(1, 1), (2, 2)]
    # Unknown owner and empty input are best-effort [].
    assert seed_series_for(df, "Nobody") == []
    assert seed_series_for(pd.DataFrame(), "Gridiron") == []
