"""History page: standings card shows a Playoff Bracket tab when bracket data exists.

The history-standings-panel card uses the card-tabs pattern (tab-strip / tab-btn /
tab-panel) so users can toggle between the regular-season standings table and the
playoff bracket, the same way the standings page shows a Playoff Picture tab.

Skipped when Flask/openai aren't installed; runs in CI with the full stack.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from unittest import mock  # noqa: E402

import pandas as pd  # noqa: E402


def _mock_history_ctx():
    """Minimal history_ctx dict for testing bracket rendering.

    Includes regular-season weeks (1-2) plus a playoff week (14) so that
    _build_match_scores can resolve scores for the bracket.
    """
    df = pd.DataFrame({
        "week":       [1, 1, 2, 2, 14, 14],
        "owner":      ["Team A", "Team B", "Team A", "Team B", "Team A", "Team B"],
        "roster_id":  [1, 2, 1, 2, 1, 2],
        "points":     [100.0, 90.0, 110.0, 95.0, 120.5, 105.3],
        "matchup_id": [1, 1, 1, 1, 1, 1],
        "finalized":  [True, True, True, True, True, True],
    })
    return {
        "platform": "sleeper",
        "season": 2024,
        "league_id": "test_league",
        "resolved_league_id": "test_league",
        "league": {
            "name": "Test League",
            "league_id": "test_league",
            "settings": {"playoff_week_start": 14},
        },
        "df_weekly": df,
        "roster_map": {"1": "Team A", "2": "Team B"},
        "users": [],
        "rosters": [],
        "offseason_mode": False,
    }


def _sample_bracket():
    """A minimal completed winners bracket."""
    return [
        {"r": 1, "m": 1, "t1": 1, "t2": 2, "w": 1, "l": 2, "p": 1},
    ]


def test_build_match_scores_maps_playoff_points(offline_client):
    """_build_match_scores resolves points from df_weekly for each bracket match."""
    from dashboard_services.pages import history_page as H

    ctx = _mock_history_ctx()
    bracket = _sample_bracket()
    scores = H._build_match_scores(bracket, ctx["df_weekly"], ctx["league"])

    assert 1 in scores, "match id 1 should have scores"
    assert abs(scores[1]["t1_score"] - 120.5) < 0.01
    assert abs(scores[1]["t2_score"] - 105.3) < 0.01


def test_build_match_scores_empty_without_roster_id(offline_client):
    """_build_match_scores returns {} when df_weekly lacks roster_id column."""
    from dashboard_services.pages import history_page as H

    df_no_rid = pd.DataFrame({
        "week": [14, 14], "owner": ["A", "B"], "points": [100, 90],
    })
    scores = H._build_match_scores(_sample_bracket(), df_no_rid, {"settings": {}})
    assert scores == {}


def test_bracket_html_returned_when_bracket_exists(offline_client):
    """When get_bracket returns data, _get_history_bracket_html produces
    bracket markup."""
    from dashboard_services.pages import history_page as H

    ctx = _mock_history_ctx()

    with mock.patch.object(H, "get_bracket", return_value=_sample_bracket()):
        html = H._get_history_bracket_html(ctx)

    assert html, "bracket HTML should not be empty when bracket data exists"
    assert "bracket" in html


def test_bracket_html_includes_scores(offline_client):
    """The bracket HTML should include the playoff matchup scores."""
    from dashboard_services.pages import history_page as H

    ctx = _mock_history_ctx()

    with mock.patch.object(H, "get_bracket", return_value=_sample_bracket()):
        html = H._get_history_bracket_html(ctx)

    assert "120.50" in html, "t1 score should appear in bracket HTML"
    assert "105.30" in html, "t2 score should appear in bracket HTML"


def test_bracket_html_empty_when_no_bracket(offline_client):
    """When get_bracket returns [], the bracket function returns empty string."""
    from dashboard_services.pages import history_page as H

    ctx = _mock_history_ctx()

    with mock.patch.object(H, "get_bracket", return_value=[]):
        html = H._get_history_bracket_html(ctx)

    assert html == "", "bracket HTML should be empty when no bracket data"


def test_bracket_handles_exception_gracefully(offline_client):
    """If get_bracket raises, _get_history_bracket_html returns empty string."""
    from dashboard_services.pages import history_page as H

    ctx = _mock_history_ctx()

    with mock.patch.object(H, "get_bracket", side_effect=Exception("API down")):
        html = H._get_history_bracket_html(ctx)

    assert html == ""


def test_bracket_html_empty_when_all_empty_bracket(offline_client):
    """If get_bracket returns data but playoff_bracket renders the po-empty
    sentinel, _get_history_bracket_html returns empty string."""
    from dashboard_services.pages import history_page as H

    ctx = _mock_history_ctx()

    with mock.patch.object(H, "get_bracket", return_value=[{"r": None}]):
        html = H._get_history_bracket_html(ctx)

    assert html == ""


def test_standings_panel_has_tabs_with_bracket(offline_client):
    """build_history_body renders the standings card with card-tabs when bracket
    data is available."""
    import app
    from dashboard_services.pages import history_page as H

    ctx = _mock_history_ctx()

    with app.app.test_request_context("/"):
        with mock.patch.object(H, "get_bracket", return_value=_sample_bracket()):
            with mock.patch.object(H, "get_league_season_summary", return_value="Test recap"):
                body = H.build_history_body(
                    history_ctx=ctx,
                    available_seasons=[2024],
                    base_platform="sleeper",
                    base_season=2024,
                    base_league_id="test_league",
                    selected_history_season=2024,
                    resolved_history_league_id="test_league",
                    prerendered={
                        "summary": "<div>awards</div>",
                        "standings": "<div>standings</div>",
                        "chart": "<div>chart</div>",
                    },
                )

    assert 'data-tab="standings"' in body
    assert 'data-tab="bracket"' in body
    assert "Playoff Bracket" in body
    assert "tab-strip" in body


def test_standings_panel_no_tabs_without_bracket(offline_client):
    """build_history_body omits the bracket tab when no bracket data exists."""
    import app
    from dashboard_services.pages import history_page as H

    ctx = _mock_history_ctx()

    with app.app.test_request_context("/"):
        with mock.patch.object(H, "get_bracket", return_value=[]):
            with mock.patch.object(H, "get_league_season_summary", return_value="Test recap"):
                body = H.build_history_body(
                    history_ctx=ctx,
                    available_seasons=[2024],
                    base_platform="sleeper",
                    base_season=2024,
                    base_league_id="test_league",
                    selected_history_season=2024,
                    resolved_history_league_id="test_league",
                    prerendered={
                        "summary": "<div>awards</div>",
                        "standings": "<div>standings</div>",
                        "chart": "<div>chart</div>",
                    },
                )

    assert 'data-tab="standings"' in body
    assert 'data-tab="bracket"' not in body
    assert "Playoff Bracket" not in body
