"""Recap power rankings must match the standings page for the same week.

The standings page renders a week two ways: the latest finalized week is
its default load (the live blended list), and any older week selected in
the week-selector ranks by the performance PowerScore stored in the
week-capped team_stats. The recap follows the same split, and
/api/standings-week applies it too, so all three surfaces agree.
"""
from pathlib import Path

import pytest

from dashboard_services.pages.recap_page import _power_teams_from_team_stats

ROOT = Path(__file__).resolve().parents[1]

OWNER_TO_RID = {"Alpha": "1", "Bravo": "2", "Charlie": "3"}


def test_performance_power_orders_by_score_then_pf():
    team_stats = [
        {"owner": "Bravo", "PowerScore": 0.50, "PF": 300.0},
        {"owner": "Alpha", "PowerScore": 0.50, "PF": 350.0},
        {"owner": "Charlie", "PowerScore": 1.25, "PF": 200.0},
    ]
    teams = _power_teams_from_team_stats(team_stats, OWNER_TO_RID)
    # Charlie leads on score; Alpha wins the 0.50 tie on PF, exactly how
    # render_power_and_playoffs sorts the standings week view.
    assert [t["roster_id"] for t in teams] == ["3", "1", "2"]
    assert [t["power_score"] for t in teams] == [1.25, 0.50, 0.50]
    assert [t["team_name"] for t in teams] == ["Charlie", "Alpha", "Bravo"]


def test_performance_power_skips_unmappable_and_unscored_rows():
    team_stats = [
        {"owner": "Alpha", "PowerScore": 0.10, "PF": 100.0},
        {"owner": "Ghost", "PowerScore": 9.99, "PF": 999.0},
        {"owner": "Bravo", "PF": 100.0},
        {"owner": "Charlie", "PowerScore": float("nan"), "PF": 50.0},
    ]
    teams = _power_teams_from_team_stats(team_stats, OWNER_TO_RID)
    assert [t["roster_id"] for t in teams] == ["1", "3"]
    assert teams[1]["power_score"] == 0.0  # NaN reads as no score


def test_performance_power_accepts_team_stats_dataframe():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([
        {"owner": "Alpha", "PowerScore": -0.25, "PF": 90.0},
        {"owner": "Bravo", "PowerScore": 0.75, "PF": 80.0},
    ])
    teams = _power_teams_from_team_stats(frame, OWNER_TO_RID)
    assert [t["roster_id"] for t in teams] == ["2", "1"]


def test_performance_power_empty_inputs():
    assert _power_teams_from_team_stats(None, OWNER_TO_RID) == []
    assert _power_teams_from_team_stats([], OWNER_TO_RID) == []


def test_recap_and_standings_week_api_share_the_same_split():
    recap = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    assert "_power_is_live = selected_week >= _latest_final_week" in recap
    assert "build_power_rankings_context(capped_ctx)" in recap
    assert "power_teams = _power_teams_for(historical_ctx)" in recap
    assert "prior_power_teams = _power_teams_for(prior_ctx)" in recap

    api = (ROOT / "routes/league_pages_bp.py").read_text()
    week_api = api.split("def api_standings_week", 1)[1]
    assert "_power_rankings = None" in week_api
    assert "week >= _weeks[-1]" in week_api
    assert "build_power_rankings_context(capped)" in week_api
