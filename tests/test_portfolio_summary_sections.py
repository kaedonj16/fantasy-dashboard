import pytest

pd = pytest.importorskip("pandas")

from dashboard_services.portfolio_summary import first_non_null, recent_streak


def test_recent_streak_prefers_canonical_weekly_score_columns():
    weekly = pd.DataFrame([
        {"week": 1, "roster_id": "1", "points": 120, "points_against": 100, "finalized": True},
        {"week": 2, "roster_id": "1", "points": 98, "points_against": 110, "finalized": True},
        {"week": 3, "roster_id": "1", "points": 130, "points_against": 125, "finalized": True},
    ])
    assert recent_streak(weekly, "1") == ["W", "L", "W"]


def test_recent_streak_supports_legacy_aliases_and_real_zeroes():
    weekly = pd.DataFrame([
        {"week": 1, "roster_id": 7, "pts": 0.0, "opp_pts": 3.0, "finalized": True},
        {"week": 2, "roster_id": 7, "PF": 4.0, "PA": 0.0, "finalized": True},
    ])
    assert recent_streak(weekly, "7") == ["L", "W"]
    assert first_non_null({"points": 0.0, "pts": 99}, ("points", "pts")) == 0.0


def test_progressive_hydration_keeps_streak_slot_and_resets_manual_retry():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    # render() replaces the stats container, so its replacement must recreate
    # the streak target before the section patch runs.
    render = source[source.index("function render(c,d)"):source.index("function load(c)")]
    assert "data-summary-streak" in render
    assert "c._summaryAttempt=0;q.unshift(c)" in source
