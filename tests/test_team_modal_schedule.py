"""Regression coverage for provider-authored team-modal schedules."""
from pathlib import Path
from dashboard_services.team_schedule import build_team_schedule

ROOT = Path(__file__).parents[1]


def fixture(details=False, rows=None, season=2026, current_week=2):
    rows = rows if rows is not None else [
        {"roster_id": 10, "matchup_id": "m1", "points": 0, "starters": ["p-old"],
         "players_points": {"p-old": -2.5}, "custom_points": -1},
        {"roster_id": 20, "matchup_id": "m1", "points": -1, "starters": ["p2"],
         "players_points": {"p2": 0}},
    ]
    return build_team_schedule(
        platform="sleeper", league_id="L", season=season, roster_id="10",
        league={"settings": {"total_weeks": 2, "playoff_week_start": 2}},
        rosters=[{"roster_id": 10, "owner_id": "u1", "players": ["current"]},
                 {"roster_id": 20, "owner_id": "u2"}],
        users=[{"user_id": "u1", "display_name": "One"}, {"user_id": "u2", "display_name": "Two"}],
        current_season=2026, current_week=current_week,
        get_week=lambda week: rows if week == 1 else [], details=details,
    )


def test_schedule_uses_ids_authoritative_totals_and_states(monkeypatch):
    monkeypatch.setattr("utils.utils.load_players_index", lambda: {"p-old": {"name": "Traded Player", "pos": "RB"}, "p2": {"name": "Defense", "pos": "DEF"}})
    data = fixture(details=True)
    week = data["weeks"][0]
    assert week["opponent"]["roster_id"] == "20"
    assert week["team_points"] == 0.0 and week["opponent_points"] == -1.0
    assert week["state"] == "final" and week["result"] == "W"
    assert week["team_adjustment"] == -1.0
    assert week["team_lineup"] == [{"player_id": "p-old", "name": "Traded Player", "position": "RB", "slot": None, "points": -2.5}]
    assert all(p["player_id"] != "current" for p in week["team_lineup"])
    assert data["weeks"][1] == {"week": 2, "state": "unpublished", "published": False}


def test_tie_missing_projection_and_missing_score_are_not_invented():
    tied = [{"roster_id": 10, "matchup_id": 7, "points": 0}, {"roster_id": 20, "matchup_id": 7, "points": 0}]
    week = fixture(rows=tied)["weeks"][0]
    assert week["result"] == "T"
    assert week["team_projection"] is None
    missing = [{"roster_id": 10, "matchup_id": 7, "points": None}, {"roster_id": 20, "matchup_id": 7, "points": 0}]
    week = fixture(rows=missing)["weeks"][0]
    assert week["state"] == "unavailable" and week["result"] is None


def test_frontend_has_no_mock_schedule_or_bye_conflicts_and_hides_details():
    js = (ROOT / "static/app.js").read_text()
    css = (ROOT / "static/dashboard.css").read_text()
    app = (ROOT / "app.py").read_text()
    for forbidden in ("_tmSeededRng", "_tmGenPts", "_tmScoreLineup", "_TM_NAME_POOL", "schedule_opponents", "bye_conflicts"):
        assert forbidden not in js and forbidden not in app
    assert ".tm-menu[hidden]" in css
    assert ".tm-sched-detail[hidden]" in css
    assert "tmLoadSchedule(false)" in js
    assert "/week/${week}" in js


def test_menu_teardown_and_accessibility_contract():
    js = (ROOT / "static/app.js").read_text()
    assert "tmCloseMenu(false);" in js
    assert "document.removeEventListener('click', _tmMenuOutside)" in js
    assert "document.removeEventListener('keydown', _tmMenuKeydown)" in js
    assert "if (e.key === 'Escape')" in js and "tmCloseMenu(true)" in js
    assert "trigger.setAttribute('aria-expanded', willOpen ? 'true' : 'false')" in js
    assert "window._tmScheduleAbort.abort()" in js
