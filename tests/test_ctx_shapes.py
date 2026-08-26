"""Contracts for live league-context shapes.

Opponent Scout used to read team1/team2 while production matchups_by_week
uses left/right. Keep that mismatch from coming back.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_matchups_by_week_use_left_right():
    scout = (ROOT / "dashboard_services" / "pages" / "scout_page.py").read_text(encoding="utf-8")
    assert 'm.get("left") or m.get("team1")' in scout
    assert 'm.get("right") or m.get("team2")' in scout
    tests = (ROOT / "tests" / "test_scout_page.py").read_text(encoding="utf-8")
    assert '"left":' in tests
    assert '"right":' in tests


def test_pipeline_records_skipped_and_wls_errors():
    cron = (ROOT / "cron_daily.py").read_text(encoding="utf-8")
    assert 'record_pipeline_health("build_matchup_ratings", "skipped")' in cron
    assert 'record_pipeline_health("build_weekly_rookie_data", "skipped")' in cron
    assert "CRON_STEPS" in cron
    wls = cron[cron.index("Step 9: WLS"):cron.index("Step 10:")]
    assert "except Exception as e:" not in wls


def test_notification_cron_does_not_lie_on_failure():
    push = (ROOT / "routes" / "push_bp.py").read_text(encoding="utf-8")
    assert 'return jsonify({"ok": False, "error": str(exc)}), 500' in push
