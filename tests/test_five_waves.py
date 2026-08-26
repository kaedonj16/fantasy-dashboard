"""Source contracts for the five-wave audit follow-through."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_trade_outcome_flags_nearest_board():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    ttv = (ROOT / "dashboard_services" / "trade_time_values.py").read_text(encoding="utf-8")
    assert "def get_or_persist_trade_value_meta" in ttv
    assert '"then_estimated"' in app
    assert "Approximate (nearest board)." in js


def test_prospects_paused_banner_with_rows():
    page = (ROOT / "dashboard_services" / "pages" / "rookies_page.py").read_text(encoding="utf-8")
    api = (ROOT / "dashboard_services" / "rookie_api.py").read_text(encoding="utf-8")
    assert "rkPausedBanner" in page
    assert 'payload["last_updated"]' in api


def test_schedule_idp_and_wednesday_copy():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "Ratings rebuild on Wednesdays in-season" in app
    assert '_idp_slots' in app


def test_keeper_mfl_banner():
    html = (ROOT / "dashboard_services" / "pages" / "_keeper_render.py").read_text(encoding="utf-8")
    assert "MFL draft history is not auto-imported" in html


def test_compare_does_not_call_am_a_pro_feature():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "Advanced metrics are a PRO feature" not in js
    assert "Start/Sit score" in js
