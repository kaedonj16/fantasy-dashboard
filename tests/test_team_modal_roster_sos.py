"""Roster tab + remaining-schedule SOS share this-season starter strength.

The team-modal Roster tab used to sort and display dynasty trade value.
Remaining-schedule SOS used a raw roster-value sum before week 1. Both now
read slot-legal this-season production (and remaining NFL matchup ease on
the roster rows).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _team_details_body() -> str:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find('@app.route("/api/team-details/<roster_id>")')
    end = src.find('@app.route("/api/player-league-trades/<player_id>")', start)
    assert start > 0 and end > start
    return src[start:end]


def test_schedule_strength_uses_starter_production_not_roster_sum():
    src = (ROOT / "routes" / "schedule_api_bp.py").read_text(encoding="utf-8")
    start = src.find("def api_schedule_strength")
    assert start > 0
    body = src[start:start + 5000]
    assert "preseason_opponent_strength" in body
    assert "roster_val / 50.0" not in body
    assert "this-season starter" in body or "starter production" in body


def test_team_details_roster_uses_this_season_production_and_sos():
    body = _team_details_body()
    assert "this_season_production" in body
    assert '"sos_ease"' in body
    assert '"sos_rank"' in body
    assert "prod_value" in body
    # Display/sort field is this-season production, not dynasty trade value.
    assert '"value": round(float(prod_value), 1) if prod_value else None' in body
    assert '"trade_value": round(float(value), 1) if value else None' in body


def test_team_modal_roster_js_shows_sos_not_just_value():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "tm-roster-sos" in js
    assert "tm-roster-note" in js
    assert "this-season starter strength" in js
    assert "player.sos_ease" in js


def test_roster_intel_ranks_rooms_by_production():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _api_roster_intel_compute")
    end = src.find("def api_trade_targets", start)
    body = src[start:end]
    assert "this_season_production" in body
    assert '"prod_value"' in body
    assert 'pos_vals[rid][info["position"]].append(' in body
    assert "prod_value" in body
    js = (ROOT / "static" / "teams.js").read_text(encoding="utf-8")
    assert "p.prod_value" in js
    assert "this-season starter strength" in js
