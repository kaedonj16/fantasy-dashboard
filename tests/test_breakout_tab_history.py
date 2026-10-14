"""Breakout tab for players surfaced in ANY week, not just the current board.

A player the board called in Week 2 keeps their Breakout tab after they
roll off the latest board: the tab serves that week's stored card,
labeled with its week, with the finished grade's verdict stamp and the
grade receipt (why this grade) attached. Canonical "surfaced" is
weekly_grading.surfaced_player_ids_by_week -- the same set the track
record counts.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _patch_surfaced(monkeypatch, by_week):
    import data_building.breakout_engine.weekly_grading as wg

    monkeypatch.setattr(wg, "surfaced_player_ids_by_week", lambda season: by_week)


# ---------------------------------------------------------------------------
# breakout_call_history / _surfaced_weeks_for
# ---------------------------------------------------------------------------

def test_call_history_returns_earliest_surfaced_week(monkeypatch):
    import dashboard_services.breakout_api as api

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: 2026)
    _patch_surfaced(monkeypatch, {2: {"love"}, 3: set(), 5: {"love", "other"}})

    assert api.breakout_call_history("love", 2026) == {
        "kind": "weekly", "season": 2026, "as_of_week": 2, "label": "Week 2",
    }
    assert api._surfaced_weeks_for("love", 2026) == [2, 5]


def test_call_history_none_when_never_surfaced(monkeypatch):
    import dashboard_services.breakout_api as api

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: 2026)
    _patch_surfaced(monkeypatch, {2: {"someone-else"}})

    assert api.breakout_call_history("love", 2026) is None


def test_call_history_fails_soft(monkeypatch):
    import dashboard_services.breakout_api as api
    import data_building.breakout_engine.weekly_grading as wg

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: 2026)

    def _boom(season):
        raise RuntimeError("db down")

    monkeypatch.setattr(wg, "surfaced_player_ids_by_week", _boom)
    assert api.breakout_call_history("love", 2026) is None


# ---------------------------------------------------------------------------
# Weekly detail for a specific week: grade attached
# ---------------------------------------------------------------------------

def test_weekly_detail_for_week_attaches_grade_and_call_label(monkeypatch):
    import dashboard_services.breakout_api as api

    monkeypatch.setattr(
        "data_building.breakout_engine.weekly_store.get_weekly_candidate",
        lambda pid, season, as_of_week=None: {
            "player_id": pid, "breakout_score": 62, "confidence": 70,
            "classification": "emerging_breakout", "evidence": {},
            "as_of_week": as_of_week,
        },
    )
    monkeypatch.setattr(api, "_weekly_series_for_payload", lambda pid, season: [])
    monkeypatch.setattr(api, "_breakout_ranks", lambda season: {})
    monkeypatch.setattr(api, "_weekly_grades_for_week", lambda season, week: {
        "love": {"grade": "hit", "grade_detail": {
            "ppg_delta": 4.2, "snap_delta": 11.0, "opp_delta": 2.5,
            "outcome_games": 3, "reason": "Role and production both cleared the bar.",
        }},
    })

    detail = api.get_weekly_breakout_detail("love", 2026, as_of_week=2)

    assert detail["available"] is True
    assert detail["historical_call"] is True
    assert detail["breakout_call"]["label"] == "Week 2"
    assert detail["as_of_week"] == 2
    assert detail["grade"] == "hit"
    assert detail["grade_detail"]["ppg_delta"] == 4.2


def test_historical_detail_ranks_within_that_weeks_board(monkeypatch):
    import dashboard_services.breakout_api as api

    monkeypatch.setattr(api, "get_weekly_breakout_detail", lambda pid, season, as_of_week=None: {
        "player_id": pid, "available": True, "breakout_rank": {"overall": 99},
    })
    monkeypatch.setattr(api, "get_weekly_breakout_candidates", lambda season, as_of_week=None: {
        "candidates": [
            {"player_id": "a", "position": "RB"},
            {"player_id": "love", "position": "RB", "forecast": {"band_label": "Tracking to hit"}},
            {"player_id": "b", "position": "WR"},
        ],
    })

    detail = api.get_historical_weekly_breakout_detail("love", 2026, 2)

    assert detail["breakout_rank"]["overall"] == 2
    assert detail["breakout_rank"]["overall_total"] == 3
    assert detail["breakout_rank"]["pos"] == 2
    assert detail["forecast"] == {"band_label": "Tracking to hit"}


# ---------------------------------------------------------------------------
# Candidate detail prefers the historical call over a later non-call row
# ---------------------------------------------------------------------------

def _patch_detail_env(monkeypatch, surfaced_weeks, completed=(2, 5)):
    import dashboard_services.breakout_api as api

    monkeypatch.setattr(api, "_weekly_breakout_available", lambda season: True)
    monkeypatch.setattr(api, "get_weekly_breakout_detail",
                        lambda pid, season, as_of_week=None: {"available": True, "marker": "latest"})
    monkeypatch.setattr(api, "_surfaced_weeks_for", lambda pid, season: surfaced_weeks)
    monkeypatch.setattr(api, "get_historical_weekly_breakout_detail",
                        lambda pid, season, week: {"available": True, "marker": "historical", "week": week})
    monkeypatch.setattr(
        "data_building.breakout_engine.weekly_store.list_completed_weeks",
        lambda season: [{"as_of_week": w} for w in completed],
    )
    return api


def test_candidate_detail_serves_earliest_call_when_not_current(monkeypatch):
    api = _patch_detail_env(monkeypatch, surfaced_weeks=[2])

    out = api.get_breakout_candidate_detail("love", season=2026)

    assert out["marker"] == "historical"
    assert out["week"] == 2


def test_candidate_detail_serves_latest_when_surfaced_latest_week(monkeypatch):
    api = _patch_detail_env(monkeypatch, surfaced_weeks=[2, 5])

    out = api.get_breakout_candidate_detail("love", season=2026)

    assert out["marker"] == "latest"


def test_candidate_detail_no_history_keeps_latest(monkeypatch):
    api = _patch_detail_env(monkeypatch, surfaced_weeks=[])

    out = api.get_breakout_candidate_detail("love", season=2026)

    assert out["marker"] == "latest"


# ---------------------------------------------------------------------------
# weekly_store: an explicit week serves its own run, any scoring version
# ---------------------------------------------------------------------------

def test_get_weekly_candidate_explicit_week_uses_serving_run(monkeypatch):
    from data_building.breakout_engine import weekly_store

    captured = {}

    class _Conn:
        def execute(self, query, params):
            captured["query"] = query
            captured["params"] = params

            class _Row:
                def fetchone(self):
                    return {"player_id": "love", "as_of_week": 2, "run_id": 42}

            return _Row()

    @contextmanager
    def _conn():
        yield _Conn()

    monkeypatch.setattr(weekly_store, "init_weekly_breakout_db", lambda: None)
    monkeypatch.setattr(weekly_store, "get_conn", _conn)
    monkeypatch.setattr(weekly_store, "get_serving_run",
                        lambda season, week: {"id": 42, "scoring_version": "weekly-v3"})

    row = weekly_store.get_weekly_candidate("love", 2026, as_of_week=2)

    assert row["as_of_week"] == 2
    assert captured["params"] == ("love", 42)
    assert "scoring_version" not in captured["query"]


# ---------------------------------------------------------------------------
# Player endpoint membership carries the any-week call
# ---------------------------------------------------------------------------

def test_player_endpoint_membership_includes_history(monkeypatch):
    flask = pytest.importorskip("flask")
    import dashboard_services.breakout_api as api
    import dashboard_services.subscriptions as subscriptions

    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(api.breakout_bp)
    monkeypatch.setattr(
        api, "breakout_board_membership",
        lambda player_id, season: (False, {"season": 2026, "data_available": True}),
    )
    monkeypatch.setattr(
        api, "breakout_call_history",
        lambda player_id, season: {"kind": "weekly", "season": 2026,
                                    "as_of_week": 2, "label": "Week 2"},
    )
    monkeypatch.setattr(subscriptions, "has_premium_for_viewer", lambda *args: False)

    body = app.test_client().get("/api/breakout/player/love?season=2026").get_json()

    assert body["board_eligible"] is False
    assert body["has_breakout_call"] is True
    assert body["breakout_call"]["label"] == "Week 2"


# ---------------------------------------------------------------------------
# Modal JS: gate, week label, verdict stamp + receipt
# ---------------------------------------------------------------------------

def test_modal_gates_tab_on_any_breakout_call():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "data.on_breakout_board === true || data.has_breakout_call === true" in js
    assert "tabBreakout.style.display = _onBreakoutBoard ? '' : 'none'" in js


def test_modal_weekly_card_labels_the_call_week():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "data.breakout_call && data.breakout_call.label" in js
    assert "_pmEsc(callLabel) + ' call · '" in js


def test_modal_weekly_card_renders_verdict_pill_and_receipt():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "function _pmBkGradeHtml" in js
    # Solid grade pill in the board's chosen specs (hit green / partial
    # amber / miss red), styled inline so merge order cannot unstyle it.
    assert "'#16a34a'" in js and "'#d97706'" in js and "'#dc2626'" in js
    assert "font-weight:800;letter-spacing:0.05em;padding:3px 10px" in js
    # aria-label, never title: a title would be upgraded into a tooltip
    # bubble pinned to the card edge (the board's stamp bug).
    assert 'aria-label="Why this grade?"' in js
    assert 'title="Why this grade?"' not in js
    assert "function _pmBkGradeReceiptHtml" in js
    assert "data.grade_detail" in js
    # Receipt bars mirror the breakout page's grader bars.
    assert "'Production'" in js and "'Role (touches)'" in js and "'Role (snap share)'" in js
    assert "function pmToggleBkGradeReceipt" in js
    assert "html += _pmBkGradeHtml(data);" in js


def test_player_details_payload_carries_breakout_call():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '"has_breakout_call": bool(on_breakout_board or breakout_call)' in app
    assert '"breakout_call": breakout_call' in app
    assert "breakout_call = breakout_call_history(player_id, season)" in app
