"""When roster_changes is empty, do not serve readiness-only 'breakout' lists."""
from pathlib import Path

import pytest

from data_building.breakout_opportunity_guard import (
    UNAVAILABLE_BREAKOUT_REASON,
    roster_changes_cover_season,
)


def test_zero_rows_does_not_cover_a_season():
    assert roster_changes_cover_season(0) is False
    assert roster_changes_cover_season(None) is False
    assert roster_changes_cover_season("n/a") is False


def test_any_roster_change_row_covers_a_season():
    assert roster_changes_cover_season(1) is True
    assert roster_changes_cover_season(400) is True


def test_unavailable_reason_mentions_roster_changes():
    assert "roster-change" in UNAVAILABLE_BREAKOUT_REASON.lower()


def test_get_breakout_candidates_stays_empty_without_roster_changes(monkeypatch):
    bapi = pytest.importorskip("dashboard_services.breakout_api")
    monkeypatch.setattr(bapi, "opportunity_data_ready", lambda season: False)
    out = bapi.get_breakout_candidates(season=2026, min_score=0)
    assert out["candidates"] == []
    assert out["count"] == 0
    assert out["data_available"] is False
    assert out["data_status"] == "unavailable"
    assert "roster-change" in (out.get("reason") or "").lower()


def test_get_breakout_candidates_queries_when_roster_changes_exist(monkeypatch):
    """Guard must not fire when data is present — the DB query is the next step."""
    bapi = pytest.importorskip("dashboard_services.breakout_api")
    monkeypatch.setattr(bapi, "opportunity_data_ready", lambda season: True)

    class _Boom(Exception):
        pass

    def _boom_conn():
        raise _Boom("would query breakout_opportunity_scores")

    monkeypatch.setattr(bapi, "get_conn", _boom_conn)
    with pytest.raises(_Boom):
        bapi.get_breakout_candidates(season=2026)


def test_player_detail_unavailable_without_roster_changes(monkeypatch):
    bapi = pytest.importorskip("dashboard_services.breakout_api")
    monkeypatch.setattr(bapi, "opportunity_data_ready", lambda season: False)
    out = bapi.get_breakout_candidate_detail("1234", season=2026)
    assert out["data_available"] is False
    assert "error" in out


def test_aligned_scores_empty_without_roster_changes(monkeypatch):
    bapi = pytest.importorskip("dashboard_services.breakout_api")
    monkeypatch.setattr(bapi, "opportunity_data_ready", lambda season: False)
    monkeypatch.setattr(bapi, "_resolve_bo_season", lambda requested: 2026)
    assert bapi.aligned_breakout_scores(["1234"], requested_season=2026) == {}


def test_offseason_candidates_empty_without_roster_changes(monkeypatch):
    off = pytest.importorskip("data_building.offseason_opportunity")
    helpers = pytest.importorskip("data_building.breakout_engine.db_helpers")
    monkeypatch.setattr(helpers, "opportunity_data_ready", lambda season: False)
    assert off.get_offseason_breakout_candidates(2026) == []


def test_breakouts_page_has_unavailable_empty_copy():
    src = (Path(__file__).parents[1] / "app.py").read_text()
    assert 'id="breakoutsEmptyTitle"' in src
    assert "data.data_available === false" in src
    assert "Breakout data is not ready" in src
