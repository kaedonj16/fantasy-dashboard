"""Regression tests for league-page blueprint dependencies."""

import pytest

pytest.importorskip("flask")

from dashboard_services.pages.waivers_page import (
    build_waivers_body as service_build_waivers_body,
)
from dashboard_services.providers.base import ProviderUnavailableError
from routes.league_pages_bp import build_waivers_body


def test_waivers_page_uses_extracted_service_builder():
    """The builder no longer exists in app.py after its service extraction."""
    assert build_waivers_body is service_build_waivers_body


def _disable_exception_propagation(app_module):
    """Flask TESTING mode re-raises instead of running errorhandlers."""
    prev = app_module.app.config.get("PROPAGATE_EXCEPTIONS")
    app_module.app.config["PROPAGATE_EXCEPTIONS"] = False
    return prev


def test_waivers_page_returns_503_when_fleaflicker_unavailable(offline_client, monkeypatch):
    """Production 500: FetchLeagueStandings HTML 403 became an uncaught 500."""
    import app as appmod

    def _boom(*_args, **_kwargs):
        raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")

    monkeypatch.setattr(appmod, "get_league", _boom)
    monkeypatch.setattr(appmod, "get_users", _boom)
    monkeypatch.setattr(appmod, "get_rosters", _boom)
    appmod.DASHBOARD_CACHE.clear()

    prev = _disable_exception_propagation(appmod)
    try:
        response = offline_client.get("/fleaflicker/2026/92916/waivers")
    finally:
        appmod.app.config["PROPAGATE_EXCEPTIONS"] = prev

    assert response.status_code == 503
    html = response.get_data(as_text=True)
    assert "temporarily unavailable" in html.lower()
    assert "Something went wrong" not in html


def test_waivers_page_returns_503_when_ctx_cache_raises(offline_client, monkeypatch):
    """Any league page that loads context should map provider outages to 503."""
    import app as appmod

    def _boom(*_args, **_kwargs):
        raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")

    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", _boom)
    prev = _disable_exception_propagation(appmod)
    try:
        response = offline_client.get("/fleaflicker/2026/92916/waivers")
    finally:
        appmod.app.config["PROPAGATE_EXCEPTIONS"] = prev

    assert response.status_code == 503
    assert b"Fleaflicker is temporarily unavailable" in response.data


def test_build_week_activity_survives_provider_outage(monkeypatch):
    """The waivers 500 stack was get_users inside build_week_activity."""
    pytest.importorskip("pandas")
    from dashboard_services.service import build_week_activity

    def _boom(*_args, **_kwargs):
        raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")

    monkeypatch.setattr("dashboard_services.players.get_users", _boom)
    monkeypatch.setattr("dashboard_services.players.get_rosters", _boom)
    frame = build_week_activity("92916", "fleaflicker", 2026, {})
    assert frame.empty
    assert list(frame.columns) == ["kind", "week", "ts", "data"]


def test_build_week_activity_skips_refetch_when_rosters_provided(monkeypatch):
    pytest.importorskip("pandas")
    from dashboard_services.service import build_week_activity

    def _boom(*_args, **_kwargs):
        raise AssertionError("should not refetch users/rosters")

    monkeypatch.setattr("dashboard_services.players.get_users", _boom)
    monkeypatch.setattr("dashboard_services.players.get_rosters", _boom)
    monkeypatch.setattr(
        "dashboard_services.service.get_transactions_by_week",
        lambda *_args, **_kwargs: {},
    )
    frame = build_week_activity(
        "92916", "fleaflicker", 2026, {}, users=[], rosters=[],
    )
    assert frame.empty


def test_get_transactions_by_week_logs_once_for_many_week_failures(caplog, monkeypatch):
    """18 parallel week fetches used to print the same Fleaflicker outage."""
    import logging
    from dashboard_services.providers.base import ProviderUnavailableError
    from dashboard_services import service as svc

    def _boom(*_a, **_k):
        raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")

    monkeypatch.setattr(svc, "platform_get_transactions", _boom)
    with caplog.at_level(logging.WARNING, logger=svc.logger.name):
        result = svc.get_transactions_by_week(
            "92916", list(range(0, 19)), platform="fleaflicker", season=2026,
        )
    assert result[1] == []
    assert result[18] == []
    warnings = [r for r in caplog.records if "[transactions]" in r.getMessage()]
    assert len(warnings) == 1
    assert "19 week(s) failed" in warnings[0].getMessage()


def test_build_league_context_draft_warning_is_rate_limited(caplog, monkeypatch):
    """A Fleaflicker blip used to reprint the same drafts warning per page build."""
    import logging
    import app as appmod

    appmod._CTX_TASK_WARN_TS.clear()
    with caplog.at_level(logging.WARNING, logger=appmod.logger.name):
        appmod._warn_league_ctx_once(
            "drafts", "92916",
            "[build_league_context] failed to load drafts for league %s: %s",
            "92916", "Fleaflicker is temporarily unavailable.",
        )
        appmod._warn_league_ctx_once(
            "drafts", "92916",
            "[build_league_context] failed to load drafts for league %s: %s",
            "92916", "Fleaflicker is temporarily unavailable.",
        )
    warnings = [r for r in caplog.records if "failed to load drafts" in r.getMessage()]
    assert len(warnings) == 1
