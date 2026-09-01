"""Smoke tests for the UI audit mock league and /ui-audit hub."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from utils.ui_audit_fixture import (
    UI_AUDIT_LEAGUE_ID,
    all_audit_hrefs,
    bootstrap_viewer_session,
    build_ui_audit_league_context,
    install_ui_audit_hooks,
    league_page_href,
)


@pytest.fixture
def ui_audit_client(monkeypatch, offline_client):
    monkeypatch.setenv("UI_AUDIT", "1")
    install_ui_audit_hooks()
    import app as appmod

    monkeypatch.setattr(appmod, "daily_completed", __import__("datetime").date.today(), raising=False)
    return offline_client


def test_build_ui_audit_league_context_has_rosters():
    ctx = build_ui_audit_league_context()
    assert ctx["league_id"] == UI_AUDIT_LEAGUE_ID
    assert ctx["offseason_mode"] is False
    assert ctx["current_week"] == 11
    assert len(ctx.get("rosters") or []) == 10
    assert len(ctx.get("users") or []) == 10
    assert not ctx["df_weekly"].empty
    assert ctx["team_stats"] is not None and not ctx["team_stats"].empty


def test_ui_audit_hub_requires_flag(offline_client):
    r = offline_client.get("/ui-audit")
    assert r.status_code == 404


def test_ui_audit_hub_lists_pages(ui_audit_client):
    r = ui_audit_client.get("/ui-audit")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "UI Audit Hub" in html
    assert league_page_href("dashboard") in html
    assert "/pricing" in html


def test_bootstrap_seeds_session(ui_audit_client):
    r = ui_audit_client.get("/ui-audit/bootstrap", follow_redirects=False)
    assert r.status_code in (302, 303)
    with ui_audit_client.session_transaction() as sess:
        assert sess.get("viewer_team_name") == "Audit Team A"
        assert sess.get("last_league_id") == UI_AUDIT_LEAGUE_ID


def test_weekly_hub_renders_in_season(ui_audit_client):
    with ui_audit_client.session_transaction() as sess:
        bootstrap_viewer_session(sess)
    r = ui_audit_client.get(league_page_href("weekly"), follow_redirects=True)
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Weekly Hub Unavailable" not in html


@pytest.mark.parametrize("href,label", all_audit_hrefs())
def test_audit_route_renders(ui_audit_client, href, label):
    with ui_audit_client.session_transaction() as sess:
        bootstrap_viewer_session(sess)
    r = ui_audit_client.get(href, follow_redirects=True)
    assert r.status_code == 200, f"{label} ({href}) -> {r.status_code}"
