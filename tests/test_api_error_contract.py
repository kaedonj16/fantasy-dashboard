"""API error contract: /api/* failures must be JSON with 4xx/5xx codes,
never branded HTML, and bad query params must 400 instead of 500."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")


def _as_json(resp):
    assert resp.content_type.startswith("application/json"), resp.content_type
    return resp.get_json()


def test_api_404_returns_json_envelope(offline_client):
    resp = offline_client.get("/api/definitely-not-a-route")
    assert resp.status_code == 404
    body = _as_json(resp)
    assert body["error"]
    assert body["code"] == "not_found"


def test_page_404_stays_html(offline_client):
    resp = offline_client.get("/definitely-not-a-page")
    assert resp.status_code == 404
    assert resp.content_type.startswith("text/html")


def test_page_404_json_when_accept_json(offline_client):
    resp = offline_client.get(
        "/definitely-not-a-page", headers={"Accept": "application/json"}
    )
    assert resp.status_code == 404
    body = _as_json(resp)
    assert body["code"] == "not_found"


def test_api_500_returns_json_envelope(offline_client, monkeypatch):
    import app as appmod

    endpoint = "health.api_health_errors"

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setitem(appmod.app.view_functions, endpoint, _boom)
    monkeypatch.setitem(appmod.app.config, "TESTING", False)
    monkeypatch.setitem(appmod.app.config, "PROPAGATE_EXCEPTIONS", False)
    resp = offline_client.get("/api/health/errors")
    assert resp.status_code == 500
    body = _as_json(resp)
    assert body["code"] == "internal_error"
    assert "Traceback" not in body["error"]


def test_api_400_on_bad_season_param(offline_client):
    resp = offline_client.get("/api/sleeper-user-leagues?username=abc&season=abc")
    assert resp.status_code == 400
    body = _as_json(resp)
    assert body["code"] == "bad_request"
    assert "season" in body["error"]


def test_api_400_on_bad_history_season_param(offline_client):
    resp = offline_client.get(
        "/api/history/sleeper/2026/12345/chart?history_season=abc"
    )
    assert resp.status_code == 400
    body = _as_json(resp)
    assert body["code"] == "bad_request"


def test_api_method_not_allowed_is_json(offline_client):
    resp = offline_client.post("/api/sleeper-user-leagues")
    assert resp.status_code == 405
    body = _as_json(resp)
    assert body["code"] == "method_not_allowed"


def test_history_chart_empty_state_is_200_without_error_key(offline_client):
    # No history seasons available -> empty-state HTML, 200, no "error" key.
    resp = offline_client.get("/api/history/sleeper/2026/12345/chart")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "error" not in body
    assert "history-empty" in body.get("html", "")
