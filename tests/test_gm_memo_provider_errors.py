"""Regression tests: /api/gm-memo must surface provider outages as 503/403/404 JSON.

Incident (2026-09-27): Fleaflicker's API errored during a GM memo request and
/api/gm-memo returned 500 {"error": "Internal error"} after ~20s. The endpoint
caught every exception and flattened provider errors, even though the app
already maps ProviderUnavailableError to a 503 for page routes.
"""

import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("pandas")

from dashboard_services.providers.base import (
    LeagueNotFoundError,
    ProviderAuthenticationError,
    ProviderUnavailableError,
)


def _post_memo(client, monkeypatch, exc):
    import app as appmod

    monkeypatch.setattr(appmod, "has_premium_for_viewer", lambda *a, **k: True)

    def _raise(*a, **k):
        raise exc

    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", _raise)
    return client.post(
        "/api/gm-memo",
        json={
            "league_id": "92916",
            "season": 2026,
            "platform": "fleaflicker",
            "viewer_roster_id": "1",
        },
    )


def test_gm_memo_provider_unavailable_is_503_json(offline_client, monkeypatch):
    resp = _post_memo(
        offline_client, monkeypatch,
        ProviderUnavailableError("Fleaflicker is temporarily unavailable."),
    )
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["success"] is False
    assert "temporarily unavailable" in data["error"]
    assert data["error"] != "Internal error"


def test_gm_memo_provider_auth_is_403_json(offline_client, monkeypatch):
    resp = _post_memo(
        offline_client, monkeypatch,
        ProviderAuthenticationError("This Fleaflicker league is private or requires authentication."),
    )
    assert resp.status_code == 403
    data = resp.get_json()
    assert data["success"] is False
    assert "private" in data["error"]


def test_gm_memo_league_not_found_is_404_json(offline_client, monkeypatch):
    resp = _post_memo(
        offline_client, monkeypatch,
        LeagueNotFoundError("No Fleaflicker league was found for that ID and season."),
    )
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["success"] is False
    assert "No Fleaflicker league" in data["error"]


def test_gm_memo_unexpected_error_still_500(offline_client, monkeypatch):
    resp = _post_memo(offline_client, monkeypatch, RuntimeError("boom"))
    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Internal error"
