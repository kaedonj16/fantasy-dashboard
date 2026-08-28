"""Cross-device league list sync: /api/my-leagues must stay fresh and clients
must refetch when the tab regains focus."""
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


def test_my_leagues_is_not_cacheable(offline_client, monkeypatch):
    import dashboard_services.accounts as accounts

    monkeypatch.setattr(
        accounts,
        "resolve_my_leagues",
        lambda viewer_user_id, account_id, current_season: (
            [{"platform": "sleeper", "league_id": "abc", "season": 2026, "name": "Test"}],
            2026,
        ),
    )
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 42

    response = offline_client.get("/api/my-leagues")
    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert len(data["leagues"]) == 1
    assert response.headers.get("Cache-Control") == "no-store"


def test_clients_refetch_my_leagues_without_browser_cache():
    script = (ROOT / "static" / "app.js").read_text()
    assert 'fetch("/api/my-leagues", { cache: "no-store" })' in script
    assert "fetch('/api/my-leagues', { cache: 'no-store' })" in script
    assert "window.refreshLeagueSwitcher" in script
    assert "refreshSavedLeaguesFromServer" in script
    assert "window.refreshHomeLeagues?.();" in script
    assert "window.refreshLeagueSwitcher?.();" in script


def test_my_leagues_endpoint_sets_no_store_header_in_source():
    source = (ROOT / "routes" / "league_meta_bp.py").read_text()
    block = source[source.index('def api_my_leagues'):source.index("@league_meta_bp.route(\"/api/weekly-trends\")")]
    assert 'resp.headers["Cache-Control"] = "no-store"' in block
