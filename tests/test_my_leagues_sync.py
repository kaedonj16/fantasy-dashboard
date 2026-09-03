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
    assert data["leagues"][0]["name"] == "Test"
    assert "format" not in data["leagues"][0]
    assert response.headers.get("Cache-Control") == "no-store"


def test_my_leagues_includes_sleeper_format_when_slots_known(offline_client, monkeypatch):
    import dashboard_services.accounts as accounts

    monkeypatch.setattr(
        accounts,
        "resolve_my_leagues",
        lambda viewer_user_id, account_id, current_season: (
            [{
                "platform": "sleeper",
                "league_id": "1389346724446756865",
                "season": 2026,
                "name": "KC fantasy-yearly",
                "total_rosters": 8,
                "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX"],
            }],
            2026,
        ),
    )
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 42

    response = offline_client.get("/api/my-leagues")
    assert response.status_code == 200
    row = response.get_json()["leagues"][0]
    assert row["name"] == "KC fantasy-yearly"
    assert row["sf"] is True
    assert row["size"] == 8
    assert row["format"] == "8tm SF"


def test_clients_refetch_my_leagues_without_browser_cache():
    script = (ROOT / "static" / "app.js").read_text()
    assert 'fetch("/api/my-leagues", { cache: "no-store" })' in script
    assert "fetch('/api/my-leagues', { cache: 'no-store' })" in script
    assert "window.refreshLeagueSwitcher" in script
    assert "refreshSavedLeaguesFromServer" in script
    assert "window.refreshHomeLeagues?.();" in script
    assert "window.refreshLeagueSwitcher?.();" in script
    assert "function paintLeagueChromeChip" in script
    assert "paintLeagueChromeChip(cur)" in script


def test_my_leagues_endpoint_sets_no_store_header_in_source():
    source = (ROOT / "routes" / "league_meta_bp.py").read_text()
    block = source[source.index('def api_my_leagues'):source.index("@league_meta_bp.route(\"/api/weekly-trends\")")]
    assert 'resp.headers["Cache-Control"] = "no-store"' in block
