"""Regression coverage for cache-only, consolidated Portfolio hydration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _membership():
    return {"platform": "sleeper", "league_id": "L1", "season": 2026, "name": "League"}


def test_cold_summary_uses_cache_only_loader(offline_client, monkeypatch):
    import routes.user_pages_bp as pages
    seen = []
    monkeypatch.setattr("dashboard_services.accounts.resolve_account_leagues", lambda *a, **k: [_membership()])
    monkeypatch.setattr("dashboard_services.portfolio_summary.get_cached_summary", lambda *a: None)
    monkeypatch.setattr(pages, "get_league_ctx_from_cache",
                        lambda *a, **k: seen.append(k.get("allow_build")) or {})
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 7
    response = offline_client.get("/api/portfolio/summary?platform=sleeper&league_id=L1&season=2026")
    assert response.status_code == 200
    assert response.json["pending"] is True
    assert seen == [False]


def test_cold_card_returns_last_good_while_warming(offline_client, monkeypatch):
    import routes.user_pages_bp as pages
    stale = {"state": "ready", "record": "8-2", "league_id": "L1"}
    monkeypatch.setattr("dashboard_services.accounts.resolve_account_leagues", lambda *a, **k: [_membership()])
    monkeypatch.setattr("dashboard_services.portfolio_summary.get_cached_summary", lambda *a: stale)
    monkeypatch.setattr(pages, "get_league_ctx_from_cache", lambda *a, **k: {})
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 7
    response = offline_client.get("/api/portfolio/card?platform=sleeper&league_id=L1&season=2026")
    assert response.status_code == 200
    assert response.json["state"] == "pending"
    assert response.json["summary"]["record"] == "8-2"
    assert response.json["matchup"]["pending"] is True


def test_card_rejects_unlinked_league(offline_client, monkeypatch):
    monkeypatch.setattr("dashboard_services.accounts.resolve_account_leagues", lambda *a, **k: [])
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 7
    response = offline_client.get("/api/portfolio/card?platform=sleeper&league_id=NOPE&season=2026")
    assert response.status_code == 403


def test_client_has_one_bounded_queue_and_stable_matchup_failure():
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body", 1)[1].split("\ndef ", 1)[0]
    assert "active=0,MAX=2,q=[]" in fn
    assert "/api/portfolio/card?" in fn
    assert "25000" in fn
    assert "Matchup temporarily unavailable" in fn
    assert "data-matchup-retry" in fn
    assert "cards=[].slice.call(document.querySelectorAll('.pf-lg-card" in fn
    assert "for(var k=0;k<3;k++)pump();" not in fn
    assert "[3000,6000,10000,15000,25000]" in fn


def test_my_leagues_loader_is_single_flight_and_force_invalidates():
    source = (ROOT / "static" / "app.js").read_text()
    block = source.split("window.brGetMyLeagues = function", 1)[1].split("})();", 1)[0]
    assert "if (inflight) return inflight" in block
    assert "if (force) { value = null; expiresAt = 0; }" in block
    assert ".finally(function () { inflight = null; })" in block
    assert "expiresAt = Date.now() + 5000" in block
