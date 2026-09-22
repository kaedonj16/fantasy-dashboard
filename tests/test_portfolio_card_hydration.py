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
    source = (ROOT / "static" / "app.js").read_text()
    assert "var MAX_REQUESTS = 2" in source
    assert "/api/portfolio/card?" in source
    assert "Matchup temporarily unavailable" in source
    assert "RETRY_DELAYS = [3000, 6000, 10000, 15000, 25000]" in source
    assert "owner.active < MAX_REQUESTS" in source


def test_my_leagues_loader_is_single_flight_and_force_invalidates():
    source = (ROOT / "static" / "app.js").read_text()
    block = source.split("window.brGetMyLeagues = function", 1)[1].split("})();", 1)[0]
    assert "if (inflight) return inflight" in block
    assert "if (force) { value = null; expiresAt = 0; }" in block
    assert ".finally(function(){ inflight=null; })" in block
    assert "expiresAt = Date.now() + 5000" in block


def test_hydration_liveness_guard_matches_rendered_grid():
    """The queue must not self-disable before issuing its first card request."""
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body", 1)[1].split("\ndef ", 1)[0]
    assert "<div class='pf-lg-grid'>" in fn
    assert "Portfolio cards are started by initPageRoot" in fn
    assert "pf-leagues-grid" not in fn


def test_live_polling_reuses_card_queue_after_initial_hydration():
    source = (ROOT / "static" / "app.js").read_text()
    assert "slot._isLive = data.status === 'in'" in source
    assert "if (slot && slot._isLive) schedule(owner, card)" in source
    assert "if (!document.hidden) pump(owner)" in source
    assert "window.__pfQueueCard" in source


def test_warm_and_loading_cards_have_the_same_refresh_hooks():
    """Rendered HTML, rather than source text, proves mixed cards are eligible."""
    import app
    warm = {"league_id": "W", "platform": "sleeper", "season": 2026,
            "name": "Warm", "wins": 2, "losses": 1, "record": "2-1",
            "rank": 2, "total_teams": 10, "streak": ["W"],
            "pos_user_vals": {}, "pos_league_avgs": {}, "pos_user_rank": {}}
    cold = {"league_id": "C", "platform": "espn", "season": 2026,
            "name": "Cold", "loading": True}
    with app.app.test_request_context("/portfolio"):
        rendered = app.build_portfolio_body("viewer", [warm], [warm, cold], 2026)
    assert rendered.count("data-summary-card") == 2
    assert rendered.count("data-summary-stats") == 2
    assert "data-platform='sleeper' data-league-id='W' data-season='2026'" in rendered
    assert "data-platform='espn' data-league-id='C' data-season='2026'" in rendered
