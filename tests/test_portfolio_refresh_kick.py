"""Refresh Data kick-and-return contract (2026-10-27 reliability fix).

POST /api/portfolio/refresh used to bust each league's cache and then
rebuild it synchronously on the request thread: 14-120s per batch of 4
against the client's 30s timeout, un-cancellable once accepted, and a
re-tap piled a second build onto the same semaphores. It now busts and
kicks the existing background summary-refresh machinery (bounded pool,
per-league single-flight) and returns accepted/pending immediately;
the client polls /api/portfolio/card for completion.
"""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")


def _membership(league_id="L1"):
    return {"platform": "sleeper", "league_id": league_id, "season": 2026,
            "name": f"League {league_id}"}


def _wire(monkeypatch, leagues=("L1", "L2")):
    import routes.user_pages_bp as pages
    monkeypatch.setattr(
        "dashboard_services.accounts.resolve_account_leagues",
        lambda *a, **k: [_membership(lid) for lid in leagues])
    return pages


def _sign_in(client):
    with client.session_transaction() as sess:
        sess["account_id"] = 7


def test_refresh_requires_account(offline_client, monkeypatch):
    _wire(monkeypatch)
    resp = offline_client.post(
        "/api/portfolio/refresh",
        json={"leagues": [{"platform": "sleeper", "league_id": "L1", "season": 2026}]})
    assert resp.status_code == 401


def test_refresh_validation_contract_unchanged(offline_client, monkeypatch):
    _wire(monkeypatch)
    _sign_in(offline_client)
    # account_id in the payload is still rejected.
    resp = offline_client.post(
        "/api/portfolio/refresh",
        json={"account_id": 7,
              "leagues": [{"platform": "sleeper", "league_id": "L1", "season": 2026}]})
    assert resp.status_code == 400
    # More than 4 leagues per POST is still rejected.
    resp = offline_client.post(
        "/api/portfolio/refresh",
        json={"leagues": [{"platform": "sleeper", "league_id": f"L{i}", "season": 2026}
                          for i in range(5)]})
    assert resp.status_code == 400
    # An unlinked league is still a 403.
    resp = offline_client.post(
        "/api/portfolio/refresh",
        json={"leagues": [{"platform": "sleeper", "league_id": "NOPE", "season": 2026}]})
    assert resp.status_code == 403


def test_refresh_kicks_and_returns_without_sync_build(offline_client, monkeypatch):
    import app as appmod
    pages = _wire(monkeypatch)
    _sign_in(offline_client)

    def _no_sync_build(*a, **k):
        raise AssertionError("refresh must not build summaries on the request thread")

    monkeypatch.setattr(
        "dashboard_services.portfolio_summary.build_league_summary", _no_sync_build)
    kicks = []
    monkeypatch.setattr(
        pages, "_kick_summary_refresh",
        lambda account_id, membership: kicks.append(membership["league_id"]))
    # Seed a warm cache entry for L1: the refresh must bust it (ts -> 0).
    cache_key = appmod._cache_key("sleeper", 2026, "L1")
    appmod.DASHBOARD_CACHE[cache_key] = {"ctx": {}, "ts": 12345.0, "page_html": {}}
    try:
        resp = offline_client.post(
            "/api/portfolio/refresh",
            json={"leagues": [
                {"platform": "sleeper", "league_id": "L1", "season": 2026},
                {"platform": "sleeper", "league_id": "L2", "season": 2026},
            ]})
        # The seeded warm entry was busted by the refresh (ts reset to 0).
        assert appmod.DASHBOARD_CACHE[cache_key]["ts"] == 0
    finally:
        appmod.DASHBOARD_CACHE.pop(cache_key, None)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["accepted"] is True
    assert body["refreshed_at"] is None
    # Positional per-league results, pending, claiming no summary.
    assert [r["league_id"] for r in body["results"]] == ["L1", "L2"]
    assert all(r["ok"] is True and r["pending"] is True for r in body["results"])
    assert all("summary" not in r for r in body["results"])
    # One background kick per requested league, and L1's cache was busted.
    assert kicks == ["L1", "L2"]


def test_summary_refresh_kick_is_single_flight_per_league(monkeypatch):
    """A second kick for a league whose rebuild is already in flight
    must not stack another build on the bounded pool; a different
    league still kicks."""
    pytest.importorskip("flask")
    import routes.user_pages_bp as pages

    submitted = []

    class _FakePool:
        def submit(self, fn):
            submitted.append(fn)  # never runs: the build stays "in flight"

    monkeypatch.setattr(pages, "_portfolio_refresh_pool", lambda: _FakePool())
    pages._SUMMARY_REFRESH_INFLIGHT.clear()
    try:
        pages._kick_summary_refresh(7, _membership("L1"))
        pages._kick_summary_refresh(7, _membership("L1"))  # duplicate: coalesced
        pages._kick_summary_refresh(7, _membership("L2"))
        assert len(submitted) == 2
    finally:
        pages._SUMMARY_REFRESH_INFLIGHT.clear()
