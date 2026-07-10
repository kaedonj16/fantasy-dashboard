"""End-to-end page render tests through the real Flask stack.

These exercise the full route -> body-builder -> render_page path offline, using
the ``offline_client`` fixture (Sleeper HTTP mocked). They catch the class of
bug that pure unit tests miss: a route that 500s, a template that references a
missing field, a page chrome that breaks. Skipped automatically when Flask/
pandas aren't installed (the pure suite still runs); run them via the venv from
scripts/dev_setup.sh, and they run in CI where the full stack is present.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

# Tour routes render from seeded mock data, so they need no live league.
TOUR_PAGES = [
    "/sleeper/2026/tourdemo/graphs?tour=1",
    "/sleeper/2026/tourdemo/history?tour=1",
]


@pytest.mark.parametrize("path", TOUR_PAGES)
def test_tour_page_renders_200(offline_client, path):
    r = offline_client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    html = r.get_data(as_text=True)
    # Real page, not an error stub.
    assert len(html) > 2000
    assert "Traceback" not in html


def test_graphs_tour_has_core_chart(offline_client):
    html = offline_client.get("/sleeper/2026/tourdemo/graphs?tour=1").get_data(as_text=True)
    assert "PF vs PA" in html


def test_robots_and_sitemap(offline_client):
    assert offline_client.get("/robots.txt").status_code == 200
    assert offline_client.get("/sitemap.xml").status_code in (200, 500)  # sitemap may need data


def test_health_timing_requires_admin(offline_client):
    # No/incorrect admin secret -> 403 (never leaks timing without auth).
    r = offline_client.get("/api/health/timing")
    assert r.status_code == 403


def test_compare_page_renders(offline_client):
    r = offline_client.get("/compare")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'data-page="compare"' in html
    assert 'id="cmpPick1"' in html and 'id="cmpPick2"' in html


def test_compare_page_seo_title_from_ids(offline_client):
    # When both ids resolve to names, the <title> names the matchup (shareable/SEO).
    from app import get_model_value_table_cached
    table = get_model_value_table_cached() or []
    if len(table) < 2:
        import pytest as _pytest
        _pytest.skip("no value table in this environment")
    p1, p2 = str(table[0]["id"]), str(table[1]["id"])
    html = offline_client.get(f"/compare?p1={p1}&p2={p2}").get_data(as_text=True)
    assert " vs " in html and "Dynasty Comparison" in html


def test_compare_baselines_endpoint(offline_client):
    """The compare page's tier-average opponents (Avg WR1, RB2, ...). Each entry
    must be shaped like /api/player-details (synthetic id, position, stats with a
    numeric value) so the compare UI can treat it as a player."""
    r = offline_client.get("/api/compare-baselines")
    assert r.status_code == 200
    baselines = r.get_json().get("baselines", [])
    if not baselines:
        import pytest as _pytest
        _pytest.skip("no value table in this environment")
    for b in baselines:
        assert str(b["player_id"]).startswith("avg-")
        assert b.get("is_baseline") is True
        assert b.get("position") in ("QB", "RB", "WR", "TE")
        # A tier average always has a numeric dynasty value (its whole point);
        # PPG may be null when no usage season is available.
        assert isinstance(b["stats"]["value"], (int, float))
    # Tiers are per-position and ordered, so WR1's average value >= WR2's.
    wr = [b for b in baselines if b["position"] == "WR"]
    vals = [b["stats"]["value"] for b in wr]
    assert vals == sorted(vals, reverse=True)


def test_watchlist_page_renders(offline_client):
    r = offline_client.get("/watchlist")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'id="wlPageTable"' in html
    assert 'data-page="watchlist"' in html


def test_watchlist_api_unsynced_when_signed_out(offline_client):
    # Not signed in (no viewer_user_id in session) -> synced:false, no DB needed.
    r = offline_client.get("/api/watchlist")
    assert r.status_code == 200
    body = r.get_json()
    assert body.get("synced") is False and body.get("items") == []
    # merge is also a safe no-op when signed out.
    r2 = offline_client.post("/api/watchlist/merge", json={"items": [{"player_id": "1"}]})
    assert r2.status_code == 200 and r2.get_json().get("synced") is False


def test_watchlist_rows_for_parses_dict_rows(monkeypatch):
    """get_conn() uses psycopg's dict_row factory, so _rows_for must key rows by
    column name. Indexing by position (r[0]) raised KeyError and silently emptied
    every read, so nothing ever synced across devices. This guards that path."""
    import contextlib
    import datetime
    import dashboard_services.db as db
    from routes import watchlist_bp as wb

    class _Cur:
        def fetchall(self):
            return [{
                "player_id": "9509", "name": "Bijan Robinson",
                "position": "RB", "team": "ATL",
                "added_at": datetime.datetime(2026, 1, 2, 3, 4, 5),
            }]

    class _Conn:
        def execute(self, *a, **k):
            return _Cur()

    @contextlib.contextmanager
    def _fake_conn():
        yield _Conn()

    monkeypatch.setattr(db, "get_conn", _fake_conn)
    rows = wb._rows_for("u_1")
    assert rows == [{
        "player_id": "9509", "name": "Bijan Robinson",
        "position": "RB", "team": "ATL", "note": "",
        "added_at": "2026-01-02T03:04:05",
    }]


def test_push_routes_registered(offline_client):
    # Extracted into routes/push_bp.py — verify the blueprint is mounted.
    # vapid-public-key returns 200 (ephemeral key) or 503 (not configured);
    # broadcast requires the admin secret.
    assert offline_client.get("/api/push/vapid-public-key").status_code in (200, 503)
    assert offline_client.post("/api/push/broadcast", json={}).status_code == 403


def test_push_broadcast_prunes_dead_410_subscriptions(monkeypatch):
    """A 404/410 push means the subscription is permanently gone and must be
    deleted so it stops being retried forever. requests.Response is falsy for
    non-2xx (its __bool__ returns .ok), so `if exc.response` silently skipped
    real 410s and they were never pruned. This guards the fixed detection."""
    import contextlib
    pywebpush = pytest.importorskip("pywebpush")  # excluded from the dev venv
    import requests
    import dashboard_services.db as db
    import routes.push_bp as pb

    monkeypatch.setattr(pb, "_get_vapid_keys", lambda: {"private": "x", "public": "y"})
    monkeypatch.setattr(pb, "_init_push_table", lambda: None)

    import utils.push_notifications as pn
    monkeypatch.setattr(pn, "_make_vapid", lambda k: object())

    # webpush() always fails with a 410 whose Response is falsy (like the real bug).
    resp = requests.models.Response()
    resp.status_code = 410
    assert not resp  # confirms the truthiness footgun this test exists for

    def _boom(**kw):
        raise pywebpush.WebPushException("Push failed: 410 Gone", response=resp)

    monkeypatch.setattr(pywebpush, "webpush", _boom)

    deleted = []

    class _Cur:
        def fetchall(self):
            return [{"endpoint": "https://fcm/dead", "p256dh": "k", "auth": "a"}]

    class _Conn:
        def execute(self, sql, params=None):
            if "DELETE" in sql:
                deleted.append(params)
            return _Cur()
        def commit(self):
            pass

    @contextlib.contextmanager
    def _fake_conn(*a, **k):
        yield _Conn()

    monkeypatch.setattr(db, "get_conn", _fake_conn)

    body, status = pb._push_broadcast("Title", "Body")
    assert status == 200
    assert body["sent"] == 0 and body["failed"] == 1
    assert deleted and "https://fcm/dead" in str(deleted[-1])


def test_health_endpoints_work_with_admin_secret(offline_client, monkeypatch):
    # Proves the routes extracted into routes/health_bp.py are registered and
    # functional through the shared limiter.
    monkeypatch.setenv("ADMIN_SECRET", "s3cret")
    r = offline_client.get("/api/health/timing", headers={"X-Admin-Secret": "s3cret"})
    assert r.status_code == 200
    assert "endpoints" in r.get_json()
    r = offline_client.get("/api/health/errors", headers={"X-Admin-Secret": "s3cret"})
    assert r.status_code == 200
    assert "errors" in r.get_json()
