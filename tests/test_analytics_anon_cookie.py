"""End-to-end tests for the persistent anonymous analytics cookie.

The pageview hook lives in app.py; these tests drive the real Flask app
with track_event faked out (no live database). Probe routes are trivial
HTML pages; the hook under test is the app's own after_request handler.
"""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

import app as app_module  # noqa: E402
from dashboard_services import analytics  # noqa: E402

_flask_app = app_module.app


def _probe_page():
    return "<html><body>probe</body></html>"


def _probe_api():
    return "<html><body>api</body></html>"


_flask_app.add_url_rule(
    "/_anon_cookie_probe", endpoint="anon_cookie_probe_page",
    view_func=_probe_page,
)
_flask_app.add_url_rule(
    "/api/_anon_cookie_probe", endpoint="anon_cookie_probe_api",
    view_func=_probe_api,
)
_flask_app.add_url_rule(
    "/admin/_anon_cookie_probe", endpoint="anon_cookie_probe_admin",
    view_func=_probe_page,
)


@pytest.fixture()
def recorded(monkeypatch):
    calls = []

    def fake_track_event(event, account_id=None, session_id=None,
                          path=None, props=None):
        calls.append({
            "event": event,
            "account_id": account_id,
            "session_id": session_id,
            "path": path,
            "props": props,
        })

    monkeypatch.setattr(analytics, "track_event", fake_track_event)
    monkeypatch.setattr(
        "dashboard_services.admin_auth.is_admin", lambda: False
    )
    return calls


@pytest.fixture()
def client():
    return _flask_app.test_client()


def _anon_cookie_headers(resp):
    return [h for h in resp.headers.getlist("Set-Cookie")
            if h.startswith(analytics.ANON_COOKIE_NAME + "=")]


def test_first_anonymous_pageview_sets_persistent_cookie(client, recorded):
    resp = client.get("/_anon_cookie_probe")
    assert resp.status_code == 200
    cookies = _anon_cookie_headers(resp)
    assert len(cookies) == 1
    header = cookies[0]
    value = header.split(";", 1)[0].split("=", 1)[1]
    assert len(value) == 32
    assert "Max-Age=31536000" in header
    assert "Path=/" in header
    assert "SameSite=Lax" in header
    assert "HttpOnly" in header
    if _flask_app.config.get("SESSION_COOKIE_SECURE"):
        assert "Secure" in header
    # The recorded pageview carries the same identity the cookie persists.
    assert len(recorded) == 1
    assert recorded[0]["event"] == analytics.EVENT_PAGEVIEW
    assert recorded[0]["account_id"] is None
    assert recorded[0]["session_id"] == value


def test_return_visit_reuses_cookie_identity(client, recorded):
    client.get("/_anon_cookie_probe")
    assert len(recorded) == 1
    sid = recorded[0]["session_id"]
    recorded.clear()
    resp = client.get("/_anon_cookie_probe")
    assert len(recorded) == 1
    assert recorded[0]["session_id"] == sid
    # The cookie already matches, so it is not re-stamped.
    assert _anon_cookie_headers(resp) == []


def test_session_only_sid_is_adopted_into_cookie(client, recorded):
    adopted = "0123456789abcdef" * 2
    with client.session_transaction() as sess:
        sess["_analytics_sid"] = adopted
    resp = client.get("/_anon_cookie_probe")
    assert len(recorded) == 1
    assert recorded[0]["session_id"] == adopted
    cookies = _anon_cookie_headers(resp)
    assert len(cookies) == 1
    assert cookies[0].split(";", 1)[0] == "_analytics_sid=" + adopted


def test_signed_in_pageview_records_account_identity(client, recorded):
    with client.session_transaction() as sess:
        sess["account_id"] = 42
    resp = client.get("/_anon_cookie_probe")
    assert resp.status_code == 200
    assert len(recorded) == 1
    assert recorded[0]["account_id"] == 42
    # The anon id is still resolved alongside, exactly as before.
    assert recorded[0]["session_id"]


def test_api_and_admin_paths_record_and_set_nothing(client, recorded):
    resp_api = client.get("/api/_anon_cookie_probe")
    resp_admin = client.get("/admin/_anon_cookie_probe")
    assert resp_api.status_code == 200
    assert resp_admin.status_code == 200
    assert recorded == []
    assert _anon_cookie_headers(resp_api) == []
    assert _anon_cookie_headers(resp_admin) == []


def test_admin_session_records_and_sets_nothing(client, recorded, monkeypatch):
    monkeypatch.setattr(
        "dashboard_services.admin_auth.is_admin", lambda: True
    )
    resp = client.get("/_anon_cookie_probe")
    assert resp.status_code == 200
    assert recorded == []
    assert _anon_cookie_headers(resp) == []
