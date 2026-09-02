import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("espn_api")

from dashboard_services.providers import espn_api


class ESPNAccessDenied(Exception):
    pass


class ESPNInvalidLeague(Exception):
    pass


@pytest.fixture(autouse=True)
def clear_league_cache():
    espn_api._league_cached.cache_clear()
    yield
    espn_api._league_cached.cache_clear()


def test_public_espn_league_is_loaded_without_server_cookies(monkeypatch):
    calls = []

    def fake_league(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name="Public league")

    monkeypatch.setattr(espn_api, "League", fake_league)
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: ("secret", "{owner}"))

    league = espn_api._league_cached(2026, "123")

    assert league.name == "Public league"
    assert calls == [{"league_id": 123, "year": 2026}]


def test_private_espn_league_retries_with_server_cookies(monkeypatch):
    calls = []

    def fake_league(**kwargs):
        calls.append(kwargs)
        if "espn_s2" not in kwargs:
            raise ESPNAccessDenied()
        return SimpleNamespace(name="Private league")

    monkeypatch.setattr(espn_api, "League", fake_league)
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: ("secret", "{owner}"))

    league = espn_api._league_cached(2026, "456")

    assert league.name == "Private league"
    assert calls == [
        {"league_id": 456, "year": 2026},
        {"league_id": 456, "year": 2026, "espn_s2": "secret", "swid": "{owner}"},
    ]


def test_private_league_retries_after_espn_api_none_cookies_bug(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=espn_api.__name__)
    calls = []

    def fake_league(**kwargs):
        calls.append(kwargs)
        if "espn_s2" not in kwargs:
            raise AttributeError("'NoneType' object has no attribute 'get'")
        return SimpleNamespace(name="Private league")

    monkeypatch.setattr(espn_api, "League", fake_league)
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: ("secret", "{owner}"))

    league = espn_api._league_cached(2026, "456")

    assert league.name == "Private league"
    assert calls == [
        {"league_id": 456, "year": 2026},
        {"league_id": 456, "year": 2026, "espn_s2": "secret", "swid": "{owner}"},
    ]
    assert "anonymous None-cookies AttributeError as access denied" in caplog.text


def test_private_league_skips_anonymous_retry_and_reuses_auth_cache(monkeypatch, caplog):
    """Player-modal / dashboard traffic must not re-pay a failed anonymous ESPN hit."""
    caplog.set_level(logging.INFO, logger=espn_api.__name__)
    calls = []

    def fake_league(**kwargs):
        calls.append(kwargs)
        if "espn_s2" not in kwargs:
            raise AttributeError("'NoneType' object has no attribute 'get'")
        return SimpleNamespace(name=f"Private-{len(calls)}")

    monkeypatch.setattr(espn_api, "League", fake_league)
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: ("secret", "{owner}"))

    first = espn_api._league_cached(2026, "887776065")
    second = espn_api._league_cached(2026, "887776065")
    third = espn_api._league_cached(2026, "887776065")

    assert first is second is third
    # One doomed anonymous attempt + one authenticated load; later calls reuse cache.
    assert calls == [
        {"league_id": 887776065, "year": 2026},
        {"league_id": 887776065, "year": 2026, "espn_s2": "secret", "swid": "{owner}"},
    ]
    # AttributeError compatibility log once per denial window, not per request.
    assert caplog.text.count("anonymous None-cookies AttributeError as access denied") == 1


def test_auth_league_cache_is_scoped_to_credential_fingerprint(monkeypatch):
    calls = []

    def fake_league(**kwargs):
        calls.append(kwargs)
        if "espn_s2" not in kwargs:
            raise ESPNAccessDenied()
        return SimpleNamespace(name=kwargs["espn_s2"])

    monkeypatch.setattr(espn_api, "League", fake_league)
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: ("secret-a", "{owner-a}"))
    a = espn_api._league_cached(2026, "456")
    assert a.name == "secret-a"

    monkeypatch.setattr(espn_api, "_espn_creds", lambda: ("secret-b", "{owner-b}"))
    b = espn_api._league_cached(2026, "456")
    assert b.name == "secret-b"
    assert a is not b
    # Anonymous denial is remembered, so only authenticated loads after the first.
    assert calls == [
        {"league_id": 456, "year": 2026},
        {"league_id": 456, "year": 2026, "espn_s2": "secret-a", "swid": "{owner-a}"},
        {"league_id": 456, "year": 2026, "espn_s2": "secret-b", "swid": "{owner-b}"},
    ]


def test_private_guest_dashboard_uses_staged_credentials_after_anonymous_bug(monkeypatch):
    flask = pytest.importorskip("flask")
    app = flask.Flask(__name__)
    app.secret_key = "test"
    calls = []

    def fake_league(**kwargs):
        calls.append(kwargs)
        if "espn_s2" not in kwargs:
            raise AttributeError("'NoneType' object has no attribute 'get'")
        return SimpleNamespace(name="Private league")

    monkeypatch.setattr(espn_api, "League", fake_league)
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: (None, None))
    import dashboard_services.accounts as accounts
    monkeypatch.setattr(accounts, "peek_private_espn_connection", lambda *a: {
        "espn_s2": "staged-secret", "swid": "{staged-owner}",
    })

    with app.test_request_context("/espn/2026/456/dashboard"):
        flask.session["pending_provider_connection_token"] = "opaque-token"
        league = espn_api._league_cached(2026, "456")

    assert league.name == "Private league"
    assert calls[-1] == {
        "league_id": 456, "year": 2026,
        "espn_s2": "staged-secret", "swid": "{staged-owner}",
    }


def test_unrelated_attribute_error_is_not_treated_as_access_denied(monkeypatch):
    error = AttributeError("settings")
    monkeypatch.setattr(espn_api, "League", lambda **kwargs: (_ for _ in ()).throw(error))
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: ("secret", "{owner}"))

    with pytest.raises(AttributeError) as caught:
        espn_api._league_cached(2026, "456")

    assert caught.value is error


def test_authenticated_access_denial_does_not_rethrow_cookie_bearing_message(monkeypatch):
    calls = []

    def fake_league(**kwargs):
        calls.append(kwargs)
        if "espn_s2" not in kwargs:
            raise ESPNAccessDenied("anonymous")
        raise ESPNAccessDenied("cannot access with espn_s2=super-secret and swid={owner}")

    monkeypatch.setattr(espn_api, "League", fake_league)
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: ("super-secret", "{owner}"))

    with pytest.raises(espn_api.ESPNAccessDenied) as caught:
        espn_api._league_cached(2026, "456")

    assert "super-secret" not in str(caught.value)
    assert "{owner}" not in str(caught.value)
    assert len(calls) == 2


def test_private_espn_league_preserves_access_denied_without_cookies(monkeypatch):
    monkeypatch.setattr(espn_api, "League", lambda **kwargs: (_ for _ in ()).throw(ESPNAccessDenied()))
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: (None, None))

    # The provider deliberately replaces espn-api's exception because recent
    # releases include cookie values in its message.
    with pytest.raises(espn_api.ESPNAccessDenied, match="anonymous access"):
        espn_api._league_cached(2026, "789")


def test_invalid_league_is_not_retried_with_credentials(monkeypatch):
    calls = []

    def fake_league(**kwargs):
        calls.append(kwargs)
        raise ESPNInvalidLeague()

    monkeypatch.setattr(espn_api, "League", fake_league)
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: ("secret", "{owner}"))

    with pytest.raises(ESPNInvalidLeague):
        espn_api._league_cached(2026, "999")

    assert calls == [{"league_id": 999, "year": 2026}]


def test_explicit_public_connection_never_uses_credentials(monkeypatch):
    calls = []
    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, ok=True, json=lambda: {
            "id": 123, "seasonId": 2026, "settings": {"name": "Public"},
        })
    monkeypatch.setattr(espn_api.requests, "get", fake_get)
    result = espn_api.connect_league(2026, "123")
    assert result["name"] == "Public"
    assert calls[0][1]["cookies"] is None
    assert calls[0][1]["timeout"] == espn_api.ESPN_REQUEST_TIMEOUT


def test_explicit_private_connection_uses_submitted_credentials(monkeypatch):
    calls = []
    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, ok=True, json=lambda: {
            "id": 456, "seasonId": 2026, "settings": {"name": "Private"},
        })
    monkeypatch.setattr(espn_api.requests, "get", fake_get)
    result = espn_api.connect_league(2026, "456", swid="owner", espn_s2="secret")
    assert result["name"] == "Private"
    assert calls[0][1]["cookies"] == {"SWID": "{owner}", "espn_s2": "secret"}


@pytest.mark.parametrize("status,error", [
    (403, espn_api.ESPNAccessDenied),
    (404, espn_api.ESPNInvalidLeague),
    (429, espn_api.ESPNRateLimited),
    (503, espn_api.ESPNUnavailable),
])
def test_connection_maps_espn_http_errors(monkeypatch, status, error):
    monkeypatch.setattr(espn_api.requests, "get", lambda *a, **k: SimpleNamespace(status_code=status, ok=False))
    with pytest.raises(error):
        espn_api.connect_league(2026, "123")


def test_connection_rejects_empty_espn_response(monkeypatch):
    monkeypatch.setattr(espn_api.requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, ok=True, json=lambda: None,
    ))
    with pytest.raises(espn_api.ESPNMalformedResponse):
        espn_api.connect_league(2026, "123")


def test_connection_logs_safe_upstream_response_metadata(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=espn_api.__name__)
    response = SimpleNamespace(
        status_code=200,
        ok=True,
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=b"<html>login</html>",
        json=lambda: (_ for _ in ()).throw(ValueError("not json")),
    )
    monkeypatch.setattr(espn_api.requests, "get", lambda *a, **k: response)

    with pytest.raises(espn_api.ESPNMalformedResponse) as caught:
        espn_api.connect_league(2026, "123", swid="owner", espn_s2="super-secret")

    assert caught.value.debug_reference
    assert "outcome=json_decode_failed" in caplog.text
    assert "content_type='text/html; charset=utf-8'" in caplog.text
    assert "content_length=18" in caplog.text
    assert "authenticated=True" in caplog.text
    assert "super-secret" not in caplog.text


@pytest.mark.parametrize("swid,espn_s2", [("x", None), (None, "x")])
def test_private_connection_rejects_partial_credentials(swid, espn_s2):
    with pytest.raises(espn_api.ESPNRequestValidationError):
        espn_api.ESPNFantasyClient(swid=swid, espn_s2=espn_s2)
