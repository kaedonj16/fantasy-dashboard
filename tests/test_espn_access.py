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


def test_private_espn_league_preserves_access_denied_without_cookies(monkeypatch):
    monkeypatch.setattr(espn_api, "League", lambda **kwargs: (_ for _ in ()).throw(ESPNAccessDenied()))
    monkeypatch.setattr(espn_api, "_espn_creds", lambda: (None, None))

    with pytest.raises(ESPNAccessDenied):
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


@pytest.mark.parametrize("swid,espn_s2", [("x", None), (None, "x")])
def test_private_connection_rejects_partial_credentials(swid, espn_s2):
    with pytest.raises(espn_api.ESPNRequestValidationError):
        espn_api.ESPNFantasyClient(swid=swid, espn_s2=espn_s2)
