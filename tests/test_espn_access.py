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
    monkeypatch.setattr(espn_api, "League", lambda **kwargs: calls.append(kwargs) or SimpleNamespace(name="Public"))
    result = espn_api.connect_league(2026, "123")
    assert result["name"] == "Public"
    assert calls == [{"league_id": 123, "year": 2026}]


def test_explicit_private_connection_uses_submitted_credentials(monkeypatch):
    calls = []
    monkeypatch.setattr(espn_api, "League", lambda **kwargs: calls.append(kwargs) or SimpleNamespace(name="Private"))
    result = espn_api.connect_league(2026, "456", swid="owner", espn_s2="secret")
    assert result["name"] == "Private"
    assert calls == [{"league_id": 456, "year": 2026, "swid": "{owner}", "espn_s2": "secret"}]


@pytest.mark.parametrize("swid,espn_s2", [("x", None), (None, "x")])
def test_private_connection_rejects_partial_credentials(swid, espn_s2):
    with pytest.raises(espn_api.ESPNRequestValidationError):
        espn_api.ESPNFantasyClient(swid=swid, espn_s2=espn_s2)
