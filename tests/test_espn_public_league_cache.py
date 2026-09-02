"""ESPN public League cache must not freeze empty post-draft rosters."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("espn_api")

from dashboard_services.providers import espn_api


@pytest.fixture(autouse=True)
def clear_caches():
    espn_api.clear_espn_league_caches()
    yield
    espn_api.clear_espn_league_caches()


def test_public_league_cache_expires_by_ttl(monkeypatch):
    calls = []

    def fake_league(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name=f"League-{len(calls)}", roster=[])

    monkeypatch.setattr(espn_api, "League", fake_league)
    monkeypatch.setattr(espn_api, "_PUBLIC_LEAGUE_TTL", 30)

    clock = {"t": 1000.0}
    monkeypatch.setattr(espn_api.time, "time", lambda: clock["t"])

    a = espn_api._public_league_cached(2026, "99")
    b = espn_api._public_league_cached(2026, "99")
    assert a is b
    assert len(calls) == 1

    clock["t"] = 1040.0  # past TTL
    c = espn_api._public_league_cached(2026, "99")
    assert c is not a
    assert len(calls) == 2
    assert c.name == "League-2"


def test_clear_espn_league_caches_forces_reload(monkeypatch):
    calls = []

    def fake_league(**kwargs):
        calls.append(1)
        return SimpleNamespace(name="x")

    monkeypatch.setattr(espn_api, "League", fake_league)
    espn_api._public_league_cached(2026, "1")
    espn_api._public_league_cached(2026, "1")
    assert len(calls) == 1
    espn_api.clear_espn_league_caches()
    espn_api._public_league_cached(2026, "1")
    assert len(calls) == 2


def test_league_cached_cache_clear_alias(monkeypatch):
    monkeypatch.setattr(espn_api, "League", lambda **kw: SimpleNamespace())
    espn_api._public_league_cached(2026, "7")
    assert espn_api._public_league_cache
    espn_api._league_cached.cache_clear()
    assert not espn_api._public_league_cache
    assert not espn_api._auth_league_cache
    assert not espn_api._anon_denied_until
    assert not espn_api._league_globals_cache


def test_get_league_globals_reuses_ttl_cache(monkeypatch):
    fetches = {"n": 0}

    class Request:
        def league_get(self, params):
            fetches["n"] += 1
            return {
                "settings": {
                    "scoringSettings": {"scoringItems": [{"statId": 53, "points": 1}]},
                    "rosterSettings": {"lineupSlotCounts": {"0": 1}},
                }
            }

    league = SimpleNamespace(
        settings=SimpleNamespace(
            scoring_type="ppr",
            position_slot_counts={},
            playoff_team_count=4,
            reg_season_count=14,
        ),
        teams=[object()] * 10,
        espn_request=Request(),
    )
    monkeypatch.setattr(espn_api, "_league", lambda season, league_id: league)

    a = espn_api.get_league_globals(2026, "55")
    b = espn_api.get_league_globals(2026, "55")
    assert a is b
    assert fetches["n"] == 1
    assert a["scoring_settings"]["rec"] == 1.0

    espn_api.clear_espn_league_caches("55", 2026)
    c = espn_api.get_league_globals(2026, "55")
    assert c is not a
    assert fetches["n"] == 2
