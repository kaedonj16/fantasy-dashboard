"""RedZone's 15-second browser poll must not sit behind minute-scale caches."""
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("requests")

import dashboard_services.api as api


def test_ttl_cache_allows_live_callers_to_bound_existing_entry_age(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(api.time, "time", lambda: now[0])
    calls = []

    @api.ttl_cache(ttl=300)
    def load(key):
        calls.append(key)
        return len(calls)

    assert load("game") == 1
    now[0] += 13
    assert load("game") == 1  # ordinary pages retain the five-minute cache
    assert load("game", _cache_max_age=12) == 2
    assert load("game") == 2  # live refresh replaces the shared entry


def test_redzone_collection_requests_short_provider_and_score_cache_age():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    block = source[source.index("def _redzone_collect("):source.index("def _redzone_fetch(")]
    assert "cache_ttl=_RZ_LIVE_CACHE_TTL" in block
    assert "_cache_max_age=_RZ_LIVE_CACHE_TTL" in block
    box = source[source.index("def _redzone_boxscore("):source.index("_RZ_PROJ_CACHE")]
    assert "_cache_max_age=use_ttl" in box


def test_redzone_data_responses_explicitly_disable_http_caching():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    route = source[source.index('def api_redzone_data('):source.index('def api_redzone_player(')]
    assert route.count('response.headers["Cache-Control"]') >= 2
    assert "no-store, no-cache, must-revalidate, max-age=0" in route
