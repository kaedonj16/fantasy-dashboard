"""Regression tests for the player league-trades caching/timeout fixes.

The modal's trade-history endpoints used to rescan the league's entire
history chain on every request, and the tab's fetch waited out a wedged
origin (~100s CDN timeout) before showing its retry card. Covered here:

(a) /api/player-league-trades + /api/player-acquisition response cache
(c) get_transactions_by_week per-(platform, league, season, weeks) cache
(b) no Trades prefetch: pmPrefetchTabs stays a no-op, tabs load on demand
(d) _pmFetchTradesInto aborts at 15s so the retry card shows fast
"""
from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")

from dashboard_services import service

ROOT = Path(__file__).resolve().parents[1]
MODAL = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_tx_cache():
    service._TX_BY_WEEK_CACHE.clear()
    yield
    service._TX_BY_WEEK_CACHE.clear()


def _counting_provider(calls, fail_weeks=()):
    def fake(*, platform, league_id, week, season):
        calls.append(week)
        if week in fail_weeks:
            raise RuntimeError("provider down")
        return [{"transaction_id": f"tx-{season}-{week}"}]

    return fake


# ── (c) get_transactions_by_week cache ──────────────────────────────────────

def test_transactions_by_week_second_call_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "platform_get_transactions", _counting_provider(calls))
    first = service.get_transactions_by_week("L1", range(0, 19), platform="sleeper", season=2026)
    assert len(calls) == 19
    second = service.get_transactions_by_week("L1", range(0, 19), platform="sleeper", season=2026)
    assert len(calls) == 19  # served from cache, no new provider calls
    assert second == first


def test_transactions_by_week_cache_keyed_by_season_and_weeks(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "platform_get_transactions", _counting_provider(calls))
    service.get_transactions_by_week("L1", [1, 2], platform="sleeper", season=2026)
    assert len(calls) == 2
    service.get_transactions_by_week("L1", [1, 2], platform="sleeper", season=2025)
    assert len(calls) == 4  # different season refetches
    service.get_transactions_by_week("L1", [1, 2, 3], platform="sleeper", season=2026)
    assert len(calls) == 7  # different week set refetches


def test_transactions_by_week_partial_failure_not_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(
        service, "platform_get_transactions", _counting_provider(calls, fail_weeks={3}))
    out = service.get_transactions_by_week("L1", [1, 2, 3], platform="sleeper", season=2026)
    assert out[3] == []  # failed week still degrades to empty, as before
    assert len(calls) == 3
    service.get_transactions_by_week("L1", [1, 2, 3], platform="sleeper", season=2026)
    assert len(calls) == 6  # partial result was NOT cached: provider retried


def test_transactions_by_week_cached_copy_cannot_be_poisoned(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "platform_get_transactions", _counting_provider(calls))
    first = service.get_transactions_by_week("L1", [1], platform="sleeper", season=2026)
    first[1].append({"transaction_id": "poison"})
    second = service.get_transactions_by_week("L1", [1], platform="sleeper", season=2026)
    assert second[1] == [{"transaction_id": "tx-2026-1"}]
    second[1].append({"transaction_id": "poison-2"})
    third = service.get_transactions_by_week("L1", [1], platform="sleeper", season=2026)
    assert third[1] == [{"transaction_id": "tx-2026-1"}]


# ── (a) route response caches ───────────────────────────────────────────────

@pytest.fixture()
def appmod():
    mod = pytest.importorskip("app")
    mod._PLAYER_TRADES_CACHE.clear()
    yield mod
    mod._PLAYER_TRADES_CACHE.clear()


def test_player_league_trades_route_cached(monkeypatch, appmod):
    import dashboard_services.player_league_trades as plt

    calls = []

    def stub(**kwargs):
        calls.append(kwargs)
        return {"trades": [], "total": 0, "source": "league"}

    monkeypatch.setattr(plt, "get_player_league_trades", stub)
    client = appmod.app.test_client()
    url = "/api/player-league-trades/8112?platform=sleeper&league_id=L1&season=2026&limit=10"
    assert client.get(url).status_code == 200
    assert client.get(url).status_code == 200
    assert len(calls) == 1  # second identical request served from cache
    client.get("/api/player-league-trades/8112?platform=sleeper&league_id=L1&season=2026&limit=50")
    assert len(calls) == 2  # different limit is a different cache entry
    client.get("/api/player-league-trades/9999?platform=sleeper&league_id=L1&season=2026&limit=10")
    assert len(calls) == 3  # different player is a different cache entry


def test_player_acquisition_route_cached(monkeypatch, appmod):
    import dashboard_services.player_league_trades as plt

    calls = []

    def stub(player_id, **kwargs):
        calls.append(player_id)
        return {"events": []}

    monkeypatch.setattr(plt, "get_player_acquisition_events", stub)
    client = appmod.app.test_client()
    url = "/api/player-acquisition/8112?platform=sleeper&league_id=L1&season=2026"
    assert client.get(url).status_code == 200
    assert client.get(url).status_code == 200
    assert len(calls) == 1


# ── (b) + (d) player_modal.js ───────────────────────────────────────────────

def test_trades_fetch_has_15s_abort_timeout():
    body = MODAL.split("function _pmFetchTradesInto", 1)[1].split("// ── Team tab", 1)[0]
    assert "AbortController" in body
    assert "15000" in body
    assert "ctrl.signal" in body


def test_no_trades_prefetch_remains():
    body = MODAL.split("function pmPrefetchTabs()", 1)[1].split("// ── Weekly", 1)[0]
    assert "pmLoadTradesTab" not in body
    assert "_pmFetchTradesInto" not in body
    # The stale comment describing the removed background-activation prefetch
    # (which sent readers hunting for a prefetch that no longer exists) is gone.
    assert "briefly activating" not in MODAL


def test_trades_tab_still_loads_on_demand():
    assert "pmLoadTradesTab(panel, playerId, season, pmTabBar)" in MODAL
