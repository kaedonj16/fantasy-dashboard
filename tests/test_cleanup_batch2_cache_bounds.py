"""Cleanup Batch 2a: the four previously unbounded caches stay bounded.

_RZ_BOX_CACHE (app.py), archetype _SIM_CACHE, news _CACHE, and the ESPN
draft relay _STORE each keyed on an effectively unbounded id space while
only TTL-checking on read of the exact key, so memory grew one entry per
new game / league / athlete / draft forever. Each now prunes on write;
the relay additionally honours its advertised TTL on the local get path.
"""
import time

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")


# ── app.py _RZ_BOX_CACHE ─────────────────────────────────────────────────

def test_rz_box_cache_stays_bounded(monkeypatch):
    import app
    import dashboard_services.api as api

    monkeypatch.setattr(
        api,
        "fetch_tank_boxscore",
        lambda game_id, **kwargs: {"game_id": game_id},
    )
    app._RZ_BOX_CACHE.clear()
    try:
        total = app._RZ_BOX_CACHE_MAX + 40
        for i in range(total):
            app._scorezone_boxscore(f"game-{i}")
        assert len(app._RZ_BOX_CACHE) <= app._RZ_BOX_CACHE_MAX
        # The freshest game survives; the oldest slate was evicted.
        assert f"game-{total - 1}" in app._RZ_BOX_CACHE
        assert "game-0" not in app._RZ_BOX_CACHE
    finally:
        app._RZ_BOX_CACHE.clear()


# ── archetype_engine _SIM_CACHE ──────────────────────────────────────────

def test_sim_cache_prune_evicts_oldest():
    from dashboard_services import archetype_engine as ae

    saved = dict(ae._SIM_CACHE)
    ae._SIM_CACHE.clear()
    try:
        total = ae._SIM_CACHE_MAX + 25
        for i in range(total):
            ae._SIM_CACHE[f"espn:league-{i}:2026"] = {
                "sim_state": None,
                "base_odds": {},
                "ts": float(i),
                "sched_sig": "fallback",
            }
        ae._prune_sim_cache()
        assert len(ae._SIM_CACHE) == ae._SIM_CACHE_MAX
        assert f"espn:league-{total - 1}:2026" in ae._SIM_CACHE
        assert "espn:league-0:2026" not in ae._SIM_CACHE
    finally:
        ae._SIM_CACHE.clear()
        ae._SIM_CACHE.update(saved)


# ── news _CACHE ──────────────────────────────────────────────────────────

def test_news_cache_stays_bounded():
    from dashboard_services import news

    saved = dict(news._CACHE)
    news._CACHE.clear()
    try:
        total = news._CACHE_MAX + 200
        for i in range(total):
            news._cache_put(f"athlete_{i}", float(i), [])
        assert len(news._CACHE) <= news._CACHE_MAX
        assert f"athlete_{total - 1}" in news._CACHE
        assert "athlete_0" not in news._CACHE
    finally:
        news._CACHE.clear()
        news._CACHE.update(saved)


# ── espn_draft_relay _STORE ──────────────────────────────────────────────

def test_relay_store_stays_bounded():
    from dashboard_services import espn_draft_relay as relay

    with relay._STORE_LOCK:
        saved = dict(relay._STORE)
        relay._STORE.clear()
    try:
        total = relay._STORE_MAX + 10
        for i in range(total):
            relay.put_relay_snapshot(f"league-{i}", 2026, {"picks": []})
        with relay._STORE_LOCK:
            size = len(relay._STORE)
        assert size <= relay._STORE_MAX
        assert relay.get_relay_snapshot(f"league-{total - 1}", 2026) is not None
    finally:
        with relay._STORE_LOCK:
            relay._STORE.clear()
            relay._STORE.update(saved)


def test_relay_local_get_honours_ttl(monkeypatch):
    from dashboard_services import espn_draft_relay as relay

    monkeypatch.delenv("REDIS_URL", raising=False)
    relay.put_relay_snapshot("ttl-league", 2026, {"picks": [{"pick_no": 1}]})
    try:
        # Fresh entry is served locally.
        assert relay.get_relay_snapshot("ttl-league", 2026) is not None
        # Backdate past the advertised TTL: the local get path used to
        # ignore it and serve the stale snapshot forever.
        key = relay._store_key("ttl-league", 2026)
        with relay._STORE_LOCK:
            relay._STORE[key]["updated_at"] = (
                int(time.time()) - relay.DEFAULT_TTL_SECONDS - 60
            )
        assert relay.get_relay_snapshot("ttl-league", 2026) is None
        with relay._STORE_LOCK:
            assert key not in relay._STORE
    finally:
        relay.clear_relay_snapshot("ttl-league", 2026)
