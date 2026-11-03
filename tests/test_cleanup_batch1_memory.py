"""Cleanup Batch 1: cache env split, players generation release, index TTL,
and get_nfl_players segregation from the shared provider TTL budget."""
import gc
import json
import os
import subprocess
import sys
import weakref

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")


def _import_constants(env):
    code = (
        "import json\n"
        "import app\n"
        "import dashboard_services.api as api\n"
        "print(json.dumps({\n"
        "  'ctx': app.LEAGUE_CTX_CACHE_MAX,\n"
        "  'ctx_alias': app.DASHBOARD_CACHE_MAX,\n"
        "  'provider': api.PROVIDER_TTL_CACHE_MAX_ENTRIES,\n"
        "}))\n"
    )
    full_env = {k: v for k, v in os.environ.items()
                if k not in ("DASHBOARD_CACHE_MAX", "LEAGUE_CTX_CACHE_MAX",
                             "PROVIDER_TTL_CACHE_MAX_ENTRIES")}
    full_env.update(env)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=full_env, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_env_split_defaults_preserve_behavior():
    vals = _import_constants({})
    assert vals == {"ctx": 4, "ctx_alias": 4, "provider": 75}


def test_env_split_old_name_applies_to_ctx_only():
    vals = _import_constants({"DASHBOARD_CACHE_MAX": "9"})
    assert vals["ctx"] == 9
    assert vals["ctx_alias"] == 9
    # The provider budget no longer follows the old shared name.
    assert vals["provider"] == 75


def test_env_split_new_names_win_independently():
    vals = _import_constants({
        "DASHBOARD_CACHE_MAX": "9",
        "LEAGUE_CTX_CACHE_MAX": "7",
        "PROVIDER_TTL_CACHE_MAX_ENTRIES": "33",
    })
    assert vals["ctx"] == 7
    assert vals["provider"] == 33


def test_serve_swap_releases_old_players_generation(monkeypatch):
    import app

    class Gen(dict):
        """dict subclass: plain dicts do not support weak references."""

    gen_old = Gen({"p1": {"full_name": "Old Gen"}})
    gen_new = Gen({"p1": {"full_name": "New Gen"}})
    new_map = {"p1": {"name": "New Gen", "team": "BUF", "pos": "QB"}}
    ctx = {"players": gen_old, "players_map": {"p1": {"name": "Old Gen"}}}
    old_ref = weakref.ref(gen_old)

    monkeypatch.setattr(app, "get_players_global", lambda: gen_new)
    monkeypatch.setattr(
        app, "get_players_map",
        lambda players: new_map if players is gen_new else {},
    )
    app._league_ctx_refresh_players(ctx)
    assert ctx["players"] is gen_new
    assert ctx["players_map"] is new_map

    del gen_old
    gc.collect()
    assert old_ref() is None, "old players generation still pinned after serve swap"


def test_serve_swap_is_noop_when_already_current(monkeypatch):
    import app

    gen = {"p1": {"full_name": "Current"}}
    pmap = {"p1": {"name": "Current"}}
    ctx = {"players": gen, "players_map": pmap}
    monkeypatch.setattr(app, "get_players_global", lambda: gen)
    monkeypatch.setattr(app, "get_players_map", lambda players: pmap)
    app._league_ctx_refresh_players(ctx)
    assert ctx["players"] is gen
    assert ctx["players_map"] is pmap


def test_players_index_global_refreshes_after_ttl(monkeypatch):
    import app

    v1 = {"p1": {"team": "BUF"}}
    v2 = {"p1": {"team": "MIA"}}
    calls = []

    def fake_load():
        calls.append(1)
        return v2

    monkeypatch.setattr(app, "load_players_index", fake_load)
    monkeypatch.setattr(app, "_PLAYERS_INDEX_GLOBAL", v1)
    # Fresh within TTL: served from the global, loader untouched.
    monkeypatch.setattr(app, "_PLAYERS_INDEX_GLOBAL_TS", app.time.time())
    assert app.get_players_index_global() is v1
    assert calls == []
    # Past TTL: reloads through load_players_index (previously pinned forever).
    monkeypatch.setattr(
        app, "_PLAYERS_INDEX_GLOBAL_TS",
        app.time.time() - app._PLAYERS_INDEX_GLOBAL_TTL - 1,
    )
    assert app.get_players_index_global() is v2
    assert calls == [1]


def test_nfl_players_segregated_from_shared_budget(monkeypatch):
    import dashboard_services.api as api

    assert "get_nfl_players" not in [i["name"] for i in api._TTL_CACHES]
    assert "get_nfl_players" in [i["name"] for i in api._TTL_CACHES_SEGREGATED]

    # Seed the segregated slot directly (no network), then churn the shared
    # budget hard: the players entry must survive untouched.
    key = ("get_nfl_players", (), ())
    with api.get_nfl_players._cache_lock:
        api.get_nfl_players._cache[key] = (api.time.time(), {"p1": {}})
    monkeypatch.setattr(api, "PROVIDER_TTL_CACHE_MAX_ENTRIES", 2)

    @api.ttl_cache(ttl=60)
    def small(k):
        return k

    for i in range(6):
        small(i)
    with api.get_nfl_players._cache_lock:
        assert key in api.get_nfl_players._cache
    assert api.ttl_cache_entry_counts().get("get_nfl_players") == 1
    api.get_nfl_players.clear_cache()


def test_own_max_entries_cache_has_its_own_cap(monkeypatch):
    import dashboard_services.api as api

    @api.ttl_cache(ttl=60, own_max_entries=1)
    def solo(k):
        return k

    solo("a")
    solo("b")
    assert len(solo._cache) == 1
    assert "solo" not in [i["name"] for i in api._TTL_CACHES]
