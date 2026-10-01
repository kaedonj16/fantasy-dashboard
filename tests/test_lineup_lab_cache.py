"""Regression tests for the Lineup Lab payload/profile caching.

Pins three behaviors added with the Lab perf fix:

- ``/api/lineup-lab`` serves a repeat identical request from the
  in-memory payload cache without rebuilding, keys entries by
  week/roster (no collisions), and never caches error outcomes.
- ``build_profiles`` reuses the shared ``_PROFILE_CACHE`` per player on
  repeat calls (same key shape as ``get_player_profile``).
- Build-error fallback profiles are never cached, so a transient
  failure is retried instead of sticking for the whole TTL.
"""

import pytest

import data_building.lineup_lab as lab_mod
import data_building.player_distributions as pd_mod

pytest.importorskip("flask")
pytest.importorskip("pandas")


_CTx = {"viewer": {"viewer_roster_id": 7}, "current_week": 4}
_URL = "/api/lineup-lab?platform=sleeper&league_id=L1&season=2026&week=4"


@pytest.fixture
def lab_route(offline_client, monkeypatch):
    """Route wired to a counting fake payload builder."""
    import app as appmod

    appmod._LAB_PAYLOAD_CACHE.clear()
    monkeypatch.setattr(appmod, "_session_signed_in", lambda: True)
    monkeypatch.setattr(
        appmod, "get_league_ctx_from_cache", lambda *a, **k: dict(_CTx))
    calls = {"n": 0}

    def fake_build(**kwargs):
        calls["n"] += 1
        return {"week": kwargs["week"],
                "you": {"roster_id": kwargs["viewer_roster_id"]},
                "n_sims": 2000}

    monkeypatch.setattr(lab_mod, "build_lineup_lab_payload", fake_build)
    yield offline_client, calls
    appmod._LAB_PAYLOAD_CACHE.clear()


def test_lab_route_serves_repeat_request_from_cache(lab_route):
    client, calls = lab_route
    r1 = client.get(_URL)
    r2 = client.get(_URL)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.get_json() == r2.get_json()
    assert r1.get_json()["state"] == "loaded"
    assert calls["n"] == 1


def test_lab_route_cache_key_separates_week_and_roster(lab_route, monkeypatch):
    client, calls = lab_route
    import app as appmod

    assert client.get(_URL).status_code == 200
    assert calls["n"] == 1
    # Different week -> different cache key -> rebuild.
    assert client.get(
        "/api/lineup-lab?platform=sleeper&league_id=L1&season=2026&week=5"
    ).status_code == 200
    assert calls["n"] == 2
    # Different viewer roster -> different cache key -> rebuild.
    monkeypatch.setattr(
        appmod, "get_league_ctx_from_cache",
        lambda *a, **k: {"viewer": {"viewer_roster_id": 9},
                         "current_week": 4})
    r = client.get(_URL)
    assert r.status_code == 200
    assert calls["n"] == 3
    assert r.get_json()["you"]["roster_id"] == 9


def test_lab_route_never_caches_errors(offline_client, monkeypatch):
    import app as appmod

    appmod._LAB_PAYLOAD_CACHE.clear()
    monkeypatch.setattr(appmod, "_session_signed_in", lambda: True)
    monkeypatch.setattr(
        appmod, "get_league_ctx_from_cache", lambda *a, **k: dict(_CTx))
    calls = {"n": 0}

    def flaky_build(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LookupError("viewer roster not found")
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return {"week": kwargs["week"],
                "you": {"roster_id": kwargs["viewer_roster_id"]}}

    monkeypatch.setattr(lab_mod, "build_lineup_lab_payload", flaky_build)
    try:
        assert offline_client.get(_URL).status_code == 409
        assert offline_client.get(_URL).status_code == 503
        # Neither error was cached: the third call reaches the builder
        # again and succeeds, and that success IS cached.
        assert offline_client.get(_URL).status_code == 200
        assert calls["n"] == 3
        assert offline_client.get(_URL).status_code == 200
        assert calls["n"] == 3
    finally:
        appmod._LAB_PAYLOAD_CACHE.clear()


# ── Batch profile cache ──────────────────────────────────────────────────────


@pytest.fixture
def pd_hermetic(monkeypatch):
    rows = [{"rec_yd": s * 10.0, "rec": 1.0, "rec_tgt": 2.0}
            for s in (10, 12, 8, 15, 9)]
    cur = {"p1": rows}
    players = {"p1": {"team": "KC", "position": "WR", "injury_status": ""}}
    monkeypatch.setattr(
        pd_mod, "_week_files", lambda season: cur if season == 2026 else {})
    monkeypatch.setattr(pd_mod, "_players_index", lambda: players)
    pd_mod._PROFILE_CACHE.clear()
    yield
    pd_mod._PROFILE_CACHE.clear()


def _counting_build_one(monkeypatch):
    calls = {"n": 0}
    orig = pd_mod._build_one

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(pd_mod, "_build_one", counting)
    return calls


def test_build_profiles_reuses_cached_profiles(pd_hermetic, monkeypatch):
    calls = _counting_build_one(monkeypatch)
    reqs = [{"player_id": "p1", "pos": "WR", "mean": 12.0}]
    first = pd_mod.build_profiles(reqs, 2026, 4)
    second = pd_mod.build_profiles(reqs, 2026, 4)
    assert calls["n"] == 1
    assert first == second
    # A different week is a different cache key and rebuilds.
    pd_mod.build_profiles(reqs, 2026, 5)
    assert calls["n"] == 2
    # A different mean is a different cache key and rebuilds.
    pd_mod.build_profiles(
        [{"player_id": "p1", "pos": "WR", "mean": 13.0}], 2026, 4)
    assert calls["n"] == 3


def test_build_profiles_does_not_cache_build_errors(pd_hermetic, monkeypatch):
    reqs = [{"player_id": "p1", "pos": "WR", "mean": 12.0}]
    real_build_one = pd_mod._build_one
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("transient")

    monkeypatch.setattr(pd_mod, "_build_one", boom)
    out1 = pd_mod.build_profiles(reqs, 2026, 4)
    assert out1["p1"]["factors"].get("fallback") == "build_error"
    out2 = pd_mod.build_profiles(reqs, 2026, 4)
    assert out2["p1"]["factors"].get("fallback") == "build_error"
    # The fallback was not cached: the builder was retried.
    assert calls["n"] == 2

    # Once the builder recovers, the real profile is built once and
    # then served from the cache.
    real_calls = {"n": 0}

    def counting(*a, **k):
        real_calls["n"] += 1
        return real_build_one(*a, **k)

    monkeypatch.setattr(pd_mod, "_build_one", counting)
    out3 = pd_mod.build_profiles(reqs, 2026, 4)
    assert "fallback" not in out3["p1"]["factors"]
    pd_mod.build_profiles(reqs, 2026, 4)
    assert real_calls["n"] == 1
