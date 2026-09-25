"""Tests for deploy-time shared-cache warmup (dashboard_services.startup_warmup).

The warmup runs in a daemon thread per gunicorn worker after the port bind.
Contract: every league-independent step runs exactly once, one failing step
never blocks the others, the whole pass never raises, and
STARTUP_WARMUP_ENABLED=0 disables it.
"""
import pytest

import dashboard_services.startup_warmup as sw


@pytest.fixture()
def stubbed_loaders(monkeypatch):
    calls = []

    def _rec(name):
        def _fn(*a, **k):
            calls.append((name, a, k))
        return _fn

    import dashboard_services.api as api
    import utils.utils as uu
    import data_building.fetch_projections as fp
    import data_building.weekly_metrics as wm

    monkeypatch.setattr(api, "get_nfl_state",
                        lambda: {"season": 2026, "week": 4})
    monkeypatch.setattr(api, "get_nfl_players", _rec("nfl_players"))
    monkeypatch.setattr(uu, "load_players_index", _rec("players_index"))
    monkeypatch.setattr(fp, "fetch_sleeper_season_ppg_variants",
                        _rec("projections"))
    monkeypatch.setattr(wm, "get_usage_trends", _rec("usage_trends"))
    monkeypatch.setattr(sw, "_load_model_value_table", _rec("value_table"))
    monkeypatch.setattr(sw, "_ENABLED", True)
    return calls


def _names(calls):
    return [c[0] for c in calls]


def test_warmup_runs_every_step_once(stubbed_loaders):
    sw.warm_shared_caches()
    assert _names(stubbed_loaders) == [
        "players_index",
        "nfl_players",
        "projections",
        "usage_trends",
        "value_table",
    ]
    # season-scoped steps receive the season from nfl state
    for name, args, _kw in stubbed_loaders:
        if name in ("projections", "usage_trends"):
            assert args == (2026,), (name, args)


def test_one_failing_step_does_not_block_others(stubbed_loaders, monkeypatch):
    import utils.utils as uu

    def _boom():
        raise RuntimeError("sleeper is down")

    monkeypatch.setattr(uu, "load_players_index", _boom)
    sw.warm_shared_caches()  # must not raise
    names = _names(stubbed_loaders)
    assert "players_index" not in names
    assert len(names) == 4


def test_disabled_warmup_is_noop(stubbed_loaders, monkeypatch):
    monkeypatch.setattr(sw, "_ENABLED", False)
    sw.warm_shared_caches()
    assert stubbed_loaders == []
    assert sw.warm_shared_caches_async() is None


def test_warmup_async_spawns_daemon_thread(stubbed_loaders):
    thread = sw.warm_shared_caches_async()
    assert thread is not None
    assert thread.daemon
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert len(stubbed_loaders) == 5
