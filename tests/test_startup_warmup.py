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
    monkeypatch.setattr(sw, "_warm_recent_leagues", _rec("recent_leagues"))
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
        "recent_leagues",
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
    assert len(names) == 5


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
    assert len(stubbed_loaders) == 6


# ── Recent-league warmup ──────────────────────────────────────────────────

def test_recent_league_visits_returns_empty_on_db_error(monkeypatch):
    import dashboard_services.db as db

    def _boom():
        raise RuntimeError("db is down")

    monkeypatch.setattr(db, "get_conn", _boom)
    assert sw._recent_league_visits() == []


def test_recent_league_visits_parses_rows(monkeypatch):
    import dashboard_services.db as db

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, _q, _params):
            class _R:
                def fetchall(self):
                    return [
                        ("sleeper", "12345", 2026),
                        ("espn", "67890", 2026),
                        ("sleeper", None, 2026),  # malformed: skipped
                    ]
            return _R()

    monkeypatch.setattr(db, "get_conn", lambda: _Conn())
    assert sw._recent_league_visits() == [
        ("sleeper", "12345", 2026),
        ("espn", "67890", 2026),
    ]


def test_warm_recent_leagues_dispatches_each_visit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sw, "_recent_league_visits",
        lambda: [("sleeper", "1", 2026), ("yahoo", "2", 2026)],
    )

    import sys
    import types

    fake_app = types.ModuleType("app")
    fake_app._warm_league_ctx_async = lambda p, l, s: calls.append((p, l, s))
    monkeypatch.setitem(sys.modules, "app", fake_app)

    sw._warm_recent_leagues()  # must not raise
    assert calls == [("sleeper", "1", 2026), ("yahoo", "2", 2026)]


def test_warm_recent_leagues_never_raises(monkeypatch):
    monkeypatch.setattr(
        sw, "_recent_league_visits",
        lambda: [("sleeper", "1", 2026)],
    )

    import sys
    import types

    fake_app = types.ModuleType("app")

    def _boom(*a):
        raise RuntimeError("warm failed")

    fake_app._warm_league_ctx_async = _boom
    monkeypatch.setitem(sys.modules, "app", fake_app)

    sw._warm_recent_leagues()  # must not raise
