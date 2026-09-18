import threading
import time
import types


def test_stable_lock_key_is_deterministic_and_league_scoped():
    from dashboard_services.league_singleflight import stable_lock_key
    assert stable_lock_key("sleeper", 2026, "a") == stable_lock_key("SLEEPER", 2026, "a")
    assert stable_lock_key("sleeper", 2026, "a") != stable_lock_key("sleeper", 2026, "b")


def test_file_singleflight_is_bounded_and_released(monkeypatch):
    from dashboard_services.league_singleflight import LeagueBuildBusy, league_build_lock
    monkeypatch.delenv("DATABASE_URL", raising=False)
    entered = threading.Event()
    release = threading.Event()

    def owner():
        with league_build_lock("sleeper", 2026, "lock-test", timeout=.5):
            entered.set()
            release.wait(1)

    thread = threading.Thread(target=owner)
    thread.start()
    assert entered.wait(1)
    started = time.monotonic()
    try:
        with league_build_lock("sleeper", 2026, "lock-test", timeout=.1):
            raise AssertionError("contending build acquired the lock")
    except LeagueBuildBusy:
        pass
    assert time.monotonic() - started < .5
    release.set()
    thread.join(1)
    with league_build_lock("sleeper", 2026, "lock-test", timeout=.2):
        pass


def test_postgres_contention_never_falls_back_to_a_second_lock_domain(monkeypatch):
    from dashboard_services import league_singleflight as singleflight

    class Connection:
        def execute(self, _query, _params):
            return self

        def fetchone(self):
            return (False,)

        def close(self):
            pass

    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/app")
    monkeypatch.setitem(
        __import__("sys").modules,
        "dashboard_services.db",
        types.SimpleNamespace(get_conn=lambda: Connection()),
    )
    monkeypatch.setattr(singleflight.os, "open", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("file-lock fallback must not run after Postgres contention")
    ))
    try:
        with singleflight.league_build_lock("sleeper", 2026, "pg-busy", timeout=.01):
            raise AssertionError("contending advisory lock was acquired")
    except singleflight.LeagueBuildBusy:
        pass


def test_dashboard_cache_limit_and_lock_registry_are_conservative():
    import app
    assert app.DASHBOARD_CACHE_MAX >= 1
    assert app.DASHBOARD_CACHE_MAX == 24
    assert app._CTX_LOCKS_MAX >= app.DASHBOARD_CACHE_MAX


def test_context_lock_registry_prunes_after_many_releases(monkeypatch):
    import app
    monkeypatch.setattr(app, "_CTX_LOCKS_MAX", 3)
    app._CTX_LOCKS.clear()
    for index in range(20):
        key = ("sleeper", 2026, str(index))
        state, _ = app._acquire_context_lock(key)
        app._release_context_lock(key, state)
    assert len(app._CTX_LOCKS) <= 3


def test_portfolio_refresh_contract_is_targeted():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    js = (root / "static" / "app.js").read_text()
    py = (root / "app.py").read_text()
    routes = (root / "routes" / "user_pages_bp.py").read_text()
    assert "window.brRefreshCurrentPage" in js
    assert "/api/portfolio/refresh" in py
    assert "style.display!=='none'" in py
    assert '@user_pages_bp.route("/api/portfolio/refresh", methods=["POST"])' in routes
    assert 'max_workers=min(2, len(requested))' in routes
    assert '"account_id" in payload' in routes
