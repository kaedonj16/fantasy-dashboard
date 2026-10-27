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
    assert app.DASHBOARD_CACHE_MAX == 4
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
    # The app.js card owner is the SINGLE refresh handler (2026-10-27):
    # the inline app.py handler that raced it is deleted.
    assert "window.brRefreshCurrentPage = refresh" in js
    assert "brRefreshCurrentPage" not in py
    assert "/api/portfolio/refresh" in js
    assert '@user_pages_bp.route("/api/portfolio/refresh", methods=["POST"])' in routes
    assert '"account_id" in payload' in routes
    # Kick-and-return: the endpoint busts + kicks the background summary
    # refresh and answers accepted/pending immediately; it must not run
    # the old synchronous per-league build pool on the request thread.
    endpoint = routes.split("def api_portfolio_refresh")[1].split("\ndef ")[0]
    assert '"accepted": True' in endpoint
    assert '"pending": True' in endpoint
    assert "_kick_summary_refresh(account_id, membership)" in endpoint
    assert "ThreadPoolExecutor" not in endpoint
    assert "build_league_summary(" not in endpoint


def test_portfolio_refresh_covers_every_visible_card():
    """The old inline handler collected keys from data-summary-card
    attributes, which warm cards render valueless, so an all-warm
    portfolio produced no keys and Refresh silently no-op'd (or posted
    all 11 leagues unbatched and got a 400). The app.js owner handler
    instead derives the batch from its visible card set and posts in
    batches of at most 4, the server's per-POST cap."""
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    owner = js.split("/* Portfolio cards: one generation owns requests")[1]
    refresh = owner.split("function refresh(ownerArg)")[1].split("function bind(owner)")[0]
    assert "owner.cards(true)" in refresh
    assert "keys.slice(at, at + 4)" in refresh
    # Completion is the per-card tracker phase polling /api/portfolio/card,
    # not the batch response: a timed-out card request must still resolve
    # its tracker exactly once (the old spin-forever hole).
    assert "fireTrackers" in owner
    assert "timedOut: true" in owner


def test_portfolio_refresh_tracker_fires_exactly_once():
    """Structural contract for the tracker fix in the app.js card owner:

    * schedule() attaches the tracker BEFORE the queued/inflight dedupe
      early-return, so a tracker for an already-running card is not
      silently discarded (it fires when that card settles).
    * fireTrackers deletes the per-card list BEFORE invoking, so no
      settle can fire a tracker twice.
    * requestCard's timeout catch fires the trackers when the retry
      budget is exhausted; with budget left it reschedules WITHOUT
      re-passing the tracker (it lives on the owner, not the queue item).
    """
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    owner = js.split("/* Portfolio cards: one generation owns requests")[1]

    schedule = owner.split("function schedule(owner, card, delay, reset, tracker)")[1]
    schedule = schedule.split("function applyCard")[0]
    assert schedule.index("attachTracker(owner, key, tracker)") < schedule.index(
        "owner.queued.has(key)")

    fire = owner.split("function fireTrackers(owner, key, result)")[1]
    fire = fire.split("function schedule")[0]
    assert fire.index("delete owner.trackers[key]") < fire.index("list.forEach")

    request = owner.split("function requestCard(owner, item)")[1]
    request = request.split("function pump(owner)")[0]
    timeout_branch = request.split("error.name === 'TimeoutError'")[1]
    timeout_branch = timeout_branch.split("var payload = error.payload")[0]
    assert "fireTrackers(owner, key" in timeout_branch
    assert "timedOut: true" in timeout_branch
    # Intermediate pending settles must not fire: the success path
    # re-schedules a pending card without firing, and fires trackers
    # only on the non-pending (terminal) branches.
    assert "if (result.pending) {" in request
    pending_branch = request.split("if (result.pending) {")[1].split("} else if")[0]
    assert "fireTrackers" not in pending_branch
    assert "else fireTrackers(owner, key, result);" in request


def test_league_builds_are_concurrency_capped_process_wide():
    """A portfolio refresh rebuilds every league; combined with per-card summary
    hydration and background warms, unbounded concurrent context builds OOM a
    memory-limited worker. A process-wide semaphore must bound how many
    build_league_context calls run at once, whatever endpoint triggered them."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert "_LEAGUE_BUILD_SEM" in src
    assert "LEAGUE_BUILD_CONCURRENCY" in src
    # The cap actually wraps the expensive build, not just declared. Since
    # 2026-10-27 the acquire is manual and BOUNDED (10s): a thread that
    # cannot get a build slot degrades to stale/pending instead of
    # queueing indefinitely behind builds on a 4-thread worker.
    helper = src.split("def get_league_ctx_from_cache(")[1].split("\ndef ")[0]
    assert "_LEAGUE_BUILD_SEM.acquire(timeout=10)" in helper
    guarded = helper.split("_LEAGUE_BUILD_SEM.acquire(timeout=10)")[1][:1200]
    assert "build_league_context(" in guarded
    assert "_LEAGUE_BUILD_SEM.release()" in guarded
