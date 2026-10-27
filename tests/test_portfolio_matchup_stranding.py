"""Portfolio matchup previews stuck on their loading skeleton.

Kaedon (2026-10-18, Sunday ~19:20 ET): the score/result band on My Leagues
cards "often struggles to load or doesn't load at all" while the rest of
each card (record / standing / streak / position strength, all
server-rendered) is already populated.

Verified stranding paths behind that symptom:

Client (static/app.js portfolio card owner):
  * The success path only retried on pending or a failed SUMMARY. A card
    whose summary loaded but whose matchup came back incomplete reset its
    retry budget to 0 and scheduled nothing -- only the 45s poller could
    ever revisit it.
  * renderMatchup's incomplete shape ({live: false} with no applicable /
    error state) left the skeleton in place and set no state: no error, no
    Retry button, ever.
  * A never-loaded card stuck in the pending shape kept its skeleton
    forever; the ~90s escape existed only for the has-last-good note.
  * Timed-out card requests were misclassified: brFetchWithTimeout aborts
    with a TimeoutError DOMException, but the catch only treated AbortError
    as a transient timeout, so a real timeout fell into the generic error
    path and fabricated a summary payload that overwrote the
    server-rendered record row.

Server (routes/user_pages_bp.py _build_live_matchups):
  * The 30s TTL cache had no single-flight or stale serve. On game day,
    once an entry expired, every concurrent card request for that league
    re-paid the full provider build in parallel, pinning request threads;
    builds could outlast the client's 25s abort, so from the client's
    side the preview never landed.
  * 2026-10-27 reliability fix: the request thread never builds and never
    waits. A miss kicks one background rebuild and returns None (callers
    translate it to the pending shape); any cached entry, inside the
    stale grace or beyond it, is served stale while the rebuild lands.
"""

import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── Client contracts ─────────────────────────────────────────────────────────

def _portfolio_client():
    source = (ROOT / "static" / "app.js").read_text()
    return source.split("/* Portfolio cards: one generation owns requests")[1]


def test_card_retries_when_matchup_fails_even_if_summary_loaded():
    client = _portfolio_client()
    # The retry decision is summary OR matchup, not summary alone.
    assert "else if (!result.summary || !result.matchup)" in client


def test_card_timeout_is_transient_not_a_fabricated_summary_error():
    client = _portfolio_client()
    # brFetchWithTimeout aborts with a TimeoutError DOMException; the catch
    # must treat it like AbortError (quiet bounded retry), not fall through
    # to the generic error path that overwrites the record row.
    assert "error.name === 'AbortError' || error.name === 'TimeoutError'" in client


def test_render_matchup_incomplete_shape_shows_retry_not_skeleton():
    client = _portfolio_client()
    incomplete = client.split("if (!data.live || !data.you)")[1].split("return true;")[0]
    # With no last-good score, an incomplete answer renders the recovery
    # state (with Retry) instead of leaving the skeleton up forever.
    assert "data-matchup-retry" in incomplete
    assert "Matchup temporarily unavailable" in incomplete
    # The has-last-good behavior is unchanged: keep showing the last score.
    assert "showing last score" in incomplete


def test_render_matchup_pending_eventually_surfaces_slow_state():
    client = _portfolio_client()
    pending = client.split("if (data.pending)")[1].split("delete slot.dataset.pendingSince;")[0]
    # A never-loaded card stamps when its pending stretch began and, past
    # ~90s, swaps the skeleton for an honest slow state with a Retry button.
    assert "pendingSince" in pending
    assert "90000" in pending
    assert "data-matchup-retry" in pending
    # A later non-pending render clears the stamp so the next pending
    # stretch is timed from its own start.
    assert "delete slot.dataset.pendingSince;" in client


# ── Server: live-matchup build coalescing ────────────────────────────────────

CTX = {}


@pytest.fixture()
def live_matchup_env(monkeypatch):
    pytest.importorskip("flask")
    import routes.user_pages_bp as pages

    pages._LIVE_MATCHUP_CACHE.clear()
    getattr(pages, "_LIVE_MATCHUP_LOCKS", {}).clear()
    getattr(pages, "_LIVE_MATCHUP_REBUILDING", set()).clear()
    calls = {"build": 0}

    def _preview(**_kwargs):
        calls["build"] += 1
        return [{"left": {"roster_id": "1"}, "right": {"roster_id": "2"}}]

    monkeypatch.setattr(
        "dashboard_services.matchups.build_matchup_preview", _preview)
    monkeypatch.setattr(
        "utils.utils.build_status_for_week", lambda *a, **k: {})
    monkeypatch.setattr(
        pages, "build_projections_by_week",
        lambda season, week, _settings=None: {week: {"projections": {}}})
    return pages, calls


def _key():
    return ("sleeper", "L1", 2026, 6)


def test_live_matchup_miss_kicks_background_build_and_caches(live_matchup_env):
    pages, calls = live_matchup_env
    # A cold miss never builds on the request thread: it returns the
    # pending signal (None) and kicks exactly one background rebuild.
    first = pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX)
    assert first is None
    assert _wait_until(lambda: _key() in pages._LIVE_MATCHUP_CACHE)
    assert _wait_until(lambda: _key() not in pages._LIVE_MATCHUP_REBUILDING)
    # The landed entry is then served from cache within the TTL, with no
    # second build.
    second = pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX)
    third = pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX)
    assert second is third
    assert second[0]  # the preview payload the background build computed
    assert calls["build"] == 1


def test_live_matchup_serves_stale_while_rebuild_in_flight(live_matchup_env):
    pages, calls = live_matchup_env
    stale = ([{"stale": True}], {}, {})
    # Past the 30s TTL but inside the stale grace window.
    pages._LIVE_MATCHUP_CACHE[_key()] = (time.time() - 60, stale)
    lock = pages._live_matchup_lock(_key())
    assert lock.acquire(blocking=False)
    try:
        served = pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX)
    finally:
        lock.release()
    # The request coalesced onto the in-flight rebuild: it served last-good
    # immediately instead of paying a second parallel provider build.
    assert served is stale
    assert calls["build"] == 0


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_live_matchup_stale_serves_immediately_and_rebuilds_in_background(
        live_matchup_env, monkeypatch):
    pages, calls = live_matchup_env
    stale = ([{"stale": True}], {}, {})
    # Past the 30s TTL but inside the stale grace window.
    pages._LIVE_MATCHUP_CACHE[_key()] = (time.time() - 60, stale)
    gate = threading.Event()
    entered = threading.Event()

    def _slow_preview(**_kwargs):
        entered.set()
        gate.wait(timeout=5)
        calls["build"] += 1
        return [{"left": {"roster_id": "1"}, "right": {"roster_id": "2"}}]

    monkeypatch.setattr(
        "dashboard_services.matchups.build_matchup_preview", _slow_preview)
    try:
        served = pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX)
        # The request returned the stale payload while the rebuild was still
        # gated shut behind it: it never blocked on the provider build.
        assert served is stale
        assert entered.wait(timeout=2.0)  # a background rebuild was scheduled
        assert pages._LIVE_MATCHUP_CACHE[_key()][1] is stale
        assert calls["build"] == 0
    finally:
        gate.set()
    # The rebuild lands in the background and replaces the entry.
    assert _wait_until(lambda: pages._LIVE_MATCHUP_CACHE[_key()][1] is not stale)
    assert _wait_until(lambda: _key() not in pages._LIVE_MATCHUP_REBUILDING)
    assert calls["build"] == 1


def test_live_matchup_no_entry_returns_pending_without_blocking(
        live_matchup_env, monkeypatch):
    pages, calls = live_matchup_env
    gate = threading.Event()
    entered = threading.Event()

    def _slow_preview(**_kwargs):
        entered.set()
        gate.wait(timeout=5)
        calls["build"] += 1
        return [{"left": {"roster_id": "1"}, "right": {"roster_id": "2"}}]

    monkeypatch.setattr(
        "dashboard_services.matchups.build_matchup_preview", _slow_preview)
    try:
        started = time.time()
        served = pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX)
        # The request thread returned the pending signal immediately even
        # though the provider build it kicked is still gated shut: the
        # build happens on a background thread, not this one.
        assert served is None
        assert time.time() - started < 2.0
        assert entered.wait(timeout=2.0)
        assert _key() in pages._LIVE_MATCHUP_REBUILDING
    finally:
        gate.set()
    assert _wait_until(lambda: _key() in pages._LIVE_MATCHUP_CACHE)
    assert _wait_until(lambda: _key() not in pages._LIVE_MATCHUP_REBUILDING)
    assert calls["build"] == 1


def test_live_matchup_fresh_entry_returns_without_compute(live_matchup_env):
    pages, calls = live_matchup_env
    fresh = ([{"fresh": True}], {}, {})
    pages._LIVE_MATCHUP_CACHE[_key()] = (time.time(), fresh)
    served = pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX)
    assert served is fresh
    assert calls["build"] == 0
    # A fresh entry schedules no background rebuild either.
    assert _key() not in pages._LIVE_MATCHUP_REBUILDING


def test_live_matchup_miss_does_not_wait_for_inflight_builder(live_matchup_env):
    pages, calls = live_matchup_env
    fresh = ([{"fresh": True}], {}, {})
    lock = pages._live_matchup_lock(_key())
    assert lock.acquire(blocking=False)

    def _builder():
        time.sleep(0.3)
        pages._LIVE_MATCHUP_CACHE[_key()] = (time.time(), fresh)
        lock.release()

    threading.Thread(target=_builder).start()
    started = time.time()
    served = pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX)
    # Nothing cached and a builder in flight: the caller no longer waits
    # on the per-key lock (the old unbounded acquire pinned a request
    # thread for the whole provider build). It defers with the pending
    # signal and starts no duplicate build.
    assert served is None
    assert time.time() - started < 0.25
    assert calls["build"] == 0
    # Once the builder lands its entry, the next call serves it fresh.
    assert _wait_until(lambda: _key() in pages._LIVE_MATCHUP_CACHE)
    assert pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX) is fresh
    assert calls["build"] == 0


def test_live_matchup_failed_builder_rekicks_on_next_call(live_matchup_env):
    pages, calls = live_matchup_env
    lock = pages._live_matchup_lock(_key())
    assert lock.acquire(blocking=False)

    def _failed_builder():
        time.sleep(0.2)
        lock.release()  # finished without storing (it raised)

    threading.Thread(target=_failed_builder).start()
    # While the failed builder still holds the lock, the call defers and
    # cannot schedule its own rebuild.
    assert pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX) is None
    assert calls["build"] == 0
    # After it releases without storing, the next call defers again but
    # this time kicks the background rebuild, which lands the entry.
    assert _wait_until(lambda: not lock.locked())
    assert pages._build_live_matchups("sleeper", "L1", 2026, 6, CTX) is None
    assert _wait_until(lambda: _key() in pages._LIVE_MATCHUP_CACHE)
    assert calls["build"] == 1
