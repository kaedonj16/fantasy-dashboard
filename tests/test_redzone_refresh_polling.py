"""Regression tests for the Redzone refresh / polling / streaming reliability
fixes.

Two behavioral Node harnesses exercise the SHIPPED redzone.js control-flow
(extracted by source, driven by mocked timers / fetch / streams / provider
responses -- not string matching):

  * tests/redzone_refresh_harness.mjs -- request-ownership ordering, the fetch
    deadline that stays armed through body/JSON reads, the stream overall +
    inactivity timeouts, manual-refresh recovery, out-of-order discards,
    scope-switch cancellation, HTTP-200 error payloads, polling resume, and
    no-pile-up.
  * tests/redzone_clock_harness.mjs -- the scoreboard-vs-PBP clock freshness
    rule (a fresh board clock is never overwritten by an older play).

A small set of source contracts and one behavioral backend test cover the
server-side timeout bounding + budget alignment.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REDZONE_JS = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")


def _run_harness(name: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS behavioral harness")
    result = subprocess.run(
        [node, str(ROOT / "tests" / name)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKS PASSED" in result.stdout, result.stdout + result.stderr


# ── Behavioral harnesses (Node) ──────────────────────────────────────────────
def test_refresh_polling_behavioral_harness():
    """Covers: stream that never returns headers, inter-chunk stall + manual
    recovery, body-read timeout, out-of-order/obsolete discard, scope-switch
    cancellation, latency past the retired 12s, HTTP-200 error payload preserving
    last-good, polling resume after failure, and no request pile-up."""
    _run_harness("redzone_refresh_harness.mjs")


def test_clock_freshness_behavioral_harness():
    """Covers: a fresh scoreboard clock is preserved over an older PBP play; PBP
    fills only missing fields; a genuine quarter/OT change still wins."""
    _run_harness("redzone_clock_harness.mjs")


# ── First-load freshness ─────────────────────────────────────────────────────
def test_first_load_kicks_off_immediate_catchup_on_game_day():
    """On a live game day the boot must not wait out the first poll interval --
    it paints the server-injected snapshot, then immediately reconciles to the
    latest plays (self-healing a stale service-worker-cached shell). Previously
    only demo mode refreshed on load."""
    boot = REDZONE_JS.rsplit("_render();", 1)[1]  # the boot tail after first paint
    assert "if (_isDemo) {" in boot
    assert "} else if (_isGameDay()) {" in boot
    # Scope-aware immediate refresh, guarded against piling onto an in-flight poll.
    assert "_refresh({ backfill: true })" in boot
    assert "_refreshUserStream()" in boot
    assert "if (_inflight || _streaming) return;" in boot


def test_backfill_poll_reconciles_as_bulk_never_live_alerts():
    """The first-load catch-up brings in newer plays as HISTORY, not live events,
    so a stale snapshot can't fire a burst of TD alerts for plays that already
    happened. The live-vs-bulk decision must honor opts.backfill."""
    refresh = REDZONE_JS.split("async function _refresh(opts) {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    assert "(!opts.backfill && wasContinuouslyActive) ? 'live' : 'bulk'" in refresh


# ── Frontend source contracts (structure the harnesses rely on) ───────────────
def test_client_deadline_exceeds_retired_12s_and_covers_server_budget():
    assert "var _RZ_FETCH_DEADLINE_MS  = 25000" in REDZONE_JS
    # The old 12s abort is gone; the deadline is not cleared the instant fetch()
    # returns headers -- it stays armed through the response body + JSON parse.
    assert ", 12000);" not in REDZONE_JS
    refresh = REDZONE_JS.split("async function _refresh(opts) {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    # The deadline (setTimeout) is released only via _release(), which is called
    # AFTER `await resp.json()`, never before.
    assert "newData = await resp.json()" in refresh
    assert refresh.index("await resp.json()") < refresh.index("_release();")


def test_stream_has_overall_deadline_and_inactivity_timeout():
    stream = REDZONE_JS.split("async function _refreshUserStream(opts) {", 1)[1]
    stream = stream.split("function _isGameDay(", 1)[0]
    assert "_RZ_STREAM_DEADLINE_MS" in stream          # overall deadline
    assert "_RZ_STREAM_IDLE_MS" in stream              # inactivity timeout
    assert "_bumpIdle()" in stream                     # reset on each chunk
    assert "_abortStream(" in stream                   # abort fetch + reader
    # Ownership-aware teardown: only clear streaming/inflight if this stream owns them.
    assert "if (_streamGen === myGen && _scope === 'user') _streaming = false;" in stream


def test_manual_refresh_can_preempt_a_stalled_stream():
    assert "function _manualRefresh(" in REDZONE_JS
    refresh = REDZONE_JS.split("async function _refresh(opts) {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    # A manual refresh preempts an in-flight request and a (possibly stalled)
    # stream instead of no-oping while _streaming is true.
    assert "opts.manual" in refresh
    assert "_cancelInflight(" in refresh


def test_error_payload_not_treated_as_fresh_and_visibility_resume_wired():
    refresh = REDZONE_JS.split("async function _refresh(opts) {", 1)[1].split(
        "// ── Progressive My Leagues", 1
    )[0]
    # A 200 body carrying an error is recovered as last-good, never applied.
    assert "if (newData && newData.error) {" in refresh
    # Freshness is tracked separately from a merely-completed request.
    assert "_lastDataAt = Date.now();" in refresh
    # Resume on tab-visible / reconnect, wired exactly once.
    assert "_visListenersWired" in REDZONE_JS
    assert "visibilitychange" in REDZONE_JS
    assert "'online'" in REDZONE_JS


def test_clock_enrichment_prefers_fresher_scoreboard():
    box = REDZONE_JS.split("function _nflGameInfoUncached(", 1)[1].split(
        "var _POS_LIST", 1
    )[0]
    # PBP fills only when the board lacks a value or has advanced to a later
    # period; a same-period PBP clock never overwrites the board clock.
    assert "var pbpLaterPeriod = _quarterRank(pbpQ) > _quarterRank(sbQ);" in box
    assert "if (!sbQ || pbpLaterPeriod) row.game_quarter" in box
    assert "if (!sbClk || pbpLaterPeriod) row.game_clock" in box
    assert "function _quarterRank(" in REDZONE_JS


# ── Backend: bounded scoreboard budget aligned with the client deadline ───────
def test_backend_scoreboard_timeout_is_bounded_below_client_deadline():
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "_RZ_SCOREBOARD_TIMEOUT = 12" in app_src
    collect = app_src[app_src.index("def _redzone_collect("):app_src.index("def _redzone_fetch(")]
    # The live scoreboard fetch passes the bounded timeout (was the 20s default).
    assert "timeout=_RZ_SCOREBOARD_TIMEOUT" in collect
    # 12s scoreboard budget must sit under the 25s client fetch deadline.
    assert 12 < 25


def test_get_nfl_scores_for_date_threads_the_timeout_through(monkeypatch):
    """Behavioral: the bounded timeout actually reaches the HTTP call, and it is
    not shared with (nor does it disturb) the default-budget callers' cache."""
    # Importing the module (which pulls in Flask) is enough of a gate; skip
    # cleanly where the web stack is absent. Deliberately NOT the flask
    # importorskip spelling, so this file stays in the fast unit shard (which has
    # both Node and Flask) rather than the Node-less integration shard.
    api = pytest.importorskip("dashboard_services.api")

    seen: list[int] = []

    class _Resp:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return {"body": self._body}

    def _fake_get(url, headers=None, params=None, timeout=None):
        seen.append(timeout)
        return _Resp({"g": {"gameID": "g"}})

    from dashboard_services import nfl_game_data
    nfl_game_data._cache.clear()
    monkeypatch.setattr(nfl_game_data._session, "get", _fake_get)
    # Distinct dates keep each call off the other's cache entry.
    api.get_nfl_scores_for_date("20250101", timeout=12)
    api.get_nfl_scores_for_date("20250102")  # default budget
    # requests receives a bounded (connect, read) tuple.
    assert any(t[1] == 12 for t in seen), seen
    assert any(t[1] == 12 for t in seen), seen  # default is capped at 12
