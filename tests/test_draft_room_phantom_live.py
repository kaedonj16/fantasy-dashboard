"""Phantom-live Draft Room fix: pre_draft must not poll like a live draft.

JS behaviour is guarded at source level (repo convention for draft_room.js);
the history=1 detect caching/parallel walk is exercised through the real
Flask route with the provider layer faked.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

REPO = Path(__file__).resolve().parents[1]
ROOM_JS = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
WIZARD_JS = (REPO / "static" / "draft_room_wizard.js").read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


# ── JS: pre_draft is upcoming, not live ──────────────────────────────────────

def test_poll_gate_helpers_use_15_minute_window():
    assert "function _preDraftInWindow(startMs)" in ROOM_JS
    assert "delta <= _START_WINDOW_MS && delta >= -_START_WINDOW_MS" in ROOM_JS
    assert "function _livePollAllowed(status, startMs)" in ROOM_JS
    gate = _section(ROOM_JS, "function _livePollAllowed", "function _clearUpcomingRecheck")
    assert "s === 'drafting' || s === 'paused'" in gate
    assert "if (s === 'pre_draft') return _preDraftInWindow(startMs);" in gate


def test_connect_gates_polling_and_arms_window_recheck():
    section = _section(ROOM_JS, "function _connectLiveApply", "function liveSig")
    assert "if (_livePollAllowed(String(d.status), state.startTime)){" in section
    assert "_armUpcomingRecheck();" in section
    # Outside the window there is no startPolling on the connect path other
    # than the gated one.
    assert section.count("startPolling();") == 1


def test_connect_loads_player_pool_exactly_once():
    section = _section(ROOM_JS, "function _connectLiveApply", "function liveSig")
    assert section.count("loadPlayers();") == 1


def test_start_polling_itself_refuses_far_future_predraft():
    section = _section(ROOM_JS, "function startPolling(){", "function schedulePoll(){")
    assert "!_livePollAllowed(state.status, state.startTime)" in section
    assert "_armUpcomingRecheck();" in section


def test_poll_loop_stops_when_start_moves_outside_window():
    section = _section(ROOM_JS, "function pollOnce(){", "function stopPolling(){")
    assert "String(state.status) === 'pre_draft' && !_preDraftInWindow(state.startTime)" in section
    assert "stopPolling(); _armUpcomingRecheck();" in section


def test_syncing_indicator_hidden_for_upcoming_board():
    section = _section(ROOM_JS, "function updatePollStatus(){", "function updateDraftBanner")
    assert "String(state.status) === 'pre_draft' && !_preDraftInWindow(state.startTime)" in section


# ── JS: resume revalidation ──────────────────────────────────────────────────

def test_resume_revalidates_instead_of_polling_blindly():
    section = _section(ROOM_JS, "function resumeFromSession(){", "function _revalidateResumedLive")
    assert "_revalidateResumedLive();" in section
    assert "startPolling()" not in section


def test_revalidate_only_polls_when_currently_live():
    section = _section(ROOM_JS, "function _revalidateResumedLive(){", "function buildTeamTip")
    assert "if (_livePollAllowed(status, state.startTime)){" in section
    assert "startPolling();" in section
    assert "state.isComplete = true;" in section
    assert "_armUpcomingRecheck();" in section
    assert "Could not confirm this draft is live - not syncing" in section


# ── JS: detectLive completed fallback removed ────────────────────────────────

def test_detect_live_never_falls_back_to_completed_as_live():
    assert "active.length ? active : all" not in ROOM_JS
    section = _section(ROOM_JS, "function detectLive(){", "function buildOwnedFromResponse")
    assert "Completed drafts - view only" in section
    assert "No live or upcoming draft found for this league." in section
    assert "String(d.status) !== 'complete'" in section
    # The explicit user Connect flow keeps its heavy sync (contrast: wizard
    # load probe below must not have it).
    assert "&sync=1" in section


# ── JS: Sleeper poll error backoff + stop ────────────────────────────────────

def test_sleeper_poll_errors_backoff_and_stop():
    assert "var SLEEPER_POLL_ERR_LIMIT = 4;" in ROOM_JS
    assert "function _registerSleeperPollError(){" in ROOM_JS
    assert ROOM_JS.count("_registerSleeperPollError();") >= 3
    assert "Connection lost - stopped checking" in ROOM_JS
    section = _section(ROOM_JS, "function schedulePoll(){", "function pollOnce(){")
    assert "_sleeperPollErrs > 0" in section
    assert "Math.pow(2, _sleeperPollErrs)" in section
    # A successful poll resets the consecutive-error count.
    assert "_sleeperPollErrs = 0; _pollStopped = false;" in ROOM_JS


# ── Wizard: light probe, skipped on resumed live board ───────────────────────

def test_wizard_probe_is_light_and_skips_resumed_live():
    assert "&sync=1" not in WIZARD_JS   # the probe URL must stay light
    assert "_wzResumedLive" in WIZARD_JS
    assert "_wzSaved.mode === 'live' && _wzSaved.sourceDraftId" in WIZARD_JS


# ── Backend: history=1 detect cache + parallel walk ──────────────────────────

import routes.draft_api_bp as bp  # noqa: E402


def _draft(lid, draft_id=None):
    return {
        "draft_id": draft_id or f"D-{lid}",
        "status": "complete",
        "type": "snake",
        "season": 2026,
        "start_time": 123,
        "league_id": lid,
        "settings": {"rounds": 15, "teams": 12},
    }


@pytest.fixture
def detect_client(monkeypatch):
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(bp.draft_api_bp)
    bp._detect_history_cache.clear()
    calls = []
    delays = {"L3": 0.15, "L2": 0.0, "L1": 0.0}   # newest league is SLOWEST

    def fake_get_drafts(platform, league_id, season):
        calls.append(league_id)
        time.sleep(delays.get(league_id, 0.0))
        return [_draft(league_id)]

    monkeypatch.setattr(bp, "get_drafts", fake_get_drafts)
    monkeypatch.setattr(
        bp, "build_league_history_map",
        lambda platform, league_id, season: {2025: "L2", 2024: "L1"},
    )
    monkeypatch.setattr(
        bp, "_live_draft_type",
        lambda rounds, platform, league_id, season, cache_key=None: "redraft",
    )
    monkeypatch.setattr(bp, "_order_from_sleeper", lambda d: "snake")
    with app.test_client() as client:
        client._calls = calls  # type: ignore[attr-defined]
        yield client
    bp._detect_history_cache.clear()


def test_history_detect_shape_and_order_preserved(detect_client):
    resp = detect_client.get(
        "/api/draft/detect?platform=sleeper&league_id=L3&season=2026&history=1")
    assert resp.status_code == 200
    drafts = resp.get_json()["drafts"]
    # Newest league first even though it answered last (parallel map order).
    assert [d["draft_id"] for d in drafts] == ["D-L3", "D-L2", "D-L1"]
    first = drafts[0]
    assert first["status"] == "complete"
    assert first["draft_type"] == "redraft"
    assert first["teams"] == 12
    assert first["rounds"] == 15
    assert first["order"] == "snake"
    assert first["start_time"] == 123


def test_history_detect_second_load_is_cached(detect_client):
    url = "/api/draft/detect?platform=sleeper&league_id=L3&season=2026&history=1"
    first = detect_client.get(url).get_json()
    assert len(detect_client._calls) == 3
    second = detect_client.get(url).get_json()
    assert second == first
    assert len(detect_client._calls) == 3   # no provider calls on cache hit


def test_detect_without_history_is_not_cached(detect_client):
    url = "/api/draft/detect?platform=sleeper&league_id=L3&season=2026"
    detect_client.get(url)
    detect_client.get(url)
    assert detect_client._calls == ["L3", "L3"]   # current league only, live


def test_history_detect_dedupes_across_seasons(detect_client, monkeypatch):
    def dup_get_drafts(platform, league_id, season):
        if league_id == "L2":
            return [_draft("L2", draft_id="D-L3"), _draft("L2")]
        return [_draft(league_id)]

    monkeypatch.setattr(bp, "get_drafts", dup_get_drafts)
    resp = detect_client.get(
        "/api/draft/detect?platform=sleeper&league_id=L3&season=2026&history=1")
    ids = [d["draft_id"] for d in resp.get_json()["drafts"]]
    assert ids == ["D-L3", "D-L2", "D-L1"]


def test_history_detect_tolerates_one_season_failing(detect_client, monkeypatch):
    def flaky_get_drafts(platform, league_id, season):
        if league_id == "L2":
            raise RuntimeError("provider down")
        return [_draft(league_id)]

    monkeypatch.setattr(bp, "get_drafts", flaky_get_drafts)
    resp = detect_client.get(
        "/api/draft/detect?platform=sleeper&league_id=L3&season=2026&history=1")
    ids = [d["draft_id"] for d in resp.get_json()["drafts"]]
    assert ids == ["D-L3", "D-L1"]
