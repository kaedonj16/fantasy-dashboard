"""Concurrency-cap proof for the modal league warm fan-out.

_kick_modal_league_warm fires once per league from
get_league_ctx_from_cache, and one warm runs the player modal's full-season
cache builds (weekly points x2 seasons, rank rows, weekstat pools,
trade-history inputs). A cold My Leagues load lands 11 league contexts in
one worker, so uncapped kicks stacked 11 heavy build threads at once --
the fan-out behind the 2026-10-26 OOM restarts. The cap
(_MODAL_WARM_SEM, 2) must bound concurrent warms per worker, DROP kicks
that cannot get a slot (not queue them), and clear the dropped league's
dedupe mark so a later context read re-kicks it.
"""
import importlib
import threading
import time

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

import app as appmod  # noqa: E402

N_LEAGUES = 6
CAP = 2


class _Tracker:
    """Counts concurrent executions of the blocking weekly-points build."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cur = 0
        self.max_seen = 0
        self.calls = 0
        self.gate = threading.Event()  # while unset, builds block inside

    def build(self, season, scoring):
        with self.lock:
            self.cur += 1
            self.calls += 1
            self.max_seen = max(self.max_seen, self.cur)
        self.gate.wait(timeout=15)
        with self.lock:
            self.cur -= 1
        return {}


def _marked_league_ids():
    return {key[1] for key in appmod._MODAL_LEAGUE_WARMED}


def _wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture()
def _warm_mocks(monkeypatch):
    svc_api = importlib.import_module("dashboard_services.api")
    plt = importlib.import_module("dashboard_services.player_league_trades")
    svc = importlib.import_module("dashboard_services.service")

    monkeypatch.setitem(appmod.app.config, "TESTING", False)
    appmod._MODAL_LEAGUE_WARMED.clear()

    scoring_by_lid = {
        f"L{i}": {"rec": 0.5, "pass_yd": 0.04 + i} for i in range(N_LEAGUES)
    }

    def _get_league(*a, **k):
        lid = a[0] if a else k.get("league_id")
        return {"scoring_settings": scoring_by_lid[str(lid)]}

    monkeypatch.setattr(svc_api, "get_league", _get_league)
    monkeypatch.setattr(
        svc_api, "build_league_history_map",
        lambda *a, **k: {2026: str(a[1])})
    tracker = _Tracker()
    monkeypatch.setattr(appmod, "_load_season_weekly_points", tracker.build)
    monkeypatch.setattr(appmod, "_season_rank_rows", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_season_weekstat_points", lambda *a, **k: None)
    monkeypatch.setattr(svc, "get_transactions_by_week", lambda *a, **k: None)
    monkeypatch.setattr(plt, "_roster_names", lambda *a, **k: None)
    monkeypatch.setattr(plt, "build_draft_resolution_map", lambda *a, **k: None)
    monkeypatch.setattr(plt, "_slot_map_for_league", lambda *a, **k: None)
    yield tracker
    tracker.gate.set()  # never leave a daemon warm blocked at teardown
    appmod._MODAL_LEAGUE_WARMED.clear()


def _kick(i):
    ctx = {"scoring_settings": {"rec": 0.5, "pass_yd": 0.04 + i}}
    appmod._kick_modal_league_warm("sleeper", f"L{i}", 2026, ctx)


def test_modal_warm_cap_drops_and_rekicks(_warm_mocks):
    tracker = _warm_mocks

    # Fire more simultaneous distinct-league kicks than the cap allows.
    for i in range(N_LEAGUES):
        _kick(i)

    # Exactly CAP warms get inside the heavy build; the rest never start.
    assert _wait_for(lambda: tracker.calls >= CAP)
    time.sleep(0.5)
    assert tracker.calls == CAP
    assert tracker.max_seen == CAP

    # Dropped kicks are un-marked (re-kickable); running kicks stay marked.
    marked = _marked_league_ids()
    assert len(marked) == CAP
    dropped = {f"L{i}" for i in range(N_LEAGUES)} - marked
    assert len(dropped) == N_LEAGUES - CAP

    # Let the running warms finish; each warm calls the weekly build twice
    # (current + previous season), so the call count lands at 2 * CAP.
    tracker.gate.set()
    assert _wait_for(lambda: tracker.calls >= 2 * CAP and tracker.cur == 0)

    # A dropped league can be kicked again once a slot has freed, and its
    # builds actually run (two more weekly calls).
    target = sorted(dropped)[0]
    idx = int(target[1:])
    _kick(idx)
    assert _wait_for(lambda: tracker.calls >= 2 * CAP + 2)
    assert target in _marked_league_ids()
    assert tracker.max_seen == CAP  # the cap held throughout
