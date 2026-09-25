"""Regression: sync_league_globals must never leak one league's config into another.

dashboard_services/platform_api.sync_league_globals carried a TODO about
process-global league configuration. The context is request-scoped (flask.g)
with a thread-local fallback; these tests pin the isolation guarantees at the
seam:

1. A failed/empty provider sync clears the context instead of leaving the
   previous league's config behind (the cross-league contamination vector).
2. Two sequential syncs for different leagues leave exactly the second
   league's config -- no residue from the first.
3. set_league_globals copies values in; readers get copies out (no aliasing
   with provider caches or caller-owned objects).
4. clear_league_globals is an explicit reset point for multi-league,
   out-of-request processing on one thread.
"""
import pytest

# dashboard_services.api pulls in the app stack (requests/flask); skip on the
# slim lint shard that doesn't install them, like the other app-stack tests.
pytest.importorskip("requests")
pytest.importorskip("flask")

from dashboard_services import platform_api
from dashboard_services.api import (
    clear_league_globals,
    get_effective_scoring_settings,
    get_league_settings,
    get_provider_scoring_settings,
    get_roster_positions,
    get_total_rosters,
    set_league_globals,
)

_LEAGUE_A = {
    "scoring_settings": {"rec": 1.0, "zz_marker": "league-a"},
    "roster_positions": ["QB", "RB", "WR"],
    "league_settings": {"playoff_week_start": 15},
    "total_rosters": 12,
}
_LEAGUE_B = {
    "scoring_settings": {"rec": 0.0, "zz_marker": "league-b"},
    "roster_positions": ["QB", "RB", "RB", "WR", "TE"],
    "league_settings": {"playoff_week_start": 14},
    "total_rosters": 10,
}


class _FakeProvider:
    def __init__(self, payload=None, exc=None):
        self._payload = payload
        self._exc = exc
        self.metadata = type("Meta", (), {"key": "fake"})()

    def get_league_globals(self, league_id, season):
        if self._exc is not None:
            raise self._exc
        return self._payload


def _sync(monkeypatch, payload=None, exc=None):
    monkeypatch.setattr(
        platform_api, "get_provider",
        lambda platform: _FakeProvider(payload, exc))
    platform_api.sync_league_globals("fake", "L1", 2026)


@pytest.fixture(autouse=True)
def _clean_state():
    # Thread-local fallback persists across tests on one thread; reset it.
    clear_league_globals()
    yield
    clear_league_globals()


def _assert_empty():
    assert get_roster_positions() == []
    assert get_league_settings() == {}
    assert get_total_rosters() == 0
    assert "zz_marker" not in get_effective_scoring_settings()


def test_failed_sync_does_not_leave_previous_league(monkeypatch):
    _sync(monkeypatch, payload=_LEAGUE_A)
    assert get_provider_scoring_settings()["zz_marker"] == "league-a"
    _sync(monkeypatch, payload={})  # provider failure shape: empty dict
    _assert_empty()


def test_none_sync_does_not_leave_previous_league(monkeypatch):
    _sync(monkeypatch, payload=_LEAGUE_A)
    _sync(monkeypatch, payload=None)  # e.g. Sleeper adapter, league missing
    _assert_empty()


def test_exception_sync_does_not_leave_previous_league(monkeypatch):
    _sync(monkeypatch, payload=_LEAGUE_A)
    _sync(monkeypatch, exc=RuntimeError("provider down"))  # must not raise
    _assert_empty()


def test_sequential_syncs_leave_only_latest_league(monkeypatch):
    _sync(monkeypatch, payload=_LEAGUE_A)
    _sync(monkeypatch, payload=_LEAGUE_B)
    assert get_roster_positions() == ["QB", "RB", "RB", "WR", "TE"]
    assert get_league_settings() == {"playoff_week_start": 14}
    assert get_total_rosters() == 10
    effective = get_effective_scoring_settings()
    assert effective["zz_marker"] == "league-b"
    assert effective["rec"] == 0.0


def test_set_copies_values_in():
    scoring = {"rec": 1.0}
    positions = ["QB"]
    settings = {"playoff_week_start": 15}
    set_league_globals(scoring_settings=scoring, roster_positions=positions,
                       league_settings=settings, total_rosters=12)
    # Mutating the caller's objects afterwards must not corrupt the context.
    scoring["rec"] = 99.0
    positions.append("K")
    settings["playoff_week_start"] = 1
    assert get_provider_scoring_settings()["rec"] == 1.0
    assert get_roster_positions() == ["QB"]
    assert get_league_settings() == {"playoff_week_start": 15}


def test_readers_return_copies():
    set_league_globals(scoring_settings=_LEAGUE_A["scoring_settings"],
                       roster_positions=_LEAGUE_A["roster_positions"],
                       league_settings=_LEAGUE_A["league_settings"],
                       total_rosters=_LEAGUE_A["total_rosters"])
    get_roster_positions().append("K")
    get_league_settings()["injected"] = True
    get_provider_scoring_settings()["rec"] = 99.0
    # The stored context is untouched by caller-side mutation.
    assert get_roster_positions() == ["QB", "RB", "WR"]
    assert get_league_settings() == {"playoff_week_start": 15}
    assert get_provider_scoring_settings()["rec"] == 1.0


def test_clear_resets_context():
    set_league_globals(scoring_settings=_LEAGUE_A["scoring_settings"],
                       roster_positions=_LEAGUE_A["roster_positions"],
                       league_settings=_LEAGUE_A["league_settings"],
                       total_rosters=_LEAGUE_A["total_rosters"])
    clear_league_globals()
    _assert_empty()
