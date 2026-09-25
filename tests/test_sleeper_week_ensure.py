"""On-demand backfill of runtime-fetched Sleeper weekly stat files.

Regression test: cache/sleeper_stats/sleeper_stats_s{season}_w*.json for the
live season are fetched at runtime (only past seasons ship in the repo) and
Render has no persistent disk, so every deploy wipes them. The player-modal
paths only globbed those files -- they never fetched -- so after a deploy
the game log lost its actuals (projections skip finished weeks) and season
PPG/total went N/A until something else happened to fetch those weeks.
"""
import time

import pytest

flask = pytest.importorskip("flask")
app = pytest.importorskip("app")


class _Policy:
    def __init__(self, weeks):
        self.completed_weeks = tuple(weeks)


@pytest.fixture()
def ensure_state(monkeypatch):
    """Reset the helper's cooldown maps and stub its collaborators.

    Files are simulated as missing (fresh deploy: cache wiped).
    """
    monkeypatch.setattr(app, "_SLEEPER_WEEK_ENSURE_TS", {})
    monkeypatch.setattr(app, "_SLEEPER_LIVE_WEEK_TS", {})
    calls = []

    def fake_policy(season):
        return _Policy([1, 2])

    def fake_fetch(season, week, force=False):
        calls.append((season, week, force))

    monkeypatch.setattr("utils.season_qualification.qualification_policy", fake_policy)
    monkeypatch.setattr(
        "data_building.external_data.sleeper_bulk_stats.fetch_week_stats", fake_fetch
    )
    monkeypatch.setattr(app, "_sleeper_week_cache_populated", lambda s, w: False)
    # The live-week kickoff check reads the real schedule cache; stub it so
    # these tests don't depend on what week it really is.
    monkeypatch.setattr(app, "_live_week_kickoff_passed", lambda s, w: False)
    return calls


def test_ensure_fetches_each_missing_completed_week(ensure_state):
    app._ensure_sleeper_week_files(2026)
    assert sorted(ensure_state) == [(2026, 1, False), (2026, 2, False)]


def test_ensure_skips_populated_files_without_fetch(monkeypatch):
    """Steady state: files on disk -> zero network, zero parsing.

    The whole point of the stat() pre-check: a modal open must not
    re-read/re-parse ~780KB files (or hit Sleeper) when there is nothing
    to backfill.
    """
    monkeypatch.setattr(app, "_SLEEPER_WEEK_ENSURE_TS", {})
    monkeypatch.setattr(app, "_SLEEPER_LIVE_WEEK_TS", {})
    monkeypatch.setattr(app, "_live_week_kickoff_passed", lambda s, w: False)
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy", lambda s: _Policy([1, 2])
    )
    monkeypatch.setattr(app, "_sleeper_week_cache_populated", lambda s, w: True)

    def boom(season, week, force=False):
        raise AssertionError("fetch must not run when files are populated")

    monkeypatch.setattr(
        "data_building.external_data.sleeper_bulk_stats.fetch_week_stats", boom
    )
    app._ensure_sleeper_week_files(2026)
    app._ensure_sleeper_week_files(2026)
    # No fetch, and the cooldowns stay unarmed (nothing to retry).
    assert app._SLEEPER_WEEK_ENSURE_TS == {}
    assert app._SLEEPER_LIVE_WEEK_TS == {}


def test_ensure_cooldown_skips_repeat(ensure_state):
    app._ensure_sleeper_week_files(2026)
    app._ensure_sleeper_week_files(2026)
    assert len(ensure_state) == 2


def test_ensure_fetches_missing_weeks_in_parallel(monkeypatch):
    """Cold post-deploy backfill must not fetch weeks one at a time.

    Sequential 20s-timeout fetches made a cold backfill take 60s+ and risk
    tripping gunicorn's 120s worker timeout on the request that triggered
    it (the Teams page first click). Weeks are independent files, so they
    fetch concurrently.
    """
    import threading

    monkeypatch.setattr(app, "_SLEEPER_WEEK_ENSURE_TS", {})
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy", lambda s: _Policy([1, 2, 3])
    )
    monkeypatch.setattr(app, "_sleeper_week_cache_populated", lambda s, w: False)

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()
    calls = []

    def fake_fetch(season, week):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            time.sleep(0.2)
            calls.append(week)
        finally:
            with lock:
                in_flight -= 1
        return {}

    monkeypatch.setattr(
        "data_building.external_data.sleeper_bulk_stats.fetch_week_stats", fake_fetch
    )
    app._ensure_sleeper_week_files(2026)

    assert sorted(calls) == [1, 2, 3]
    # Sequential fetches would never overlap; parallel ones do.
    assert max_in_flight >= 2


def test_ensure_refetches_after_cooldown(ensure_state):
    app._ensure_sleeper_week_files(2026)
    app._SLEEPER_WEEK_ENSURE_TS[2026] = time.time() - app._SLEEPER_WEEK_ENSURE_COOLDOWN_S - 1
    app._ensure_sleeper_week_files(2026)
    assert len(ensure_state) == 4


def test_ensure_never_raises_and_cools_down_on_failure(monkeypatch):
    monkeypatch.setattr(app, "_SLEEPER_WEEK_ENSURE_TS", {})
    monkeypatch.setattr(app, "_SLEEPER_LIVE_WEEK_TS", {})
    monkeypatch.setattr(app, "_live_week_kickoff_passed", lambda s, w: False)
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy", lambda s: _Policy([1])
    )
    monkeypatch.setattr(app, "_sleeper_week_cache_populated", lambda s, w: False)

    def boom(season, week, force=False):
        raise ConnectionError("sleeper down")

    monkeypatch.setattr(
        "data_building.external_data.sleeper_bulk_stats.fetch_week_stats", boom
    )
    app._ensure_sleeper_week_files(2026)  # must not raise
    # failure still arms the cooldown: no hot retry loop on the next call
    app._ensure_sleeper_week_files(2026)
    assert 2026 in app._SLEEPER_WEEK_ENSURE_TS


def test_ensure_ignores_bad_season(ensure_state):
    app._ensure_sleeper_week_files("not-a-season")
    assert ensure_state == []


def test_sleeper_stats_by_week_triggers_ensure(monkeypatch, tmp_path):
    """The game-log lookup itself backfills, covering _player_nfl_eligibility."""
    monkeypatch.setattr(app, "_SLEEPER_WEEK_ENSURE_TS", {})
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy", lambda s: _Policy([])
    )
    monkeypatch.setattr(
        "data_building.external_data.sleeper_bulk_stats.fetch_week_stats",
        lambda s, w: None,
    )
    seen = []
    monkeypatch.setattr(
        app, "_ensure_sleeper_week_files", lambda season: seen.append(season)
    )
    # Point CACHE_DIR at an empty dir so the glob finds nothing; the point
    # is the ensure call, not the parse.
    monkeypatch.setattr(app, "CACHE_DIR", str(tmp_path))
    assert app._sleeper_stats_by_week("10229", 2026) == {}
    assert seen == [2026]


def test_ensure_refreshes_live_week_after_kickoff(ensure_state, monkeypatch):
    """Regression: a Thursday-night game must land in the Stats tab.

    Week 3 is in progress (never "completed"), so the old helper never
    fetched its file and the game log kept showing the projection. Once the
    week's first kickoff has passed, the live week refetches with force=True.
    """
    monkeypatch.setattr(app, "_live_week_kickoff_passed", lambda s, w: True)
    app._ensure_sleeper_week_files(2026)
    assert (2026, 3, True) in ensure_state


def test_ensure_skips_live_week_before_kickoff(ensure_state):
    """No live fetch when the week's games haven't started yet."""
    app._ensure_sleeper_week_files(2026)
    assert all(week != 3 for _, week, _ in ensure_state)
    assert app._SLEEPER_LIVE_WEEK_TS == {}


def test_ensure_live_week_cooldown(ensure_state, monkeypatch):
    """The live-week refresh is cooldown-guarded, not per-modal-open."""
    monkeypatch.setattr(app, "_live_week_kickoff_passed", lambda s, w: True)
    app._ensure_sleeper_week_files(2026)
    app._ensure_sleeper_week_files(2026)
    live = [c for c in ensure_state if c[1] == 3]
    assert live == [(2026, 3, True)]


def test_ensure_live_week_never_raises(monkeypatch):
    """A Sleeper outage during the live refresh must not break modal opens."""
    monkeypatch.setattr(app, "_SLEEPER_WEEK_ENSURE_TS", {})
    monkeypatch.setattr(app, "_SLEEPER_LIVE_WEEK_TS", {})
    monkeypatch.setattr(app, "_live_week_kickoff_passed", lambda s, w: True)
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy", lambda s: _Policy([1, 2])
    )
    monkeypatch.setattr(app, "_sleeper_week_cache_populated", lambda s, w: True)

    def boom(season, week, force=False):
        raise ConnectionError("sleeper down")

    monkeypatch.setattr(
        "data_building.external_data.sleeper_bulk_stats.fetch_week_stats", boom
    )
    app._ensure_sleeper_week_files(2026)  # must not raise
    assert 2026 in app._SLEEPER_LIVE_WEEK_TS


def test_ensure_skips_live_week_when_season_complete(ensure_state, monkeypatch):
    """No live week exists once all 18 weeks are completed."""
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy",
        lambda s: _Policy(list(range(1, 19))),
    )
    monkeypatch.setattr(app, "_sleeper_week_cache_populated", lambda s, w: True)
    monkeypatch.setattr(app, "_live_week_kickoff_passed", lambda s, w: True)
    app._ensure_sleeper_week_files(2026)
    assert ensure_state == []
    assert app._SLEEPER_LIVE_WEEK_TS == {}
