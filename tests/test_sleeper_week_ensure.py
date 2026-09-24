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
    """Reset the helper's cooldown map and stub its two collaborators."""
    monkeypatch.setattr(app, "_SLEEPER_WEEK_ENSURE_TS", {})
    calls = []

    def fake_policy(season):
        return _Policy([1, 2])

    def fake_fetch(season, week):
        calls.append((season, week))

    monkeypatch.setattr("utils.season_qualification.qualification_policy", fake_policy)
    monkeypatch.setattr(
        "data_building.external_data.sleeper_bulk_stats.fetch_week_stats", fake_fetch
    )
    return calls


def test_ensure_fetches_each_completed_week(ensure_state):
    app._ensure_sleeper_week_files(2026)
    assert sorted(ensure_state) == [(2026, 1), (2026, 2)]


def test_ensure_cooldown_skips_repeat(ensure_state):
    app._ensure_sleeper_week_files(2026)
    app._ensure_sleeper_week_files(2026)
    assert len(ensure_state) == 2


def test_ensure_refetches_after_cooldown(ensure_state):
    app._ensure_sleeper_week_files(2026)
    app._SLEEPER_WEEK_ENSURE_TS[2026] = time.time() - app._SLEEPER_WEEK_ENSURE_COOLDOWN_S - 1
    app._ensure_sleeper_week_files(2026)
    assert len(ensure_state) == 4


def test_ensure_never_raises_and_cools_down_on_failure(monkeypatch):
    monkeypatch.setattr(app, "_SLEEPER_WEEK_ENSURE_TS", {})
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy", lambda s: _Policy([1])
    )

    def boom(season, week):
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
