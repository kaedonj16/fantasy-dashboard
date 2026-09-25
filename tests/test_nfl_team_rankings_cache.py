"""NFL team rankings: the cold-cache path must not stall the page.

Root cause of the first-click blank page: ``_compute_team_offense_ranks``
does a 60s+ cold compute (Sleeper week files + nflverse CSV downloads) and
the in-memory cache is per gunicorn worker, so every worker paid it on its
first hit. Near gunicorn's 120s worker timeout the request could even be
killed, so the page never populated until a refresh. The payload is now also
cached on the container-shared disk (one worker's compute serves the rest),
the post-deploy script warms it, and missing Sleeper weeks fetch in
parallel instead of sequentially.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

SEASON = 2026


def _payload():
    return {
        "stats_season": SEASON,
        "season": SEASON,
        "data_mode": "actual",
        "completed_weeks": [1, 2, 3],
        "teams_index": {"KC": {"city": "Kansas City", "name": "Chiefs"}},
        "ranks": {"points": {"KC": {"rank": 1, "total": 2, "value": 30.0}}},
        "team_games": {"KC": 3},
        "available_seasons": [SEASON],
    }


@pytest.fixture()
def cold_env(monkeypatch, tmp_path):
    """app with an empty memory cache and CACHE_DIR pointed at tmp."""
    import app

    monkeypatch.setattr(app, "CACHE_DIR", str(tmp_path))
    app._TEAM_OFFENSE_RANKS_CACHE.pop((SEASON,), None)
    return app


def test_disk_cache_round_trip(cold_env, tmp_path):
    payload = _payload()
    cold_env._write_team_offense_ranks_disk(SEASON, payload)
    assert (tmp_path / f"team_offense_ranks_{SEASON}.json").exists()
    hit = cold_env._read_team_offense_ranks_disk(SEASON)
    assert hit is not None
    computed_at, back = hit
    assert back == payload
    assert time.time() - computed_at < cold_env._TEAM_OFFENSE_RANKS_TTL


def test_stale_disk_cache_ignored(cold_env, tmp_path):
    stale = {
        "computed_at": time.time() - cold_env._TEAM_OFFENSE_RANKS_TTL - 1,
        "payload": _payload(),
    }
    (tmp_path / f"team_offense_ranks_{SEASON}.json").write_text(json.dumps(stale))
    assert cold_env._read_team_offense_ranks_disk(SEASON) is None


def test_corrupt_disk_cache_ignored(cold_env, tmp_path):
    (tmp_path / f"team_offense_ranks_{SEASON}.json").write_text("not json{{{")
    assert cold_env._read_team_offense_ranks_disk(SEASON) is None
    # A corrupt file must not break the compute path either.
    assert cold_env._read_team_offense_ranks_disk(1999) is None


def _stub_compute_inputs(monkeypatch, app, forbid_compute=False):
    from utils import team_offense_ranks
    from utils import utils as utils_mod

    def _boom(*a, **k):
        raise AssertionError("heavy compute must not run on a warm disk cache")

    if forbid_compute:
        monkeypatch.setattr(team_offense_ranks, "compute_team_offense", _boom)
        monkeypatch.setattr(team_offense_ranks, "rank_offense_table", _boom)
    else:
        monkeypatch.setattr(
            team_offense_ranks, "compute_team_offense",
            lambda *a, **k: {"data_mode": "actual", "completed_weeks": [1],
                             "teams": {"KC": {"games": 1}}},
        )
        monkeypatch.setattr(
            team_offense_ranks, "rank_offense_table",
            lambda table: {"points_pg": {"KC": {"rank": 1, "total": 1,
                                                "value": 30.0}}},
        )
    monkeypatch.setattr(app, "_nflverse_team_games_rows", lambda: [])
    monkeypatch.setattr(app, "_sleeper_team_week_rows", lambda s, w: {})
    monkeypatch.setattr(app, "_aggregate_projected_team_offense", lambda s: {})
    monkeypatch.setattr(app, "_team_plays_pg_map", lambda s: {})
    monkeypatch.setattr(app, "_has_team_offense_projections", lambda s: False)
    monkeypatch.setattr(utils_mod, "load_teams_index", lambda: {})


def test_compute_prefers_disk_over_recompute(cold_env, monkeypatch):
    app = cold_env
    _stub_compute_inputs(monkeypatch, app, forbid_compute=True)
    payload = _payload()
    app._write_team_offense_ranks_disk(SEASON, payload)
    # Memory cache is empty (cold worker); the disk hit must win and the
    # heavy compute must never run.
    assert app._compute_team_offense_ranks(SEASON) == payload


def test_compute_writes_disk_on_miss(cold_env, monkeypatch, tmp_path):
    app = cold_env
    _stub_compute_inputs(monkeypatch, app)
    out = app._compute_team_offense_ranks(SEASON)
    assert out["season"] == SEASON
    assert out["data_mode"] == "actual"
    disk = app._read_team_offense_ranks_disk(SEASON)
    assert disk is not None
    assert disk[1] == out


def test_post_deploy_warms_team_rankings():
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "post_deploy.py").read_text(encoding="utf-8")
    assert "_warm_team_rankings_cache" in src
    assert "/api/nfl-team-rankings" in src
