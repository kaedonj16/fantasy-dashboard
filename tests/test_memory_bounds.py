"""Regression tests for the 2GB memory-bound fixes.

Covers:
- _WEEK_STAT_INDEX season bounding: the per-(season, week) parsed stat index
  measured ~300MB per gunicorn worker for 10 seasons of history. It is now
  capped at the N most recent seasons (WEEK_STAT_INDEX_MAX_SEASONS, default 4);
  older seasons re-parse on demand. Stat keys / player ids are interned.
- LEAGUE_HISTORY_CACHE bound: previously unbounded; now evicts oldest-first
  past LEAGUE_HISTORY_CACHE_MAX entries.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

# app.py imports pandas at module load. The fast "lint" CI shard has flask but
# not pandas, so guard on pandas first - otherwise importing app here raises at
# COLLECTION and aborts the whole run.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

import app as appmod

ROOT = Path(__file__).resolve().parents[1]


def _write_week(tmp_path: Path, season: int, week: int, payload: dict) -> Path:
    d = tmp_path / "sleeper_stats"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"sleeper_stats_s{season}_w{week}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture()
def stat_index_env(tmp_path, monkeypatch):
    """Point app's CACHE_DIR at a tmp dir and start with an empty index."""
    monkeypatch.setattr(appmod, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(appmod, "_WEEK_STAT_INDEX", {})
    return tmp_path


def test_week_stat_index_evicts_old_seasons(stat_index_env):
    tmp_path = stat_index_env
    payload = {"7": {"rush_yd": 42, "rush_td": 0}}
    for season in range(2019, 2027):  # 8 seasons; cap is 4
        _write_week(tmp_path, season, 1, payload)
        appmod._week_stat_index_rows(season, 1)
    seasons = sorted({s for (s, _w) in appmod._WEEK_STAT_INDEX})
    assert seasons == [2023, 2024, 2025, 2026], seasons
    # 4 seasons x 1 week each
    assert len(appmod._WEEK_STAT_INDEX) == 4


def test_week_stat_index_old_season_still_correct(stat_index_env):
    """Evicted seasons re-parse from disk on demand: slower, still correct."""
    tmp_path = stat_index_env
    for season in range(2019, 2027):
        _write_week(tmp_path, season, 1, {"7": {"rush_yd": 10 + season}})
        appmod._week_stat_index_rows(season, 1)
    assert (2019, 1) not in appmod._WEEK_STAT_INDEX
    rows = appmod._week_stat_index_rows(2019, 1)
    assert rows["7"] == {"rush_yd": 2029}
    # Re-inserting 2019 makes it the oldest season again, so the cap evicts it
    # right back out; the returned rows were still correct. Cap still holds.
    seasons = sorted({s for (s, _w) in appmod._WEEK_STAT_INDEX})
    assert seasons == [2023, 2024, 2025, 2026]


def test_week_stat_index_semantics_preserved(stat_index_env):
    """Nonzero stats kept (incl. non-display keys), all-zero rows -> None."""
    tmp_path = stat_index_env
    _write_week(
        tmp_path, 2025, 1,
        {
            "7": {"rush_yd": 42, "rush_td": 0, "custom_bonus": 3},
            "9": {"rush_yd": 0, "rush_td": 0},
        },
    )
    rows = appmod._week_stat_index_rows(2025, 1)
    # Every nonzero stat is kept, even ones the display layer doesn't know.
    assert rows["7"] == {"rush_yd": 42, "custom_bonus": 3}
    # Present-but-all-zero row is a genuine 0.0 game, not DNP.
    assert rows["9"] is None
    # Consumer pattern from _sleeper_stats_by_week still works on interned rows.
    merged = {"rush_yd": 0, "rush_td": 0}
    merged.update(rows["7"])
    assert merged == {"rush_yd": 42, "rush_td": 0, "custom_bonus": 3}


def test_week_stat_index_mtime_reparse(stat_index_env):
    tmp_path = stat_index_env
    p = _write_week(tmp_path, 2025, 1, {"7": {"rush_yd": 42}})
    assert appmod._week_stat_index_rows(2025, 1)["7"] == {"rush_yd": 42}
    # Bump mtime and change content: the cached parse must be refreshed.
    time.sleep(0.02)
    p.write_text(json.dumps({"7": {"rush_yd": 99}}), encoding="utf-8")
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert appmod._week_stat_index_rows(2025, 1)["7"] == {"rush_yd": 99}


def test_week_stat_index_interns_keys(stat_index_env):
    """Stat keys and player ids are sys.intern()ed (shared, not duplicated)."""
    tmp_path = stat_index_env
    _write_week(tmp_path, 2025, 1, {"7": {"rush_yd": 42}})
    _write_week(tmp_path, 2025, 2, {"7": {"rush_yd": 43}})
    r1 = appmod._week_stat_index_rows(2025, 1)
    r2 = appmod._week_stat_index_rows(2025, 2)
    k1 = next(iter(r1["7"].keys()))
    k2 = next(iter(r2["7"].keys()))
    assert k1 == "rush_yd" and k1 is k2  # identical object: interned
    p1 = next(iter(r1.keys()))
    p2 = next(iter(r2.keys()))
    assert p1 == "7" and p1 is p2


def test_league_history_cache_bounded():
    api = pytest.importorskip("dashboard_services.api")
    api.LEAGUE_HISTORY_CACHE.clear()
    try:
        for i in range(api.LEAGUE_HISTORY_CACHE_MAX + 50):
            api.LEAGUE_HISTORY_CACHE[f"sleeper:lg{i}"] = {
                "ts": float(i),
                "map": {2025: f"lg{i}"},
            }
        api._prune_league_history_cache()
        assert len(api.LEAGUE_HISTORY_CACHE) <= api.LEAGUE_HISTORY_CACHE_MAX
        # Oldest-first eviction: the surviving entries are the newest.
        assert "sleeper:lg0" not in api.LEAGUE_HISTORY_CACHE
        assert f"sleeper:lg{api.LEAGUE_HISTORY_CACHE_MAX + 49}" in api.LEAGUE_HISTORY_CACHE
    finally:
        api.LEAGUE_HISTORY_CACHE.clear()
