"""Determinism tests for the parallelized game-log week/schedule parsing.

Covers the parallelization of the game-log request path:
- ``_sleeper_stats_by_week`` now warms the shared per-week stat index through
  a thread pool; it must return byte-identical results to the old sequential
  loop (zero-fill of display keys, present-but-all-zero -> zeros, nonzero
  overlay for custom scoring).
- ``_load_schedule_week_file`` must match the route's old inline schedule
  parse, including the first-file-wins week dedup over sorted files.
- Concurrent warmers must not corrupt the shared index (lock path).
"""
from __future__ import annotations

import concurrent.futures as futures
import glob
import json
import os
import re
from pathlib import Path

import pytest

# app.py imports pandas (and flask/openai) at module load. The fast "lint" CI
# shard has flask but not pandas, so guard here - otherwise importing app
# raises at COLLECTION and aborts the whole run.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

import app as appmod

W1 = {
    "101": {"pass_yd": 312, "pass_td": 3, "pass_att": 38, "pass_cmp": 38,
            "rush_yd": 9, "rec": 0, "fum_lost": 0},
    "102": {"pass_yd": 0, "pass_td": 0, "pass_att": 0, "pass_cmp": 0,
            "rush_yd": 0, "rec": 0, "fum_lost": 0},
    "103": "not-a-dict-row",
}
W2 = {
    "101": {"pass_yd": 275, "pass_td": 2, "pass_att": 33,
            "rush_yd": 21, "rush_td": 1, "fum_lost": 1},
}


@pytest.fixture()
def stat_files(tmp_path, monkeypatch):
    """Two week files for season 2024 under a fake CACHE_DIR."""
    cache = tmp_path / "cache"
    stats_dir = cache / "sleeper_stats"
    stats_dir.mkdir(parents=True)
    for week, payload in ((1, W1), (2, W2)):
        (stats_dir / f"sleeper_stats_s2024_w{week}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    monkeypatch.setattr(appmod, "CACHE_DIR", str(cache))
    monkeypatch.setattr(appmod, "_WEEK_STAT_INDEX", {})
    # Keep the test hermetic: no week-file backfill fetches.
    monkeypatch.setattr(appmod, "_ensure_sleeper_week_files", lambda season: None)
    return stats_dir


def _sequential_stats_by_week(player_id, season):
    """The pre-parallel loop body, for parity comparison."""
    pid = str(player_id)
    season = int(season)
    out: dict = {}
    pattern = os.path.join(
        appmod.CACHE_DIR, "sleeper_stats", f"sleeper_stats_s{season}_w*.json"
    )
    for path in glob.glob(pattern):
        match = re.match(r"sleeper_stats_s\d+_w(\d+)", os.path.basename(path))
        if not match:
            continue
        week = int(match.group(1))
        rows = appmod._week_stat_index_rows(season, week)
        if pid in rows:
            stored = rows[pid]
            if stored is None:
                out[week] = {k: 0 for k in appmod._GAMELOG_STAT_KEYS}
            else:
                merged = {k: 0 for k in appmod._GAMELOG_STAT_KEYS}
                merged.update(stored)
                out[week] = merged
    return out


def test_parallel_weeks_match_sequential(stat_files):
    """Pool-backed _sleeper_stats_by_week == old sequential loop, per player."""
    for pid in ("101", "102", "103", "999"):
        appmod._WEEK_STAT_INDEX.clear()
        parallel = appmod._sleeper_stats_by_week(pid, 2024)
        appmod._WEEK_STAT_INDEX.clear()
        sequential = _sequential_stats_by_week(pid, 2024)
        assert parallel == sequential
    # Spot-check the semantics, not just parity with itself:
    out = appmod._sleeper_stats_by_week("101", 2024)
    assert out[1]["pass_yd"] == 312
    assert out[1]["pass_cmp"] == 38  # nonzero overlay kept for custom scoring
    assert out[2]["rush_td"] == 1
    zeroed = appmod._sleeper_stats_by_week("102", 2024)
    assert set(zeroed) == {1}  # all-zero row -> 0.0 game, not DNP
    assert all(v == 0 for v in zeroed[1].values())


def test_concurrent_warm_is_consistent(stat_files):
    """Simultaneous cold warmers agree (exercises the index lock path)."""
    appmod._WEEK_STAT_INDEX.clear()
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda pid: appmod._sleeper_stats_by_week(pid, 2024),
            ["101", "102", "101", "102"],
        ))
    assert results[0] == results[2]
    assert results[1] == results[3]
    assert results[0][1]["pass_yd"] == 312


def test_repeat_calls_stable(stat_files):
    """Warm-index calls through the pool are stable."""
    first = appmod._sleeper_stats_by_week("101", 2024)
    second = appmod._sleeper_stats_by_week("101", 2024)
    assert first == second


@pytest.fixture()
def schedule_files(tmp_path):
    sched_dir = tmp_path / "sched"
    sched_dir.mkdir()
    games_a = [{"home": "TB", "away": "ATL", "gameDate": "2024-09-08"}]
    games_b = [{"home": "TB", "away": "CAR", "gameDate": "2024-09-08"}]
    (sched_dir / "schedule_s2024_w1.json").write_text(json.dumps(games_a))
    # Same week number via a different name: first (sorted) file must win.
    (sched_dir / "schedule_s2024_w01.json").write_text(json.dumps(games_b))
    (sched_dir / "schedule_s2024_w2.json").write_text(json.dumps({"n": 1}))
    (sched_dir / "schedule_s2024_w3.json").write_text("{not json")
    return sched_dir


def test_schedule_week_file_parse(schedule_files):
    d = str(schedule_files)
    parsed = appmod._load_schedule_week_file(
        os.path.join(d, "schedule_s2024_w1.json"))
    assert parsed is not None
    assert parsed[0] == 1
    assert isinstance(parsed[1], list)
    assert parsed[1][0]["home"] == "TB"
    # Non-list JSON, corrupt JSON, and missing files are all skipped.
    assert appmod._load_schedule_week_file(
        os.path.join(d, "schedule_s2024_w2.json")) is None
    assert appmod._load_schedule_week_file(
        os.path.join(d, "schedule_s2024_w3.json")) is None
    assert appmod._load_schedule_week_file(
        os.path.join(d, "schedule_s2024_w9.json")) is None


def test_schedule_dedup_first_sorted_file_wins(schedule_files):
    """The route's first-file-wins dedup, over sorted files, is deterministic."""
    files = sorted(glob.glob(str(schedule_files / "schedule_s2024_w*.json")))
    schedule_by_week: dict = {}
    for parsed in map(appmod._load_schedule_week_file, files):
        if parsed is None:
            continue
        wk, games = parsed
        if wk not in schedule_by_week:
            schedule_by_week[wk] = games
    assert set(schedule_by_week) == {1}
    # schedule_s2024_w01.json sorts before schedule_s2024_w1.json ("0" < "1").
    assert schedule_by_week[1][0]["away"] == "CAR"
