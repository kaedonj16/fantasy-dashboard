"""Regression tests for the shared game-log stat index.

The game-log route used to glob + fully parse every weekly stat file for every
season on each request (~180 files / ~90MB of JSON) just to extract one
player's rows. app._week_stat_index_rows parses each (season, week) file once
per worker (mtime-guarded) and app._sleeper_stats_by_week reads from it.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import pytest

# app.py imports pandas at module load. The fast "lint" CI shard has flask but
# not pandas, so guard on pandas first - otherwise importing app here raises at
# COLLECTION and aborts the whole run. The index helpers themselves are pure;
# these tests run in the full-dependency shard.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

import app as appmod

ROOT = Path(__file__).resolve().parents[1]

STAT_KEYS = [
    "pass_yd", "pass_td", "pass_int", "pass_att",
    "rush_att", "rush_yd", "rush_td",
    "rec", "rec_tgt", "rec_yd", "rec_td", "fum_lost",
]

W1 = {
    "101": {"pass_yd": 312, "pass_td": 3, "pass_int": 1, "pass_att": 38,
            "pass_cmp": 38,
            "rush_att": 2, "rush_yd": 9, "rush_td": 0,
            "rec": 0, "rec_tgt": 0, "rec_yd": 0, "rec_td": 0, "fum_lost": 0,
            "extra_key_not_needed": "ignored"},
    "102": {"pass_yd": 0, "pass_td": 0, "pass_int": 0, "pass_att": 0,
            "rush_att": 0, "rush_yd": 0, "rush_td": 0,
            "rec": 0, "rec_tgt": 0, "rec_yd": 0, "rec_td": 0, "fum_lost": 0},
    "103": "not-a-dict-row",
}
W2 = {
    "101": {"pass_yd": 275, "pass_td": 2, "pass_int": 0, "pass_att": 33,
            "rush_att": 4, "rush_yd": 21, "rush_td": 1,
            "rec": 0, "rec_tgt": 0, "rec_yd": 0, "rec_td": 0, "fum_lost": 1},
}


@pytest.fixture()
def week_files(tmp_path, monkeypatch):
    """Two seasons x two week files under a fake CACHE_DIR."""
    cache = tmp_path / "cache"
    stats_dir = cache / "sleeper_stats"
    stats_dir.mkdir(parents=True)
    payloads = {
        (2024, 1): W1, (2024, 2): W2,
        (2023, 1): {"101": dict(W2["101"])}, (2023, 2): {},
    }
    for (season, week), payload in payloads.items():
        (stats_dir / f"sleeper_stats_s{season}_w{week}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    monkeypatch.setattr(appmod, "CACHE_DIR", str(cache))
    monkeypatch.setattr(appmod, "_WEEK_STAT_INDEX", {})
    return stats_dir


def _old_logic(player_id, season, cache_dir):
    """The pre-index implementation: full row dicts, one parse per call."""
    out = {}
    pattern = os.path.join(
        str(cache_dir), "sleeper_stats", f"sleeper_stats_s{int(season)}_w*.json"
    )
    for path in glob.glob(pattern):
        match = re.match(r"sleeper_stats_s\d+_w(\d+)", os.path.basename(path))
        if not match:
            continue
        with open(path) as handle:
            weekly = json.load(handle) or {}
        row = weekly.get(str(player_id)) or weekly.get(player_id)
        if isinstance(row, dict):
            out[int(match.group(1))] = dict(row)
    return out


def _norm(row):
    """Normalize for comparison: missing and zero are the same stat."""
    return {k: (v or 0) for k, v in row.items()}


def test_parity_with_direct_parse(week_files):
    """Indexed lookup returns the same stat values the old per-request parse did.

    The index stores nonzero stats only; the lookup zero-fills the display
    keys, so normalize zeros-vs-missing before comparing.
    """
    for season in (2023, 2024):
        for pid in ("101", "102", "103", "999"):
            new = appmod._sleeper_stats_by_week(pid, season)
            old = _old_logic(pid, season, week_files.parent)
            assert set(new) == set(old)
            for week in old:
                assert _norm(new[week]) == _norm(old[week])
                # Every key the old row had is still visible (custom scoring).
                assert set(old[week]) <= set(new[week]) | {
                    k for k, v in old[week].items() if not v
                }


def test_custom_scoring_points_parity(week_files):
    """score_stats sees identical values for custom-scoring leagues.

    The index must not truncate to the display keys: score_stats scores any
    stat key the league's settings name (pass_cmp, first downs, 2pt
    conversions, return yards...). A 12-key truncation silently undercounted
    these leagues (caught: 33.0 pts/game on a 1-pt-per-completion league).
    """
    from utils.fantasy_scoring import score_stats

    scoring = {
        "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
        "rush_yd": 0.1, "rush_td": 6.0, "rec": 1.0,
        "rec_yd": 0.1, "rec_td": 6.0, "fum_lost": -2.0,
        "pass_cmp": 1.0, "pass_2pt": 2.0, "rush_2pt": 2.0,
    }
    new = appmod._sleeper_stats_by_week("101", 2024)
    old = _old_logic("101", 2024, week_files.parent)
    assert set(new) == {1, 2}
    for week in (1, 2):
        assert score_stats(new[week], scoring) == score_stats(old[week], scoring)
    # And the truncation bug, directly: the stored row keeps pass_cmp.
    assert new[1]["pass_cmp"] == 38


def test_each_file_parsed_once_across_players(week_files, monkeypatch):
    """The second player's lookup performs zero JSON parses (the perf win)."""
    real_load = json.load
    calls = []

    def counting(handle, *a, **k):
        calls.append(1)
        return real_load(handle, *a, **k)

    monkeypatch.setattr(json, "load", counting)
    first = appmod._sleeper_stats_by_week("101", 2024)
    assert len(calls) == 2  # one parse per week file of the season
    calls.clear()
    second = appmod._sleeper_stats_by_week("102", 2024)
    assert calls == []
    assert set(first) == {1, 2}
    assert set(second) == {1}  # 102 only has a (zero) row in week 1


def test_zero_row_preserved_not_dropped(week_files):
    """A present-but-all-zero row renders as a 0.0 game, not DNP."""
    out = appmod._sleeper_stats_by_week("102", 2024)
    assert set(out) == {1}
    assert out[1] == {k: 0 for k in STAT_KEYS}
    assert all(v == 0 for v in out[1].values())


def test_mtime_change_reparses(week_files):
    """A refetched week file (new mtime) is re-parsed on next access."""
    before = appmod._sleeper_stats_by_week("101", 2024)
    assert before[1]["pass_yd"] == 312
    w1_path = week_files / "sleeper_stats_s2024_w1.json"
    updated = dict(W1)
    updated["101"] = dict(W1["101"], pass_yd=400)
    w1_path.write_text(json.dumps(updated), encoding="utf-8")
    # Force a newer mtime (some filesystems have coarse granularity).
    st = w1_path.stat()
    os.utime(w1_path, (st.st_atime, st.st_mtime + 5))
    after = appmod._sleeper_stats_by_week("101", 2024)
    assert after[1]["pass_yd"] == 400


def test_missing_and_corrupt_files(week_files):
    """Seasons with no files yield {}; a corrupt week file is skipped."""
    assert appmod._sleeper_stats_by_week("101", 1999) == {}
    (week_files / "sleeper_stats_s2024_w2.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    out = appmod._sleeper_stats_by_week("101", 2024)
    assert set(out) == {1}  # week 1 survives, corrupt week 2 skipped


def test_index_keys_match_endpoint_stats_dict():
    """The index must carry every key the game-log endpoint's _stats_dict
    picks; a future key added to the endpoint but not the index would silently
    vanish from game logs."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    m = re.search(
        r"def _stats_dict\(s\):\s+return \{k: s\.get\(k\) for k in\s+\[(.*?)\]\}",
        src,
        re.DOTALL,
    )
    assert m, "endpoint _stats_dict key list not found"
    endpoint_keys = set(re.findall(r'"([a-z_]+)"', m.group(1)))
    assert endpoint_keys == set(appmod._GAMELOG_STAT_KEYS)
