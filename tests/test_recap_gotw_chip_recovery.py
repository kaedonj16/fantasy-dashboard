"""Regression tests for the recap scoreboard GOTW chip disappearing.

The chip (PR #1867) resolves from the GOTW selection cached during the
previous week. That cache lived only in the AI file cache, which sits on the
web service's ephemeral disk: any deploy between the pick and the recap
erased it, and no writer can re-cache a completed week's pick. These tests
cover the two recovery layers: the durable Redis copy of the selection, and
the deterministic historical reconstruction used when both copies are gone.
"""
import inspect
import json

import pytest

# weekly_recap imports pandas at module load (full-stack shard convention).
pytest.importorskip("pandas")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import pandas as pd

from dashboard_services.ai import cache as ai_cache
from dashboard_services.ai import weekly_recap


def _selection(**overrides):
    value = {
        "platform": "sleeper", "league_id": "league-1", "season": "2025",
        "source_week": 1, "target_week": 2, "matchup_id": 7,
        "roster_ids": ["10", "20"],
    }
    value.update(overrides)
    return value


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


def _use_tmp_cache(tmp_path, monkeypatch):
    # Reads resolve AI_CACHE_DIR through weekly_recap's imported name, writes
    # through cache.py's own global; point both at the tmp dir.
    monkeypatch.setattr(weekly_recap, "AI_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ai_cache, "AI_CACHE_DIR", tmp_path)


def test_get_cached_gotw_selection_falls_back_to_redis_and_backfills_file(tmp_path, monkeypatch):
    _use_tmp_cache(tmp_path, monkeypatch)
    fake = _FakeRedis()
    monkeypatch.setattr(weekly_recap, "_gotw_redis_client", lambda: fake)
    key = weekly_recap._gotw_cache_key("sleeper", "league-1", 2025, 2)
    fake.store[weekly_recap._gotw_redis_key(key)] = json.dumps(_selection())

    got = weekly_recap.get_cached_gotw_selection("sleeper", "league-1", 2025, 2)
    assert got and got["roster_ids"] == ["10", "20"]

    # The Redis hit backfills the local file cache.
    obj = json.loads((tmp_path / f"{key}.json").read_text())
    assert obj["metadata"]["gotw_selection"]["matchup_id"] == 7


def test_get_cached_gotw_selection_rejects_mismatched_redis_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly_recap, "AI_CACHE_DIR", tmp_path)
    fake = _FakeRedis()
    monkeypatch.setattr(weekly_recap, "_gotw_redis_client", lambda: fake)
    # A selection stored under week 3's key but claiming target_week 2 (or a
    # different league) must not validate for week 3.
    key3 = weekly_recap._gotw_cache_key("sleeper", "league-1", 2025, 3)
    fake.store[weekly_recap._gotw_redis_key(key3)] = json.dumps(_selection())
    assert weekly_recap.get_cached_gotw_selection("sleeper", "league-1", 2025, 3) is None


def test_save_gotw_selection_writes_file_and_redis(tmp_path, monkeypatch):
    _use_tmp_cache(tmp_path, monkeypatch)
    fake = _FakeRedis()
    monkeypatch.setattr(weekly_recap, "_gotw_redis_client", lambda: fake)

    weekly_recap.save_gotw_selection("sleeper", "league-1", 2025, _selection())

    key = weekly_recap._gotw_cache_key("sleeper", "league-1", 2025, 2)
    obj = json.loads((tmp_path / f"{key}.json").read_text())
    assert obj["metadata"]["gotw_selection"]["roster_ids"] == ["10", "20"]
    stored = json.loads(fake.store[weekly_recap._gotw_redis_key(key)])
    assert stored["roster_ids"] == ["10", "20"]


def _df_weekly():
    # Weeks 1-2: teams 1 and 2 go 2-0 with the league's best scoring;
    # teams 3 and 4 go 0-2.
    rows = [
        # week, rid, pts, opp_pts, matchup_id
        (1, 1, 130.0, 90.0, 1), (1, 3, 90.0, 130.0, 1),
        (1, 2, 125.0, 85.0, 2), (1, 4, 85.0, 125.0, 2),
        (2, 1, 128.0, 88.0, 1), (2, 4, 88.0, 128.0, 1),
        (2, 2, 122.0, 95.0, 2), (2, 3, 95.0, 122.0, 2),
    ]
    return pd.DataFrame([
        {"roster_id": rid, "owner": f"Team {rid}", "week": wk, "points": pts,
         "points_against": opp, "finalized": True, "matchup_id": mid}
        for wk, rid, pts, opp, mid in rows
    ])


def _matchups_by_week():
    return {3: [
        {"matchup_id": 1, "left": {"roster_id": 1, "starters": []},
         "right": {"roster_id": 2, "starters": []}},
        {"matchup_id": 2, "left": {"roster_id": 3, "starters": []},
         "right": {"roster_id": 4, "starters": []}},
    ]}


_TEAM_BY_RID = {str(i): f"Team {i}" for i in range(1, 5)}
_LEAGUE = {"settings": {"playoff_teams": 4, "playoff_week_start": 14}}


def test_compute_historical_gotw_selection_picks_undefeated_clash():
    sel = weekly_recap.compute_historical_gotw_selection(
        _df_weekly(), _matchups_by_week(), 3, _TEAM_BY_RID, _LEAGUE)
    assert sel is not None
    assert set(sel["roster_ids"]) == {"1", "2"}
    assert sel["target_week"] == 3 and sel["source_week"] == 2
    assert sel["recovered"] is True


def test_compute_historical_gotw_selection_week_one_has_no_pick():
    assert weekly_recap.compute_historical_gotw_selection(
        _df_weekly(), _matchups_by_week(), 1, _TEAM_BY_RID, _LEAGUE) is None


def test_compute_historical_gotw_selection_no_matchups_no_pick():
    assert weekly_recap.compute_historical_gotw_selection(
        _df_weekly(), {}, 3, _TEAM_BY_RID, _LEAGUE) is None


def test_recap_page_recovers_and_persists_missing_gotw_selection():
    from dashboard_services.pages import recap_page
    src = inspect.getsource(recap_page)
    assert "compute_historical_gotw_selection" in src
    assert "save_gotw_selection" in src


def test_all_gotw_writers_use_durable_save():
    from dashboard_services.pages import history_page, weekly_hub_page
    for module in (history_page, weekly_hub_page):
        src = inspect.getsource(module)
        assert "save_gotw_selection" in src
        assert 'save_cached_ai_text(\n                    _gotw_cache_key' not in src
