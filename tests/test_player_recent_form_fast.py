"""Tests for the lightweight Recent Form endpoint.

The player modal's Recent Form graph used to fetch /api/player-game-logs,
the heaviest modal endpoint: eligibility + league sync before its cache
check, and on a miss every season's schedule files, 18 weekly projection
files, matchup ratings, and a per-year advanced-metrics join, when the
graph only needs the current season's real games. /api/player-recent-form
serves that one season with the game-log route's exact row semantics,
behind its own short-TTL cache.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

# app.py imports pandas (and flask/openai) at module load. The fast "lint"
# CI shard has flask but not pandas, so guard here - otherwise importing
# app raises at COLLECTION and aborts the whole run.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

import app as appmod

PID = "4046"
SEASON = 2026

W1_STATS = {
    "pass_yd": 300, "pass_td": 3, "pass_int": 1, "pass_att": 35,
    "rush_att": 3, "rush_yd": 20, "rush_td": 1,
    "rec": 0, "rec_tgt": 0, "rec_yd": 0, "rec_td": 0, "fum_lost": 0,
}
W2_STATS = {
    "pass_yd": 100, "pass_td": 1, "pass_int": 0, "pass_att": 20,
    "rush_att": 1, "rush_yd": 5, "rush_td": 0,
    "rec": 0, "rec_tgt": 0, "rec_yd": 0, "rec_td": 0, "fum_lost": 0,
}
# Default scoring (no league): W1 = 300*.04 + 3*4 - 2 + 20*.1 + 6 = 30.0
#                              W2 = 100*.04 + 1*4 + 5*.1 = 8.5

SCHEDULES = {
    1: [{"home": "BUF", "away": "KC", "gameDate": "20260913"}],
    2: [{"home": "MIA", "away": "BUF", "gameDate": "20260920"}],
    3: [{"home": "NE", "away": "NYJ", "gameDate": "20260927"}],
    4: [{"home": "BUF", "away": "NE", "gameDate": "20261004"}],
}


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Stands in for the weekly-metrics DB connection (dict rows)."""

    def execute(self, sql, params=None):
        if "player_weekly_advanced_metrics" in sql:
            return _FakeResult([{
                "week": 1, "avg_depth_of_target": 8.5, "ybc_per_carry": 3.0,
                "xfp": 20.5, "broken_tackles": 1.0, "wopr": 0.1,
            }])
        return _FakeResult([{
            "week": 1, "snap_pct": 92.0, "target_share": 0.0,
            "carry_share": 10.0, "rz_targets": 0, "rz_carries": 2,
        }])


class _FakeConnCtx:
    def __enter__(self):
        return _FakeConn()

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def rf_env(tmp_path, monkeypatch):
    """Isolated recent-form route: stubbed stats, schedule, team, DB."""
    appmod.app.config.update(TESTING=True)
    monkeypatch.setattr(appmod, "_RECENT_FORM_CACHE", {})

    env = SimpleNamespace(stats_calls=[], stats={1: dict(W1_STATS), 2: dict(W2_STATS)})

    def fake_stats(player_id, season):
        env.stats_calls.append((str(player_id), int(season)))
        return {wk: dict(s) for wk, s in env.stats.items()}

    monkeypatch.setattr(appmod, "_sleeper_stats_by_week", fake_stats)

    sched_dir = tmp_path / "schedule"
    sched_dir.mkdir()
    paths = []
    for week, games in SCHEDULES.items():
        p = sched_dir / f"schedule_s{SEASON}_w{week}.json"
        p.write_text(json.dumps(games), encoding="utf-8")
        paths.append(str(p))
    monkeypatch.setattr(
        appmod, "glob",
        SimpleNamespace(glob=lambda pattern: list(paths)),
    )

    players = {PID: {"team": "BUF", "pos": "QB", "name": "Josh Allen"}}
    monkeypatch.setattr("utils.utils.load_relevant_index", lambda: dict(players))
    monkeypatch.setattr("utils.data_cache.load_relevant_index", lambda: dict(players))
    monkeypatch.setattr(appmod, "load_players_index", lambda: {})
    monkeypatch.setattr(
        "data_building.external_data.player_team_history.team_for_week",
        lambda pid, season, week: "BUF",
    )
    monkeypatch.setattr(
        "data_building.weekly_metrics.init_weekly_metrics_db", lambda: None)
    monkeypatch.setattr(
        "data_building.advanced_metrics.init_weekly_advanced_metrics_db",
        lambda: None)
    monkeypatch.setattr(
        "data_building.advanced_metrics.get_conn", lambda: _FakeConnCtx())

    env.client = appmod.app.test_client()
    return env


def test_recent_form_route_maps_to_api_player_recent_form():
    """Exactly one /api/player-recent-form/<player_id> rule, serving the
    new view (the game-logs route regression shape: a helper inserted
    between decorator and view would silently re-point the route)."""
    rules = [
        r for r in appmod.app.url_map.iter_rules()
        if str(r) == "/api/player-recent-form/<player_id>"
    ]
    assert len(rules) == 1, "expected exactly one recent-form route"
    view = appmod.app.view_functions[rules[0].endpoint]
    assert view.__name__ == "api_player_recent_form", (
        f"route serves {view.__name__}, expected api_player_recent_form"
    )


def test_recent_form_returns_only_requested_season(rf_env):
    resp = rf_env.client.get(f"/api/player-recent-form/{PID}?season={SEASON}")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert set(payload["game_logs_by_year"]) == {str(SEASON)}
    assert payload["season_teams"] == {str(SEASON): "BUF"}

    logs = payload["game_logs_by_year"][str(SEASON)]
    assert [g["week"] for g in logs] == [1, 2, 3, 4]
    by_week = {g["week"]: g for g in logs}

    w1 = by_week[1]
    assert w1["fantasy_pts"] == 30.0
    assert w1["opponent"] == "KC"  # home: bare code
    assert w1["date"] == "20260913"
    assert w1["stats"]["pass_yd"] == 300
    # Advanced join rides along: the Overview Role card reads these.
    assert w1["adv"]["snap_pct"] == 92.0
    assert w1["adv"]["xfp"] == 20.5
    assert w1["adv"]["rz_carries"] == 2

    w2 = by_week[2]
    assert w2["fantasy_pts"] == 8.5
    assert w2["opponent"] == "@MIA"  # away: @ prefix

    w3 = by_week[3]  # BUF has no game and no stats: a bye row
    assert w3["is_bye"] is True
    assert w3["opponent"] == "BYE"
    assert w3["fantasy_pts"] is None and w3["stats"] is None
    assert w3["date"] == "20260927"  # date of any game that week, for sorting

    w4 = by_week[4]  # BUF plays but the player has no stats: a DNP row
    assert "is_bye" not in w4
    assert w4["opponent"] == "NE"
    assert w4["fantasy_pts"] is None and w4["stats"] is None


def test_recent_form_second_request_served_from_cache(rf_env):
    url = f"/api/player-recent-form/{PID}?season={SEASON}"
    first = rf_env.client.get(url).get_json()
    assert len(rf_env.stats_calls) == 1
    second = rf_env.client.get(url).get_json()
    assert second == first
    assert len(rf_env.stats_calls) == 1, (
        "repeat request must hit _RECENT_FORM_CACHE, not recompute stats"
    )
    assert len(appmod._RECENT_FORM_CACHE) == 1


def test_recent_form_no_stats_returns_empty(rf_env):
    rf_env.stats = {}
    resp = rf_env.client.get(f"/api/player-recent-form/{PID}?season={SEASON}")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["game_logs_by_year"] == {}
    assert payload["season_teams"] == {}


def test_recent_form_route_skips_heavy_game_log_work():
    """The light route must not reference the full route's expensive steps."""
    src = inspect.getsource(appmod.api_player_recent_form)
    assert "_prefetch_week_projections" not in src
    assert "_player_nfl_eligibility" not in src
    assert "_load_matchup_ratings" not in src
    assert "_matchup_rank_table" not in src
