"""Beat-the-market: bounded response cache over the value-history deltas.

The endpoint used to re-run the full player_value_history delta query
(plus provider rosters/users) on every Teams-page open. Repeat opens
with the same window/dates/rosters now serve the cached payload.
"""
from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


class _Result:
    def __init__(self, one=None, all_rows=None):
        self._one = one
        self._all = all_rows or []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._all


def test_beat_the_market_second_open_is_cached(offline_client, monkeypatch):
    import app as appmod
    import data_building.player_value_history as pvh
    import dashboard_services.db as dbmod

    counts = {"delta": 0, "rosters": 0}

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=()):
            if "MAX(as_of_date)" in sql:
                return _Result(one={"d": date(2026, 9, 20)})
            counts["delta"] += 1
            return _Result(all_rows=[
                {"player_id": "p1", "delta": 10.0},
                {"player_id": "p2", "delta": -6.0},
                {"player_id": "p3", "delta": 0.0},
            ])

    monkeypatch.setattr(pvh, "init_value_history_db", lambda *a, **k: None)
    monkeypatch.setattr(pvh, "get_latest_snapshot_date", lambda **k: date(2026, 10, 20))
    monkeypatch.setattr(dbmod, "get_conn", lambda *a, **k: FakeConn())
    monkeypatch.setattr(appmod, "get_nfl_state", lambda *a, **k: {"season": "2026"})

    def fake_rosters(platform, league_id, season):
        counts["rosters"] += 1
        return [
            {"roster_id": 1, "owner_id": "u1", "players": ["p1", "p2"]},
            {"roster_id": 2, "owner_id": "u2", "players": ["p3"]},
        ]

    monkeypatch.setattr(appmod, "get_rosters", fake_rosters)
    monkeypatch.setattr(appmod, "get_users", lambda *a, **k: [
        {"user_id": "u1", "display_name": "One", "metadata": {}},
        {"user_id": "u2", "display_name": "Two", "metadata": {}},
    ])
    monkeypatch.setattr(appmod, "load_players_index", lambda *a, **k: {
        "p1": {"pos": "QB", "name": "One QB"},
        "p2": {"pos": "RB", "name": "Two RB"},
        "p3": {"pos": "WR", "name": "Three WR"},
    })
    appmod._BEAT_MARKET_CACHE.clear()

    url = "/api/beat-the-market?platform=sleeper&league_id=BM1&season=2026&days=30"
    r1 = offline_client.get(url)
    assert r1.status_code == 200
    body1 = r1.get_json()
    assert body1["latest_date"] == "2026-10-20"
    assert body1["baseline_date"] == "2026-09-20"
    assert body1["league_avg_delta"] == 2.0
    assert counts == {"delta": 1, "rosters": 1}

    r2 = offline_client.get(url)
    assert r2.status_code == 200
    assert r2.get_json() == body1
    assert counts["delta"] == 1  # delta query skipped on the warm open
