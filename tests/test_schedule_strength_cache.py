"""Schedule-strength: ctx-first rosters + bounded response cache.

One handler run used to re-fetch rosters/users from the provider AND
loop provider matchups for every regular-season week on every open,
even though the page's resolved ctx already carried rosters/users/
roster_map. The matchup loop stays (ctx matchups are preview-shaped),
but repeat opens within the TTL serve the cached response.
"""
from __future__ import annotations

import pytest
from flask import Flask

pytest.importorskip("pandas")

import routes.schedule_api_bp as bp
import dashboard_services.platform_api as platform_api


ROSTERS = [
    {"roster_id": 1, "owner_id": "u1", "players": ["p1"], "metadata": {"team_name": "One"}},
    {"roster_id": 2, "owner_id": "u2", "players": ["p2"], "metadata": {"team_name": "Two"}},
]
USERS = [
    {"user_id": "u1", "display_name": "One"},
    {"user_id": "u2", "display_name": "Two"},
]


def _matchups(week):
    return [
        {"matchup_id": 1, "roster_id": 1, "points": 100.0 + week},
        {"matchup_id": 1, "roster_id": 2, "points": 90.0 + week},
    ]


@pytest.fixture()
def harness(monkeypatch):
    calls = {"matchups": 0, "rosters": 0, "users": 0}
    bp._SCHED_STRENGTH_CACHE.clear()

    def fake_matchups(platform, league_id, week, season):
        calls["matchups"] += 1
        return _matchups(week)

    def fake_rosters(*a, **k):
        calls["rosters"] += 1
        return [dict(r) for r in ROSTERS]

    def fake_users(*a, **k):
        calls["users"] += 1
        return [dict(u) for u in USERS]

    monkeypatch.setattr(platform_api, "get_matchups", fake_matchups)
    monkeypatch.setattr(bp, "get_rosters", fake_rosters)
    monkeypatch.setattr(bp, "get_users", fake_users)
    monkeypatch.setattr(bp, "get_nfl_state",
                        lambda *a, **k: {"season": "2026", "leg": 3, "week": 3})
    monkeypatch.setattr(bp, "_projected_starter_avgs", lambda *a, **k: {})
    monkeypatch.setattr(bp, "_prune_ttl_cache", lambda cache, max_entries: None)
    monkeypatch.setattr(bp, "_roster_sig", lambda ctx: "sig-fixed")
    app = Flask(__name__)
    app.register_blueprint(bp.schedule_api_bp)
    return app.test_client(), calls, monkeypatch


def test_ctx_first_and_response_cached(harness):
    client, calls, monkeypatch = harness
    ctx = {
        "rosters": [dict(r) for r in ROSTERS],
        "users": [dict(u) for u in USERS],
        "roster_map": {"1": "One", "2": "Two"},
        "league_settings": {"playoff_week_start": 15},
    }
    monkeypatch.setattr(bp, "get_league_ctx_from_cache", lambda *a, **k: ctx)

    url = "/api/schedule-strength?platform=sleeper&league_id=SS1&season=2026"
    r1 = client.get(url)
    assert r1.status_code == 200
    body1 = r1.get_json()
    assert {t["roster_id"] for t in body1["teams"]} == {"1", "2"}
    # reg_weeks = playoff_week_start - 1 = 14 matchup fetches; provider
    # rosters/users never fetched because ctx carried them.
    assert calls == {"matchups": 14, "rosters": 0, "users": 0}

    r2 = client.get(url)
    assert r2.status_code == 200
    assert r2.get_json() == body1
    assert calls == {"matchups": 14, "rosters": 0, "users": 0}  # fully cached


def test_provider_fallback_when_ctx_empty(harness):
    client, calls, monkeypatch = harness
    monkeypatch.setattr(bp, "get_league_ctx_from_cache", lambda *a, **k: {})

    url = "/api/schedule-strength?platform=sleeper&league_id=SS2&season=2026"
    r1 = client.get(url)
    assert r1.status_code == 200
    assert calls["rosters"] == 1 and calls["users"] == 1
    assert calls["matchups"] == 14
    r2 = client.get(url)
    assert r2.get_json() == r1.get_json()
    assert calls["matchups"] == 14  # second open served from the cache
