"""Failure-path tests for build_espn_team_game_lookup.

ESPN scoreboard failures must be loud: a warning-level log with season/week
context plus a degraded status ("stale" when serving cached data after a
failed refresh, "failed" when there is no data at all) -- never a silent
empty dict like the old ESPN-403 → empty-scores outage.
"""
import logging
import time
import types

import pytest
import requests

import utils.redzone_alt_pbp as alt


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    monkeypatch.setattr(alt, "_ESPN_SB_CACHE", {})


def _resp(status_code=200, payload=None):
    return types.SimpleNamespace(status_code=status_code, json=lambda: payload or {})


def _scoreboard_payload():
    return {
        "events": [
            {
                "id": "401700000",
                "date": "2026-09-11T00:20Z",
                "status": {
                    "displayClock": "12:34",
                    "period": 2,
                    "type": {"state": "in", "completed": False, "shortDetail": "12:34 - 2nd"},
                },
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "SEA"}, "score": "10"},
                            {"homeAway": "away", "team": {"abbreviation": "NE"}, "score": "7"},
                        ]
                    }
                ],
            },
        ]
    }


def _warned_with(caplog, *needles):
    return any(
        r.levelno == logging.WARNING and all(n in r.getMessage() for n in needles)
        for r in caplog.records
    )


def test_non_200_no_cache_returns_failed_and_warns(monkeypatch, caplog):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _resp(403))

    with caplog.at_level(logging.WARNING, logger="utils.redzone_alt_pbp"):
        lookup, status = alt.build_espn_team_game_lookup(2026, 3)

    assert lookup == {}
    assert status == "failed"
    assert _warned_with(caplog, "403", "2026")


def test_non_200_with_expired_cache_serves_stale_and_warns(monkeypatch, caplog):
    stale = {"NE": {"gameID": "20260911_NE@SEA"}}
    alt._ESPN_SB_CACHE["2026:3:2"] = (time.time() - 3600, stale)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _resp(403))

    with caplog.at_level(logging.WARNING, logger="utils.redzone_alt_pbp"):
        lookup, status = alt.build_espn_team_game_lookup(2026, 3)

    assert lookup == stale
    assert status == "stale"
    assert _warned_with(caplog, "403", "2026")


def test_exception_returns_failed_and_warns(monkeypatch, caplog):
    def _boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "get", _boom)

    with caplog.at_level(logging.WARNING, logger="utils.redzone_alt_pbp"):
        lookup, status = alt.build_espn_team_game_lookup(2026, 3)

    assert lookup == {}
    assert status == "failed"
    assert _warned_with(caplog, "ConnectionError", "2026")


def test_200_with_games_returns_fresh_and_caches(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _resp(200, _scoreboard_payload()))

    lookup, status = alt.build_espn_team_game_lookup(2026, 3)

    assert status == "fresh"
    assert lookup["SEA"]["gameID"] == "20260911_NE@SEA"
    assert lookup["NE"]["homePts"] == "10"
    # cached for the next poll
    assert alt._ESPN_SB_CACHE["2026:3:2"][1] is lookup


def test_fresh_cache_hit_returns_fresh_without_fetch(monkeypatch):
    cached = {"NE": {"gameID": "cached"}}
    alt._ESPN_SB_CACHE["2026:3:2"] = (time.time(), cached)

    def _must_not_fetch(*a, **k):
        raise AssertionError("must not fetch on a fresh cache hit")
    monkeypatch.setattr(requests, "get", _must_not_fetch)

    lookup, status = alt.build_espn_team_game_lookup(2026, 3)

    assert lookup == cached
    assert status == "fresh"
