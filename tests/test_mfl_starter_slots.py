"""MFL starters must be published in lineup-slot order with starters_slots.

MFL flags starters in weeklyResults but returns them in its own player
order (and the roster derivation fallback returns roster order), while
index-based consumers pair starters with roster_positions by list
position -- the Yahoo raw-order failure shape. These tests pin the
adapter seating starters into slot order with parallel starters_slots,
points realigned, and a fail-soft path when seating is impossible.
"""
import sys
import types

import pytest

from dashboard_services.providers.mfl_api import MFLProvider, _CACHE

_INDEX = {
    "qb1": {"position": "QB"},
    "rb1": {"position": "RB"},
    "rb2": {"position": "RB"},
    "wr1": {"position": "WR"},
    "wr2": {"position": "WR"},
    "te1": {"position": "TE"},
    "k1": {"position": "K"},
    "def1": {"position": "DEF"},
}

_XWALK = {
    "101": "k1", "102": "wr1", "103": "def1", "104": "qb1",
    "105": "te1", "106": "rb1", "107": "wr2", "108": "rb2",
}

_LEAGUE = {"league": {"id": "123", "name": "Dynasty", "size": "2",
                      "lastRegularSeasonWeek": "14",
                      "starters": "QB,RB,WR,TE,FLEX,PK,Def",
                      "franchises": {"franchise": [{"id": "0001", "name": "Owls"}]}}}

_SLOT_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF"]


@pytest.fixture(autouse=True)
def clear_cache():
    _CACHE.clear()


@pytest.fixture
def players_index(monkeypatch):
    stub = types.ModuleType("utils.utils")
    stub.load_players_index = lambda: dict(_INDEX)
    monkeypatch.setitem(sys.modules, "utils.utils", stub)


def _provider(monkeypatch, payloads):
    provider = MFLProvider()
    monkeypatch.setattr(provider, "_export", lambda kind, *a, **k: payloads[kind])
    monkeypatch.setattr(provider, "_canonical_map", lambda *a, **k: dict(_XWALK))
    return provider


def _scrambled_block():
    # MFL order: K, WR, DEF, QB, TE, RB, flex WR, plus a bench RB.
    rows = [
        ("101", "starter", "8"), ("102", "starter", "11"), ("103", "starter", "6"),
        ("104", "starter", "20"), ("105", "starter", "9"), ("106", "starter", "15"),
        ("107", "starter", "12"), ("108", "nonstarter", "3"),
    ]
    return [{"id": pid, "status": status, "score": score} for pid, status, score in rows]


def test_get_matchups_seats_scrambled_starters_into_slot_order(monkeypatch, players_index):
    payloads = {
        "league": _LEAGUE,
        "rules": {},
        "weeklyResults": {"weeklyResults": {"matchup": [{"franchise": [
            {"id": "0001", "score": "87", "player": _scrambled_block()},
            {"id": "0002", "score": "50"},
        ]}]}},
    }
    provider = _provider(monkeypatch, payloads)
    row = next(r for r in provider.get_matchups("123", 2026, 1) if r["roster_id"] == 1)
    assert row["starters"] == ["qb1", "rb1", "wr1", "te1", "wr2", "k1", "def1"]
    assert row["starters_slots"] == _SLOT_ORDER
    assert row["starters_points"] == [20.0, 15.0, 11.0, 9.0, 12.0, 8.0, 6.0]


def test_get_rosters_seats_weekly_results_starters(monkeypatch, players_index):
    payloads = {
        "league": _LEAGUE,
        "rules": {},
        "rosters": {"rosters": {"franchise": [
            {"id": "0001", "player": [{"id": pid} for pid in _XWALK]},
        ]}},
        "weeklyResults": {"weeklyResults": {"matchup": [{"franchise": [
            {"id": "0001", "score": "87", "player": _scrambled_block()},
        ]}]}},
    }
    provider = _provider(monkeypatch, payloads)
    roster = provider.get_rosters("123", 2026)[0]
    assert roster["starters"] == ["qb1", "rb1", "wr1", "te1", "wr2", "k1", "def1"]
    assert roster["starters_slots"] == _SLOT_ORDER


def test_get_rosters_derived_starters_are_seated(monkeypatch, players_index):
    payloads = {
        "league": _LEAGUE,
        "rules": {},
        "rosters": {"rosters": {"franchise": [
            {"id": "0001", "player": [{"id": pid} for pid in _XWALK]},
        ]}},
        "weeklyResults": {"weeklyResults": {}},
    }
    provider = _provider(monkeypatch, payloads)
    roster = provider.get_rosters("123", 2026)[0]
    starters = roster["starters"]
    assert roster["starters_slots"] == _SLOT_ORDER
    assert len(starters) == 7
    assert starters[0] == "qb1"
    assert starters[1] == "rb1"
    assert starters[3] == "te1"
    assert starters[5] == "k1"
    assert starters[6] == "def1"
    assert {starters[2], starters[4]} == {"wr1", "wr2"}


def test_get_matchups_fail_soft_without_league_slots(monkeypatch, players_index):
    payloads = {
        "weeklyResults": {"weeklyResults": {"matchup": [{"franchise": [
            {"id": "0001", "score": "87", "player": _scrambled_block()},
            {"id": "0002", "score": "50"},
        ]}]}},
    }
    provider = _provider(monkeypatch, payloads)

    def _no_league(*a, **k):
        raise RuntimeError("league export unavailable")

    monkeypatch.setattr(provider, "get_league", _no_league)
    row = next(r for r in provider.get_matchups("123", 2026, 1) if r["roster_id"] == 1)
    # Raw MFL order preserved; no claimed slots.
    assert row["starters"] == ["k1", "wr1", "def1", "qb1", "te1", "rb1", "wr2"]
    assert row["starters_slots"] == []
    assert row["starters_points"] == [8.0, 11.0, 6.0, 20.0, 9.0, 15.0, 12.0]
