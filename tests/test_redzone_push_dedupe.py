"""Server-side RedZone scoring push: owner targeting + canonical dedupe.

The device push for a live touchdown must:
  * target only the affected fantasy owner (never a league-wide broadcast),
  * send exactly once per canonical play+owner even when many clients/workers
    poll the same live game (atomic app_state claim), and
  * never push an unrostered player's touchdown to anyone.
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

import utils.push_notifications as pn


def _fake_db_module():
    """A fake dashboard_services.db whose get_conn emulates the atomic claim
    (INSERT ... ON CONFLICT (key) DO NOTHING RETURNING key) with an in-memory
    set, so _app_state_claim is exercised for real."""
    claimed: set = set()

    class R:
        def __init__(self, row): self._row = row
        def fetchone(self): return self._row

    class FakeConn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, q, params=None):
            if "ON CONFLICT (key) DO NOTHING" in q:
                key = params[0]
                if key in claimed:
                    return R(None)
                claimed.add(key)
                return R({"key": key})
            return R(None)
        def commit(self): pass

    mod = types.ModuleType("dashboard_services.db")
    mod.get_conn = lambda: FakeConn()
    return mod, claimed


def _run_notify(pbp_by_game, rosters, player_info, scoring, sends):
    fake_db, claimed = _fake_db_module()
    pkg = sys.modules.get("dashboard_services") or types.ModuleType("dashboard_services")
    with mock.patch.dict(sys.modules, {"dashboard_services": pkg,
                                       "dashboard_services.db": fake_db}), \
         mock.patch.object(
             pn, "_broadcast_owner",
             side_effect=lambda league_id, owner_id, title, body, url="/", tag="update", notif_type=None:
             sends.append({"league": league_id, "owner": owner_id, "title": title,
                           "notif_type": notif_type}) or 1):
        return pn.notify_redzone_scores(
            "L1", "sleeper", pbp_by_game, player_info, rosters, scoring,
            season=2026, week=1,
        )


ROSTERS = [
    {"roster_id": 1, "owner_id": "O1", "starters": ["rb"], "players": ["rb"]},
    {"roster_id": 2, "owner_id": "O2", "starters": ["wr", "qb"], "players": ["wr", "qb"]},
]
PLAYER_INFO = {
    "rb": {"name": "Aaron Jones", "pos": "RB"},
    "wr": {"name": "Justin Jefferson", "pos": "WR"},
    "qb": {"name": "Carson Wentz", "pos": "QB"},
}
SCORING = {"rush_td": 6.0, "rec_td": 6.0, "pass_td": 4.0, "rush_yd": 0.1}


def _td_row(pid, play_id, **line):
    return {"pid": pid, "play_id": play_id, "is_td": True, "play_state": "VALID",
            "stat_line": line, "play_text": "TOUCHDOWN"}


def test_owner_targeted_not_league_wide():
    sends = []
    n = _run_notify({"g1": [_td_row("rb", "p1", rush_td=1, rush_yds=3)]},
                    ROSTERS, PLAYER_INFO, SCORING, sends)
    assert n == 1
    assert len(sends) == 1
    assert sends[0]["owner"] == "O1"           # only the RB's owner
    assert sends[0]["notif_type"] == "redzone_scores"


def test_dedupe_across_repeated_polls():
    sends = []
    pbp = {"g1": [_td_row("rb", "p1", rush_td=1)]}
    # Two workers / two polls observe the same touchdown.
    fake_db, _ = _fake_db_module()
    pkg = sys.modules.get("dashboard_services") or types.ModuleType("dashboard_services")
    with mock.patch.dict(sys.modules, {"dashboard_services": pkg,
                                       "dashboard_services.db": fake_db}), \
         mock.patch.object(
             pn, "_broadcast_owner",
             side_effect=lambda *a, **k: sends.append(k.get("notif_type")) or 1):
        first = pn.notify_redzone_scores("L1", "sleeper", pbp, PLAYER_INFO, ROSTERS, SCORING)
        second = pn.notify_redzone_scores("L1", "sleeper", pbp, PLAYER_INFO, ROSTERS, SCORING)
    assert first == 1
    assert second == 0          # atomic claim already taken → no duplicate send
    assert len(sends) == 1


def test_unrostered_player_td_not_broadcast():
    sends = []
    n = _run_notify({"g1": [_td_row("nobody", "p9", rush_td=1)]},
                    ROSTERS, PLAYER_INFO, SCORING, sends)
    assert n == 0
    assert sends == []


def test_passing_td_notifies_each_owner_once():
    # QB (owner O2) + WR (owner O2) on the same canonical play → one push (same
    # owner). Put the QB on O1 to prove two distinct owners each get one.
    rosters = [
        {"roster_id": 1, "owner_id": "O1", "starters": ["qb"], "players": ["qb"]},
        {"roster_id": 2, "owner_id": "O2", "starters": ["wr"], "players": ["wr"]},
    ]
    pbp = {"g1": [
        {"pid": "qb", "play_id": "p1", "is_td": True, "play_state": "VALID",
         "stat_line": {"pass_td": 1}, "play_text": "TD pass"},
        {"pid": "wr", "play_id": "p1", "is_td": True, "play_state": "VALID",
         "stat_line": {"rec_td": 1}, "play_text": "TD catch"},
    ]}
    sends = []
    n = _run_notify(pbp, rosters, PLAYER_INFO, SCORING, sends)
    owners = sorted(s["owner"] for s in sends)
    assert owners == ["O1", "O2"]
    assert n == 2


def test_nullified_td_is_not_pushed():
    sends = []
    row = _td_row("rb", "p1", rush_td=1)
    row["play_state"] = "OVERTURNED"
    n = _run_notify({"g1": [row]}, ROSTERS, PLAYER_INFO, SCORING, sends)
    assert n == 0
    assert sends == []


def test_app_state_claim_is_atomic_one_shot():
    """The dedupe primitive returns True once, then False for the same key."""
    fake_db, _ = _fake_db_module()
    conn = fake_db.get_conn()
    assert pn._app_state_claim(conn, "redzone_td:L1:g1:p1:O1") is True
    assert pn._app_state_claim(conn, "redzone_td:L1:g1:p1:O1") is False
    assert pn._app_state_claim(conn, "redzone_td:L1:g1:p1:O2") is True
