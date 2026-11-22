"""Server-side ScoreZone big-play push: 30+ yard gains, owner targeting.

The big-play twin of the TD push (tests/test_scorezone_push_dedupe.py):
a starter who gains 30+ yards on one play pushes exactly once to his
owner, bench players stay quiet, touchdown plays stay with the TD alert,
and the atomic claim keeps repeat polls from double-sending.
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import utils.push_notifications as pn

try:
    import dashboard_services.api  # noqa: F401
    import dashboard_services.db  # noqa: F401
except Exception:
    pass


def _fake_db_module():
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


def _patch(sends, fake_db):
    pkg = sys.modules.get("dashboard_services") or types.ModuleType("dashboard_services")
    return (
        mock.patch.dict(sys.modules, {"dashboard_services": pkg,
                                      "dashboard_services.db": fake_db}),
        mock.patch.object(
            pn, "_broadcast_owner",
            side_effect=lambda league_id, owner_id, title, body, url="/", tag="update", notif_type=None:
            sends.append({"league": league_id, "owner": owner_id, "title": title,
                          "body": body, "notif_type": notif_type}) or 1),
    )


def _run_notify(pbp_by_game, rosters, player_info, scoring, sends):
    fake_db, _claimed = _fake_db_module()
    p1, p2 = _patch(sends, fake_db)
    with p1, p2:
        return pn.notify_scorezone_big_plays(
            "L1", "sleeper", pbp_by_game, player_info, rosters, scoring,
            season=2026, week=11,
        )


ROSTERS = [
    {"roster_id": 1, "owner_id": "O1", "starters": ["rb"], "players": ["rb", "benchwr"]},
    {"roster_id": 2, "owner_id": "O2", "starters": ["wr", "qb"], "players": ["wr", "qb"]},
]
PLAYER_INFO = {
    "rb": {"name": "Javonte Williams", "pos": "RB"},
    "wr": {"name": "Justin Jefferson", "pos": "WR"},
    "qb": {"name": "Carson Wentz", "pos": "QB"},
    "benchwr": {"name": "Bench Receiver", "pos": "WR"},
}
SCORING = {"rush_yd": 0.1, "rec_yd": 0.1, "pass_yd": 0.04, "rush_td": 6.0, "rec_td": 6.0}


def _row(pid, play_id, **line):
    return {"pid": pid, "play_id": play_id, "is_td": False, "play_state": "VALID",
            "stat_line": line, "play_text": "big gain"}


def test_thirty_yard_run_alerts_the_owner():
    sends = []
    n = _run_notify({"g1": [_row("rb", "p1", rush_yds=30, carries=1)]},
                    ROSTERS, PLAYER_INFO, SCORING, sends)
    assert n == 1
    assert len(sends) == 1
    assert sends[0]["owner"] == "O1"
    assert sends[0]["title"].startswith("Big play: Javonte Williams")
    assert "30-yd run" in sends[0]["body"]
    assert sends[0]["notif_type"] == "redzone_scores"


def test_twenty_nine_yards_stays_quiet():
    sends = []
    n = _run_notify({"g1": [_row("rb", "p1", rush_yds=29, carries=1)]},
                    ROSTERS, PLAYER_INFO, SCORING, sends)
    assert n == 0
    assert sends == []


def test_touchdown_play_is_left_to_the_td_alert():
    sends = []
    td_catch = {"pid": "wr", "play_id": "p1", "is_td": True, "play_state": "VALID",
                "stat_line": {"rec_td": 1, "rec_yds": 45}, "play_text": "TD"}
    n = _run_notify({"g1": [td_catch]}, ROSTERS, PLAYER_INFO, SCORING, sends)
    assert n == 0
    assert sends == []


def test_bench_players_big_play_stays_quiet():
    sends = []
    n = _run_notify({"g1": [_row("benchwr", "p1", rec_yds=52)]},
                    ROSTERS, PLAYER_INFO, SCORING, sends)
    assert n == 0
    assert sends == []


def test_qb_and_receiver_same_owner_one_push_headlined_by_the_catch():
    sends = []
    pbp = {"g1": [
        _row("qb", "p1", pass_yds=38, pass_att=1, pass_cmp=1),
        _row("wr", "p1", rec_yds=38, rec=1),
    ]}
    n = _run_notify(pbp, ROSTERS, PLAYER_INFO, SCORING, sends)
    assert n == 1
    assert len(sends) == 1
    assert sends[0]["owner"] == "O2"
    assert "Justin Jefferson" in sends[0]["title"]
    assert "38-yd catch" in sends[0]["body"]


def test_dedupe_across_repeated_polls():
    sends = []
    pbp = {"g1": [_row("rb", "p1", rush_yds=41, carries=1)]}
    fake_db, _claimed = _fake_db_module()
    p1, p2 = _patch(sends, fake_db)
    with p1, p2:
        first = pn.notify_scorezone_big_plays(
            "L1", "sleeper", pbp, PLAYER_INFO, ROSTERS, SCORING, season=2026, week=11)
        second = pn.notify_scorezone_big_plays(
            "L1", "sleeper", pbp, PLAYER_INFO, ROSTERS, SCORING, season=2026, week=11)
    assert first == 1
    assert second == 0
    assert len(sends) == 1
