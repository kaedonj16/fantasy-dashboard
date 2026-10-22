"""Tests for Start/Sit advice tracking (dashboard_services.start_sit_tracking).

Pure transforms are locked here with payloads shaped exactly like the real
``_build_start_sit_options`` output (dict rows, float scores, swap entries of
``{start: {player_id, name, position, proj}, sit: {...} | None, gain, slot}``)
and provider matchup rows shaped like Sleeper's (``players_points`` dict).
The DB paths run against a fake psycopg-style connection (dict rows,
``execute(...).fetchone()/fetchall()``, explicit ``commit()``), mirroring the
real ``dashboard_services.db.get_conn`` contract.
"""
from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from dashboard_services import start_sit_tracking as trk


# ---------------------------------------------------------------------------
# Fixtures shaped like the real payloads
# ---------------------------------------------------------------------------

ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                    "BN", "BN", "BN", "BN"]


def _row(pid, name, *, start=False, flex=False, score=50.0, pct=75,
         proj=10.0, close=None, guest=False):
    row = {
        "player_id": pid, "name": name, "team": "KC", "opponent": "vs BUF",
        "on_bye": False, "proj_pts": proj, "start_score": score,
        "start_score_pct": pct, "start": start, "flex_start": flex,
        "flex_slot": "FLEX" if flex else None,
    }
    if close:
        row["close_call"] = close
    if guest:
        row["guest"] = True
    return row


def _positions():
    return {
        "QB": [
            _row("qb1", "QB One", start=True, score=92.0, pct=95, proj=22.0,
                 close={"vs_name": "QB Two", "vs_player_id": "qb2",
                        "win_prob": 0.78}),
            _row("qb2", "QB Two", score=70.0, pct=60, proj=17.0),
        ],
        "RB": [
            _row("rb1", "RB One", start=True, score=88.0, pct=90, proj=16.0),
            _row("rb2", "RB Two", start=True, score=80.0, pct=82, proj=14.0),
            _row("rb3", "RB Three", start=True, flex=True, score=75.0, pct=78,
                 proj=12.0),
            _row("rb4", "RB Four", score=40.0, pct=35, proj=6.0),
            _row("rb9", "Guest Back", start=True, score=99.0, pct=99,
                 proj=25.0, guest=True),
        ],
        "WR": [
            _row("wr1", "WR One", start=True, score=85.0, pct=88, proj=15.0),
            _row("wr2", "WR Two", start=True, score=78.0, pct=80, proj=13.0),
            _row("wr3", "WR Three", score=55.0, pct=50, proj=9.0),
        ],
        "TE": [
            _row("te1", "TE One", start=True, score=72.0, pct=74, proj=10.0),
            _row("te2", "TE Two", score=45.0, pct=40, proj=5.0),
        ],
    }


def _advice():
    return {
        "has_current": True,
        "optimal_pts": 210.5,
        "current_pts": 195.0,
        "delta": 15.5,
        "swaps": [
            {"start": {"player_id": "rb3", "name": "RB Three",
                       "position": "RB", "proj": 75.0},
             "sit": {"player_id": "wr3", "name": "WR Three",
                     "position": "WR", "proj": 55.0},
             "gain": 20.0, "slot": "FLEX"},
        ],
    }


def _snapshot(**over):
    kw = dict(
        season=2026, week=6, platform="sleeper", league_id="L1", roster_id=3,
        positions=_positions(), lineup_advice=_advice(),
        roster_positions=ROSTER_POSITIONS,
        current_starters=["qb1", "rb1", "rb2", "wr1", "wr2", "wr3", "te1"],
        captured_at="2026-10-11T15:00:00+00:00",
    )
    kw.update(over)
    return trk.build_snapshot_payload(**kw)


def _fake_optimal(pts_map, pos_map, roster_positions, all_pids):
    """Greedy slot fill shaped like compute_optimal_lineup's contract."""
    slots = [s for s in roster_positions if s not in ("BN", "IR", "TAXI")]
    elig = {
        "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"},
        "FLEX": {"RB", "WR", "TE"}, "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
        "K": {"K"}, "DEF": {"DEF"},
    }
    used, chosen = set(), set()
    # Most restrictive slots first.
    for slot in sorted(slots, key=lambda s: len(elig.get(s, set()))):
        cands = [p for p in all_pids
                 if p not in used and pos_map.get(p) in elig.get(slot, set())]
        if not cands:
            continue
        best = max(cands, key=lambda p: pts_map.get(p, 0.0))
        used.add(best)
        chosen.add(best)
    return chosen, round(sum(pts_map.get(p, 0.0) for p in chosen), 2)


ACTUALS = {
    "qb1": 25.0, "qb2": 10.0,
    "rb1": 20.0, "rb2": 5.0, "rb3": 18.0, "rb4": 22.0,
    "wr1": 15.0, "wr2": 12.0, "wr3": 8.0,
    "te1": 4.0, "te2": 14.0,
}


# ---------------------------------------------------------------------------
# Snapshot building
# ---------------------------------------------------------------------------

def test_build_snapshot_verdicts_guests_and_current():
    snap = _snapshot()
    assert snap["key"] == "2026:6:sleeper:L1:3"
    by_pid = {p["player_id"]: p for p in snap["players"]}
    # The guest row never enters the snapshot, even flagged as a starter.
    assert "rb9" not in by_pid
    assert by_pid["qb1"]["verdict"] == "START"
    assert by_pid["rb3"]["verdict"] == "FLEX"
    assert by_pid["rb3"]["flex_slot"] == "FLEX"
    assert by_pid["wr3"]["verdict"] == "BENCH"
    assert set(snap["recommended_starters"]) == {
        "qb1", "rb1", "rb2", "rb3", "wr1", "wr2", "te1"}
    # Caller-supplied current starters pass through verbatim.
    assert snap["current_starters"] == [
        "qb1", "rb1", "rb2", "wr1", "wr2", "wr3", "te1"]
    # Close call survives with its pairing.
    assert by_pid["qb1"]["close_call"]["vs_player_id"] == "qb2"
    # Swap normalized to ids + names.
    assert snap["swaps"] == [{
        "start_player_id": "rb3", "start_name": "RB Three",
        "sit_player_id": "wr3", "sit_name": "WR Three",
        "slot": "FLEX", "proj_gain": 20.0,
    }]
    assert snap["advice_score_delta"] == 15.5


def test_build_snapshot_current_derived_when_not_supplied():
    snap = _snapshot(current_starters=None)
    # (recommended - swap-ins) + swap-outs
    assert set(snap["current_starters"]) == {
        "qb1", "rb1", "rb2", "wr1", "wr2", "te1", "wr3"}


# ---------------------------------------------------------------------------
# Grading (pure)
# ---------------------------------------------------------------------------

def test_grade_snapshot_totals_swaps_close_and_projection_error():
    snap = _snapshot()
    grade = trk.grade_snapshot_payload(snap, ACTUALS, optimal_fn=_fake_optimal)
    assert grade["complete"] is True
    # Recommended: qb1 25 + rb1 20 + rb2 5 + rb3 18 + wr1 15 + wr2 12 + te1 4
    assert grade["recommended_actual"] == pytest.approx(99.0)
    # Snapshot-time lineup swaps rb3 out for wr3: 99 - 18 + 8
    assert grade["current_actual"] == pytest.approx(89.0)
    assert grade["advice_delta"] == pytest.approx(10.0)
    # Optimal by actuals: qb1, rb4+rb1, wr1+wr2, te2, flex rb3 = 25+22+20+15+12+14+18
    assert grade["optimal_actual"] == pytest.approx(126.0)
    assert grade["efficiency"] == pytest.approx(round(99.0 / 126.0 * 100, 1))
    (swap,) = grade["swaps"]
    assert swap["outcome"] == "hit"
    assert swap["delta"] == pytest.approx(10.0)  # rb3 18 over wr3 8
    (close,) = grade["close_calls"]
    assert close["player_id"] == "qb1" and close["outcome"] == "hit"
    assert close["band"] == "90-100"
    qb1 = next(p for p in grade["players"] if p["player_id"] == "qb1")
    assert qb1["proj_error"] == pytest.approx(3.0)  # 25 actual vs 22 proj


def test_grade_snapshot_incomplete_when_any_score_missing():
    snap = _snapshot()
    partial = {k: v for k, v in ACTUALS.items() if k != "te2"}
    grade = trk.grade_snapshot_payload(snap, partial, optimal_fn=_fake_optimal)
    assert grade["complete"] is False
    assert grade["missing_player_ids"] == ["te2"]


def test_grade_snapshot_none_scores_are_missing_not_zero():
    snap = _snapshot()
    with_none = dict(ACTUALS, te1=None)
    grade = trk.grade_snapshot_payload(snap, with_none, optimal_fn=_fake_optimal)
    assert grade["complete"] is False
    assert grade["missing_player_ids"] == ["te1"]


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def test_summarize_tracking_season_and_calibration():
    snap6 = _snapshot()
    grade6 = trk.grade_snapshot_payload(snap6, ACTUALS, optimal_fn=_fake_optimal)
    snap7 = _snapshot(week=7)
    out = trk.summarize_tracking([
        {"snapshot": snap6, "grade": grade6},
        {"snapshot": snap7, "grade": None},
    ])
    assert out["state"] == "graded"
    assert out["pending_weeks"] == [7]
    season = out["season"]
    assert season["weeks_graded"] == 1
    assert season["weeks_snapshotted"] == 2
    assert season["recommended_actual"] == pytest.approx(99.0)
    assert season["advice_delta"] == pytest.approx(10.0)
    assert season["swap_record"] == {"hits": 1, "pushes": 0, "total": 1}
    assert season["close_record"] == {"hits": 1, "pushes": 0, "total": 1}
    assert season["proj_mae"] is not None
    (band,) = out["calibration"]
    assert band["band"] == "90-100" and band["n"] == 1
    assert band["hit_rate"] == pytest.approx(100.0)
    # Newest week first in the page list.
    assert [w["week"] for w in out["weeks"]] == [7, 6]


def test_summarize_tracking_states():
    assert trk.summarize_tracking([])["state"] == "no_snapshots"
    snap = _snapshot()
    out = trk.summarize_tracking([{"snapshot": snap, "grade": None}])
    assert out["state"] == "awaiting_grades"
    assert out["season"]["recommended_actual"] is None


# ---------------------------------------------------------------------------
# Persistence against a fake psycopg-style connection
# ---------------------------------------------------------------------------

class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    """Dict-row fake shaped like the real pooled psycopg connection."""

    def __init__(self):
        self.snapshots = {}  # key -> payload dict
        self.grades = {}     # key -> payload dict
        self.meta = {}       # key -> (platform, league_id, season, roster_id, week)

    def execute(self, sql, params=()):
        q = " ".join(sql.split()).upper()
        if q.startswith("CREATE"):
            return _Cursor([])
        if "FROM START_SIT_GRADES WHERE SNAPSHOT_KEY" in q:
            key = params[0]
            return _Cursor([{"snapshot_key": key}] if key in self.grades else [])
        if q.startswith("INSERT INTO START_SIT_SNAPSHOTS"):
            (key, season, week, platform, league_id, roster_id,
             _source, payload) = params
            self.snapshots[key] = json.loads(payload)
            self.meta[key] = (platform, league_id, season, roster_id, week)
            return _Cursor([])
        if q.startswith("INSERT INTO START_SIT_GRADES"):
            key, payload = params
            self.grades[key] = json.loads(payload)
            return _Cursor([])
        if "FROM START_SIT_SNAPSHOTS" in q and "LEFT JOIN" in q:
            rows = []
            for key, payload in self.snapshots.items():
                platform, league_id, season, roster_id, _week = self.meta[key]
                if params[0] != platform or params[1] != league_id \
                        or params[2] != season:
                    continue
                if len(params) > 3 and params[3] != roster_id:
                    continue
                rows.append({
                    "payload": json.dumps(payload),
                    "grade_payload": (
                        json.dumps(self.grades[key]) if key in self.grades
                        else None),
                })
            return _Cursor(rows)
        raise AssertionError(f"unexpected SQL: {q[:80]}")

    def commit(self):
        pass


@pytest.fixture
def fake_db(monkeypatch):
    conn = _FakeConn()

    @contextmanager
    def _cm():
        yield conn

    monkeypatch.setattr(trk, "_get_conn", lambda: _cm())
    monkeypatch.setattr(trk, "_TABLES_READY", True)
    return conn


def test_save_snapshot_refreshes_until_graded_then_locks(fake_db):
    snap = _snapshot()
    assert trk.save_snapshot(snap) is True
    # A later view the same week refreshes the capture.
    snap2 = _snapshot(captured_at="2026-10-12T10:00:00+00:00")
    assert trk.save_snapshot(snap2) is True
    assert fake_db.snapshots[snap["key"]]["captured_at"] == \
        "2026-10-12T10:00:00+00:00"
    # Once graded, the snapshot locks: saves are refused, payload untouched.
    assert trk._save_grade(snap["key"], {"complete": True}) is True
    snap3 = _snapshot(captured_at="2026-10-13T10:00:00+00:00")
    assert trk.save_snapshot(snap3) is False
    assert fake_db.snapshots[snap["key"]]["captured_at"] == \
        "2026-10-12T10:00:00+00:00"


def test_grade_league_week_writes_once_and_summary_reads_it(fake_db):
    snap = _snapshot()
    assert trk.save_snapshot(snap) is True
    matchups = [{"roster_id": 3, "players_points": ACTUALS,
                 "players": list(ACTUALS), "starters": []}]
    assert trk.grade_league_week("sleeper", "L1", 2026, 6,
                                 matchups=matchups) == 1
    # Already graded: a second run is a no-op, not a rewrite.
    assert trk.grade_league_week("sleeper", "L1", 2026, 6,
                                 matchups=matchups) == 0
    summary = trk.get_tracking_summary("sleeper", "L1", 2026, 3)
    assert summary["state"] == "graded"
    assert summary["season"]["recommended_actual"] == pytest.approx(99.0)


def test_grade_league_week_skips_incomplete_matchups(fake_db):
    snap = _snapshot()
    assert trk.save_snapshot(snap) is True
    partial = {k: v for k, v in ACTUALS.items() if k != "rb4"}
    matchups = [{"roster_id": 3, "players_points": partial}]
    assert trk.grade_league_week("sleeper", "L1", 2026, 6,
                                 matchups=matchups) == 0
    assert fake_db.grades == {}
