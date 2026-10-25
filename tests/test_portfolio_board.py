"""Focused tests for the Portfolio Front Office board (v25 redesign).

Covers the pure helpers in dashboard_services/portfolio_board.py (receipts,
posture, weakest room, heatmap, move keys, grouping, timing/locks, done
verification, grading core, grade line) and the DB stores behind a fake
connection, following tests/test_action_dismissals.py's pattern.
"""
import pytest

pytest.importorskip("flask")

from dashboard_services import portfolio_board as pb


# ── Receipts ────────────────────────────────────────────────────────────────

def test_receipt_tag_tossup_band():
    assert pb.receipt_tag(121.8, 120.9) == "Toss-up"
    assert pb.receipt_tag(100.0, 103.0) == "Toss-up"   # exactly at the band
    assert pb.receipt_tag(100.0, 103.1) == "L"
    assert pb.receipt_tag(124.6, 108.2) == "W"
    assert pb.receipt_tag(None, 10) is None


def test_receipts_tile_reconciles():
    rows = [{"tag": "W"}, {"tag": "L"}, {"tag": "Toss-up"}, {"tag": None}, {}]
    tile = pb.receipts_tile(rows)
    assert tile == {"w": 1, "l": 1, "t": 1}
    assert pb.tile_str(tile) == "1-1"


# ── Posture / weakest room / heatmap ────────────────────────────────────────

def test_posture_thresholds():
    assert pb.posture_for(74) == ("Contend", "contend")
    assert pb.posture_for(38) == ("Fringe", "fringe")
    assert pb.posture_for(18) == ("Rebuild", "rebuild")
    assert pb.posture_for(None, wins=5, losses=2) == ("Contend", "contend")
    assert pb.posture_for(None, wins=0, losses=0) == (None, "")


def test_weakest_room_picks_worst_rank():
    room = pb.weakest_room({"QB": 4, "RB": 9, "WR": 2, "TE": 6}, 10)
    assert room == {"pos": "RB", "rank": 9, "total": 10}
    assert pb.weakest_room({}, 10) is None


def test_heatmap_rows_skip_empty_and_tier():
    rows = pb.heatmap_rows([
        {"name": "A", "pos_user_rank": {"QB": 1, "RB": 10, "WR": 5, "TE": None},
         "total_teams": 10},
        {"name": "B", "pos_user_rank": {}, "total_teams": 10},
    ])
    assert len(rows) == 1
    cells = rows[0]["cells"]
    assert cells["QB"] == {"rank": 1, "tier": "good"}
    assert cells["RB"] == {"rank": 10, "tier": "weak"}
    assert cells["WR"]["tier"] == "mid"
    assert cells["TE"] is None


# ── Move keys (client parity: goldens computed with node) ──────────────────

def test_move_key_prefers_dismiss_key():
    assert pb.move_key({"dismiss_key": "wv:123", "kind": "waiver",
                        "title": "Add X"}) == "wv:123"
    # Sanitized to the dismissal store's item_key alphabet so the key
    # round-trips through both stores.
    assert pb.move_key({"dismiss_key": "swap:100>200", "kind": "lineup",
                        "title": "Start X"}) == "swap:100-200"


def test_move_key_fallback_matches_client_hash():
    a = {"kind": "waiver", "title": "Add Top Player (RB)"}
    assert pb.move_key(a) == "pf:waiver:nhcgry"
    b = {"kind": "lineup", "title": "J. Smith is out"}
    assert pb.move_key(b) == "pf:lineup:beym9w"


# ── Grouping ────────────────────────────────────────────────────────────────

def test_group_actions_global_ranks_and_meta():
    actions = [
        {"kind": "injury", "platform": "sleeper", "league_id": "1",
         "league_name": "Alpha", "title": "Move RB to IR"},
        {"kind": "waiver", "platform": "sleeper", "league_id": "2",
         "league_name": "Beta", "title": "Add RB"},
        {"kind": "lineup", "platform": "sleeper", "league_id": "1",
         "league_name": "Alpha", "title": "Fix lineup"},
    ]
    meta = {"sleeper|1": {"league_name": "Alpha", "record_line": "5-2 · 2nd"}}
    groups = pb.group_actions(actions, meta)
    assert [g["league_name"] for g in groups] == ["Alpha", "Beta"]
    assert [m["rank"] for m in groups[0]["moves"]] == [1, 3]
    assert groups[1]["moves"][0]["rank"] == 2
    assert groups[0]["meta"]["record_line"] == "5-2 · 2nd"
    assert groups[0]["moves"][0]["move_key"]


# ── Timing / locks ──────────────────────────────────────────────────────────

def _et_epoch(year, month, day, hour, minute=0):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime(year, month, day, hour, minute,
                    tzinfo=ZoneInfo("America/New_York")).timestamp()


SUN_1PM_ET = _et_epoch(2026, 10, 25, 13)  # Sunday 1 PM ET kickoff


def test_timing_waiver_never_locks():
    out = pb.timing_for({"kind": "waiver", "lock_epoch": 100}, now_ts=200)
    assert out == {"label": "Before waivers", "locked": False, "lock_label": ""}


def test_timing_locked_after_kickoff():
    out = pb.timing_for({"kind": "lineup", "lock_epoch": SUN_1PM_ET},
                        now_ts=SUN_1PM_ET + 60)
    assert out["locked"] is True
    assert out["lock_label"].startswith("Locked ·")


def test_timing_lock_countdown_same_day():
    out = pb.timing_for({"kind": "lineup", "lock_epoch": SUN_1PM_ET},
                        now_ts=SUN_1PM_ET - 3600)
    assert out["locked"] is False
    assert out["label"].startswith("Locks ")


def test_timing_never_locks_before_deadline():
    out = pb.timing_for({"kind": "injury"}, now_ts=SUN_1PM_ET + 99999)
    assert out["locked"] is False
    assert out["label"] == "Do now"


# ── Done verification ───────────────────────────────────────────────────────

ROSTER = {"players": ["1", "2", "3"], "starters": ["1", "2"],
          "reserve": ["3"]}


def test_verify_swap():
    a = {"platform": "sleeper", "verify": {"type": "swap", "in": "2", "out": "9"}}
    assert pb.verify_done(a, ROSTER)["verified"] is True
    a2 = {"platform": "sleeper", "verify": {"type": "swap", "in": "3", "out": "1"}}
    out = pb.verify_done(a2, ROSTER)
    assert out["verified"] is False and "swap" in out["label"]


def test_verify_ir_and_drop():
    ir = {"platform": "sleeper", "verify": {"type": "ir", "pid": "3"}}
    out = pb.verify_done(ir, ROSTER)
    assert out == {"verified": True, "label": "IR slot set in Sleeper"}
    drop = {"verify": {"type": "drop", "pid": "3"}}
    assert pb.verify_done(drop, ROSTER)["verified"] is False
    drop2 = {"verify": {"type": "drop", "pid": "99"}}
    assert pb.verify_done(drop2, ROSTER)["verified"] is True


def test_verify_claim_never_verified():
    out = pb.verify_done({"verify": {"type": "claim", "pid": "5"}}, ROSTER)
    assert out == {"verified": False, "label": "Claims have not processed yet"}


def test_verify_without_roster_is_marked():
    out = pb.verify_done({"verify": {"type": "ir", "pid": "3"}}, None)
    assert out["verified"] is False


# ── Grading core + grade line ───────────────────────────────────────────────

MOVES = [
    {"platform": "sleeper", "league_id": "1", "move_key": "swap:2>1",
     "kind": "lineup", "verify": {"type": "swap", "in": "2", "out": "1"}},
    {"platform": "sleeper", "league_id": "1", "move_key": "swap:4>3",
     "kind": "lineup", "verify": {"type": "swap", "in": "4", "out": "3"}},
    {"platform": "sleeper", "league_id": "1", "move_key": "wv:9",
     "kind": "waiver", "verify": {"type": "claim", "pid": "9"}},
]


def test_grade_snapshot_moves_followed_and_skipped():
    done = {("sleeper", "1", "swap:2>1"), ("sleeper", "1", "wv:9")}
    points = {("sleeper", "1"): {"1": 10.0, "2": 22.4, "3": 5.0, "4": 11.1}}
    grade = pb.grade_snapshot_moves(MOVES, done, points)
    assert grade["followed"] == 2
    assert grade["total"] == 3
    assert grade["delta_pts"] == 12.4
    assert grade["skipped_cost"] == 6.1


def test_grade_snapshot_moves_ungradeable_is_none():
    grade = pb.grade_snapshot_moves(MOVES, set(), {("sleeper", "1"): {}})
    assert grade is None


def test_grade_line_format():
    line = pb.grade_line({"week": 7, "followed": 4, "total": 5,
                          "delta_pts": 12.4, "skipped_cost": 6.1})
    assert line == ("Week 7 board: you followed 4 of 5 moves · worth +12.4 pts "
                    "vs leaving the lineups as they were. "
                    "The moves you skipped cost 6.1.")
    line2 = pb.grade_line({"week": 7, "followed": 5, "total": 5,
                           "delta_pts": 3.0, "skipped_cost": 0.0})
    assert "skipped" not in line2
    assert pb.grade_line(None) == ""


# ── Done store (fake conn, same pattern as test_action_dismissals) ─────────

class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn


class _FakeConn:
    """Implements exactly the statements portfolio_board issues."""

    def __init__(self):
        self.states = {}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def commit(self):
        pass

    def execute(self, sql, params=()):
        s = " ".join(sql.split()).upper()
        self._last = (s, params)
        return self

    def fetchall(self):
        s, p = self._last
        if s.startswith("SELECT PLATFORM, LEAGUE_ID, MOVE_KEY"):
            user_key, season, week = p
            return [
                {"platform": plat, "league_id": lid, "move_key": mk,
                 "verified": v["verified"], "verify_label": v["label"],
                 "done_at": "2026-10-25T13:41:00"}
                for (uk, plat, lid, sn, wk, mk), v in self.states.items()
                if uk == user_key and sn == season and wk == week
            ]
        return []

    def fetchone(self):
        return None


@pytest.fixture()
def board_store(monkeypatch):
    conn = _FakeConn()
    import dashboard_services.db as db_mod
    monkeypatch.setattr(db_mod, "get_conn", lambda: conn)
    pb._DONE_TABLE_READY = True
    # Route the module's INSERT/DELETE through the fake's state dict.
    orig_execute = _FakeConn.execute

    def execute(self, sql, params=()):
        s = " ".join(sql.split()).upper()
        if s.startswith("INSERT INTO PORTFOLIO_MOVE_STATES"):
            (uk, plat, lid, sn, wk, mk, verified, label) = params
            self.states[(uk, plat, lid, sn, wk, mk)] = {
                "verified": verified, "label": label}
        elif s.startswith("DELETE FROM PORTFOLIO_MOVE_STATES"):
            self.states.pop(tuple(params), None)
        return orig_execute(self, sql, params)

    monkeypatch.setattr(_FakeConn, "execute", execute)
    return conn


def test_done_store_roundtrip(board_store):
    assert pb.set_done("acct:1", "sleeper", "L1", 2026, 8, "swap:2>1",
                       True, "Lineup set in Sleeper") is True
    states = pb.load_done_states("acct:1", 2026, 8)
    key = ("sleeper", "L1", "swap:2>1")
    assert states[key]["verified"] is True
    assert states[key]["label"] == "Lineup set in Sleeper"
    assert pb.clear_done("acct:1", "sleeper", "L1", 2026, 8, "swap:2>1") is True
    assert pb.load_done_states("acct:1", 2026, 8) == {}


def test_done_store_guest_noop():
    assert pb.load_done_states("", 2026, 8) == {}
    assert pb.set_done("", "sleeper", "L1", 2026, 8, "k", True, "x") is False
