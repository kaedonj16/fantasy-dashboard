"""Server-side ScoreZone scoring push: owner targeting + canonical dedupe.

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

# Pre-import the real dashboard_services submodules: notify_scorezone_scores
# lazily imports utils.fantasy_scoring, whose chain imports
# dashboard_services.api at module load. If that first import happens while
# the fake bare ``dashboard_services`` module below is patched into
# sys.modules, it fails and notify silently returns 0. Caching the real
# submodules first mirrors full-suite state and keeps this file green when
# run standalone.
try:
    import dashboard_services.api  # noqa: F401
    import dashboard_services.db  # noqa: F401
except Exception:
    pass


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
        return pn.notify_scorezone_scores(
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
        first = pn.notify_scorezone_scores("L1", "sleeper", pbp, PLAYER_INFO, ROSTERS, SCORING)
        second = pn.notify_scorezone_scores("L1", "sleeper", pbp, PLAYER_INFO, ROSTERS, SCORING)
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


def test_bench_player_td_is_not_pushed():
    """Only starting-lineup players trigger TD alerts -- a scorer riding the
    bench stays quiet while a starter on the same roster still notifies."""
    sends = []
    rosters = [
        {"roster_id": 1, "owner_id": "O1", "starters": ["rb"], "players": ["rb", "wr"]},
    ]
    pbp = {"g1": [
        _td_row("rb", "p1", rush_td=1),
        _td_row("wr", "p2", rec_td=1),
    ]}
    n = _run_notify(pbp, rosters, PLAYER_INFO, SCORING, sends)
    assert n == 1
    assert len(sends) == 1
    assert sends[0]["owner"] == "O1"
    assert "Aaron Jones" in sends[0]["title"]  # the starter, not the benched WR


def test_starter_wins_over_bench_for_same_player():
    """The same NFL player owned by two teams: the starter's owner is
    notified, the owner with him on the bench is not."""
    sends = []
    rosters = [
        {"roster_id": 1, "owner_id": "O1", "starters": [], "players": ["rb"]},
        {"roster_id": 2, "owner_id": "O2", "starters": ["rb"], "players": ["rb"]},
    ]
    pbp = {"g1": [_td_row("rb", "p1", rush_td=1)]}
    n = _run_notify(pbp, rosters, PLAYER_INFO, SCORING, sends)
    assert n == 1
    assert [s["owner"] for s in sends] == ["O2"]


def test_all_bench_td_plays_send_nothing():
    sends = []
    rosters = [
        {"roster_id": 1, "owner_id": "O1", "starters": ["qb"], "players": ["qb", "wr"]},
    ]
    pbp = {"g1": [_td_row("wr", "p1", rec_td=1)]}
    n = _run_notify(pbp, rosters, PLAYER_INFO, SCORING, sends)
    assert n == 0
    assert sends == []


def test_app_state_claim_is_atomic_one_shot():
    """The dedupe primitive returns True once, then False for the same key."""
    fake_db, _ = _fake_db_module()
    conn = fake_db.get_conn()
    assert pn._app_state_claim(conn, "redzone_td:L1:g1:p1:O1") is True
    assert pn._app_state_claim(conn, "redzone_td:L1:g1:p1:O1") is False
    assert pn._app_state_claim(conn, "redzone_td:L1:g1:p1:O2") is True


# ── Wrong-player TD headline regression (2026-11-15) ─────────────────────────
# A live push read "TD: Nico Collins" (no points) over the body "11-yd pass
# to D.Schultz". The extractors emit one row per player resolved from the
# booth line, all sharing the play text: the passer and the receiver carry
# the TD stats, while a player merely MENTIONED in the line (here a declined
# penalty clause) carries an empty stat line but still inherits is_td=True
# (see _row_inherits_td in utils/scorezone.py). The rows below are shaped
# exactly as extract_espn_pbp_plays emits them for that play. A mention row
# is not a scorer: it must never headline a push, and when the play's scorer
# is identified and belongs to someone else, the mentioned player's owner
# gets no TD push for the play at all.

HOU_TEXT = ("(Shotgun) C.Stroud pass short right to D.Schultz for 11 yards, "
            "TOUCHDOWN. PENALTY on HOU-N.Collins, Illegal Formation, "
            "5 yards, declined.")
HOU_INFO = {
    "qb_hou": {"name": "C.J. Stroud", "pos": "QB"},
    "wr_hou": {"name": "Nico Collins", "pos": "WR"},
    "te_hou": {"name": "Dalton Schultz", "pos": "TE"},
}
HOU_SCORING = {"pass_td": 4.0, "rec_td": 6.0, "rec": 1.0,
               "rec_yd": 0.1, "pass_yd": 0.04}


def _hou_td_rows():
    return [
        {"pid": "qb_hou", "play_id": "p_td", "is_td": True,
         "play_state": "VALID",
         "stat_line": {"pass_yds": 11, "pass_cmp": 1, "pass_att": 1,
                       "pass_td": 1},
         "play_text": HOU_TEXT},
        {"pid": "te_hou", "play_id": "p_td", "is_td": True,
         "play_state": "VALID",
         "stat_line": {"rec": 1, "rec_yds": 11, "targets": 1, "rec_td": 1},
         "play_text": HOU_TEXT},
        # Mention-only row: no stats, still is_td (extractor contract).
        {"pid": "wr_hou", "play_id": "p_td", "is_td": True,
         "play_state": "VALID", "stat_line": {}, "play_text": HOU_TEXT},
    ]


def _run_notify_full(pbp_by_game, rosters, player_info, scoring, sends):
    """Like _run_notify but also records the body and pins the league name
    so the settled body format "{desc} · {league}" is assertable."""
    fake_db, claimed = _fake_db_module()
    pkg = sys.modules.get("dashboard_services") or types.ModuleType("dashboard_services")
    with mock.patch.dict(sys.modules, {"dashboard_services": pkg,
                                       "dashboard_services.db": fake_db}), \
         mock.patch.object(pn, "_league_display_name", return_value="blackedraw"), \
         mock.patch.object(
             pn, "_broadcast_owner",
             side_effect=lambda league_id, owner_id, title, body, url="/", tag="update", notif_type=None:
             sends.append({"league": league_id, "owner": owner_id,
                           "title": title, "body": body,
                           "notif_type": notif_type}) or 1):
        return pn.notify_scorezone_scores(
            "L1", "sleeper", pbp_by_game, player_info, rosters, scoring,
            season=2026, week=11,
        )


def test_mentioned_non_scorer_starter_gets_no_td_push():
    """(a) The owner's rostered starter is neither the scorer nor the
    passer -- he is only mentioned on the play. He gets no push; the
    passer's owner still gets his own correct one."""
    sends = []
    rosters = [
        {"roster_id": 1, "owner_id": "O_MENTION",
         "starters": ["wr_hou"], "players": ["wr_hou"]},
        {"roster_id": 2, "owner_id": "O_QB",
         "starters": ["qb_hou"], "players": ["qb_hou"]},
    ]
    n = _run_notify_full({"g1": _hou_td_rows()}, rosters, HOU_INFO,
                         HOU_SCORING, sends)
    assert n == 1
    assert [s["owner"] for s in sends] == ["O_QB"]
    assert sends[0]["title"] == "TD: C.J. Stroud (+4.4)"
    assert all("Collins" not in s["title"] for s in sends)


def test_receiver_td_titles_receiver_not_mentioned_wr():
    """(b) The screenshot shape: a receiving TD with another same-team WR
    rostered and started by a different owner. The receiver's owner gets
    the settled format naming the receiver with his points; the WR's owner
    gets nothing."""
    sends = []
    rosters = [
        {"roster_id": 1, "owner_id": "O_MENTION",
         "starters": ["wr_hou"], "players": ["wr_hou"]},
        {"roster_id": 2, "owner_id": "O_TE",
         "starters": ["te_hou"], "players": ["te_hou"]},
    ]
    n = _run_notify_full({"g1": _hou_td_rows()}, rosters, HOU_INFO,
                         HOU_SCORING, sends)
    assert n == 1
    assert len(sends) == 1
    assert sends[0]["owner"] == "O_TE"
    assert sends[0]["title"] == "TD: Dalton Schultz (+8.1)"
    assert sends[0]["body"] == "11-yd pass to D.Schultz · blackedraw"


def test_same_owner_mention_row_never_shadows_own_scorer():
    """One owner starts both the mentioned WR and the scoring TE, and the
    mention row arrives first: the push still headlines the scorer."""
    sends = []
    rows = _hou_td_rows()
    rows = [rows[2], rows[1], rows[0]]  # mention row first
    rosters = [
        {"roster_id": 1, "owner_id": "O_BOTH",
         "starters": ["wr_hou", "te_hou"], "players": ["wr_hou", "te_hou"]},
    ]
    n = _run_notify_full({"g1": rows}, rosters, HOU_INFO, HOU_SCORING, sends)
    assert n == 1
    assert len(sends) == 1
    assert sends[0]["title"] == "TD: Dalton Schultz (+8.1)"
    assert sends[0]["body"] == "11-yd pass to D.Schultz · blackedraw"


def test_unparseable_scorer_falls_back_to_generic_title():
    """No row on the play carries a TD stat (the stat parse missed the
    credit entirely), so the scorer cannot be named honestly: the alert
    still goes to the mentioned starter's owner, but with a generic title
    consistent with the body instead of a possibly-wrong name."""
    sends = []
    rows = [
        {"pid": "qb_hou", "play_id": "p_td", "is_td": True,
         "play_state": "VALID",
         "stat_line": {"pass_yds": 11, "pass_cmp": 1, "pass_att": 1},
         "play_text": HOU_TEXT},
        {"pid": "te_hou", "play_id": "p_td", "is_td": True,
         "play_state": "VALID", "stat_line": {}, "play_text": HOU_TEXT},
        {"pid": "wr_hou", "play_id": "p_td", "is_td": True,
         "play_state": "VALID", "stat_line": {}, "play_text": HOU_TEXT},
    ]
    rosters = [
        {"roster_id": 1, "owner_id": "O_MENTION",
         "starters": ["wr_hou"], "players": ["wr_hou"]},
    ]
    n = _run_notify_full({"g1": rows}, rosters, HOU_INFO, HOU_SCORING, sends)
    assert n == 1
    assert len(sends) == 1
    assert sends[0]["title"] == "Touchdown!"
    assert sends[0]["body"] == "11-yd pass to D.Schultz · blackedraw"
