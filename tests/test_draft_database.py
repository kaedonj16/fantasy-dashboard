"""Focused tests for the Draft Database (routes/draft_db_bp.py,
utils/draft_db_grades.py, crawler metadata extraction).

The API tests run the real blueprint on a bare Flask app with a fake
DB connection, following the stubbed-deps pattern of the trade-database
tests. The board YOU rule is asserted both ways: a viewer who owns a
slot gets markers, a viewer who does not gets none.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")

from flask import Flask

import routes.draft_db_bp as ddb
from utils.draft_db_grades import build_grade_summary, headline_superlatives


# ── Fake DB ─────────────────────────────────────────────────────────────────

class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    """Routes execute() by SQL substring to canned rows."""

    def __init__(self, drafts, picks=None, adp_rows=None, recent=0):
        self.drafts = drafts
        self.picks = picks or []
        self.adp_rows = adp_rows or []
        self.recent = recent
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, query, params=()):
        self.queries.append((query, list(params or ())))
        q = " ".join(query.split())
        if "COUNT(*) AS n FROM draft_adp_drafts d" in q:
            return _Cursor([{"n": len(self.drafts)}])
        if "FROM draft_adp_drafts d" in q and "ORDER BY" in q:
            limit = params[-2]
            offset = params[-1]
            return _Cursor(self.drafts[offset:offset + limit])
        if "SELECT * FROM draft_adp_drafts" in q:
            did = params[0]
            return _Cursor([d for d in self.drafts
                            if d["draft_id"] == did])
        if "FROM draft_adp_picks WHERE draft_id" in q:
            return _Cursor(self.picks)
        if "COUNT(*) AS n FROM draft_adp_drafts" in q:
            return _Cursor([{"n": self.recent}])
        if "FROM draft_adp " in q or "FROM draft_adp\n" in q:
            return _Cursor(self.adp_rows)
        if "draft_started_at >= %s" in q and "GROUP BY" in q:
            return _Cursor(self.adp_rows)
        return _Cursor([])


def _draft(**kw):
    base = {
        "draft_id": "d1", "league_id": "L1", "league_name": "Alpha League",
        "season": 2026, "draft_type": "startup", "league_format": "dynasty",
        "scoring_type": "ppr", "is_superflex": True, "num_teams": 12,
        "rounds": 15, "total_picks": 180, "draft_started_at": None,
        "grade_summary": None, "roster_names": {"1": "Team One", "2": "Team Two"},
        "roster_owners": {"1": "user-me", "2": "user-other"},
        "slot_to_roster_id": {"1": "1", "2": "2"},
        "draft_order_type": "snake",
    }
    base.update(kw)
    return base


def _app(monkeypatch, conn, session_vals=None):
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(ddb.draft_db_bp)
    import dashboard_services.db as dbmod
    monkeypatch.setattr(dbmod, "get_conn", lambda: conn)
    monkeypatch.setattr(ddb, "_players_map", lambda: {
        "p1": {"name": "Player One", "pos": "WR", "team": "KC", "espnID": "111"},
        "p2": {"name": "Player Two", "pos": "RB", "team": "SF", "espnID": "222"},
        "p3": {"name": "Player Three", "pos": "QB", "team": "BUF"},
    })
    ddb._COUNT_CACHE.clear()
    client = app.test_client()
    if session_vals is not None:
        with client.session_transaction() as s:
            s.update(session_vals)
    return client


# ── Filter parsing / SQL ────────────────────────────────────────────────────

def test_parse_filters_defaults_and_caps():
    f = ddb.parse_filters({"limit": "500", "page": "-3"})
    assert f["limit"] == 50
    assert f["page"] == 0
    assert f["sort"] == "recent"
    assert f["draft_type"] == "all"


def test_parse_filters_rejects_unknown_values():
    f = ddb.parse_filters({"draft_type": "auction", "format": "bestball",
                           "league_type": "2qb", "sort": "random"})
    assert f["draft_type"] == "all"
    assert f["format"] == "all"
    assert f["league_type"] == "all"
    assert f["sort"] == "recent"


def test_build_filter_sql_null_format_never_matches():
    f = ddb.parse_filters({"format": "keeper", "league_type": "sf",
                           "num_teams": "10", "season": "2025"})
    where, params = ddb.build_filter_sql(f)
    assert "d.league_format = %s" in where
    assert "d.is_superflex = TRUE" in where
    assert "d.num_teams = %s" in where
    assert params == [2025, "keeper", 10]


def test_build_filter_sql_q_player_probe():
    f = ddb.parse_filters({"q": "one"})
    where, params = ddb.build_filter_sql(f, match_ids=["p1"])
    assert "league_name ILIKE" in where
    assert "draft_adp_picks" in where
    assert params == ["%one%", ["p1"]]


# ── List API ────────────────────────────────────────────────────────────────

def test_list_api_pagination_shape(monkeypatch):
    drafts = [_draft(draft_id=f"d{i}", league_name=f"League {i}")
              for i in range(3)]
    conn = _FakeConn(drafts)
    client = _app(monkeypatch, conn)
    data = client.get("/api/draft-database?limit=2&page=0").get_json()
    assert data["total"] == 3
    assert len(data["drafts"]) == 2
    assert data["has_more"] is True
    assert data["pagination"]["total_pages"] == 2
    assert data["drafts"][0]["league_name"] == "League 0"


def test_list_api_unnamed_league_fallback(monkeypatch):
    conn = _FakeConn([_draft(league_name=None)])
    client = _app(monkeypatch, conn)
    d = client.get("/api/draft-database").get_json()["drafts"][0]
    assert d["league_name"] == "League L1"
    assert d["league_name_known"] is False


def test_list_api_count_uses_cache(monkeypatch):
    conn = _FakeConn([_draft()])
    client = _app(monkeypatch, conn)
    client.get("/api/draft-database")
    client.get("/api/draft-database")
    counts = [q for q, _ in conn.queries if "COUNT(*)" in q]
    assert len(counts) == 1  # second request served from the fingerprint cache


def test_list_api_superlatives_label_family(monkeypatch):
    summary = {
        "at_draft": {
            "family": "at_draft",
            "best_draft": {"roster_id": "1", "team": "Team One",
                           "grade": "A", "avg_adp_diff": 9.0,
                           "graded_picks": 15},
            "biggest_steal": {"player_id": "p1", "pick_no": 48, "round": 4,
                              "roster_id": "1", "team": "Team One",
                              "grade": "A", "adp_diff": 30.0},
            "biggest_bust": {"player_id": "p2", "pick_no": 5, "round": 1,
                             "roster_id": "2", "team": "Team Two",
                             "grade": "F", "adp_diff": -20.0},
        },
        "performance": None, "performance_note": None, "per_pick": {},
    }
    conn = _FakeConn([_draft(grade_summary=summary)])
    client = _app(monkeypatch, conn)
    d = client.get("/api/draft-database").get_json()["drafts"][0]
    assert d["superlative_family"] == "at_draft"
    assert d["best_draft_family"] == "at_draft"
    assert d["biggest_steal"]["name"] == "Player One"
    assert d["biggest_steal"]["headshot"].endswith("/111.png")


# ── Board API + YOU rule ────────────────────────────────────────────────────

_PICKS = [
    {"player_id": "p1", "pick_no": 1, "round": 1, "pick_in_round": 1, "roster_id": "1"},
    {"player_id": "p2", "pick_no": 2, "round": 1, "pick_in_round": 2, "roster_id": "2"},
    {"player_id": "p3", "pick_no": 3, "round": 2, "pick_in_round": 2, "roster_id": "2"},
]


def test_board_viewer_in_draft_gets_you_markers(monkeypatch):
    conn = _FakeConn([_draft()], picks=_PICKS)
    client = _app(monkeypatch, conn, session_vals={"viewer_user_id": "user-me"})
    data = client.get("/api/draft-database/d1/board").get_json()
    assert data["viewer_in_draft"] is True
    you_teams = [t for t in data["teams"] if t["is_you"]]
    assert len(you_teams) == 1 and you_teams[0]["roster_id"] == "1"
    cells = [c for row in data["rounds"] for c in row["cells"] if c]
    assert any(c["is_you"] for c in cells)
    assert not any(c["is_you"] for c in cells if c["roster_id"] == "2")


def test_board_viewer_not_in_draft_gets_no_you(monkeypatch):
    conn = _FakeConn([_draft()], picks=_PICKS)
    client = _app(monkeypatch, conn,
                  session_vals={"viewer_user_id": "someone-else"})
    data = client.get("/api/draft-database/d1/board").get_json()
    assert data["viewer_in_draft"] is False
    assert all(not t["is_you"] for t in data["teams"])
    cells = [c for row in data["rounds"] for c in row["cells"] if c]
    assert all(not c["is_you"] for c in cells)


def test_board_guest_gets_no_you(monkeypatch):
    conn = _FakeConn([_draft()], picks=_PICKS)
    client = _app(monkeypatch, conn)
    data = client.get("/api/draft-database/d1/board").get_json()
    assert data["viewer_in_draft"] is False


def test_board_snake_second_round_reverses(monkeypatch):
    conn = _FakeConn([_draft()], picks=_PICKS)
    client = _app(monkeypatch, conn)
    data = client.get("/api/draft-database/d1/board").get_json()
    r2 = [row for row in data["rounds"] if row["round"] == 2][0]
    # Snake: round 2 runs slot 2 -> slot 1, so roster 2's cell leads and
    # roster 1's (empty) slot trails.
    assert r2["cells"][0]["player_id"] == "p3"
    assert r2["cells"][1] is None


def test_board_missing_draft_404(monkeypatch):
    conn = _FakeConn([])
    client = _app(monkeypatch, conn)
    assert client.get("/api/draft-database/nope/board").status_code == 404


def test_board_headshot_fallback_when_no_espn_id(monkeypatch):
    conn = _FakeConn([_draft()], picks=_PICKS)
    client = _app(monkeypatch, conn)
    data = client.get("/api/draft-database/d1/board").get_json()
    cells = {c["player_id"]: c for row in data["rounds"]
             for c in row["cells"] if c}
    assert cells["p1"]["headshot"].endswith("/111.png")
    assert cells["p3"]["headshot"] is None  # initials fallback in the UI


# ── ADP basis ───────────────────────────────────────────────────────────────

def test_choose_adp_basis_threshold():
    assert ddb.choose_adp_basis(ddb.ADP_RECENT_MIN_DRAFTS) == "last_30_days"
    assert ddb.choose_adp_basis(ddb.ADP_RECENT_MIN_DRAFTS - 1) == "season_aggregate"
    assert ddb.choose_adp_basis(0) == "season_aggregate"


def test_adp_api_fallback_labels_season_aggregate(monkeypatch):
    adp_rows = [{"player_id": "p1", "avg_pick": 3.5, "std_pick": 1.2,
                 "avg_round": 1.0, "sample_size": 40, "drafts": 25}]
    conn = _FakeConn([], adp_rows=adp_rows, recent=2)
    client = _app(monkeypatch, conn)
    data = client.get("/api/draft-database/adp?season=2026").get_json()
    assert data["basis"] == "season_aggregate"
    assert "Season aggregate" in data["basis_line"]
    assert data["items"][0]["name"] == "Player One"
    assert data["items"][0]["adp"] == 3.5


# ── Grade summary builder ───────────────────────────────────────────────────

def _summary_picks():
    return [
        {"player_id": "p1", "pick_no": 40, "round": 4, "roster_id": "1", "team": "Team One"},
        {"player_id": "p2", "pick_no": 3, "round": 1, "roster_id": "2", "team": "Team Two"},
        {"player_id": "p3", "pick_no": 10, "round": 1, "roster_id": "1", "team": "Team One"},
    ]


def test_grade_summary_at_draft_family():
    adp = {"p1": 12.0, "p2": 25.0, "p3": 11.0}
    pos = {"p1": "WR", "p2": "RB", "p3": "QB"}
    s = build_grade_summary(_summary_picks(), adp, pos, num_teams=12,
                            draft_type="startup")
    ad = s["at_draft"]
    assert ad["family"] == "at_draft"
    # p1 fell 28 picks past ADP: the steal. p2 went 22 early: the bust.
    assert ad["biggest_steal"]["player_id"] == "p1"
    assert ad["biggest_bust"]["player_id"] == "p2"
    assert ad["best_draft"] is not None
    assert s["performance"] is None
    head = headline_superlatives(s)
    assert head["family"] == "at_draft"


def test_grade_summary_performance_family_wins_headline():
    adp = {"p1": 12.0, "p2": 25.0, "p3": 11.0}
    pos = {"p1": "WR", "p2": "RB", "p3": "QB"}
    # p2 (RB, drafted RB1) finishes as the only RB with a big PPG edge is
    # not how ranks work with one RB; give every position two players by
    # adding picks instead: simpler - p1 WR class rank moves via PPG.
    picks = _summary_picks() + [
        {"player_id": "p4", "pick_no": 50, "round": 5, "roster_id": "2", "team": "Team Two"},
    ]
    pos["p4"] = "WR"
    adp["p4"] = 55.0
    ppg = {"p1": 5.0, "p4": 20.0, "p2": 15.0, "p3": 18.0}
    s = build_grade_summary(picks, adp, pos, num_teams=12,
                            draft_type="startup", ppg_by_player=ppg)
    perf = s["performance"]
    assert perf is not None and perf["family"] == "performance"
    # p4 was the 2nd WR drafted but finished WR1: delta +1... p1 was WR1
    # drafted, finished WR2: delta -1. Steal = max delta.
    assert perf["biggest_steal"]["player_id"] == "p4"
    assert s["per_pick"]["p4"]["verdict"] in ("Hit", "As drafted", "Steal")
    head = headline_superlatives(s)
    assert head["family"] == "performance"


def test_grade_summary_rookie_pending_note():
    s = build_grade_summary(_summary_picks(), {"p1": 12.0}, {"p1": "WR"},
                            draft_type="rookie")
    assert s["performance"] is None
    assert s["performance_note"] == "Performance grades land after Week 18."
