"""Regression tests: trend gates must ignore future-dated provider rows.

Bug (2026-09-30): PR #2119 restricted xfp_trend / opportunity_trend to the
player's newest snapshot row so an old build's forced 0.0 cannot resurrect
when the newest build stores NULL. But scripts/sync_nflverse_metrics.py
stamps its season rows March 1 of the following year (so provider values
win the coalesce), and those rows never carry trend columns. "Newest" was
therefore always the sync row, its trend always NULL, and every trend
leaderboard (Is This Breakout Real?, Waiver Wire) came back empty. The
position-ranks supplement had the twin bug in its Python merge.

Fix: "newest" means the newest row dated today or earlier, in both places:
the SQL gate caps its MAX subquery at CURRENT_DATE, and the ranks merge
precomputes the newest non-future date per player. The stale-zero guard
itself is unchanged: when the newest real build stores NULL, older values
still may not fill in (player rb2 below pins that intent).
"""

import re
from datetime import date, timedelta

import pytest

import data_building.advanced_metrics as am

_TODAY = date.today()
_NEW = _TODAY
_OLD = _TODAY - timedelta(days=7)
_SYNC = date(2027, 3, 1)  # nflverse season-sync stamp for the 2026 season

_COLS = [
    "player_id", "position", "season", "as_of_date", "games", "snap_share",
    "xfp_trend", "opportunity_trend", "total_targets", "total_receptions",
    "total_carries", "total_pass_att", "completion_pct",
]


def _row(pid, as_of, **kw):
    r = {c: None for c in _COLS}
    r.update({"player_id": pid, "position": "RB", "season": 2026,
              "as_of_date": as_of})
    r.update(kw)
    return r


def _lb_rows():
    return [
        # rb1: healthy trend on the newest build; the future sync row must
        # not silence it, and the older build's forced 0.0 must not win.
        _row("rb1", _OLD, games=3, xfp_trend=0.0, opportunity_trend=0.0),
        _row("rb1", _NEW, games=4, xfp_trend=0.25, opportunity_trend=0.10,
             total_carries=60, total_targets=20, total_receptions=15),
        _row("rb1", _SYNC, snap_share=0.62),
        # rb2: newest build deliberately stores NULL (no signal yet); the
        # older build's 0.30 must stay buried under both gate designs.
        _row("rb2", _OLD, games=3, xfp_trend=0.30, opportunity_trend=0.22),
        _row("rb2", _NEW, games=4),
        _row("rb2", _SYNC),
        # rb3: no sync row at all; plain control.
        _row("rb3", _NEW, games=4, xfp_trend=-0.05, opportunity_trend=-0.02,
             total_carries=50),
    ]


class _SemanticsConn:
    """Fake conn that runs the leaderboard's trend-gate semantics in Python,
    driven by the SQL the code actually generates (whether the MAX gate is
    present and whether it is capped at CURRENT_DATE), so the behavioral
    tests fail on the pre-fix SQL and pass on the fixed SQL."""

    def __init__(self, rows):
        self.rows = rows
        self.main_sql = None
        self._rows = []

    def execute(self, sql, params=None):
        if "information_schema" in sql:
            self._rows = [{"column_name": c} for c in _COLS]
        elif "DISTINCT ON" in sql:
            self.main_sql = sql
            self._rows = self._run_main(sql, params or ())
        elif "SELECT 1 FROM player_advanced_metrics" in sql:
            season = params[0]
            self._rows = ([{"present": 1}] if any(
                r["season"] == season and (r["games"] or 0) > 0
                for r in self.rows) else [])
        else:
            self._rows = []
        return self

    def _run_main(self, sql, params):
        metric = re.search(r"WHERE m\.(\w+) IS NOT NULL", sql).group(1)
        gated = "MAX(cx.as_of_date)" in sql
        capped = "cx.as_of_date <= CURRENT_DATE" in sql
        season, pos, limit = params[1], params[2], params[-1]
        today_iso = _TODAY.isoformat()
        by_player = {}
        for r in self.rows:
            if r["season"] == season and r["position"] == pos:
                by_player.setdefault(r["player_id"], []).append(r)
        out = []
        for pid, rows in by_player.items():
            games_vals = [r["games"] for r in rows if r["games"] is not None]
            vol = max(games_vals) if games_vals else None
            if vol is None or vol <= 0:
                continue  # season has vol data: the v.vol > 0 gate applies
            dates = [r["as_of_date"] for r in rows]
            if capped:
                dates = [d for d in dates if str(d)[:10] <= today_iso]
            newest = max(dates) if dates else None
            cands = [r for r in rows
                     if r[metric] is not None
                     and (not gated or r["as_of_date"] == newest)]
            if not cands:
                continue
            pick = max(cands, key=lambda r: r["as_of_date"])
            out.append({
                "player_id": pid, "position": pos,
                "games": pick["games"] if pick["games"] is not None else vol,
                "value": float(pick[metric]),
                "ctx_receptions": None, "ctx_targets": None,
                "ctx_carries": None, "ctx_attempts": None,
                "ctx_completions": None,
            })
        out.sort(key=lambda r: r["value"], reverse=True)
        return out[:limit]

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def lb_conn(monkeypatch):
    conn = _SemanticsConn(_lb_rows())
    monkeypatch.setattr(am, "get_conn", lambda: conn)
    am._METRIC_LEADERBOARD_CACHE.clear()
    yield conn
    am._METRIC_LEADERBOARD_CACHE.clear()


@pytest.mark.parametrize("metric", ["xfp_trend", "opportunity_trend"])
def test_trend_gate_sql_caps_newest_at_current_date(lb_conn, metric):
    am.get_metric_leaderboard(metric, position="RB", season=2026)
    assert lb_conn.main_sql is not None
    assert "MAX(cx.as_of_date)" in lb_conn.main_sql
    assert "cx.as_of_date <= CURRENT_DATE" in lb_conn.main_sql


@pytest.mark.parametrize("metric,expected", [
    ("xfp_trend", {"rb1": 0.25, "rb3": -0.05}),
    ("opportunity_trend", {"rb1": 0.10, "rb3": -0.02}),
])
def test_trend_leaderboard_ignores_future_sync_row(lb_conn, metric, expected):
    board = am.get_metric_leaderboard(metric, position="RB", season=2026)
    got = {r["player_id"]: r["value"] for r in board}
    # rb1 speaks from the newest real build, rb3 is the control, and rb2
    # stays off the board: its newest build is NULL, so the stale 0.30 from
    # the older build must not resurrect.
    assert got == expected


# --------------------------------------------------------------------------- #
# Position-ranks supplement: same rule in the Python merge
# --------------------------------------------------------------------------- #

class _Res:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _RanksConn:
    """Serves get_player_metric_ranks: an empty window-query result (so all
    ranks come from the supplement) plus the supplement's raw season rows,
    ordered player_id, as_of_date DESC exactly as its query asks."""

    def __init__(self, raw_rows):
        self.raw_rows = raw_rows

    def execute(self, sql, params=None):
        if "SELECT position, season" in sql:
            return _Res([{"position": "RB", "season": 2026}])
        if "WITH snapshot AS" in sql:
            return _Res([])
        if "SELECT * FROM player_advanced_metrics" in sql:
            return _Res(self.raw_rows)
        return _Res([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _srow(pid, as_of, **kw):
    r = {c: None for c in _COLS}
    r.update({"player_id": pid, "position": "RB", "season": 2026,
              "as_of_date": as_of, "total_rec_tds": None,
              "yards_per_reception": None, "total_routes": None,
              "explosive_runs_10_plus": None, "avoided_tackles": None,
              "total_touches": None, "total_snaps": None, "passing_epa": None})
    r.update(kw)
    return r


def _raw_rows():
    return [
        _srow("rb1", _SYNC, snap_share=0.62),
        _srow("rb1", _NEW, games=4, xfp_trend=0.25, opportunity_trend=0.10,
              total_carries=60, total_targets=20, total_receptions=15,
              total_touches=80, total_snaps=200),
        _srow("rb1", _OLD, games=3, xfp_trend=0.0, opportunity_trend=0.0,
              total_carries=40),
        _srow("rb2", _SYNC),
        _srow("rb2", _NEW, games=4, total_carries=55),
        _srow("rb2", _OLD, games=3, xfp_trend=0.30, opportunity_trend=0.22,
              total_carries=38),
        _srow("rb3", _NEW, games=4, xfp_trend=-0.05, opportunity_trend=-0.02,
              total_carries=50),
    ]


@pytest.fixture()
def ranks_conn(monkeypatch):
    conn = _RanksConn(_raw_rows())
    monkeypatch.setattr(am, "get_conn", lambda: conn)
    am._POSITION_RANKS_CACHE.clear()
    am._POSITION_BOUNDS_CACHE.clear()
    yield conn
    am._POSITION_RANKS_CACHE.clear()
    am._POSITION_BOUNDS_CACHE.clear()


def test_ranks_supplement_trends_ignore_future_sync_row(ranks_conn):
    rb1 = am.get_player_metric_ranks("rb1", season=2026)["ranks"]
    # rb1's trend comes from the newest real build (0.25), ranking first.
    assert rb1.get("xfp_trend") == 1
    assert rb1.get("opportunity_trend") == 1

    rb3 = am.get_player_metric_ranks("rb3", season=2026)["ranks"]
    assert rb3.get("xfp_trend") == 2

    # rb2's newest build stores NULL: the older build's 0.30 stays buried.
    rb2 = am.get_player_metric_ranks("rb2", season=2026)["ranks"]
    assert "xfp_trend" not in rb2
    assert "opportunity_trend" not in rb2
