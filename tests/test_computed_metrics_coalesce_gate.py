"""Regression tests for the computed-metric coalesce + snap-gate fix.

Audit findings (2026-09-30):

1. Five computed metrics (passing_epa_per_att, explosive_run_rate,
   intended_air_yards_per_game, unrealized_air_yards,
   unrealized_air_yards_per_game) were permanently empty: their components
   live on different snapshot rows (volume totals on the daily base row,
   EPA/NGS values on provider sync rows), and get_metric_leaderboard
   evaluated computed_sql against a single physical row, so one component
   was always NULL. The fix evaluates computed_sql over a per-player
   coalesced season row (latest non-null value per component column).

2. The snap-share efficiency gate sat on the physical row inside the
   DISTINCT ON query, so a player whose current row failed the gate was
   not dropped: they silently fell back to their most recent passing
   (older) row and showed stale values. The fix gates on the player's
   coalesced (latest non-null) snap_share.

The fake connection below implements the leaderboard query semantics in
Python, driven by the SQL the code actually generates (which alias the
gate reads, whether the coalesce join exists), so these tests fail on the
pre-fix SQL and pass on the fixed SQL. A DuckDB-backed test at the bottom
additionally executes the real generated SQL end to end when duckdb is
installed (it is not a CI dependency, so it skips there).
"""

import re

import pytest

import data_building.advanced_metrics as am


_ALL_COLS = [
    "player_id", "position", "season", "as_of_date", "games", "snap_share",
    "passing_epa", "total_pass_att", "catch_rate", "total_targets",
    "total_receptions", "total_carries", "explosive_runs_10_plus",
    "ngs_avg_intended_air_yards", "yards_per_reception", "completion_pct",
    "total_tds", "total_snaps", "total_rush_tds", "total_rec_tds",
    "total_pass_tds", "total_touches",
]


def _row(pid, pos, as_of, **kw):
    r = {"player_id": pid, "position": pos, "season": 2026, "as_of_date": as_of}
    r.update(kw)
    return r


# QB whose passing_epa sits on the nflverse sync row (2027-03-01) while
# total_pass_att sits on the daily base row: no single row has both.
SPLIT_QB = [
    _row("q1", "QB", "2026-09-29", games=4, snap_share=0.95, total_pass_att=140),
    _row("q1", "QB", "2027-03-01", passing_epa=21.0),
]
# RB whose explosive_runs_10_plus sits on the sync row, carries on the base row.
SPLIT_RB = [
    _row("r1", "RB", "2026-09-29", games=4, snap_share=0.60, total_carries=60),
    _row("r1", "RB", "2027-03-01", explosive_runs_10_plus=9),
]
# WR whose NGS intended air yards sit on the sync row, receiving volume on base.
SPLIT_WR = [
    _row("w4", "WR", "2027-03-01", ngs_avg_intended_air_yards=12.5),
    _row("w4", "WR", "2026-09-29", games=4, total_targets=40,
         total_receptions=25, yards_per_reception=11.0),
]
# Snap-gate rollback cast (catch_rate, an efficiency metric):
#  w1: current row fails the gate (snap_share 0.01, the broken scale from
#      the snap-share bug), older row passes via NULL with a stale 1.000.
#  w2: healthy control, must show 0.80 under both gate designs.
#  w3: latest row snap NULL, older row 0.30: coalesced gate passes and the
#      current value (0.70) shows.
GATE_CAST = [
    _row("w1", "WR", "2026-09-10", games=2, catch_rate=1.000, total_targets=10),
    _row("w1", "WR", "2026-09-29", games=4, snap_share=0.01,
         catch_rate=0.917, total_targets=24),
    _row("w2", "WR", "2026-09-29", games=4, snap_share=0.75,
         catch_rate=0.80, total_targets=30),
    _row("w3", "WR", "2026-09-10", games=2, snap_share=0.30,
         catch_rate=0.60, total_targets=12),
    _row("w3", "WR", "2026-09-29", games=4, catch_rate=0.70, total_targets=26),
]


class _Res:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


def _eval_sql_expr(expr, row, coal, swap_m_to_coal=False):
    """Evaluate the small SQL subset used by the catalog's computed_sql /
    computed_null expressions against dict contexts (m=row, c=coalesced).

    Mirrors SQL NULL semantics: a NULL operand poisons arithmetic to NULL
    and makes a predicate not-true.
    """
    e = expr
    e = re.sub(r"::[A-Za-z]+", "", e)
    e = re.sub(r"\bc\.(\w+)", r"C.get('\1')", e)
    e = re.sub(r"\bm\.(\w+)", r"M.get('\1')", e)
    e = e.replace("IS NOT NULL", "is not None").replace("IS NULL", "is None")
    e = re.sub(r"\bAND\b", "and", e)
    e = re.sub(r"\bOR\b", "or", e)
    e = re.sub(r"\bNOT\b", "not", e)
    ns = {
        "M": coal if swap_m_to_coal else row,
        "C": coal if coal is not None else {},
        "COALESCE": lambda *a: next((x for x in a if x is not None), None),
        "NULLIF": lambda a, b: None if a == b else a,
        "ROUND": lambda a, n=0: (round(a, n) if a is not None else None),
    }
    try:
        return eval(e, {"__builtins__": {}}, ns)  # noqa: S307 - test-only, fixed catalog strings
    except (TypeError, ZeroDivisionError):
        return None


class _LeaderboardFakeConn:
    """Implements get_metric_leaderboard's query semantics for a fixture.

    Behavior is driven by the generated SQL text: the coalesced row exists
    only when the SQL carries the ARRAY_AGG join, and the snap gate reads
    whichever alias (m./c.) the SQL names, exactly like Postgres would.
    """

    def __init__(self, rows, metric):
        self.rows = rows
        self.metric = metric
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, tuple(params or ())))
        if "information_schema.columns" in sql:
            return _Res([{"column_name": c} for c in _ALL_COLS])
        if sql.startswith("SELECT 1 FROM player_advanced_metrics"):
            m = re.search(r"AND (\w+) IS NOT NULL AND \1 > 0", sql)
            col = m.group(1) if m else None
            hit = bool(col) and any((r.get(col) or 0) > 0 for r in self.rows)
            return _Res([{"x": 1}] if hit else [])
        if "DISTINCT ON" in sql:
            return _Res(self._run(sql))
        return _Res([])

    def _run(self, sql):
        spec = am.LEADERBOARD_METRICS[self.metric]
        vol_col = (spec.get("min_vol") or {}).get("col") or "games"
        has_coal = "ARRAY_AGG" in sql
        gate_on_coal = "(c.snap_share" in sql
        # The main WHERE is the last one before the gate's season clause
        # (the vol/coalesce joins carry their own WHEREs earlier in the SQL).
        where_sql = sql.split(" AND m.season = ")[0].rsplit(" WHERE ", 1)[1]

        by_player = {}
        for r in self.rows:
            if r.get("season") == 2026:
                by_player.setdefault(r["player_id"], []).append(r)

        out = []
        for pid, prows in by_player.items():
            prows.sort(key=lambda r: r["as_of_date"], reverse=True)
            coal = None
            if has_coal:
                coal = {}
                cols = set().union(*(r.keys() for r in prows))
                for c in cols:
                    for r in prows:  # latest first: first non-null wins
                        if r.get(c) is not None:
                            coal[c] = r[c]
                            break
            vols = [r[vol_col] for r in prows if r.get(vol_col) is not None]
            vol = max(vols) if vols else None
            for r in prows:  # DISTINCT ON: latest row passing WHERE + gate
                if spec.get("computed_sql"):
                    ok = _eval_sql_expr(where_sql, r, coal)
                else:
                    ok = r.get(self.metric) is not None
                if not ok:
                    continue
                if spec.get("efficiency"):
                    snap = (coal or {}).get("snap_share") if gate_on_coal \
                        else r.get("snap_share")
                    if not (snap is None
                            or snap >= am._MIN_SNAP_FOR_EFFICIENCY):
                        continue
                if vol is None or vol <= 0:
                    continue
                if spec.get("computed_sql"):
                    value = _eval_sql_expr(
                        spec["computed_sql"], r, coal, swap_m_to_coal=has_coal)
                else:
                    value = r.get(self.metric)
                if value is None:
                    continue
                out.append({
                    "player_id": pid,
                    "position": r.get("position"),
                    "games": r.get("games") if r.get("games") is not None
                             else (coal or {}).get("games"),
                    "vol": r.get(vol_col) if r.get(vol_col) is not None else vol,
                    "value": value,
                })
                break
        out.sort(key=lambda d: d["value"], reverse=True)
        return out


@pytest.fixture(autouse=True)
def _no_player_index(monkeypatch):
    monkeypatch.setattr("utils.utils.load_players_index", lambda: {})
    cache = getattr(am, "_METRIC_LEADERBOARD_CACHE", None)
    if cache is not None:
        cache.clear()


def _board(monkeypatch, rows, metric):
    conn = _LeaderboardFakeConn(rows, metric)
    monkeypatch.setattr(am, "get_conn", lambda: conn)
    board = am.get_metric_leaderboard(metric, season=2026)
    return {r["player_id"]: r["value"] for r in board}, conn


# --------------------------------------------------------------------------- #
# Finding 1: computed metrics over split snapshot rows
# --------------------------------------------------------------------------- #
def test_passing_epa_per_att_from_split_rows(monkeypatch):
    board, _ = _board(monkeypatch, SPLIT_QB, "passing_epa_per_att")
    assert board.get("q1") == pytest.approx(21.0 / 140)


def test_explosive_run_rate_from_split_rows(monkeypatch):
    board, _ = _board(monkeypatch, SPLIT_RB, "explosive_run_rate")
    assert board.get("r1") == pytest.approx(9.0 / 60)


def test_unrealized_air_yards_from_split_rows(monkeypatch):
    board, _ = _board(monkeypatch, SPLIT_WR, "unrealized_air_yards")
    # 12.5 intended AY/target * 40 targets - 11.0 YPR * 25 receptions.
    assert board.get("w4") == pytest.approx(12.5 * 40 - 11.0 * 25)


def test_intended_air_yards_per_game_from_split_rows(monkeypatch):
    board, _ = _board(monkeypatch, SPLIT_WR, "intended_air_yards_per_game")
    assert board.get("w4") == pytest.approx(12.5 * 40 / 4)


# --------------------------------------------------------------------------- #
# Finding 2: efficiency gate on coalesced snap_share, no stale fallback
# --------------------------------------------------------------------------- #
def test_snap_gate_excludes_player_whose_latest_share_fails(monkeypatch):
    board, _ = _board(monkeypatch, GATE_CAST, "catch_rate")
    # Pre-fix, w1 appeared with the stale 1.000 from the older passing row.
    assert "w1" not in board


def test_snap_gate_keeps_healthy_and_coalesced_passing_players(monkeypatch):
    board, _ = _board(monkeypatch, GATE_CAST, "catch_rate")
    assert board.get("w2") == pytest.approx(0.80)
    # w3's latest row has NULL snap_share; the coalesced 0.30 passes and the
    # value shown is the current row's 0.70, not the older row's 0.60.
    assert board.get("w3") == pytest.approx(0.70)


# --------------------------------------------------------------------------- #
# SQL contract: the generated query really is the coalesced one
# --------------------------------------------------------------------------- #
def _captured_main_sql(monkeypatch, rows, metric):
    _, conn = _board(monkeypatch, rows, metric)
    main = [sql for sql, _ in conn.calls if "DISTINCT ON" in sql]
    assert main, "leaderboard never issued its main query"
    return main[0], [p for sql, p in conn.calls if "DISTINCT ON" in sql][0]


def test_computed_sql_evaluated_on_coalesced_alias(monkeypatch):
    sql, params = _captured_main_sql(monkeypatch, SPLIT_QB, "passing_epa_per_att")
    assert "ARRAY_AGG(cx.passing_epa" in sql
    assert "ARRAY_AGG(cx.total_pass_att" in sql
    assert "c.passing_epa::float / NULLIF(c.total_pass_att, 0) AS value" in sql
    assert "c.passing_epa IS NOT NULL" in sql
    # Placeholder order must match parameter order: vol season, coalesce
    # season, gate season, snap threshold, limit.
    assert sql.count("%s") == len(params)
    assert params[0] == params[1] == params[2] == 2026
    assert params[3] == am._MIN_SNAP_FOR_EFFICIENCY
    assert params[-1] == 500


def test_efficiency_gate_reads_coalesced_snap_share(monkeypatch):
    sql, _ = _captured_main_sql(monkeypatch, GATE_CAST, "catch_rate")
    assert "(c.snap_share IS NULL OR c.snap_share >= %s)" in sql
    assert "m.snap_share >= %s" not in sql
    # Plain metrics keep reading their value from the physical row.
    assert "m.catch_rate AS value" in sql


def test_plain_non_efficiency_metric_gets_no_coalesce_join(monkeypatch):
    rows = [_row("r9", "RB", "2026-09-29", games=4, total_carries=50,
                 explosive_runs_10_plus=7)]
    sql, _ = _captured_main_sql(monkeypatch, rows, "explosive_runs_10_plus")
    assert "ARRAY_AGG" not in sql
    assert "snap_share >= %s" not in sql
    assert "m.explosive_runs_10_plus AS value" in sql


# --------------------------------------------------------------------------- #
# Real-SQL execution (DuckDB stand-in for Postgres) when available
# --------------------------------------------------------------------------- #
def test_generated_sql_executes_against_real_engine(monkeypatch):
    duckdb = pytest.importorskip("duckdb")

    class _Res2:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

    class _DuckConn:
        def __init__(self, con):
            self.con = con

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=None):
            if "information_schema.columns" in sql:
                cols = [r[0] for r in self.con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='player_advanced_metrics'").fetchall()]
                return _Res2([{"column_name": c} for c in cols])
            cur = self.con.execute(sql.replace("%s", "?"), list(params or ()))
            names = [d[0] for d in cur.description]
            return _Res2([dict(zip(names, r)) for r in cur.fetchall()])

    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE player_advanced_metrics ("
        "player_id VARCHAR, position VARCHAR, season INTEGER, "
        "as_of_date DATE, games DOUBLE, snap_share DOUBLE, "
        "passing_epa DOUBLE, total_pass_att DOUBLE, catch_rate DOUBLE, "
        "total_targets DOUBLE, total_receptions DOUBLE, total_carries DOUBLE, "
        "explosive_runs_10_plus DOUBLE, ngs_avg_intended_air_yards DOUBLE, "
        "yards_per_reception DOUBLE, completion_pct DOUBLE, total_tds DOUBLE)"
    )
    fixtures = SPLIT_QB + SPLIT_WR + GATE_CAST
    cols = ["player_id", "position", "season", "as_of_date", "games",
            "snap_share", "passing_epa", "total_pass_att", "catch_rate",
            "total_targets", "total_receptions", "total_carries",
            "explosive_runs_10_plus", "ngs_avg_intended_air_yards",
            "yards_per_reception", "completion_pct", "total_tds"]
    for r in fixtures:
        con.execute(
            f"INSERT INTO player_advanced_metrics ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})",
            [r.get(c) for c in cols],
        )
    monkeypatch.setattr(am, "get_conn", lambda: _DuckConn(con))

    epa = {r["player_id"]: r["value"]
           for r in am.get_metric_leaderboard("passing_epa_per_att", season=2026)}
    assert epa.get("q1") == pytest.approx(0.15)
    unrl = {r["player_id"]: r["value"]
            for r in am.get_metric_leaderboard("unrealized_air_yards", season=2026)}
    assert unrl.get("w4") == pytest.approx(225.0)
    cr = {r["player_id"]: r["value"]
          for r in am.get_metric_leaderboard("catch_rate", season=2026)}
    assert "w1" not in cr
    assert cr.get("w2") == pytest.approx(0.80)
    assert cr.get("w3") == pytest.approx(0.70)
