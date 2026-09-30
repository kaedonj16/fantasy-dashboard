"""Regression tests for the trend-gate future-row fix (2026-09-30).

PR #2119 gated the trend metrics (xfp_trend / opportunity_trend) to each
player's MAX(as_of_date) row so an old build's forced 0.0 could not be
resurrected by the DISTINCT ON latest-non-null pick. But the provider
sync writers store their rows future-dated on purpose
(sync_nflverse_metrics: as_of_date = {season+1}-03-01;
sync_pff_advanced_metrics: {season+1}-02-15) so provider values win the
coalesce, and those rows never carry trend columns. The raw MAX therefore
always landed on a sync row, its NULL trend filtered out every qualified
player, and the breakout_check / waiver_wire boards (and the Usage Trend /
xFP Trend columns on populated boards) rendered empty despite healthy
snapshot data.

The fix defines the trend authority row as the newest SNAPSHOT row:
snapshot rows are marked by a non-null games value (the snapshot builder
only writes players with games >= 1; neither sync writer ever sets
games), capped at CURRENT_DATE so no future-dated row can speak. The
marker is load-bearing for past seasons: once a season ends, its sync
rows are in the past, so a date cap alone cannot exclude them.

The leaderboard fake below implements the query semantics in Python,
driven by the SQL the code actually generates (whether the trend gate's
MAX subquery carries the games marker / CURRENT_DATE cap), so these
tests fail on the pre-fix SQL and pass on the fixed SQL, mirroring
tests/test_computed_metrics_coalesce_gate.py.
"""

import re
from datetime import datetime, timezone

import pytest

pytest.importorskip("flask")

import data_building.advanced_metrics as am
import utils.season_qualification as sq
import utils.utils as uu

_ALL_COLS = [
    "player_id", "position", "season", "as_of_date", "games", "snap_share",
    "xfp_trend", "opportunity_trend", "target_share",
    "total_targets", "total_receptions", "total_carries", "total_pass_att",
    "completion_pct",
]


def _row(pid, pos, season, as_of, **kw):
    r = {"player_id": pid, "position": pos, "season": season,
         "as_of_date": as_of}
    r.update(kw)
    return r


# Season 2026 cast (WRs). Every player has a future-dated nflverse sync
# row (2027-03-01, games NULL, trends NULL) exactly as upsert_season
# writes it, plus snapshot rows from the daily build.
CAST_2026 = [
    # Riser: current snapshot carries real trend values.
    _row("p1", "WR", 2026, "2026-09-29", games=4, xfp_trend=24.0,
         opportunity_trend=12.0, target_share=0.28),
    _row("p1", "WR", 2026, "2027-03-01"),
    # Faller: negative xFP trend, strong usage trend.
    _row("p2", "WR", 2026, "2026-09-29", games=4, xfp_trend=-6.0,
         opportunity_trend=30.0, target_share=0.22),
    _row("p2", "WR", 2026, "2027-03-01"),
    # Stale-zero player (#2119): the newest snapshot row stores NULL
    # trends; an older snapshot row stores the forced 0.0 from the first
    # build weeks. The 0.0 must stay buried.
    _row("p3", "WR", 2026, "2026-09-15", games=2, xfp_trend=0.0,
         opportunity_trend=0.0),
    _row("p3", "WR", 2026, "2026-09-29", games=4),
    _row("p3", "WR", 2026, "2027-03-01"),
    # Sync-only player: no snapshot row at all, so no trend authority.
    _row("p4", "WR", 2026, "2027-03-01"),
]

# Past-season cast (2025). The season is over, so BOTH provider rows are
# in the past relative to today: a CURRENT_DATE cap alone cannot exclude
# them; only the games marker can.
CAST_2025 = [
    _row("p9", "WR", 2025, "2025-11-30", games=11, xfp_trend=15.0,
         opportunity_trend=8.0),
    _row("p9", "WR", 2025, "2026-02-15"),   # PFF sync row (games NULL)
    _row("p9", "WR", 2025, "2026-03-01"),   # nflverse sync row (games NULL)
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
    and makes a predicate not-true. (Same helper as
    tests/test_computed_metrics_coalesce_gate.py.)
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
        return eval(e, {"__builtins__": {}}, ns)
    except (TypeError, ZeroDivisionError):
        return None


class _LeaderboardFakeConn:
    """Implements get_metric_leaderboard's query semantics for a fixture.

    The trend gate is parsed from the generated SQL: the authority date is
    the max as_of_date over rows passing whichever restrictions the SQL
    carries (games marker, CURRENT_DATE cap), exactly like Postgres would
    evaluate the subquery.
    """

    def __init__(self, rows, season, metric):
        self.rows = rows
        self.season = season
        self.metric = metric
        self.main_sql = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        if "information_schema.columns" in sql:
            return _Res([{"column_name": c} for c in _ALL_COLS])
        if sql.startswith("SELECT 1 FROM player_advanced_metrics"):
            m = re.search(r"AND (\w+) IS NOT NULL AND \1 > 0", sql)
            col = m.group(1) if m else None
            hit = bool(col) and any(
                r.get("season") == self.season and (r.get(col) or 0) > 0
                for r in self.rows)
            return _Res([{"x": 1}] if hit else [])
        if "DISTINCT ON" in sql:
            self.main_sql.append(sql)
            return _Res(self._run(sql))
        return _Res([])

    def _run(self, sql):
        spec = am.LEADERBOARD_METRICS[self.metric]
        today = datetime.now(tz=timezone.utc).date().isoformat()
        gated = "MAX(cx.as_of_date)" in sql
        marker = "cx.games IS NOT NULL" in sql
        cap = "cx.as_of_date <= CURRENT_DATE" in sql

        by_player = {}
        for r in self.rows:
            if r.get("season") == self.season:
                by_player.setdefault(r["player_id"], []).append(r)

        out = []
        for pid, prows in by_player.items():
            desc = sorted(prows, key=lambda r: r["as_of_date"], reverse=True)
            # Coalesced season row (latest non-null per column).
            coal = {}
            for c in set().union(*(r.keys() for r in desc)):
                for r in desc:
                    if r.get(c) is not None:
                        coal[c] = r[c]
                        break
            vol = coal.get("games")
            if vol is None or vol <= 0:
                continue
            authority = None
            if gated:
                cand = [r["as_of_date"] for r in desc
                        if (not marker or r.get("games") is not None)
                        and (not cap or r["as_of_date"] <= today)]
                authority = max(cand) if cand else None
            for r in desc:  # DISTINCT ON: newest row passing WHERE + gate
                if spec.get("computed_sql"):
                    if not _eval_sql_expr(spec["computed_null"], r, coal,
                                          swap_m_to_coal=True):
                        continue
                    value = _eval_sql_expr(spec["computed_sql"], r, coal,
                                           swap_m_to_coal=True)
                else:
                    if r.get(self.metric) is None:
                        continue
                    if gated and r["as_of_date"] != authority:
                        continue
                    value = r.get(self.metric)
                if value is None:
                    continue
                out.append({
                    "player_id": pid,
                    "position": r.get("position"),
                    "games": r.get("games") if r.get("games") is not None
                             else vol,
                    "value": value,
                })
                break
        out.sort(key=lambda d: d["value"],
                 reverse=not spec.get("lower_better"))
        return out


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(uu, "load_players_index", dict)
    am._METRIC_LEADERBOARD_CACHE.clear()
    am._POSITION_RANKS_CACHE.clear()
    am._POSITION_BOUNDS_CACHE.clear()
    yield
    am._METRIC_LEADERBOARD_CACHE.clear()
    am._POSITION_RANKS_CACHE.clear()
    am._POSITION_BOUNDS_CACHE.clear()


def _board(monkeypatch, rows, season, metric):
    conn = _LeaderboardFakeConn(rows, season, metric)
    monkeypatch.setattr(am, "get_conn", lambda: conn)
    board = am.get_metric_leaderboard(metric, season=season)
    return {r["player_id"]: r["value"] for r in board}, conn


# --------------------------------------------------------------------------- #
# (a) Future-dated sync rows no longer silence trend boards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric,expected", [
    ("xfp_trend", {"p1": 24.0, "p2": -6.0}),
    ("opportunity_trend", {"p1": 12.0, "p2": 30.0}),
])
def test_trend_board_reads_snapshot_row_despite_future_sync_row(
        monkeypatch, metric, expected):
    board, _ = _board(monkeypatch, CAST_2026, 2026, metric)
    for pid, value in expected.items():
        assert board.get(pid) == pytest.approx(value)
    # The sync-only player has no snapshot authority row: stays absent.
    assert "p4" not in board


def test_trend_values_present_on_populated_nontrend_board(monkeypatch):
    # The screenshot scenario: a populated non-trend board (here
    # target_share; the Buy Low preset's FPOE/G primary is served from
    # weekly rows, but the merge mechanism is the same) whose rows show
    # dashes in the Usage Trend / xFP Trend columns. The page merges each
    # trend column's own leaderboard response into the board by
    # player_id, so the merged payload must carry the trend values.
    primary, _ = _board(monkeypatch, CAST_2026, 2026, "target_share")
    assert primary.get("p1") == pytest.approx(0.28)
    merged = {pid: {"value": v} for pid, v in primary.items()}
    for metric in ("xfp_trend", "opportunity_trend"):
        col, _ = _board(monkeypatch, CAST_2026, 2026, metric)
        for pid, value in col.items():
            if pid in merged:
                merged[pid][metric] = value
    assert merged["p1"]["xfp_trend"] == pytest.approx(24.0)
    assert merged["p1"]["opportunity_trend"] == pytest.approx(12.0)
    assert merged["p2"]["xfp_trend"] == pytest.approx(-6.0)


def test_past_season_trend_board_ignores_past_dated_sync_rows(monkeypatch):
    # Both sync rows predate today, so only the games marker (not the
    # CURRENT_DATE cap) can keep them from becoming the authority row.
    for metric, value in (("xfp_trend", 15.0), ("opportunity_trend", 8.0)):
        board, _ = _board(monkeypatch, CAST_2025, 2025, metric)
        assert board.get("p9") == pytest.approx(value)


# --------------------------------------------------------------------------- #
# (b) #2119 stays fixed: a stale 0.0 never resurrects over a current NULL
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", ["xfp_trend", "opportunity_trend"])
def test_newest_snapshot_null_keeps_stale_zero_buried(monkeypatch, metric):
    board, _ = _board(monkeypatch, CAST_2026, 2026, metric)
    # p3's newest snapshot row stores NULL; the older row's forced 0.0
    # must not reappear. (The healthy players on the same board prove the
    # gate is reading snapshot rows at all.)
    assert "p3" not in board
    assert "p1" in board


# --------------------------------------------------------------------------- #
# SQL contract: the gate's MAX is restricted to snapshot rows, capped today
# --------------------------------------------------------------------------- #
def test_trend_gate_sql_restricts_authority_to_snapshot_rows(monkeypatch):
    _, conn = _board(monkeypatch, CAST_2026, 2026, "xfp_trend")
    assert len(conn.main_sql) == 1
    sql = conn.main_sql[0]
    gate_sql = sql[sql.index("MAX(cx.as_of_date)"):]
    assert "cx.games IS NOT NULL" in gate_sql
    assert "cx.as_of_date <= CURRENT_DATE" in gate_sql


# --------------------------------------------------------------------------- #
# (c) Position-ranks supplement fills trends from the authority row
# --------------------------------------------------------------------------- #
class _RanksFakeConn:
    """Serves get_player_metric_ranks: position lookup, an empty main
    window query, and the supplement's SELECT * rows (pre-sorted
    player_id, as_of_date DESC as the SQL orders them)."""

    def __init__(self, rows, position_row):
        self.rows = rows
        self.position_row = position_row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        if "SELECT position, season FROM player_advanced_metrics" in sql:
            return _Res([self.position_row] if self.position_row else [])
        if "WITH snapshot AS" in sql:
            return _Res([])
        if "SELECT * FROM player_advanced_metrics" in sql:
            return _Res([dict(r) for r in self.rows])
        return _Res([])


def _ranks_rows(cast):
    rows = [dict(r) for r in cast]
    rows.sort(key=lambda r: (r["player_id"], r["as_of_date"]), reverse=True)
    # Stable sort by player keeps the as_of DESC order within each player.
    rows.sort(key=lambda r: r["player_id"])
    return rows


def test_ranks_supplement_fills_trends_from_snapshot_row(monkeypatch):
    monkeypatch.setattr(
        sq, "qualification_policy",
        lambda season, **kw: sq.QualificationPolicy(
            int(season), (1, 2, 3, 4), 4))
    conn = _RanksFakeConn(_ranks_rows(CAST_2026),
                          {"position": "WR", "season": 2026})
    monkeypatch.setattr(am, "get_conn", lambda: conn)

    res = am.get_player_metric_ranks("p1", season=2026)
    ranks = res["ranks"]
    # p1 (24.0) outranks p2 (-6.0) on xFP trend; p2 (30.0) outranks
    # p1 (12.0) on usage trend.
    assert ranks.get("xfp_trend") == 1
    assert ranks.get("opportunity_trend") == 2

    # p3's newest snapshot row stores NULL trends: no trend rank, even
    # though an older snapshot row stores 0.0 (served from the position
    # cache the first call populated).
    res3 = am.get_player_metric_ranks("p3", season=2026)
    assert "xfp_trend" not in res3["ranks"]
    assert "opportunity_trend" not in res3["ranks"]
