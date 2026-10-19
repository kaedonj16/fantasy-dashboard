"""YPTPA (yards per team pass attempt) — season finalize, weekly writer,
week-range aggregation, registry contract, and snapshot persistence.

Definition: a receiver's receiving yards divided by his team's pass
attempts over the weeks in his own weekly series (games-played basis —
bye / inactive weeks are absent from the series and count in neither
the numerator nor the denominator). Team attempts pool every player's
stored pass_att on the team (QB attempts plus trick-play attempts),
which reproduces Sleeper's TEAM_ aggregate rows exactly.

Golden values are fixture expectations taken from the real-data check
(2025 Sleeper weekly caches, the files build_weekly_metrics consumes):
JSN 1793 rec yds / 481 SEA att = 3.73, Puka 1715 / 565 LA att (games
played) = 3.04, Zay Flowers 1211 / 422 BAL att = 2.87.
"""

import re
import sqlite3

import pytest


def _week(week, **kw):
    """A weekly-series row shaped like get_weekly_series_by_player's
    dict rows (values arrive as Decimal/float from psycopg)."""
    base = {"week": week, "position": "WR", "ppr_pts": 10.0, "snaps": 50.0,
            "snap_pct": 80.0, "targets": 5.0, "touches": 5.0,
            "rz_targets": 0.0, "rz_carries": 0.0, "rec_tds": 0.0,
            "rush_tds": 0.0, "pass_tds": 0.0, "rec_yards": 0.0,
            "pass_att": 0.0, "receptions": 0.0, "carries": 0.0}
    base.update(kw)
    return base


def _run_finalize(monkeypatch, series, metrics, usage):
    import data_building.weekly_metrics as wm
    from data_building import advanced_metrics as am

    monkeypatch.setattr(wm, "get_weekly_series_by_player",
                        lambda season, through_week: series)
    am.finalize_weekly_series_metrics(metrics, usage, 2025, 18)
    return {m["player_id"]: m for m in metrics}


# --------------------------------------------------------------------------- #
# Season finalize
# --------------------------------------------------------------------------- #
def test_finalize_yptpa_pooling_rules(monkeypatch):
    series = {
        # KC QB: attempts form the team denominator (30 + 40), plus a
        # week-3 row that must NOT count for teammates absent that week.
        "1001": [_week(1, position="QB", pass_att=30.0),
                 _week(2, position="QB", pass_att=40.0),
                 _week(3, position="QB", pass_att=35.0)],
        # KC WR: 90 + 110 rec yards, and a trick-play attempt in week 2
        # that counts toward team attempts (40 + 1). No week-3 row:
        # he was inactive, so week 3 is excluded from his denominator.
        "1002": [_week(1, rec_yards=90.0),
                 _week(2, rec_yards=110.0, pass_att=1.0)],
        # KC RB: receiving-only numerator over the same team weeks.
        "1004": [_week(1, position="RB", rec_yards=10.0),
                 _week(2, position="RB", rec_yards=20.0)],
        # NYJ WR: no QB rows anywhere and no attempts of his own, so the
        # team attempted zero passes in his weeks -> None, never 0.0.
        "1003": [_week(1, rec_yards=50.0)],
    }
    metrics = [{"player_id": pid, "position": pos} for pid, pos in
               [("1001", "QB"), ("1002", "WR"), ("1004", "RB"), ("1003", "WR")]]
    usage = [{"id": "1001", "team": "KC"}, {"id": "1002", "team": "KC"},
             {"id": "1004", "team": "KC"}, {"id": "1003", "team": "NYJ"}]
    by_pid = _run_finalize(monkeypatch, series, metrics, usage)

    # Denominator: (30) + (40 + 1 trick-play) = 71; week 3 excluded.
    assert by_pid["1002"]["team_pass_att_gp"] == 71.0
    assert by_pid["1002"]["yards_per_team_pass_att"] == pytest.approx(200.0 / 71.0)
    assert by_pid["1004"]["yards_per_team_pass_att"] == pytest.approx(30.0 / 71.0)
    # RB/WR/TE only: the QB gets no YPTPA even though his attempts pooled.
    assert by_pid["1001"]["yards_per_team_pass_att"] is None
    assert by_pid["1001"]["team_pass_att_gp"] is None
    # Zero denominator -> None.
    assert by_pid["1003"]["yards_per_team_pass_att"] is None
    assert by_pid["1003"]["team_pass_att_gp"] is None


def test_finalize_yptpa_golden_values(monkeypatch):
    # (player, team, rec yards split, QB att split, golden YPTPA)
    golden = [
        ("9488", "SEA", (1000.0, 793.0), (300.0, 181.0), 3.73),   # JSN
        ("9493", "LA", (1000.0, 715.0), (300.0, 265.0), 3.04),    # Puka (GP)
        ("9997", "BAL", (700.0, 511.0), (200.0, 222.0), 2.87),    # Flowers
    ]
    series, metrics, usage = {}, [], []
    for i, (pid, team, rec_split, att_split, _g) in enumerate(golden):
        qb = f"7{i:03d}"
        series[qb] = [_week(1, position="QB", pass_att=att_split[0]),
                      _week(2, position="QB", pass_att=att_split[1])]
        series[pid] = [_week(1, rec_yards=rec_split[0]),
                       _week(2, rec_yards=rec_split[1])]
        metrics.append({"player_id": pid, "position": "WR"})
        usage += [{"id": qb, "team": team}, {"id": pid, "team": team}]
    by_pid = _run_finalize(monkeypatch, series, metrics, usage)
    for pid, _team, rec_split, att_split, want in golden:
        got = by_pid[pid]["yards_per_team_pass_att"]
        assert got == pytest.approx(sum(rec_split) / sum(att_split))
        assert round(got, 2) == want
        assert by_pid[pid]["team_pass_att_gp"] == sum(att_split)


# --------------------------------------------------------------------------- #
# Weekly writer + week-range aggregation
# --------------------------------------------------------------------------- #
class _FakeConn:
    """psycopg-shaped stand-in: records executes, context manager."""

    def __init__(self):
        self.executes = []

    def execute(self, sql, params=None):
        self.executes.append((sql, params))
        return self

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _weekly_series():
    return {
        "1001": [_week(1, position="QB", pass_att=30.0),
                 _week(2, position="QB", pass_att=40.0)],
        "1002": [_week(1, rec_yards=90.0),
                 _week(2, rec_yards=110.0, pass_att=1.0)],
        "1003": [_week(1, rec_yards=50.0)],
    }


def _capture_weekly_rows(monkeypatch):
    import data_building.weekly_metrics as wm
    from data_building import advanced_metrics as am

    monkeypatch.setattr(wm, "get_weekly_series_by_player",
                        lambda season, through_week: _weekly_series())
    monkeypatch.setattr(am, "init_weekly_advanced_metrics_db", lambda: None)
    conn = _FakeConn()
    monkeypatch.setattr(am, "get_conn", lambda: conn)
    usage = [{"id": "1001", "team": "KC"}, {"id": "1002", "team": "KC"},
             {"id": "1003", "team": "NYJ"}]

    n = am.upsert_yptpa_weekly_metrics(2025, 2, usage)
    # Only RB/WR/TE player-weeks: KC WR x2 + NYJ WR x1 (QB rows skipped).
    assert n == 3
    rows = {}
    for sql, params in conn.executes:
        assert "INSERT INTO player_weekly_advanced_metrics" in sql
        # Coexistence with the nflverse writer: the upsert sets ONLY
        # position + this writer's two columns, each COALESCE-guarded.
        set_clause = sql.split("DO UPDATE SET", 1)[1]
        set_cols = {c.strip().split("=")[0].strip()
                    for c in set_clause.split("),")}
        assert set_cols <= {"position", "yards_per_team_pass_att",
                            "w_team_pass_att"}
        assert "expected_ppr" not in sql and "w_targets" not in set_cols
        pid, _season, week, _pos, value, weight = params
        rows[(pid, week)] = (value, weight)
    # Week values: rec_yards / that week's team attempts (incl. trick play).
    assert rows[("1002", 1)] == (pytest.approx(90.0 / 30.0), 30.0)
    assert rows[("1002", 2)] == (pytest.approx(110.0 / 41.0), 41.0)
    # Zero team attempts: value None (absent), weight 0.
    assert rows[("1003", 1)] == (None, 0.0)
    return rows


def test_weekly_writer_rows_and_coexistence(monkeypatch):
    _capture_weekly_rows(monkeypatch)


def test_week_range_weighted_mean_is_exact_ratio_of_sums(monkeypatch):
    from data_building import advanced_metrics as am

    rows = _capture_weekly_rows(monkeypatch)
    value_sql, min_col = am._adv_weekly_agg_sql("yards_per_team_pass_att")
    assert min_col == "SUM(w_team_pass_att)"
    # Run the range aggregation the leaderboard generates (dialect-only
    # ::float cast stripped) against the writer's stored rows in sqlite.
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE player_weekly_advanced_metrics "
               "(player_id TEXT, yards_per_team_pass_att REAL, "
               " w_team_pass_att REAL)")
    for (pid, _week_no), (value, weight) in rows.items():
        if pid == "1002":
            db.execute("INSERT INTO player_weekly_advanced_metrics "
                       "VALUES (?, ?, ?)", (pid, value, weight))
    got = db.execute(
        f"SELECT {value_sql.replace('::float', '')} "
        "FROM player_weekly_advanced_metrics").fetchone()[0]
    # Exact ratio of sums: (90 + 110) / (30 + 41).
    assert got == pytest.approx(200.0 / 71.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Registry / dispatch contract
# --------------------------------------------------------------------------- #
def test_registry_and_range_contract():
    import inspect

    from data_building import advanced_metrics as am

    spec = am.LEADERBOARD_METRICS["yards_per_team_pass_att"]
    assert spec["label"] == "Yards / Team Pass Att"
    assert spec["category"] == "Receiving"
    assert spec["positions"] == ["WR", "RB", "TE"]
    assert spec["efficiency"] is True
    assert spec["min_vol"] == am._V_GAMES
    assert "basic" not in spec  # opt-in column, not a default
    assert spec["desc"] == (
        "Receiving yards per pass attempt by the player's team in games "
        "he played. Combines target share, efficiency, and team passing "
        "volume in one production rate; the closest available stand-in "
        "for yards per route run, which needs route data the feed does "
        "not carry.")
    assert "yards_per_team_pass_att" not in am.PRO_METRICS

    # Week range routes to the adv-weekly path: registered weighted, in
    # the weekly table's column lists, and NOT in _WEEKLY_METRICS (the
    # player_weekly_metrics path has no team dimension).
    assert "yards_per_team_pass_att" in am.ADV_WEEKLY_METRIC_KEYS
    assert am.adv_weekly_metric_supported("yards_per_team_pass_att")
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["yards_per_team_pass_att"] \
        == "w_team_pass_att"
    assert "yards_per_team_pass_att" in am.WEEKLY_ADV_METRIC_COLS
    assert "w_team_pass_att" in am.WEEKLY_ADV_WEIGHT_COLS
    assert am.adv_weekly_vol_spec("yards_per_team_pass_att") == \
        {"label": "Min Team Att", "opts": [50, 100, 200, 400]}
    assert "yards_per_team_pass_att" not in am._WEEKLY_METRICS

    # Career aggregation list + modal ranks gate carry the key.
    career_src = inspect.getsource(am.get_player_career_metrics)
    assert "'yards_per_team_pass_att'" in career_src
    assert '"yards_per_team_pass_att": ("games", 4)' in inspect.getsource(am)


# --------------------------------------------------------------------------- #
# Snapshot persistence (round-trip / COALESCE preservation)
# --------------------------------------------------------------------------- #
class _SnapshotConn(_FakeConn):
    def execute(self, sql, params=None):
        self.executes.append((sql, params))
        return _FakeConn()


def _insert_rows(conn):
    rows = []
    for sql, p in conn.executes:
        if "INSERT INTO player_advanced_metrics" not in sql:
            continue
        cols = [c.strip() for c in
                re.search(r"\((.*?)\)\s*VALUES", sql, re.S).group(1).split(",")]
        assert len(cols) == len(p)  # placeholders/values stay aligned
        rows.append((sql, dict(zip(cols, p))))
    return rows


def test_snapshot_round_trip_and_coalesce(monkeypatch):
    from data_building import advanced_metrics as am

    monkeypatch.setattr(am, "init_advanced_metrics_db", lambda: None)
    conn = _SnapshotConn()
    monkeypatch.setattr(am, "get_conn", lambda: conn)

    am.save_metrics_snapshot([{
        "player_id": "1002", "position": "WR", "games": 2,
        "yards_per_team_pass_att": 200.0 / 71.0, "team_pass_att_gp": 71.0,
    }], "2025-10-05", season=2025)
    rows = _insert_rows(conn)
    assert len(rows) == 1
    sql, vals = rows[0]
    assert vals["yards_per_team_pass_att"] == pytest.approx(200.0 / 71.0)
    assert vals["team_pass_att_gp"] == 71.0
    # A later partial build without the keys writes NULLs and the
    # COALESCE update keeps the stored values.
    assert "yards_per_team_pass_att = COALESCE(EXCLUDED.yards_per_team_pass_att" in sql
    assert "team_pass_att_gp = COALESCE(EXCLUDED.team_pass_att_gp" in sql

    conn2 = _SnapshotConn()
    monkeypatch.setattr(am, "get_conn", lambda: conn2)
    am.save_metrics_snapshot([{
        "player_id": "1002", "position": "WR", "games": 2,
    }], "2025-10-06", season=2025)
    _sql2, vals2 = _insert_rows(conn2)[0]
    assert vals2["yards_per_team_pass_att"] is None
    assert vals2["team_pass_att_gp"] is None
