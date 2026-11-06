"""Focused contracts for the per-snap family additions and First Down Rate.

Points / Snap and Yards / Snap complete the per-snap family next to
Targets / Snap and Touches / Snap. Their weekly gates must be in snaps:
the leaderboard handler derives its default min_vol from the SEASON spec
(_V_SNAPS, col total_snaps), so a weekly gate in any other unit demands a
season's worth of that unit inside the range and empties the board (the
PR #3042 failure mode).

First Down Rate is the combined chain-moving rate: receiving + rushing
first downs over targets + carries. The denominator is targets + carries
explicitly because total_touches in this store is carries + receptions;
the numerator is total_first_downs, the same count behind First Downs/G.
"""
import contextlib
from pathlib import Path

from dashboard_services.pages.advanced_metrics_page import ADVANCED_METRIC_PRESETS
from data_building.advanced_metrics import (
    LEADERBOARD_METRICS,
    PRO_METRICS,
    _SEASON_FROM_WEEKLY,
    _WEEKLY_METRICS,
)

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
ADVANCED_METRICS_PY = (
    ROOT / "data_building" / "advanced_metrics.py").read_text(encoding="utf-8")

PER_SNAP = ("pts_per_snap", "yards_per_snap")


def test_weekly_per_snap_specs_gate_on_snaps():
    for key in PER_SNAP:
        spec = _WEEKLY_METRICS[key]
        assert spec["min_col"] == "SUM(snaps)", key
        assert spec["min_label"] == "Min Snaps", key
        assert spec["min_opts"] == [50, 100, 200, 400], key


def test_weekly_sql_shapes():
    assert _WEEKLY_METRICS["pts_per_snap"]["sql"] == (
        "SUM(ppr_pts)::float / NULLIF(SUM(snaps), 0)")
    assert _WEEKLY_METRICS["yards_per_snap"]["sql"] == (
        "(SUM(rush_yards) + SUM(rec_yards))::float / NULLIF(SUM(snaps), 0)")
    assert _WEEKLY_METRICS["first_down_rate"]["sql"] == (
        "SUM(first_downs)::float / NULLIF(SUM(targets) + SUM(carries), 0)")
    fd = _WEEKLY_METRICS["first_down_rate"]
    assert fd["min_col"] == "SUM(touches)"
    assert fd["min_label"] == "Min Touches"
    assert fd["min_opts"] == [5, 10, 20, 40]


def test_season_registry_entries_are_free_and_snaps_gated():
    for key, label in (("pts_per_snap", "Points / Snap"),
                       ("yards_per_snap", "Yards / Snap")):
        spec = LEADERBOARD_METRICS[key]
        assert spec["label"] == label, key
        assert spec["category"] == "General", key
        assert spec["positions"] == ["RB", "WR", "TE"], key
        assert spec["min_vol"]["col"] == "total_snaps", key
        assert key not in PRO_METRICS, key
    # pts_per_snap must NOT carry a season-table computed_sql: PPR points
    # have no column on player_advanced_metrics (they live in
    # player_weekly_metrics), so "m.ppr_pts / m.total_snaps" is an
    # UndefinedColumn 500 in production. Season mode delegates to the
    # weekly table via _SEASON_FROM_WEEKLY, exactly like ppr_pts itself.
    assert "computed_sql" not in LEADERBOARD_METRICS["pts_per_snap"]
    assert "pts_per_snap" in _SEASON_FROM_WEEKLY
    assert LEADERBOARD_METRICS["yards_per_snap"]["computed_sql"] == (
        "(COALESCE(m.yards_per_carry * m.total_carries, 0) + "
        "COALESCE(m.yards_per_reception * m.total_receptions, 0)) "
        "/ NULLIF(m.total_snaps, 0)")


def test_first_down_rate_formula_is_targets_plus_carries():
    spec = LEADERBOARD_METRICS["first_down_rate"]
    assert spec["label"] == "First Down Rate"
    assert spec["category"] == "General"
    assert spec["positions"] == ["RB", "WR", "TE"]
    assert spec["pct"] is True and spec["pct_frac"] is True
    assert spec["min_vol"]["col"] == "total_touches"
    assert "first_down_rate" not in PRO_METRICS
    # Numerator is the stored first-down count; the denominator is
    # targets + carries explicitly, NOT total_touches (carries + recs).
    assert spec["computed_sql"] == (
        "m.total_first_downs::float / "
        "NULLIF(COALESCE(m.total_targets, 0) + "
        "COALESCE(m.total_carries, 0), 0)")
    assert "m.total_touches" not in spec["computed_sql"]
    # The formula itself: 45 first downs on 80 targets + 20 carries.
    assert 45 / (80 + 20) == 0.45


def test_rank_cache_supplement_materializes_the_new_metrics():
    for key in ("pts_per_snap", "yards_per_snap", "first_down_rate"):
        assert f'_row["{key}"]' in ADVANCED_METRICS_PY, key
    assert '"pts_per_snap": ("total_snaps", 100)' in ADVANCED_METRICS_PY
    assert '"yards_per_snap": ("total_snaps", 100)' in ADVANCED_METRICS_PY
    assert '"first_down_rate": ("total_touches", 40)' in ADVANCED_METRICS_PY


def test_presets_surface_the_per_snap_pair():
    general = ADVANCED_METRIC_PRESETS["general"]["metrics"]
    assert general.index("pts_per_snap") == general.index("snap_share") + 1
    assert general.index("yards_per_snap") == general.index("pts_per_snap") + 1
    rb = ADVANCED_METRIC_PRESETS["rb"]["metrics"]
    assert rb.index("pts_per_snap") == rb.index("yards_per_touch") + 1
    assert rb.index("yards_per_snap") == rb.index("pts_per_snap") + 1
    for preset_id in ("wr", "te"):
        metrics = ADVANCED_METRIC_PRESETS[preset_id]["metrics"]
        assert metrics.index("pts_per_snap") == (
            metrics.index("targets_per_snap") + 1), preset_id
        assert metrics.index("yards_per_snap") == (
            metrics.index("pts_per_snap") + 1), preset_id
    # First Down Rate is in no preset: neither of its component rates
    # (rec_first_down_rate / rush_first_down_rate) is in one either.
    for preset in ADVANCED_METRIC_PRESETS.values():
        assert "first_down_rate" not in preset["metrics"]


def test_compare_fallback_entries_in_app_js():
    rb = APP_JS[APP_JS.index("const rbMetrics = ["):]
    rb = rb[: rb.index("];")]
    wr_te = APP_JS[APP_JS.index("const wrTeMetrics = ["):]
    wr_te = wr_te[: wr_te.index("];")]
    for key in ("pts_per_snap", "yards_per_snap", "first_down_rate"):
        assert f"'{key}'" in rb, key
        assert f"'{key}'" in wr_te, key
    assert "pts_per_snap:'Points/Snap'" in APP_JS
    assert "yards_per_snap:'Yards/Snap'" in APP_JS
    assert "first_down_rate:'First Down Rate'" in APP_JS
    # Ranges from the proof-computed 2026 leaders (pts .53, yards 2.92,
    # first-down rate .625 at the default gates), with headroom.
    assert "'pts_per_snap': 0.6" in APP_JS
    assert "'yards_per_snap': 3.2" in APP_JS
    assert "'first_down_rate': 0.7" in APP_JS


def _weekly_rows(monkeypatch, metric, value, vol):
    import data_building.advanced_metrics as metrics

    captured = {}

    class Conn:
        def execute(self, sql, params=()):
            captured["sql"] = sql
            captured["params"] = params
            return self

        def fetchall(self):
            return [{
                "player_id": "p1", "position": "RB", "weeks_played": 8,
                "value": value, "ctx_receptions": 30, "ctx_targets": 40,
                "ctx_carries": 50, "_wvol": vol,
            }]

    @contextlib.contextmanager
    def conn():
        yield Conn()

    monkeypatch.setattr(metrics, "get_conn", conn)
    rows = metrics.get_weekly_range_leaderboard(
        metric, position="RB", season=2026,
        week_start=1, week_end=8, min_vol=50)
    return rows, captured


def test_weekly_leaderboard_gates_per_snap_on_snaps(monkeypatch):
    """A default-scaled gate (50) must be applied to in-range snaps.

    Player in range: 280 snaps. Under a targets/touches gate a min_vol
    of 50 could still pass or fail on the wrong unit; the emitted SQL
    must gate on SUM(snaps) and report vol in snaps.
    """
    rows, captured = _weekly_rows(monkeypatch, "pts_per_snap", 150 / 280, 280)
    assert "SUM(snaps) AS _wvol" in captured["sql"]
    assert 50 in captured["params"]
    assert len(rows) == 1
    assert rows[0]["vol"] == 280
    assert rows[0]["value"] == 150 / 280

    rows, captured = _weekly_rows(monkeypatch, "yards_per_snap", 710 / 280, 280)
    assert "SUM(snaps) AS _wvol" in captured["sql"]
    assert len(rows) == 1
    assert rows[0]["value"] == 710 / 280


def test_weekly_first_down_rate_gates_on_touches(monkeypatch):
    rows, captured = _weekly_rows(monkeypatch, "first_down_rate", 45 / 90, 90)
    assert "SUM(first_downs)" in captured["sql"]
    assert "SUM(touches) AS _wvol" in captured["sql"]
    assert len(rows) == 1
    assert rows[0]["value"] == 45 / 90


# ── Regression: season board must not reference columns the store lacks ──
# Production failure (2026-11-06): GET /api/advanced-metrics/leaderboard
# ?metric=pts_per_snap returned 500, psycopg UndefinedColumn "m.ppr_pts
# does not exist". The season spec's computed_sql named a column that
# only exists on player_weekly_metrics. The tests above pinned the SQL
# string, so nothing caught it; the tests below validate the catalog
# against the store's actual DDL and exercise the season entry point.

def _season_store_columns():
    """player_advanced_metrics columns from the DDL in advanced_metrics.py
    (its CREATE TABLE block plus every ADD COLUMN migration in the file;
    no other module alters this table)."""
    import re

    cols = set()
    in_create = False
    for ln in ADVANCED_METRICS_PY.splitlines():
        if "CREATE TABLE IF NOT EXISTS player_advanced_metrics" in ln:
            in_create = True
            continue
        if in_create:
            if ln.strip() == ");":
                break
            m = re.match(
                r"\s+([a-z_][a-z0-9_]*)\s+"
                r"(NUMERIC|VARCHAR|INTEGER|DATE|SERIAL|TEXT|REAL|BOOLEAN)",
                ln)
            if m:
                cols.add(m.group(1))
    cols |= set(re.findall(
        r"ADD COLUMN IF NOT EXISTS\s+([a-z_][a-z0-9_]*)",
        ADVANCED_METRICS_PY))
    return cols


def test_every_computed_sql_reference_exists_in_the_season_store():
    import re

    cols = _season_store_columns()
    assert "total_snaps" in cols and "total_first_downs" in cols  # sanity
    assert "ppr_pts" not in cols  # points live in player_weekly_metrics
    for key, spec in LEADERBOARD_METRICS.items():
        expr = (spec.get("computed_sql") or "") + " " + (
            spec.get("computed_null") or "")
        if not expr.strip():
            continue
        refs = set(re.findall(r"\bm\.([A-Za-z_][A-Za-z0-9_]*)", expr))
        missing = refs - cols
        assert not missing, (
            f"{key} references columns player_advanced_metrics lacks: "
            f"{sorted(missing)}")


def test_season_pts_per_snap_board_aggregates_the_weekly_table(monkeypatch):
    """The season entry point (get_metric_leaderboard, no week range)
    must serve Points / Snap from player_weekly_metrics — SUM(ppr_pts) /
    SUM(snaps) gated on SUM(snaps) — and never issue SQL against
    player_advanced_metrics for this metric."""
    import data_building.advanced_metrics as metrics

    captured = []

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=()):
            captured.append(sql)
            return self

        def fetchall(self):
            return [{
                "player_id": "p1", "position": "RB", "weeks_played": 8,
                "value": 150 / 280, "ctx_receptions": 30, "ctx_targets": 40,
                "ctx_carries": 50, "_wvol": 280,
            }]

        def fetchone(self):
            return None

    monkeypatch.setattr(metrics, "get_conn", lambda: Conn())
    metrics._METRIC_LEADERBOARD_CACHE.clear()
    rows = metrics.get_metric_leaderboard(
        "pts_per_snap", position="RB", season=2026, min_vol=100)
    assert len(rows) == 1
    assert rows[0]["value"] == 150 / 280
    assert rows[0]["vol"] == 280
    assert captured, "no SQL issued"
    assert all("player_weekly_metrics" in sql for sql in captured)
    assert all("player_advanced_metrics" not in sql for sql in captured)
    assert any("SUM(ppr_pts)" in sql and "SUM(snaps)" in sql
               for sql in captured)
