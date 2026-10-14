"""Avg Sep Score: replaces the retired Created Sep display metric.

Created Sep (average separation minus pre-snap cushion) is negative by
construction: on real NGS data every qualified player in 2025 was
negative (range -4.54..-0.56, median -3.04), and the value correlates
more with the cushion defenses allow (-0.65) than with separation
itself (+0.55), so its leaderboard mostly ranked who got pressed.

Avg Sep Score is a 0-100 percentile of average separation among
qualified players at the SAME position, computed from the pool at
serve time (leaderboard, player payloads, modal ranks), never stored.
The qualification gate is the qualification policy's target minimum,
the same gate the modal's rank supplement uses for this metric family.
Created Sep survives only as a stored data-layer field (historical
warehouse inputs); it must not appear in any display config or payload.
"""
import inspect
from pathlib import Path

import data_building.advanced_metrics as am

ROOT = Path(__file__).resolve().parents[1]


# ── percentile math ─────────────────────────────────────────────────────────

def test_percentile_map_known_values_and_ties():
    pairs = [("a", 4.62), ("b", 3.23), ("c", 2.77), ("d", 2.63), ("e", 2.63)]
    scores = am._sep_percentile_map(pairs)
    assert scores == {"a": 90.0, "b": 70.0, "c": 50.0, "d": 20.0, "e": 20.0}


def test_percentile_map_single_player_pool_is_50():
    assert am._sep_percentile_map([("x", 3.0)]) == {"x": 50.0}
    assert am._sep_percentile_map([]) == {}


def test_pool_scores_partition_by_position_and_gate():
    pool = [
        # WR pool (gate 15): five qualified.
        {"player_id": "w1", "position": "WR", "ngs_avg_separation": 4.62, "total_targets": 60},
        {"player_id": "w2", "position": "WR", "ngs_avg_separation": 3.23, "total_targets": 141},
        {"player_id": "w3", "position": "WR", "ngs_avg_separation": 3.13, "total_targets": 185},
        {"player_id": "w4", "position": "WR", "ngs_avg_separation": 2.77, "total_targets": 117},
        {"player_id": "w5", "position": "WR", "ngs_avg_separation": 2.63, "total_targets": 88},
        # TE pool: different separation scale, must not leak into WR scores.
        {"player_id": "t1", "position": "TE", "ngs_avg_separation": 4.08, "total_targets": 68},
        {"player_id": "t2", "position": "TE", "ngs_avg_separation": 3.63, "total_targets": 69},
        {"player_id": "t3", "position": "TE", "ngs_avg_separation": 3.06, "total_targets": 169},
        {"player_id": "t4", "position": "TE", "ngs_avg_separation": 2.92, "total_targets": 86},
        # Unqualified: below the target gate, missing targets, missing sep.
        {"player_id": "u1", "position": "WR", "ngs_avg_separation": 5.50, "total_targets": 10},
        {"player_id": "u2", "position": "TE", "ngs_avg_separation": 3.90, "total_targets": None},
        {"player_id": "u3", "position": "WR", "ngs_avg_separation": None, "total_targets": 90},
    ]
    scores = am._sep_scores_from_pool(pool, gate=15)
    # WR: sorted seps [2.63, 2.77, 3.13, 3.23, 4.62] -> leader 90, median 50.
    assert scores["w1"] == 90.0
    assert scores["w3"] == 50.0
    assert scores["w5"] == 10.0
    # TE: sorted [2.92, 3.06, 3.63, 4.08] -> t4 bottom (12.5), t1 top (87.5).
    assert scores["t4"] == 12.5
    assert scores["t1"] == 87.5
    for pid in ("u1", "u2", "u3"):
        assert pid not in scores


# ── leaderboards ────────────────────────────────────────────────────────────

_POOL = [
    {"player_id": "w1", "position": "WR", "ngs_avg_separation": 4.62, "total_targets": 60, "games": 12},
    {"player_id": "w2", "position": "WR", "ngs_avg_separation": 2.63, "total_targets": 88, "games": 14},
    {"player_id": "t1", "position": "TE", "ngs_avg_separation": 4.08, "total_targets": 68, "games": 13},
    {"player_id": "t2", "position": "TE", "ngs_avg_separation": 2.92, "total_targets": 86, "games": 15},
    {"player_id": "u1", "position": "WR", "ngs_avg_separation": 5.50, "total_targets": 3, "games": 4},
]


def _patch_index(monkeypatch):
    import utils.utils as uu
    monkeypatch.setattr(uu, "load_players_index", lambda: {
        pid: {"name": pid.upper(), "team": "T", "espnHeadshot": ""}
        for pid in ("w1", "w2", "t1", "t2", "u1")
    })


def test_season_leaderboard_scores_and_filters(monkeypatch):
    _patch_index(monkeypatch)
    monkeypatch.setattr(am, "_sep_season_pool", lambda season: [dict(r) for r in _POOL])
    rows = am.get_sep_score_leaderboard(season=2025)
    assert [r["player_id"] for r in rows] == ["w1", "t1", "w2", "t2"]
    assert [r["value"] for r in rows] == [75.0, 75.0, 25.0, 25.0]
    assert all(r["season"] == 2025 for r in rows)
    # Position filter keeps the position-partitioned scores unchanged.
    tes = am.get_sep_score_leaderboard(position="TE", season=2025)
    assert [(r["player_id"], r["value"]) for r in tes] == [("t1", 75.0), ("t2", 25.0)]
    # Display min_vol filters rows without recomputing the pool.
    strict = am.get_sep_score_leaderboard(season=2025, min_vol=80)
    assert [r["player_id"] for r in strict] == ["w2", "t2"]


def test_range_leaderboard_scores(monkeypatch):
    _patch_index(monkeypatch)
    monkeypatch.setattr(am, "_sep_range_pool", lambda s, a, b: [dict(r) for r in _POOL])
    rows = am.get_sep_score_range_leaderboard(season=2025, week_start=1, week_end=18)
    assert [r["player_id"] for r in rows] == ["w1", "t1", "w2", "t2"]
    assert rows[0]["value"] == 75.0


def test_leaderboard_dispatch(monkeypatch):
    monkeypatch.setattr(am, "get_sep_score_leaderboard",
                        lambda **kw: [{"player_id": "w1", "value": 90.0}])
    rows = am.get_metric_leaderboard("ngs_avg_sep_score", season=2025)
    assert rows[0]["value"] == 90.0 and rows[0]["season"] == 2025
    monkeypatch.setattr(am, "get_sep_score_range_leaderboard",
                        lambda **kw: [{"player_id": "w1", "value": 80.0}])
    rows = am.get_adv_weekly_range_leaderboard(
        "ngs_avg_sep_score", season=2025, week_start=1, week_end=6)
    assert rows[0]["value"] == 80.0
    assert am.adv_weekly_metric_supported("ngs_avg_sep_score") is True


# ── payload swap ────────────────────────────────────────────────────────────

class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _SeasonConn:
    """Serves the single get_player_metrics_by_season SELECT * query."""

    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        assert "player_id = %s AND season = %s" in sql
        return _Cursor(self._rows)


def test_player_season_payload_swaps_created_for_score(monkeypatch):
    stored = [{
        "player_id": "w1", "season": 2025, "position": "WR",
        "ngs_avg_separation": 4.62, "ngs_created_separation": -2.70,
        "total_targets": 60,
    }]
    monkeypatch.setattr(am, "get_conn", lambda: _SeasonConn(stored))
    monkeypatch.setattr(am, "_avg_sep_scores_for_season", lambda season: {"w1": 99.4})
    out = am.get_player_metrics_by_season("w1", 2025)
    assert out["ngs_avg_sep_score"] == 99.4
    assert "ngs_created_separation" not in out
    # Unqualified players simply have no score key (never an invented one).
    monkeypatch.setattr(am, "_avg_sep_scores_for_season", lambda season: {})
    out = am.get_player_metrics_by_season("w1", 2025)
    assert "ngs_avg_sep_score" not in out
    assert "ngs_created_separation" not in out


# ── config and display-surface contracts ────────────────────────────────────

def test_config_contract():
    spec = am.LEADERBOARD_METRICS["ngs_avg_sep_score"]
    assert spec["label"] == "Avg Sep Score"
    assert spec["positions"] == ["WR", "TE"]
    assert spec["integer"] is True
    assert "percentile" in spec["desc"]
    assert "ngs_created_separation" not in am.LEADERBOARD_METRICS
    # Retired from the weekly advanced plumbing too (the stored DB column
    # and the historical warehouse fields are data layer, untouched).
    assert "ngs_created_separation" not in am.WEEKLY_ADV_METRIC_COLS
    assert "ngs_created_separation" not in am._ADV_WEEKLY_WEIGHTED_METRICS
    assert "ngs_created_separation" not in am.ADV_WEEKLY_METRIC_KEYS
    # The score is registered as week-range capable on a targets basis so
    # the page offers ranges and the Min Targets filter for it.
    assert "ngs_avg_sep_score" in am.ADV_WEEKLY_METRIC_KEYS
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["ngs_avg_sep_score"] == "w_targets"
    vol = am.adv_weekly_vol_spec("ngs_avg_sep_score")
    assert vol and vol["label"] == "Min Targets"


def test_ranks_supplement_contract():
    src = inspect.getsource(am.get_player_metric_ranks)
    assert '"ngs_avg_sep_score": ("total_targets", 15)' in src
    assert '"ngs_created_separation": ("total_targets", 15)' not in src
    career_src = inspect.getsource(am.get_player_career_metrics)
    assert "'ngs_avg_sep_score'" in career_src
    assert "'ngs_created_separation'" not in career_src


def test_display_files_have_no_created_sep():
    page = (ROOT / "dashboard_services" / "pages" / "advanced_metrics_page.py").read_text(encoding="utf-8")
    assert "ngs_created_separation" not in page
    assert "'ngs_avg_sep_score': ['Gets open', 'Blanketed']" in page
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "ngs_created_separation" not in app_js
    assert "Created Sep" not in app_js
    assert "ngs_avg_sep_score:'Avg Sep Score'" in app_js
    assert "'ngs_avg_sep_score': 100" in app_js
    modal_js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "ngs_created_separation" not in modal_js
    assert "ngs_avg_sep_score: 100" in modal_js
    assert "percentile of average separation" in modal_js
