"""Catchable-ball % (PFR advstats) metrics.

QB Catchable % = 1 - bad_throws / attempts; WR/TE Catchable Tgt % =
1 - drops / targets. Both use PFR human charting via the nflverse
pfr_advstats release (not PFF); the advstats files carry counts only, so
denominators come from Sleeper weekly volume.

Covers, with no network / no nfl_data_py:
- download_pfr_advstats_pass_csv / download_pfr_advstats_rec_csv: shared
  cache logic, cache reuse without network, failure returns None
- build_pfr_catchable_weekly: per-(sleeper_id, week) counts, REG + week
  filters, explicit zero counts kept, season floor, download/crosswalk
  failures
- _apply_catchable_weekly: pct derived from w_pass_att / w_targets, zero
  denominators skipped, 100% when the count is zero
- _merge_pfr_catchable: season totals across completed weeks (never an
  average of weekly pcts), completed_week cutoff, zero denominators
  guarded, small samples still compute, never clobbers, degrades to 0
- calculate_passing_metrics / calculate_receiving_metrics pass-through
- LEADERBOARD_METRICS entries: free Passing / Receiving metrics with the
  normal volume gates and pct formatting
- WEEKLY_ADV_METRIC_COLS + _ADV_WEEKLY_WEIGHTED_METRICS wiring, so week
  ranges re-aggregate as totals ratios
- preset contracts unchanged
"""

import csv

import pytest

from data_building import advanced_metrics as am
from data_building.external_data import nflverse_metrics as nm


_PASS_ROWS = [
    # week 1
    {"pfr_player_id": "p1", "week": "1", "game_type": "REG",
     "passing_bad_throws": "3", "passing_bad_throw_pct": "0.1"},
    {"pfr_player_id": "p2", "week": "1", "game_type": "REG",
     "passing_bad_throws": "0", "passing_bad_throw_pct": "0"},
    # week 2
    {"pfr_player_id": "p1", "week": "2", "game_type": "REG",
     "passing_bad_throws": "1", "passing_bad_throw_pct": "0.025"},
    # noise rows: unmapped id, preseason, bad week
    {"pfr_player_id": "p3", "week": "1", "game_type": "REG",
     "passing_bad_throws": "5", "passing_bad_throw_pct": "0.2"},
    {"pfr_player_id": "p1", "week": "1", "game_type": "PRE",
     "passing_bad_throws": "50", "passing_bad_throw_pct": "0.9"},
    {"pfr_player_id": "p9", "week": "0", "game_type": "REG",
     "passing_bad_throws": "4", "passing_bad_throw_pct": "0.5"},
]

_PASS_FIELDS = ["pfr_player_id", "week", "game_type",
                "passing_bad_throws", "passing_bad_throw_pct"]

_REC_ROWS = [
    # week 1
    {"pfr_player_id": "p4", "week": "1", "game_type": "REG",
     "receiving_drop": "2", "receiving_drop_pct": "0.25"},
    {"pfr_player_id": "p5", "week": "1", "game_type": "REG",
     "receiving_drop": "0", "receiving_drop_pct": "0"},
    # week 2
    {"pfr_player_id": "p4", "week": "2", "game_type": "REG",
     "receiving_drop": "1", "receiving_drop_pct": "0.143"},
    # noise: preseason row must not leak in
    {"pfr_player_id": "p4", "week": "1", "game_type": "PRE",
     "receiving_drop": "50", "receiving_drop_pct": "0.9"},
]

_REC_FIELDS = ["pfr_player_id", "week", "game_type",
               "receiving_drop", "receiving_drop_pct"]

_CROSSWALK = {"p1": "1234", "p2": "5678", "p4": "4444", "p5": "5555"}


@pytest.fixture()
def pfr_csvs(tmp_path):
    pass_path = tmp_path / "advstats_week_pass_2026.csv"
    rec_path = tmp_path / "advstats_week_rec_2026.csv"
    with pass_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PASS_FIELDS)
        writer.writeheader()
        writer.writerows(_PASS_ROWS)
    with rec_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_REC_FIELDS)
        writer.writeheader()
        writer.writerows(_REC_ROWS)
    return str(pass_path), str(rec_path)


@pytest.fixture()
def stub_pfr_sources(monkeypatch, pfr_csvs):
    """Stub the download helpers and id crosswalk on nflverse_metrics."""
    pass_path, rec_path = pfr_csvs
    monkeypatch.setattr(
        nm, "download_pfr_advstats_pass_csv",
        lambda season, max_age_hours=6.0: pass_path)
    monkeypatch.setattr(
        nm, "download_pfr_advstats_rec_csv",
        lambda season, max_age_hours=6.0: rec_path)
    monkeypatch.setattr(nm, "_pfr_to_sleeper", lambda: dict(_CROSSWALK))
    yield


# --- download helpers ---------------------------------------------------------

def _fake_urlopen(body: bytes):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return body

    def _open(req, timeout=60):
        _open.urls.append(req.full_url)
        return _Resp()

    _open.urls = []
    return _open


def test_download_helpers_share_cached_logic(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.paths.CACHE_DIR", str(tmp_path))
    fake = _fake_urlopen(b"pfr_player_id,week\np1,1\n")
    monkeypatch.setattr("urllib.request.urlopen", fake)
    pass_path = nm.download_pfr_advstats_pass_csv(2026)
    rec_path = nm.download_pfr_advstats_rec_csv(2026)
    assert pass_path.endswith("pfr_advstats_week_pass_2026.csv")
    assert rec_path.endswith("pfr_advstats_week_rec_2026.csv")
    assert any("advstats_week_pass_2026.csv" in u for u in fake.urls)
    assert any("advstats_week_rec_2026.csv" in u for u in fake.urls)
    # The rush helper (refactored onto the same downloader) still works.
    rush_path = nm.download_pfr_advstats_rush_csv(2026)
    assert rush_path.endswith("pfr_advstats_week_rush_2026.csv")
    # Second call reuses the cache: no new network fetch.
    def _boom(*a, **k):
        raise AssertionError("must use cache, not the network")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert nm.download_pfr_advstats_pass_csv(2026) == pass_path
    assert nm.download_pfr_advstats_rec_csv(2026) == rec_path


def test_download_helpers_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.paths.CACHE_DIR", str(tmp_path))

    def _down(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", _down)
    assert nm.download_pfr_advstats_pass_csv(2026) is None
    assert nm.download_pfr_advstats_rec_csv(2026) is None


# --- build_pfr_catchable_weekly ------------------------------------------------

def test_weekly_build_counts_keyed_by_sleeper_and_week(stub_pfr_sources):
    out = nm.build_pfr_catchable_weekly(2026)
    # p1: 3 bad throws week 1, 1 bad throw week 2
    assert out[("1234", 1)] == {"bad_throws": 3.0}
    assert out[("1234", 2)] == {"bad_throws": 1.0}
    # p2: explicit zero bad throws is meaningful (charted, none observed)
    assert out[("5678", 1)] == {"bad_throws": 0.0}
    # p4: 2 drops week 1, 1 drop week 2; p5: zero drops
    assert out[("4444", 1)] == {"drops": 2.0}
    assert out[("4444", 2)] == {"drops": 1.0}
    assert out[("5555", 1)] == {"drops": 0.0}
    # unmapped p3 dropped; preseason and week-0 rows excluded
    assert all(k[1] > 0 for k in out)
    assert len(out) == 6


def test_weekly_build_season_floor(stub_pfr_sources):
    assert nm.build_pfr_catchable_weekly(2017) == {}


def test_weekly_build_download_failure(monkeypatch):
    monkeypatch.setattr(nm, "download_pfr_advstats_pass_csv",
                        lambda s, max_age_hours=6.0: None)
    monkeypatch.setattr(nm, "download_pfr_advstats_rec_csv",
                        lambda s, max_age_hours=6.0: None)
    assert nm.build_pfr_catchable_weekly(2026) == {}


def test_weekly_build_no_crosswalk(monkeypatch, pfr_csvs):
    pass_path, rec_path = pfr_csvs
    monkeypatch.setattr(nm, "download_pfr_advstats_pass_csv",
                        lambda s, max_age_hours=6.0: pass_path)
    monkeypatch.setattr(nm, "download_pfr_advstats_rec_csv",
                        lambda s, max_age_hours=6.0: rec_path)
    monkeypatch.setattr(nm, "_pfr_to_sleeper", lambda: {})
    assert nm.build_pfr_catchable_weekly(2026) == {}


# --- _apply_catchable_weekly ----------------------------------------------------

def test_weekly_apply_derives_pct_from_pbp_weights():
    out = {
        ("1234", 1): {"w_pass_att": 30.0},
        ("4444", 1): {"w_targets": 8.0},
        ("5678", 1): {},  # counts but no volume: nothing to divide by
        ("9999", 1): {"w_pass_att": 0.0},  # explicit zero denominator
    }
    counts = {
        ("1234", 1): {"bad_throws": 3.0},
        ("4444", 1): {"drops": 2.0},
        ("5678", 1): {"bad_throws": 1.0},
        ("9999", 1): {"bad_throws": 0.0},
    }
    nm._apply_catchable_weekly(out, counts)
    # (30 - 3) / 30 * 100 = 90.0; (8 - 2) / 8 * 100 = 75.0
    assert out[("1234", 1)]["catchable_pass_pct"] == pytest.approx(90.0)
    assert out[("4444", 1)]["catchable_tgt_pct"] == pytest.approx(75.0)
    assert "catchable_pass_pct" not in out[("5678", 1)]
    assert "catchable_pass_pct" not in out[("9999", 1)]


def test_weekly_apply_zero_count_gives_hundred(stub_pfr_sources):
    out = {("5678", 1): {"w_pass_att": 25.0}, ("5555", 1): {"w_targets": 6.0}}
    counts = nm.build_pfr_catchable_weekly(2026)
    nm._apply_catchable_weekly(out, counts)
    assert out[("5678", 1)]["catchable_pass_pct"] == pytest.approx(100.0)
    assert out[("5555", 1)]["catchable_tgt_pct"] == pytest.approx(100.0)


# --- _merge_pfr_catchable (season snapshot) --------------------------------------

def test_merge_totals_then_divides(stub_pfr_sources):
    # p1: 3 + 1 bad throws over 70 attempts -> (70-4)/70*100
    # p4: 2 + 1 drops over 15 targets -> (15-3)/15*100
    usage_map = {
        "1234": {"pass_att": 70},
        "5678": {"pass_att": 25},   # 0 bad throws -> 100%
        "4444": {"targets": 15},
        "5555": {"targets": 6},     # 0 drops -> 100%
    }
    merged = am._merge_pfr_catchable(usage_map, 2026, completed_week=2)
    assert merged == 4
    assert usage_map["1234"]["catchable_pass_pct"] == pytest.approx(
        round((70 - 4) / 70 * 100, 1))
    assert usage_map["5678"]["catchable_pass_pct"] == pytest.approx(100.0)
    assert usage_map["4444"]["catchable_tgt_pct"] == pytest.approx(
        round((15 - 3) / 15 * 100, 1))
    assert usage_map["5555"]["catchable_tgt_pct"] == pytest.approx(100.0)


def test_merge_never_averages_weekly_pcts(stub_pfr_sources):
    # Weekly pcts would be 90.0 (wk1) and 97.5 (wk2); the naive mean is
    # 93.75. The merge must divide season totals instead: 94.3.
    usage_map = {"1234": {"pass_att": 70}}
    am._merge_pfr_catchable(usage_map, 2026, completed_week=2)
    val = usage_map["1234"]["catchable_pass_pct"]
    assert val == pytest.approx(94.3)
    assert val != pytest.approx(round((90.0 + 97.5) / 2, 1))


def test_merge_respects_completed_week(stub_pfr_sources):
    usage_map = {"1234": {"pass_att": 30}, "4444": {"targets": 8}}
    am._merge_pfr_catchable(usage_map, 2026, completed_week=1)
    assert usage_map["1234"]["catchable_pass_pct"] == pytest.approx(90.0)
    assert usage_map["4444"]["catchable_tgt_pct"] == pytest.approx(75.0)


def test_merge_zero_denominators_guarded(stub_pfr_sources):
    usage_map = {"1234": {"pass_att": 0}, "4444": {"targets": 0}}
    merged = am._merge_pfr_catchable(usage_map, 2026, completed_week=2)
    assert merged == 0
    assert "catchable_pass_pct" not in usage_map["1234"]
    assert "catchable_tgt_pct" not in usage_map["4444"]


def test_merge_small_samples_still_compute(stub_pfr_sources):
    # 3 attempts, 3 bad throws -> 0.0 (not skipped for low volume)
    usage_map = {"1234": {"pass_att": 3}}
    am._merge_pfr_catchable(usage_map, 2026, completed_week=1)
    assert usage_map["1234"]["catchable_pass_pct"] == pytest.approx(0.0)


def test_merge_never_clobbers(stub_pfr_sources):
    usage_map = {"1234": {"pass_att": 70, "catchable_pass_pct": 99.9},
                 "4444": {"targets": 15, "catchable_tgt_pct": 88.8}}
    merged = am._merge_pfr_catchable(usage_map, 2026, completed_week=2)
    assert merged == 0
    assert usage_map["1234"]["catchable_pass_pct"] == pytest.approx(99.9)
    assert usage_map["4444"]["catchable_tgt_pct"] == pytest.approx(88.8)


def test_merge_download_failure_is_zero(monkeypatch):
    monkeypatch.setattr(
        nm, "download_pfr_advstats_pass_csv", lambda s, max_age_hours=6.0: None)
    monkeypatch.setattr(
        nm, "download_pfr_advstats_rec_csv", lambda s, max_age_hours=6.0: None)
    usage_map = {"1234": {"pass_att": 70}}
    assert am._merge_pfr_catchable(usage_map, 2026, 2) == 0
    assert usage_map == {"1234": {"pass_att": 70}}


def test_merge_crosswalk_failure_is_zero(monkeypatch, stub_pfr_sources):
    monkeypatch.setattr(nm, "_pfr_to_sleeper",
                        lambda: (_ for _ in ()).throw(RuntimeError("id map down")))
    usage_map = {"1234": {"pass_att": 70}}
    assert am._merge_pfr_catchable(usage_map, 2026, 2) == 0
    assert "catchable_pass_pct" not in usage_map["1234"]


def test_merge_empty_map_short_circuits(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not download for an empty map")

    monkeypatch.setattr(nm, "download_pfr_advstats_pass_csv", _boom)
    assert am._merge_pfr_catchable({}, 2026, 2) == 0


# --- calculate_* pass-through ---------------------------------------------------

def test_calculate_functions_pass_through_merge_values():
    p = am.calculate_passing_metrics(
        {"avg_pass_att": 30, "catchable_pass_pct": 94.3})
    assert p["catchable_pass_pct"] == 94.3
    r = am.calculate_receiving_metrics(
        {"avg_targets": 8, "catchable_tgt_pct": 75.0})
    assert r["catchable_tgt_pct"] == 75.0
    # absent merge value stays absent
    assert am.calculate_passing_metrics({})["catchable_pass_pct"] is None
    assert am.calculate_receiving_metrics({})["catchable_tgt_pct"] is None


# --- metadata / wiring ----------------------------------------------------------

def _metric(key):
    return am.LEADERBOARD_METRICS[key]


def test_leaderboard_entries_are_free():
    m = _metric("catchable_pass_pct")
    assert m["label"] == "Catchable %"
    assert m["category"] == "Passing"
    assert m.get("pro", False) is False
    assert m.get("efficiency") is True
    assert m.get("pct") is True
    assert m["min_vol"]["col"] == "total_pass_att"
    assert m["positions"] == ["QB"]

    m = _metric("catchable_tgt_pct")
    assert m["label"] == "Catchable Tgt %"
    assert m["category"] == "Receiving"
    assert m.get("pro", False) is False
    assert m.get("efficiency") is True
    assert m.get("pct") is True
    assert m["min_vol"]["col"] == "total_targets"
    assert m["positions"] == ["WR", "TE"]


def test_weekly_wiring_supports_ranges():
    assert "catchable_pass_pct" in am.WEEKLY_ADV_METRIC_COLS
    assert "catchable_tgt_pct" in am.WEEKLY_ADV_METRIC_COLS
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["catchable_pass_pct"] == "w_pass_att"
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["catchable_tgt_pct"] == "w_targets"
    assert am.adv_weekly_metric_supported("catchable_pass_pct") is True
    assert am.adv_weekly_metric_supported("catchable_tgt_pct") is True
    # Weighted re-aggregation SQL is a totals ratio, not an average of ratios.
    sql, _weight_sql = am._adv_weekly_agg_sql("catchable_pass_pct")
    assert "SUM(catchable_pass_pct * w_pass_att)" in sql
    assert "SUM(CASE WHEN catchable_pass_pct IS NOT NULL THEN w_pass_att END)" in sql
    sql, _weight_sql = am._adv_weekly_agg_sql("catchable_tgt_pct")
    assert "SUM(catchable_tgt_pct * w_targets)" in sql


def test_career_numeric_aggregation_covers_new_fields():
    # _NUMERIC_METRICS lives inside get_player_career_metrics; assert both
    # fields flow through it by checking the source list directly.
    import inspect
    src = inspect.getsource(am.get_player_career_metrics)
    assert "'catchable_pass_pct'" in src and "'catchable_tgt_pct'" in src


def test_preset_contracts_unchanged():
    from dashboard_services.pages.advanced_metrics_page import ADVANCED_METRIC_PRESETS
    for preset in ADVANCED_METRIC_PRESETS.values():
        assert "catchable_pass_pct" not in preset["metrics"]
        assert "catchable_tgt_pct" not in preset["metrics"]
