"""Yards before/after contact (PFR advstats) metrics.

Covers, with no network / no nfl_data_py:
- build_pfr_contact_yards_weekly: per-week ybc/yac per carry, w_carries weight,
  REG + week filters, zero-carry skips, season floor, download failure
- _merge_pfr_contact_yards: season totals across completed weeks (never an
  average of averages), completed_week cutoff, never clobbers, degrades to 0
- LEADERBOARD_METRICS entries: free Rushing metrics with the normal
  minimum-carries threshold and w_carries weekly weighting
- WEEKLY_ADV_METRIC_COLS + _ADV_WEEKLY_WEIGHTED_METRICS wiring, so week
  ranges re-aggregate correctly
"""

import csv

import pytest

from data_building import advanced_metrics as am
from data_building.external_data import nflverse_metrics as nm


_CSV_ROWS = [
    # week 1
    {"pfr_player_id": "p1", "week": "1", "game_type": "REG",
     "carries": "10", "rushing_yards_before_contact": "30",
     "rushing_yards_after_contact": "15"},
    {"pfr_player_id": "p2", "week": "1", "game_type": "REG",
     "carries": "5", "rushing_yards_before_contact": "-5",
     "rushing_yards_after_contact": "40"},
    # week 2
    {"pfr_player_id": "p1", "week": "2", "game_type": "REG",
     "carries": "20", "rushing_yards_before_contact": "90",
     "rushing_yards_after_contact": "30"},
    # noise rows: preseason, zero carries, bad week
    {"pfr_player_id": "p1", "week": "1", "game_type": "PRE",
     "carries": "8", "rushing_yards_before_contact": "80",
     "rushing_yards_after_contact": "80"},
    {"pfr_player_id": "p3", "week": "1", "game_type": "REG",
     "carries": "0", "rushing_yards_before_contact": "0",
     "rushing_yards_after_contact": "0"},
    {"pfr_player_id": "p9", "week": "0", "game_type": "REG",
     "carries": "4", "rushing_yards_before_contact": "12",
     "rushing_yards_after_contact": "4"},
]

_FIELDNAMES = ["pfr_player_id", "week", "game_type", "carries",
               "rushing_yards_before_contact", "rushing_yards_after_contact"]

_CROSSWALK = {"p1": "1234", "p2": "5678", "p3": "9999"}


@pytest.fixture()
def pfr_csv(tmp_path):
    path = tmp_path / "advstats_week_rush_2026.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_CSV_ROWS)
    return str(path)


@pytest.fixture()
def stub_pfr_sources(monkeypatch, pfr_csv):
    """Stub the download helper and id crosswalk on nflverse_metrics."""
    monkeypatch.setattr(nm, "download_pfr_advstats_rush_csv", lambda season, max_age_hours=6.0: pfr_csv)
    monkeypatch.setattr(nm, "_pfr_to_sleeper", lambda: dict(_CROSSWALK))
    yield


# --- build_pfr_contact_yards_weekly -------------------------------------------

def test_weekly_build_per_week_math(stub_pfr_sources):
    out = nm.build_pfr_contact_yards_weekly(2026)
    # p1 week 1: 30 YBC / 10 carries = 3.0; 15 YAC / 10 = 1.5
    assert out[("1234", 1)] == {
        "ybc_per_carry": 3.0, "yac_per_carry": 1.5, "w_carries": 10.0}
    # p1 week 2: 90 / 20 = 4.5; 30 / 20 = 1.5
    assert out[("1234", 2)]["ybc_per_carry"] == pytest.approx(4.5)
    assert out[("1234", 2)]["yac_per_carry"] == pytest.approx(1.5)
    # p2 week 1: negative YBC is real (hit behind the line)
    assert out[("5678", 1)]["ybc_per_carry"] == pytest.approx(-1.0)
    assert out[("5678", 1)]["yac_per_carry"] == pytest.approx(8.0)
    # noise rows excluded
    assert ("9999", 1) not in out  # zero carries
    assert all(k[1] > 0 for k in out)  # no week-0 row
    assert len(out) == 3  # preseason 80/8 row did not leak in


def test_weekly_build_season_floor(stub_pfr_sources):
    assert nm.build_pfr_contact_yards_weekly(2017) == {}


def test_weekly_build_download_failure(monkeypatch):
    monkeypatch.setattr(nm, "download_pfr_advstats_rush_csv", lambda s, max_age_hours=6.0: None)
    assert nm.build_pfr_contact_yards_weekly(2026) == {}


def test_weekly_build_no_crosswalk(monkeypatch, pfr_csv):
    monkeypatch.setattr(nm, "download_pfr_advstats_rush_csv", lambda s, max_age_hours=6.0: pfr_csv)
    monkeypatch.setattr(nm, "_pfr_to_sleeper", lambda: {})
    assert nm.build_pfr_contact_yards_weekly(2026) == {}


# --- _merge_pfr_contact_yards (season snapshot) --------------------------------

def test_merge_totals_then_divides(stub_pfr_sources):
    # 10 + 20 carries; YBC 30 + 90 = 120; YAC 15 + 30 = 45.
    usage_map = {"1234": {"games": 2}, "5678": {"games": 1}}
    merged = am._merge_pfr_contact_yards(usage_map, 2026, completed_week=2)
    assert merged == 2
    assert usage_map["1234"]["ybc_per_carry"] == pytest.approx(120 / 30)
    assert usage_map["1234"]["yac_per_carry"] == pytest.approx(45 / 30)
    assert usage_map["5678"]["ybc_per_carry"] == pytest.approx(-1.0)
    assert usage_map["5678"]["yac_per_carry"] == pytest.approx(8.0)


def test_merge_respects_completed_week(stub_pfr_sources):
    usage_map = {"1234": {"games": 2}}
    am._merge_pfr_contact_yards(usage_map, 2026, completed_week=1)
    assert usage_map["1234"]["ybc_per_carry"] == pytest.approx(3.0)
    assert usage_map["1234"]["yac_per_carry"] == pytest.approx(1.5)


def test_merge_never_clobbers(stub_pfr_sources):
    usage_map = {"1234": {"games": 2, "ybc_per_carry": 9.99, "yac_per_carry": 1.11}}
    merged = am._merge_pfr_contact_yards(usage_map, 2026, completed_week=2)
    assert merged == 0
    assert usage_map["1234"]["ybc_per_carry"] == pytest.approx(9.99)
    assert usage_map["1234"]["yac_per_carry"] == pytest.approx(1.11)


def test_merge_download_failure_is_zero(monkeypatch):
    monkeypatch.setattr(
        nm, "download_pfr_advstats_rush_csv", lambda s, max_age_hours=6.0: None)
    usage_map = {"1234": {"games": 2}}
    assert am._merge_pfr_contact_yards(usage_map, 2026, 2) == 0
    assert usage_map == {"1234": {"games": 2}}


def test_merge_empty_map_short_circuits(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not download for an empty map")
    monkeypatch.setattr(nm, "download_pfr_advstats_rush_csv", _boom)
    assert am._merge_pfr_contact_yards({}, 2026, 2) == 0


# --- metadata / wiring -------------------------------------------------------

def _metric(key):
    return am.LEADERBOARD_METRICS[key]


def test_leaderboard_entries_are_free_and_rushing():
    for key, label in (("ybc_per_carry", "YBC / Carry"),
                       ("yac_per_carry", "YAC / Carry")):
        m = _metric(key)
        assert m["label"] == label
        assert m["category"] == "Rushing"
        assert m.get("pro", False) is False
        assert m.get("efficiency") is True
        # normal minimum-carries volume gate, shared with yards_per_carry
        assert m["min_vol"]["col"] == "total_carries"
        assert m["positions"] == ["RB", "QB"]


def test_weekly_wiring_supports_ranges():
    assert "ybc_per_carry" in am.WEEKLY_ADV_METRIC_COLS
    assert "yac_per_carry" in am.WEEKLY_ADV_METRIC_COLS
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["ybc_per_carry"] == "w_carries"
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["yac_per_carry"] == "w_carries"
    assert am.adv_weekly_metric_supported("ybc_per_carry") is True
    assert am.adv_weekly_metric_supported("yac_per_carry") is True
    # Weighted re-aggregation SQL is a totals ratio, not an average of ratios:
    # weeks 1-2 for p1 -> (10*3.0 + 20*4.5) / 30 = 4.0
    sql, _weight_sql = am._adv_weekly_agg_sql("ybc_per_carry")
    assert "SUM(ybc_per_carry * w_carries)" in sql
    assert "SUM(CASE WHEN ybc_per_carry IS NOT NULL THEN w_carries END)" in sql


def test_career_numeric_aggregation_covers_new_fields():
    # _NUMERIC_METRICS lives inside get_player_career_metrics; assert both
    # fields flow through it by checking the source list directly.
    import inspect
    src = inspect.getsource(am.get_player_career_metrics)
    assert "'ybc_per_carry'" in src and "'yac_per_carry'" in src
