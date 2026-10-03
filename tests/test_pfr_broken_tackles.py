"""Broken Tackles (PFR advstats) behind the avoided_tackles column.

The public Broken Tackles / Broken Tackles/Carry metrics were empty on the
live site: their only source was a manual PFF export sync, and the daily
snapshot build never wrote the column at all. They are now sourced from
PFR's rushing_broken_tackles charting (nflverse pfr_advstats release),
merged by the same pass that fills YBC/YAC per carry, and the column left
PREMIUM_METRICS (the drop_rate / contested_catch_rate precedent: a metric
populated from free data is public-safe).

Covers, with no network / no nfl_data_py:
- _merge_pfr_contact_yards totals rushing_broken_tackles only (the rush
  file's receiving_broken_tackles column and the rec file never count
  toward a per-carry rushing metric)
- completed_week cutoff, never clobbers, missing column fabricates nothing
- calculate_rushing_metrics passes the total through
- metadata: honest Broken Tackles labels, PFR credit, not premium-gated
- save_metrics_snapshot actually persists the column (INSERT + params +
  monotonic-counter guard)
"""

import csv
import inspect

import pytest

from data_building import advanced_metrics as am
from data_building.external_data import nflverse_metrics as nm


_CSV_ROWS = [
    # week 1: 4 rushing broken tackles; the 9 receiving ones in the same
    # rush-file row must NOT leak into a rushing metric.
    {"pfr_player_id": "p1", "week": "1", "game_type": "REG",
     "carries": "10", "rushing_yards_before_contact": "30",
     "rushing_yards_after_contact": "15",
     "rushing_broken_tackles": "4", "receiving_broken_tackles": "9"},
    {"pfr_player_id": "p2", "week": "1", "game_type": "REG",
     "carries": "5", "rushing_yards_before_contact": "-5",
     "rushing_yards_after_contact": "40",
     "rushing_broken_tackles": "1", "receiving_broken_tackles": "0"},
    # week 2
    {"pfr_player_id": "p1", "week": "2", "game_type": "REG",
     "carries": "20", "rushing_yards_before_contact": "90",
     "rushing_yards_after_contact": "30",
     "rushing_broken_tackles": "3", "receiving_broken_tackles": "2"},
    # noise: preseason production never counts
    {"pfr_player_id": "p1", "week": "1", "game_type": "PRE",
     "carries": "8", "rushing_yards_before_contact": "80",
     "rushing_yards_after_contact": "80",
     "rushing_broken_tackles": "7", "receiving_broken_tackles": "7"},
]

_FIELDNAMES = ["pfr_player_id", "week", "game_type", "carries",
               "rushing_yards_before_contact", "rushing_yards_after_contact",
               "rushing_broken_tackles", "receiving_broken_tackles"]

_CROSSWALK = {"p1": "1234", "p2": "5678"}


def _write_csv(tmp_path, rows, fieldnames, name="advstats_week_rush_2026.csv"):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


@pytest.fixture()
def stub_btk_sources(monkeypatch, tmp_path):
    path = _write_csv(tmp_path, _CSV_ROWS, _FIELDNAMES)
    monkeypatch.setattr(nm, "download_pfr_advstats_rush_csv",
                        lambda season, max_age_hours=6.0: path)
    monkeypatch.setattr(nm, "_pfr_to_sleeper", lambda: dict(_CROSSWALK))
    return path


def test_merge_totals_rushing_broken_tackles_only(stub_btk_sources):
    usage_map = {"1234": {"games": 2}, "5678": {"games": 1}}
    merged = am._merge_pfr_contact_yards(usage_map, 2026, completed_week=2)
    assert merged == 2
    # 4 + 3 rushing over weeks 1-2; receiving (9 + 2) and preseason (7) excluded.
    assert usage_map["1234"]["avoided_tackles"] == pytest.approx(7.0)
    assert usage_map["5678"]["avoided_tackles"] == pytest.approx(1.0)
    # The per-carry derivative the leaderboard computes from the total.
    assert usage_map["1234"]["avoided_tackles"] / 30 == pytest.approx(7 / 30)


def test_merge_broken_tackles_respects_completed_week(stub_btk_sources):
    usage_map = {"1234": {"games": 1}}
    am._merge_pfr_contact_yards(usage_map, 2026, completed_week=1)
    assert usage_map["1234"]["avoided_tackles"] == pytest.approx(4.0)


def test_merge_broken_tackles_never_clobbers(stub_btk_sources):
    # A real PFF import (or an earlier fill) wins over the PFR merge.
    usage_map = {"1234": {"games": 2, "avoided_tackles": 42.0,
                          "ybc_per_carry": 9.99, "yac_per_carry": 1.11}}
    am._merge_pfr_contact_yards(usage_map, 2026, completed_week=2)
    assert usage_map["1234"]["avoided_tackles"] == pytest.approx(42.0)


def test_merge_without_broken_tackles_column_fabricates_nothing(monkeypatch, tmp_path):
    legacy_fields = [c for c in _FIELDNAMES
                     if c not in ("rushing_broken_tackles", "receiving_broken_tackles")]
    legacy_rows = [{k: r[k] for k in legacy_fields} for r in _CSV_ROWS]
    path = _write_csv(tmp_path, legacy_rows, legacy_fields)
    monkeypatch.setattr(nm, "download_pfr_advstats_rush_csv",
                        lambda season, max_age_hours=6.0: path)
    monkeypatch.setattr(nm, "_pfr_to_sleeper", lambda: dict(_CROSSWALK))
    usage_map = {"1234": {"games": 2}}
    am._merge_pfr_contact_yards(usage_map, 2026, completed_week=2)
    assert "avoided_tackles" not in usage_map["1234"]
    # The contact splits still fill from the legacy file.
    assert usage_map["1234"]["ybc_per_carry"] == pytest.approx(120 / 30)


def test_calculate_rushing_metrics_passes_broken_tackles_through():
    out = am.calculate_rushing_metrics({
        "avg_carries": 20.0, "avg_rush_yards": 90.0, "avg_rush_tds": 0.5,
        "avg_targets": 3.0, "avg_receptions": 2.0, "avg_rec_yards": 15.0,
        "avoided_tackles": 11.0,
    })
    assert out["avoided_tackles"] == pytest.approx(11.0)
    out = am.calculate_rushing_metrics({"avg_carries": 20.0})
    assert out["avoided_tackles"] is None


def test_metadata_names_the_actual_stat_and_source():
    per_carry = am.LEADERBOARD_METRICS["avoided_tackles_per_carry"]
    assert per_carry["label"] == "Broken Tackles/Carry"
    assert per_carry["category"] == "Rushing"
    assert "PFR" in per_carry["desc"]
    assert "PFF" not in per_carry["desc"]
    # The derivative still divides the stored total by the volume column.
    assert per_carry["computed_sql"] == "m.avoided_tackles::float / NULLIF(v.vol, 0)"
    assert per_carry["computed_null"] == "m.avoided_tackles IS NOT NULL"


def test_broken_tackles_is_public_now_that_pfr_sources_it():
    assert "avoided_tackles_per_carry" not in am.PREMIUM_METRICS


def test_snapshot_save_persists_broken_tackles():
    src = inspect.getsource(am.save_metrics_snapshot)
    insert = src[src.index("INSERT INTO player_advanced_metrics"):]
    columns = insert[:insert.index("VALUES")]
    assert "avoided_tackles" in columns
    assert 'metrics.get("avoided_tackles")' in src
    # A PFR-outage build must not wipe a stored total on conflict.
    assert "avoided_tackles = COALESCE(EXCLUDED.avoided_tackles" in src
    assert "avoided_tackles" in am._MONOTONIC_SNAPSHOT_COUNTERS
