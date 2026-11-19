"""Aggregation tests against a real 2026 PBP slice.

Fixture: tests/fixtures/pbp_slice_2026.csv, extracted from the real
nflverse play_by_play_2026.parquet (weeks 1-3): all pass/run plays for
D. Prescott, C. Lamb and J. Gibbs plus a deterministic league baseline
sample. GSIS ids are used as player keys (identity crosswalk).
"""
import csv
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from dashboard_services import player_roles as pr  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "pbp_slice_2026.csv"


def _rows():
    with open(FIXTURE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ids(rows):
    qb = next(r["passer_player_id"] for r in rows
              if r["passer_player_name"] == "D.Prescott")
    wr = next(r["receiver_player_id"] for r in rows
              if r["receiver_player_name"] == "C.Lamb")
    rb = next(r["rusher_player_id"] for r in rows
              if r["rusher_player_name"] == "J.Gibbs")
    return qb, wr, rb


@pytest.fixture(scope="module")
def tables():
    rows = _rows()
    weeks = {int(float(r["week"])) for r in rows}
    return pr.build_pbp_tables_from_rows(
        rows, {}, last4_weeks={max(weeks)}, base_weeks=weeks - {max(weeks)})


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def test_zone_shares_sum_to_100(tables):
    _qb, wr, _rb = _ids(_rows())
    zones = tables["receiving"]["zones"][wr]
    total = sum(c["primary"] for c in zones.values())
    assert total > 0
    share_sum = sum(c["primary"] / total * 100.0 for c in zones.values())
    assert share_sum == pytest.approx(100.0, abs=0.01)


def test_receiver_catch_rate_matches_raw(tables):
    rows = _rows()
    _qb, wr, _rb = _ids(rows)
    raw = [r for r in rows if r["play_type"] == "pass"
           and r["receiver_player_id"] == wr and _num(r["qb_spike"]) != 1
           and r["pass_location"] in ("left", "middle", "right")]
    expected = sum(1 for r in raw if _num(r["complete_pass"]) == 1) / len(raw)
    zones = tables["receiving"]["zones"][wr]
    att = sum(c["primary"] for c in zones.values())
    comp = sum(c["secondary"] for c in zones.values())
    assert att == len(raw)
    assert comp / att == pytest.approx(expected, abs=1e-9)


def test_qb_rating_by_zone_matches_raw(tables):
    rows = _rows()
    qb, _wr, _rb = _ids(rows)
    zones = tables["passing"]["zones"][qb]
    # Recompute one zone's rating straight from the raw slice.
    cell_key = max(zones, key=lambda k: zones[k]["primary"])
    raw = [r for r in rows if r["play_type"] == "pass"
           and r["passer_player_id"] == qb and _num(r["qb_spike"]) != 1
           and r["receiver_player_id"] not in ("", "nan")
           and r["pass_location"] == cell_key[1]
           and pr.zone_row_for_air_yards(_num(r["air_yards"])) == cell_key[0]]
    att = len(raw)
    comp = sum(1 for r in raw if _num(r["complete_pass"]) == 1)
    yds = sum(_num(r["yards_gained"]) for r in raw)
    td = sum(1 for r in raw if _num(r["pass_touchdown"]) == 1)
    inter = sum(1 for r in raw if _num(r["interception"]) == 1)
    assert zones[cell_key]["primary"] == att
    assert pr.passer_rating(att, comp, yds, td, inter) == pytest.approx(
        pr.passer_rating(zones[cell_key]["primary"], zones[cell_key]["secondary"],
                         zones[cell_key]["yards"], zones[cell_key]["td"],
                         zones[cell_key]["int"]), abs=0.05)


def test_lane_sums_and_rates_match_raw(tables):
    rows = _rows()
    _qb, _wr, rb = _ids(rows)
    raw = [r for r in rows if r["play_type"] == "run"
           and r["rusher_player_id"] == rb and _num(r["qb_kneel"]) != 1
           and _num(r["qb_spike"]) != 1 and _num(r["two_point_attempt"]) != 1]
    lanes = tables["rushing"]["lanes"][rb]
    carries = sum(c["carries"] for c in lanes.values())
    assert carries == len(raw)
    yards = sum(_num(r["yards_gained"]) for r in raw)
    assert sum(c["yards"] for c in lanes.values()) == pytest.approx(yards)
    stuffed = sum(1 for r in raw if _num(r["yards_gained"]) <= 0)
    assert sum(c["stuffed"] for c in lanes.values()) == stuffed
    explosive = sum(1 for r in raw if _num(r["yards_gained"]) >= 10)
    assert sum(c["explosive"] for c in lanes.values()) == explosive
    success = sum(1 for r in raw if pr.is_success(
        r["down"], r["ydstogo"], r["yards_gained"]))
    assert sum(c["success"] for c in lanes.values()) == success
    share_sum = sum(c["carries"] / carries * 100.0 for c in lanes.values())
    assert share_sum == pytest.approx(100.0, abs=0.01)


def test_middle_lane_ignores_gap(tables):
    rows = _rows()
    _qb, _wr, rb = _ids(rows)
    raw_mid = [r for r in rows if r["play_type"] == "run"
               and r["rusher_player_id"] == rb
               and r["run_location"] == "middle"
               and _num(r["qb_kneel"]) != 1 and _num(r["two_point_attempt"]) != 1]
    lanes = tables["rushing"]["lanes"][rb]
    assert lanes["M"]["carries"] == len(raw_mid)


def test_league_baselines_exist(tables):
    league_zones = tables["receiving"]["league"]
    assert sum(c["primary"] for c in league_zones.values()) > 100
    league_lanes = tables["rushing"]["league"]
    assert sum(c["carries"] for c in league_lanes.values()) > 50


def test_pbp_feats_present(tables):
    _qb, wr, rb = _ids(_rows())
    assert "deep_rate" in tables["feats"][wr]
    assert "explosive_run_rate" in tables["feats"][rb]
