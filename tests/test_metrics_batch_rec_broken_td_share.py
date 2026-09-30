"""Family F tests: PFR receiving broken tackles per reception + TD share.

rec_broken_tackles_per_reception = PFR receiving_broken_tackles / receptions
(receiving plays only, never the combined rush+rec figure; the PFR reader
keeps explicit zero rows because a charted zero is a real 0.00, not missing).
td_share = (rush TDs + rec TDs) / team offensive TDs; every offensive TD is
exactly one player's rush or rec TD, so pass TDs are never added (that
would double count every passing TD).
"""

import csv
import sys
import types

import pytest

pytest.importorskip("pandas")

import pandas as pd

import data_building.external_data.nflverse_metrics as nvm

QB, WR = "00-0000001", "00-0000002"
SL = {QB: "1001", WR: "1002"}


class _FakeNfl(types.SimpleNamespace):
    def __init__(self, pbp=None):
        super().__init__()
        self._pbp = pbp

    def import_ftn_data(self, years):
        return pd.DataFrame({"nflverse_game_id": [], "nflverse_play_id": []})

    def import_pbp_data(self, years, columns=None, downcast=False):
        return self._pbp.copy()

    def import_ngs_data(self, stat_type=None, years=None):
        return pd.DataFrame({"season_type": [], "week": []})


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(nvm, "_gsis_to_sleeper", lambda: dict(SL))
    monkeypatch.setattr(nvm, "build_pfr_contact_yards_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_catchable_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_broken_tackles_weekly", lambda season: {})


def _write_rec_csv(tmp_path, rows, with_column=True):
    path = tmp_path / "pfr_rec.csv"
    fields = ["pfr_player_id", "week", "game_type"]
    if with_column:
        fields.append("receiving_broken_tackles")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(path)


def test_rec_broken_reader_keeps_explicit_zeros(tmp_path, monkeypatch):
    path = _write_rec_csv(tmp_path, [
        {"pfr_player_id": "PFR2", "week": 1, "game_type": "REG",
         "receiving_broken_tackles": 3},
        {"pfr_player_id": "PFR2", "week": 2, "game_type": "REG",
         "receiving_broken_tackles": 0},
        {"pfr_player_id": "PFR2", "week": 3, "game_type": "POST",
         "receiving_broken_tackles": 9},
    ])
    monkeypatch.setattr(nvm, "download_pfr_advstats_rec_csv", lambda season: path)
    monkeypatch.setattr(nvm, "_pfr_to_sleeper", lambda: {"PFR2": "1002"})
    out = nvm.build_pfr_rec_broken_tackles_weekly(2025)
    assert out[("1002", 1)] == 3.0
    # The charted zero week is KEPT (a real 0.00 per reception downstream).
    assert out[("1002", 2)] == 0.0
    # Postseason rows are excluded.
    assert ("1002", 3) not in out


def test_rec_broken_reader_missing_column_skips(tmp_path, monkeypatch, capsys):
    path = _write_rec_csv(tmp_path, [
        {"pfr_player_id": "PFR2", "week": 1, "game_type": "REG"},
    ], with_column=False)
    monkeypatch.setattr(nvm, "download_pfr_advstats_rec_csv", lambda season: path)
    monkeypatch.setattr(nvm, "_pfr_to_sleeper", lambda: {"PFR2": "1002"})
    out = nvm.build_pfr_rec_broken_tackles_weekly(2025)
    assert out == {}
    assert "WARNING" in capsys.readouterr().out


def test_rec_broken_weekly_apply_and_integration(monkeypatch):
    # Direct apply: 2 broken tackles on 4 receptions = 0.5; no receptions
    # means the metric stays absent.
    out = {("1002", 1): {"w_receptions": 4.0}, ("1003", 1): {"w_receptions": 0.0}}
    nvm._apply_rec_broken_tackles_weekly(out, {("1002", 1): 2.0, ("1003", 1): 1.0})
    assert out[("1002", 1)]["rec_broken_tackles_per_reception"] == 0.5
    assert "rec_broken_tackles_per_reception" not in out[("1003", 1)]

    # End to end through the weekly builder: WR catches 4 of 5 targets.
    rows = []
    for i in range(5):
        rows.append({
            "game_id": "g1", "play_id": i + 1, "week": 1, "season_type": "REG",
            "play_type": "pass", "epa": 0.1, "qb_epa": 0.1, "cpoe": 1.0,
            "success": 1, "sack": 0, "qb_scramble": 0, "qb_hit": 0,
            "rush_attempt": 0, "pass_attempt": 1, "qb_dropback": 1,
            "rushing_yards": 0, "air_yards": 8, "yards_gained": 8,
            "complete_pass": 0 if i == 4 else 1, "yards_after_catch": 2,
            "passing_yards": 8, "pass_touchdown": 0, "interception": 0,
            "first_down_pass": 0, "first_down_rush": 0,
            "posteam": "KC", "yardline_100": 50, "down": 1,
            "third_down_converted": 0,
            "passer_player_id": QB, "rusher_player_id": None,
            "receiver_player_id": WR,
        })
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(pbp=pd.DataFrame(rows)))
    monkeypatch.setattr(nvm, "build_pfr_rec_broken_tackles_weekly",
                        lambda season: {("1002", 1): 2.0})
    weekly = nvm.build_nflverse_weekly_metrics_for_season(2025)
    assert weekly[("1002", 1)]["rec_broken_tackles_per_reception"] == 0.5


def test_rec_broken_season_merge_totals_first(tmp_path, monkeypatch):
    from data_building import advanced_metrics as am

    path = _write_rec_csv(tmp_path, [
        {"pfr_player_id": "PFR2", "week": 1, "game_type": "REG",
         "receiving_broken_tackles": 5},
        {"pfr_player_id": "PFR2", "week": 2, "game_type": "REG",
         "receiving_broken_tackles": 3},
        {"pfr_player_id": "PFR2", "week": 9, "game_type": "REG",
         "receiving_broken_tackles": 100},
    ])
    monkeypatch.setattr(nvm, "download_pfr_advstats_rec_csv", lambda season: path)
    monkeypatch.setattr(nvm, "_pfr_to_sleeper", lambda: {"PFR2": "1002"})
    usage_map = {
        "1002": {"total_receptions": 40.0, "games": 8},
        "1003": {"total_receptions": 0.0, "games": 8},
    }
    merged = am._merge_pfr_rec_broken_tackles(usage_map, 2025, completed_week=2)
    assert merged == 1
    # (5 + 3) / 40 = 0.2; the week-9 row is beyond completed_week.
    assert usage_map["1002"]["rec_broken_tackles_per_reception"] == 0.2
    # Zero receptions: absent, never 0.0.
    assert "rec_broken_tackles_per_reception" not in usage_map["1003"]


def _week(**kw):
    base = {"ppr_pts": 10.0, "snaps": 50.0, "rz_targets": 0.0, "rz_carries": 0.0,
            "rec_tds": 0.0, "rush_tds": 0.0, "pass_tds": 0.0,
            "snap_pct": 80.0, "targets": 5.0, "touches": 5.0}
    base.update(kw)
    return base


def test_td_share_no_double_count(monkeypatch):
    import data_building.weekly_metrics as wm
    from data_building import advanced_metrics as am

    series = {
        # WR: 2 rec TDs + 1 rush TD. QB on the same team: 3 pass TDs (the
        # same scores as the WR's rec TDs) + 1 rush TD.
        "1002": [_week(rec_tds=2.0, rush_tds=1.0)],
        "1001": [_week(rec_tds=0.0, rush_tds=1.0, pass_tds=3.0)],
        # Scoreless team: denominator zero means absent, never 0.0.
        "1003": [_week()],
    }
    monkeypatch.setattr(wm, "get_weekly_series_by_player",
                        lambda season, through_week: series)
    metrics = [{"player_id": "1002", "position": "WR"},
               {"player_id": "1003", "position": "WR"}]
    usage = [{"id": "1002", "team": "KC"}, {"id": "1001", "team": "KC"},
             {"id": "1003", "team": "NYJ"}]
    am.finalize_weekly_series_metrics(metrics, usage, 2025, 3)
    by_pid = {m["player_id"]: m for m in metrics}
    # Team total = 2 + 1 + 1 = 4 (pass TDs NOT added). WR share = 3/4.
    assert by_pid["1002"]["td_share"] == pytest.approx(0.75)
    assert by_pid["1003"]["td_share"] is None


def test_family_f_specs_free_and_wired():
    from data_building import advanced_metrics as am

    spec = am.LEADERBOARD_METRICS["rec_broken_tackles_per_reception"]
    assert spec["label"] == "Broken Tackles / Catch"
    assert spec["category"] == "Receiving"
    assert not spec.get("pct")
    td = am.LEADERBOARD_METRICS["td_share"]
    assert td["label"] == "TD Share"
    assert td["pct"] is True and td["pct_frac"] is True
    w = am._ADV_WEEKLY_WEIGHTED_METRICS
    assert w["rec_broken_tackles_per_reception"] == "w_receptions"
    # td_share is season-only (weekly-series derived like the RZ shares).
    assert "td_share" not in am.WEEKLY_ADV_METRIC_COLS
    # Both are in the career aggregation list (a local in
    # get_player_career_metrics).
    import inspect
    career_src = inspect.getsource(am.get_player_career_metrics)
    assert "'td_share'" in career_src
    assert "'rec_broken_tackles_per_reception'" in career_src
    for key in ("rec_broken_tackles_per_reception", "td_share"):
        assert key not in am.PRO_METRICS
        assert key not in am.PREMIUM_METRICS
