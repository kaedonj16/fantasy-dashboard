"""Family C tests: QB rating when targeted + PFR pressure rate faced.

qb_rating_when_targeted applies the standard NFL passer rating formula to
the AGGREGATE counting stats of throws targeting a receiver (totals first,
never an average of per-play/per-week ratings).

pressure_rate_faced = PFR times_pressured / (attempts + times_sacked),
totals first, season and weekly. A PFR file missing the pressure columns
skips the metric loudly instead of fabricating zeros.
"""

import sys
import types

import pytest

pytest.importorskip("pandas")

import pandas as pd

import data_building.external_data.nflverse_metrics as nvm

_REAL_CATCHABLE_WEEKLY = nvm.build_pfr_catchable_weekly

QB, WR = "00-0000001", "00-0000002"
SL = {QB: "1001", WR: "1002"}


class _FakeNfl(types.SimpleNamespace):
    def __init__(self, pbp=None):
        super().__init__()
        self._pbp = pbp

    def import_pbp_data(self, years, columns=None, downcast=False):
        return self._pbp.copy()

    def import_ngs_data(self, stat_type=None, years=None):
        return pd.DataFrame({"season_type": [], "week": []})

    def import_ftn_data(self, years):
        return pd.DataFrame()


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(nvm, "_gsis_to_sleeper", lambda: dict(SL))
    monkeypatch.setattr(nvm, "build_pfr_contact_yards_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_catchable_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_broken_tackles_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_rec_broken_tackles_weekly", lambda season: {})


def _play(**kw):
    base = {
        "game_id": "g1", "play_id": 1, "week": 1, "season_type": "REG",
        "play_type": "pass", "epa": 0.1, "qb_epa": 0.1, "cpoe": 1.0,
        "success": 1, "sack": 0, "qb_scramble": 0, "qb_hit": 0,
        "rush_attempt": 0, "pass_attempt": 1, "qb_dropback": 1,
        "rushing_yards": 0, "air_yards": 8, "yards_gained": 8,
        "complete_pass": 1, "yards_after_catch": 2, "passing_yards": 8,
        "pass_touchdown": 0, "interception": 0,
        "first_down_pass": 0, "first_down_rush": 0,
        "passer_player_id": None, "rusher_player_id": None,
        "receiver_player_id": None,
    }
    base.update(kw)
    return base


def test_passer_rating_helper():
    assert nvm._passer_rating(10, 10, 200, 3, 0) == pytest.approx(158.3)
    assert nvm._passer_rating(0, 0, 0, 0, 0) is None
    # Totals-first: rating of the aggregates, not the mean of two ratings.
    agg = nvm._passer_rating(20, 12, 200, 2, 1)
    mean = (nvm._passer_rating(10, 8, 150, 2, 0)
            + nvm._passer_rating(10, 4, 50, 0, 1)) / 2
    assert agg != pytest.approx(mean)


def _target_frame():
    # WR: 10 targets, 8 completions, 120 yards on completions, 2 TD, 1 INT.
    plays = []
    for i in range(10):
        comp = i < 8
        plays.append(_play(
            play_id=i + 1, passer_player_id=QB, receiver_player_id=WR,
            complete_pass=1 if comp else 0,
            yards_gained=15 if comp else 0,
            yards_after_catch=3 if comp else 0,
            pass_touchdown=1 if i < 2 else 0,
            interception=1 if i == 9 else 0,
        ))
    return pd.DataFrame(plays)


def test_qb_rating_when_targeted_season_and_weekly(monkeypatch):
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(pbp=_target_frame()))
    expected = nvm._passer_rating(10, 8, 120, 2, 1)
    season = nvm.build_pbp_metrics_for_season(2024)
    assert season["1002"]["qb_rating_when_targeted"] == pytest.approx(expected)
    weekly = nvm.build_nflverse_weekly_metrics_for_season(2024)
    assert weekly[("1002", 1)]["qb_rating_when_targeted"] == pytest.approx(expected)


def test_weekly_pressure_opps_weight(monkeypatch):
    # QB: 10 attempts + 2 sacks = 12 pressure opportunities.
    plays = []
    for i in range(12):
        plays.append(_play(play_id=i + 1, passer_player_id=QB,
                           pass_attempt=0 if i >= 10 else 1,
                           sack=1 if i >= 10 else 0))
    monkeypatch.setitem(sys.modules, "nfl_data_py",
                        _FakeNfl(pbp=pd.DataFrame(plays)))
    weekly = nvm.build_nflverse_weekly_metrics_for_season(2024)
    assert weekly[("1001", 1)]["w_pressure_opps"] == pytest.approx(12.0)


def test_apply_pressure_weekly():
    out = {("1001", 1): {"w_pressure_opps": 12.0}, ("1001", 2): {}}
    counts = {("1001", 1): {"pressured": 3.0}, ("1001", 2): {"pressured": 1.0}}
    nvm._apply_pressure_weekly(out, counts)
    assert out[("1001", 1)]["pressure_rate_faced"] == pytest.approx(25.0)
    # No denominator that week: skipped, not fabricated.
    assert "pressure_rate_faced" not in out[("1001", 2)]


_PASS_HEADER = "player,pfr_player_id,week,game_type,passing_bad_throws,times_pressured,times_sacked\n"
_REC_HEADER = "player,pfr_player_id,week,game_type,receiving_drop\n"


def _write_csv(tmp_path, name, header, rows):
    p = tmp_path / name
    p.write_text(header + "".join(rows))
    return str(p)


def test_pfr_weekly_reader_pressure_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(nvm, "build_pfr_catchable_weekly", _REAL_CATCHABLE_WEEKLY)
    pass_csv = _write_csv(tmp_path, "pass.csv", _PASS_HEADER, [
        "QB One,PFRQB,1,REG,2,7,1\n",
        "QB One,PFRQB,2,REG,1,5,2\n",
    ])
    rec_csv = _write_csv(tmp_path, "rec.csv", _REC_HEADER, [])
    monkeypatch.setattr(nvm, "download_pfr_advstats_pass_csv", lambda s: pass_csv)
    monkeypatch.setattr(nvm, "download_pfr_advstats_rec_csv", lambda s: rec_csv)
    monkeypatch.setattr(nvm, "_pfr_to_sleeper", lambda: {"PFRQB": "1001"})
    out = nvm.build_pfr_catchable_weekly(2024)
    assert out[("1001", 1)]["pressured"] == pytest.approx(7.0)
    assert out[("1001", 1)]["sacked_taken"] == pytest.approx(1.0)
    assert out[("1001", 2)]["pressured"] == pytest.approx(5.0)


def test_pfr_weekly_reader_missing_pressure_column_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(nvm, "build_pfr_catchable_weekly", _REAL_CATCHABLE_WEEKLY)
    pass_csv = _write_csv(
        tmp_path, "pass.csv",
        "player,pfr_player_id,week,game_type,passing_bad_throws\n",
        ["QB One,PFRQB,1,REG,2\n"])
    rec_csv = _write_csv(tmp_path, "rec.csv", _REC_HEADER, [])
    monkeypatch.setattr(nvm, "download_pfr_advstats_pass_csv", lambda s: pass_csv)
    monkeypatch.setattr(nvm, "download_pfr_advstats_rec_csv", lambda s: rec_csv)
    monkeypatch.setattr(nvm, "_pfr_to_sleeper", lambda: {"PFRQB": "1001"})
    out = nvm.build_pfr_catchable_weekly(2024)
    assert out[("1001", 1)]["bad_throws"] == pytest.approx(2.0)
    assert "pressured" not in out[("1001", 1)]


def test_season_pressure_merge_totals_first(tmp_path, monkeypatch):
    from data_building import advanced_metrics as am

    pass_csv = _write_csv(tmp_path, "pass.csv", _PASS_HEADER, [
        "QB One,PFRQB,1,REG,4,18,6\n",
        "QB One,PFRQB,2,REG,6,12,4\n",
    ])
    rec_csv = _write_csv(tmp_path, "rec.csv", _REC_HEADER, [])
    monkeypatch.setattr(nvm, "download_pfr_advstats_pass_csv", lambda s: pass_csv)
    monkeypatch.setattr(nvm, "download_pfr_advstats_rec_csv", lambda s: rec_csv)
    monkeypatch.setattr(nvm, "_pfr_to_sleeper", lambda: {"PFRQB": "1001"})
    usage_map = {"1001": {"catchable_pass_pct": None,
                          "catchable_tgt_pct": None,
                          "pressure_rate_faced": None,
                          "total_pass_att": 100.0, "avg_pass_att": 0,
                          "games": 2}}
    am._merge_pfr_catchable(usage_map, 2024, completed_week=17)
    # Totals first: pressured 30 / (100 att + 10 sacked) = 27.3.
    assert usage_map["1001"]["pressure_rate_faced"] == pytest.approx(27.3)
    # The catchable merge still works off the same file: (100 - 10) / 100.
    assert usage_map["1001"]["catchable_pass_pct"] == pytest.approx(90.0)


def test_family_c_specs_free_and_wired():
    from data_building import advanced_metrics as am

    spec = am.LEADERBOARD_METRICS["pressure_rate_faced"]
    assert spec["label"] == "Pressure Rate Faced"
    assert spec["category"] == "Passing"
    assert spec["lower_better"] is True
    spec = am.LEADERBOARD_METRICS["qb_rating_when_targeted"]
    assert spec["label"] == "QB Rating vs Tgt"
    assert spec["category"] == "Receiving"
    assert spec["positions"] == ["WR", "RB", "TE"]
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["pressure_rate_faced"] == "w_pressure_opps"
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["qb_rating_when_targeted"] == "w_targets"
    assert "w_pressure_opps" in am.WEEKLY_ADV_WEIGHT_COLS
    for key in ("pressure_rate_faced", "qb_rating_when_targeted"):
        assert key not in am.PRO_METRICS
        assert key not in am.PREMIUM_METRICS
