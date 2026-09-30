"""Family E tests: FTN charting additions.

turnover_worthy_rate = is_interception_worthy dropbacks / dropbacks (QB)
contested_target_rate = is_contested_ball targets / charted targets
screen_target_rate    = is_screen_pass targets / charted targets
Same FTN merges as drop_rate / contested_catch_rate, season and weekly.
A missing FTN flag column skips the metric (no fabricated zeros).
"""

import sys
import types

import pytest

pytest.importorskip("pandas")

import pandas as pd

import data_building.external_data.nflverse_metrics as nvm

QB, WR = "00-0000001", "00-0000002"
SL = {QB: "1001", WR: "1002"}


class _FakeNfl(types.SimpleNamespace):
    def __init__(self, ftn=None, pbp=None):
        super().__init__()
        self._ftn = ftn
        self._pbp = pbp

    def import_ftn_data(self, years):
        return self._ftn.copy()

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
    monkeypatch.setattr(nvm, "build_pfr_rec_broken_tackles_weekly", lambda season: {})


def _pbp_row(play_id, **kw):
    base = {
        "game_id": "g1", "play_id": play_id, "week": 1, "season_type": "REG",
        "play_type": "pass", "epa": 0.1, "qb_epa": 0.1, "cpoe": 1.0,
        "success": 1, "sack": 0, "qb_scramble": 0, "qb_hit": 0,
        "rush_attempt": 0, "pass_attempt": 1, "qb_dropback": 1,
        "rushing_yards": 0, "air_yards": 8, "yards_gained": 8,
        "complete_pass": 1, "yards_after_catch": 2, "passing_yards": 8,
        "pass_touchdown": 0, "interception": 0,
        "first_down_pass": 0, "first_down_rush": 0,
        "posteam": "KC", "yardline_100": 50, "down": 1,
        "third_down_converted": 0,
        "passer_player_id": QB, "rusher_player_id": None,
        "receiver_player_id": WR,
    }
    base.update(kw)
    return base


def _ftn_row(play_id, **kw):
    base = {
        "nflverse_game_id": "g1", "nflverse_play_id": play_id,
        "is_catchable_ball": 1, "is_drop": 0, "is_contested_ball": 0,
        "is_screen_pass": 0, "is_interception_worthy": 0,
        "is_throw_away": 0, "is_play_action": 0, "is_qb_out_of_pocket": 0,
        "n_blitzers": 0, "n_defense_box": 6,
    }
    base.update(kw)
    return base


def _frames():
    # 10 charted WR targets thrown by QB: contested 4 (2 caught), screens 2,
    # 1 drop, QB: 1 turnover-worthy throw in 10.
    pbp, ftn = [], []
    for i in range(10):
        pid = i + 1
        pbp.append(_pbp_row(pid, complete_pass=0 if i in (0, 1, 5) else 1))
        ftn.append(_ftn_row(
            pid,
            is_contested_ball=1 if i < 4 else 0,
            is_screen_pass=1 if i in (8, 9) else 0,
            is_interception_worthy=1 if i == 0 else 0,
            is_drop=1 if i == 5 else 0,
        ))
    return pd.DataFrame(ftn), pd.DataFrame(pbp)


def test_ftn_season_trio(monkeypatch):
    ftn, pbp = _frames()
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(ftn=ftn, pbp=pbp))
    out = nvm.build_ftn_charting_for_season(2025)
    assert out["1002"]["contested_target_rate"] == pytest.approx(40.0)
    assert out["1002"]["screen_target_rate"] == pytest.approx(20.0)
    # Existing contested CATCH rate is a different metric and still works:
    # 2 caught of 4 contested.
    assert out["1002"]["contested_catch_rate"] == pytest.approx(50.0)
    assert out["1001"]["turnover_worthy_rate"] == pytest.approx(10.0)


def test_ftn_weekly_trio(monkeypatch):
    ftn, pbp = _frames()
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(ftn=ftn, pbp=pbp))
    out = nvm.build_nflverse_weekly_metrics_for_season(2025)
    assert out[("1002", 1)]["contested_target_rate"] == pytest.approx(40.0)
    assert out[("1002", 1)]["screen_target_rate"] == pytest.approx(20.0)
    assert out[("1001", 1)]["turnover_worthy_rate"] == pytest.approx(10.0)


def test_ftn_missing_flag_column_skips_metric(monkeypatch):
    ftn, pbp = _frames()
    ftn = ftn.drop(columns=["is_interception_worthy"])
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(ftn=ftn, pbp=pbp))
    out = nvm.build_ftn_charting_for_season(2025)
    assert "turnover_worthy_rate" not in out.get("1001", {})
    # Other FTN outputs are unaffected.
    assert out["1002"]["contested_target_rate"] == pytest.approx(40.0)


def test_family_e_specs_free_and_wired():
    from data_building import advanced_metrics as am

    spec = am.LEADERBOARD_METRICS["turnover_worthy_rate"]
    assert spec["label"] == "Turnover-Worthy %"
    assert spec["lower_better"] is True
    assert am.LEADERBOARD_METRICS["contested_target_rate"]["label"] == "Contested Tgt %"
    assert am.LEADERBOARD_METRICS["screen_target_rate"]["label"] == "Screen Tgt %"
    # Distinct from the pre-existing contested CATCH rate.
    assert "contested_catch_rate" in am.LEADERBOARD_METRICS
    w = am._ADV_WEEKLY_WEIGHTED_METRICS
    assert w["turnover_worthy_rate"] == "w_dropbacks"
    assert w["contested_target_rate"] == "w_targets"
    assert w["screen_target_rate"] == "w_targets"
    for key in ("turnover_worthy_rate", "contested_target_rate", "screen_target_rate"):
        assert key not in am.PRO_METRICS
        assert key not in am.PREMIUM_METRICS
