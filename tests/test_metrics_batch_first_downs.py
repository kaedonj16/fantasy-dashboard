"""Family B tests: first-down metrics from nflverse play-by-play.

rec_first_down_rate  = first_down_pass on targets / targets
rush_first_down_rate = first_down_rush / carries
first_downs (weekly) / total_first_downs (season) = passer + rusher +
receiver first downs from the player's own plays, each play counted once.
All ratios are totals-first; a zero denominator leaves the metric absent.
"""

import sys
import types

import pytest

pytest.importorskip("pandas")

import pandas as pd

import data_building.external_data.nflverse_metrics as nvm

QB, WR, RB = "00-0000001", "00-0000002", "00-0000003"
SL = {QB: "1001", WR: "1002", RB: "1003"}


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


def _frame():
    plays = []
    pid = 0
    # QB: 10 dropbacks, 4 first downs passing; 4 carries, 1 rushing FD.
    for i in range(10):
        pid += 1
        plays.append(_play(play_id=pid, passer_player_id=QB,
                           first_down_pass=1 if i < 4 else 0))
    for i in range(4):
        pid += 1
        plays.append(_play(play_id=pid, play_type="run", pass_attempt=0,
                           rush_attempt=1, qb_dropback=0, complete_pass=0,
                           rusher_player_id=QB, rushing_yards=5,
                           first_down_rush=1 if i == 0 else 0))
    # WR: 10 targets, 6 first downs.
    for i in range(10):
        pid += 1
        plays.append(_play(play_id=pid, passer_player_id="00-0099999",
                           receiver_player_id=WR,
                           first_down_pass=1 if i < 6 else 0))
    # RB: 10 carries, 3 first downs.
    for i in range(10):
        pid += 1
        plays.append(_play(play_id=pid, play_type="run", pass_attempt=0,
                           rush_attempt=1, qb_dropback=0, complete_pass=0,
                           rusher_player_id=RB, rushing_yards=4,
                           first_down_rush=1 if i < 3 else 0))
    return pd.DataFrame(plays)


def test_season_first_down_metrics(monkeypatch):
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(pbp=_frame()))
    out = nvm.build_pbp_metrics_for_season(2024)
    assert out["1002"]["rec_first_down_rate"] == pytest.approx(60.0)
    assert out["1003"]["rush_first_down_rate"] == pytest.approx(30.0)
    # Totals: QB 4 pass + 1 rush, WR 6, RB 3. No double counting.
    assert out["1001"]["total_first_downs"] == 5
    assert out["1002"]["total_first_downs"] == 6
    assert out["1003"]["total_first_downs"] == 3


def test_weekly_first_down_metrics(monkeypatch):
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(pbp=_frame()))
    out = nvm.build_nflverse_weekly_metrics_for_season(2024)
    assert out[("1002", 1)]["rec_first_down_rate"] == pytest.approx(60.0)
    assert out[("1003", 1)]["rush_first_down_rate"] == pytest.approx(30.0)
    assert out[("1001", 1)]["first_downs"] == pytest.approx(5.0)
    assert out[("1002", 1)]["first_downs"] == pytest.approx(6.0)


def test_zero_denominator_rush_rate_absent(monkeypatch):
    # A "rusher" whose rows carry rush_attempt = 0 has no carries: the rate
    # must be absent, not a fabricated 0.0.
    plays = [_play(play_id=1, play_type="pass", rusher_player_id=RB,
                   rush_attempt=0, first_down_rush=0)]
    monkeypatch.setitem(sys.modules, "nfl_data_py",
                        _FakeNfl(pbp=pd.DataFrame(plays)))
    out = nvm.build_pbp_metrics_for_season(2024)
    assert "rush_first_down_rate" not in out.get("1003", {})


def test_first_down_specs_free_and_wired():
    from data_building import advanced_metrics as am

    assert am.LEADERBOARD_METRICS["rec_first_down_rate"]["category"] == "Receiving"
    assert am.LEADERBOARD_METRICS["rec_first_down_rate"]["positions"] == ["WR", "RB", "TE"]
    assert am.LEADERBOARD_METRICS["rush_first_down_rate"]["category"] == "Rushing"
    assert am.LEADERBOARD_METRICS["rush_first_down_rate"]["positions"] == ["RB", "QB"]
    spec = am.LEADERBOARD_METRICS["first_downs_per_game"]
    assert spec["category"] == "General"
    assert "total_first_downs" in spec["computed_sql"]
    for key in ("rec_first_down_rate", "rush_first_down_rate", "first_downs_per_game"):
        assert key not in am.PRO_METRICS
        assert key not in am.PREMIUM_METRICS
