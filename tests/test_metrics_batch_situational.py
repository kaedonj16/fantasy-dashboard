"""Family D tests: situational play-by-play metrics.

goal_line_opp_share  = player carries+targets inside the opponent 5 /
                       team total of the same
end_zone_target_rate = targets with air_yards >= yardline_100 / targets
deep_target_rate     = targets with air_yards >= 20 / targets
stuffed_rate         = carries with yards_gained <= 0 / carries
third_down_conv_rate = third-down dropbacks producing a first down /
                       third-down dropbacks
All totals-first; zero denominators leave the metric absent.
"""

import sys
import types

import pytest

pytest.importorskip("pandas")

import pandas as pd

import data_building.external_data.nflverse_metrics as nvm

QB, WR, RB, RB2 = "00-0000001", "00-0000002", "00-0000003", "00-0000004"
SL = {QB: "1001", WR: "1002", RB: "1003", RB2: "1004"}


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
        "posteam": "KC", "yardline_100": 50, "down": 1,
        "third_down_converted": 0,
        "passer_player_id": None, "rusher_player_id": None,
        "receiver_player_id": None,
    }
    base.update(kw)
    return base


def _frame():
    plays = []
    pid = 0

    def add(**kw):
        nonlocal pid
        pid += 1
        plays.append(_play(play_id=pid, **kw))

    # RB: 10 KC carries; yards 5,0,-2,3,1,0,7,2,4,1 (3 stuffed); first
    # three snapped inside the 5.
    for i, (yds, yl) in enumerate([(5, 3), (0, 2), (-2, 4), (3, 50), (1, 50),
                                   (0, 50), (7, 50), (2, 50), (4, 50), (1, 50)]):
        add(play_type="run", pass_attempt=0, rush_attempt=1, qb_dropback=0,
            complete_pass=0, rusher_player_id=RB, rushing_yards=yds,
            yards_gained=yds, yardline_100=yl)
    # WR: 10 KC targets (air, yardline): deep 4, end-zone 3, goal-line 2.
    for air, yl in [(25, 40), (22, 20), (5, 4), (30, 45), (2, 3),
                    (18, 15), (40, 55), (3, 60), (12, 30), (8, 25)]:
        add(passer_player_id="00-0099999", receiver_player_id=WR,
            air_yards=air, yardline_100=yl)
    # One more KC goal-line target to a receiver outside the crosswalk:
    # team GL total = 3 (RB) + 2 (WR) + 1 = 6.
    add(passer_player_id="00-0099999", receiver_player_id="00-0099998",
        air_yards=3, yardline_100=2)
    # QB: 10 dropbacks, 4 on third down; conversions via first_down_pass,
    # third_down_converted, and a first_down_rush scramble = 3 of 4.
    for i in range(10):
        kw = {}
        if i < 4:
            kw["down"] = 3
        if i == 0:
            kw["first_down_pass"] = 1
        if i == 1:
            kw["third_down_converted"] = 1
        if i == 2:
            kw["first_down_rush"] = 1
        add(passer_player_id=QB, **kw)
    # RB2 on NYJ: 5 midfield carries; NYJ has no goal-line plays at all.
    for i in range(5):
        add(play_type="run", pass_attempt=0, rush_attempt=1, qb_dropback=0,
            complete_pass=0, rusher_player_id=RB2, rushing_yards=3,
            yards_gained=3, posteam="NYJ", yardline_100=50)
    return pd.DataFrame(plays)


def test_season_situational_metrics(monkeypatch):
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(pbp=_frame()))
    out = nvm.build_pbp_metrics_for_season(2024)
    assert out["1003"]["stuffed_rate"] == pytest.approx(30.0)
    assert out["1003"]["goal_line_opp_share"] == pytest.approx(50.0)
    assert out["1002"]["goal_line_opp_share"] == pytest.approx(33.3)
    assert out["1002"]["deep_target_rate"] == pytest.approx(40.0)
    assert out["1002"]["end_zone_target_rate"] == pytest.approx(30.0)
    assert out["1001"]["third_down_conv_rate"] == pytest.approx(75.0)
    # NYJ never reached the goal line: no denominator, no fabricated share.
    assert "goal_line_opp_share" not in out.get("1004", {})


def test_weekly_situational_metrics_and_weights(monkeypatch):
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(pbp=_frame()))
    out = nvm.build_nflverse_weekly_metrics_for_season(2024)
    assert out[("1003", 1)]["stuffed_rate"] == pytest.approx(30.0)
    assert out[("1003", 1)]["goal_line_opp_share"] == pytest.approx(50.0)
    assert out[("1003", 1)]["w_team_gl_opps"] == pytest.approx(6.0)
    assert out[("1002", 1)]["deep_target_rate"] == pytest.approx(40.0)
    assert out[("1002", 1)]["end_zone_target_rate"] == pytest.approx(30.0)
    assert out[("1001", 1)]["third_down_conv_rate"] == pytest.approx(75.0)
    assert out[("1001", 1)]["w_third_down_dropbacks"] == pytest.approx(4.0)


def test_family_d_specs_free_and_wired():
    from data_building import advanced_metrics as am

    assert am.LEADERBOARD_METRICS["goal_line_opp_share"]["category"] == "General"
    assert am.LEADERBOARD_METRICS["goal_line_opp_share"]["positions"] == ["RB", "WR", "TE"]
    assert am.LEADERBOARD_METRICS["end_zone_target_rate"]["category"] == "Receiving"
    assert am.LEADERBOARD_METRICS["deep_target_rate"]["category"] == "Receiving"
    assert am.LEADERBOARD_METRICS["stuffed_rate"]["lower_better"] is True
    assert am.LEADERBOARD_METRICS["third_down_conv_rate"]["category"] == "Passing"
    w = am._ADV_WEEKLY_WEIGHTED_METRICS
    assert w["goal_line_opp_share"] == "w_team_gl_opps"
    assert w["third_down_conv_rate"] == "w_third_down_dropbacks"
    assert w["stuffed_rate"] == "w_carries"
    assert w["end_zone_target_rate"] == "w_targets"
    assert w["deep_target_rate"] == "w_targets"
    assert "w_team_gl_opps" in am.WEEKLY_ADV_WEIGHT_COLS
    assert "w_third_down_dropbacks" in am.WEEKLY_ADV_WEIGHT_COLS
    for key in ("goal_line_opp_share", "end_zone_target_rate", "deep_target_rate",
                "stuffed_rate", "third_down_conv_rate"):
        assert key not in am.PRO_METRICS
        assert key not in am.PREMIUM_METRICS
