"""Tests for the shared player distribution model.

The module reads real cache files; tests monkeypatch the data accessors so
they run hermetically.
"""

import math

import pytest

from data_building import player_distributions as pd


def _rows(scores):
    """Build fake raw weekly rows from fixed-map fantasy points.

    We invert the fixed map through rec yards: 1 pt ~= 10 rec yards.
    """
    return [{"rec_yd": s * 10.0, "rec": 1.0, "rec_tgt": 2.0} for s in scores]


@pytest.fixture
def hermetic(monkeypatch):
    cur = {
        "p1": _rows([10, 12, 8, 15, 9]),       # steady-ish WR
        "p2": _rows([4, 25, 3, 30, 2]),        # boom/bust WR
        "p3": _rows([14, 16, 15]),             # 3-game player (small sample)
    }
    hist = {
        "p1": _rows([11, 9, 13, 10, 12, 8]),
        "p2": _rows([12, 11, 13, 10]),
        "p3": [],
    }
    players = {
        "p1": {"team": "KC", "position": "WR", "injury_status": ""},
        "p2": {"team": "KC", "position": "WR", "injury_status": ""},
        "p3": {"team": "BUF", "position": "WR", "injury_status": "Questionable"},
        "pk": {"team": "KC", "position": "K", "injury_status": ""},
    }

    def fake_week_files(season):
        if season == 2026:
            return cur
        if season == 2025:
            return hist
        return {}

    monkeypatch.setattr(pd, "_week_files", fake_week_files)
    monkeypatch.setattr(pd, "_players_index", lambda: players)
    # Clear the single-player cache between tests.
    pd._PROFILE_CACHE.clear()
    return players


def _req(pid, pos="WR", mean=12.0):
    return {"player_id": pid, "pos": pos, "mean": mean}


def test_mean_passthrough_never_modified(hermetic):
    profs = pd.build_profiles([_req("p1", mean=17.3)], 2026, 4)
    assert profs["p1"]["mean"] == 17.3


def test_steady_vs_volatile_spread(hermetic):
    profs = pd.build_profiles([_req("p1"), _req("p2")], 2026, 4)
    # Same mean, but p2's scores swing wildly -> wider std.
    assert profs["p2"]["std"] > profs["p1"]["std"]
    assert profs["p2"]["factors"].get("profile") == "boom-or-bust"
    assert profs["p1"]["factors"].get("profile") == "steady"


def test_std_rescales_with_mean(hermetic):
    a = pd.build_profiles([_req("p1", mean=10.0)], 2026, 4)["p1"]["std"]
    b = pd.build_profiles([_req("p1", mean=20.0)], 2026, 4)["p1"]["std"]
    assert b > a  # shape preserved, size follows the projection


def test_td_dependency_raises_skew(hermetic, monkeypatch):
    # All points from TDs: 2 rec TDs a week.
    rows = [{"rec_td": 2.0, "rec_tgt": 3.0} for _ in range(5)]
    monkeypatch.setattr(pd, "_week_files", lambda season: {"ptd": rows} if season == 2026 else {})
    profs = pd.build_profiles([_req("ptd", mean=14.0)], 2026, 4)
    p = profs["ptd"]
    assert p["skew_alpha"] > 2.0
    assert p["factors"]["td_share"] > 0.5


def test_questionable_sets_dud_risk(hermetic):
    profs = pd.build_profiles([_req("p3", mean=11.0)], 2026, 4)
    assert profs["p3"]["dud_risk"] == pytest.approx(0.12)
    assert profs["p3"]["factors"].get("questionable") is True


def test_teammate_out_narrows_spread(hermetic):
    base = pd.build_profiles([_req("p1", mean=12.0)], 2026, 4)["p1"]["std"]
    with_out = pd.build_profiles(
        [_req("p1", mean=12.0)], 2026, 4, ctx={"teammate_out": {"p1": True}}
    )["p1"]["std"]
    assert with_out < base


def test_qb_out_widens_pass_catchers(hermetic):
    base = pd.build_profiles([_req("p1", mean=12.0)], 2026, 4)["p1"]["std"]
    wid = pd.build_profiles(
        [_req("p1", mean=12.0)], 2026, 4, ctx={"qb_out": {"KC": True}}
    )["p1"]["std"]
    assert wid > base


def test_kicker_falls_back_to_baseline(hermetic):
    profs = pd.build_profiles([_req("pk", pos="K", mean=8.0)], 2026, 4)
    assert profs["pk"]["std"] == pytest.approx(0.00 * 8.0 + 4.0)
    assert profs["pk"]["n_games"] == 0.0


def test_unknown_player_gets_baseline(hermetic):
    profs = pd.build_profiles([_req("ghost", pos="RB", mean=9.0)], 2026, 4)
    assert profs["ghost"]["std"] == pytest.approx(0.45 * 9.0 + 2.0)


def test_role_change_flag_recorded(hermetic):
    profs = pd.build_profiles(
        [_req("p1", mean=12.0)], 2026, 4, ctx={"role_change": {"p1": True}}
    )
    assert profs["p1"]["factors"].get("role_change") is True


def test_game_script_favored_rb_narrows(hermetic):
    base = pd.build_profiles([_req("p1", pos="RB", mean=14.0)], 2026, 4)["p1"]["std"]
    fav = pd.build_profiles(
        [_req("p1", pos="RB", mean=14.0)], 2026, 4, ctx={"vegas_spread": {"KC": 9.5}}
    )["p1"]["std"]
    assert fav < base


def test_correlation_pairs_same_team(hermetic):
    players = hermetic
    players["pqb"] = {"team": "KC", "position": "QB", "injury_status": ""}
    pairs = pd.correlation_pairs(["p1", "p2", "p3", "pqb"], 2026)
    assert pairs[("p1", "pqb")] == pytest.approx(0.30)
    # p3 is BUF: no cross-team correlation.
    assert all("p3" not in pair for pair in pairs)


def test_signature_stable_and_week_sensitive(hermetic):
    s1 = pd.profile_inputs_signature(2026, 4)
    s2 = pd.profile_inputs_signature(2026, 4)
    s3 = pd.profile_inputs_signature(2026, 5)
    assert s1 == s2
    assert s1 != s3


def test_std_bounded(hermetic):
    profs = pd.build_profiles([_req("p2", mean=60.0)], 2026, 4)
    assert 1.0 <= profs["p2"]["std"] <= 30.0
    assert math.isfinite(profs["p2"]["std"])


def test_spike_is_season_best_game(hermetic):
    # Fake rows add 1.0 pt per reception on top of the yardage score.
    profs = pd.build_profiles([_req("p1"), _req("p2"), _req("p3")], 2026, 4)
    assert profs["p1"]["spike"] == 16.0   # max of [10,12,8,15,9] + 1
    assert profs["p2"]["spike"] == 31.0   # max of [4,25,3,30,2] + 1
    assert profs["p3"]["spike"] == 17.0   # max of [14,16,15] + 1


def test_spike_none_for_single_game(hermetic, monkeypatch):
    # One played game is a fluke guard: no spike recorded.
    monkeypatch.setattr(
        pd, "_week_files",
        lambda season: {"px": _rows([22])} if season == 2026 else {},
    )
    pd._PROFILE_CACHE.clear()
    profs = pd.build_profiles([_req("px")], 2026, 4)
    assert profs["px"]["spike"] is None


def test_spike_none_for_kicker_baseline(hermetic):
    profs = pd.build_profiles([_req("pk", pos="K", mean=8.0)], 2026, 4)
    assert profs["pk"]["spike"] is None
