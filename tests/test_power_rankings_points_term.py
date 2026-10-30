"""Guards the points-based power rankings roster term.

In-season rank is 0.60 all-play + 0.40 current-roster lineup points
(``starter_ppg_available``: the injury-adjusted optimal lineup's projected
points in league scoring, averaged over the remaining weeks) when any team
carries a points term. A mid-season trade for a top scorer moves the
ranking before the new roster plays. When no team carries a points term
(projection maps unavailable), the blend falls back to the legacy
0.80 all-play + 0.20 injury-adjusted starter value. Preseason stays on
value either way.
"""
import pytest

pd = pytest.importorskip("pandas")

from dashboard_services.power_score import blended_team_scores

# starter_lineup_points / available_starter_lineup_points are imported
# inside the unit tests so the blend-ordering tests below still fail with
# plain assertion errors on pre-change main (not a collection error).


# ── lineup points unit behavior ─────────────────────────────────────────────

_PTS = {"qb": 22.0, "rb1": 18.0, "rb2": 12.0, "wr1": 16.0, "wr2": 11.0,
        "te": 9.0, "flex": 10.0, "bench_rb": 13.0, "bench_wr": 8.0}
_POS = {"qb": "QB", "rb1": "RB", "rb2": "RB", "wr1": "WR", "wr2": "WR",
        "te": "TE", "flex": "WR", "bench_rb": "RB", "bench_wr": "WR"}
_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
_ALL = list(_PTS)


def test_lineup_points_optimal_fill():
    from dashboard_services.power_score import starter_lineup_points
    total = starter_lineup_points(_ALL, _PTS, _POS, roster_positions=_SLOTS)
    # FLEX takes bench_rb (13.0) over the remaining WRs (10.0 / 8.0).
    assert total == 22.0 + 18.0 + 12.0 + 16.0 + 11.0 + 9.0 + 13.0


def test_lineup_points_bye_player_cannot_be_picked():
    from dashboard_services.power_score import starter_lineup_points
    # rb1 has no projection this week (bye): he is absent from the map and
    # the bench refills his slot instead of his points counting anyway.
    pts = {k: v for k, v in _PTS.items() if k != "rb1"}
    total = starter_lineup_points(_ALL, pts, _POS, roster_positions=_SLOTS)
    assert total == 22.0 + 13.0 + 12.0 + 16.0 + 11.0 + 9.0 + 10.0


def test_lineup_points_ir_star_refills_from_bench():
    from dashboard_services.power_score import (
        available_starter_lineup_points,
        starter_lineup_points,
    )
    adj = available_starter_lineup_points(
        _ALL, _PTS, _POS, roster_positions=_SLOTS,
        injury_by_pid={"rb1": "IR"},
    )
    assert adj == 22.0 + 13.0 + 12.0 + 16.0 + 11.0 + 9.0 + 10.0
    raw = starter_lineup_points(_ALL, _PTS, _POS, roster_positions=_SLOTS)
    assert adj < raw


def test_lineup_points_questionable_counts_in_full():
    from dashboard_services.power_score import (
        available_starter_lineup_points,
        starter_lineup_points,
    )
    adj = available_starter_lineup_points(
        _ALL, _PTS, _POS, roster_positions=_SLOTS,
        injury_by_pid={"rb1": "Questionable"},
    )
    assert adj == starter_lineup_points(_ALL, _PTS, _POS, roster_positions=_SLOTS)


def test_lineup_points_no_projections_is_zero():
    from dashboard_services.power_score import starter_lineup_points
    assert starter_lineup_points(_ALL, {}, _POS, roster_positions=_SLOTS) == 0.0


# ── blend ordering: the #2-buys-#1's-scorer scenario ────────────────────────

_OWNERS = ["Seller", "Buyer"] + [f"Team{i}" for i in range(3, 11)]
_ALLPLAY = [0.620, 0.580, 0.560, 0.540, 0.520, 0.500, 0.470, 0.440, 0.410, 0.380]
_AVG = [128.0, 124.0, 121.0, 119.0, 117.0, 114.0, 111.0, 108.0, 105.0, 102.0]
_LINEUP_PPG = [128.0, 124.0, 121.0, 119.0, 117.0, 114.0, 111.0, 108.0, 105.0, 102.0]
_VALUES = [5200.0, 5000.0, 4900.0, 4850.0, 4800.0, 4700.0, 4600.0, 4550.0, 4450.0, 4350.0]


def _teams(lineup_ppg, with_points=True):
    out = []
    for i, name in enumerate(_OWNERS):
        t = {
            "team_name": name, "luck_adj_win": _ALLPLAY[i], "avg": _AVG[i],
            "starter_value_available": _VALUES[i],
        }
        if with_points:
            t["starter_ppg_available"] = lineup_ppg[i]
        out.append(t)
    return out


def test_top_scorer_trade_flips_number_two_over_number_one():
    before = blended_team_scores(_teams(_LINEUP_PPG), phase="mid")
    assert before[0]["team_name"] == "Seller"

    # Buyer acquires Seller's 24.5 PPG scorer for picks: Buyer +10.5 lineup
    # PPG (displaces a 14.0 flex), Seller -9.5 (bench refills at 15.0).
    after_ppg = list(_LINEUP_PPG)
    after_ppg[0] -= 9.5
    after_ppg[1] += 10.5
    after = blended_team_scores(_teams(after_ppg), phase="mid")
    assert after[0]["team_name"] == "Buyer"
    assert after[0]["power_components"]["points"] > after[1]["power_components"]["points"]


def test_points_term_outweighs_value_gap():
    # Buyer has the worse starter VALUE but the better lineup points at
    # equal all-play: the points term, not value, must rank Buyer first.
    teams = _teams([120.0, 130.0] + _LINEUP_PPG[2:])
    teams[0]["luck_adj_win"] = teams[1]["luck_adj_win"] = 0.55
    teams[0]["avg"] = teams[1]["avg"] = 120.0
    ranked = blended_team_scores(teams, phase="mid")
    assert ranked[0]["team_name"] == "Buyer"


def test_no_points_anywhere_falls_back_to_legacy_value_blend():
    # Same trade expressed in value only (no points term anywhere): the
    # legacy 0.80/0.20 value blend closes the gap but does not flip Buyer.
    vals = list(_VALUES)
    vals[0] -= 300.0
    vals[1] += 250.0
    teams = _teams(_LINEUP_PPG, with_points=False)
    for i, t in enumerate(teams):
        t["starter_value_available"] = vals[i]
    ranked = blended_team_scores(teams, phase="mid")
    assert ranked[0]["team_name"] == "Seller"
    assert "points" not in ranked[0]["power_components"]


def test_preseason_ignores_points_term():
    teams = _teams([90.0, 140.0] + _LINEUP_PPG[2:])
    teams[0]["luck_adj_win"] = teams[1]["luck_adj_win"] = 0.5
    teams[0]["avg"] = teams[1]["avg"] = 100.0
    ranked = blended_team_scores(teams, phase="preseason")
    # Seller's starters carry the higher value; preseason ranks on value.
    assert ranked[0]["team_name"] == "Seller"
