"""Guards the unified PowerScore engine upgrades."""
import pytest

pd = pytest.importorskip("pandas")

from dashboard_services.power_score import (
    approximate_power_score_frame,
    blended_team_scores,
    performance_power_scores,
    season_phase_from_progress,
    starter_lineup_value,
    z_scores,
)


def test_z_scores_population():
    assert z_scores([1.0]) == [0.0]
    assert z_scores([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]
    zs = z_scores([0.0, 2.0])
    assert zs[0] == pytest.approx(-1.0)
    assert zs[1] == pytest.approx(1.0)


def test_season_phase_from_progress():
    assert season_phase_from_progress(games_played=0) == "preseason"
    assert season_phase_from_progress(games_played=3) == "early"
    assert season_phase_from_progress(games_played=7) == "mid"
    assert season_phase_from_progress(games_played=12) == "late"


def test_performance_prefers_avg_signal_over_total_pf_shape():
    # Two teams, same win% / last3 / consistency / ceiling; different AVG.
    scores = performance_power_scores(
        win_pct=[0.5, 0.5],
        avg=[100.0, 120.0],
        last3=[100.0, 100.0],
        consistency=[0.0, 0.0],
        ceiling=[130.0, 130.0],
    )
    assert scores[1] > scores[0]


def test_performance_includes_past_sos_when_provided():
    without = performance_power_scores(
        win_pct=[0.5, 0.5],
        avg=[100.0, 100.0],
        last3=[100.0, 100.0],
        consistency=[0.0, 0.0],
        ceiling=[100.0, 100.0],
    )
    with_sos = performance_power_scores(
        win_pct=[0.5, 0.5],
        avg=[100.0, 100.0],
        last3=[100.0, 100.0],
        consistency=[0.0, 0.0],
        ceiling=[100.0, 100.0],
        past_sos=[90.0, 110.0],
    )
    assert with_sos[1] > with_sos[0]
    assert without[0] == without[1]


def test_starter_lineup_value_uses_slots_when_available():
    lookup = {
        "qb": {"position": "QB", "redraft_value_1qb": 50},
        "rb1": {"position": "RB", "redraft_value_1qb": 100},
        "rb2": {"position": "RB", "redraft_value_1qb": 80},
        "wr1": {"position": "WR", "redraft_value_1qb": 90},
        "te": {"position": "TE", "redraft_value_1qb": 40},
        "bench": {"position": "WR", "redraft_value_1qb": 200},  # high but not started
    }
    # Without slots: top-8 core includes the 200 bench WR.
    no_slots = starter_lineup_value(
        list(lookup), lookup, redraft_key="redraft_value_1qb"
    )
    assert no_slots == 560.0  # all six core players

    # With a 1QB/1RB/1WR/1TE lineup, the 200 WR is a starter — but if we only
    # have one WR slot and also a FLEX, both WRs start. Use no FLEX so the
    # lower WR is excluded? Actually WR1=90 and bench=200 — with one WR slot
    # the 200 should start and 90 sit.
    with_slots = starter_lineup_value(
        list(lookup),
        lookup,
        redraft_key="redraft_value_1qb",
        roster_positions=["QB", "RB", "WR", "TE"],
    )
    # QB50 + best RB100 + best WR200 + TE40 = 390
    assert with_slots == 390.0
    assert with_slots < no_slots


def test_blended_phase_weights_shift_value_vs_record():
    # Equal everything except value vs record extremes.
    base = {
        "avg": 100.0,
        "luck_adj_win": 0.5,
        "starter_value": 100.0,
        "momentum": 0.0,
        "consistency": 0.0,
        "sos": 0.5,
    }
    teams_pre = [
        {**base, "team": "Value", "starter_value": 300.0, "luck_adj_win": 0.2},
        {**base, "team": "Record", "starter_value": 50.0, "luck_adj_win": 0.9},
    ]
    teams_late = [dict(t) for t in teams_pre]
    pre = {t["team"]: t for t in blended_team_scores(teams_pre, phase="preseason")}
    late = {t["team"]: t for t in blended_team_scores(teams_late, phase="late")}
    # Preseason: roster value dominates → Value ranks higher.
    assert pre["Value"]["rank"] < pre["Record"]["rank"]
    # Late: record dominates more → Record should close the gap / flip.
    assert late["Record"]["power_score"] > late["Value"]["power_score"]


def test_blended_includes_ros_and_playoff_when_present():
    teams = [
        {
            "avg": 100.0,
            "luck_adj_win": 0.5,
            "starter_value": 100.0,
            "momentum": 0.0,
            "consistency": 0.0,
            "sos": 0.5,
            "ros_ease": 0.8,
            "playoff_pct": 80.0,
        },
        {
            "avg": 100.0,
            "luck_adj_win": 0.5,
            "starter_value": 100.0,
            "momentum": 0.0,
            "consistency": 0.0,
            "sos": 0.5,
            "ros_ease": 0.2,
            "playoff_pct": 20.0,
        },
    ]
    out = blended_team_scores(teams, phase="mid")
    assert "ros" in out[0]["power_components"]
    assert "playoff" in out[0]["power_components"]
    assert out[0]["power_score"] > out[1]["power_score"]


def test_blended_uses_avg_not_total_pf_for_scoring_component():
    # Identical PF totals but different games → different AVG → scoring z differs.
    teams = [
        {"pf": 600.0, "avg": 100.0, "luck_adj_win": 0.5, "starter_value": 100.0,
         "momentum": 0.0, "consistency": 0.0, "sos": 0.5},
        {"pf": 600.0, "avg": 150.0, "luck_adj_win": 0.5, "starter_value": 100.0,
         "momentum": 0.0, "consistency": 0.0, "sos": 0.5},
    ]
    out = blended_team_scores(teams, phase="mid")
    assert out[0]["power_components"]["pf"] > out[1]["power_components"]["pf"]
    assert out[0]["avg"] == 150.0


def test_approximate_frame_sets_powerscore():
    df = pd.DataFrame({
        "owner": ["A", "B"],
        "Win%": [0.8, 0.2],
        "AVG": [120.0, 90.0],
        "MAX": [150.0, 110.0],
        "STD": [10.0, 20.0],
    })
    out = approximate_power_score_frame(df)
    assert "PowerScore" in out.columns
    assert out.loc[0, "PowerScore"] > out.loc[1, "PowerScore"]
