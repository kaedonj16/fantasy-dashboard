"""Redraft roster grades ignore age and draft capital."""
from dashboard_services.ai.context_builders import calculate_roster_grade


def _players(ages, values=None):
    values = values or [800, 700, 600, 500, 400, 300, 200, 150]
    return [
        {"position": pos, "value": val, "age": age}
        for pos, val, age in zip(
            ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "TE"], values, ages
        )
    ]


def test_redraft_grade_ignores_age_and_capital():
    young = _players([22, 23, 23, 24, 24, 25, 25, 26])
    old = _players([31, 32, 30, 29, 33, 31, 28, 34])
    picks = [{"id": "2027_1_1", "display": "2027 1st"}] * 4

    young_g = calculate_roster_grade(
        young, picks,
        dynasty_pct_val=0.4,
        redraft_pct_val=0.8,
        position_ranks={"QB": 2, "RB": 3, "WR": 2, "TE": 4},
        num_teams=12,
        scoring_type="redraft",
    )
    old_g = calculate_roster_grade(
        old, [],
        dynasty_pct_val=0.4,
        redraft_pct_val=0.8,
        position_ranks={"QB": 2, "RB": 3, "WR": 2, "TE": 4},
        num_teams=12,
        scoring_type="redraft",
    )
    assert young_g["score"] == old_g["score"]
    assert young_g["grade"] == old_g["grade"]
    assert young_g["win_window"] in ("Contend", "Bubble", "Long Shot")
    assert old_g["win_window"] in ("Contend", "Bubble", "Long Shot")
    assert young_g["breakdown"]["scoring_type"] == "redraft"


def test_dynasty_grade_still_rewards_youth_and_picks():
    young = _players([22, 23, 23, 24, 24, 25, 25, 26])
    old = _players([31, 32, 30, 29, 33, 31, 28, 34])
    picks = [{"id": "2027_1_1", "display": "2027 1st"}] * 4

    young_g = calculate_roster_grade(
        young, picks,
        dynasty_pct_val=0.4,
        redraft_pct_val=0.5,
        scoring_type="dynasty",
    )
    old_g = calculate_roster_grade(
        old, [],
        dynasty_pct_val=0.4,
        redraft_pct_val=0.5,
        scoring_type="dynasty",
    )
    assert young_g["score"] > old_g["score"]
    assert young_g["win_window"] not in ("Contend", "Bubble", "Long Shot")
