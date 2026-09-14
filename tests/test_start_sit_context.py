from utils.start_sit_context import expected_plays_context, role_confidence_from_trend
from utils.start_sit_score import compute_start_score


def test_real_team_volume_reaches_scorer_and_missing_stays_neutral():
    teams = {"NE": {"off_plays_pg": 68}, "NYJ": {"plays_faced_pg": 66, "plays_faced_l4_pg": 67}}
    ctx = expected_plays_context(teams, "NWE", "NYJ", 64)
    _, factors, _ = compute_start_score(
        10, expected_team_plays=ctx["expected_team_plays"], league_average_plays=ctx["league_average_plays"])
    assert factors["expected_plays"] > 1
    missing = expected_plays_context({}, "NE", "NYJ", 64)
    _, neutral, _ = compute_start_score(10, expected_team_plays=missing.get("expected_team_plays"),
                                        league_average_plays=missing.get("league_average_plays"))
    assert neutral["expected_plays"] == 1.0


def test_role_confidence_recognizes_stability_and_confirmed_promotion():
    assert role_confidence_from_trend({"series": [8, 8, 8]}) == 1.0
    assert role_confidence_from_trend({"series": [2, 3, 12, 14]}) >= 0.75
    assert role_confidence_from_trend({"series": [2]}) is None
