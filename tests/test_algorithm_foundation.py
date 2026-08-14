from utils.injury_lineup import healthy_lineup_total
from utils.market_plausibility import market_plausibility
from utils.optimal_lineup import compute_optimal_lineup, slot_eligibility
from utils.simulation import ScenarioBank
from utils.standings import resolve_standings


def test_exact_assignment_avoids_greedy_restricted_flex_trap():
    values = {"dual": 20, "wr": 19}
    positions = {"dual": ["RB", "WR"], "wr": "WR"}
    starters, total = compute_optimal_lineup(values, positions, ["WR_TE", "WR_RB"], values)
    assert starters == {"dual", "wr"} and total == 39


def test_provider_aliases_have_explicit_distinct_eligibility():
    assert slot_eligibility("WRTE_FLEX") == {"WR", "TE"}
    assert slot_eligibility("WRRB_FLEX") == {"WR", "RB"}
    assert slot_eligibility("OP") == {"QB", "RB", "WR", "TE"}


def test_injuries_are_removed_simultaneously_before_reoptimization():
    v = {"rb1": 20, "rb2": 15, "rb3": 10}
    p = {x: "RB" for x in v}
    assert healthy_lineup_total(v, p, ["RB", "RB"], v, ["rb1", "rb2"])[1] == 10


def test_scenario_bank_is_reproducible_and_correlated():
    a = ScenarioBank.create(200, 3, 4, seed=17).weekly_draws()
    b = ScenarioBank.create(200, 3, 4, seed=17).weekly_draws()
    assert a == b


def test_standings_are_lexicographic_not_float_packed():
    rows = [{"roster_id": "a", "wins": 9, "points_for": 1},
            {"roster_id": "b", "wins": 8, "points_for": 99999999}]
    assert resolve_standings(rows)[0]["roster_id"] == "a"


def test_unlabeled_data_is_truthfully_market_plausibility():
    result = market_plausibility(100, 100)
    assert result["label"] == "market plausibility" and not result["calibrated"]
