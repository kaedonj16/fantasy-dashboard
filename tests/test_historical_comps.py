"""Comparable-player matching and smoothed board probabilities (slim CI)."""
from copy import deepcopy
from pathlib import Path

from dashboard_services.historical.comps import (
    COMP_FEATURE_FIELDS,
    build_comp_aggregates,
    extract_comp_query,
    lookup_board_probabilities,
    match_comps,
)
from dashboard_services.historical.definitions import (
    DEFAULT_BAYES_PRIOR_N,
    MIN_COMP_CELL_N,
    empirical_bayes,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(**kwargs):
    base = {
        "sleeper_id": kwargs.get("sleeper_id", "p"),
        "name": kwargs.get("name", "Player"),
        "season": kwargs.get("season", 2022),
        "position": kwargs.get("position", "WR"),
        "games": 16,
    }
    base.update(kwargs)
    return base


def test_extract_omits_missing_and_ignores_same_season_actuals():
    rookie = extract_comp_query(_row(
        years_experience=0,
        previous_season_finish=None,
        draft_capital_bucket=None,
        age=None,
        target_share=0.40,
        snap_pct=0.99,
        ppr_positional_finish=2,
        ppr_points=300,
    ))
    assert rookie["position"] == "WR"
    assert rookie["career_stage"] == "rookie"
    assert rookie["prior_finish"] == "none"
    assert "draft_capital" not in rookie
    from_round = extract_comp_query(_row(
        years_experience=0,
        draft_capital_bucket=None,
        draft_round=1,
        draft_pick=4,
    ))
    assert from_round["draft_capital"] == "round_1"
    assert "age_bucket" not in rookie
    assert "target_share" not in rookie
    assert "snap_pct" not in rookie
    assert set(rookie).issubset(set(COMP_FEATURE_FIELDS))

    veteran_missing = extract_comp_query(_row(
        years_experience=4,
        previous_season_finish=None,
        draft_capital_bucket=None,
    ))
    assert "prior_finish" not in veteran_missing
    assert "draft_capital" not in veteran_missing

    known = extract_comp_query(_row(
        years_experience=1,
        age=23.4,
        draft_capital_bucket="round_1",
        previous_season_finish=18,
        previous_season_year=2023,
        previous_season_target_share=0.22,
        previous_season_snap_pct=0.85,
        target_share=0.01,
        snap_pct=0.10,
    ))
    assert known["career_stage"] == "year_2"
    assert known["draft_capital"] == "round_1"
    assert known["age_bucket"] == "23-24"
    assert known["prior_finish"] == "top_24"
    assert known["target_share"] == "20-25%"
    assert known["snap_pct"] == "80%+"

    qb = extract_comp_query(_row(
        position="QB",
        years_experience=2,
        previous_season_target_share=0.0,
        previous_season_year=2023,
        previous_season_snap_pct=0.95,
    ))
    assert "target_share" not in qb
    assert qb["snap_pct"] == "80%+"

    early_snap = extract_comp_query(_row(
        years_experience=2,
        previous_season_year=2021,
        previous_season_snap_pct=0.90,
    ))
    assert "snap_pct" not in early_snap


def test_comp_features_do_not_leak_same_season_actuals():
    row = _row(
        sleeper_id="wr",
        season=2024,
        years_experience=2,
        age=24.1,
        draft_capital_bucket="round_1",
        previous_season_finish=20,
        previous_season_year=2023,
        previous_season_target_share=0.18,
        previous_season_snap_pct=0.70,
        target_share=0.28,
        snap_pct=0.91,
        ppr_positional_finish=8,
        ppr_points=250,
        adot=12.0,
    )
    before = extract_comp_query(row)
    mutated = deepcopy(row)
    mutated["target_share"] = 0.99
    mutated["snap_pct"] = 0.05
    mutated["ppr_positional_finish"] = 80
    mutated["ppr_points"] = 10
    mutated["adot"] = 3.0
    after = extract_comp_query(mutated)
    assert before == after
    assert after["target_share"] == "15-20%"
    assert after["snap_pct"] == "60-80%"
    assert set(before) <= set(COMP_FEATURE_FIELDS)
    assert "projected_points" not in before
    assert "adp" not in before


def test_match_comps_excludes_self_and_future_seasons():
    query = _row(
        sleeper_id="q",
        season=2022,
        years_experience=1,
        age=23.0,
        draft_capital_bucket="round_1",
        previous_season_finish=20,
        ppr_positional_finish=5,
        ppr_points=200,
    )
    pool = [
        query,
        _row(
            sleeper_id="q",
            season=2021,
            name="Self Past",
            years_experience=0,
            draft_capital_bucket="round_1",
            previous_season_finish=None,
            ppr_positional_finish=20,
        ),
        _row(
            sleeper_id="future",
            name="Future Star",
            season=2024,
            years_experience=1,
            age=23.2,
            draft_capital_bucket="round_1",
            previous_season_finish=18,
            ppr_positional_finish=3,
            ppr_points=280,
        ),
        _row(
            sleeper_id="peer",
            name="Same Year",
            season=2022,
            years_experience=1,
            age=23.4,
            draft_capital_bucket="round_1",
            previous_season_finish=19,
            ppr_positional_finish=7,
            ppr_points=210,
        ),
        _row(
            sleeper_id="old",
            name="Past Peer",
            season=2020,
            years_experience=1,
            age=23.1,
            draft_capital_bucket="round_1",
            previous_season_finish=22,
            ppr_positional_finish=9,
            ppr_points=190,
        ),
        _row(
            sleeper_id="rb",
            position="RB",
            season=2022,
            years_experience=1,
            draft_capital_bucket="round_1",
            previous_season_finish=20,
        ),
    ]
    names = {c["name"] for c in match_comps(query, pool, limit=8)}
    assert "Same Year" in names
    assert "Past Peer" in names
    assert "Future Star" not in names
    assert "Self Past" not in names
    ids = {c["sleeper_id"] for c in match_comps(query, pool)}
    assert "q" not in ids
    assert "rb" not in ids


def test_tiny_cell_relaxes_and_smooths_toward_position_baseline():
    rows = []
    # Exact cell: year-2 / round_1 / prior top_24 / age 23-24 / 20-25% share.
    # 2 hits / 4 seasons — below MIN_COMP_CELL_N.
    for i in range(2):
        rows.append(_row(
            sleeper_id=f"exact-hit-{i}",
            name=f"Exact Hit {i}",
            season=2022 + i,
            years_experience=1,
            age=23.2,
            draft_capital_bucket="round_1",
            previous_season_finish=18,
            previous_season_year=2023,
            previous_season_target_share=0.22,
            previous_season_snap_pct=0.85,
            ppr_positional_finish=4,
            ppr_points=260,
        ))
    for i in range(2):
        rows.append(_row(
            sleeper_id=f"exact-miss-{i}",
            name=f"Exact Miss {i}",
            season=2022 + i,
            years_experience=1,
            age=23.2,
            draft_capital_bucket="round_1",
            previous_season_finish=18,
            previous_season_year=2023,
            previous_season_target_share=0.22,
            previous_season_snap_pct=0.85,
            ppr_positional_finish=40,
            ppr_points=80,
        ))
    # Same stage/capital/prior, different age — fills the relaxed cell.
    for i in range(16):
        rows.append(_row(
            sleeper_id=f"older-{i}",
            name=f"Older {i}",
            season=2020,
            years_experience=1,
            age=28.0,
            draft_capital_bucket="round_1",
            previous_season_finish=20,
            previous_season_year=2023,
            previous_season_target_share=0.22,
            previous_season_snap_pct=0.85,
            ppr_positional_finish=40,
            ppr_points=70,
        ))
    # Other WRs so the position baseline is below the year-2 / round-1 cell.
    for i in range(10):
        rows.append(_row(
            sleeper_id=f"vet-{i}",
            name=f"Vet {i}",
            season=2019,
            years_experience=6,
            age=31.0,
            draft_capital_bucket="day_3",
            previous_season_finish=50,
            ppr_positional_finish=40,
            ppr_points=40,
        ))
    payload = build_comp_aggregates(rows)
    assert payload["walk_forward"] is False
    query = _row(
        sleeper_id="prospect",
        years_experience=1,
        age=23.4,
        draft_capital_bucket="round_1",
        previous_season_finish=18,
        previous_season_year=2023,
        previous_season_target_share=0.22,
        previous_season_snap_pct=0.85,
    )
    exact = lookup_board_probabilities(query, payload, min_n=1)
    assert exact["n"] == 4
    assert abs(exact["rates"]["top_12"]["raw_rate"] - 0.5) < 1e-9
    assert exact["fallback"] is False

    looked = lookup_board_probabilities(query, payload, min_n=MIN_COMP_CELL_N)
    assert looked["fallback"] is True
    assert "age_bucket" in looked["dropped"] or "target_share" in looked["dropped"]
    assert looked["n"] >= MIN_COMP_CELL_N
    assert looked["kind"] == "conditional"
    assert looked["rates"]["top_12"]["kind"] == "conditional"
    raw = looked["rates"]["top_12"]["raw_rate"]
    smoothed = looked["rates"]["top_12"]["smoothed_rate"]
    baseline = payload["by_position"]["WR"]["baseline"]["top_12"]["raw_rate"]
    assert raw is not None and smoothed is not None and baseline is not None
    assert looked["n"] == 20
    assert abs(raw - 0.10) < 1e-9
    assert abs(baseline - (2 / 30)) < 1e-5
    assert min(raw, baseline) <= smoothed <= max(raw, baseline)
    assert looked["rates"]["top_12"]["sample_size"] == looked["n"]
    expected = empirical_bayes(
        looked["rates"]["top_12"]["successes"],
        looked["n"],
        baseline * DEFAULT_BAYES_PRIOR_N,
        DEFAULT_BAYES_PRIOR_N,
    )
    assert abs(smoothed - expected) < 1e-9
    assert "prospect" not in {ex["sleeper_id"] for ex in looked["examples"]}
    assert looked["examples"]


def test_lookup_empty_position_is_none_not_zero():
    payload = build_comp_aggregates([_row(position="WR", ppr_positional_finish=1)])
    empty = lookup_board_probabilities({"position": "K"}, payload)
    assert empty["n"] == 0
    assert empty["rates"]["top_12"]["raw_rate"] is None
    assert empty["rates"]["top_12"]["smoothed_rate"] is None
    assert empty["rates"]["top_12"]["display_pct"] is None


def test_board_prob_is_conditional_not_distribution():
    rows = [
        _row(
            sleeper_id="hit",
            years_experience=2,
            draft_capital_bucket="day_2",
            previous_season_finish=8,
            ppr_positional_finish=3,
            ppr_points=300,
        ),
        _row(
            sleeper_id="miss",
            years_experience=2,
            draft_capital_bucket="day_2",
            previous_season_finish=8,
            ppr_positional_finish=40,
            ppr_points=50,
        ),
    ]
    payload = build_comp_aggregates(rows)
    looked = lookup_board_probabilities(rows[0], payload, min_n=1)
    # Conditional: 1/2 of this profile finished top-12. Distribution would be
    # "100% of top-12s had this profile" (1/1).
    assert abs(looked["rates"]["top_12"]["raw_rate"] - 0.5) < 1e-9
    assert looked["kind"] == "conditional"
    assert "distribution" not in looked
    assert payload["descriptive_only"] is True


def test_include_named_false_skips_examples():
    rows = [
        _row(sleeper_id="a", ppr_positional_finish=2, years_experience=1),
        _row(sleeper_id="b", ppr_positional_finish=40, years_experience=1),
    ]
    named = build_comp_aggregates(rows, include_named=True)
    skip = build_comp_aggregates(rows, include_named=False)
    assert named["named_examples"] is True
    assert skip["named_examples"] is False
    named_ex = named["by_position"]["WR"]["leaves"][0]["examples"]
    skip_ex = skip["by_position"]["WR"]["leaves"][0]["examples"]
    assert named_ex
    assert skip_ex == []


def test_comps_modules_stay_pure_and_skip_adp_projections():
    hist = ROOT / "dashboard_services" / "historical"
    text = (hist / "comps.py").read_text(encoding="utf-8")
    assert "import pandas" not in text
    assert "import nfl_data_py" not in text
    assert "flask" not in text.lower()
    assert "projected_" not in text
    assert "resolve_market_adp" not in text
    assert "adp_service" not in text
    assert "breakout_engine" not in text
    assert "build_player_history_features" not in text
    assert "031_" not in text
