"""Walk-forward History vs Market vs baseline, Pick Score gate (slim CI)."""
from pathlib import Path

from dashboard_services.historical.comps import (
    build_comp_aggregates,
    lookup_board_probabilities,
)
from dashboard_services.historical.signals import probability_from_rate
from dashboard_services.historical.walkforward import (
    brier_score,
    evaluate_pick_score_gate,
    roc_auc,
    run_walk_forward,
    split_train_test,
)


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "dashboard_services" / "historical"


def _row(**kwargs):
    base = {
        "sleeper_id": kwargs.get("sleeper_id", "p"),
        "name": kwargs.get("name", "Player"),
        "season": kwargs.get("season", 2020),
        "position": kwargs.get("position", "WR"),
        "games": 16,
        "years_experience": 1,
        "age": 24.0,
        "draft_capital_bucket": "round_1",
        "previous_season_finish": 20,
        "previously_top12": False,
        "prior_top12_count": 0,
    }
    base.update(kwargs)
    return base


def test_roc_auc_perfect_ties_and_missing():
    assert roc_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
    assert roc_auc([0.9], [1]) is None
    assert roc_auc([None, 0.9, None, 0.1], [1, 1, 0, 0]) == 1.0
    # Missing scores are skipped, never treated as 0.
    assert roc_auc([None, None], [1, 0]) is None


def test_brier_skips_missing_not_zero():
    assert abs(brier_score([1.0, 0.0], [1, 0]) - 0.0) < 1e-12
    assert abs(brier_score([0.0, 1.0], [1, 0]) - 1.0) < 1e-12
    assert brier_score([None, None], [1, 0]) is None
    # A missing 1.0 hit is not scored as p=0.
    only_miss = brier_score([None, 0.0], [1, 0])
    assert abs(only_miss - 0.0) < 1e-12


def test_split_train_is_strictly_before_test_season():
    rows = [
        _row(sleeper_id="a", season=2019),
        _row(sleeper_id="b", season=2020),
        _row(sleeper_id="c", season=2021),
        _row(sleeper_id="d", season=2022),
    ]
    train, test = split_train_test(rows, 2021)
    assert {r["season"] for r in train} == {2019, 2020}
    assert {r["season"] for r in test} == {2021}
    assert all(r["season"] < 2021 for r in train)


def _fold(season, *, hist_auc, market_auc, n_both=80, n_hist=80, n_pos=8):
    return {
        "test_season": season,
        "by_label": {
            "league_winner": {
                "hist_auc": hist_auc,
                "market_auc": market_auc,
                "n_both": n_both,
                "n_hist": n_hist,
                "n_pos": n_pos,
            }
        },
    }


def test_pick_score_gate_fails_when_hist_does_not_beat_market():
    folds = [
        _fold(2021, hist_auc=0.62, market_auc=0.61),
        _fold(2022, hist_auc=0.58, market_auc=0.60),
        _fold(2023, hist_auc=0.54, market_auc=0.50),
    ]
    verdict = evaluate_pick_score_gate(folds)
    assert verdict["validated"] is False
    assert verdict["in_live_ranking"] is False
    assert verdict["qualifying_seasons"] == []


def test_pick_score_gate_passes_only_with_margin_and_coverage():
    folds = [
        _fold(2021, hist_auc=0.62, market_auc=0.55),
        _fold(2022, hist_auc=0.60, market_auc=0.56),
        _fold(2023, hist_auc=0.66, market_auc=0.58),
    ]
    verdict = evaluate_pick_score_gate(folds)
    assert verdict["validated"] is True
    assert verdict["qualifying_seasons"] == [2021, 2022, 2023]
    assert verdict["in_live_ranking"] is False


def test_pick_score_gate_fails_thin_seasons_even_if_auc_looks_good():
    folds = [
        _fold(2021, hist_auc=0.90, market_auc=0.50, n_both=10, n_hist=10),
        _fold(2022, hist_auc=0.90, market_auc=0.50, n_both=10, n_hist=10),
        _fold(2023, hist_auc=0.90, market_auc=0.50, n_both=10, n_hist=10),
    ]
    verdict = evaluate_pick_score_gate(folds)
    assert verdict["validated"] is False


def _profile_rows(season, n, *, hit, adp, prefix):
    rows = []
    for i in range(n):
        rows.append(_row(
            sleeper_id=f"{prefix}{i}",
            season=season,
            ppr_positional_finish=2 if hit else 40,
            ppr_points=250 if hit else 40,
            adp_overall=adp,
            adp_source="sleeper",
            adp_bucket="round_1" if adp < 13 else "rounds_8_10",
        ))
    return rows


def test_walkforward_does_not_leak_test_season_outcomes():
    # 2020: this profile never hits. 2021: the same profile always hits.
    # If 2021 outcomes leaked into train, History P would jump toward 1.
    train_miss = _profile_rows(2020, 20, hit=False, adp=90.0, prefix="m")
    test_hit = _profile_rows(2021, 20, hit=True, adp=90.0, prefix="h")
    rows = train_miss + test_hit
    wf = run_walk_forward(rows, test_seasons=(2021,))
    assert wf["n_folds"] == 1
    fold = wf["folds"][0]
    assert fold["n_train"] == 20
    assert fold["n_test"] == 20
    assert fold["train_season_range"] == [2020, 2020]

    comps_train = build_comp_aggregates(train_miss, include_named=False)
    comps_all = build_comp_aggregates(rows, include_named=False)
    query = test_hit[0]
    p_train = probability_from_rate(
        lookup_board_probabilities(query, comps_train, min_n=1)["rates"]["top_5"]
    )
    p_all = probability_from_rate(
        lookup_board_probabilities(query, comps_all, min_n=1)["rates"]["top_5"]
    )
    assert p_train is not None and p_all is not None
    assert p_all > p_train
    # Walk-forward History P is the train-only (low) rate, not the leaked mix.
    # With n=20 the cell may relax; raw train rate is 0 so smoothed P is small.
    assert p_train < 0.25
    assert wf["pick_score"]["validated"] is False
    assert wf["not_a_second_engine"] is True
    assert wf["pick_score"]["in_live_ranking"] is False


def test_walkforward_skips_missing_market_p():
    rows = []
    for season in (2020, 2021):
        for i in range(6):
            rows.append(_row(
                sleeper_id=f"adp{season}-{i}",
                season=season,
                ppr_positional_finish=3 if i < 2 else 40,
                ppr_points=200 if i < 2 else 30,
                adp_overall=8.0 if i < 3 else None,
                adp_source="sleeper" if i < 3 else None,
            ))
    wf = run_walk_forward(rows, test_seasons=(2021,))
    block = wf["folds"][0]["by_label"]["league_winner"]
    assert block["n_test"] == 6
    assert block["n_hist"] == 6
    assert block["n_market"] == 3
    assert block["n_both"] == 3


def test_walkforward_module_stays_pure():
    text = (HIST / "walkforward.py").read_text(encoding="utf-8")
    assert "import pandas" not in text
    assert "import nfl_data_py" not in text
    assert "import flask" not in text.lower()
    assert "import sklearn" not in text
    assert "from sklearn" not in text
    assert "breakout_engine" not in text
    assert "build_player_history_features" not in text
    assert "031_" not in text
    assert "static/pick_score" not in text
    assert "from utils.projection_resolver" not in text
    pick = (ROOT / "static" / "pick_score.js").read_text(encoding="utf-8")
    py = (ROOT / "utils" / "pick_score.py").read_text(encoding="utf-8")
    core = (ROOT / "static" / "draft_board_core.js").read_text(encoding="utf-8")
    assert "p_hit_pct" not in pick
    assert "p_hit_pct" not in py
    assert "walkforward" not in pick
    assert "walkforward" not in core
