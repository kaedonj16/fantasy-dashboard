"""History vs Projection vs Market comparison (slim CI)."""
from pathlib import Path

from dashboard_services.historical.career_profiles import assemble_profile_aggregates
from dashboard_services.historical.comps import extract_comp_query
from dashboard_services.historical.definitions import (
    SIGNAL_HISTORY_BULLISH_P,
    SIGNAL_HISTORY_SKEPTICAL_P,
    SIGNAL_PROB_ALIGN_DELTA,
    SIGNAL_RANK_ALIGN_SPOTS,
)
from dashboard_services.historical.signals import (
    compare_board_signals,
    compare_history_vs_market,
    compare_player_signals,
    compare_projection_vs_history,
    compare_projection_vs_market,
    implied_adp_positional_ranks,
    implied_projection_ranks,
    normalize_projected_ppg,
    projected_ppg_of,
    signal_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _wh(**kwargs):
    row = {
        "sleeper_id": kwargs.get("sleeper_id", "p"),
        "name": kwargs.get("name", "Player"),
        "season": kwargs.get("season", 2022),
        "position": kwargs.get("position", "WR"),
        "games": 16,
    }
    row.update(kwargs)
    return row


def test_normalize_projected_ppg_missing_is_none():
    assert normalize_projected_ppg(14.2) == 14.2
    assert normalize_projected_ppg(None) is None
    assert normalize_projected_ppg(0) is None
    assert normalize_projected_ppg(-1) is None
    assert normalize_projected_ppg("") is None


def test_projected_ppg_ignores_actuals_and_season_totals():
    row = {
        "sleeper_id": "x",
        "position": "WR",
        "ppg": 22.0,
        "ppr_ppg": 21.0,
        "projected_points": 280.0,
        "ppr_points": 300.0,
    }
    assert projected_ppg_of(row) is None
    row["projected_ppg"] = 16.4
    assert projected_ppg_of(row) == 16.4
    row["projected_ppg"] = 0
    row["proj_ppg"] = 11.1
    assert projected_ppg_of(row) == 11.1


def test_extract_comp_query_ignores_projections_and_adp():
    feats = extract_comp_query(_wh(
        years_experience=1,
        age=23.0,
        draft_capital_bucket="round_1",
        previous_season_finish=18,
        projected_ppg=19.0,
        adp_overall=4.0,
        ppr_ppg=20.0,
    ))
    assert feats.get("position") == "WR"
    assert "projected_ppg" not in feats
    assert "adp_overall" not in feats
    assert "ppr_ppg" not in feats


def test_implied_ranks_are_competition_rank_and_skip_missing():
    board = [
        {"sleeper_id": "a", "position": "WR", "projected_ppg": 18.0, "adp": 12.0},
        {"sleeper_id": "b", "position": "WR", "projected_ppg": 15.0, "adp": 8.0},
        {"sleeper_id": "c", "position": "WR", "projected_ppg": 15.0, "adp": 8.0},
        {"sleeper_id": "d", "position": "WR", "ppr_ppg": 30.0, "adp": 999},
        {"sleeper_id": "e", "position": "RB", "projected_ppg": 16.0, "adp": 3.0},
    ]
    proj = implied_projection_ranks(board)
    assert proj == [1, 2, 2, None, 1]
    adp = implied_adp_positional_ranks(board)
    assert adp == [3, 1, 1, None, 1]


def test_history_vs_market_labels_and_missing():
    aligned = compare_history_vs_market(0.40, 0.42)
    assert aligned["label"] == "aligned"
    assert aligned["unit"] == "probability"
    assert abs(aligned["delta"] - (-0.02)) < 1e-9
    hist = compare_history_vs_market(0.50, 0.30)
    assert hist["label"] == "history_higher"
    mkt = compare_history_vs_market(0.10, 0.35)
    assert mkt["label"] == "market_higher"
    missing = compare_history_vs_market(None, 0.40)
    assert missing["label"] == "unknown"
    assert missing["delta"] is None
    edge = compare_history_vs_market(0.40, 0.40 - SIGNAL_PROB_ALIGN_DELTA)
    assert edge["label"] == "history_higher"


def test_projection_vs_market_rank_space():
    aligned = compare_projection_vs_market(8, 10)
    assert aligned["label"] == "aligned"
    assert aligned["unit"] == "positional_rank"
    assert aligned["delta"] == 2
    proj = compare_projection_vs_market(4, 4 + SIGNAL_RANK_ALIGN_SPOTS + 1)
    assert proj["label"] == "projection_higher"
    mkt = compare_projection_vs_market(20, 5)
    assert mkt["label"] == "market_higher"
    missing = compare_projection_vs_market(None, 3)
    assert missing["label"] == "unknown"
    assert missing["delta"] is None


def test_projection_vs_history_is_qualitative_not_a_probability():
    skeptical = compare_projection_vs_history(True, SIGNAL_HISTORY_SKEPTICAL_P - 0.01)
    assert skeptical["label"] == "history_skeptical"
    assert "not converted" in skeptical["note"]
    bullish = compare_projection_vs_history(False, SIGNAL_HISTORY_BULLISH_P)
    assert bullish["label"] == "history_bullish"
    agree = compare_projection_vs_history(True, 0.40)
    assert agree["label"] == "agree_hit"
    miss = compare_projection_vs_history(False, 0.10)
    assert miss["label"] == "agree_miss"
    unknown = compare_projection_vs_history(True, None)
    assert unknown["label"] == "unknown"
    assert unknown["unit"] == "qualitative"


def _warehouse_for_signals():
    rows = []
    for i in range(12):
        rows.append(_wh(
            sleeper_id=f"r1{i}",
            season=2022,
            position="WR",
            adp_overall=float(i + 1),
            adp_source="sleeper",
            ppr_positional_finish=4 if i < 6 else 40,
            ppr_points=220 if i < 6 else 40,
            years_experience=1,
            age=23.0,
            draft_capital_bucket="round_1",
            previous_season_finish=18,
        ))
    for i in range(12):
        rows.append(_wh(
            sleeper_id=f"late{i}",
            season=2022,
            position="WR",
            adp_overall=130.0 + i,
            adp_source="mfl",
            ppr_positional_finish=40,
            ppr_points=30,
            years_experience=5,
            age=28.0,
            draft_capital_bucket="day_3",
            previous_season_finish=50,
        ))
    return rows


def test_board_signals_keep_native_units_and_missing_unknown():
    payload = assemble_profile_aggregates(_warehouse_for_signals())
    assert payload["phase"] == 8
    assert payload["definitions"]["no_projections"] is True
    assert payload["definitions"]["projections_in_comps"] is False
    assert payload["definitions"]["projections_in_ranking"] is False
    assert payload["signals"]["no_blended_score"] is True
    assert payload["signals"]["warehouse_has_projections"] is False
    assert payload["signals"]["native_units"]["projection_vs_history"] == "qualitative_only"

    board = [
        {
            "sleeper_id": "star",
            "position": "WR",
            "projected_ppg": 18.5,
            "adp": 3.0,
            "years_experience": 1,
            "age": 23.0,
            "draft_capital_bucket": "round_1",
            "previous_season_finish": 18,
        },
        *[
            {
                "sleeper_id": f"fill{i}",
                "position": "WR",
                "projected_ppg": 12.0 - (i * 0.2),
                "adp": 20.0 + i,
            }
            for i in range(12)
        ],
        {
            "sleeper_id": "late",
            "position": "WR",
            "projected_ppg": 8.0,
            "adp": 140.0,
            "years_experience": 5,
            "age": 28.0,
            "draft_capital_bucket": "day_3",
            "previous_season_finish": 50,
        },
        {"sleeper_id": "ghost", "position": "WR"},
        {
            "sleeper_id": "actuals",
            "position": "WR",
            "ppg": 22.0,
            "ppr_ppg": 22.0,
            "projected_points": 280.0,
            "adp": 999,
        },
    ]
    signals = compare_board_signals(board, payload)
    by_id = {row["player_id"]: row for row in signals}
    assert len(signals) == 16

    star = by_id["star"]
    assert star["projection"]["ppg"] == 18.5
    assert star["projection"]["implied_positional_rank"] == 1
    assert star["projection"]["implies_top_12"] is True
    assert "p_top_12" not in star["projection"]
    assert star["market"]["adp_bucket"] == "round_1"
    assert star["market"]["p_top_12"] is not None
    assert star["history"]["p_top_12"] is not None
    assert star["comparison"]["history_vs_market"]["unit"] == "probability"
    assert star["comparison"]["projection_vs_market"]["unit"] == "positional_rank"
    assert star["comparison"]["projection_vs_history"]["unit"] == "qualitative"
    assert star["comparison"]["blended_score"] is None
    assert star["comparison"]["no_blended_score"] is True
    assert star["comparison"]["projection_vs_market"]["label"] in {
        "aligned",
        "projection_higher",
        "market_higher",
    }

    late = by_id["late"]
    assert late["projection"]["implied_positional_rank"] == 14
    assert late["projection"]["implies_top_12"] is False
    assert late["market"]["adp_bucket"] == "rounds_11_plus"
    assert late["comparison"]["projection_vs_history"]["label"] in {
        "agree_miss",
        "history_bullish",
        "unknown",
    }

    ghost = by_id["ghost"]
    assert ghost["projection"]["ppg"] is None
    assert ghost["projection"]["implied_positional_rank"] is None
    assert ghost["projection"]["unknown_reason"] == "missing_ppg"
    assert ghost["market"]["unknown_reason"] == "missing_adp"
    assert ghost["market"]["p_top_12"] is None
    assert ghost["comparison"]["history_vs_market"]["label"] == "unknown"
    assert ghost["comparison"]["projection_vs_market"]["label"] == "unknown"
    assert ghost["comparison"]["blended_score"] is None

    actuals = by_id["actuals"]
    assert actuals["projection"]["ppg"] is None
    assert actuals["projection"]["implied_positional_rank"] is None
    assert actuals["market"]["overall_adp"] is None
    assert actuals["market"]["unknown_reason"] == "missing_adp"


def test_single_player_does_not_invent_a_blended_score():
    payload = assemble_profile_aggregates(_warehouse_for_signals())
    out = compare_player_signals(
        {"sleeper_id": "x", "position": "WR", "adp_overall": 2.0},
        payload,
    )
    assert out["comparison"]["blended_score"] is None
    assert out["projection"]["ppg"] is None
    assert out["comparison"]["projection_vs_market"]["label"] == "unknown"


def test_signal_contract_and_modules_stay_pure():
    contract = signal_contract()
    assert contract["no_blended_score"] is True
    assert contract["no_historical_projection_backfill"] is True
    assert "not a probability" in contract["native_units"]["projection"]
    hist = ROOT / "dashboard_services" / "historical"
    text = (hist / "signals.py").read_text(encoding="utf-8")
    assert "import pandas" not in text
    assert "import nfl_data_py" not in text
    assert "import flask" not in text.lower()
    assert "from utils.projection_resolver" not in text
    assert "fetch_projections" not in text
    assert "031_" not in text
    assert "static/pick_score" not in text
    assert "build_player_history_features" not in text
    for name in ("definitions.py", "comps.py", "adp.py", "career_profiles.py"):
        src = (hist / name).read_text(encoding="utf-8")
        assert "import pandas" not in src
        assert "from utils.projection_resolver" not in src
