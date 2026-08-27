"""Phase 8 compact board payload and deep panel (slim CI)."""
from pathlib import Path

import pytest

from dashboard_services.historical.board import (
    attach_historical_signals,
    build_deep_panel,
    build_hist_panel_copy,
    build_hist_trends,
    build_preseason_profiles,
    compact_signal,
    live_redraft_adp,
    query_for_board_player,
)
from dashboard_services.historical.career_profiles import assemble_profile_aggregates
from dashboard_services.historical.comps import extract_comp_query


ROOT = Path(__file__).resolve().parents[1]


def _wh(**kwargs):
    row = {
        "sleeper_id": kwargs.get("sleeper_id", "p"),
        "name": kwargs.get("name", "Player"),
        "season": kwargs.get("season", 2024),
        "position": kwargs.get("position", "WR"),
        "games": 16,
    }
    row.update(kwargs)
    return row


def test_preseason_profile_steps_forward_from_last_observed_season():
    rows = [
        _wh(
            sleeper_id="a",
            season=2023,
            years_experience=1,
            draft_year=2022,
            age=22.5,
            draft_capital_bucket="round_1",
            ppr_positional_finish=9,
            target_share=0.22,
            snap_pct=0.81,
        ),
        _wh(
            sleeper_id="a",
            season=2024,
            years_experience=2,
            draft_year=2022,
            age=23.5,
            draft_capital_bucket="round_1",
            ppr_positional_finish=5,
            target_share=0.28,
            snap_pct=0.90,
        ),
    ]
    packed = build_preseason_profiles(rows, upcoming_season=2026)
    assert packed["upcoming_season"] == 2026
    assert packed["prior_season_floor"] == 2024
    rec = packed["by_player"]["a"]
    assert rec["years_experience"] == 4  # 2026 - 2022
    assert rec["age"] == 25.5
    assert rec["previous_season_finish"] == 5
    assert rec["previous_season_year"] == 2024
    assert rec["draft_capital_bucket"] == "round_1"
    assert "projected_ppg" not in rec
    assert "ppr_ppg" not in rec


def test_live_redraft_adp_ignores_dynasty_and_sleeper_999():
    assert live_redraft_adp({"avg_pick": 4.0, "redraft_avg_pick": 12.2}) == 12.2
    assert live_redraft_adp({"avg_pick": 4.0}) is None
    assert live_redraft_adp({"redraft_avg_pick": 999}) is None
    assert live_redraft_adp({
        "adp_by_source": {"consensus": {"redraft_avg_pick": 8.4, "avg_pick": 1.2}},
    }) == 8.4


def test_query_uses_json_priors_and_live_proj_not_actuals():
    profiles = {"x": {
        "position": "WR",
        "years_experience": 2,
        "draft_capital_bucket": "round_1",
        "previous_season_finish": 11,
        "age": 24.0,
    }}
    q = query_for_board_player(
        {
            "id": "x",
            "position": "WR",
            "ppg": 22.0,
            "ppr_ppg": 21.0,
            "proj_ppg": 16.4,
            "redraft_avg_pick": 7.0,
            "age": 40.0,
        },
        profiles,
    )
    assert q["projected_ppg"] == 16.4
    assert q["adp_overall"] == 7.0
    assert q["age"] == 24.0  # warehouse Sept-1, not live age-today
    assert q["previous_season_finish"] == 11
    feats = extract_comp_query(q)
    assert "projected_ppg" not in feats
    assert "adp_overall" not in feats


def test_attach_compact_payload_and_deep_panel_are_descriptive():
    rows = []
    for i in range(12):
        rows.append(_wh(
            sleeper_id=f"r{i}",
            season=2024,
            position="WR",
            adp_overall=float(i + 1),
            ppr_positional_finish=4 if i < 6 else 40,
            ppr_points=200 if i < 6 else 40,
            years_experience=1,
            draft_year=2023,
            age=23.0,
            draft_capital_bucket="round_1",
            previous_season_finish=18,
        ))
    payload = assemble_profile_aggregates(rows)
    assert payload["phase"] == 9
    assert payload["board"]["not_in_ranking"] is True
    assert payload["board"]["not_in_pick_score"] is True
    assert payload["preseason_profiles"]["n_players"] == 12

    board = [
        {
            "id": "r0",
            "position": "WR",
            "proj_ppg": 18.0,
            "redraft_avg_pick": 3.0,
        },
        {
            "id": "r11",
            "position": "WR",
            "proj_ppg": 8.0,
            "redraft_avg_pick": 140.0,
        },
        {"id": "k", "position": "K", "proj_ppg": 9.0, "redraft_avg_pick": 12.0},
    ]
    compact = attach_historical_signals(board, payload)
    assert "historical" in board[0]
    assert board[0]["historical"]["p_hit_pct"] is not None
    assert board[0]["historical"]["h_vs_m"] in {
        "aligned", "history_higher", "market_higher", "unknown",
    }
    assert "examples" not in board[0]["historical"]
    assert compact[2] == {}
    assert "historical" not in board[2]

    panel = build_deep_panel("r0", payload)
    assert panel["available"] is True
    assert panel["no_blended_score"] is True
    assert panel["not_in_ranking"] is True
    assert panel["history"]["kind"] == "conditional"
    assert isinstance(panel["history"]["examples"], list)
    copy = panel["copy"]
    assert copy["hit_rates"]
    assert copy["hit_rates"][0]["label"].startswith("Then finished ")
    assert copy["headline"].startswith("Among ")
    assert "not this player's odds" in copy["cohort_note"]
    assert isinstance(copy["trends"], list)
    assert all("_" not in row["label"] for row in copy["profile"])
    assert all("_" not in row["label"] for row in copy["relaxed"])
    # Warehouse profiles have no current ADP, so Market is unknown until the
    # live board pick is passed in. That is the modal dash, not a missing rate.
    assert panel["market"]["unknown_reason"] == "missing_adp"
    assert panel["market"]["p_top_12"] is None
    with_adp = build_deep_panel("r0", payload, extra={"redraft_avg_pick": 3.0})
    assert with_adp["market"]["unknown_reason"] is None
    assert with_adp["market"]["p_top_12"] is not None
    assert "Players drafted in Round 1" in with_adp["copy"]["market_sentence"]


def test_hist_panel_copy_uses_bucket_hit_rates_not_snake_case():
    history = {
        "n": 35,
        "key_used": {
            "position": "RB",
            "career_stage": "year_4",
            "draft_capital": "round_1",
            "prior_finish": "top_12",
            "age_bucket": "23-24",
            "target_share": "20-25%",
            "snap_pct": "80%+",
        },
        "dropped": ["target_share", "snap_pct"],
        "fallback": True,
        "rates": {
            "top_5": {"display_pct": 18, "sample_size": 35, "confidence": "moderate"},
            "top_12": {"display_pct": 37, "sample_size": 35, "confidence": "moderate"},
            "top_24": {"display_pct": 62, "sample_size": 35, "confidence": "moderate"},
        },
    }
    market = {
        "p_top_12": 0.82,
        "adp_bucket": "round_1",
        "sample_size": 140,
        "confidence": "strong",
        "overall_adp": 3.2,
    }
    copy = build_hist_panel_copy(history, market)
    labels = [row["label"] for row in copy["profile"]]
    values = [row["value"] for row in copy["profile"]]
    assert labels == [
        "Position",
        "Career stage",
        "Draft capital",
        "Last year finish",
        "Age",
        "Last year target share",
        "Last year snaps",
    ]
    assert "Year 4" in values
    assert "Round 1" in values
    assert "23–24" in values
    assert "20-25%" in values
    assert "80%+" in values
    assert copy["hit_rates"][1]["label"] == "Then finished top-12"
    assert copy["hit_rates"][1]["pct"] == 37
    assert "Among RBs" in copy["headline"]
    assert "not this player's odds" in copy["cohort_note"]
    assert copy["relaxed"][0]["label"] == "Last year target share"
    shown = " ".join(
        f"{row['label']} {row['value']}" for row in copy["profile"]
    ) + " " + " ".join(row["label"] for row in copy["relaxed"])
    assert "age_bucket" not in shown
    assert "career_stage" not in shown
    assert "draft_capital" not in shown
    assert "Players drafted in Round 1 historically finished top-12 82%" in copy["market_sentence"]
    missing = build_hist_panel_copy(history, {})
    assert "no live ADP" in missing["market_sentence"]


def test_compact_signal_never_blends():
    full = {
        "history": {"p_top_12": 0.14, "confidence": "moderate", "sample_size": 30},
        "market": {"p_top_12": 0.58, "adp_bucket": "round_1"},
        "projection": {"implied_positional_rank": 4, "implies_top_12": True},
        "comparison": {
            "history_vs_market": {"label": "market_higher"},
            "projection_vs_market": {"label": "aligned", "adp_positional_rank": 3},
            "projection_vs_history": {"label": "history_skeptical"},
            "blended_score": 0.99,
        },
    }
    out = compact_signal(full)
    assert out["p_hit_pct"] == 14
    assert out["mkt_pct"] == 58
    assert "Players drafted in Round 1 historically finished top-12 58%" in out["mkt_sentence"]
    assert "blended_score" not in out
    assert out["h_vs_m"] == "market_higher"


def test_board_modules_stay_pure():
    hist = ROOT / "dashboard_services" / "historical"
    for name in ("board.py", "aggregates_store.py"):
        text = (hist / name).read_text(encoding="utf-8")
        assert "import pandas" not in text
        assert "import nfl_data_py" not in text
        assert "import flask" not in text.lower()
        assert "from utils.projection_resolver" not in text
        assert "build_player_history_features" not in text
        assert "static/pick_score" not in text
    bp = (ROOT / "routes" / "historical_api_bp.py").read_text(encoding="utf-8")
    assert "/api/historical-player/" in bp
    assert "read_parquet" not in bp
    assert "load_player_history_df" not in bp
    pick = (ROOT / "static" / "pick_score.js").read_text(encoding="utf-8")
    core = (ROOT / "static" / "draft_board_core.js").read_text(encoding="utf-8")
    assert "p_hit_pct" not in pick
    assert "historical" not in pick
    assert "p_hit_pct" not in core
    assert "compare_board_signals" not in core
    app_py = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _league_players_response" in app_py
    assert "stamp_historical_on_payload" in app_py
    assert "historical_api_bp" in app_py
    assert "return _league_players_response(payload)" in app_py


def test_deep_panel_route_serves_json_leaves():
    pytest.importorskip("flask")
    from flask import Flask
    from routes.historical_api_bp import historical_api_bp

    app = Flask(__name__)
    app.register_blueprint(historical_api_bp)
    aggs = None
    from dashboard_services.historical.aggregates_store import load_profile_aggregates
    aggs = load_profile_aggregates()
    if not aggs:
        pytest.skip("profile JSON missing")
    pid = next(iter((aggs.get("preseason_profiles") or {}).get("by_player") or {}), None)
    if not pid:
        pytest.skip("no preseason profiles")
    with app.test_client() as client:
        resp = client.get(f"/api/historical-player/{pid}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["available"] is True
    assert body["no_blended_score"] is True
    assert body["history"]["kind"] == "conditional"
    assert "examples" in body["history"]
    assert body["copy"]["hit_rates"]
    assert all("_" not in row["label"] for row in body["copy"]["profile"])
    assert isinstance(body["copy"]["trends"], list)
    with app.test_client() as client:
        resp2 = client.get(f"/api/historical-player/{pid}?adp=3&redraft_avg_pick=3&position=RB")
    body2 = resp2.get_json()
    kinds = {row["kind"] for row in body2["copy"]["trends"]}
    assert "adp" in kinds or body2["market"]["p_top_12"] is None
    shown = " ".join(row["sentence"] for row in body2["copy"]["trends"])
    assert "age_bucket" not in shown
    assert "snap_pct" not in shown


def test_hist_trends_are_descriptive_bucket_slices():
    from dashboard_services.historical.aggregates_store import load_profile_aggregates

    aggs = load_profile_aggregates()
    if not aggs:
        pytest.skip("profile JSON missing")
    panel = build_deep_panel("9221", aggs, extra={"redraft_avg_pick": 2.0, "position": "RB"})
    copy = panel["copy"]
    assert copy["headline"].startswith("Among RBs")
    assert "not this player's odds" in copy["cohort_note"]
    kinds = [row["kind"] for row in copy["trends"]]
    assert "adp" in kinds
    assert "career_stage" in kinds
    assert "draft_capital" in kinds
    assert "age" in kinds
    sentences = [row["sentence"] for row in copy["trends"]]
    assert any("taken in fantasy Round 1 finished top-12" in s for s in sentences)
    assert any("target share last year" in s for s in sentences)
    assert not any("targets last year" in s for s in sentences)
    assert all("_" not in row["label"] for row in copy["trends"])
    with_proj = build_deep_panel(
        "9221",
        aggs,
        extra={
            "redraft_avg_pick": 2.0,
            "position": "RB",
            "proj_ppg": 18.4,
            "projected_positional_rank": 2,
            "adp_positional_rank": 1,
        },
    )
    proj = with_proj["copy"]["projection_trends"]
    kinds = [row["kind"] for row in proj]
    assert "projection_ppg" in kinds
    assert "projection_rank" in kinds
    assert any("18.4 PPG" == row.get("display") for row in proj)
    assert any(row.get("display") == "#2" for row in proj)
    assert all("p_top_12" not in row for row in proj)
    query = {
        "position": "RB",
        "years_experience": 3,
        "age": 24.4,
        "draft_capital_bucket": "round_1",
        "previous_season_finish": 3,
        "previous_season_target_share": 0.17,
        "previous_season_snap_pct": 0.61,
        "previous_season_year": 2025,
        "adp_overall": 2.0,
    }
    trends = build_hist_trends(query, aggs, panel["market"])
    assert trends
    assert all(row.get("pct") is not None for row in trends)
