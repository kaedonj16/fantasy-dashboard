"""Phase 8 compact board payload and deep panel (slim CI)."""
from pathlib import Path
import json

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
    assert rec["prior_top12_count"] == 2
    assert "projected_ppg" not in rec
    assert "ppr_ppg" not in rec


def test_preseason_profile_counts_rookie_smash_after_a_down_year():
    """WR4 as a rookie, then WR42, is still previously top-12 (BTJ 2024/2025)."""
    rows = [
        _wh(
            sleeper_id="btj",
            season=2024,
            years_experience=0,
            draft_year=2024,
            age=21.8,
            draft_capital_bucket="round_1",
            ppr_positional_finish=4,
        ),
        _wh(
            sleeper_id="btj",
            season=2025,
            years_experience=1,
            draft_year=2024,
            age=22.8,
            draft_capital_bucket="round_1",
            ppr_positional_finish=42,
        ),
    ]
    packed = build_preseason_profiles(rows, upcoming_season=2026)
    rec = packed["by_player"]["btj"]
    assert rec["years_experience"] == 2
    assert rec["previous_season_finish"] == 42
    assert rec["previous_season_year"] == 2025
    assert rec["prior_top12_count"] == 1


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


def test_query_defaults_rookie_top12_count_without_claiming_veterans():
    rookie = query_for_board_player(
        {"id": "r", "position": "WR", "years_experience": 0},
        {},
    )
    assert rookie["prior_top12_count"] == 0
    veteran = query_for_board_player(
        {
            "id": "v",
            "position": "WR",
            "years_experience": 2,
            "previous_season_finish": 42,
        },
        {},
    )
    assert "prior_top12_count" not in veteran


def test_query_fills_live_draft_capital_for_unprofiled_rookies():
    q = query_for_board_player(
        {
            "id": "13287",
            "position": "RB",
            "years_exp": 0,
            "draft_round": 1,
            "draft_pick": 4,
            "age": 21.0,
        },
        {},
    )
    assert q["draft_capital_bucket"] == "round_1"
    assert q["years_experience"] == 0
    assert q["prior_top12_count"] == 0
    feats = extract_comp_query(q)
    assert feats["draft_capital"] == "round_1"
    assert feats["career_stage"] == "rookie"


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
    assert "this player's historical chance" in copy["cohort_note"].lower()
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
    assert "23-24" in values
    assert "20-25%" in values
    assert "80%+" in values
    assert copy["hit_rates"][1]["label"] == "Then finished top-12"
    assert copy["hit_rates"][1]["pct"] == 37
    assert "Among RBs" in copy["headline"]
    assert "this player's historical chance" in copy["cohort_note"].lower()
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
    for name in ("board.py", "aggregates_store.py", "career_path.py", "cohorts.py", "filters.py"):
        text = (hist / name).read_text(encoding="utf-8")
        assert "import pandas" not in text
        assert "import nfl_data_py" not in text
        assert "import flask" not in text.lower()
        assert "from utils.projection_resolver" not in text
        assert "build_player_history_features" not in text
        assert "static/pick_score" not in text
    bp = (ROOT / "routes" / "historical_api_bp.py").read_text(encoding="utf-8")
    assert "/api/historical-player/" in bp
    assert "/api/historical-trends" in bp
    assert "/api/historical-cohort" in bp
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
    if body["history"].get("closest_examples"):
        assert body["copy"].get("examples_heading") == "Closest historical examples"
    assert all("_" not in row["label"] for row in body["copy"]["profile"])
    assert isinstance(body["copy"]["trends"], list)
    with app.test_client() as client:
        resp2 = client.get(f"/api/historical-player/{pid}?adp=3&redraft_avg_pick=3&position=RB")
    body2 = resp2.get_json()
    kinds = {row["kind"] for row in body2["copy"]["trends"]}
    assert "adp" not in kinds
    assert "adp_positional" not in kinds
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
    assert "this player's historical chance" in copy["cohort_note"].lower()
    kinds = [row["kind"] for row in copy["trends"]]
    assert "adp" not in kinds
    assert "adp_positional" not in kinds
    assert "career_stage" in kinds
    assert "draft_capital" in kinds
    assert "age" in kinds
    sentences = [row["sentence"] for row in copy["trends"]]
    assert not any("taken in fantasy Round 1 finished top-12" in s for s in sentences)
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
            "prior_top12_count": 3,
            "previous_season_ngs_rush_yards_over_expected_per_att": 0.7,
        },
    )
    assert "projection_trends" not in with_proj["copy"]
    shown_copy = json.dumps(with_proj["copy"], ensure_ascii=False)
    assert "This board's projection" not in shown_copy
    assert "Sleeper projection for this season" not in shown_copy
    assert "both point at a top-12" not in shown_copy
    modal_kinds = [row["kind"] for row in with_proj["copy"]["trends"]]
    assert "adp" not in modal_kinds
    assert "adp_positional" not in modal_kinds
    assert "capital_miss" in modal_kinds
    assert "top12_as_rookie" in modal_kinds
    assert "top12_by_year_2" in modal_kinds
    assert "two_plus" in modal_kinds
    assert "ryoe" in modal_kinds
    assert "league_winner_smash" not in modal_kinds
    assert any(row.get("vs_label") for row in with_proj["copy"]["trends"])
    assert any(row.get("secondary") for row in with_proj["copy"]["trends"])
    shown = json.dumps(with_proj["copy"], ensure_ascii=False)
    assert "–" not in shown
    assert "—" not in shown
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
    hist_kinds = [row["kind"] for row in trends]
    assert "age" in hist_kinds
    assert "age_exact" not in hist_kinds
    assert "prime" not in hist_kinds
    titles = [row["title"] for row in trends]
    assert len(titles) == len(set(titles))
    assert titles.count("NFL Round 1") == 1
    assert "Miss rate" in titles
    assert "Hit top-12 as a rookie" in titles
    assert "Hit top-12 by year 2" in titles
    assert "Top-12 again" in titles
    assert "Then top-5" in titles
    assert "Age 24" not in titles
    assert any(str(t).startswith("Age ") and "-" in str(t) for t in titles)
    breakout_q = dict(query)
    breakout_q["previous_season_finish"] = 28
    breakout_q["years_experience"] = 4
    smash = build_hist_trends(breakout_q, aggs, panel["market"])
    smash_kinds = [row["kind"] for row in smash]
    assert "league_winner_smash" in smash_kinds
    assert "breakout" in smash_kinds
    assert "first_time_elite" not in smash_kinds
    assert "repeat" not in smash_kinds
    never_elite_q = dict(breakout_q)
    never_elite_q["prior_top12_count"] = 0
    never_kinds = [row["kind"] for row in build_hist_trends(never_elite_q, aggs, panel["market"])]
    assert "first_time_elite" in never_kinds
    prior_smash_q = dict(breakout_q)
    prior_smash_q["prior_top12_count"] = 1
    prior_kinds = [row["kind"] for row in build_hist_trends(prior_smash_q, aggs, panel["market"])]
    assert "breakout" in prior_kinds
    assert "league_winner_smash" in prior_kinds
    assert "first_time_elite" not in prior_kinds


def test_btj_hist_does_not_claim_never_previously_top12():
    from dashboard_services.historical.aggregates_store import load_profile_aggregates

    aggs = load_profile_aggregates()
    if not aggs:
        pytest.skip("profile JSON missing")
    profiles = ((aggs.get("preseason_profiles") or {}).get("by_player") or {})
    if "11631" not in profiles:
        pytest.skip("BTJ preseason profile missing")
    assert profiles["11631"].get("prior_top12_count") == 1
    panel = build_deep_panel("11631", aggs, extra={"position": "WR"})
    kinds = [row["kind"] for row in panel["copy"]["trends"]]
    assert "first_time_elite" not in kinds
    assert "breakout" in kinds
    assert "league_winner_smash" in kinds
    headline = str(panel["copy"].get("headline") or "").lower()
    assert "year 3" in headline
    assert "outside the top 36 last year" in headline
    assert "already been top-12" in headline
    assert "never" not in headline
    assert panel["history"].get("career_path") == "bounce_back"
    assert panel["history"].get("examples") == []
    assert panel["history"].get("closest_examples")
    top12 = next(
        row for row in panel["copy"]["hit_rates"] if row.get("tier") == "top_12"
    )
    assert top12.get("pct") is not None
    from dashboard_services.historical.board import query_for_board_player
    from dashboard_services.historical.comps import lookup_board_probabilities
    query = query_for_board_player({"id": "11631", "position": "WR"}, profiles)
    comps_pct = (
        (lookup_board_probabilities(query, aggs.get("comps") or aggs).get("rates") or {})
        .get("top_12") or {}
    ).get("display_pct")
    assert top12["pct"] > (comps_pct or 0)
    labels = [row["label"] for row in panel["copy"]["profile"]]
    assert "Career elite" in labels


def test_historical_trends_tab_is_position_wide_and_descriptive():
    from dashboard_services.historical.aggregates_store import load_profile_aggregates
    from dashboard_services.historical.board import board_contract, build_historical_trends

    assert board_contract()["trends_tab"] == "/api/historical-trends"
    assert board_contract()["cohort"] == "/api/historical-cohort"
    aggs = load_profile_aggregates()
    if not aggs:
        pytest.skip("profile JSON missing")
    payload = build_historical_trends(aggs)
    assert payload["available"] is True
    assert payload["descriptive_only"] is True
    assert payload["not_in_ranking"] is True
    assert payload["not_in_pick_score"] is True
    assert payload["positions"] == ["QB", "RB", "WR", "TE"]
    rb = payload["by_position"]["RB"]
    assert rb["baseline_pct"] is None or isinstance(rb["baseline_pct"], (int, float))
    ids = [sec["id"] for sec in rb["sections"]]
    assert "adp" not in ids
    assert "adp_positional" not in ids
    assert "repeat" in ids
    assert "league_winner" in ids
    assert "career_stage" in ids
    assert "draft_capital" in ids
    assert "top12_as_rookie" in ids
    assert "capital_miss" in ids
    assert "age" in ids
    assert "ryoe" in ids
    wr = payload["by_position"]["WR"]
    wr_ids = [sec["id"] for sec in wr["sections"]]
    qb_ids = [sec["id"] for sec in payload["by_position"]["QB"]["sections"]]
    te_ids = [sec["id"] for sec in payload["by_position"]["TE"]["sections"]]
    assert "touches" in ids
    assert "carries" in ids
    assert "games" in ids
    assert "receptions" in wr_ids
    assert "targets" in wr_ids
    assert "receptions" in te_ids
    assert "pass_attempts" in qb_ids
    assert "touches" not in wr_ids
    assert "receptions" not in ids
    assert "pass_attempts" not in ids
    assert "not fantasy ADP" in next(
        sec["note"] for sec in rb["sections"] if sec["id"] == "draft_capital"
    )
    winners = next(sec for sec in rb["sections"] if sec["id"] == "league_winner")
    assert any("top-5" in row["label"] for row in winners["rows"])
    repeat = next(sec for sec in rb["sections"] if sec["id"] == "repeat")
    never_row = next(row for row in repeat["rows"] if "Never-elite" in row["label"])
    assert never_row["match"] == {"group": "never_elite", "field": "prior_top12_count", "eq": 0}
    assert "null_as" not in never_row["match"]
    miss = next(sec for sec in rb["sections"] if sec["id"] == "capital_miss")
    assert miss["polarity"] == "miss"
    assert rb["highlights"]
    assert all("ranking_edge" in h for h in rb["highlights"])
    assert rb.get("red_flags") is not None
    assert rb.get("finish_tier_copy")
    assert "Top 24 is the flex line" not in payload["by_position"]["QB"]["finish_tier_copy"]
    assert "Top 24 is the flex line" not in payload["by_position"]["TE"]["finish_tier_copy"]
    assert "streaming" in payload["by_position"]["QB"]["finish_tier_copy"]
    dumped = json.dumps(payload)
    assert "observations" not in dumped
    assert "cohort_index" not in dumped
    assert rb["age_curve"]
    assert any(pt.get("age") for pt in rb["age_curve"])
    assert "top_5" in (rb.get("baselines") or {})
    assert "top_24" in (rb.get("baselines") or {})
    capital = next(sec for sec in rb["sections"] if sec["id"] == "draft_capital")
    assert capital.get("finish_tied") is True
    cap_row = capital["rows"][0]
    assert cap_row.get("match")
    assert cap_row["match"]["field"] == "draft_capital"
    assert cap_row.get("ranking_edge") is not None or cap_row.get("adjusted_edge") is not None
    assert cap_row.get("pcts", {}).get("top_12") is not None
    assert cap_row.get("pcts", {}).get("top_5") is not None
    assert cap_row.get("pcts", {}).get("top_24") is not None
    assert any(row.get("match") for row in next(sec for sec in rb["sections"] if sec["id"] == "repeat")["rows"])
    feats = payload.get("player_features") or {}
    assert feats
    sample_pid = next(iter(feats))
    assert feats[sample_pid].get("position") in ("QB", "RB", "WR", "TE")
    from dashboard_services.historical.board import matches_trend_filter
    assert any(f.get("ryoe") for f in feats.values())
    assert any(f.get("adot") for f in feats.values())
    assert any(f.get("position") == "RB" and f.get("touches") == "400+" for f in feats.values())
    assert any(f.get("position") == "WR" and f.get("receptions") for f in feats.values())
    touches_sec = next(sec for sec in rb["sections"] if sec["id"] == "touches")
    cliff = next(row for row in touches_sec["rows"] if (row.get("match") or {}).get("eq") == "400+")
    assert cliff.get("pct") is not None
    assert cliff.get("pcts", {}).get("top_5") is not None
    assert cliff.get("pcts", {}).get("top_24") is not None
    assert any(
        f.get("position") == "RB" and matches_trend_filter(f, cliff["match"])
        for f in feats.values()
    )
    rec_sec = next(sec for sec in payload["by_position"]["WR"]["sections"] if sec["id"] == "receptions")
    rec_row = rec_sec["rows"][0]
    assert any(
        f.get("position") == "WR" and matches_trend_filter(f, rec_row["match"])
        for f in feats.values()
    )
    ryoe_sec = next(sec for sec in rb["sections"] if sec["id"] == "ryoe")
    below = next(row for row in ryoe_sec["rows"] if (row.get("match") or {}).get("eq") == "below expected")
    above = next(row for row in ryoe_sec["rows"] if "above" in str((row.get("match") or {}).get("eq") or ""))
    assert any(
        f.get("position") == "RB" and matches_trend_filter(f, below["match"])
        for f in feats.values()
    )
    assert any(
        f.get("position") == "RB" and matches_trend_filter(f, above["match"])
        for f in feats.values()
    )
    adot_sec = next(sec for sec in payload["by_position"]["WR"]["sections"] if sec["id"] == "adot")
    adot_row = adot_sec["rows"][0]
    assert any(
        f.get("position") == "WR" and matches_trend_filter(f, adot_row["match"])
        for f in feats.values()
    )
    wr = payload["by_position"]["WR"]
    wr_ids = [sec["id"] for sec in wr["sections"]]
    assert "adot" in wr_ids
    for pos_page in payload["by_position"].values():
        for sec in pos_page["sections"]:
            assert "_" not in sec["heading"]
            for row in sec["rows"]:
                assert "_" not in row["label"]
                assert row.get("pct") is not None
    blob = " ".join(
        f"{sec['heading']} {sec['note']} " + " ".join(row["label"] for row in sec["rows"])
        for pos_page in payload["by_position"].values()
        for sec in pos_page["sections"]
    )
    assert "age_bucket" not in blob
    assert "snap_pct" not in blob
    assert "target_share" not in blob


def test_historical_trends_route_serves_json_leaves():
    pytest.importorskip("flask")
    from flask import Flask
    from dashboard_services.historical.aggregates_store import load_profile_aggregates
    from routes.historical_api_bp import historical_api_bp

    if not load_profile_aggregates():
        pytest.skip("profile JSON missing")
    app = Flask(__name__)
    app.register_blueprint(historical_api_bp)
    with app.test_client() as client:
        resp = client.get("/api/historical-trends")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["available"] is True
    assert body["descriptive_only"] is True
    assert "RB" in body["by_position"]
    assert body["by_position"]["RB"]["sections"]
    assert body.get("player_features")


def test_historical_cohort_route_counts_matched_rows():
    pytest.importorskip("flask")
    from flask import Flask
    from dashboard_services.historical.aggregates_store import load_profile_aggregates
    from routes.historical_api_bp import historical_api_bp

    aggs = load_profile_aggregates()
    if not aggs:
        pytest.skip("profile JSON missing")
    if not ((aggs.get("cohort_index") or {}).get("observations")):
        pytest.skip("cohort index missing")
    app = Flask(__name__)
    app.register_blueprint(historical_api_bp)
    with app.test_client() as client:
        resp = client.post(
            "/api/historical-cohort",
            json={
                "position": "WR",
                "filters": [
                    {"group": "age_bucket", "field": "age_bucket", "eq": "23-24"},
                    {"group": "draft_capital", "field": "draft_capital", "eq": "day_2"},
                ],
            },
        )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["descriptive_only"] is True
    assert body["not_in_ranking"] is True
    assert body["kind"] == "player_season"
    assert "observations" not in body
    assert body.get("available") is True
    assert body.get("sample_size", 0) > 0
    assert body["n_players"] <= body["sample_size"]
    assert body["raw_rate"] == body["rates"]["top_12"]["raw_rate"]
    assert body["ci_low"] is not None
    assert body["adjusted_edge"] is not None


def test_hist_trend_titles_keep_distinct_capital_and_age_rows():
    from dashboard_services.historical.board import format_hist_trend_title

    assert format_hist_trend_title(kind="draft_capital", label="Draft capital", bucket="Round 1") == "NFL Round 1"
    assert format_hist_trend_title(kind="capital_miss", label="Miss rate", bucket="Round 1") == "Miss rate"
    assert format_hist_trend_title(
        kind="top12_as_rookie", label="Hit top-12 as a rookie", bucket="Round 1"
    ) == "Hit top-12 as a rookie"
    assert format_hist_trend_title(kind="age", label="Age", bucket="23-24") == "Age 23-24"
    assert format_hist_trend_title(kind="age_exact", label="Age", bucket="24") == "Age 24"
    from dashboard_services.historical.board import matches_trend_filter

    feats = {
        "position": "RB",
        "draft_capital": "round_1",
        "career_stage": "year_2",
        "prior_finish": "top_24",
        "age_bucket": "23-24",
        "prior_top12_count": 0,
    }
    assert matches_trend_filter(feats, {"field": "draft_capital", "eq": "round_1"})
    assert not matches_trend_filter(feats, {"field": "draft_capital", "eq": "day_2"})
    assert matches_trend_filter(
        feats, {"field": "prior_finish", "in": ["none", "top_24", "top_36", "outside_36"]}
    )
    assert matches_trend_filter(
        {"prior_top12_count": None},
        {"field": "prior_top12_count", "eq": 0, "null_as": 0},
    )
    assert not matches_trend_filter(
        {"prior_top12_count": None},
        {"field": "prior_top12_count", "eq": 0},
    )
    assert not matches_trend_filter(
        {"prior_top12_count": 1},
        {"field": "prior_top12_count", "eq": 0},
    )
    assert matches_trend_filter({"age": 24}, {"field": "age", "between": [23, 27]})
    assert not matches_trend_filter({"age": 31}, {"field": "age", "between": [23, 27]})
    assert matches_trend_filter({"age": 22}, {"field": "age", "lte": 24})
    assert not matches_trend_filter({"age": 25}, {"field": "age", "lte": 24})
    assert not matches_trend_filter({"age": None}, {"field": "age", "lte": 24})
