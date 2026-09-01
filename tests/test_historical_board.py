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
    prefer_selective_hist_tiles,
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
    assert q["nfl_draft_pick"] == 4
    assert q["years_experience"] == 0
    assert q["prior_top12_count"] == 0
    feats = extract_comp_query(q)
    assert feats["draft_capital"] == "round_1"
    assert feats["career_stage"] == "rookie"
    from dashboard_services.historical.filters import extract_trend_features

    trend = extract_trend_features(q)
    assert trend["nfl_draft_pick"] == 4


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
    assert "trend_feats" in board[0]["historical"]
    assert board[0]["historical"]["trend_feats"].get("position") == "WR"
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
    assert "Sample: 140" in copy["market_sentence"]
    assert "n=140" not in copy["market_sentence"]
    assert copy["market_compare_heading"] == "Two groups, not one chance"
    assert copy["history_group_label"] == "Players like this"
    assert copy["history_group_hint"] == "this career and situation"
    assert copy["market_group_label"] == "Round 1"
    assert copy["market_group_hint"] == "anyone taken in that fantasy round"
    assert copy["history_pct"] == 37
    assert copy["market_pct"] == 82
    assert copy["history_vs_market_pts"] == -45
    assert "Round 1 hits 45 percent more often" in copy["gap_note"]
    assert "Early ADP is a high bar" in copy["gap_note"]
    assert "not a combined chance" in copy["gap_note"].lower()
    assert "—" not in copy["gap_note"]
    assert "–" not in copy["gap_note"]
    assert "Grouped by Career, Capital, Roster, Offense, and Usage" in copy["trends_note"]
    assert "Each tile is a bucket this player matches this year" in copy["trends_note"]
    assert "Analog rows" not in copy["trends_note"]
    assert "—" not in copy["trends_note"]
    assert "–" not in copy["trends_note"]
    missing = build_hist_panel_copy(history, {})
    assert "no live ADP" in missing["market_sentence"]
    assert missing["gap_note"] == "Need live ADP to show the other group."
    assert missing["market_group_label"] == "That ADP round"
    aligned = build_hist_panel_copy(
        history,
        {"p_top_12": 0.40, "adp_bucket": "round_3"},
    )
    assert aligned["gap_note"].startswith("Round 3 and players like this are in line")
    ahead = build_hist_panel_copy(
        history,
        {"p_top_12": 0.20, "adp_bucket": "rounds_8_10"},
    )
    assert "Players like this hit 17 percent more often than Rounds 8-10" in ahead["gap_note"]
    assert "Early ADP is a high bar" not in ahead["gap_note"]


def test_compact_signal_never_blends():
    full = {
        "history": {"p_top_12": 0.14, "confidence": "moderate", "sample_size": 30},
        "market": {"p_top_12": 0.58, "adp_bucket": "round_1"},
        "projection": {"implied_positional_rank": 4, "implies_top_12": True},
        "comparison": {
            "history_vs_market": {"label": "market_higher", "delta": -0.44},
            "projection_vs_market": {"label": "aligned", "adp_positional_rank": 3},
            "projection_vs_history": {"label": "history_skeptical"},
            "blended_score": 0.99,
        },
    }
    out = compact_signal(full)
    assert out["p_hit_pct"] == 14
    assert out["mkt_pct"] == 58
    assert out["h_vs_m_pts"] == -44
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
    assert "return _league_players_response" in app_py


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
    groups = copy.get("trend_groups") or []
    group_ids = [sec["id"] for sec in groups]
    assert group_ids[0] == "career"
    assert "roster" in group_ids or "capital" in group_ids
    grouped_n = sum(len(sec["rows"]) for sec in groups)
    assert grouped_n == len(copy["trends"])
    assert "this player's historical chance" in copy["cohort_note"].lower()
    kinds = [row["kind"] for row in copy["trends"]]
    assert "adp" not in kinds
    assert "adp_positional" not in kinds
    assert "career_stage" in kinds
    assert "draft_capital" in kinds or any(str(k).startswith("capital_roster") for k in kinds)
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
    assert "capital_miss" not in modal_kinds
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
    assert titles.count("Drafted NFL Round 1, any season") == 1
    assert "Drafted NFL Round 1, miss (any season)" not in titles
    assert "Drafted NFL Round 1, year 1" not in titles
    assert "Drafted NFL Round 1, year 2" not in titles
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


def test_gibbs_hist_does_not_shrink_tiny_cell_toward_all_rbs():
    from dashboard_services.historical.aggregates_store import load_profile_aggregates
    from dashboard_services.historical.signals import lookup_history_probability

    aggs = load_profile_aggregates()
    if not aggs:
        pytest.skip("profile JSON missing")
    extra = {"redraft_avg_pick": 2.0, "position": "RB"}
    panel = build_deep_panel("9221", aggs, extra=extra)
    top12 = next(row for row in panel["copy"]["hit_rates"] if row["tier"] == "top_12")
    assert top12["n"] == 2
    assert abs((panel["history"]["rates"]["top_12"]["raw_rate"] or 0) - 0.5) < 1e-9
    assert panel["history"].get("prior_source") == "parent_cell"
    prior_key = panel["history"].get("prior_key") or {}
    assert prior_key.get("age_bucket") == "23-24"
    assert prior_key.get("prior_finish") == "top_5"
    assert (panel["history"].get("prior_n") or 0) >= 8
    assert top12["pct"] >= 50
    assert panel["copy"]["history_pct"] >= 50
    assert panel["copy"]["history_pct"] != 15
    note = str(panel["copy"].get("sample_prior_note") or "")
    assert "Only 2 exact matches" in note
    assert "not every RB" in note
    typical = str(panel["copy"].get("typical_note") or "")
    assert "high historical hit rate" in typical
    assert "typical RB" in typical
    assert "—" not in note + typical
    assert "–" not in note + typical
    hist = lookup_history_probability(
        {
            "position": "RB",
            "years_experience": 3,
            "age": 24.4,
            "draft_capital_bucket": "round_1",
            "previous_season_finish": 3,
            "previous_season_target_share": 0.17,
            "previous_season_snap_pct": 0.61,
            "previous_season_year": 2025,
        },
        aggs,
    )
    assert hist["sample_size"] == 2
    assert hist["p_top_12"] is not None and hist["p_top_12"] >= 0.50


def test_kelce_hist_does_not_headline_self_repeat_as_77():
    from dashboard_services.historical.aggregates_store import load_profile_aggregates
    from dashboard_services.historical.signals import lookup_history_probability

    aggs = load_profile_aggregates()
    if not aggs:
        pytest.skip("profile JSON missing")
    extra = {"redraft_avg_pick": 100.0, "position": "TE"}
    panel = build_deep_panel("1466", aggs, extra=extra)
    top12 = next(row for row in panel["copy"]["hit_rates"] if row["tier"] == "top_12")
    assert panel["history"].get("prior_source") == "parent_displayed"
    assert (panel["history"].get("exact_n") or 0) == 2
    assert (panel["history"].get("n") or 0) >= 15
    assert top12["n"] >= 15
    key_used = panel["history"].get("key_used") or {}
    profile_key = panel["history"].get("profile_key") or {}
    assert key_used.get("prior_finish") == "top_5"
    assert key_used.get("career_stage") == "year_6_plus"
    assert "age_bucket" not in key_used
    assert profile_key.get("age_bucket") == "32+"
    assert panel["copy"]["history_pct"] != 77
    assert panel["copy"]["history_pct"] < 70
    note = str(panel["copy"].get("sample_prior_note") or "")
    assert "Only 2 exact matches" in note
    assert "veteran repeating" in note
    assert "not declining vets mixed in" not in note
    assert "own seasons" in note
    assert "—" not in note
    assert "–" not in note
    hist = lookup_history_probability(
        {
            "position": "TE",
            "years_experience": 13,
            "age": 36.9,
            "draft_capital_bucket": "day_2",
            "previous_season_finish": 3,
            "previous_season_snap_pct": 0.70,
            "previous_season_year": 2025,
            "sleeper_id": "1466",
        },
        aggs,
    )
    assert hist["sample_size"] >= 15
    assert hist["p_top_12"] is not None and hist["p_top_12"] < 0.70
    assert hist["p_top_12"] != pytest.approx(0.77, abs=0.02)


def test_hist_panel_keeps_draft_capital_when_the_cell_has_seasons():
    from dashboard_services.historical.aggregates_store import load_profile_aggregates

    aggs = load_profile_aggregates()
    if not aggs:
        pytest.skip("profile JSON missing")
    panel = build_deep_panel(
        "preview-love",
        aggs,
        extra={
            "position": "RB",
            "years_experience": 0,
            "nfl_draft_pick": 8,
            "draft_capital_bucket": "round_1",
            "age": 21,
            "adp": 18,
            "redraft_avg_pick": 18,
            "team": "ARI",
            "roster_spot": 1,
            "proj_ppg": 14.0,
        },
    )
    dropped = list(panel["history"].get("dropped") or [])
    assert "draft_capital" not in dropped
    assert "round 1" in panel["copy"]["headline"].lower()
    assert panel["copy"]["history_pct"] != 4
    assert panel["copy"]["history_pct"] >= 35
    titles = [row["title"] for row in panel["copy"]["trends"]]
    assert "Drafted NFL Top 10, year 1" not in titles
    assert "Drafted NFL Top 10, any season" not in titles
    assert "Drafted NFL Top 10, year 2" not in titles
    assert "Drafted NFL Round 1, miss (any season)" not in titles
    assert "Drafted NFL Top 10, RB1, year 1" in titles
    year1 = next(row for row in panel["copy"]["trends"] if row["title"] == "Drafted NFL Top 10, RB1, year 1")
    assert year1.get("n")
    assert year1.get("pct") is not None
    offense_titles = [row["title"] for row in panel["copy"]["trends"] if "offense" in row["title"].lower()]
    assert "Top-10 projected offense" not in offense_titles
    assert "Top-10 projected offense, year 1" not in offense_titles
    assert "Top-10 projected offense, RB1" not in offense_titles
    assert "Top-10 projected offense, RB2" not in offense_titles
    assert "Top-10 projected offense, RB3+" not in offense_titles
    assert "Top-10 projected offense, RB1, year 1" not in offense_titles
    assert "21-32 projected offense, year 1" not in offense_titles
    assert "21-32 projected offense, RB1, year 1" in offense_titles
    assert not any("offense last year" in title for title in offense_titles)
    assert "Drafted NFL Round 1, RB1, year 1" not in titles
    assert "Drafted NFL Round 1, RB3+" not in titles
    assert "Drafted NFL Round 1, RB2" not in titles
    assert "Drafted NFL Top 10, RB1, year 1" in titles
    assert "21-32 projected offense, Round 1" not in titles
    assert "21-32 projected offense, NFL Top 10" in titles
    assert "Top-10 projected offense, Round 1" not in titles
    assert "Top-10 projected offense, NFL Top 10" not in titles
    assert not any(
        "projected offense" in title and "Round 1" in title and "year 1" in title
        for title in titles
    )
    assert all(row.get("role") == "this" for row in panel["copy"]["trends"])
    assert not any(row.get("polarity") == "miss" for row in panel["copy"]["trends"])
    assert not any("RB3+" in title for title in titles)
    by_title = {row["title"]: row for row in panel["copy"]["trends"]}
    assert by_title["Drafted NFL Top 10, RB1, year 1"]["role"] == "this"
    assert by_title["21-32 projected offense, NFL Top 10"]["role"] == "this"
    named = by_title["Drafted NFL Top 10, RB1, year 1"].get("examples") or []
    assert 1 <= len(named) <= 3
    assert named[0].get("name")
    group_ids = [sec["id"] for sec in (panel["copy"].get("trend_groups") or [])]
    assert group_ids[0] == "career"
    assert "roster" in group_ids
    assert "offense" in group_ids
    note = str(panel["copy"].get("examples_vs_cohort_note") or "")
    if note:
        assert "Sample:" in note
        assert "n=" not in note
    board = [{
        "id": "preview-love",
        "position": "RB",
        "years_experience": 0,
        "nfl_draft_pick": 8,
        "draft_capital_bucket": "round_1",
        "age": 21,
        "adp": 18,
        "redraft_avg_pick": 18,
        "team": "ARI",
        "roster_spot": 1,
        "proj_ppg": 14.0,
    }]
    attach_historical_signals(board, aggs)
    assert board[0]["historical"]["p_hit_pct"] == panel["copy"]["history_pct"]


def test_prefer_selective_hist_tiles_drops_same_band_without_roster():
    kept = [row["title"] for row in prefer_selective_hist_tiles([
        {"kind": "offense_year_1", "bucket": "21-32", "title": "21-32 projected offense, year 1"},
        {"kind": "offense_roster_1", "bucket": "21-32, RB1", "title": "21-32 projected offense, RB1, year 1"},
        {"kind": "offense_capital", "bucket": "21-32, Top 10", "title": "21-32 projected offense, NFL Top 10"},
        {"kind": "top12_as_rookie", "bucket": "Top 10", "title": "Drafted NFL Top 10, year 1"},
        {"kind": "capital_roster_1", "bucket": "Top 10, RB1", "title": "Drafted NFL Top 10, RB1, year 1"},
        {"kind": "draft_capital", "bucket": "Day 2", "title": "Drafted NFL Day 2, any season"},
    ])]
    assert kept == [
        "21-32 projected offense, RB1, year 1",
        "21-32 projected offense, NFL Top 10",
        "Drafted NFL Top 10, RB1, year 1",
        "Drafted NFL Day 2, any season",
    ]


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
    titles = [row["title"] for row in panel["copy"]["trends"]]
    assert "11-20 projected offense, WR1" in titles
    assert "11-20 projected offense" not in titles
    assert "Drafted NFL Picks 11-25, any season" not in titles
    assert "Drafted NFL Picks 11-25, WR1" in titles
    assert not any("Top-10 projected offense" in title for title in titles)
    assert "Top-10 projected offense, WR2" not in titles
    assert "Top-10 projected offense, WR3+" not in titles
    assert "Drafted NFL Round 1, WR1" not in titles
    assert "Drafted NFL Round 1, WR2" not in titles
    assert "Drafted NFL Round 1, WR3+" not in titles
    assert "Drafted NFL Picks 11-25, WR1" in titles
    assert "Outside top 36 last year, WR1" in titles
    assert "Outside top 36 last year, WR3+" not in titles
    assert all(row.get("role") == "this" for row in panel["copy"]["trends"])
    assert not any(row.get("polarity") == "miss" for row in panel["copy"]["trends"])
    assert not any("WR3+" in title for title in titles)
    by_title = {row["title"]: row for row in panel["copy"]["trends"]}
    assert by_title["Outside top 36 last year, WR1"]["role"] == "this"
    assert by_title["Drafted NFL Picks 11-25, WR1"]["role"] == "this"
    bounce_names = by_title["Outside top 36 last year, WR1"].get("examples") or []
    assert 1 <= len(bounce_names) <= 3
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
    assert "offense" in ids
    assert "offense_roster" in ids
    assert "capital_roster" in ids
    assert "offense_capital" in ids
    assert "bounce_roster" in ids
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
    cap_labels = [row["label"] for row in capital["rows"]]
    assert "Top 10, any season" in cap_labels
    assert "Picks 11-25, any season" in cap_labels
    assert "Rest of Round 1, any season" in cap_labels
    assert "Round 1" not in cap_labels
    assert "Day 2 (rounds 2-3), any season" in cap_labels
    cap_row = capital["rows"][0]
    assert cap_row.get("match")
    assert cap_row["match"]["group"] == "draft_capital"
    assert cap_row["match"]["field"] == "nfl_draft_pick"
    year1 = next(sec for sec in rb["sections"] if sec["id"] == "top12_as_rookie")
    assert year1["heading"] == "Drafted, year 1"
    assert any(row["label"] == "Top 10, year 1" for row in year1["rows"])
    offense = next(sec for sec in rb["sections"] if sec["id"] == "offense")
    assert offense["heading"] == "Projected offense"
    off_labels = [row["label"] for row in offense["rows"]]
    assert "Top 10 projected" in off_labels
    assert "Top 10 projected, year 1" in off_labels
    assert "11-20 projected" in off_labels
    assert "season-long" in (offense.get("note") or "")
    assert "regular-season" in (offense.get("note") or "")
    assert "implied" in (offense.get("note") or "")
    roster = next(sec for sec in rb["sections"] if sec["id"] == "offense_roster")
    assert roster["heading"] == "Projected offense by roster spot"
    roster_labels = [row["label"] for row in roster["rows"]]
    assert "Top 10 projected, RB1" in roster_labels
    assert "Top 10 projected, RB2" in roster_labels
    assert "Top 10 projected, RB3+" in roster_labels
    assert "21-32 projected, RB3+" in roster_labels
    assert all("_" not in label for label in roster_labels)
    assert "preseason ADP" in (roster.get("note") or "")
    assert "depth chart" in (roster.get("note") or "")
    rb1 = next(row for row in roster["rows"] if row["label"] == "Top 10 projected, RB1")
    rb3 = next(row for row in roster["rows"] if row["label"] == "Top 10 projected, RB3+")
    assert rb1.get("n") and rb3.get("n")
    assert rb1["match"]["group"] == "offense_roster"
    assert rb1["match"]["all"]
    for pos, starter, depth in (
        ("WR", "WR1", "WR3+"),
        ("TE", "TE1", "TE2"),
        ("QB", "QB1", "QB2"),
    ):
        pos_roster = next(
            sec for sec in payload["by_position"][pos]["sections"] if sec["id"] == "offense_roster"
        )
        pos_labels = [row["label"] for row in pos_roster["rows"]]
        assert f"Top 10 projected, {starter}" in pos_labels
        assert f"Top 10 projected, {depth}" in pos_labels
        assert all("_" not in lab for lab in pos_labels)
        assert f"{pos}1 is the lowest ADP" in (pos_roster.get("note") or "")
        starter_row = next(row for row in pos_roster["rows"] if row["label"] == f"Top 10 projected, {starter}")
        assert starter_row.get("n") and starter_row.get("pct") is not None
    wr_cap = next(sec for sec in wr["sections"] if sec["id"] == "capital_roster")
    wr_cap_labels = [row["label"] for row in wr_cap["rows"]]
    assert "Round 1, WR1" in wr_cap_labels
    assert "Round 1, WR3+" in wr_cap_labels
    assert "Top 10, WR1" in wr_cap_labels
    assert "Day 2, WR1" in wr_cap_labels
    assert "Day 3, WR3+" in wr_cap_labels
    assert all("_" not in lab for lab in wr_cap_labels)
    wr1_r1 = next(row for row in wr_cap["rows"] if row["label"] == "Round 1, WR1")
    wr3_r1 = next(row for row in wr_cap["rows"] if row["label"] == "Round 1, WR3+")
    assert wr1_r1.get("n") and wr3_r1.get("n")
    assert wr1_r1["match"]["group"] == "capital_roster"
    assert wr1_r1["match"]["all"]
    qb_cap_ids = [sec["id"] for sec in payload["by_position"]["QB"]["sections"]]
    assert "capital_roster" in qb_cap_ids
    assert "capital_roster" not in te_ids
    rb_off_cap = next(sec for sec in rb["sections"] if sec["id"] == "offense_capital")
    off_cap_labels = [row["label"] for row in rb_off_cap["rows"]]
    assert "Top 10 projected, Round 1" in off_cap_labels
    assert "Top 10 projected, NFL Top 10" in off_cap_labels
    assert "21-32 projected, Round 1" in off_cap_labels
    assert "21-32 projected, NFL Top 10" in off_cap_labels
    assert all("_" not in lab for lab in off_cap_labels)
    assert "offense_capital" not in wr_ids
    assert "offense_capital" not in te_ids
    wr_bounce = next(sec for sec in wr["sections"] if sec["id"] == "bounce_roster")
    bounce_labels = [row["label"] for row in wr_bounce["rows"]]
    assert "Outside top 36, WR1" in bounce_labels
    assert "Outside top 36, WR3+" in bounce_labels
    assert all("_" not in lab for lab in bounce_labels)
    assert "bounce_roster" in ids
    assert "bounce_roster" not in te_ids
    last_year = next(sec for sec in rb["sections"] if sec["id"] == "offense_last_year")
    assert last_year["heading"] == "Offense last year"
    last_labels = [row["label"] for row in last_year["rows"]]
    assert "Top 10 last year" in last_labels
    assert "Top 10 last year, year 1" in last_labels
    assert cap_row["match"]["between"] == [1, 10]
    assert cap_row.get("ranking_edge") is not None or cap_row.get("adjusted_edge") is not None
    assert cap_row.get("pcts", {}).get("top_12") is not None
    assert cap_row.get("pcts", {}).get("top_5") is not None
    assert cap_row.get("pcts", {}).get("top_24") is not None
    from dashboard_services.historical.board import matches_trend_filter as _match_pick
    assert any(
        f.get("position") == "RB" and _match_pick(f, cap_row["match"])
        for f in (payload.get("player_features") or {}).values()
    )
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

    assert format_hist_trend_title(kind="draft_capital", label="Draft capital", bucket="Round 1") == "Drafted NFL Round 1, any season"
    assert format_hist_trend_title(kind="draft_capital", label="Draft capital", bucket="Top 10") == "Drafted NFL Top 10, any season"
    assert format_hist_trend_title(kind="draft_capital", label="Draft capital", bucket="Picks 11-25") == "Drafted NFL Picks 11-25, any season"
    assert format_hist_trend_title(kind="capital_miss", label="Miss rate", bucket="Round 1") == "Drafted NFL Round 1, miss (any season)"
    assert format_hist_trend_title(
        kind="top12_as_rookie", label="Hit top-12 as a rookie", bucket="Round 1"
    ) == "Drafted NFL Round 1, year 1"
    assert format_hist_trend_title(kind="age", label="Age", bucket="23-24") == "Age 23-24, any season"
    assert format_hist_trend_title(kind="age_exact", label="Age", bucket="24") == "Age 24, any season"
    assert format_hist_trend_title(kind="offense", label="Team offense", bucket="Top 10") == "Top-10 projected offense"
    assert format_hist_trend_title(kind="offense_year_1", label="Team offense", bucket="Top 10") == "Top-10 projected offense, year 1"
    assert format_hist_trend_title(kind="offense_last_year", label="Team offense", bucket="Top 10") == "Top-10 offense last year"
    assert format_hist_trend_title(kind="offense_roster", label="Team offense", bucket="Top 10, RB1") == "Top-10 projected offense, RB1"
    assert format_hist_trend_title(
        kind="capital_roster", label="NFL", bucket="Round 1, WR3+"
    ) == "Drafted NFL Round 1, WR3+"
    assert format_hist_trend_title(
        kind="capital_roster_1", label="NFL", bucket="Top 10, RB1"
    ) == "Drafted NFL Top 10, RB1, year 1"
    assert format_hist_trend_title(
        kind="offense_capital", label="Offense", bucket="21-32, Round 1"
    ) == "21-32 projected offense, Round 1"
    assert format_hist_trend_title(
        kind="offense_capital", label="Offense", bucket="Top 10, Top 10"
    ) == "Top-10 projected offense, NFL Top 10"
    assert format_hist_trend_title(
        kind="bounce_roster", label="Last year", bucket="WR1"
    ) == "Outside top 36 last year, WR1"
    assert "_" not in format_hist_trend_title(
        kind="capital_roster", label="NFL", bucket="Day 2, WR1"
    )
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
