"""My Leagues pending-card viewer matching guards."""

import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_uses_account_viewer_resolution():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    fn = source.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "resolve_account_viewer_for_league" in fn
    assert "match_viewer_roster" in fn
    assert 'owner_id=viewer_user_id if lg_platform == "sleeper"' in fn
    assert 'str(r.get("owner_id")) == str(viewer_user_id)' not in fn


def test_portfolio_undrafted_leagues_use_startup_draft_phase():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    summary = source.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "startup_draft_phase" in summary
    assert "draft_start_ms" in summary
    assert '"draft_phase": draft_phase' in summary
    assert 'draft_phase != "drafted"' in summary
    # Thin pre-draft shells must not fall through to positional ranks.
    assert "pos_user_rank" in summary
    phase_gate = summary.split('draft_phase != "drafted"')[0]
    ranks_after = summary.split('draft_phase != "drafted"')[1]
    assert "pos_user_rank" in ranks_after
    assert "pos_user_rank" not in phase_gate


def test_portfolio_positional_strength_uses_in_league_percentiles():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    summary = source.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "from utils.roster_strength import strength_percentile" in summary
    assert "pos_user_pctile[pos] = strength_percentile(" in summary
    assert '"pos_user_pctile": pos_user_pctile' in summary

    blend = source.split("Cross-league positional strength")[1].split("valid_leagues.sort")[0]
    assert "average_league_percentiles" in blend
    assert "pos_user_pctile" in blend
    # Ratio-vs-median blend is what made stacked leagues read negative.
    assert "u / a" not in blend


def test_portfolio_actions_api_wired():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    assert '@user_pages_bp.route("/api/portfolio-actions")' in source
    assert "rank_cross_league_actions" in source
    assert "lineup_actions_from_issues" in source
    assert "injury_stash_action" in source
    assert "_portfolio_viewer_has_pro" in source
    assert '"paywall": True' in source


def test_portfolio_body_moves_card_pro_gated():
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    assert "showPaywall" in fn
    assert "__brctx.isPremium" in fn or "isPremium" in fn


def test_portfolio_body_includes_moves_card():
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    assert "pfMovesCard" in fn
    assert "/api/portfolio-actions" in fn
    assert "moves_card" in fn
    assert "top_strip + moves_card + league_card" in fn
    assert "pf-move-row" in fn
    assert "pf-moves-list" in fn


def test_portfolio_record_and_rank_accepts_seed_int_standings_map():
    from dashboard_services.ai.context_builders import portfolio_record_and_rank

    lctx = {
        "standings_map": {1: 3, 2: 1},
        "roster_map": {"1": "Team A", "2": "Team B"},
        "rosters": [
            {
                "roster_id": 1,
                "settings": {
                    "wins": 2, "losses": 1, "ties": 0,
                    "fpts": 120, "fpts_decimal": 50,
                },
            },
            {"roster_id": 2, "settings": {"wins": 3, "losses": 0, "fpts": 140}},
        ],
    }
    wins, losses, ties, pf, rank = portfolio_record_and_rank(lctx, "1", lctx["rosters"][0])
    assert wins == 2 and losses == 1 and ties == 0
    assert pf == pytest.approx(120.5)
    assert rank == 3


def test_portfolio_record_and_rank_accepts_dict_standings_map():
    from dashboard_services.ai.context_builders import portfolio_record_and_rank

    lctx = {
        "standings_map": {
            "1": {"wins": 5, "losses": 2, "ties": 0, "pf": 800.0},
            "2": {"wins": 4, "losses": 3, "ties": 0, "pf": 750.0},
        },
        "rosters": [
            {"roster_id": 1, "settings": {}},
            {"roster_id": 2, "settings": {}},
        ],
    }
    wins, losses, ties, pf, rank = portfolio_record_and_rank(lctx, "1", lctx["rosters"][0])
    assert wins == 5 and losses == 2
    assert pf == pytest.approx(800.0)
    assert rank == 1
