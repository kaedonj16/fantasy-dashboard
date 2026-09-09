"""My Leagues pending-card viewer matching guards."""

import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_uses_account_viewer_resolution():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    fn = source.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "resolve_account_viewer_for_league" in fn
    assert "match_viewer_roster" in fn
    assert "sleeper_owner_id_for_account" in fn
    assert "owner_id=sleeper_owner" in fn
    assert 'str(r.get("owner_id")) == str(viewer_user_id)' not in fn
    assert "team_label_from_user" in fn
    assert '"team_name": team_name' in fn


def test_sleeper_owner_id_ignores_unlinked_session_viewer(monkeypatch):
    pytest.importorskip("flask")
    from routes.user_pages_bp import sleeper_owner_id_for_account
    import dashboard_services.accounts as accounts

    monkeypatch.setattr(accounts, "list_account_platform_ids", lambda *a, **k: ["linked-sleeper"])
    assert sleeper_owner_id_for_account(42, "linked-sleeper", "sleeper") == "linked-sleeper"
    assert sleeper_owner_id_for_account(42, "1020439", "sleeper") is None
    assert sleeper_owner_id_for_account(42, "1020439", "fleaflicker") is None
    assert sleeper_owner_id_for_account(None, "sleeper-only", "sleeper") == "sleeper-only"


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
    assert "rank_rosters_by_position" in summary
    assert "league_format_value_lookup" in summary
    assert "from utils.roster_strength import" in summary
    assert "strength_percentile" in summary
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
    assert "select_waiver_add" in source
    assert "already_on_ir=" in source
    assert 'viewer_roster.get("reserve")' in source
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


def test_portfolio_empty_digest_shows_all_caught_up_not_hidden():
    """A PRO user whose leagues have nothing to do must still see the card with
    an all-caught-up note, not have it vanish (which reads as 'missing')."""
    source = (ROOT / "app.py").read_text()
    fn = source.split("def build_portfolio_body")[1].split("\ndef ")[0]
    # The old behavior hid the card on an empty digest; that must be gone.
    assert "if(!acts.length){card.hidden=true;return;}" not in fn
    # New empty state renders inside a visible card.
    assert "pf-moves-empty" in fn
    assert "all caught up" in fn.lower()
    assert ".pf-moves-empty{" in source  # style is defined


def test_changelog_announces_portfolio_moves_empty_state():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        e for e in CHANGELOG
        if "this week's moves" in e.get("text", "").lower()
        and "empty" in e.get("text", "").lower()
    )
    assert entry["tag"] == "fix"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


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


def test_portfolio_record_and_rank_falls_back_to_roster_settings():
    from dashboard_services.ai.context_builders import portfolio_record_and_rank

    lctx = {
        "standings_map": {},
        "rosters": [
            {"roster_id": 1, "settings": {"wins": 2, "losses": 4, "fpts": 90, "fpts_decimal": 0}},
            {"roster_id": 2, "settings": {"wins": 5, "losses": 1, "fpts": 140, "fpts_decimal": 0}},
            {"roster_id": 3, "settings": {"wins": 5, "losses": 1, "fpts": 150, "fpts_decimal": 0}},
        ],
    }
    wins, losses, ties, pf, rank = portfolio_record_and_rank(lctx, "1", lctx["rosters"][0])
    assert wins == 2 and losses == 4
    assert rank == 3
