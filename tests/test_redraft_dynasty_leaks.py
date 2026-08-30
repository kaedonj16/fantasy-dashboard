"""Redraft surfaces must not show dynasty window / pick / age-core signals."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_graphs_page_skips_value_vs_age_for_redraft():
    src = (ROOT / "dashboard_services" / "pages" / "graphs_page.py").read_text(encoding="utf-8")
    assert 'ctx_scoring_type(ctx) != "redraft"' in src
    assert "Dynasty Value vs Age" in src


def test_team_modal_skips_value_vs_age_for_redraft():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("@app.route(\"/api/team-details/<roster_id>\")")
    end = src.find("def api_player_league_trades", start)
    body = src[start:end]
    assert "if not is_redraft:" in body
    assert "value_age_svg" in body


def test_offseason_dashboard_clears_picks_for_redraft():
    src = (ROOT / "dashboard_services" / "pages" / "offseason_dashboard_page.py").read_text(
        encoding="utf-8"
    )
    assert "if _league_is_redraft(ctx):" in src
    assert "picks_by_roster = {}" in src
    assert "_waiver_value_keys(ctx)" in src


def test_offseason_standings_hides_draft_capital_for_redraft():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _build_offseason_standings_body")
    end = src.find("def _waiver_value_keys", start)
    body = src[start:end]
    assert "is_redraft = _league_is_redraft(ctx)" in body
    assert "picks_by_roster = {} if is_redraft" in body
    assert "_waiver_value_keys(ctx)" in body
    assert 'this-season redraft value' in body
    assert "dyn-capital" in body


def test_share_card_uses_redraft_labels_not_espn_only():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def page_share_card")
    end = src.find("def share_card_og_image", start)
    if end < 0:
        end = src.find("@app.route", start + 10)
    body = src[start:end]
    assert "is_redraft = _league_is_redraft(ctx)" in body
    assert '"Roster Rank" if is_redraft else "Dynasty Rank"' in body
    assert '"Roster Value" if is_redraft else "Dynasty Value"' in body
    assert "_picks = [] if is_redraft" in body
    assert "if _pick_labels and not is_redraft:" in body
    assert 'league_info.get("name") or "League"' in body


def test_trade_window_card_skips_age_for_redraft():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _trade_window_card_html")
    end = src.find("def _render_season_review_card")
    body = src[start:end]
    assert "is_redraft = _league_is_redraft(ctx)" in body
    assert 'titles = {"buy": "Playoff push"' in body
    assert "if not is_redraft:" in body
    assert "stacking picks" in body  # dynasty copy still exists
    assert "if not is_redraft:" in body
    # Week 1 ESPN was painting "Trade deadline: Playoff push" with no deadline.
    assert "redraft_deadline_card_visible(weeks_to)" in body
    assert "trade_deadline_ts" in body


def test_trade_calc_hides_rebuilding_chip_for_redraft():
    from dashboard_services.pages.trade_calculator_page import build_trade_calculator_body

    redraft = build_trade_calculator_body("L1", 2026, scoring_type="redraft")
    dynasty = build_trade_calculator_body("L1", 2026, scoring_type="dynasty")
    assert 'data-arch="rebuilding"' not in redraft
    assert 'data-arch="rebuilding"' in dynasty
    assert "Choose Contending, Consolidate, or Distribute above." in redraft
    assert "Rebuilding" in dynasty
    assert "Personalized to your playoff standing and roster" in redraft
    assert "Personalized to your team direction and roster lens" in dynasty
    guest_rd = build_trade_calculator_body(None, 2026, scoring_type="redraft")
    guest_dyn = build_trade_calculator_body(None, 2026, scoring_type="dynasty")
    assert "AI-powered trade analysis for this season" in guest_rd
    assert "AI-powered trade analysis for dynasty leagues" in guest_dyn


def test_strategy_js_has_redraft_copy_and_hides_rebuild():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "function _archDesc()" in js
    assert "Win this season. Trade for proven weekly production" in js
    assert 'rebuildChip.style.display = redraft ? "none" : ""' in js
    assert 'getScoringType() === "redraft" && arch === "rebuilding"' in js


def test_roster_grade_uses_scoring_type_on_teams_page():
    src = (ROOT / "dashboard_services" / "pages" / "teams_page.py").read_text(encoding="utf-8")
    assert "scoring_type=_scoring" in src


def test_trade_intel_skips_rebuild_window_for_redraft():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def api_trade_intel_player_packages")
    end = src.find("def _real_trade_packages_for_target", start)
    body = src[start:end]
    assert "is_redraft = bool(ctx) and _league_is_redraft(ctx)" in body
    assert "if not is_redraft:" in body
    assert "redraft_value_sf" in body
    assert '_pkg_has_pick' in body
    assert "PICK" in body


def test_roster_intel_uses_league_is_redraft():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _api_roster_intel_compute")
    end = src.find("def api_trade_targets", start)
    body = src[start:end]
    assert "is_redraft = _league_is_redraft(ctx)" in body
    assert "redraft_value_sf" in body


def test_brctx_and_compare_flip_scoring_type():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "scoringType:{league_scoring_type_js}" in app
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "function brScoringType()" in js
    assert "function _cmpIsRedraft()" in js
    assert "Redraft Value (1QB)" in js
    ranks = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
    assert "__leagueScoringType" in ranks
    assert "this-season redraft value" in ranks


def test_roster_grade_badge_skips_age_for_redraft():
    src = (ROOT / "dashboard_services" / "ai" / "renderer.py").read_text(encoding="utf-8")
    start = src.find("def get_roster_grade")
    end = src.find("def get_power_rankings_html", start)
    body = src[start:end]
    assert "scoring_type = ctx_scoring_type(ctx)" in body
    assert "future_picks = [] if is_redraft" in body
    assert "scoring_type=scoring_type" in body
    assert "redraft_window_label(" in body
    assert 'if scoring != "redraft":' in body
    assert "Age:" in body
