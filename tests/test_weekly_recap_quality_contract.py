from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_recap_prompt_has_natural_record_and_power_rules():
    source = (ROOT / "dashboard_services/ai/weekly_recap.py").read_text()
    assert "Format records naturally as 1-0" in source
    assert "Use % rather than the word percent" in source
    assert "Standings describe what has happened" in source
    assert "Power rank describes how strong a team looks" in source
    assert "rank_gap" in source
    assert "v11_power_divisions" in source


def test_lineup_selectors_supply_six_real_candidates():
    source = (ROOT / "app.py").read_text()
    assert "busts = bust_pool[:6]" in source
    assert "sleepers = all_bench[:6]" in source


def test_recap_uses_shared_historical_and_picture_resolvers():
    source = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    assert 'recap_ctx["df_weekly"] = fin_df' in source
    assert "build_standings_as_of_week(recap_ctx, selected_week)" in source
    assert "build_power_rankings_context(historical_ctx)" in source
    assert "team_avatar(_platform, roster, users)" in source
    assert "recap-rank-grid" in source
