from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_recap_prompt_has_natural_record_and_power_rules():
    source = (ROOT / "dashboard_services/ai/weekly_recap.py").read_text()
    assert "Format records naturally as 1-0" in source
    assert "Use % rather than the word percent" in source
    assert "Standings describe what has happened" in source
    assert "Power rank describes how strong a team looks" in source
    assert "rank_gap" in source
    assert "v12_top_performers" in source
    assert 'em dash character "—"' in source
    assert "does NOT apply to regular hyphens in compound words" in source
    assert 'Format records naturally as 2-0, never as tuples or "2, 0"' in source


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


def test_recap_sections_follow_editorial_order_and_end_with_up_next():
    """The recap moves from the completed week to its forward-looking ending."""
    source = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    returned = source.split("    return (preview_banner + history_banner + week_selector", 1)[1]
    expected = ["cards_html", "story_html", "scoreboard_html", "lineup_html",
                "standings_html", "around_html", "up_next_html"]
    positions = [returned.index(item) for item in expected]
    assert positions == sorted(positions)
    assert returned.index("up_next_html") < returned.index(")\n", returned.index("up_next_html"))


def test_historical_movement_is_week_capped_and_week_one_safe():
    source = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    assert "if selected_week > 1:" in source
    assert "build_standings_as_of_week(recap_ctx, selected_week - 1)" in source
    assert "build_power_rankings_context(prior_ctx)" in source
    assert "prior_standings else ''" in source


def test_recap_keeps_preview_entitlements_divisions_and_new_section_names():
    source = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    for text in ("get_weekly_ai_recap_preview", "get_weekly_ai_recap_teaser",
                 "resolve_divisions", "Week at a Glance", "The Story of Week",
                 "How the Week Finished", "Decisions That Mattered", "What Changed",
                 "Around the League", "Up Next — Week"):
        assert text in source


def test_matchup_badges_are_selective_and_do_not_use_fake_upsets():
    from dashboard_services.pages.recap_page import _matchup_badges

    games = [
        {"margin": 1, "w_pts": 100, "l_pts": 99},
        {"margin": 40, "w_pts": 140, "l_pts": 100},
        {"margin": 10, "w_pts": 120, "l_pts": 110},
    ]
    labels = _matchup_badges(games)
    assert len(labels) < len(games)
    assert all(len(badges) <= 2 for badges in labels.values())
    assert all("Upset" not in badge for badges in labels.values() for badge in badges)


def test_mobile_recap_no_longer_depends_on_sidebar_grid():
    page = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    css = (ROOT / "static/dashboard.css").read_text()
    assert "recap-scoreboard-grid" not in page
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in css
    assert "grid-template-columns: minmax(0, 1fr) minmax(88px, 30vw) minmax(0, 1fr)" in css
