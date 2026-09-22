from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_recap_prompt_has_natural_record_and_power_rules():
    source = (ROOT / "dashboard_services/ai/weekly_recap.py").read_text()
    assert "Format records naturally as 1-0" in source
    assert "Use % rather than the word percent" in source
    assert "Standings describe what has happened" in source
    assert "Power rank describes how strong a team looks" in source
    assert "rank_gap" in source
    assert "v13_story" in source
    assert "Do not recap the award cards" in source


def test_lineup_selectors_share_one_historical_calculation():
    source = (ROOT / "dashboard_services/recap_calculations.py").read_text()
    app = (ROOT / "app.py").read_text()
    assert '"underperformers": underperformers' in source
    assert '"bench_gems": gems' in source
    assert "build_lineup_analysis(" in app


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
    returned = source.split("    return ('<main class=\"weekly-recap\">'", 1)[1]
    expected = ["week_selector", "cards_html", "story_html", "scoreboard_html",
                "lineup_html", "standings_html", "up_next_html"]
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
                 "resolve_divisions", "Week at a Glance", "Weekly Story",
                 "Scoreboard", "Lineup Review", "Standings &amp; Power Rankings",
                 "Up Next — Week"):
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
    assert ".weekly-recap" in css
    assert "max-width: 1240px" in css
    assert "grid-template-columns:1fr" in css


def test_all_power_ranked_teams_are_rendered():
    source = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    loop = source.split("for i, p in enumerate(power_teams, 1):", 1)[1].split("power_html =", 1)[0]
    assert "power_rows.append" in loop
    assert "if i not in" not in loop


def test_weekly_story_and_lineup_review_layout_contract():
    """Recap-specific styles keep the story wide and lineup groups content-sized."""
    app = (ROOT / "app.py").read_text()
    css = (ROOT / "static/dashboard.css").read_text()

    assert ".weekly-recap .recap-story > .card { width:100%; max-width:none;" in css
    assert ".weekly-recap .recap-story [data-br-reveal-text] > *" in css
    assert ".weekly-recap .recap-lineup-cols { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); align-items:start; }" in css
    assert ".weekly-recap .rlc-section:nth-child(3) { grid-column:1 / -1; }" in css
    assert "@media (max-width:900px)" in css
    assert ".weekly-recap .recap-lineup-cols { display:block; }" in css
    assert ".rlc-section + .rlc-section { border-top:1px solid var(--border); }" in css
    assert "grid-template-columns:repeat(3,minmax(0,1fr))" not in css

    # Each missed opportunity is emitted as one expandable unit with both
    # players and its own gain, rather than as independently counted rows.
    missed = app.split("    def missed_row(item):", 1)[1].split("    missed_rows =", 1)[0]
    assert "rc-swap-row" in missed
    assert 'swap_player(started, "Started", "loss")' in missed
    assert 'swap_player(reserve, "Benched", "win")' in missed
    assert "Potential gain" in missed
    assert "player_row(" not in missed


def test_lineup_sections_have_independent_accessible_expansion_controls():
    app = (ROOT / "app.py").read_text()
    section = app.split("    def section(title, note, rows):", 1)[1].split("    viewer =", 1)[0]

    assert "rows[:3]" in section and "rows[3:]" in section
    assert "<details class='recap-lineup-more'>" in section
    assert "Show all" in section and "Show less" in section
    assert "<section class='rlc-section'>" in section
