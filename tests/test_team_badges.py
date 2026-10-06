"""Tests for logo-free team badges (replaces trademarked NFL logos)."""

import pytest


class TestTeamBadgeHtml:
    def test_badge_renders_abbreviation(self):
        from utils.nfl import team_badge_html
        html = team_badge_html("CHI")
        assert "CHI" in html
        assert "team-badge" in html

    def test_badge_uses_team_colors(self):
        from utils.nfl import team_badge_html
        html = team_badge_html("BUF")
        # Bills: primary #00338D, secondary #C60C30
        assert "#00338D" in html
        assert "#C60C30" in html

    def test_badge_sizes(self):
        from utils.nfl import team_badge_html
        for size in ("xs", "sm", "md", "lg"):
            html = team_badge_html("KC", size=size)
            assert f"team-badge-{size}" in html

    def test_badge_invalid_size_defaults_to_sm(self):
        from utils.nfl import team_badge_html
        html = team_badge_html("KC", size="xxl")
        assert "team-badge-sm" in html

    def test_badge_empty_team_returns_empty(self):
        from utils.nfl import team_badge_html
        assert team_badge_html("") == ""
        assert team_badge_html(None) == ""

    def test_badge_canonicalizes_wsh(self):
        from utils.nfl import team_badge_html
        html = team_badge_html("WSH")
        assert "WAS" in html

    def test_badge_all_32_teams(self):
        from utils.nfl import team_badge_html
        from utils.rb_usage import TEAM_COLORS
        assert len(TEAM_COLORS) == 32
        for abbr in TEAM_COLORS:
            html = team_badge_html(abbr)
            assert abbr in html, f"Badge missing for {abbr}"
            assert "team-badge" in html

    def test_badge_has_stripe_styles(self):
        from utils.nfl import team_badge_html
        html = team_badge_html("DAL")
        assert "--badge-primary:" in html
        assert "--badge-secondary:" in html
        assert 'aria-label="DAL"' in html

    def test_badge_no_logo_urls(self):
        from utils.nfl import team_badge_html
        html = team_badge_html("NE")
        assert "espncdn" not in html
        assert "team_logos" not in html
        assert "<img" not in html
