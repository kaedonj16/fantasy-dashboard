"""Tests for the fallback prospect modal's parity with the report card.

The prospects page renders its report card with the rkRc* functions in
rookies_page.py. Every other page reaches the modal through
openProspectModal in static/app.js, which renders the same report card
(the rcfRc* port) from the player endpoint's merged payload. The
report-card styles are shared from static/dashboard.css (they used to
live in the prospects page's own <style> block, which no other page
could see). These tests pin that arrangement; rendering behavior is
verified by the Playwright QA harness.

Synthetic data only; no DB needed.
"""

import os
import sys

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def _fallback_region(app_js):
    start = app_js.index("Prospect report card modal (shared)")
    end = app_js.index("// closePlayerModal lives in static/player_modal.js")
    return app_js[start:end]


class TestSharedStyles:
    def test_report_card_css_lives_in_dashboard_css(self):
        css = _read("static/dashboard.css")
        assert ".rc-hero {" in css
        assert ".rc-arch-ev" in css
        assert ".rc-season-grid" in css
        # The shared block must sit before the merged icons.css block,
        # which the merge script owns.
        assert css.index(".rc-hero {") < css.index("MERGED: icons.css")

    def test_page_no_longer_embeds_the_modal_css(self):
        from dashboard_services.pages.rookies_page import build_prospects_body

        html = build_prospects_body(False)
        assert ".rc-hero {" not in html
        # The page's own renderers and structure are untouched.
        assert "function rkRcRenderGrade" in html
        assert 'id="rkRcGradeSection"' in html
        assert "function rkOpenModal" in html


class TestFallbackPort:
    def test_open_prospect_modal_renders_the_report_card(self):
        app_js = _read("static/app.js")
        region = _fallback_region(app_js)
        assert "function openProspectModal" in region
        assert "function rcfOpenReportCard" in region
        assert "/api/prospects/player/" in region
        for fn in ("rcfRcRenderArch", "rcfRcRenderGrade", "rcfRcRenderChart",
                   "rcfRcRenderSeasonTable", "rcfRcRenderMetrics",
                   "rcfRcRenderNotes", "rcfRcRenderComps", "rcfRcSetupMini"):
            assert "function " + fn in region, fn

    def test_old_fallback_design_is_gone(self):
        app_js = _read("static/app.js")
        assert "rkModalStyles" not in app_js
        region = _fallback_region(app_js)
        assert "rk-comp-row" not in region
        assert "Draft Cap." not in region
        # Kaedon's colors only: no amber anywhere in the port.
        assert "#f59e0b" not in region

    def test_shell_matches_the_page_shell(self):
        region = _fallback_region(_read("static/app.js"))
        assert "max-width:1080px" in region
        assert "function createRkModalIfMissing" in region
