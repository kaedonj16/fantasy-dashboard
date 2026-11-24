"""Tests for the mobile report-card treatment in the prospect modal.

Kaedon approved the settled mobile mock ("good, build it", 2026-11-24):
de-crowded single-column phone layout, a Grade Breakdown section
(component percentiles with a generated soft-spot line), a strict
Season Lines phone grid (Dom column, YoY delta), and a sticky
mini-header that appears once the hero scrolls out of the modal's
own scroll container. These tests pin the shipped page structure;
behavior is verified by the Playwright QA harness.

Synthetic data only; no DB needed.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services.pages.rookies_page import build_prospects_body


def _html():
    return build_prospects_body(False)


class TestGradeBreakdown:
    def test_section_and_renderer_present(self):
        html = _html()
        assert 'id="rkRcGradeSection"' in html
        assert 'id="rkRcGradeBreakdown"' in html
        assert "Grade Breakdown" in html
        assert "function rkRcRenderGrade" in html

    def test_grade_section_precedes_production(self):
        # Same relative position in both layouts: the Grade Breakdown
        # section comes before Production by Year in the main column.
        html = _html()
        assert html.index("rkRcGradeSection") < html.index("Production by Year")

    def test_soft_spot_line_is_generated_from_values(self):
        html = _html()
        assert "is the soft spot" in html
        assert "every other component" in html
        assert "or higher." in html

    def test_side_note_names_the_pool(self):
        html = _html()
        assert "Percentile vs drafted" in html

    def test_components_are_the_payload_percentiles(self):
        html = _html()
        for key in ("production", "efficiency", "breakout", "competition"):
            assert "key: '" + key + "'" in html


class TestMobileStructure:
    def test_in_season_pill_hook(self):
        html = _html()
        assert 'id="rkRcInSeason"' in html
        assert "rc-inseason-pill" in html

    def test_class_standing_is_a_rail_sibling(self):
        # The standing panel reorders independently of the comparables
        # on phones, so it sits beside the comps section in the rail,
        # not inside it.
        html = _html()
        comps_close = html.index('id="rkRcComps"')
        standing = html.index('id="rkRcStanding"')
        assert comps_close < standing

    def test_phone_section_order_classes(self):
        html = _html()
        for cls in ("rc-sec-grade", "rc-sec-prod", "rc-sec-season",
                    "rc-sec-metrics", "rc-sec-notes", "rc-comps-section"):
            assert cls in html

    def test_season_grid_builder_and_dom_source(self):
        html = _html()
        assert "function rkRcSeasonGrid" in html
        assert "rc-season-grid" in html
        assert "advanced_seasons" in html
        assert "function rkRcRateDelta" in html

    def test_metric_percentile_word_span(self):
        # The word "percentile" rides in a span the phone layout
        # hides, leaving the compact "97th" label at the bar's end.
        html = _html()
        assert "rc-mpct-word" in html


class TestStickyMiniHeader:
    def test_mini_markup_present(self):
        html = _html()
        assert 'id="rkRcMini"' in html
        assert 'id="rkRcMiniGrade"' in html
        assert 'id="rkRcMiniBand"' in html
        assert "rc-mini-initials" in html

    def test_observer_watches_hero_in_scroll_container(self):
        html = _html()
        assert "function rkRcSetupMini" in html
        assert "IntersectionObserver" in html
        assert "root: content" in html
        assert "rkModalContent" in html

    def test_reduced_motion_disables_the_slide(self):
        html = _html()
        assert "prefers-reduced-motion" in html
