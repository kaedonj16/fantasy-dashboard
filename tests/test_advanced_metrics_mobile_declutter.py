"""Mobile declutter contracts for the Advanced Metrics toolbar.

On phones the toolbar used to render ~4 rows of set-management controls and
toggles plus three wrapped rows of Decide pills before the table. The
secondary controls now hide behind the existing Filters toggle
(`am-mobile-filter`, same pattern as the week/team/sort controls) and the
Decide pills scroll as one row.
"""
import re

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS

# Controls that collapse behind Filters on mobile (<=600px).
COLLAPSED_IDS = (
    "amSavedSet",
    "amActiveSet",
    "amSaveSetBtn",
    "amDeleteSetBtn",
    "amTrendToggleWrap",
    "amRosterToggleWrap",
)

# Controls that stay visible on mobile.
VISIBLE_IDS = (
    "amFiltersBtn",   # the toggle itself
    "amAddStatBtn",   # + Metric
    "amPositions",
    "amDecisionPills",
)


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _tag(html, el_id):
    m = re.search(r'<[^>]*\bid="%s"[^>]*>' % re.escape(el_id), html)
    assert m, f"#{el_id} missing from page"
    return m.group(0)


def test_secondary_controls_collapse_behind_filters_on_mobile():
    html = _html()
    for el_id in COLLAPSED_IDS:
        tag = _tag(html, el_id)
        assert "am-mobile-filter" in tag, f"#{el_id} should collapse on mobile: {tag[:120]}"


def test_primary_controls_stay_visible_on_mobile():
    html = _html()
    for el_id in VISIBLE_IDS:
        tag = _tag(html, el_id)
        assert "am-mobile-filter" not in tag, f"#{el_id} must stay visible on mobile"


def test_mobile_filter_css_hides_collapsed_controls():
    html = _html()
    assert "#amToolbar:not(.am-open) .am-mobile-filter" in html


def test_decide_pills_scroll_as_one_row_on_mobile():
    html = _html()
    style = html[html.index("<style>"):html.index("</style>")]
    # Find the 600px media block containing the decisions rule.
    assert ".am-decisions { flex-wrap:nowrap; overflow-x:auto;" in style
    assert ".am-decisions .am-decisions-label, .am-decisions .am-pill { flex-shrink:0; }" in style