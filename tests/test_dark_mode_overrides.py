"""Dark-mode override contract: every component flagged in the 2026-09-30
dark-mode audit must carry a [data-theme="dark"] variant.

The app's base theme is light; dark mode is a [data-theme="dark"]
attribute on <html>. Components that hardcode light backgrounds or dark
text must each have a matching dark rule, otherwise they render as bright
bands or washed-out text on the dark ground. This happened with the Trade
Calculator (.otc-* cluster), the pastel status chips, the Front Office
grade colors, and the Plotly charts (the toggle only restyled the two
team-modal charts).

These tests pin the dark overrides in static/dashboard.css and the
all-charts behavior of updatePlotlyChartsTheme() in static/app.js so a
future light-only addition fails loudly instead of shipping dark-broken.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "dashboard.css"
JS = ROOT / "static" / "app.js"


@pytest.fixture(scope="module")
def css():
    return CSS.read_text()


@pytest.fixture(scope="module")
def js():
    return JS.read_text()


def has_dark_override(css_text, selector):
    """A [data-theme="dark"] rule mentioning this exact selector exists."""
    pattern = r'\[data-theme="dark"\]\s*' + re.escape(selector) + r'(?![\w-])'
    return re.search(pattern, css_text) is not None


# ---------------------------------------------------------------------------
# Systemic: color-scheme + scrollbars
# ---------------------------------------------------------------------------

def test_color_scheme_dark_declared(css):
    assert re.search(
        r':root\[data-theme="dark"\]\s*\{[^}]*color-scheme:\s*dark', css
    ), "dark theme must declare color-scheme: dark so native scrollbars and form controls render dark"


def test_scrollbox_dark_scrollbar(css):
    assert has_dark_override(css, ".scroll-box"), \
        "the .scroll-box hardcodes a light scrollbar-color with no dark variant"


# ---------------------------------------------------------------------------
# Trade Calculator (.otc-*) cluster
# ---------------------------------------------------------------------------

OTC_SELECTORS = [
    ".otc-team-selector",
    ".otc-team-pill",
    ".otc-page-badge",
    ".otc-dropdown-rank-inline",
    '.otc-dropdown-item[data-position="PICK"]',
    ".otc-mini-tab",
    ".otc-slot-empty",
    ".otc-slot-empty-sub",
    ".otc-team-label",
    ".otc-value-sub",
    ".otc-summary-sub",
    ".otc-chip-meta",
    ".otc-chip-remove",
    ".otc-mini-sub",
    ".otc-dropdown-sub",
    ".otc-balance-labels",
    ".otc-balance-fair-label",
    ".otc-team-owner-tag",
    ".otc-team-owner-tag-muted",
    ".otc-guest-link",
    ".otc-view-all-link:hover",
    ".otc-inline-player",
    ".otc-sugg-pkg-value.fair",
    ".otc-sugg-pkg-value.great",
    ".otc-sugg-pkg-value.overpay",
    ".otc-mini-row.up .otc-mini-delta",
    ".otc-mini-row.down .otc-mini-delta",
    ".otc-chip-delta-positive",
    ".otc-chip-delta-negative",
    ".otc-share-btn-success",
    ".otc-delta.po",
]


@pytest.mark.parametrize("selector", OTC_SELECTORS)
def test_otc_dark_override(css, selector):
    assert has_dark_override(css, selector), \
        f"{selector} hardcodes light colors with no dark variant"


# ---------------------------------------------------------------------------
# Pastel status chips
# ---------------------------------------------------------------------------

PASTEL_CHIP_SELECTORS = [
    ".chip.diff-pos",
    ".chip.diff-neg",
    ".outcome-win",
    ".outcome-got",
    ".outcome-loss",
    ".outcome-gave",
    ".outcome-even",
    ".urgency-high",
    ".urgency-medium",
    ".urgency-low",
    ".suggestion-asset.give",
    ".suggestion-asset.get",
    ".io.add",
    ".io.drop",
    ".depth-danger",
    ".depth-caution",
    ".streak-cold .chip-streak",
    ".streak-hot .chip-streak",
]


@pytest.mark.parametrize("selector", PASTEL_CHIP_SELECTORS)
def test_pastel_chip_dark_override(css, selector):
    assert has_dark_override(css, selector), \
        f"{selector} is a pastel chip that glows on the dark ground"


# ---------------------------------------------------------------------------
# Front Office grades / trends / roles
# ---------------------------------------------------------------------------

FOR_SELECTORS = [
    ".for-grade-a",
    ".for-grade-b",
    ".for-grade-c",
    ".for-grade-d",
    ".for-grade-f",
    ".for-inj",
    ".for-role-starter",
    ".for-role-depth",
    ".for-trend-up",
    ".for-trend-down",
]


@pytest.mark.parametrize("selector", FOR_SELECTORS)
def test_front_office_dark_override(css, selector):
    assert has_dark_override(css, selector), \
        f"{selector} is a dark-toned literal with ~3:1 contrast on dark cards"


# ---------------------------------------------------------------------------
# Small utility chips, badges, and saturated text
# ---------------------------------------------------------------------------

UTILITY_SELECTORS = [
    ".badge",
    ".highlight-box",
    ".link-pill",
    ".changelog-tag-fix",
    ".luck-chip.luck-neu",
    ".game-log-proj-badge",
    ".opt-pos-qb",
    ".opt-pos-rb",
    ".opt-pos-wr",
    ".opt-pos-te",
    ".wt-contend .grade-window-label",
    ".wt-rebuilding .grade-window-label",
    ".rz-game-pill.is-live .rz-gp-status",
    ".team-strength-card .tsc-pi-num.dn",
    ".settings-menu-logout",
    ".analytics-bar-val.analytics-bar-neg",
    ".watchlist-clear-btn:hover",
    ".wl-page-remove:hover",
    ".sched-remove:hover",
    ".game-log-matchup.mt4",
    ".game-log-table-total td:nth-child(2)",
    ".podium .slot.second .podium-header h3",
    ".podium .slot.third .podium-header h3",
    ".rank-third",
    ".rivalry-chip-streak",
    ".rivalry-chip-blowout",
    ".card-collapse-toggle",
    ".card-collapse-toggle:hover",
    ".legend-label",
    ".home-bullets",
    ".history-table thead th",
    ".viewer-toggle",
    ".os-section-subtitle",
    ".os-waiver-sub",
    ".pick-via",
    ".dvt-tier-depth",
]


@pytest.mark.parametrize("selector", UTILITY_SELECTORS)
def test_utility_dark_override(css, selector):
    assert has_dark_override(css, selector), \
        f"{selector} hardcodes light colors with no dark variant"


# ---------------------------------------------------------------------------
# Inverted-token active pills (inline page styles)
#
# A second failure class the literal-color scan missed: active states styled
# background:var(--text); color:var(--card). In light mode that is a dark
# pill; in dark mode --text is near-white, so the pill glares (the Trade Hub
# "Suggestions" subtab). Each one needs a dark override in its page file.
# ---------------------------------------------------------------------------

TRADE_PAGE = ROOT / "dashboard_services" / "pages" / "trade_calculator_page.py"
AM_PAGE = ROOT / "dashboard_services" / "pages" / "advanced_metrics_page.py"


@pytest.fixture(scope="module")
def trade_page():
    return TRADE_PAGE.read_text()


@pytest.fixture(scope="module")
def am_page():
    return AM_PAGE.read_text()


@pytest.mark.parametrize("selector", [
    ".otc-sugg-subtab.is-active",
    ".otc-sugg-subtab-toggle .br-slide-ind",
    ".otc-mode-btn.is-active",
])
def test_trade_page_inverted_pill_dark_override(trade_page, selector):
    assert has_dark_override(trade_page, selector), \
        f"{selector} uses the inverted --text fill with no dark variant"


@pytest.mark.parametrize("selector", [
    ".am-pos.active",
    ".am-chip.am-chip-primary",
    ".am-positions.am-segmented .am-pos.active",
])
def test_am_page_inverted_pill_dark_override(am_page, selector):
    assert has_dark_override(am_page, selector), \
        f"{selector} uses the inverted --text fill with no dark variant"


# ---------------------------------------------------------------------------
# Plotly: theme toggle must restyle every chart, not just the team pair
# ---------------------------------------------------------------------------

def _toggle_fn(js_text):
    start = js_text.index("function updatePlotlyChartsTheme()")
    end = js_text.index("function updateThemeIcons()")
    return js_text[start:end]


def test_plotly_toggle_restyles_all_charts(js):
    fn = _toggle_fn(js)
    assert "querySelectorAll('.js-plotly-plot')" in fn, \
        "updatePlotlyChartsTheme must restyle every Plotly chart, not a hardcoded pair"


def test_plotly_toggle_not_limited_to_team_charts(js):
    fn = _toggle_fn(js)
    assert "getElementById('teamWeeklyChart')" not in fn, \
        "the toggle must not be limited to the two team-modal charts anymore"
    assert "getElementById('teamRadarChart')" not in fn


def test_plotly_toggle_covers_polar_charts(js):
    fn = _toggle_fn(js)
    assert "polar.radialaxis.gridcolor" in fn, \
        "radar charts still need polar chrome restyled on toggle"
    assert "hoverlabel.bgcolor" in fn


# ---------------------------------------------------------------------------
# Medal and rank badges: the last cluster from the 2026-09-30 audit.
# .dyn-rank.rank-1/2/3 and the .os-snap-medal top-3 variants carry light
# metallic gradients with near-black text in their base rules. The dark
# overrides use the deepened metallic variants from the approved visual
# improvements mock (Dark Mode tab); the base light rules must not move.
# ---------------------------------------------------------------------------

MEDAL_DARK_SELECTORS = [
    ".dyn-rank.rank-1",
    ".dyn-rank.rank-2",
    ".dyn-rank.rank-3",
    ".os-snap-row.is-1 .os-snap-medal",
    ".os-snap-row.is-2 .os-snap-medal",
    ".os-snap-row.is-3 .os-snap-medal",
]


@pytest.mark.parametrize("selector", MEDAL_DARK_SELECTORS)
def test_medal_dark_override(css, selector):
    assert has_dark_override(css, selector), \
        f"{selector} is a light metallic badge with no dark variant"


def _dark_rule_body(css_text, selector):
    pattern = r'\[data-theme="dark"\]\s*' + re.escape(selector) + r'\s*\{([^}]*)\}'
    match = re.search(pattern, css_text)
    assert match, f"no dark rule for {selector}"
    return match.group(1)


MEDAL_DARK_VALUES = [
    # selector, gradient stops, text color, box-shadow (os-snap only)
    (".dyn-rank.rank-1", "#b45309", "#92400e", "#fde68a", None),
    (".dyn-rank.rank-2", "#475569", "#334155", "#e2e8f0", None),
    (".dyn-rank.rank-3", "#9a5b2e", "#7c4522", "#fed7aa", None),
    (".os-snap-row.is-1 .os-snap-medal", "#b45309", "#92400e", "#fde68a",
     "0 3px 10px rgba(245, 158, 11, .22)"),
    (".os-snap-row.is-2 .os-snap-medal", "#475569", "#334155", "#e2e8f0",
     "0 3px 10px rgba(148, 163, 184, .18)"),
    (".os-snap-row.is-3 .os-snap-medal", "#9a5b2e", "#7c4522", "#fed7aa",
     "0 3px 10px rgba(194, 128, 63, .2)"),
]


@pytest.mark.parametrize(
    "selector,stop1,stop2,color,shadow", MEDAL_DARK_VALUES
)
def test_medal_dark_values(css, selector, stop1, stop2, color, shadow):
    body = _dark_rule_body(css, selector)
    assert f"background: linear-gradient(135deg, {stop1}, {stop2});" in body
    assert f"color: {color};" in body
    if shadow is not None:
        assert f"box-shadow: {shadow};" in body


def test_medal_dark_rules_before_icons_merge_marker(css):
    marker = css.index("MERGED: icons.css")
    for selector in MEDAL_DARK_SELECTORS:
        pattern = r'\[data-theme="dark"\]\s*' + re.escape(selector) + r'\s*\{'
        match = re.search(pattern, css)
        assert match and match.start() < marker, \
            f"dark rule for {selector} must sit before the icons merge marker"


def test_medal_light_base_rules_unchanged(css):
    """Light mode must be visually unchanged: the base metallic rules stay."""
    assert (
        ".dyn-rank.rank-1 {\n"
        "    background: linear-gradient(135deg, #fcd34d, #f59e0b);\n"
        "    color: #3f2d03;\n}"
    ) in css
    assert (
        ".dyn-rank.rank-2 {\n"
        "    background: linear-gradient(135deg, #eef2f7, #a9b6c7);\n"
        "    color: #27303d;\n}"
    ) in css
    assert (
        ".dyn-rank.rank-3 {\n"
        "    background: linear-gradient(135deg, #eabb92, #c2803f);\n"
        "    color: #3a2408;\n}"
    ) in css
    assert (
        ".os-snap-row.is-1 .os-snap-medal {\n"
        "    background: linear-gradient(135deg, #fcd34d, #f59e0b);\n"
        "    color: #3f2d03;"
    ) in css
    assert (
        ".os-snap-row.is-2 .os-snap-medal {\n"
        "    background: linear-gradient(135deg, #eef2f7, #a9b6c7);\n"
        "    color: #27303d;"
    ) in css
    assert (
        ".os-snap-row.is-3 .os-snap-medal {\n"
        "    background: linear-gradient(135deg, #eabb92, #c2803f);\n"
        "    color: #3a2408;"
    ) in css


def test_medal_dark_mode_changelog_entry():
    from pin_helpers import changelog_entry

    entry = changelog_entry("top 3 dynasty rank badges", "deepened metallic", tag="fix")
    assert entry["date"] == "2026-10-29"
    assert "snapshot medals" in entry["text"]
