"""Pins for the Get started card header composition.

The header used to force the trust line onto its own full-width centered
row (an orphan line with dead flanks). The lead and the trust line now
form one text block on the left with the Google pill on the right, in
both stylesheets, and the lead's old flex basis lives on the wrapper:
as a column flex item it would otherwise become the lead's HEIGHT.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
LITE = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")
DASH = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_account_text_wraps_lead_and_trust_before_the_pill():
    start = APP_PY.index('<div class="home-account-entry home-account-top">')
    block = APP_PY[start:APP_PY.index("</div>", APP_PY.index("home-hero-trust", start))]
    assert '<div class="home-account-text">' in block
    assert block.index("home-account-text") < block.index("home-account-lead")
    assert block.index("home-account-lead") < block.index("home-hero-trust")
    entry_end = APP_PY.index("</div>", APP_PY.index("home-hero-trust", start))
    after = APP_PY[entry_end:entry_end + 400]
    assert "google-continue-btn" in after


def test_header_composition_rules_in_both_sheets():
    for css in (LITE, DASH):
        assert re.search(
            r"\.home-account-text \{ display: flex; flex-direction: column;"
            r" gap: 2px; min-width: 0; flex: 1 1 240px; \}", css)
        assert ".home-account-top .home-account-lead { flex: none;" in css
        assert ".home-account-top .home-hero-trust { margin: 0; text-align: left; }" in css
        assert "justify-content: space-between" in css
        # In the stacked phone layout the wrapper's row-basis would become
        # its height; the media query must neutralize its flex sizing.
        assert ".home-account-text { flex: none; }" in css


def test_get_started_is_one_card_in_light_mode():
    """The approved mock has the header INSIDE one card with the columns;
    a light-mode override used to make .home-card transparent and leave
    the surface on the columns grid alone, splitting the section in two.
    In light mode .home-card carries the surface and the grid drops to a
    divider-topped inner grid; dark mode keeps its slab + panel."""
    for css in (LITE, DASH):
        assert ('html:not([data-theme="dark"]) .home-card { background: var(--card);'
                ' color: #122d4b; border: 1px solid var(--border);') in css
        assert ('html:not([data-theme="dark"]) .ld-connect-grid { background: transparent;'
                ' border: none; border-top: 1px solid var(--border);') in css
        assert ('html:not([data-theme="dark"]) .home-account-top { margin-bottom: 0;'
                ' padding: 20px 22px 16px; border-bottom: none; }') in css
        assert 'html:not([data-theme="dark"]) .home-card { background: transparent' not in css


def test_orphan_trust_row_rule_is_gone():
    for css in (LITE, DASH):
        assert ".home-account-top .home-hero-trust { flex-basis: 100%" not in css
        assert not re.search(
            r"\.home-account-top \{[^}]*flex-wrap: wrap;[^}]*justify-content: center;", css)
