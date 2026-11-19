"""Pins for the connect card title and returning-row composition.

The guest card is titled "Load your league." (the approved mock's
heading). The old top-of-card account block (lead + trust line + Google
pill above the connect flow) is gone: sign-in now lives in one quiet
"Returning?" row at the bottom of connect Page 1, under the trust line,
with the same real Google button at auto width beside the text.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
LITE = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")
DASH = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_guest_card_title_is_load_your_league():
    at = APP_PY.index('id="homeCardTitle"')
    title = APP_PY[at - 80:at + 160]
    # The signed-in branch of the same element keeps its own wording.
    assert "Your leagues" in title
    assert "Load your league." in title
    assert "Get started" not in title


def test_returning_row_sits_at_bottom_of_connect_page1():
    # The top-of-card account block must not come back.
    assert "home-account-top" not in APP_PY
    assert "home-account-entry" not in APP_PY
    page1 = APP_PY[APP_PY.index('id="connectPage1"'):APP_PY.index('id="connectPage2"')]
    # The row is the last thing on Page 1: after the Find button and
    # after the trust line, under its hairline.
    assert page1.index('id="smartFindBtn"') < page1.index("smart-returning")
    assert page1.index("smart-trust") < page1.index("smart-returning")
    row = page1[page1.index("smart-returning"):]
    assert "Returning? Sign in with Google to load your saved leagues." in row
    assert 'class="google-continue-btn" href="/auth/google?intent=login' in row
    # Exactly one account Google button in the home source: the row's.
    assert APP_PY.count(
        'class="google-continue-btn" href="/auth/google?intent=login') == 1


def test_returning_row_composition_rules_in_both_sheets():
    for css in (LITE, DASH):
        start = css.index(".smart-returning {")
        block = css[start:css.index("}", start)]
        assert "display: flex" in block
        assert "justify-content: space-between" in block
        assert "border-top: 1px solid rgba(255,255,255,.12)" in block
        # The pill is auto width beside the text, not a full-width button.
        assert ".smart-returning .google-continue-btn { flex: 0 1 auto; width: auto; }" in css


def test_connect_card_is_the_dark_home_card_in_every_theme():
    """The reimagine mock's connect card is the site's dark .home-card
    shell with the title inside it, in light mode too. The ld landing's
    light re-skin (white card surface, re-skinned contents, grid as the
    surface) is gone: .home-card carries the dark surface in every
    theme and Page 2's grid sits directly on it with no surface."""
    for css in (LITE, DASH):
        assert "background: #0d1626;" in css
        assert ('.home-page .home-card { padding: 18px 20px 16px;'
                ' border-radius: var(--radius); }') in css
        assert ('.home-card .ld-connect-grid { background: transparent;'
                ' border: none;') in css
        assert ('html:not([data-theme="dark"]) .home-card { background: var(--card);'
                ' color: #122d4b;') not in css
        assert ('html:not([data-theme="dark"]) .ld-connect-grid { background: transparent;'
                ' border: none; border-top: 1px solid var(--border);') not in css
        assert ('html:not([data-theme="dark"]) .home-account-top { margin-bottom: 0;'
                ' padding: 20px 22px 16px; border-bottom: none; }') not in css
        assert 'html:not([data-theme="dark"]) .home-card { background: transparent' not in css


def test_orphan_trust_row_rule_is_gone():
    for css in (LITE, DASH):
        assert ".home-account-top .home-hero-trust { flex-basis: 100%" not in css
        assert not re.search(
            r"\.home-account-top \{[^}]*flex-wrap: wrap;[^}]*justify-content: center;", css)
