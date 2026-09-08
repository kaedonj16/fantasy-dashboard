"""Regression guards for UI audit polish items U1–U6 (improvement plan)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _any_media_has(css: str, max_width: int, pattern: str) -> bool:
    parts = re.split(rf"@media\s*\(\s*max-width:\s*{max_width}px\s*\)\s*\{{", css)
    for chunk in parts[1:]:
        depth = 1
        body = []
        for ch in chunk:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth >= 1:
                body.append(ch)
        if re.search(pattern, "".join(body)):
            return True
    return False


def test_u1_jump_nav_scrolls_with_fade_hint():
    assert ".os-jump-nav" in CSS
    assert "overflow-x: auto" in CSS
    assert "-webkit-mask-image: linear-gradient" in CSS or "mask-image: linear-gradient" in CSS
    # Last tab stays reachable: tabs do not shrink to fit.
    assert re.search(r"\.os-jump-nav\s+button\s*\{[^}]*flex:\s*0\s+0\s+auto", CSS)


def test_u2_empty_activity_cards_reduce_min_height_on_mobile():
    assert "min-height: 120px" in CSS
    assert re.search(
        r"\.card\.activity-card\s+\.scroll-box,\s*\.card\.central\s+\.scroll-box\s*\{[^}]*min-height:\s*120px",
        CSS,
    )


def test_u3_body_font_at_least_14px_on_narrow():
    assert _any_media_has(CSS, 480, r"\bbody\s*\{[^}]*font-size:\s*14px")


def test_u4_footer_links_have_44px_tap_targets():
    assert re.search(
        r"\.site-footer-links\s+a\s*\{[^}]*min-height:\s*44px",
        CSS,
    )


def test_u5_trade_chips_and_waiver_rows_expose_full_name_title():
    assert "nameEl.title = p.name || \"Unknown\"" in APP_JS
    assert "nameEl.title = pk.display || pk.id || \"Pick\"" in APP_JS
    app_py = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'title="{html.escape(p[\'name\'], quote=True)}"' in app_py
    assert 'title="{html.escape(subline, quote=True)}"' in app_py
    # Trending strip already titles full name (waivers page).
    wv = (ROOT / "dashboard_services" / "pages" / "waivers_page.py").read_text(encoding="utf-8")
    assert 'title="${{tipName}} · ${{tipAdds}}"' in wv


def test_u6_bubble_badge_has_stronger_contrast():
    m = re.search(r"\.pp-t-bub\s*\{([^}]+)\}", CSS)
    assert m, "missing .pp-t-bub"
    body = m.group(1)
    assert "var(--warning)" in body
    assert "border:" in body
