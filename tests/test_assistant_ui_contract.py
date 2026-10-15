"""Contract guards for the BR Assistant popup UI (Stage 1 frontend).

The popup is built dynamically in static/app.js (initAssistant, next to
initPills) and styled by a delimited section in static/dashboard.css.
These tests lock the mock geometry, the API contract paths, and the
copy rules (no em dashes, no 999px radius) without needing a browser.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

_CSS_START = CSS.index("BR ASSISTANT (Stage 1 popup chat)")
_CSS_END = CSS.index("/* == BR ASSISTANT END == */")
ASSISTANT_CSS = CSS[_CSS_START:_CSS_END]

_JS_START = JS.index("function initAssistant()")
_JS_END = JS.index("function init()", _JS_START)
ASSISTANT_JS = JS[_JS_START:_JS_END]


def _rule(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
    assert match, f"missing CSS rule for {selector}"
    return match.group(1)


def test_assistant_css_selectors_exist():
    for selector in (
        ".br-asst-launcher",
        ".br-asst-dot",
        ".br-asst-teaser",
        ".br-asst-popup",
        ".br-asst-scrim",
        ".br-asst-grab",
        ".br-asst-head",
        ".br-asst-body",
        ".br-asst-composer",
        ".br-asst-msg-user",
        ".br-asst-msg",
        ".br-asst-chip",
        ".br-asst-sys",
        ".br-asst-fb-btn",
        ".br-asst-memory",
        ".br-asst-pref-row",
        ".br-asst-starters",
        ".br-asst-starters-label",
    ):
        assert selector in ASSISTANT_CSS, f"missing assistant CSS for {selector}"


def test_desktop_geometry_matches_mock():
    launcher = _rule(ASSISTANT_CSS, ".br-asst-launcher")
    assert "position: fixed" in launcher
    assert "width: 56px" in launcher
    assert "height: 56px" in launcher
    assert "bottom: 66px" in launcher  # stacked above #floating-pill-group
    assert "right: 16px" in launcher

    popup = _rule(ASSISTANT_CSS, ".br-asst-popup")
    assert "width: 392px" in popup
    assert "height: 600px" in popup
    assert "bottom: 66px" in popup
    assert "right: 16px" in popup

    teaser = _rule(ASSISTANT_CSS, ".br-asst-teaser")
    assert "width: 326px" in teaser
    assert "bottom: 136px" in teaser


def test_mobile_sheet_uses_dock_safe_token():
    # Everything mobile anchors to --dock-safe-bottom instead of raw pixels.
    assert ASSISTANT_CSS.count("var(--dock-safe-bottom)") >= 4
    media = ASSISTANT_CSS[ASSISTANT_CSS.index("@media (max-width: 768px)"):]
    # Launcher: dock-safe + 14px gap, which resolves to 86px at the
    # default 72px dock-safe value.
    launcher = _rule(media, ".br-asst-launcher")
    assert "bottom: calc(var(--dock-safe-bottom) + 14px)" in launcher
    # Sheet: dock-safe + 8px, which resolves to 80px at the default value.
    popup = _rule(media, ".br-asst-popup")
    assert "bottom: calc(var(--dock-safe-bottom) + 8px)" in popup
    assert "height: 452px" in popup
    assert "left: 8px" in popup
    assert "right: 8px" in popup
    # Scrim stops at the dock-safe line so the dock is never dimmed.
    scrim = _rule(media, ".br-asst-scrim")
    assert "bottom: var(--dock-safe-bottom)" in scrim


def test_assistant_js_endpoints_and_wiring():
    for path in (
        "/api/assistant/context",
        "/api/assistant/chat",
        "/api/assistant/feedback",
        "/api/assistant/correction",
        "/api/assistant/preferences",
    ):
        assert path in ASSISTANT_JS, f"missing endpoint {path}"
    # initAssistant is defined before the public bundle marker, on the
    # same side as initPills, and wired from the same init().
    assert JS.index("function initAssistant()") < JS.index("// @public-js:core-end")
    init_body = JS[JS.index("function init()"):]
    assert "initAssistant();" in init_body[:400]
    assert "initPills();" in init_body[:400]


def test_assistant_js_renders_contract_pieces():
    for needle in (
        "trade-ai-verdict-accept",
        "trade-ai-verdict-decline",
        "ns-item",
        "ns-priority gold",
        "bo-receipt",
        "bo-rc-row",
        "See the plan",
        "What BR remembers about you",
        "What was wrong?",
        "Wrong player data",
        "Bad reasoning",
        "Ignored my situation",
        "Tap again to confirm",
        "Forget everything BR remembers",
        "sessionStorage",
        "player_id",
        "BR_Mark_dark.png",
        # Starter chips (empty-thread state): the three core asks.
        "Try asking",
        "What should I do with my team this week?",
        "Who should I start at FLEX this week?",
        "Which player is a buy-low right now?",
        "hideStarters",
        # A wrong-data correction can return a revised answer to append.
        "data.revision",
        "data.revision.answer",
    ):
        assert needle in ASSISTANT_JS, f"missing assistant JS piece {needle}"
    # Cap (429) and unavailable (503) states are handled explicitly.
    assert "res.status === 429" in ASSISTANT_JS
    assert "res.status === 503" in ASSISTANT_JS


def test_answer_contract_fields_consumed():
    # The backend composer contract (tests/test_assistant_contract.py on the
    # backend branch pins the producing side): act_link is an OBJECT with
    # href/label, and this-week actions carry href + cta_label + key_number.
    for needle in (
        "answer.act_link.href",
        "answer.act_link.label",
        "action.href",
        "action.cta_label",
        "action.key_number",
    ):
        assert needle in ASSISTANT_JS, f"UI does not consume contract field {needle}"


def test_no_banned_values_in_new_sections():
    assert "999px" not in ASSISTANT_CSS
    assert "999px" not in ASSISTANT_JS
    assert "\u2014" not in ASSISTANT_CSS  # em dash
    assert "\u2014" not in ASSISTANT_JS  # em dash


def test_server_text_is_not_injected_as_html():
    # innerHTML is allowed only for static skeleton markup and icons.
    # Answer rendering must go through textContent (the el() helper).
    assert "answer.text" in ASSISTANT_JS
    assert ".innerHTML = SVG_" in ASSISTANT_JS or "innerHTML = SVG_" in ASSISTANT_JS
    # No dynamic server field is concatenated into an innerHTML string.
    for bad in ("innerHTML = answer", "innerHTML += answer", "innerHTML = data."):
        assert bad not in ASSISTANT_JS
