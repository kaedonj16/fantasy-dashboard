"""Trade Analyst verdict badge: renderer class -> CSS contract.

Bug (Kaedon screenshot, 2026-10-21): the "ACCEPT" pill on the BR Trade
Analyst card rendered ghosted - white text on no background. The renderer
carried the verdict color only in an inline style="background:..."
attribute, and sanitize_ai_html (the allowlist every AI HTML payload
passes through before innerHTML) strips style attributes, so the badge's
background never reached the page while .trade-ai-verdict-badge's
color: #fff did.

The fix carries the color on modifier classes
(.trade-ai-verdict-badge-accept / -decline / -counter and the matching
.trade-ai-conf-dot-* classes). These tests pin that contract end to end:
the class the renderer emits must survive the sanitizer and must have a
background rule in static/dashboard.css.
"""
import re
from pathlib import Path

import pytest

pytest.importorskip("flask")

# NOTE: dashboard_services imports stay inside the helpers below. The
# renderer chain imports the openai package at module level, which the
# pure-Python lint CI shard does not install; a module-level import here
# would fail collection there (pytest exit 2) even though this file is
# auto-marked integration.
ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")

EXPECTED = {
    "ACCEPT": ("accept", "#16a34a"),
    "DECLINE": ("decline", "#dc2626"),
    "COUNTER": ("counter", "#d97706"),
}


def _render(verdict: str) -> str:
    from dashboard_services.ai.renderer import render_trade_ai_html

    return render_trade_ai_html({
        "verdict": verdict,
        "summary": "A fair swap.",
        "helps": ["Adds depth."],
        "risks": ["Costs upside."],
        "counter": "",
        "confidence": "high",
    })


def _rule_body(selector: str) -> str:
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", CSS)
    assert m, f"{selector} has no rule in dashboard.css"
    return m.group(1)


@pytest.mark.parametrize("verdict", sorted(EXPECTED))
def test_badge_color_survives_sanitizer_via_class(verdict):
    mod, _hex = EXPECTED[verdict]
    from dashboard_services.ai.html_sanitize import sanitize_ai_html

    clean = sanitize_ai_html(_render(verdict))
    badge = re.search(r'<div class="([^"]*trade-ai-verdict-badge[^"]*)"', clean)
    assert badge, "verdict badge missing after sanitize"
    classes = badge.group(1).split()
    assert "trade-ai-verdict-badge" in classes
    assert f"trade-ai-verdict-badge-{mod}" in classes
    # The color must not ride on an inline style (the sanitizer strips it).
    badge_tag = re.search(r'<div class="[^"]*trade-ai-verdict-badge[^"]*"[^>]*>', clean)
    assert "style=" not in badge_tag.group(0)
    dot = re.search(r'<span class="([^"]*trade-ai-conf-dot[^"]*)"', clean)
    assert dot, "confidence dot missing after sanitize"
    assert f"trade-ai-conf-dot-{mod}" in dot.group(1).split()


def test_unknown_verdict_falls_back_to_counter_class():
    from dashboard_services.ai.html_sanitize import sanitize_ai_html

    clean = sanitize_ai_html(_render("MAYBE"))
    assert "trade-ai-verdict-badge-counter" in clean
    assert "trade-ai-conf-dot-counter" in clean


@pytest.mark.parametrize("verdict", sorted(EXPECTED))
def test_modifier_classes_have_background_rules(verdict):
    mod, hex_color = EXPECTED[verdict]
    badge_body = _rule_body(f".trade-ai-verdict-badge-{mod}")
    assert f"background: {hex_color}" in badge_body
    dot_body = _rule_body(f".trade-ai-conf-dot-{mod}")
    assert f"background: {hex_color}" in dot_body


def test_base_badge_rule_keeps_white_text():
    body = _rule_body(".trade-ai-verdict-badge")
    assert "color: #fff" in body
