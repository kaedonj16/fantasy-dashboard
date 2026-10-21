"""Contracts for the Emil Kowalski motion pack (2026-10-20 audit fixes).

Source-level guards, same style as test_css_health: they lock the
invariants the audit established so a later edit can't quietly regress
them - one global reduced-motion block that keeps fades, no
`transition: all`, a session-gated page entrance, an asymmetric
assistant motion budget, instant keyboard closes, and the coarse
pointer input/hover rules.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(rel):
    with open(os.path.join(_ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def test_single_global_reduced_motion_block_keeps_fades():
    css = _read("static/dashboard.css")
    # The old kill-switch zeroed transition durations globally, twice.
    assert "transition-duration: 0.01ms" not in css
    # Exactly one global block, and it restricts transitions to
    # non-movement properties instead of zeroing them (fades stay).
    assert css.count("transition-property: opacity, color, background-color") == 1
    assert css.count("@media (prefers-reduced-motion: reduce) {") >= 1


def test_no_transition_all_anywhere():
    targets = [
        "static/dashboard.css",
        "static/landing_lite.css",
        "static/seo_lite.css",
        "static/paywall.css",
        "dashboard_services/pages/advanced_metrics_page.py",
        "dashboard_services/pages/draft_room_page.py",
        "dashboard_services/pages/players_page.py",
        "dashboard_services/pages/trade_calculator_page.py",
        "dashboard_services/pages/trade_intel_page.py",
        "routes/trade_bp.py",
    ]
    for rel in targets:
        text = _read(rel)
        assert not re.search(r"transition:\s*all\b", text), rel


def test_page_entrance_stagger_is_session_gated():
    js = _read("static/app.js")
    assert "brPageEntered" in js
    gate = js.index("brPageEntered")
    add = js.index("main.classList.add('br-stagger')")
    # The gate is consulted before the class is added.
    assert gate < add


def test_assistant_motion_budget_asymmetric():
    css = _read("static/dashboard.css")
    assert "br-asst-in .28s" in css
    assert "br-asst-out .18s" in css
    assert "br-asst-in .38s" not in css
    assert "br-asst-out .38s" not in css
    js = _read("static/app.js")
    # Escape minimizes instantly (no exit animation), with the
    # minimize semantics preserved.
    assert "minimizePopupInstant" in js
    # Count-ups on data values run on the short budget.
    assert "opts.dur || 400" in js


def test_custom_select_escape_closes_instantly():
    for rel in ("static/app.js", "static/custom_selects.js"):
        js = _read(rel)
        assert "function close(instant)" in js, rel
        assert "close(true)" in js, rel


def test_mobile_pack_rules_present():
    css = _read("static/dashboard.css")
    assert "touch-action: manipulation" in css
    assert "@media (hover: none), (pointer: coarse)" in css
    assert "@media (pointer: coarse)" in css
    # The assistant composer is one of the explicitly pinned 16px fields.
    coarse = css[css.index("@media (pointer: coarse)"):]
    assert ".br-asst-input" in coarse


def test_tooltip_first_delay_with_skip_delay():
    js = _read("static/app.js")
    assert "showTimer" in js
    assert "lastHideAt" in js
    assert "}, 350);" in js
