"""Trade Calculator sticky bar dock clearance + Fit grading state.

Two live bugs from a phone screenshot, pinned at the source level (the
behavioral contracts live in tests/trade_fit_harness.mjs):

- The sticky balance bar floated at a flat bottom offset on mobile and
  the bottom tab bar dock covered its lower half on home-indicator
  phones. Bottom-fixed UI on this site anchors to
  --dock-safe-bottom (the space the dock owns: 60px bar + float gap +
  safe-area) plus a gap; the sticky bar now does the same.
- While a package re-graded, the Fit tile, acceptance chip, and sticky
  bar kept painting the previous trade's grade. The eval payload is
  signature-stamped on landing and the surfaces enter a grading state
  (_fitPending) for the in-flight selection.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _sticky_bar_mobile_rule() -> str:
    """The .otc-sticky-bar rule inside the max-width:768px block."""
    idx = CSS.index(".otc-sticky-bar { left: 12px;")
    return CSS[idx:CSS.index("}", idx)]


def test_sticky_bar_clears_the_mobile_dock():
    rule = _sticky_bar_mobile_rule()
    assert "calc(var(--dock-safe-bottom)" in rule, (
        "the sticky bar must anchor to the dock-safe line, not a flat "
        "offset the dock can cover")
    assert not re.search(r"bottom:\s*88px", rule), (
        "the old flat 88px offset hid the bar behind the dock on "
        "home-indicator phones")


def test_dock_safe_variable_is_the_site_pattern():
    # The variable the bar anchors to is the shared one: 0 on desktop,
    # dock height + float gap + safe-area on mobile.
    assert "--dock-safe-bottom: 0px;" in CSS
    assert "--dock-safe-bottom: calc(60px + env(safe-area-inset-bottom) + 12px);" in CSS


def test_fit_grading_state_wiring():
    # Entering a recompute starts grading; a landing stamps the payload
    # with the eval signature and ends it; a terminal failure resolves
    # it (in the catch, beside the bounded retry) so nothing strands.
    assert "state._fitPending = true;" in APP_JS
    assert "state._lastEvalSig = _evalSig;" in APP_JS
    assert "state._lastEvalSig = _currentEvalSig();" in APP_JS
    assert APP_JS.count("state._fitPending = false;") >= 3, (
        "landing, empty-trade reset, and terminal failure each resolve "
        "the grading state")
    assert "otc-fit--grading" in APP_JS
    assert "Grading…" in APP_JS, "the sticky verdict slot placeholders while grading"


def test_fit_components_are_signature_guarded():
    body = APP_JS.split("function _fitComponents()", 1)[1].split("function _clearPiFit", 1)[0]
    assert "state._lastEvalSig !== _currentEvalSig()" in body, (
        "a payload stamped for a previous package must yield no components")
