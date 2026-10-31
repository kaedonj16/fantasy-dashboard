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
- Follow-up: with the dock cleared, the bar's right end slid under the
  BR Assistant launcher, which anchors at the same mobile bottom with
  right: 16px and a 56px width at a higher z-index. The bar's mobile
  rule reserves that footprint plus a gap on its right; it does not
  outrank the launcher.
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


def test_sticky_bar_reserves_the_assistant_footprint():
    # Follow-up live bug: with the dock cleared, the bar's right end
    # slid under the BR Assistant launcher. The launcher anchors at the
    # same mobile bottom (dock-safe + 14px) with right: 16px and a 56px
    # width, so the bar's mobile rule must end 16 + 56 + a 12px gap
    # short of the right edge instead of the old symmetric 12px.
    rule = _sticky_bar_mobile_rule()
    assert "right: calc(16px + 56px + 12px)" in rule, (
        "the sticky bar must reserve the assistant launcher's "
        "footprint (16px offset + 56px width + 12px gap) on mobile")
    assert "right: 12px" not in rule, (
        "the symmetric 12px right inset let the verdict pill slide "
        "under the assistant launcher")


def test_assistant_footprint_derivation_matches_launcher_css():
    # The reservation numbers come from .br-asst-launcher's own rules:
    # 56px square, right: 16px, and the same dock-safe + 14px mobile
    # anchor the bar uses. If the launcher's geometry changes, this
    # pin fails so the bar's reservation gets re-derived.
    base = CSS.split(".br-asst-launcher {", 1)[1].split("}", 1)[0]
    assert "width: 56px;" in base and "right: 16px;" in base
    mobile = CSS.split("@media (max-width: 768px) {\n    .br-asst-launcher {", 1)[1].split("}", 1)[0]
    assert "right: 16px;" in mobile
    assert "bottom: calc(var(--dock-safe-bottom) + 14px);" in mobile


def test_sticky_bar_stays_below_the_assistant_z_index():
    # The fix is clearance, not stacking: the bar must never outrank
    # the launcher (9995) or the chrome above it.
    bar = CSS.split(".otc-sticky-bar {", 1)[1].split("}", 1)[0]
    assert "z-index: 9991;" in bar
    launcher = CSS.split(".br-asst-launcher {", 1)[1].split("}", 1)[0]
    assert "z-index: 9995;" in launcher


def test_sticky_bar_desktop_keeps_full_width():
    # The reservation is mobile-only: on desktop the launcher anchors
    # at bottom: 66px, clear of the bar's bottom: 0 band, so the base
    # rule keeps its full-width right: 0.
    base = CSS.split(".otc-sticky-bar {", 1)[1].split("}", 1)[0]
    assert "right: 0;" in base and "bottom: 0;" in base


def test_fit_components_are_signature_guarded():
    body = APP_JS.split("function _fitComponents()", 1)[1].split("function _clearPiFit", 1)[0]
    assert "state._lastEvalSig !== _currentEvalSig()" in body, (
        "a payload stamped for a previous package must yield no components")
