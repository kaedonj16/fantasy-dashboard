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
- Redesign: the bar's Fit slot and verdict pill are gone. The pill
  squeezed the summary's full verdict sentence into a truncated
  ellipsis, which read as pointless. The bar is now a balance strip
  only: the two side values and a thin balance band between them,
  positioned by _balanceGeometry, the same computation the summary's
  full balance bar uses (extracted from the eval-landing handler).
  The whole bar is a button that scrolls to the Trade Summary card.
- Visibility: the bar hides while the Trade Summary card itself is
  on screen (an IntersectionObserver on the card owns
  state._summaryInView; a 24px rootMargin dead band keeps the
  boundary from flickering). It only earns its place when the
  summary is scrolled out of view.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
PAGE = (ROOT / "dashboard_services" / "pages" / "trade_calculator_page.py").read_text(encoding="utf-8")


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
    # The grading state lives in the Fit TILE; the sticky bar has no
    # slots left to placeholder.
    assert "state._fitPending = true;" in APP_JS
    assert "state._lastEvalSig = _evalSig;" in APP_JS
    assert "state._lastEvalSig = _currentEvalSig();" in APP_JS
    assert APP_JS.count("state._fitPending = false;") >= 3, (
        "landing, empty-trade reset, and terminal failure each resolve "
        "the grading state")
    assert "otc-fit--grading" in APP_JS
    bar_body = APP_JS.split("function renderStickyBar()", 1)[1].split(
        "function scrollStickyBarToSummary", 1)[0]
    assert "Grading" not in bar_body, (
        "the bar's grading placeholders went away with its Fit/verdict slots")
    assert "_fitPending" not in bar_body, (
        "the bar renders the landed payload only; grading is the tile's state")


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
        "the symmetric 12px right inset let the bar's contents slide "
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


def test_sticky_bar_is_values_and_band_only():
    # The bar's anatomy is exactly: value, mini balance band, value.
    # The Fit slot and the verdict pill (and their elements) are gone
    # from the template, the renderer, and the stylesheet.
    for el_id in ("tradeStickyValueA", "tradeStickyValueB",
                  "tradeStickyBandFair", "tradeStickyBandMarker"):
        assert f'id="{el_id}"' in PAGE, f"bar element {el_id} missing"
    for gone in ("tradeStickyTotals", "tradeStickyVerdict"):
        assert gone not in PAGE, f"{gone} must not come back to the template"
        assert gone not in APP_JS, f"{gone} must not come back to the renderer"
    assert ".otc-sticky-band {" in CSS
    assert ".otc-sticky-band-fair" in CSS
    assert ".otc-sticky-band-marker" in CSS
    assert ".otc-sticky-totals" not in CSS
    assert ".otc-sticky-verdict" not in CSS
    bar_body = APP_JS.split("function renderStickyBar()", 1)[1].split(
        "function scrollStickyBarToSummary", 1)[0]
    assert "computeTradeFitScore" not in bar_body, (
        "no Fit score in the bar; the grade lives in the summary tile")
    assert "data.verdict" not in bar_body, (
        "no verdict sentence in the bar; it lives in the summary card")
    assert "acceptance_label" not in bar_body, (
        "no acceptance read in the bar; it lives in the summary card")


def test_sticky_bar_band_shares_summary_geometry():
    # One geometry computation, two call sites: the eval-landing
    # handler (summary's full bar) and renderStickyBar (mini band).
    # A second formula in the bar would let the two bars disagree.
    assert "function _balanceGeometry(diff, aEff, bEff, fairThreshold, fairPct)" in APP_JS
    assert "_balanceGeometry(diff, aEff, bEff" in APP_JS, (
        "the eval-landing handler must use the shared geometry")
    bar_body = APP_JS.split("function renderStickyBar()", 1)[1].split(
        "function scrollStickyBarToSummary", 1)[0]
    assert "_balanceGeometry(" in bar_body, (
        "the mini band must use the shared geometry")
    assert "normalizedDiff" not in bar_body, (
        "no private copy of the position math in the bar renderer")


def test_sticky_bar_scrolls_to_summary():
    # Button semantics in the markup, bindOnce wiring beside the other
    # trade buttons, and the scroll itself targets the summary card
    # with an explicit top-nav offset (scroll-margin-top is zeroed by
    # the global <=1280px reset, so it cannot carry the offset).
    bar_tag = PAGE.split('id="tradeStickyBar"', 1)[1].split(">", 1)[0]
    assert 'role="button"' in bar_tag
    assert 'tabindex="0"' in bar_tag
    assert 'aria-label="Go to trade summary"' in bar_tag
    assert "function bindStickyBar()" in APP_JS
    assert "bindStickyBar();" in APP_JS
    assert '"stickyBarClick"' in APP_JS and '"stickyBarKeys"' in APP_JS
    scroll_body = APP_JS.split("function scrollStickyBarToSummary()", 1)[1].split(
        "function renderUnvaluedStrip", 1)[0]
    assert '".otc-summary-card"' in scroll_body, "scroll target is the summary card"
    assert '".top-nav"' in scroll_body, "offset measured from the sticky top nav"
    assert 'behavior: "smooth"' in scroll_body


def test_sticky_bar_hides_while_summary_visible():
    # An IntersectionObserver on the same .otc-summary-card the tap
    # scrolls to owns state._summaryInView; renderStickyBar composes
    # it with the armed gate. The 24px rootMargin dead band is the
    # hysteresis: hide just before the card enters, return just after
    # it has fully left, so the boundary cannot flicker.
    bind_body = APP_JS.split("function bindStickyBar()", 1)[1].split(
        "function bindClearTradeButton", 1)[0]
    assert "new IntersectionObserver" in bind_body
    assert '".otc-summary-card"' in bind_body, "observer watches the summary card"
    assert "io.observe(summaryCard)" in bind_body
    assert "state._summaryInView = !!en.isIntersecting;" in bind_body
    assert 'rootMargin: "24px 0px 24px 0px"' in bind_body, (
        "the rootMargin dead band is the anti-flicker hysteresis")
    bar_body = APP_JS.split("function renderStickyBar()", 1)[1].split(
        "function scrollStickyBarToSummary", 1)[0]
    assert "state._summaryInView" in bar_body, (
        "the renderer hides the bar while the summary is on screen")
