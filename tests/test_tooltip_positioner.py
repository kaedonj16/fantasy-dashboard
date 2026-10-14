"""Global [data-tooltip] positioning contract.

Bug (2026-10-14, screenshot): several tooltips across the site render
cut off. The reported instance: the Teams card "Positional Index"
definition bubble showed only its last lines ("starting-lineup strength
sits above or below the league average, in standard deviations."), the
top sliced flat.

Mechanism, proven in a Playwright harness with the real dashboard.css:
the app.js title upgrader turns title= into a [data-tooltip] bubble that
is a pure CSS ::after -- always centered on the trigger, always opening
upward (bottom:100% + 6px), with no flip, no viewport clamp, and no
awareness of clipping ancestors. The .tsc-pi trigger sits ~67px below
its card's top edge while the bubble needs ~118px above it; the card's
overflow:hidden slices the top 51px (desktop) / 21px (390px phone).
Same class as the earlier per-surface patches: AM command bar vs the
sticky nav (#2569) and #amMoreToggle vs the right edge (#2581). The
metric-definition tip (.adv-def-tip) never had this bug because it is
positioned in JS with clamping and flipping (_advPositionTip).

Fix: one fixed-position singleton bubble (.br-tip) rendered by a JS
positioner in app.js next to the title upgrader (before the
@public-js:core-end marker, so both bundles ship it). It centers on the
trigger, clamps inside the viewport, flips below when above does not
fit (clearing the sticky .top-nav), and escapes ancestor overflow as a
body-level fixed element. html.br-tips-js suppresses the CSS ::after
while it runs; the ::after remains the no-JS fallback. .pm-adp-dot
keeps its own ::after bubble and [data-def] elements keep the
metric-definition tip; both are excluded from the singleton.
"""
import os
import re

_APP_JS = os.path.join(os.path.dirname(__file__), "..", "static", "app.js")
_CSS = os.path.join(os.path.dirname(__file__), "..", "static", "dashboard.css")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _positioner_block():
    js = _read(_APP_JS)
    start = js.index("// [tooltip-positioner:start]")
    end = js.index("// [tooltip-positioner:end]")
    return js[start:end]


def _rule_body(css, selector_pattern):
    m = re.search(selector_pattern + r"\s*\{([^}]*)\}", css, re.S)
    assert m, f"rule not found: {selector_pattern}"
    return m.group(1)


def test_positioner_ships_in_both_bundles():
    js = _read(_APP_JS)
    core_end = js.index("// @public-js:core-end")
    upgrade_end = js.index("// [title-upgrade:end]")
    start = js.index("// [tooltip-positioner:start]")
    assert upgrade_end < start < core_end, (
        "positioner must sit between the title upgrader and the "
        "@public-js:core-end marker so the public bundle ships it too"
    )


def test_positioner_clamps_and_flips():
    block = _positioner_block()
    # Horizontal clamp inside the viewport.
    assert "Math.max(8, Math.min(left, vw - tw - 8))" in block
    # Flip: below the trigger when above does not clear the sticky nav.
    assert "_navBottom()" in block
    assert "below + th <= vh - 8" in block
    # Neither side fits: roomier side, still clamped on screen.
    assert "rect.top >= vh - rect.bottom" in block


def test_positioner_exclusions():
    block = _positioner_block()
    assert "pm-adp-dot" in block, "ADP dots keep their own ::after bubble"
    assert "data-def" in block, "metric definitions keep the .adv-def-tip path"


def test_positioner_lifecycle():
    block = _positioner_block()
    for needle in ("mouseover", "mouseout", "focusin", "focusout",
                   "Escape", "scroll", "resize", "br-tips-js"):
        assert needle in block, f"positioner missing {needle} handling"
    # Touch: tap toggles non-interactive triggers, and the hover handlers
    # stand down on coarse pointers (post-tap synthetic mouse events must
    # not instantly hide a tap-shown bubble).
    assert "pointer: coarse" in block
    assert block.count("_coarsePointer()) return") >= 2


def test_singleton_css_uses_shared_tooltip_tokens():
    css = _read(_CSS)
    body = _rule_body(css, r"\.br-tip")
    assert "position: fixed" in body
    assert "z-index: var(--z-tooltip)" in body
    for token in ("--tooltip-bg", "--tooltip-fg", "--tooltip-border",
                  "--tooltip-pad", "--tooltip-radius", "--tooltip-fs",
                  "--tooltip-lh", "--tooltip-shadow"):
        assert f"var({token})" in body, f".br-tip must use var({token})"
    assert "max-width: min(240px, calc(100vw - 16px))" in body


def test_pseudo_bubble_suppressed_only_while_js_runs():
    css = _read(_CSS)
    assert re.search(
        r"html\.br-tips-js \[data-tooltip\]:not\(\.pm-adp-dot\):hover::after", css
    ), "hover ::after must be suppressed under html.br-tips-js"
    assert re.search(
        r"html\.br-tips-js \[data-tooltip\]:not\(\.pm-adp-dot\):focus-visible::after", css
    ), "focus ::after must be suppressed under html.br-tips-js"
    # The ID-specificity More actions override needs an ID-level match.
    assert re.search(
        r"html\.br-tips-js #amMoreToggle\[data-tooltip\]:hover::after", css
    ), "#amMoreToggle suppression must outrank its ID-keyed page rule"


def test_no_js_fallback_and_adp_dot_bubble_survive():
    css = _read(_CSS)
    # The base ::after fallback stays for no-JS rendering, and the
    # .pm-adp-dot ::after is the only other attr(data-tooltip) consumer.
    assert css.count("content: attr(data-tooltip)") == 2
    assert re.search(r"\.pm-adp-dot::after\s*\{", css)
