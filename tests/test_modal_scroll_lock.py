"""Background scroll lock for every overlay (Kaedon, 2026-11-26).

When any modal, popover, dropdown or sheet is open, scrolling must
move only that overlay, never the page behind it. The lock is central:
one br-scroll-lock module in app.js core watches the whole overlay
family with a MutationObserver and pins the body (position:fixed at
the saved scroll offset, the iOS-proven technique, since body
overflow:hidden alone does not stop touch scrolling in iOS Safari),
plus a touchmove guard so a drag only proceeds when a scrollable
element inside the overlay can still consume it. dashboard.css stops
scroll chaining at each overlay's boundary with overscroll-behavior.

These guards pin the wiring so a future overlay refactor cannot
silently drop a family member from the lock, and so the lock module
stays in app.js CORE (before the @public-js:core-end marker): the
More sheet, changelog and assistant live in core and on lite pages
that never load the feature bundle.
"""
from pathlib import Path

ROOT = Path(__file__).parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text()
APP_JS = (ROOT / "static" / "app.js").read_text()
TI_PAGE = (ROOT / "dashboard_services" / "pages" / "trade_intel_page.py").read_text()

LOCK_START = APP_JS.index("/* br-scroll-lock:start */")
LOCK_END = APP_JS.index("/* br-scroll-lock:end */")
LOCK_JS = APP_JS[LOCK_START:LOCK_END]
CORE_END = APP_JS.index("// @public-js:core-end")


def _block(css, selector, start=0):
    """Return the declaration block body for the first `selector { ... }`."""
    i = css.index(selector, start)
    j = css.index("{", i)
    k = css.index("}", j)
    return css[j + 1:k]


def test_lock_module_lives_in_app_js_core():
    # The More sheet / changelog / assistant are core UI; a lock that
    # only ships in the lazy feature bundle would miss them.
    assert LOCK_START < CORE_END


def test_lock_module_is_refcounted_and_restores_scroll_position():
    assert "window.brScrollLock" in LOCK_JS
    assert "acquire" in LOCK_JS and "release" in LOCK_JS
    assert "manualCount" in LOCK_JS
    # iOS technique: pin the body at the saved offset, restore exactly.
    # Renders where body itself is the scroller skip the pin (it would
    # collapse the body's scroll box) and freeze via the CSS class.
    assert "position = 'fixed'" in LOCK_JS
    assert "bodyIsScroller()" in LOCK_JS
    assert "pageScrollY()" in LOCK_JS
    assert "window.scrollTo(0, y)" in LOCK_JS
    assert "'br-scroll-locked'" in LOCK_JS


def test_lock_module_watches_the_whole_overlay_family():
    for selector in (
        ".player-modal-overlay",   # player modal + breakout modal
        ".team-modal-overlay",     # team modal / drawer
        ".for-modal-overlay",      # Front Office modal
        ".rzm-modal-overlay",      # ScoreZone / redzone modal
        ".modal-overlay",          # generic page modals
        ".signin-modal-overlay",   # sign-in / claim modal
        ".scm-overlay",            # share-card modals
        ".sub-welcome-overlay",    # subscription welcome dialog
        "#tiTradesOverlay",        # Trade Intel trades modal
        "#rkModal",                # rookie / prospect modal
        "#brMoreSheet.open",       # mobile More bottom sheet (.br-sheet)
        "#brSearchScreen.open",    # full-screen mobile search
        ".rz-bs-sheet",            # ScoreZone box-score sheet
        ".nt-sheet",               # NFL Teams box-score sheet
        ".br-asst-popup",          # assistant panel
        ".changelog-dropdown",     # scrollable popover
        ".settings-dropdown",      # popover
        '[data-br-modal-open="true"]',
    ):
        assert selector in LOCK_JS, selector
    # DOM state is the trigger (every close path shares it), observed
    # for the class/style/hidden flips the family uses to show + hide.
    assert "MutationObserver" in LOCK_JS
    assert "attributeFilter: ['class', 'style', 'hidden']" in LOCK_JS


def test_touch_guard_only_lets_overlay_scrollers_consume_drags():
    assert "addEventListener('touchmove'" in LOCK_JS
    assert "{ passive: false }" in LOCK_JS
    assert "preventDefault()" in LOCK_JS
    # The guard honours whichever axis the gesture moves on, so
    # intentional inner horizontal scrollers keep working (PR #3113).
    assert "overflowX" in LOCK_JS and "overflowY" in LOCK_JS
    # Pinch and form controls are not page scroll; never blocked.
    assert "e.touches.length !== 1" in LOCK_JS
    assert "'INPUT'" in LOCK_JS and "'CANVAS'" in LOCK_JS


def _rule(css, selector):
    """Selector prelude + declaration block for the first matching rule."""
    i = css.index(selector)
    k = css.index("}", i)
    return css[i:k]


def test_css_stops_scroll_chaining_at_overlay_boundaries():
    # The containment section must live BEFORE the first MERGED marker:
    # scripts/merge_icon_css.py rebuilds everything from that marker
    # onward, so a section inside the merged region is deleted by the
    # re-merge and test_merge_script_is_idempotent fails (PR #3285 CI).
    assert CSS.index("SCROLL CONTAINMENT") < CSS.index("MERGED: icons.css")
    # Scope to the containment section: these selector names also
    # appear in earlier rules elsewhere in the sheet.
    section = CSS[CSS.index("SCROLL CONTAINMENT"):]
    bodies = _rule(section, ".player-modal-body,")
    assert "overscroll-behavior: contain" in bodies
    for selector in (
        ".team-modal-body",
        ".for-modal-body",
        ".rzm-modal-body",
        ".td-body",
        ".br-asst-body",
        ".br-sheet-panel",
        ".changelog-dropdown",
        ".rz-bs-body",
        ".nt-sheet-body",
        "#rkModalContent",
    ):
        assert selector in bodies, selector
    shells = _rule(section, ".player-modal-overlay,")
    assert "overscroll-behavior: none" in shells
    for selector in (".team-modal-overlay", ".for-modal-overlay",
                     ".rzm-modal-overlay", ".modal-overlay",
                     "#tiTradesOverlay", ".br-sheet-scrim",
                     ".rz-bs-backdrop", ".nt-sheet-backdrop",
                     ".br-asst-scrim"):
        assert selector in shells, selector
    locked = _block(CSS, "html.br-scroll-locked body {")
    assert "overflow: hidden" in locked
    assert "overscroll-behavior: none" in locked


def test_trade_intel_trades_body_contains_overscroll():
    body = _block(TI_PAGE, ".ti-trades-body {{")
    assert "overscroll-behavior: contain" in body
