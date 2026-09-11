"""Test that _advFetch is available in all JS bundles.

The _advFetch helper is used by both app.js (trade calculator) and player_modal.js
(advanced metrics). It must be defined in app.js with @public-js:include markers
so it's available in all bundles: app.min.js, public.min.js, and app-features.min.js.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
PLAYER_MODAL_JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")


def test_advfetch_defined_in_app_js_before_core_end():
    """_advFetch must be defined in app.js before @public-js:core-end marker."""
    # Check that _advFetch is defined in app.js
    assert "function _advFetch(url, ms, init)" in APP_JS, \
        "_advFetch must be defined in app.js"
    
    # Verify the function is defined BEFORE the @public-js:core-end marker
    # so it's included in all bundles (app.min.js, public.min.js, app-features.min.js)
    core_end = APP_JS.find("// @public-js:core-end")
    advfetch_def = APP_JS.find("function _advFetch(url, ms, init)")
    
    assert core_end > 0, "app.js must have @public-js:core-end marker"
    assert advfetch_def > 0, "_advFetch must be defined in app.js"
    assert advfetch_def < core_end, \
        "_advFetch definition must come BEFORE @public-js:core-end marker"


def test_advfetch_not_duplicated_in_player_modal():
    """_advFetch must NOT be defined in player_modal.js to avoid duplication."""
    # The function should be called but not defined in player_modal.js
    assert "_advFetch(" in PLAYER_MODAL_JS, \
        "player_modal.js should still call _advFetch"
    assert "function _advFetch(url, ms, init)" not in PLAYER_MODAL_JS, \
        "_advFetch must not be defined in player_modal.js (moved to app.js)"


def test_advfetch_used_by_trade_calculator():
    """Trade calculator in app.js uses _advFetch for player deltas and indicators."""
    # These are the calls that were failing with ReferenceError
    assert "await _advFetch(`/api/player-deltas?" in APP_JS, \
        "loadPlayerDeltas must use _advFetch"
    assert "await _advFetch(`/api/player-indicators?" in APP_JS, \
        "loadPlayerIndicators must use _advFetch"


def test_advfetch_defined_before_usage():
    """_advFetch must be defined before it's used in app.js."""
    advfetch_def = APP_JS.find("function _advFetch(url, ms, init)")
    
    # Find all usages of _advFetch (excluding the definition line)
    usages = []
    search_start = 0
    while True:
        idx = APP_JS.find("_advFetch(", search_start)
        if idx == -1:
            break
        # Skip the definition line
        if idx != advfetch_def + len("function "):
            usages.append(idx)
        search_start = idx + 1
    
    assert len(usages) > 0, "app.js should have at least one usage of _advFetch"
    
    # All usages should come after the definition
    for usage_idx in usages:
        assert advfetch_def < usage_idx, \
            f"_advFetch usage at position {usage_idx} comes before definition at {advfetch_def}"
