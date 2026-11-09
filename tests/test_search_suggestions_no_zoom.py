"""Focused tests for the mobile search empty-state suggestions and the
site-wide iOS no-zoom rule.

Suggestions: with an empty query, the mobile search page / dropdown now
shows Recent searches (localStorage), Top players (by proj_ppg from the
slim search payload) and a Jump to shortcut list, instead of a blank
void. Desktop keeps the old behavior (empty query shows nothing).

No-zoom: iOS Safari zooms into any focused field under 16px, so all
three stylesheets carry one blanket 16px rule for form fields inside
the existing max-width: 768px breakpoint.

These are file-text / payload-function checks; they do not import app.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
LANDING_CSS = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")
SEO_CSS = (ROOT / "static" / "seo_lite.css").read_text(encoding="utf-8")

BLANKET = "input, select, textarea { font-size: 16px !important; }"
BLANKET_BLOCK = re.compile(
    r"@media \(max-width: 768px\) \{\s*"
    r"input, select, textarea \{ font-size: 16px !important; \}\s*\}"
)


def _rule_inside_768_block(css: str) -> bool:
    """True when the blanket rule is the body of a max-width:768px block."""
    return bool(BLANKET_BLOCK.search(css))


def test_blanket_16px_rule_in_dashboard_css_scoped_to_768():
    assert BLANKET in DASH_CSS
    assert _rule_inside_768_block(DASH_CSS)


def test_blanket_16px_rule_in_landing_lite_css_scoped_to_768():
    assert BLANKET in LANDING_CSS
    assert _rule_inside_768_block(LANDING_CSS)


def test_blanket_16px_rule_in_seo_lite_css_scoped_to_768():
    assert BLANKET in SEO_CSS
    assert _rule_inside_768_block(SEO_CSS)


def test_blanket_rule_not_applied_at_desktop_scope():
    # Exactly one blanket rule per sheet, and it lives in the 768 block
    # (checked above); a second copy would mean an unscoped/global rule.
    for css in (DASH_CSS, LANDING_CSS, SEO_CSS):
        assert css.count(BLANKET) == 1


def test_app_js_recents_storage_key_and_recorder():
    assert "'br_recent_searches'" in APP_JS
    assert "function recordRecent(" in APP_JS
    # Recorded on BOTH selection paths: the dropdown click handler and
    # selectCurrent() (Enter / keyboard).
    assert APP_JS.count("recordRecentById(") >= 3  # definition + 2 call sites
    # A way to clear the stored recents exists and is wired to a control.
    assert "data-ns-clear-recents" in APP_JS
    assert "function clearRecents(" in APP_JS


def test_app_js_render_suggestions_wired_to_empty_query():
    assert "function renderSuggestions()" in APP_JS
    # renderResults routes the empty query to suggestions instead of
    # closing the dropdown outright.
    assert "if (!q) { renderSuggestions(); return; }" in APP_JS
    # Mobile gate: desktop top-bar focus with an empty query shows nothing.
    assert "window.matchMedia('(max-width: 768px)')" in APP_JS
    assert "brSearchScreen" in APP_JS


def test_app_js_suggestion_sections_and_jump_to_subset():
    assert "Recent" in APP_JS and "Top players" in APP_JS and "Jump to" in APP_JS
    for label in ("ScoreZone", "Matchups", "Start/Sit",
                  "Trade Calculator", "Waivers", "Player Rankings"):
        assert label in APP_JS
    # Top players come from the loaded player list ordered by proj_ppg.
    assert "projPpg" in APP_JS
    assert "proj_ppg" in APP_JS  # mapped in loadPlayers()


def test_slim_search_payload_carries_proj_ppg():
    from dashboard_services.league_players_board import (
        SEARCH_PLAYER_KEYS,
        slim_search_payload,
    )

    assert "proj_ppg" in SEARCH_PLAYER_KEYS
    payload = {
        "players": [
            {
                "id": "1001", "name": "Top Guy", "position": "WR",
                "team": "DET", "espnHeadshot": "http://x/y.png",
                "value": 5000, "proj_ppg": 24.6,
            },
            {
                "id": "1002", "name": "No Proj", "position": "RB",
                "team": "MIA", "espnHeadshot": "", "value": 3000,
            },
        ]
    }
    out = slim_search_payload(payload)
    rows = {r["id"]: r for r in out["players"]}
    assert rows["1001"]["proj_ppg"] == 24.6
    # A row without a projection simply omits the key (never invented).
    assert "proj_ppg" not in rows["1002"]
