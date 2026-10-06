"""Contract tests: floating mobile dock (Facebook-style pill) + pro polish.

- The dock floats with side margins, rounded corners and a soft shadow.
- Pro pass: 60px tall, 24px icons, 10px tracked labels, and ONE active signal
  (the gliding pill; the active icon/label stay a strong neutral).
- A mobile-only search shortcut sits top-right in the top bar and opens the
  existing full-screen search.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def test_dock_floats_with_side_margins():
    assert "left: max(12px, env(safe-area-inset-left))" in CSS
    assert "right: max(12px, env(safe-area-inset-right))" in CSS
    assert "bottom: calc(env(safe-area-inset-bottom) + 10px)" in CSS


def test_dock_is_a_rounded_pill_with_shadow():
    assert "border-radius: var(--radius)" in CSS
    assert "box-shadow: 0 8px 24px" in CSS


def test_dock_reserve_space_accounts_for_float_gap():
    assert "--dock-safe-bottom: calc(60px + env(safe-area-inset-bottom) + 12px)" in CSS


def test_dock_proportions():
    assert "height: 60px;" in CSS
    assert "font-size: 11px;" in CSS
    assert "letter-spacing: .03em;" in CSS
    # Dock icons render at 24px in both the league and guest docks.
    assert APP_PY.count("_nav_icon(icon, size=24)") == 2
    assert APP_PY.count("_nav_icon('more', size=24)") == 2


def test_active_tab_uses_pill_only():
    # The gliding pill is the single active signal; no accent recolor.
    assert ".br-tabbar-item.active {\n        /* The gliding pill is the active signal;" in CSS
    assert "color: var(--text);" in CSS


def test_top_search_button_hidden_on_desktop():
    assert ".br-top-search {\n    display: none;\n}" in CSS


def test_top_search_button_shown_on_mobile():
    assert ".top-nav.br-mnav .br-top-search," in CSS or ".top-nav.br-mnav .br-top-search {" in CSS
    assert ".top-nav.br-mnav .br-top-search {\n        right: 12px;\n    }" in CSS


def test_top_search_button_rendered_and_wired():
    assert "class='br-top-search'" in APP_PY
    assert "window.brOpenSearch&&window.brOpenSearch()" in APP_PY
    assert "window.brOpenSearch = openSearch;" in APP_JS


def test_top_notif_button_hidden_on_desktop():
    assert ".br-top-notif {\n    display: none;\n}" in CSS


def test_top_notif_button_shown_on_mobile():
    assert ".top-nav.br-mnav .br-top-notif {\n        right: 58px;\n    }" in CSS
    assert ".br-top-notif-dot" in CSS


def test_top_notif_button_rendered_and_wired():
    assert "class='br-top-notif'" in APP_PY
    assert "id='brTopNotifDot'" in APP_PY
    # Wired via inline onclick to window.brToggleChangelog, the same pattern
    # as the working mobile search button (no bind-timing dependency, no
    # More-sheet detour).
    assert "window.brToggleChangelog&&window.brToggleChangelog(event)" in APP_PY
    assert "window.brToggleChangelog = function" in APP_JS
    assert "window.brOpenNotifications" not in APP_JS
    # The red dot follows the shared unread state.
    assert 'document.getElementById("brTopNotifDot")' in APP_JS


def test_recent_updates_out_of_more_sheet():
    # No "What's New" row in either mobile More sheet; the desktop gear-menu
    # row stays.
    assert "_sheet_action_row(\"What's New\"" not in APP_PY
    assert "id='settingsChangelogBtn'" in APP_PY


def test_changelog_dropdown_lives_in_top_nav_on_mobile():
    assert "topNav.appendChild(changelog)" in APP_JS
    assert ".top-nav.br-mnav .changelog-dropdown {" in CSS


def test_changelog_panel_anchored_to_nav_not_viewport():
    """Regression: the <=640px .changelog-dropdown rule sets position:fixed,
    under which top:calc(100% + 8px) resolves against the viewport height and
    the opened panel renders below the screen (bell tap looked dead). The
    top-nav rule must pin position:absolute + transform:none so 100% means
    the nav height."""
    block = CSS.split(".top-nav.br-mnav .changelog-dropdown {", 1)[1].split("}", 1)[0]
    assert "position: absolute" in block
    assert "transform: none" in block
    assert "top: calc(100% + 8px)" in block
