"""Contract tests: Dock B + the unified notification dot system.

One chrome pass:
- The mobile dock is Direction B (64px floating bar, blurred surface,
  20px radius, full-tab gliding fill, opacity press feedback).
- Every chrome dot/badge speaks one system: --dot-alert (unread/new),
  --dot-live (happening now, one shared pulse), --dot-status (system
  state). Counted badges (.count-badge, watchlist) share one spec and
  anchor to the icon's top-right corner, never the wide button box.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _block(selector):
    start = CSS.index(selector + " {")
    return CSS[start:CSS.index("}", start)]


def test_dot_tokens_exist_light_and_dark():
    assert "--dot-alert: #E24B4A;" in CSS
    assert "--dot-live: #16A34A;" in CSS
    assert "--dot-status: #F59E0B;" in CSS
    assert "--dot-alert: #FF6B69;" in CSS
    assert "--dot-live: #4ADE80;" in CSS


def test_dock_b_bar():
    # The mobile rule (the base .br-tabbar rule is just display:none).
    start = CSS.index(".br-tabbar {\n        display: flex;")
    bar = CSS[start:CSS.index("}", start)]
    assert "left: max(16px, env(safe-area-inset-left))" in bar
    assert "bottom: calc(env(safe-area-inset-bottom) + 12px)" in bar
    assert "border-radius: 20px" in bar
    assert "color-mix(in srgb, var(--bg) 92%, transparent)" in bar
    assert "backdrop-filter: blur(14px) saturate(1.5)" in bar
    assert "box-shadow: 0 16px 40px rgba(18, 45, 75, .18)" in bar


def test_dock_b_items_and_press_feedback():
    item = _block(".br-tabbar-item")
    assert "height: 64px" in item
    assert "font-size: 11px" in item
    assert "gap: 4px" in item
    active = _block(".br-tabbar-item:active")
    assert "opacity: .6" in active
    assert "scale(0.94)" not in CSS.split(".br-tabbar-item:active", 1)[1].split("}", 1)[0]


def test_indicator_fill_covers_full_tab():
    # Full-tab fill (variant C): the ::before fills the whole slot behind
    # icon AND label; the slot's 6px padding is the fill's inset.
    slot = _block(".br-tabbar-ind")
    assert "padding: 6px" in slot
    pill = _block(".br-tabbar-ind::before")
    assert "width: 100%" in pill
    assert "height: 100%" in pill
    assert "border-radius: 14px" in pill


def test_alert_dots_unified():
    for sel in (".changelog-dot", ".alert-dot"):
        block = _block(sel)
        assert "width: 8px" in block
        assert "height: 8px" in block
        assert "background: var(--dot-alert)" in block
        assert "border: 2px solid var(--card)" in block
    assert "background: var(--dot-alert)" in _block(".ni-udot")
    assert "background: var(--dot-alert)" in _block(".br-sheet-acct-dot")


def test_count_badge_spec_and_anchors():
    badge = _block(".count-badge")
    assert "height: 16px" in badge
    assert "min-width: 16px" in badge
    assert "padding: 0 5px" in badge
    assert "background: var(--dot-alert)" in badge
    assert "font-size: 10px" in badge
    assert "font-weight: 800" in badge
    assert "border: 2px solid var(--card)" in badge
    # Icon-corner anchors: bell 36px btn / 16px icon, mobile bell 38px / 20px.
    assert "top: 2px" in _block(".changelog-bell-btn .count-badge")
    assert "top: 1px" in _block(".top-nav.br-mnav .br-top-notif .count-badge")
    # Badges are rendered next to both bells and start hidden.
    assert APP_PY.count("class='count-badge'") == 3
    assert "id='changelogBadge'" in APP_PY
    assert APP_PY.count("id='brTopNotifBadge'") == 2


def test_badge_js_swaps_dot_for_count():
    assert "function setChangelogDot(hasNew, unreadCount)" in APP_JS
    assert 'count > 9 ? "9+" : String(count)' in APP_JS
    assert 'setChangelogDot(inboxUnread > 0 || changelogUnseen(), inboxUnread)' in APP_JS
    assert 'document.getElementById("changelogBadge")' in APP_JS or 'getElementById("changelogBadge")' in APP_JS


def test_watchlist_badge_matches_badge_spec():
    block = _block(".watchlist-alert-dot")
    assert "height: 16px" in block
    assert "min-width: 16px" in block
    assert "background: var(--dot-alert)" in block
    assert "font-size: 10px" in block
    assert "top: 2px" in block and "right: 2px" in block
    assert "alertCount > 9 ? '9+'" in APP_JS


def test_live_dots_unified_green_one_pulse():
    assert "@keyframes br-live-pulse" in CSS
    for sel in (".rz-mnav-dot", ".rz-live-dot-sm", ".pm-live-dot",
                ".rz-psb-live-dot", ".rz-nfl-live-dot"):
        block = _block(sel)
        assert "background: var(--dot-live)" in block, sel
        assert "width: 8px" in block, sel
    assert "background: var(--dot-live)" in _block(".live-dot")
    assert "border-radius: 50%" in _block(".live-dot")
    # Live never recolours the More tab or the sheet row/icon.
    assert ".br-more-tab.br-more-live {" not in CSS
    assert ".br-sheet-link.rz-mnav-live .nav-svg-icon" not in CSS
    # More-tab live dot anchors to the More icon corner (new dock geometry).
    more_dot = _block(".br-more-tab .rz-mnav-dot")
    assert "top: 8px" in more_dot
    assert "margin-left: 8px" in more_dot


def test_status_dots_use_status_token():
    assert "background: var(--dot-status)" in _block(".conn-banner-dot")
    assert "background: var(--dot-status)" in _block(".offline-dot")
