"""Hover-audit fix contract (2026-10-13, follow-up to the site-wide hover
audit in ~/workspace/hover-audit-2026-10-13.md).

Locks in, at the source level:
- hover-only content gained tap/focus paths (Trade Calculator PI panel and
  info tooltip, draft board tip, AM pin button and column definitions)
- name hovers follow the dim-only standard everywhere
- the global title= upgrade lives in app.js core (before the public-js
  marker) so native tooltips become the themed bubble
- dead / duplicated / divergent hover CSS stayed removed or unified

These tests read shipped sources only, like test_player_name_hover.py.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(rel):
    with open(os.path.join(_ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def _rule_body(css, selector):
    idx = css.find(selector)
    assert idx != -1, f"selector not found: {selector}"
    open_brace = css.find("{", idx)
    close_brace = css.find("}", open_brace)
    assert open_brace != -1 and close_brace != -1, f"rule body not found: {selector}"
    return css[open_brace + 1:close_brace]


_CSS = _read("static/dashboard.css")
_APP_JS = _read("static/app.js")


# ── dashboard.css ────────────────────────────────────────────────────────

def test_recap_efficiency_team_names_have_no_underline_rule():
    assert ".recap-eff-name .team-clickable:hover" not in _CSS


def test_dark_mode_toggle_does_not_grow():
    body = _rule_body(_CSS, "#darkModeToggle:hover")
    assert "scale" not in body


def test_waiver_row_hover_is_a_plain_tint():
    body = _rule_body(_CSS, ".os-waiver-row:hover").lower()
    assert "transform" not in body and "box-shadow" not in body
    assert "background" in body


def test_redzone_event_hover_does_not_slide():
    # ScoreZone fix-all (#2804) removed the .rz-event hover rule entirely:
    # play rows are not interactive (click handler and cursor:pointer are
    # gone), so there is no hover state left to slide. If a hover rule ever
    # returns, it must stay a non-moving treatment.
    if ".rz-event:hover" not in _CSS:
        base = _rule_body(_CSS, ".rz-event {").lower()
        assert "cursor: pointer" not in base
        return
    body = _rule_body(_CSS, ".rz-event:hover").lower()
    assert "transform" not in body and "box-shadow" not in body


def test_dead_hover_css_stays_dead():
    assert ".player-modal-game-log-row" not in _CSS
    assert ".br-lift" not in _CSS


def test_otc_view_all_link_has_one_base_hover_rule():
    # One base rule plus its dark-theme variant; the duplicate standalone
    # underline-only rule was deleted (appearance unchanged).
    assert _CSS.count(".otc-view-all-link:hover") == 2


def test_trend_pill_tooltip_is_capped_and_focusable():
    body = _rule_body(_CSS, ".pm-trend-pill::after")
    assert "max-width" in body
    assert ".pm-trend-pill:focus-visible::after" in _CSS


def test_floating_pills_expand_on_keyboard_focus():
    assert "#discord-pill:focus-visible" in _CSS
    assert "#cache-freshness:focus-visible" in _CSS


def test_otc_info_tooltip_is_width_capped():
    body = _rule_body(_CSS, ".otc-info-tooltip {")
    assert "max-width" in body


# ── landing / seo bundles ────────────────────────────────────────────────

def test_lite_bundles_do_not_grow_nav_icons():
    for rel in ("static/landing_lite.css", "static/seo_lite.css"):
        css = _read(rel)
        assert "scale(1.05)" not in css, rel


# ── paywall ──────────────────────────────────────────────────────────────

def test_pricing_option_hover_is_unified():
    css = _read("static/paywall.css")
    assert ".paywall-pricing .pricing-option:hover" not in css
    body = _rule_body(css, ".pricing-option:hover")
    assert "translateY(-3px)" in body


# ── app.js: OTC bindings + title upgrade ─────────────────────────────────

def test_otc_info_tooltip_has_click_and_pin():
    assert "otcInfoClick" in _APP_JS
    assert "otcInfoPinned" in _APP_JS


def test_analyze_button_no_longer_preopens_on_hover():
    assert 'btn.addEventListener("mouseenter"' not in _APP_JS


def test_title_upgrade_lives_in_core_before_the_marker():
    start = _APP_JS.find("// [title-upgrade:start]")
    end = _APP_JS.find("// [title-upgrade:end]")
    marker = _APP_JS.find("// @public-js:core-end")
    assert start != -1 and end != -1 and marker != -1
    assert start < marker and end < marker
    block = _APP_JS[start:end]
    assert "brUpgradeTitles" in block
    assert "MutationObserver" in block
    # Native-only surfaces keep their titles.
    for skipped in ("IFRAME", "INPUT", "TEXTAREA", "SELECT"):
        assert skipped in block


# ── Trade Calculator page ────────────────────────────────────────────────

def test_playoff_impact_trigger_is_a_button_with_focus_reveal():
    src = _read("dashboard_services/pages/trade_calculator_page.py")
    assert '<button type="button" class="pi-info-btn"' in src
    assert ".pi-info-wrap:focus-within .pi-tooltip" in src


# ── Draft room ───────────────────────────────────────────────────────────

def test_board_tip_has_a_tap_path_and_measures_itself():
    js = _read("static/draft_room.js")
    assert js.count("buildTeamTip(slot)") >= 2  # hover fill + tap fill
    assert "tip.offsetWidth" in js
    assert "(pointer: coarse)" in js


def test_cap_pill_remove_cue_shows_on_touch():
    src = _read("dashboard_services/pages/draft_room_page.py")
    assert "@media (pointer: coarse) { .dr-cap-pill-x" in src


# ── Advanced Metrics page ────────────────────────────────────────────────

def test_am_pin_button_reveals_without_hover():
    src = _read("dashboard_services/pages/advanced_metrics_page.py")
    assert ".am-row:focus-within .am-pin-btn" in src
    assert "@media (pointer: coarse) { .am-pin-btn" in src


def test_am_column_definitions_have_a_keyboard_path():
    src = _read("dashboard_services/pages/advanced_metrics_page.py")
    assert "_amTipFocus" in src and "_amTipBlur" in src
    assert "th.tabIndex = 0;" in src


# ── Recap / activity / portfolio name treatments ─────────────────────────

def test_recap_page_has_no_team_name_underline_rules():
    src = _read("dashboard_services/pages/recap_page.py")
    assert "team-clickable:hover" not in src


def test_activity_team_names_dim():
    src = _read("dashboard_services/pages/activity_page.py")
    assert ".act-pulse .act-tm.team-clickable:hover {{ opacity: 0.72; }}" in src


def test_portfolio_league_name_accent_only_on_the_link():
    src = _read("app.py")
    assert "a.pf-lg-name:hover{color:var(--accent);}" in src
    assert '".pf-lg-name:hover{color:var(--accent);}"' not in src


# ── Player modal ─────────────────────────────────────────────────────────

def test_trend_pill_is_keyboard_focusable():
    js = _read("static/player_modal.js")
    assert 'class=\\"pm-trend-pill\\" tabindex=\\"0\\"' in js or \
        'class="pm-trend-pill" tabindex="0"' in js


def test_wk_tip_clamps_to_the_viewport():
    js = _read("static/player_modal.js")
    assert "window.innerWidth - w / 2 - 8" in js
