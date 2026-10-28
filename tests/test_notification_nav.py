"""Notification center nav: bell, settings dropdown, week pill removal.

The desktop bell is a real borderless utility button now (its dot is its
own), the settings dropdown carries an identity header + grouped menu on
every page, the Week pill is gone from the league chip (week preserved as
a data attribute), and the phone top bell also renders on home/public nav.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")

GRAPHS = "/sleeper/2026/tourdemo/graphs?tour=1"


def _nav(monkeypatch, league_id="L", active="dashboard", *, premium=False,
         session_values=None):
    import app
    monkeypatch.setattr(app, "get_nfl_state",
                        lambda: {"season": "2026", "season_type": "off"})
    monkeypatch.setattr(app, "get_league_ctx_from_cache", lambda *a, **k: {})
    monkeypatch.setattr(app, "has_draft_ended", lambda *a, **k: False)
    monkeypatch.setattr(app, "has_premium_for_viewer",
                        lambda *a, **k: premium)
    path = "/sleeper/2026/L/dashboard" if league_id else "/"
    with app.app.test_request_context(path):
        for k, v in (session_values or {}).items():
            app.session[k] = v
        return app.build_nav(league_id, active, "sleeper", 2026)


_SIGNED_IN = {"account_id": 42, "account_first_name": "Kaedon",
              "account_email": "kaedon@example.com"}


# ── Bell ─────────────────────────────────────────────────────────────────────

def test_bell_is_a_real_button_with_aria(monkeypatch):
    html = _nav(monkeypatch, session_values=_SIGNED_IN)
    assert "id='changelogBell'" in html
    assert "aria-label='Notifications'" in html
    assert "aria-haspopup='true' aria-expanded='false'" in html
    # Its own unread dot ships in the markup (hidden until JS lights it).
    assert "changelog-dot changelog-dot-hidden" in html
    assert "id='changelogDropdown'" in html


def test_bell_unhidden_and_borderless_in_css():
    # The September-era hidden-bell block is gone entirely.
    assert "Keep changelog bell hidden" not in CSS
    assert "visibility: hidden" not in CSS.split(".changelog-bell-btn {")[1].split("}")[0]
    block = CSS.split(".changelog-bell-btn {")[1].split("}")[0]
    assert "border: none" in block
    assert "width: 36px" in block
    # Accent at rest: the same token the search + watchlist icons use.
    assert "color: var(--accent)" in block
    gear = CSS.split("#settingsGearBtn {")[1].split("}")[0]
    assert "color: var(--accent)" in gear


def test_panel_is_360_and_uncapped():
    block = CSS.split("\n.changelog-dropdown {")[1].split("}")[0]
    assert "width: 360px" in block
    assert "415px" not in block
    assert ".ni-item" in CSS and ".ni-udot" in CSS and ".ni-markall" in CSS


def test_proxy_dots_and_whats_new_row_retired():
    assert "id='gearDot'" not in APP_PY
    assert "id='settingsNotifDot'" not in APP_PY
    assert "id='settingsChangelogBtn'" not in APP_PY
    assert "gearDot" not in APP_JS
    assert "settingsNotifDot" not in APP_JS
    # The dot is driven on the bell itself.
    assert 'document.querySelector(".changelog-dot")' in APP_JS
    assert "changelog-dot-hidden" in APP_JS


def test_dead_sheet_changelog_mounts_removed():
    assert "brSheetChangelog" not in APP_PY
    assert "brSheetChangelog" not in APP_JS


# ── Settings dropdown ────────────────────────────────────────────────────────

def test_settings_menu_identity_and_pro_billing(monkeypatch):
    html = _nav(monkeypatch, premium=True, session_values=_SIGNED_IN)
    assert "settings-identity" in html
    assert "Kaedon" in html
    assert "Signed in with Google" in html
    assert "settings-plan-chip is-pro" in html
    assert ">PRO</span>" in html
    assert "id='settingsBillingBtn' data-br-action='billing-portal'" in html
    assert "Manage subscription" in html
    assert "Go PRO" not in html


def test_settings_menu_free_billing_row(monkeypatch):
    html = _nav(monkeypatch, premium=False, session_values=_SIGNED_IN)
    assert "settings-plan-chip'>Free</span>" in html
    assert "href='/pricing'" in html
    assert "Go PRO" in html
    assert "Manage subscription" not in html


def test_settings_menu_groups_in_order_sign_out_last(monkeypatch):
    html = _nav(monkeypatch, session_values=_SIGNED_IN)
    menu = html.split("id='settingsDropdown'", 1)[1]
    order = [menu.index(x) for x in (
        "settings-identity", ">League<", "id='leagueSwitcher'",
        "id='settingsLinkBtn'", ">Preferences<",
        "id='settingsNotifBtn'", "id='settingsDarkModeBtn'",
        ">Tools<", "id='settingsRefreshBtn'",
        ">Help<", "id='settingsHelpToursBtn'",
        "href='/logout'",
    )]
    assert order == sorted(order)
    # Dark Mode is a switch row that shows state; the label never renames.
    assert "role='switch' aria-checked='false'" in menu
    assert "settings-switch" in menu
    assert "theme-text" not in menu
    # Retired rows stay retired.
    assert "What's New" not in menu
    assert "Help / Support" not in menu
    assert "id='settingsTourBtn'" not in menu
    assert "id='settingsWelcomeBtn'" not in menu


def test_gear_aria_and_escape_wiring():
    assert APP_PY.count("id='settingsGearBtn'") >= 3
    assert "aria-haspopup='true' aria-expanded='false'" in APP_PY
    # Both the gear menu and the bell panel close on Escape (app.js).
    assert APP_JS.count('e.key === "Escape"') >= 2
    assert 'setAttribute("aria-expanded"' in APP_JS


def test_help_tours_dialog_folds_in_site_tour_and_welcome():
    assert "data-help-welcome" in APP_JS
    assert "PRO Welcome" in APP_JS
    assert "startSubWelcome" in APP_JS


def test_full_menu_on_home_pages(monkeypatch):
    html = _nav(monkeypatch, league_id=None, active="home",
                session_values=_SIGNED_IN)
    menu = html.split("id='settingsDropdown'", 1)[1]
    assert "settings-identity" in menu
    assert "id='settingsNotifBtn'" in menu
    assert "id='leagueSwitcher'" in menu
    assert "href='/logout'" in menu


def test_phone_top_buttons_on_home_nav(monkeypatch):
    html = _nav(monkeypatch, league_id=None, active="home",
                session_values=_SIGNED_IN)
    assert "class='br-top-search'" in html
    assert "class='br-top-notif'" in html
    assert "id='brTopNotifDot'" in html


# ── Guests + offseason negatives ─────────────────────────────────────────────

def test_guest_league_nav_has_bell_but_no_identity(monkeypatch):
    html = _nav(monkeypatch, session_values={})
    assert "id='changelogBell'" in html
    assert "settings-identity" not in html
    assert "href='/logout'" not in html
    assert "Sign In" in html


def test_guest_home_nav_has_bell_and_top_buttons(monkeypatch):
    html = _nav(monkeypatch, league_id=None, active="home", session_values={})
    assert "id='changelogBell'" in html
    assert "class='br-top-notif'" in html
    assert "settings-identity" not in html


def test_offseason_page_keeps_bell_and_panel(offline_client):
    r = offline_client.get(GRAPHS)
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "id='changelogBell'" in html
    assert "id='changelogDropdown'" in html
    assert "br-ctx-week" not in html


# ── Week pill ────────────────────────────────────────────────────────────────

def test_week_pill_replaced_by_data_attribute():
    import app
    meta = {"name": "Blackedraw", "format": "1QB", "week_label": "Week 8",
            "nfl_week_stale": False}
    out = app._render_league_chrome_chip(meta, can_switch=False)
    assert "br-ctx-week" not in out
    assert "data-week-label='Week 8'" in out
    assert "Blackedraw" in out


def test_changelog_announces_phone_panel_and_more_dot_fix():
    from dashboard_services.changelog import CHANGELOG

    top = CHANGELOG[15]  # prepended above it: rooting guide two-per-row + badge fit, Stripe period-end grant fix, awards page rework, graphs fill and audit, portfolio light resolver, trade calculator dedupe, recap/standings dedupe, sitewide speed pass, Weekly Hub dedupe refresh, waivers/start-sit call consolidation, AM batch + players cache, ScoreZone user-cache speedup, My Leagues portfolio consolidation, duplicate Tuesday dot removal, phone tabs + card odds removal
    assert top["date"] == "2026-10-27"
    assert top["tag"] == "fix"
    assert "Notifications on your phone" in top["text"]
    assert "More tab" in top["text"]
    assert "—" not in top["text"]  # no em dashes in UI copy


# ── Mobile panel: bounded height + dim ───────────────────────────────────────

def test_mobile_panel_capped_by_dynamic_viewport():
    """Regression (2026-10-27): the mobile panel inherited the desktop
    cap calc(100vh - 96px); 100vh is the LARGE viewport on iOS Safari,
    so the panel ran off-screen and its tail slid under the floating
    dock. The mobile rule caps by the dynamic viewport and reserves
    the dock's zone below the panel."""
    block = CSS.split(".top-nav.br-mnav .changelog-dropdown {")[1].split("}")[0]
    assert "max-height: calc(100vh - 176px)" in block
    assert "max-height: calc(100dvh - 176px)" in block
    # The vh fallback must come first so dvh overrides it where supported.
    assert block.index("100vh") < block.index("100dvh")
    # Desktop base rule keeps its own cap (this fix is mobile-only).
    base = CSS.split("\n.changelog-dropdown {")[1].split("}")[0]
    assert "max-height: calc(100vh - 96px)" in base


def test_mobile_panel_head_and_foot_stay_pinned():
    head = CSS.split(".top-nav.br-mnav .changelog-dropdown .ni-head {")[1].split("}")[0]
    assert "position: sticky" in head and "top: 0" in head
    foot = CSS.split(".top-nav.br-mnav .changelog-dropdown .ni-foot {")[1].split("}")[0]
    assert "position: sticky" in foot and "bottom: 0" in foot


def test_mobile_panel_dim_matches_sheet_scrim_pattern():
    # Hidden by default (desktop never shows it; JS only creates the
    # element on mobile), styled on the sheet scrim's dim.
    base = CSS.split("\n.ni-scrim {")[1].split("}")[0]
    assert "display: none" in base
    mob = CSS.split(".top-nav .ni-scrim {")[1].split("}")[0]
    assert "display: block" in mob
    assert "position: fixed" in mob and "inset: 0" in mob
    sheet = CSS.split(".br-sheet-scrim {")[1].split("}")[0]
    assert "rgba(4, 8, 17, .5)" in sheet
    assert "rgba(4, 8, 17, .5)" in mob
    open_block = CSS.split(".top-nav .ni-scrim.open {")[1].split("}")[0]
    assert "opacity: 1" in open_block and "pointer-events: auto" in open_block
    # While open, the nav is raised above the floating dock so the dim
    # covers the dock too (the .rz-bs-backdrop approach).
    assert ".top-nav.br-mnav.br-ni-open {" in CSS
    assert "z-index: calc(var(--z-chrome) + 2)" in CSS


def test_panel_js_wires_scrim_only_for_floating_mobile_panel():
    assert 'niScrim.className = "ni-scrim"' in APP_JS
    assert "setNiScrim(true)" in APP_JS and "setNiScrim(false)" in APP_JS
    assert 'classList.add("br-ni-open")' in APP_JS
    fn = _function_source(APP_JS, "niFloatingMobile")
    assert 'matchMedia("(max-width: 768px)")' in fn
    # The More sheet's inline Account panel keeps the sheet's own scrim.
    assert '"#brMoreSheet"' in fn and '"#brSheetAccount"' in fn


# ── Panel JS wiring ──────────────────────────────────────────────────────────

def test_panel_reads_inbox_api_and_marks_read():
    assert 'fetch("/api/notifications")' in APP_JS
    assert '"/api/notifications/read"' in APP_JS
    assert '"/api/notifications/read-all"' in APP_JS
    assert "niMarkAll" in APP_JS
    assert "Notification settings" in APP_JS
    # No "Held for ..." badge anywhere in the panel.
    assert "Held for" not in APP_JS


# ── Scrim behavior (node) ────────────────────────────────────────────────────

def _function_source(source: str, name: str) -> str:
    start = source.index("function %s(" % name)
    brace = source.index("{", start)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start:pos + 1]
    raise AssertionError("unterminated %s" % name)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_ni_scrim_behavior():
    gate = _function_source(APP_JS, "niFloatingMobile")
    setter = _function_source(APP_JS, "setNiScrim")
    harness = """
function ClassList(owner) { this.owner = owner; this.set = new Set(); }
ClassList.prototype.add = function (c) { this.set.add(c); };
ClassList.prototype.remove = function (c) { this.set.delete(c); };
ClassList.prototype.contains = function (c) { return this.set.has(c); };
function El(tag) {
  this.tagName = tag; this.children = []; this.parentNode = null;
  this.classList = new ClassList(this); this.attrs = {};
}
El.prototype.setAttribute = function (k, v) { this.attrs[k] = v; };
El.prototype.insertBefore = function (node, ref) {
  node.parentNode = this; this.children.push(node); return node;
};
var mobile = true, inSheet = false;
var nav = new El('header'); nav.className = 'top-nav br-mnav';
var dropdown = new El('div');
dropdown.closest = function (sel) {
  if (sel === '.top-nav') return nav;
  if (sel === '#brMoreSheet' || sel === '#brSheetAccount') return inSheet ? nav : null;
  return null;
};
var niScrim = null;
var window = { matchMedia: function () { return { matches: mobile }; } };
var document = { createElement: function (t) { return new El(t); } };
function ok(v, m) { if (!v) throw new Error(m); }
%s
%s

// 1. Mobile floating panel opening: scrim created inside the nav,
//    faded in, and the nav is raised above the dock.
setNiScrim(true);
ok(niScrim, 'scrim must be created');
ok(niScrim.parentNode === nav, 'scrim must live inside the top nav');
ok(niScrim.classList.contains('open'), 'scrim must get .open');
ok(nav.classList.contains('br-ni-open'), 'nav must get .br-ni-open');

// 2. Closing: dim and raise both come off (element stays parked).
setNiScrim(false);
ok(!niScrim.classList.contains('open'), 'close must remove .open');
ok(!nav.classList.contains('br-ni-open'), 'close must remove .br-ni-open');

// 3. Reopening reuses the same element (no duplicates pile up).
var first = niScrim;
setNiScrim(true);
ok(niScrim === first, 'reopen must reuse the parked scrim element');

// 4. Panel living in the More sheet: no scrim (the sheet has its own).
setNiScrim(false);
niScrim = null; inSheet = true;
setNiScrim(true);
ok(niScrim === null, 'sheet-inline panel must not create a scrim');
ok(!nav.classList.contains('br-ni-open'), 'sheet-inline must not raise the nav');

// 5. Desktop width: no scrim.
inSheet = false; mobile = false;
setNiScrim(true);
ok(niScrim === null, 'desktop must not create a scrim');
console.log('ni scrim behavior ok');
""" % (gate, setter)
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ni scrim behavior ok" in result.stdout
