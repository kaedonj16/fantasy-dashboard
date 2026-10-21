"""Design-system foundation contract.

The site grew forked profile / title / button geometry page by page
(player modal vs prospect hero vs NFL team chips vs team-modal tiles,
inline 22px h1 styles, and a `.btn` class dashboard.css never defined).
These tests pin the shared primitives in dashboard.css and the migrated
surfaces so the geometry cannot silently fork again.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _page(name):
    return (ROOT / "dashboard_services" / "pages" / name).read_text(encoding="utf-8")


def test_btn_family_is_defined():
    # .btn historically did not exist in dashboard.css at all.
    assert ".btn {" in CSS
    assert "height: 32px;" in CSS
    assert ".btn-primary {" in CSS
    assert ".btn-ghost {" in CSS
    assert ".btn-sm {" in CSS


def test_page_title_suppresses_global_hero_bar():
    assert ".page-title {" in CSS
    assert ".page-title::after { content: none; display: none; }" in CSS
    assert "text-align: left;" in CSS


def test_profile_system_exists_with_shared_tile_spec():
    for cls in (".profile-head {", ".profile-media {", ".profile-stats {",
                ".profile-stat {", ".profile-stat.primary {",
                ".profile-lab {", ".profile-val {", ".profile-sub {"):
        assert cls in CSS, cls
    # The one stat-tile spec: bg-alt surface, token radius, 28px value.
    assert ".profile-val {\n    font-size: 28px;" in CSS


def test_prospect_hero_matches_shared_tile_geometry():
    # dashboard.css copy (prospect surfaces + player-modal rookie tab).
    assert ".rk-hero-stat {\n    background: var(--bg-alt);" in CSS
    assert ".pm-rk-hero-v { font-size: 28px; font-weight: 700;" in CSS
    # app.js embedded copy (legacy fallback prospect modal) must not fork.
    assert "background: var(--bg-alt);" in APP_JS
    assert ".rk-hero-val {\n        font-size: 28px;" in APP_JS


def test_team_modal_tiles_match_shared_tile_spec():
    assert ".tm-stat-tile-value {\n    font-size: 28px;\n    font-weight: 700;" in CSS


def test_nfl_team_chips_are_tiles_not_borderless_chips():
    src = _page("nfl_teams_page.py")
    assert ".nt-chip{{background:var(--bg-alt);border:1px solid var(--border);" in src
    assert ".nt-chip .nt-cval{{font-size:28px;font-weight:700;" in src
    # Header keeps the team colour only as the inset top hairline.
    assert "nt-herohead\" style=\"box-shadow:inset 0 3px 0 " in src


def test_inline_page_titles_adopted_page_title():
    assert '<h1 class=\\"page-title\\">League activity</h1>' in _page("activity_page.py")
    assert '<h1 class="page-title" style="margin-bottom:8px;">League Health</h1>' in _page("commissioner_page.py")
    assert 'class="opt-page-title page-title"' in _page("optimal_page.py")


def test_shells_fields_and_chip_toggle_exist():
    for cls in (".shell-data {", ".shell-app {", ".shell-read {",
                ".field {", ".filterbar {", ".chip.on {"):
        assert cls in CSS, cls
    assert ".lh-wrap {{ max-width:1180px;" in _page("commissioner_page.py")


def test_otc_btn_uses_btn_geometry():
    block = CSS.split(".otc-btn {", 1)[1].split("}", 1)[0]
    assert "height: 32px;" in block
    assert "border-radius: var(--radius-pill);" in block
    assert "font-size: 13px;" in block


def test_underline_tabs_are_one_shared_rule():
    # .tm-tab was a byte-identical copy of .pm-tab; one rule serves both.
    assert ".pm-tab, .tm-tab {" in CSS
    assert ".pm-tab.active, .tm-tab.active {" in CSS


def test_global_table_headers_use_tbl_spec():
    block = CSS.split("\nthead th {", 1)[1].split("}", 1)[0]
    assert "font-size: 11px;" in block
    assert "text-transform: uppercase;" in block
    assert "letter-spacing: 0.06em;" in block


def test_draft_room_title_and_buttons_aligned():
    src = _page("draft_room_page.py")
    assert ".dr-title::after { content: none; display: none; }" in src
    assert "font-size: 24px; font-weight: 800" not in src.split(".dr-title")[1].split("}")[0]
    assert "display: inline-flex; align-items: center; justify-content: center; height: 32px;" in src
