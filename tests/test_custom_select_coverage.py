"""Dynamic selects injected after page load must join the site custom
select dropdown (CSD), and the player-modal team target-share bar must use
the shared themed tooltip engine instead of native title tooltips."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(*parts):
    return (ROOT.joinpath(*parts)).read_text()


def test_paywall_league_picker_select_is_enhanced():
    source = _read("static", "paywall.js")
    block = source.split("function _showLeaguePickerModal(")[1].split("\nfunction ")[0]
    assert 'id="_leaguePickerSelect"' in block
    assert "window.initCustomSelects(modal)" in block


def test_paywall_identify_select_is_enhanced():
    source = _read("static", "paywall.js")
    block = source.split("function _showIdentifyModal(")[1].split("\nfunction ")[0]
    assert 'id="_identifyLeague"' in block
    assert "window.initCustomSelects(modal)" in block
    # Exactly the two paywall modals enhance; no stray extra calls.
    assert source.count("window.initCustomSelects(modal)") == 2


def test_player_page_league_select_is_enhanced():
    source = _read("static", "player_page.js")
    block = source.split("function renderStep(")[1].split('else if (platform === "espn")')[0]
    assert "id='ppLeague'" in block
    assert "window.initCustomSelects(stepEl)" in block


def test_draft_room_pick_trade_selects_are_enhanced():
    source = _read("static", "draft_room.js")
    block = source.split("function drPickTradeOpen()")[1].split("function drPickTrade")[0]
    assert 'class="dr-pt-sel"' in block
    # The enhancer runs on the modal body right after the selects render.
    assert block.index('id="drPtResult"') < block.index("window.initCustomSelects(msg)")


def test_signin_select_wraps_fill_their_row():
    dashboard = _read("static", "dashboard.css")
    assert ".signin-modal-box .csd-wrap" in dashboard
    assert "#ppLeagueWrap .csd-wrap" in dashboard
    # Lite packs carry the context rule for the surfaces they serve.
    assert "#ppLeagueWrap .csd-wrap" in _read("static", "seo_lite.css")
    assert ".signin-modal-box .csd-wrap" in _read("static", "landing_lite.css")


def test_draft_room_pick_trade_wrap_keeps_flex_layout():
    page = _read("dashboard_services", "pages", "draft_room_page.py")
    assert ".dr-pt-picker .csd-wrap { flex: 1 1 auto; min-width: 0; }" in page


def test_team_share_bar_uses_themed_tooltip_not_native_title():
    source = _read("static", "player_modal.js")
    block = source.split("function _pmTeamShareBar(")[1].split("\nfunction ")[0]
    assert "title=" not in block
    assert 'data-def="${tip}"' in block
    assert 'data-def="Rest of offense: ${rest}%"' in block
    assert "advEnterMetricDef(event)" in block
    assert "advShowMetricDef(event)" in block
