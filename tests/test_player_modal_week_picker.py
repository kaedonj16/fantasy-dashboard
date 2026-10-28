"""Mobile week picker for the player modal Advanced tab.

On phones the inline week bar is replaced by a compact field in the Weeks
group; tapping it floats a panel over the tab content hosting a second
instance of the shared week bar (advWkBarM) plus range presets. Desktop
keeps the inline bar untouched. These are source-contract tests in the
style of test_advanced_metrics_week_range.py.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODAL = (ROOT / "static/player_modal.js").read_text()
CSS = (ROOT / "static/dashboard.css").read_text()
SW = (ROOT / "static/sw.js").read_text()


def _mobile_block():
    start = CSS.index("Mobile week picker (Advanced tab)")
    end = CSS.index(".cmp-week-bar-wrap", start)
    return CSS[start:end]


def test_desktop_inline_bar_unchanged():
    # The settled desktop control: Season button + shared bar in the row.
    assert "_wkBarBuild('advWkBar'" in MODAL
    assert "_wkBarInit('advWkBar'" in MODAL
    assert "adv-week-full-btn" in MODAL
    assert "adv-week-bar-row" in MODAL


def test_mobile_bar_is_a_second_instance_of_the_shared_bar():
    # Same shared component + same change callback, not a reimplementation.
    assert "_wkBarBuild('advWkBarM'" in MODAL
    assert "_wkBarInit('advWkBarM'" in MODAL
    assert MODAL.count("loadAdvancedMetrics(_wkPid, _wkLid, _wkSeas, ws, we)") == 2


def test_field_and_panel_markup():
    assert 'id="advWkField"' in MODAL
    assert "adv-wk-field-val" in MODAL
    assert 'id="advWkPanel"' in MODAL
    assert 'id="advWkDone"' in MODAL
    assert "Updates as you drag" in MODAL
    for preset in ("full", "last3", "last5"):
        assert f'data-wk-preset="{preset}"' in MODAL
    assert "Full season" in MODAL and "Last 3" in MODAL and "Last 5" in MODAL


def test_field_label_reflects_range():
    # Full range reads "Season" (the desktop button's word); a partial range
    # reads "W<lo> to W<hi>"; a single week reads "W<n>".
    assert "const wkFieldLabel = isFullRange ? 'Season'" in MODAL
    assert "'W' + effWS + ' to W' + effWE" in MODAL
    assert "'W' + effWS" in MODAL


def test_panel_open_state_is_scoped_to_player_and_season():
    assert "let _advWkPanelKey = null;" in MODAL
    assert "_advWkPanelKey === playerId + '|' + activeSeason" in MODAL
    # States without a Weeks group clear any leftover open state.
    assert "_advWkPanelKey = null;" in MODAL


def test_presets_call_the_same_loader():
    assert "_advWireWeekPicker(" in MODAL
    # Full season / covered window: no-week-args load, like the Season button.
    assert "loadAdvancedMetrics(playerId, leagueId, season);" in MODAL
    # Trailing window: last N available weeks ending at the latest week.
    assert "availableWeeks[availableWeeks.length - n]" in MODAL
    assert "availableWeeks[availableWeeks.length - 1]" in MODAL


def test_outside_tap_closes_panel():
    assert "window._advWkOutsideWired" in MODAL
    assert "_advCloseWeekPanel" in MODAL
    assert "wrap.contains(e.target)" in MODAL


def test_picker_is_mobile_only_in_css():
    block = _mobile_block()
    # Base (all widths): field + panel never render; desktop bar untouched.
    base = CSS.split("Mobile week picker (Advanced tab)")[1].split("@media")[0]
    assert ".adv-wk-field" in base and "display: none" in base
    # Inside the 640px block the inline bar hides and the field appears.
    assert "@media (max-width: 640px)" in block
    assert ".adv-time-group .adv-week-bar-row" in block
    assert "display: none;" in block
    assert ".adv-wk-field" in block and "display: flex" in block


def test_panel_floats_over_content_without_scrim():
    block = _mobile_block()
    m = re.search(r"\.adv-wk-panel:not\(\[hidden\]\)\s*\{([^}]*)\}", block)
    assert m, "open panel rule missing"
    rule = m.group(1)
    assert "position: absolute" in rule
    assert "top: calc(100% + 6px)" in rule
    assert "box-shadow" in rule
    # The wrap is the positioning anchor.
    assert re.search(r"\.adv-time-wrap\s*\{\s*position: relative;", CSS)


def test_panel_bar_round_thumbs_with_live_week_bubbles():
    block = _mobile_block()
    grip = re.search(r"\.adv-wk-panel \.wk-bar-grip\s*\{([^}]*)\}", block)
    assert grip and "border-radius: 50%" in grip.group(1)
    assert "width: 30px" in grip.group(1) and "height: 30px" in grip.group(1)
    # Bubble text comes from the grip's aria-valuenow, which the shared bar's
    # paint() updates during drags.
    assert "content: 'W' attr(aria-valuenow)" in block
    # Selected segment is a solid fill in the panel only.
    sel = re.search(r"\.adv-wk-panel \.wk-bar-sel\s*\{([^}]*)\}", block)
    assert sel and "background: var(--brand-blue)" in sel.group(1)


def test_sw_cache_bumped():
    assert "br-fantasy-v131" in SW
