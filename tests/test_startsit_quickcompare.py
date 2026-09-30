"""Start/Sit quick-compare button contract tests.

Once the first compare player is picked, every Start/Sit card shows a compare
button right on the row, so choosing the second player needs no expanding.
"""
import re

import pytest

from dashboard_services.pages.waivers_page import build_waivers_body


@pytest.fixture(scope="module")
def page():
    return build_waivers_body("sleeper", 2026, "12345", {})


@pytest.fixture(scope="module")
def script(page):
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    matches = [s for s in scripts if "function wvRenderStartSit(" in s]
    assert matches, "waivers inline script not found"
    return matches[0]


@pytest.fixture(scope="module")
def css():
    with open("static/dashboard.css") as f:
        return f.read()


def test_quick_compare_armed_state(script):
    # Armed only when slot 0 is filled, slot 1 is empty, and this card is not
    # the already-picked player.
    assert "const qcArmed = !!(wvCompare[0] && !wvCompare[1] && !qcPicked);" in script
    assert "const qcShow = qcPicked || qcArmed;" in script


def test_quick_compare_calls_toggle_with_player_json(script):
    # The quick button drives the same slot machine as the expanded detail
    # button: wvToggleCompare with the full player payload.
    assert 'onclick="wvToggleCompare(${{qcJson}})"' not in script  # f-string artifact guard
    assert 'onclick="wvToggleCompare(${qcJson})"' in script
    assert "const qcJson = JSON.stringify(p).replace" in script


def test_quick_compare_not_nested_in_row_button(script):
    # A <button> inside the row <button> would be reparented by the HTML
    # parser; the quick button must be a sibling after the row closes.
    rowline = script[script.index('class="wv-cx-rowline"'):]
    rowline = rowline[: rowline.index("wv-cx-detail")]
    assert rowline.index("</button>") < rowline.index("qcBtn")


def test_chevron_hidden_when_quick_shows(script):
    # While the quick-compare button shows, the chevron is omitted from the row.
    assert "${qcShow ? '' : '<span class=\"wv-cx-chev\" aria-hidden=\"true\">›</span>'}" in script


def test_picked_card_shows_filled_state(script):
    assert "wv-cx-quickcmp${qcPicked ? ' is-picked' : ''}" in script
    assert "${qcPicked ? '✓ Picked' : '+ Compare'}" in script


def test_quick_compare_css(css):
    assert ".wv-cx-rowline { display: flex; align-items: center; }" in css
    assert ".wv-cx-rowline > .wv-cx-row" in css
    assert ".wv-cx-quickcmp {" in css
    assert ".wv-cx-quickcmp.is-picked" in css
