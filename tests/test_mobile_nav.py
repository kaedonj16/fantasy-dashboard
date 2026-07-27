"""Render tests for the mobile navigation (dynamic bottom dock + More sheet).

These go through the real Flask stack via the ``offline_client`` fixture (Sleeper
HTTP mocked), so they exercise build_nav() and _mobile_nav() end to end. Skipped
automatically when Flask/pandas aren't installed; they run in CI where the full
stack is present.

The dock replaces the top nav on phones, so the assertions here are the contract
the CSS/JS relies on:
  - a dynamic dock (`.br-tabbar`) with a More button and a More sheet,
  - mount points the client moves Search and the settings menu into,
  - top-nav marker classes so CSS keeps the bar (logo only) on the dashboard and
    hides it elsewhere,
  - the page you're on always earning a dock tab (here: Graphs).
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

# Tour pages render from seeded mock data with a real league context, so they
# carry the full league chrome (dock + sheet) without needing live data. Graphs
# is deliberately not one of the dock's default slots, so it must be pulled in.
GRAPHS = "/sleeper/2026/tourdemo/graphs?tour=1"


def _html(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    return r.get_data(as_text=True)


def test_dock_and_sheet_present(offline_client):
    html = _html(offline_client, GRAPHS)
    # The dock, its More button, and the sheet.
    assert "class='br-tabbar'" in html
    assert "id='brMoreTab'" in html
    assert "id='brMoreSheet'" in html
    # Mount points app.js relocates Search and the settings menu into.
    assert "id='brSheetFind'" in html
    assert "id='brSheetAccount'" in html
    # The widgets that get relocated must exist in the server HTML.
    assert "id='navSearchWrapper'" in html
    assert "id='settingsDropdown'" in html


def test_top_bar_marker_off_dashboard(offline_client):
    # Graphs is not the dashboard: the bar is marked present-but-not-home, so the
    # CSS hides it entirely on phones.
    html = _html(offline_client, GRAPHS)
    assert "top-nav br-mnav" in html
    assert "br-mnav-home" not in html


def test_current_page_earns_a_dock_tab(offline_client):
    # Graphs isn't a default dock slot, so it takes over the last middle slot and
    # shows as an active tab rather than leaving the bar with no active state.
    html = _html(offline_client, GRAPHS)
    assert "br-tabbar-lbl'>Graphs<" in html
    assert "br-tabbar-item active" in html


def test_more_sheet_lists_core_pages(offline_client):
    html = _html(offline_client, GRAPHS)
    # A scannable, grouped list of everything else lives in the sheet.
    for section in ("Find", "League", "Draft", "Players", "Stats", "Account"):
        assert f"br-sheet-h'>{section}<" in html
    # Watchlist is a plain link to the full page (no popover to reposition).
    assert "href='/watchlist'" in html
