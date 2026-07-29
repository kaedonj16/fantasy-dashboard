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
    # Search is a row that opens the full-screen search screen; the settings
    # menu is relocated into the Account mount.
    assert "id='brSheetSearchRow'" in html
    assert "id='brSearchScreen'" in html
    assert "id='brSearchMount'" in html
    assert "id='brSheetAccount'" in html
    # The widgets that get relocated must exist in the server HTML.
    assert "id='navSearchWrapper'" in html
    assert "id='settingsDropdown'" in html


def test_top_bar_marker_off_dashboard(offline_client):
    # The bar carries the br-mnav marker on every league page (CSS slims it to a
    # logo-only bar on phones); br-mnav-home marks only the dashboard.
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


@pytest.mark.parametrize("settings,expected", [
    ({"type": 1},                    True),   # keeper league
    ({"type": 0, "max_keepers": 3},  True),   # redraft with a keeper limit
    ({"type": 0, "max_keepers": 0},  False),  # pure redraft
    ({"type": 2, "max_keepers": 0},  False),  # true dynasty
    ({"type": 2, "max_keepers": 20}, False),  # dynasty that still reports a limit
    ({},                             False),  # unknown / non-Sleeper, no limit
])
def test_nav_show_keeper_rules(monkeypatch, settings, expected):
    """Dynasty (type 2) never shows a Keeper dock tab, even with a max_keepers
    value; keeper leagues and configured keeper limits do."""
    import app
    monkeypatch.setattr(app, "get_league_ctx_from_cache",
                        lambda *a, **k: {"league_settings": settings})
    assert app._nav_show_keeper("sleeper", "L", 2026) is expected


def test_dynasty_offseason_dock_has_no_keeper(monkeypatch):
    """A dynasty league in the offseason gets the redraft dock layout
    (Draft, Trades, Teams) rather than a Keeper tab."""
    import app, re
    monkeypatch.setattr(app, "get_league_ctx_from_cache",
                        lambda *a, **k: {"league_settings": {"type": 2, "max_keepers": 20}})
    monkeypatch.setattr(app, "get_nfl_state", lambda: {"season": "2026", "season_type": "off"})
    monkeypatch.setattr(app, "has_draft_ended", lambda *a, **k: False)
    with app.app.test_request_context("/x"):
        labels = re.findall(r"br-tabbar-lbl'>([^<]+)<", app._mobile_nav("dashboard", "L", "sleeper", 2026))
    assert labels == ["Home", "Draft", "Trades", "Teams", "More"]
    assert "Keeper" not in labels


def test_preseason_is_treated_as_offseason(monkeypatch):
    """Preseason ("pre", ~August) has no fantasy games, so it gets the same
    offseason dock as "off" — not the empty in-season layout."""
    import app, re
    monkeypatch.setattr(app, "get_league_ctx_from_cache",
                        lambda *a, **k: {"league_settings": {"type": 2, "max_keepers": 20}})
    monkeypatch.setattr(app, "get_nfl_state", lambda: {"season": "2026", "season_type": "pre"})
    monkeypatch.setattr(app, "has_draft_ended", lambda *a, **k: False)
    with app.app.test_request_context("/x"):
        labels = re.findall(r"br-tabbar-lbl'>([^<]+)<", app._mobile_nav("dashboard", "L", "sleeper", 2026))
    assert labels == ["Home", "Draft", "Trades", "Teams", "More"]


@pytest.mark.parametrize("settings,shown", [
    ({"type": 1, "max_keepers": 2}, True),   # keeper league
    ({"type": 0, "max_keepers": 3}, True),   # redraft with a keeper limit
    ({"type": 2, "max_keepers": 0}, False),  # dynasty
    ({"type": 0, "max_keepers": 0}, False),  # plain redraft
])
def test_keeper_assistant_gated_everywhere(monkeypatch, settings, shown):
    """Keeper Assistant appears in the mobile sheet and the desktop Draft
    dropdown only for keeper leagues, hidden for dynasty and plain redraft."""
    import app
    monkeypatch.setattr(app, "get_league_ctx_from_cache",
                        lambda *a, **k: {"league_settings": settings})
    monkeypatch.setattr(app, "get_nfl_state", lambda: {"season": "2026", "season_type": "off"})
    monkeypatch.setattr(app, "has_draft_ended", lambda *a, **k: False)
    with app.app.test_request_context("/x"):
        sheet = app._mobile_nav("draft", "L", "sleeper", 2026)
        nav = app.build_nav("L", "draft", "sleeper", 2026)
    assert ("Keeper Assistant" in sheet) is shown
    assert ("Keeper Assistant" in nav) is shown


def test_draft_room_cfg_show_keeper():
    """The draft room only offers the Keeper draft type when show_keeper is set."""
    from dashboard_services.pages.draft_room_page import build_draft_room_body
    assert '"showKeeper": false' in build_draft_room_body("L", 2026, "sleeper", show_keeper=False)
    assert '"showKeeper": true' in build_draft_room_body("L", 2026, "sleeper", show_keeper=True)
