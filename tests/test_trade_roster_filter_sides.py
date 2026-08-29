"""Regression checks for viewer-side ownership while roster filtering is active."""

from pathlib import Path


APP_JS = Path(__file__).parents[1] / "static" / "app.js"


def test_team_selection_rebinds_roster_filter_viewer():
    source = APP_JS.read_text(encoding="utf-8")
    helper = source[source.index("function syncRosterFilterViewer(") :]
    helper = helper[: helper.index("\n  function ", 10)]

    assert "rosterFilter.viewerRid = newViewerRid" in helper
    assert 'rosterFilter.sideBRid = ""' in helper
    assert "state.sideAPlayers = []" in helper
    assert "state.sideBPlayers = []" in helper

    selector_handler = source[source.index('bindOnce(selector, "teamSelectorChange"') :]
    selector_handler = selector_handler[: selector_handler.index("\n    });")]
    assert "syncRosterFilterViewer(selectedRosterId)" in selector_handler


def test_initial_team_selection_also_syncs_async_roster_filter():
    source = APP_JS.read_text(encoding="utf-8")
    team_loader = source[source.index("function bindTeamSelector()") :]
    team_loader = team_loader[: team_loader.index("function updateAnalyzeButtonState()")]

    assert "syncRosterFilterViewer(selector.value)" in team_loader
    assert "getTradePlatform()" in team_loader
    assert "getTradeSeason()" in team_loader
    assert "&platform=sleeper" not in team_loader


def test_roster_filter_fetch_uses_league_platform_and_season():
    source = APP_JS.read_text(encoding="utf-8")
    loader = source[source.index("async function initRosterFilter()") :]
    loader = loader[: loader.index("\n  function setupSearch(")]

    assert "getTradePlatform()" in loader
    assert "getTradeSeason()" in loader
    assert "/api/league-rosters?" in loader
    assert "&platform=sleeper" not in loader
    # Rosters can finish after #teamSelect is already chosen.
    assert "syncRosterFilterViewer(teamSel?.value" in loader


def test_search_dropdown_ranks_by_scoring_type_value():
    source = APP_JS.read_text(encoding="utf-8")
    search = source[source.index("function setupSearch(side)") :]
    search = search[: search.index("\n  function bindPickButtons(")]
    assert "const valueOf = getPlayerValue;" in search
    assert "p.sf_value || p.value" not in search


def test_roster_filter_forces_team_one_as_viewer_side():
    source = APP_JS.read_text(encoding="utf-8")
    analyzer = source[source.index("async function analyzeTrade()") :]
    analyzer = analyzer[: analyzer.index("const payload = {")]

    assert "const viewerSide = rosterFilterActive()" in analyzer
    assert '? "a"' in analyzer
