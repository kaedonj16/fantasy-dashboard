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


def test_roster_filter_forces_team_one_as_viewer_side():
    source = APP_JS.read_text(encoding="utf-8")
    analyzer = source[source.index("async function analyzeTrade()") :]
    analyzer = analyzer[: analyzer.index("const payload = {")]

    assert "const viewerSide = rosterFilterActive()" in analyzer
    assert '? "a"' in analyzer
