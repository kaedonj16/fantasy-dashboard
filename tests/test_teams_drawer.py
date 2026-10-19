"""Guards for the team detail drawer (Mock 4 rework).

The drawer is a fixed right slide-over: scrim + aside shell are server-rendered,
per-team payload ships as JSON, and teams.js opens/closes it (button, card
click, scrim click, ESC). Roster intel loads lazily for the viewer's team only;
other teams get the positional breakdown plus a limitation note.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "teams.js").read_text(encoding="utf-8")
TEAMS_PAGE = (ROOT / "dashboard_services" / "pages" / "teams_page.py").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_drawer_shell_and_payload_are_server_rendered():
    assert 'id="teamDrawerScrim"' in TEAMS_PAGE
    assert 'id="teamDrawer"' in TEAMS_PAGE
    assert 'role="dialog"' in TEAMS_PAGE
    assert 'id="teamsDrawerData"' in TEAMS_PAGE
    assert "_drawer_data[str(rid)]" in TEAMS_PAGE
    # Payload carries everything the drawer renders without a fetch.
    for key in ('"positions"', '"grade"', '"window"', '"win_color"', '"pos_index"', '"initials"'):
        assert key in TEAMS_PAGE, f"drawer payload missing {key}"


def test_drawer_open_close_wiring():
    for fn in ("openTeamDrawer", "closeTeamDrawer", "wireTeamDrawer", "_tdLoadIntel", "_tdBodyHtml", "_tdHeadHtml"):
        assert "function %s(" % fn in JS, f"missing drawer fn {fn}"
    wire = JS[JS.index("function wireTeamDrawer("):]
    assert "teamDrawerScrim" in wire
    assert "keydown" in wire
    assert "Escape" in wire
    assert ".team-strength-card" in wire


def test_drawer_renders_position_accordions_with_player_rows():
    body = JS[JS.index("function _tdBodyHtml("):]
    assert "POSITIONAL BREAKDOWN" in body
    assert "td-pos" in body
    assert "players_html" in body
    assert "ROSTER INTEL" in body


def test_drawer_intel_only_for_viewer_team():
    load = JS[JS.index("function _tdLoadIntel("):]
    assert "is_viewer" in load
    assert "only computed for your team" in load
    assert "_riFetchData()" in load
    assert "_riSuggestedMoves(vt)" in load
    assert "_tdInjectIntelTags(vt)" in load


def test_drawer_tag_classes_cover_all_signals():
    tag = JS[JS.index("function _riTagClass("):]
    for sig, cls in (
        ("Core", "td-t-core"),
        ("Hold", "td-t-hold"),
        ("Sell High", "td-t-sell"),
        ("Stash", "td-t-stash"),
        ("Sleeper", "td-t-sleeper"),
        ("Breakout", "td-t-breakout"),
        ("Monitor", "td-t-monitor"),
        ("Cut", "td-t-cut"),
    ):
        assert "'%s': '%s'" % (sig, cls) in tag, f"missing tag for {sig}"
    for cls in ("td-t-core", "td-t-hold", "td-t-sell", "td-t-stash"):
        assert re.search(re.escape("." + cls) + r"\s*\{", CSS), f"missing CSS for {cls}"


def test_roster_intel_fetch_is_shared_and_league_keyed():
    fetch = JS[JS.index("function _riFetchData("):]
    assert "_syncLeagueCfg()" in fetch
    assert "_riDataKey" in fetch
    assert "viewer_roster_id" in fetch
    # The sidebar loader uses the shared fetcher (no second FC fetch inline).
    load = JS[JS.index("function loadRosterIntel()"):]
    head = load[: load.index("function ", 10)]
    assert "_riFetchData()" in head
    assert "fantasycalc.com" not in head


def test_drawer_window_pill_matches_window_color():
    """The drawer header pill must take the team's window color, not the
    hard-coded Contender green .td-contend used to carry for every team:
    Bubble amber, Long Shot grey, Rebuilding red, etc. (same palette as
    the cards' .grade-window-label rules)."""
    head = JS[JS.index("function _tdHeadHtml("):]
    assert "_TD_WIN_CLS[t.window]" in head
    for label, cls in (
        ("Contend", "td-win-contend"),
        ("Bubble", "td-win-bubble"),
        ("Long Shot", "td-win-long-shot"),
        ("Contender", "td-win-contender"),
        ("Win-Now", "td-win-win-now"),
        ("Aging Contender", "td-win-aging-contender"),
        ("Contender Window", "td-win-contender-window"),
        ("2-3 Year Window", "td-win-2yr"),
        ("Rising", "td-win-rising"),
        ("Holding Pattern", "td-win-holding"),
        ("Retooling", "td-win-retooling"),
        ("Rebuilding", "td-win-rebuilding"),
        ("Full Rebuild", "td-win-full-rebuild"),
    ):
        assert "'%s': '%s'" % (label, cls) in JS, f"missing pill class for {label}"
        assert re.search(re.escape("." + cls) + r"[,\s\{]", CSS), f"missing CSS for {cls}"
    # Spot-check the three colors Kaedon named.
    bubble = re.search(r"\.td-win-bubble[^\{]*\{([^}]*)\}", CSS)
    assert "#f59e0b" in bubble.group(1) and "#d97706" in bubble.group(1)
    long_shot = re.search(r"\.td-win-long-shot[^\{]*\{([^}]*)\}", CSS)
    assert "#94a3b8" in long_shot.group(1) and "#64748b" in long_shot.group(1)
    rebuilding = re.search(r"\.td-win-rebuilding[^\{]*\{([^}]*)\}", CSS)
    assert "#ef4444" in rebuilding.group(1) and "#dc2626" in rebuilding.group(1)
    # Long Shot is an explicit payload color, not just the .get() fallback.
    assert '"Long Shot": "#94a3b8"' in TEAMS_PAGE


def test_no_em_dashes_in_drawer_copy():
    load = JS[JS.index("function _tdLoadIntel("):]
    assert "\u2014" not in load


def test_td_bar_track_is_block_level():
    """The bar track is a <span> in teams.js. Without display:block the
    inline box ignores height/width, collapses to 0x0, and the absolutely
    positioned fill (width % of the track) renders at 0px too - only the
    2px league-average tick stayed visible at the track's left edge."""
    m = re.search(r"\.td-bar\s*\{([^}]*)\}", CSS)
    assert m, "missing .td-bar rule"
    assert "display: block" in m.group(1)
    assert "height: 8px" in m.group(1)


def test_drawer_bar_scale_comes_from_position_totals():
    """fill = total / pos_max and tick = pos_avg / pos_max in teams.js, so
    pos_avg must be the mean of the same raw position totals. Feeding it
    league_pos_avg (mean weighted_pos_strength, a weighted average of top
    players on a far smaller scale) pinned the tick near the left edge."""
    assert "_dw_pos_avg" in TEAMS_PAGE
    assert '"pos_avg": _dw_pos_avg.get(pos, 0.0)' in TEAMS_PAGE
    assert '"pos_avg": round(league_pos_avg.get(pos, 0.0), 1)' not in TEAMS_PAGE
