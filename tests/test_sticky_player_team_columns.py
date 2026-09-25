"""Source contracts for sticky identifying columns (team/player).

The Player/Team column stays pinned at left:0 during horizontal scroll so
the row stays identifiable. Only the identifying column sticks; any Rank
column scrolls underneath it. CSS for pages whose styles live in the
shared static/dashboard.css is asserted against the stylesheet directly.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
DYNASTY_PY = (ROOT / "dashboard_services" / "pages" / "dynasty_pages.py").read_text(encoding="utf-8")
HISTORY_PY = (ROOT / "dashboard_services" / "pages" / "history_page.py").read_text(encoding="utf-8")


def _css(name):
    return DASHBOARD_CSS


def test_dvt_player_column_is_targetable():
    # Trade-values table: rank th has .dvt-rank, player cells have
    # .dvt-name-cell; the Player header needs its own class to stick.
    assert '<th class="dvt-player">Player</th>' in DYNASTY_PY
    assert 'class="dvt-name-cell"' in DYNASTY_PY
    assert '<th class="dvt-rank">#' in DYNASTY_PY


def test_rnk_player_column_is_targetable():
    # Rankings variant: same shape with rnk- classes.
    assert '<th class="rnk-player">Player</th>' in DYNASTY_PY
    assert 'class="rnk-name-cell"' in DYNASTY_PY
    assert '<th class="rnk-rank">#' in DYNASTY_PY


def test_history_team_column_is_targetable():
    # Standings table: Team is the 2nd column, needs a class on th and td.
    assert '<th class="hist-team">Team</th>' in HISTORY_PY
    assert '<td class="hist-team">' in HISTORY_PY


def test_dynasty_sticky_css():
    css = _css("dynasty.css")
    # Both dynasty tables pin their player column at left:0 ...
    assert "#dvtTable td.dvt-name-cell" in css
    assert ".rnk-table td.rnk-name-cell" in css
    assert "#dvtTable th.dvt-player" in css
    assert ".rnk-table th.rnk-player" in css
    assert "position: sticky" in css
    assert "left: 0" in css
    # ... with an opaque theme-aware background (rows are transparent on
    # the page background) ...
    assert "background: var(--bg)" in css
    # ... and the translucent row-hover overlay replicated on the cell ...
    assert "tr:hover td.dvt-name-cell" in css
    assert "tr:hover td.rnk-name-cell" in css
    assert "rgba(255, 255, 255, .03)" in css
    # ... plus a separator edge so scrolling cells slide under cleanly.
    assert "border-right: 1px solid var(--border)" in css


def test_history_sticky_css():
    css = _css("history.css")
    assert ".history-table td.hist-team" in css
    assert ".history-table thead th.hist-team" in css
    assert "position: sticky" in css
    assert "left: 0" in css
    # Body cells replicate the table's var(--card) row background; the
    # transparent header row sits on the card (var(--card-soft)).
    assert "background: var(--card)" in css
    assert "background: var(--card-soft)" in css
    # Hover state matches the table's #f8fbff row hover.
    assert "tr:hover td.hist-team" in css
    assert "#f8fbff" in css
    assert "border-right: 1px solid var(--border)" in css
