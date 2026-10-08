"""Source contracts for the 2026-10-01 sticky-column audit fixes.

Tables that scroll horizontally must keep their identifying column(s)
frozen: the player-modal game log (Date + Opp at every width, not just
inside the <=768px media query), the dynasty value standings (Rank +
Team), the portfolio holdings table (Pos + Player), the Advanced
Metrics pinned-player compare (Metric), the Start/Sit and 3-way compare
tables (row labels), and the app.py All-Time standings (Team, via the
hist-team classes its history_page sibling already used).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
AM_PAGE = (ROOT / "dashboard_services" / "pages" / "advanced_metrics_page.py").read_text(
    encoding="utf-8"
)


def _strip_media_queries(css):
    """Return the CSS with every @media block removed (brace-matched)."""
    out = []
    i = 0
    while True:
        idx = css.find("@media", i)
        if idx == -1:
            out.append(css[i:])
            break
        out.append(css[i:idx])
        brace = css.index("{", idx)
        depth = 0
        j = brace
        while True:
            if css[j] == "{":
                depth += 1
            elif css[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        i = j + 1
    return "".join(out)


def _media_blocks(css):
    """Yield each @media block's full text (brace-matched), in order."""
    i = 0
    while True:
        idx = css.find("@media", i)
        if idx == -1:
            return
        brace = css.index("{", idx)
        depth = 0
        j = brace
        while True:
            if css[j] == "{":
                depth += 1
            elif css[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        yield css[idx : j + 1]
        i = j + 1


# ── Player modal game log: Date + Opp frozen at ALL widths ──────────────


def test_game_log_freeze_lives_outside_media_queries():
    # The original bug: the frozen pane existed only inside
    # @media (max-width: 768px), so desktop scrolling clipped Date/Opp.
    base = _strip_media_queries(DASHBOARD_CSS)
    # Header cells are row-scoped: tr.grp's first cell spans both frozen
    # columns; tr.cols carries the Date (1) and Opp (2) headers.
    assert ".game-log-table thead tr.grp th:first-child" in base
    assert ".game-log-table thead tr.cols th:nth-child(1)" in base
    assert ".game-log-table thead tr.cols th:nth-child(2)" in base
    assert ".game-log-table tbody td:nth-child(1)" in base
    assert ".game-log-table tbody td:nth-child(2)" in base
    assert ".game-log-table tfoot td:nth-child(2)" in base
    # Fixed offsets: Date at 0, Opp one 48px Date column in.
    assert "left: 0;" in base
    assert "left: 48px;" in base
    # Opaque backgrounds per row state, so scrolled cells never show
    # through the frozen pane.
    assert ".game-log-table tbody tr:nth-child(even) td:nth-child(-n+2)" in base
    assert ".game-log-table tbody tr.game-log-bye td:nth-child(-n+2)" in base


def test_game_log_mobile_keeps_third_frozen_column():
    # The <=768px block still freezes Pts as the third column.
    blocks = [
        b
        for b in _media_blocks(DASHBOARD_CSS)
        if "max-width: 768px" in b.split("{", 1)[0] and "game-log-table" in b
    ]
    assert blocks, "no <=768px media block touches .game-log-table"
    block = "\n".join(blocks)
    assert ".game-log-table th:nth-child(3)" in block
    assert ".game-log-table td:nth-child(3)" in block
    assert "left: 94px" in block


# ── Dynasty value standings + portfolio holdings ────────────────────────


def test_dynasty_value_standings_freeze_rank_and_team():
    css = DASHBOARD_CSS
    assert ".standings-table.dynasty-table thead th:nth-child(1)" in css
    assert ".standings-table.dynasty-table tbody td:nth-child(2)" in css
    assert "left: 52px;" in css
    # Translucent row tints get opaque twins on the frozen cells.
    assert ".standings-table.dynasty-table tbody tr.is-top3 td:nth-child(-n+2)" in css
    assert "color-mix(in srgb, var(--accent) 4%, var(--card))" in css


def test_portfolio_holdings_freeze_pos_and_player():
    css = DASHBOARD_CSS
    assert "#pfTable thead th:nth-child(1)" in css
    assert "#pfTable tbody td:nth-child(2)" in css
    assert "left: 46px;" in css


# ── All-Time (career) standings in app.py ───────────────────────────────


def test_career_standings_team_column_uses_hist_team():
    # The awards rework uses .awards-standings-table (a compact 5-column table
    # that fits without horizontal scroll, so no sticky column needed).
    assert 'class="awards-standings-table"' in APP_PY
    assert "<th>OWNER</th>" in APP_PY
    assert ".awards-standings-table" in DASHBOARD_CSS


# ── Compare tables: row labels frozen ───────────────────────────────────


def test_am_cmp_table_metric_column_sticky():
    assert ".am-cmp-table thead th:first-child" in AM_PAGE
    assert ".am-cmp-table td.am-cmp-metric" in AM_PAGE
    assert "position:sticky; left:0" in AM_PAGE
    # The metric cells the CSS targets are what the renderer emits.
    assert '<td class="am-cmp-metric">' in AM_PAGE


def test_start_sit_table_row_labels_sticky():
    assert '.ss-tbl th.ss-rowlbl{position:sticky;left:0' in APP_JS
    assert '.ss-tbl thead th.ss-rowlbl{z-index:3;}' in APP_JS
    assert '<th class="ss-rowlbl">' in APP_JS


def test_cmp3_table_row_labels_sticky():
    # #2380 relocated these rules from the inline <style> block in app.js
    # into dashboard.css; the contract is that the sticky rules still ship.
    assert ".cmp3-table th.cmp3-rowlbl {" in DASHBOARD_CSS
    body = DASHBOARD_CSS.split(".cmp3-table th.cmp3-rowlbl {")[1].split("}")[0]
    assert "position: sticky" in body
    assert "left: 0" in body
    assert ".cmp3-table thead th.cmp3-rowlbl {" in DASHBOARD_CSS
    thead = DASHBOARD_CSS.split(".cmp3-table thead th.cmp3-rowlbl {")[1].split("}")[0]
    assert "z-index: 3" in thead
    assert '<th class="cmp3-rowlbl">' in APP_JS
