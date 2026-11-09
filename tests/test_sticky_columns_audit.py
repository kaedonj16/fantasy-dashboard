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
PLAYER_MODAL_JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")


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
    # Fixed offsets: Date at 0; Opp sticks at 43px, 1px inside the 44px
    # Date column's edge, so its opaque background covers the
    # collapsed-border seam column (a scrolled column ghosting through
    # that seam was the dotted sliver at the pane edge). The Opp column
    # is 86px wide: code + matchup chip on one line, sized to the
    # longest code + chip instead of a loose round number. Widths are
    # pinned with matching min/max so every engine renders exactly the
    # width the offsets assume, and the grp corner spans the full
    # Date + Opp width (130px), not one column.
    assert "left: 0;" in base
    assert "left: 43px;" in base
    assert "width: 86px;" in base
    assert "min-width: 86px;" in base
    assert "max-width: 44px;" in base
    assert "width: 130px;" in base
    assert "min-width: 130px;" in base
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
    # Header selectors are scoped to tr.cols: unscoped th:nth-child
    # selectors also caught the tr.grp row's group labels (FANTASY /
    # PASSING) and froze them over the pane.
    assert ".game-log-table thead tr.cols th:nth-child(3)" in block
    assert ".game-log-table th:nth-child(3)" not in block
    assert ".game-log-table tbody td:nth-child(3)" in block
    # Pts sticks at 128px: the 44px Date + 86px one-line Opp columns end
    # at 130, minus a 2px overlap so its background covers the seam
    # column at Opp's right edge.
    assert "left: 128px" in block
    assert "min-width: 48px;" in block


def test_game_log_grp_cover_extends_over_pts_segment_when_scrolled():
    # The grp corner spans only Date + Opp (130px) but the mobile pane is
    # three columns: while the table scrolls, the corner's ::after patch
    # covers the Pts segment of the header row so group labels hide from
    # the stuck pane's right edge (177px: the frozen cells stick 2px
    # inside their rest position). The patch must be gated on
    # .gl-scrolled (at rest the first label legitimately sits over that
    # segment), live in the <=768px block (desktop's pane is exactly the
    # corner width), and use the corner's opaque header background. Its
    # right edge carries the pane divider as a hard-stop background line.
    blocks = [
        b
        for b in _media_blocks(DASHBOARD_CSS)
        if "max-width: 768px" in b.split("{", 1)[0] and "game-log-table" in b
    ]
    block = "\n".join(blocks)
    sel = ".table-wrap.gl-scrolled .game-log-table thead tr.grp th:first-child::after"
    assert sel in block
    rule = block[block.index(sel) :]
    rule = rule[: rule.index("}")]
    assert "left: 130px;" in rule
    assert "width: 46px;" in rule
    assert "background: var(--bg-alt);" in rule
    assert "linear-gradient(to left, var(--border) 0 1px, transparent 1px)" in rule
    # The pane divider also rides on the stuck Pts cells as an ::after
    # line: a collapsed border-right is not repainted at the stuck
    # position, so without it the divider vanishes while scrolled.
    assert ".game-log-table tbody td:nth-child(3)::after" in block
    assert ".game-log-table tfoot td:nth-child(3)::after" in block
    # Outside the media block (desktop) no such patch exists.
    assert sel not in _strip_media_queries(DASHBOARD_CSS)
    # The toggle is wired once in player_modal.js: a delegated capture
    # phase scroll listener flipping the class at the scrollLeft 0 edge,
    # plus the clip pass that keeps header labels from straddling the
    # pane edge as glyph debris.
    assert "window._glScrollCoverWired" in PLAYER_MODAL_JS
    assert "classList.toggle('gl-scrolled', w.scrollLeft > 0)" in PLAYER_MODAL_JS
    assert "clipPath" in PLAYER_MODAL_JS


def test_game_log_thead_background_matches_scrolling_header_cells():
    # Every game-log header cell is opaque (frozen cells var(--bg-alt),
    # scrolling cells var(--card) via the global thead th rule), so the
    # thead's own background only ever shows through the seam slot that
    # opens beside the frozen pane while the table scrolls sideways.
    # There it must match the scrolling cells it sits among: bg-alt
    # painted a stray gray band inside the first scrolling header cell
    # (the xFP header at phone widths), appearing only at the scroll
    # offsets where a column boundary crosses the seam.
    base = _strip_media_queries(DASHBOARD_CSS)
    sel = ".game-log-table thead {"
    assert sel in base
    rule = base[base.index(sel) :]
    rule = rule[: rule.index("}")]
    assert "background: var(--card);" in rule
    assert "background: var(--bg-alt);" not in rule


def test_game_log_pts_header_paints_card_not_pane_bg_alt():
    # Pts is a stat column, not a pane label: only the Date/Opp headers
    # keep the frozen pane's var(--bg-alt); the Pts header paints
    # var(--card) like every scrolling header (global thead th rule),
    # at every width. It stays opaque with the frozen headers' z-index
    # because below 768px it is the third frozen header.
    base = _strip_media_queries(DASHBOARD_CSS)
    assert ".game-log-table thead tr.cols th:nth-child(-n+3)" not in base
    sel = ".game-log-table thead tr.cols th:nth-child(-n+2) {"
    assert sel in base
    rule = base[base.index(sel) :]
    rule = rule[: rule.index("}")]
    assert "background: var(--bg-alt);" in rule
    sel = ".game-log-table thead tr.cols th:nth-child(3) {"
    assert sel in base
    rule = base[base.index(sel) :]
    rule = rule[: rule.index("}")]
    assert "background: var(--card);" in rule
    assert "background: var(--bg-alt);" not in rule
    assert "z-index: 3;" in rule


def test_game_log_pts_bye_row_paints_row_in_mobile_block():
    # In the <=768px frozen-pane block, a bye / no-stats row's Pts cell
    # must take the row's var(--row) paint like Date/Opp and the
    # scrolling cells, not stay var(--card) white: the rule needs the
    # tr.game-log-* selector shape to outrank the -n+3 var(--card)
    # default for the third cell.
    blocks = [
        b
        for b in _media_blocks(DASHBOARD_CSS)
        if "max-width: 768px" in b.split("{", 1)[0] and "game-log-table" in b
    ]
    block = "\n".join(blocks)
    for cls in ("game-log-no-stats", "game-log-bye"):
        sel = f".game-log-table tbody tr.{cls} td:nth-child(3)"
        assert sel in block
        rule = block[block.index(sel) :]
        rule = rule[: rule.index("}")]
        assert "background-color: var(--row);" in rule


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
