"""GUI Fix 3 (sticky mobile table columns): audit lock-ins + the one gap.

The 2026-09-25 sticky-columns pass (commit 9bd18fa2) and the 2026-10-01
audit already froze the identifying column on every wide table in the
original inventory: NFL Teams rankings + depth/box-score, Advanced
Metrics main table, dynasty trade values + rankings, history standings,
Keeper, Front Office Report roster, Commissioner Team Overview, public
O-Line rankings, playoff odds, the three Draft Room Deep Dive ledgers,
and the ScoreZone (Redzone) box-score sheet. These tests lock those
contracts in one place so a future refactor cannot silently drop one.

The one genuinely missing surface was the Standings value/production
shares table (.standings-shares-table): it renders inside the same
.st-tblscroll horizontal scroller as the main standings table and its
Team column is nowrap, but it had no frozen pane. On phones the Team
column now sticks at left:0 (the # rank scrolls underneath, mirroring
.history-table / #dvtTable), mobile only - desktop rendering is
untouched because the freeze lives inside a max-width:768px block.

The Start/Sit MAIN board is intentionally not covered: it renders as
stacked .wv-cx-row cards (waivers_page.py wvRenderStartSit), not a
horizontally scrolling table, so there is no column to freeze.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


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


def _media_blocks(css, needle):
    """Yield @media blocks whose prelude contains needle (brace-matched)."""
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
        if needle in css[idx:brace]:
            yield css[idx : j + 1]
        i = j + 1


# ── The gap this PR builds: standings shares Team column (mobile) ──────


def test_shares_team_column_sticky_on_mobile():
    blocks = [
        b
        for b in _media_blocks(DASHBOARD_CSS, "max-width: 768px")
        if "standings-shares-team" in b
    ]
    assert blocks, "no <=768px media block freezes the shares Team column"
    block = "\n".join(blocks)
    assert ".standings-shares-table thead th.standings-shares-team" in block
    assert ".standings-shares-table tbody td.standings-shares-team" in block
    assert "position: sticky" in block
    assert "left: 0" in block
    # Opaque card background so scrolled cells never show through, plus a
    # divider edge, mirroring the .history-table freeze.
    assert "background: var(--card)" in block
    assert "border-right: 1px solid var(--border)" in block
    # Header cell stacks above the frozen body cells.
    assert "z-index: 3" in block


def test_shares_freeze_is_mobile_only():
    # Desktop rendering must not change: no shares sticky rule may live
    # outside a media query.
    base = _strip_media_queries(DASHBOARD_CSS)
    assert "th.standings-shares-team" not in base
    assert "td.standings-shares-team" not in base


def test_shares_markup_carries_team_class():
    # Header and body Team cells both carry the class the CSS keys off.
    assert '<th class="standings-shares-team"' in APP_PY
    assert '<td class="standings-shares-team">' in APP_PY
    # And the table still renders inside the horizontal scroller.
    assert '<div class="st-tblscroll">' in APP_PY
    assert '<table class="standings-shares-table"' in APP_PY


# ── Audit lock-ins: inventory surfaces already frozen before this PR ───


def test_audit_nfl_teams_tables_sticky():
    src = _read("dashboard_services/pages/nfl_teams_page.py")
    flat = src.replace(" ", "")
    # Rankings table: Team column sticks.
    assert "th.nt-teamcol{{position:sticky;left:0" in flat
    assert "td.nt-teamcol{{position:sticky;left:0" in flat
    # Depth chart + box score (both render as table.nt-depth in .nt-tscroll):
    # first (Player) column sticks.
    assert "table.nt-depth>tbody>tr>td:first-child{{position:sticky" in flat


def test_audit_advanced_metrics_main_table_sticky():
    src = _read("dashboard_services/pages/advanced_metrics_page.py")
    assert ".am-table thead th.am-player, .am-table tbody td.am-player" in src
    block = src.split(".am-table thead th.am-player, .am-table tbody td.am-player")[1].split("}")[0]
    assert "position:sticky" in block.replace(" ", "")
    assert "left:0" in block.replace(" ", "")


def test_audit_draft_room_deep_dive_tables_sticky():
    src = _read("dashboard_services/pages/draft_room_page.py")
    for sel in (
        '#drDdLedger thead th[data-k="name"]',
        ".dd-ledger.dd-league tbody td.dd-plname",
        ".dd-ledger.dd-hist-table tbody td.dd-plname",
    ):
        assert sel in src, sel
    assert "position:sticky; left:0" in src


def test_audit_commissioner_and_oline_sticky():
    comm = _read("dashboard_services/pages/commissioner_page.py")
    assert ".comm-team-table thead th:first-child" in comm
    assert "position:sticky; left:0" in comm.replace("  ", " ")
    pub = _read("routes/public_bp.py")
    assert ".pub-oline-table thead th:nth-child(2)" in pub
    assert "position:sticky;left:0" in pub.replace(" ", "")


def test_audit_dashboard_css_sticky_block_intact():
    # Dynasty trade values + rankings, history standings, Front Office
    # roster, playoff odds, and the ScoreZone box-score sheet all freeze
    # from the site-wide block appended to dashboard.css.
    for sel in (
        "#dvtTable td.dvt-name-cell",
        ".rnk-table td.rnk-name-cell",
        ".history-table td.hist-team",
        ".for-table td.for-td-name",
        "table.po-table td.po-team",
        "table.rz-bs-table td.rz-bs-pname",
    ):
        assert sel in DASHBOARD_CSS, sel
