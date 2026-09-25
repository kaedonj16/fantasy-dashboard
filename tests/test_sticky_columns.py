"""Sticky identifying-column regression tests.

Source-level: the commissioner Team Overview table and the public O-Line
rankings table must carry their sticky-column classes and sticky CSS.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMM_SRC = (ROOT / "dashboard_services/pages/commissioner_page.py").read_text(
    encoding="utf-8"
)
PUB_SRC = (ROOT / "routes/public_bp.py").read_text(encoding="utf-8")


def test_commissioner_team_overview_table_has_sticky_class():
    assert 'class="comm-team-table"' in COMM_SRC


def test_commissioner_team_overview_sticky_css():
    # First (TEAM) column sticks; header above body; opaque card background.
    assert ".comm-team-table thead th:first-child" in COMM_SRC
    assert ".comm-team-table tbody td:first-child" in COMM_SRC
    assert "position:sticky" in COMM_SRC
    assert "left:0" in COMM_SRC.replace(" ", "")
    assert "z-index:3" in COMM_SRC.replace(" ", "")
    assert "background:var(--card)" in COMM_SRC.replace(" ", "")
    assert "border-right:1pxsolidvar(--border)" in COMM_SRC.replace(" ", "")


def test_public_oline_table_has_sticky_class():
    assert 'class="pub-oline-table"' in PUB_SRC


def test_public_oline_team_column_sticky_css():
    # Team is the 2nd column (after #); only it sticks.
    assert ".pub-oline-table thead th:nth-child(2)" in PUB_SRC
    assert ".pub-oline-table tbody td:nth-child(2)" in PUB_SRC
    assert "position:sticky" in PUB_SRC
    assert "left:0" in PUB_SRC.replace(" ", "")
    # Opaque section background so scrolled columns slide underneath.
    assert "background:var(--card-soft)" in PUB_SRC.replace(" ", "")


# ── Playoff odds / Draft Room Deep Dive / Redzone box-score ────────────────
# The playoff-odds and redzone sticky CSS lives in static/dashboard.css
# (merged from staging files). Draft-room CSS lives in draft_room_page.py.
import re

DASHBOARD_CSS = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
# Tail appended for the site-wide sticky columns; pill-radius checks scope here.
_STICKY_TAIL = DASHBOARD_CSS.split("Sticky identifying team/player columns (site-wide)")[-1]


def test_draft_room_sticky_player_team_css():
    raw = (ROOT / "dashboard_services/pages/draft_room_page.py").read_text(
        encoding="utf-8"
    )
    # Normalize any escaped quotes to rendered form before asserting.
    src = raw.replace('\\"', '"')
    # All three ledger tables: pick ledger, league board, historical trends.
    for sel in (
        '#drDdLedger thead th[data-k="name"]',
        "#drDdLedger tbody td.dd-plcell",
        ".dd-ledger.dd-league thead th:nth-child(2)",
        ".dd-ledger.dd-league tbody td.dd-plname",
        ".dd-ledger.dd-hist-table thead th:nth-child(2)",
        ".dd-ledger.dd-hist-table tbody td.dd-plname",
    ):
        assert sel in src, sel
    compact = src.replace(" ", "")
    assert "position:sticky;left:0" in compact
    assert "background:var(--card)" in compact
    assert "border-right:1pxsolidvar(--border)" in compact
    # Row states stay opaque on hover and on the league board "you" row.
    assert ".dd-ledger tbody tr:hover td.dd-plcell" in src
    assert ".dd-ledger.dd-league tbody tr.dd-me td.dd-plname" in src


def test_draft_room_ledger_markup_has_player_team_cells():
    raw = (ROOT / "static/draft_room.js").read_text(encoding="utf-8")
    # Normalize any escaped quotes to rendered form before asserting.
    js = raw.replace('\\"', '"')
    assert '<td class="dd-plcell">' in js  # pick ledger player cell
    assert 'data-k="name"' in js  # pick ledger player header
    assert '<td class="dd-plname">' in js  # league board team / hist player
    assert "dd-tablescroll" in js  # horizontal scroll wrapper


def test_playoff_odds_sticky_css():
    css = DASHBOARD_CSS
    # Desktop scroll container (mobile already gets one via media query).
    assert ".po-wrap" in css and "overflow-x: auto" in css
    # Sticky Team column, opaque, with divider and header stacking.
    assert "table.po-table th.po-team" in css
    assert "table.po-table td.po-team" in css
    assert "position: sticky" in css and "left: 0" in css
    assert "background: var(--card)" in css
    assert "table.po-table thead th.po-team" in css
    # Clickable-row hover stays in sync on the sticky cell.
    assert "table.po-table tr.team-clickable:hover td.po-team" in css
    assert "background: var(--accent-soft)" in css
    # No pill radii anywhere near the new styles.
    assert "999px" not in _STICKY_TAIL


def test_playoff_odds_markup_has_team_column():
    raw = (ROOT / "static/app.js").read_text(encoding="utf-8")
    js = raw.replace('\\"', '"')
    assert '<td class="po-team">' in js
    assert '<th class="po-team">Team</th>' in js
    assert '<div class="po-wrap">' in js


def test_redzone_boxscore_sticky_css():
    css = DASHBOARD_CSS
    # Player header cell has no class; target first-child. Body uses .rz-bs-pname.
    assert "table.rz-bs-table thead th:first-child" in css
    assert "table.rz-bs-table td.rz-bs-pname" in css
    assert "position: sticky" in css and "left: 0" in css
    # Opaque sheet background (var(--rz-card) resolves to var(--card)).
    assert "background: var(--rz-card)" in css
    assert "border-right: 1px solid var(--rz-border)" in css
    assert "999px" not in _STICKY_TAIL


def test_redzone_boxscore_markup_has_player_cells():
    raw = (ROOT / "static/redzone.js").read_text(encoding="utf-8")
    js = raw.replace('\\"', '"')
    assert '<td class="rz-bs-pname">' in js
    assert "rz-bs-tscroll" in js
    css = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
    assert ".rz-bs-tscroll { overflow-x: auto" in css


def test_sticky_css_appendix_is_balanced():
    # The site-wide sticky block appended to dashboard.css must have
    # balanced braces (comments stripped first).
    tail = DASHBOARD_CSS.split("Sticky identifying team/player columns (site-wide)")[-1]
    css = re.sub(r"/\*.*?\*/", "", tail, flags=re.S)
    assert css.count("{") == css.count("}"), (
        css.count("{"),
        css.count("}"),
    )
    # No pill radii in the new sticky styles (Kaedon: no border-radius:999px).
    assert "999px" not in tail
