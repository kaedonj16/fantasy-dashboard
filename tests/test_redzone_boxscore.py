"""Source contracts for the Redzone game box score button + sheet.

The button lives on the filtered-game NFL board (rendered only when the user
is filtered to one NFL game) and opens a bottom sheet with that game's box
score: a team toggle plus full skill-player (QB/RB/WR/TE) stat tables fetched
from /api/player-team-boxscore.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REDZONE_JS = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _board_fn():
    start = REDZONE_JS.index("function _renderNflBoard()")
    # Ends right before the next section comment / function.
    end = REDZONE_JS.index("// Compact centered status for an NFL game pill.")
    return REDZONE_JS[start:end]


def _boxscore_button_fn():
    # The button markup lives in this helper so both board render paths
    # (the shared app.js renderer branch and the redzone.js fallback) stay
    # in sync.
    start = REDZONE_JS.index("function _boxScoreButtonHtml()")
    end = REDZONE_JS.index("function _renderNflBoard()")
    return REDZONE_JS[start:end]


# ── Button placement: filtered-game board only ──────────────────────────
def test_boxscore_button_renders_inside_filtered_game_board():
    board = _board_fn()
    button = _boxscore_button_fn()
    # Unfiltered view renders nothing at all, so the button cannot leak there.
    assert "if (_filters.nfl === 'all') return '';" in board
    assert "data-boxscore-game" in button
    assert "Box score" in button
    # Both render paths attach it: the shared app.js renderer branch (the one
    # that actually runs on the real page) and the redzone.js fallback.
    assert board.count("_boxScoreButtonHtml()") >= 2


def test_boxscore_button_is_not_in_the_shared_board_renderer_itself():
    # The shared window._rzRenderGameBoard (app.js) is also used by the player
    # modal, so the button must not live inside it. The Redzone filtered view
    # appends the button at its own call site instead.
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "boxscore" not in app_js.lower()
    board = _board_fn()
    branch = board.split("if (window._rzRenderGameBoard) {", 1)[1].split("\n    }", 1)[0]
    assert "_boxScoreButtonHtml()" in branch


def test_boxscore_button_carries_the_filtered_game_id():
    button = _boxscore_button_fn()
    assert 'data-boxscore-game="' in button
    # Bound to the single canonical filter, not a second selection state.
    assert "_esc(_filters.nfl)" in button


# ── Click wiring ─────────────────────────────────────────────────────────
def test_board_button_click_opens_the_sheet_via_delegation():
    assert "[data-boxscore-game]" in REDZONE_JS
    assert "_openBoxScore(" in REDZONE_JS


def test_sheet_mounts_outside_rz_root():
    # The board re-renders on every poll; the sheet must survive that.
    # Two appends: the backdrop and the sheet itself.
    assert REDZONE_JS.count("document.body.appendChild") >= 2


# ── Data path ────────────────────────────────────────────────────────────
def test_sheet_fetches_the_shared_boxscore_endpoint():
    assert "/api/player-team-boxscore" in REDZONE_JS
    assert "game_id=" in REDZONE_JS or "game_id" in REDZONE_JS


def test_sheet_covers_loading_error_and_pregame_states():
    src = REDZONE_JS
    assert "Try again" in src  # error + retry
    assert "data-box-retry" in src
    assert "once the game begins" in src  # pregame message passthrough


# ── Team toggle + skill-player tables ─────────────────────────────────────
def test_sheet_has_a_team_toggle():
    assert "data-box-team" in REDZONE_JS
    assert "_BOX_SKILL_POS" in REDZONE_JS


def test_sheet_shows_skill_positions_only():
    m = re.search(r"var _BOX_SKILL_POS = \{([^}]*)\}", REDZONE_JS)
    assert m, "skill position map missing"
    body = m.group(1)
    for pos in ("QB", "RB", "WR", "TE"):
        assert pos in body
    assert '"K"' not in body and "'K'" not in body
    assert "DEF" not in body


def test_sheet_tables_render_from_endpoint_columns():
    # Columns come from the payload (columns[].label), cells from players[].cells.
    assert "gr.columns" in REDZONE_JS
    assert "p.cells" in REDZONE_JS or "(p.cells || {})" in REDZONE_JS


def test_sheet_player_rows_tap_through_to_the_player_modal():
    assert "data-pid" in REDZONE_JS.split("rz-bs-pname")[0][-2000:] or \
        'data-pid="' in REDZONE_JS
    assert "openPlayerModal" in REDZONE_JS


def test_sheet_closes_via_x_backdrop_and_escape():
    assert "data-box-close" in REDZONE_JS
    assert "_closeBoxScore" in REDZONE_JS
    assert "Escape" in REDZONE_JS


# ── CSS ──────────────────────────────────────────────────────────────────
def test_boxscore_css_exists_without_pill_radii():
    for cls in (".rz-boxscore-btn", ".rz-bs-sheet", ".rz-bs-backdrop",
                ".rz-bs-teams", ".rz-bs-team", "table.rz-bs-table"):
        assert cls in DASHBOARD_CSS, cls
    # New styles must not use the banned 999px pill radius.
    start = DASHBOARD_CSS.index("/* ── Box score button + sheet (filtered-game board) ── */")
    end = DASHBOARD_CSS.index("/* ── On-deck strip (games kicking off soon) ── */")
    block = DASHBOARD_CSS[start:end]
    assert "999px" not in block
    assert "border-radius: 12px" in block  # button radius, not a pill


# ── Behavioral harness (real shipped JS, extracted by source) ────────────
def test_frontend_boxscore_harness_passes():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS boxscore harness")
    harness = ROOT / "tests" / "redzone_boxscore_harness.mjs"
    proc = subprocess.run(
        [node, str(harness)], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
