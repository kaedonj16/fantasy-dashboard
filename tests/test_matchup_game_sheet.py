"""Matchups game sheet: the drive bar opens the game in place.

Tapping a live drive bar on the Weekly Hub carousel opens a bottom sheet
(the rz-bs pattern) over the hub instead of navigating to ScoreZone:

- Top: the real ScoreZone game board via window._rzRenderGameBoard with
  the same options the player modal's Live tab uses.
- Tabs: Stats (default) / Plays, in the Weekly Hub tab grammar.
- Stats: the ScoreZone box score (ported builder, /api/player-team-boxscore
  for both teams). Kaedon's ruling: its section line reads just
  "Player stats" here -- no "fantasy points (PPR)" suffix -- while
  ScoreZone's own box-score sheet keeps its label. The deviation rides on
  the options parameter of scorezone.js's _boxSheetBodyHtml, which
  defaults to current behavior.
- Plays: the tapped game's plays only, derived from the FULL
  scorezone-data payload (pbp_by_game) fetched through the shared
  __brSzLiveCache 'full' variant; derivation behavior is exercised by
  tests/game_sheet_events_harness.mjs against the real helper.
- Footer: a quiet "Open in ScoreZone" text link to scorezoneGameUrl.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _js() -> str:
    return (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _sheet_segment() -> str:
    js = _js()
    start = js.index("// ── Game sheet")
    end = js.index("function paint(slide, state)", start)
    return js[start:end]


def test_sheet_structure_board_tabs_footer():
    seg = _sheet_segment()
    # rz-bs sheet pattern: backdrop, sheet, handle, close button, body.
    assert "'rz-bs-backdrop'" in seg
    assert "'rz-bs-sheet gs-sheet'" in seg
    assert 'class="rz-bs-handle"' in seg
    assert 'data-gs-close="1"' in seg
    # Tabs: Stats default, Plays second.
    assert 'data-gtab="stats"' in seg
    assert 'data-gtab="plays"' in seg
    assert 'data-gpanel="stats"' in seg
    assert 'data-gpanel="plays"' in seg
    # Footer: quiet text link carrying the ScoreZone URL.
    assert 'class="gs-foot"' in seg
    assert "Open in ScoreZone \\u2192" in seg
    assert "var url = scorezoneGameUrl(game);" in seg
    assert '<div class="gs-foot"><a href="' in seg
    assert "esc(url)" in seg
    # Close on Escape and backdrop; body scroll locked while open.
    assert "e.key === 'Escape'" in seg
    assert "backdrop.addEventListener('click', gsCloseSheet)" in seg
    assert "document.body.style.overflow = 'hidden'" in seg


def test_sheet_board_uses_real_game_board_unrestyled():
    seg = _sheet_segment()
    assert (
        "window._rzRenderGameBoard(g, { modal: true, fieldBoard: true, pbpPlays: plays })"
        in seg
    )


def test_sheet_data_paths():
    seg = _sheet_segment()
    # Plays + drive trail: one FULL payload through the shared cache, the
    # player modal's mechanism -- not a hand-rolled endpoint or game id.
    assert "__brSzLiveCache.get('full', 30000, applyFull)" in seg
    assert "data.pbp_by_game[gsSheet.gid]" in seg
    # Stats: the box-score endpoint, keyed by the tapped game's id.
    assert "fetch('/api/player-team-boxscore?game_id=' + encodeURIComponent(gid))" in seg
    # Snapshot on open: no polling loop for the sheet.
    assert "setInterval" not in seg


def test_sheet_failure_states_are_quiet():
    seg = _sheet_segment()
    # Full payload failed -> Plays shows one quiet line, board + Stats stay.
    assert "Plays are unavailable for this game." in seg
    assert "fullStatus === 'loading'" in seg  # cache waiters drop silently; timer fails the tab
    # Box-score failure renders its retry state inside the Stats tab.
    assert "Could not load the box score." in seg


def test_player_stats_label_ruling():
    seg = _sheet_segment()
    # The game sheet's Stats section line is exactly "Player stats".
    assert "'<div class=\"rz-bs-sec\">Player stats</div>'" in seg
    assert "fantasy points (PPR)" not in seg
    assert "rz-bs-note" not in seg
    # ScoreZone's own sheet keeps its label, via the shared builder's
    # default (its render call passes no options).
    sz = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
    assert "function _boxSheetBodyHtml(options)" in sz
    assert "options.showPlayerStatsNote !== false" in sz
    assert "(showNote ? ' <span class=\"rz-bs-note\">\\u00b7 fantasy points (PPR)</span>' : '')" in sz
    assert "_boxSheetEls.body.innerHTML = _boxSheetBodyHtml();" in sz


def test_game_sheet_events_harness_passes():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the game sheet events harness")
    harness = ROOT / "tests" / "game_sheet_events_harness.mjs"
    result = subprocess.run(
        [node, str(harness)], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL GAME SHEET EVENT HARNESS CHECKS PASSED" in result.stdout


def test_game_sheet_css_block():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    start = css.index("Matchups game sheet (drive-bar sheet")
    end = css.index("MERGED: icons.css", start)
    block = css[start:end]
    # The new CSS lives before the icons merge marker, per repo convention.
    for sel in (".gs-sheet", ".gs-tabs .tab-bar", ".gs-tabs .tab-btn.active",
                ".gs-plays", ".gs-foot", ".gs-foot a", ".gs-hs"):
        assert sel in block, sel
    # Footer link is the quiet link blue, not a filled button.
    foot = re.search(r"\.gs-foot a \{(.*?)\}", block, flags=re.S)
    assert foot and "color: var(--brand-blue)" in foot.group(1)
    assert "border-radius: 999px" not in block
