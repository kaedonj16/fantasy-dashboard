"""Player-modal Team panel navigation uses its explicit interactive contract."""
import shutil
import subprocess
from pathlib import Path

import pytest


SRC = (Path(__file__).resolve().parents[1] / "static" / "player_modal.js").read_text()
APP_SRC = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()


def _wire():
    return SRC.split("function _pmWireTeamPanel(", 1)[1].split("function pmPrefetchTabs", 1)[0]


def test_click_and_keyboard_share_narrow_player_target_helper():
    wire = _wire()
    assert "function (target)" in wire
    assert "closest('[data-pid][role=\"button\"]')" in wire
    assert "const nav = findPlayerNavTarget(target)" in wire
    assert "const row=findPlayerNavTarget(e.target)" in wire
    assert "closest('[data-pid]')" not in wire


def test_metadata_pid_blocks_remain_non_navigation_metadata():
    assert 'id="pmAdpBlock" data-pid=' in SRC
    assert 'id="pmWeeklyTrendsWrap" data-position=' in SRC
    wire = _wire()
    # Exactly one open call lives in the one navigation helper; ordinary Team
    # controls continue through their existing branches.
    assert wire.count("openTeammate(nav)") == 1
    assert wire.count("openTeammate(row)") == 1
    for control in (".pm-team-adv-toggle", ".pm-team-sched-toggle",
                    ".pm-schedule-toggle", ".pm-boxscore-team-pill"):
        assert control in wire


def test_global_player_delegate_does_not_treat_modal_overlay_as_trigger():
    """The overlay's data-player-id owns async state; it must not make the
    entire player-modal body one giant delegated player link."""
    helper = APP_SRC.split("function _globalPlayerModalTrigger(", 1)[1].split(
        "function initGlobalPlayerModals", 1
    )[0]
    init = APP_SRC.split("function initGlobalPlayerModals()", 1)[1].split(
        "// Initialize global player modal", 1
    )[0]
    assert "closest('[data-player-id]')" in helper
    assert "!trigger.classList.contains('player-modal-overlay')" in helper
    assert "const target = _globalPlayerModalTrigger(e.target)" in init
    assert "const target = e.target.closest('[data-player-id]')" not in init
    assert "const trigger = _globalPlayerModalTrigger(el)" in init


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_global_trigger_behavior_for_modal_body_and_nested_player():
    helper = "function _globalPlayerModalTrigger(" + APP_SRC.split(
        "function _globalPlayerModalTrigger(", 1
    )[1].split("function initGlobalPlayerModals", 1)[0]
    script = helper + r"""
const overlay = { classList: { contains: c => c === 'player-modal-overlay' } };
const player = { classList: { contains: () => false }, dataset: { playerId: '456' } };
const bodyClick = { closest: selector => selector === '[data-player-id]' ? overlay : null };
const nestedPlayerClick = { closest: selector => selector === '[data-player-id]' ? player : null };
if (_globalPlayerModalTrigger(bodyClick) !== null) process.exit(1);
if (_globalPlayerModalTrigger(nestedPlayerClick) !== player) process.exit(2);
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
