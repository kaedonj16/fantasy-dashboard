"""The player modal's Redzone tab builds a game log off the Redzone page too.

Regression: everywhere except /redzone the modal passed an empty event feed, so
the game log was always "No plays recorded yet" even when the player had plays.
The behavioral assertions run in Node against the real shipped helper.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_off_page_player_log_harness_passes():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS player-log harness")
    harness = ROOT / "tests" / "redzone_player_log_harness.mjs"
    result = subprocess.run(
        [node, str(harness)], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL REDZONE PLAYER LOG HARNESS CHECKS PASSED" in result.stdout
