"""Regression coverage for how Redzone alerts set their `mine` / `opp` flags.

Every alert, feed row, and history entry in redzone.js keys off the roster tag
sets produced by `_rosterTags` (`mine: tags.my.has(rid)`, `opp: tags.opp.has(rid)`).
This drives the REAL shipped `_rosterTags` under Node (tests/redzone_alert_tags_harness.mjs)
across the four player classes: a rostered player, a non-rostered player, an
actual opponent, and a non-opponent. A source-level check also pins the call-site
contract so a rename can't silently drop the tagging.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RZ = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")


def test_alert_tag_harness_passes():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS alert-tag harness")
    harness = ROOT / "tests" / "redzone_alert_tags_harness.mjs"
    result = subprocess.run(
        [node, str(harness)], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL REDZONE ALERT TAG HARNESS CHECKS PASSED" in result.stdout


def test_roster_tags_drives_mine_and_opp_flags():
    # The flags are derived from the tag sets, not stored per event upstream.
    assert "function _rosterTags(" in RZ
    assert "mine: tags.my.has(rid)" in RZ
    assert "opp: tags.opp.has(rid)" in RZ
