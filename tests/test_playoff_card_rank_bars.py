"""Playoff Schedule bars encode the matchup RANK, not pts allowed.

Kaedon (2026-10-13): with bars scaled to pts allowed, two weeks at the same
20.6 pts allowed drew identical long bars even though their ranks were #26
and #24, while the easiest week (#3) read as only slightly longer. The bar
now fills by rank on the 1..32 scale: #1 fills the track, #32 leaves a
sliver. Pts allowed remains the text figure next to the rank.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")


def _playoff_block() -> str:
    # The card renderer is a builder now (deferred patch-in reuses it);
    # the rank-bar math lives inside it, ahead of the hero marker.
    start = JS.index("const _buildPlayoffCard = function(_ps) {")
    end = JS.index("const _h0 = _heroActive();", start)
    return JS[start:end]


def test_bar_width_comes_from_rank_not_pts():
    block = _playoff_block()
    # The pts-based scale is gone from the renderer.
    assert "_psMax" not in block
    assert "fpts_max" not in block
    assert "Number(w.opp_fpts) / " not in block
    # Rank scale: clamp to 1..32, fill = (33 - rank) / 32, 4% floor.
    assert "Math.max(1, Math.min(32, Number(w.opp_rank)))" in block
    assert "100 * (33 - _rk) / 32" in block
    # The bar renders whenever a rank exists; pts alone no longer draws one.
    assert "if (w.opp_rank != null && isFinite(Number(w.opp_rank)))" in block
    # Tier colors stay rank-based, matching the widths.
    assert "(_rk <= 10) ? ' pm-playoff-fill-easy'" in block
    assert "(_rk > 20) ? ' pm-playoff-fill-hard'" in block
    # Pts allowed still shows as the text figure.
    assert "' pts allowed'" in block


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_rank_bar_widths_for_screenshot_weeks_node():
    """Behavioral: the exact _rk/_pct lines from the renderer, run in node
    for the weeks from Kaedon's screenshot (RB card: #26, #3, #24)."""
    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    block = _playoff_block()
    rk_line = next(l.strip() for l in block.splitlines() if "const _rk =" in l)
    pct_line = next(l.strip() for l in block.splitlines() if "const _pct =" in l)
    harness = (
        "function pct(rank) { var w = { opp_rank: rank }; "
        + rk_line.replace("const _rk", "var _rk")
        + " "
        + pct_line.replace("const _pct", "var _pct")
        + " return _pct; }\n"
        "var cases = [[3, 94], [24, 28], [26, 22], [1, 100], [32, 4], [10, 72], [20, 41]];\n"
        "for (var i = 0; i < cases.length; i++) {\n"
        "  if (pct(cases[i][0]) !== cases[i][1]) { console.error('rank ' + cases[i][0] + ' -> ' + pct(cases[i][0])); process.exit(3); }\n"
        "}\n"
        "process.exit(0);\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "rank_bar_check.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=15)
        assert res.returncode == 0, (res.stderr or res.stdout)
