"""Playoff Schedule panel: header + easiest note + stacked rows (declutter).

History: the per-week bars were Kaedon's 2026-10-13 call (bar fill encodes
the matchup RANK on the 1..32 scale, never pts allowed, after two equal-pts
weeks drew identical bars at different ranks). The 2026-11-03 declutter
supersedes that rendering: the panel is a plain rail panel whose header
carries the weeks range with the aggregate as a muted note ("7th easiest
for WRs"), and each week is one stacked row -- week + opponent left,
"#rank · pts allowed" bold right. The payload and every value are
unchanged; only the visual devices (bars, big rank block) are gone.
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
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _playoff_block() -> str:
    # The card renderer is a builder (deferred patch-in reuses it).
    start = JS.index("const _buildPlayoffCard = function(_ps) {")
    end = JS.index("const _h0 = _heroActive();", start)
    return JS[start:end]


def test_rows_carry_rank_and_pts_no_bars():
    block = _playoff_block()
    # The bar machinery is gone from the renderer.
    assert "pm-playoff-track" not in block
    assert "pm-playoff-fill" not in block
    assert "_pct" not in block
    assert "33 - _rk" not in block
    # The data still renders as text: tier-colored rank + pts allowed.
    assert 'class="pm-rk pm-rk-${_rkTier(w.opp_rank)}"' in block
    assert "' pts allowed'" in block
    # A week with no opponent still renders as its own row.
    assert 'class="pm-playoff-row"' in block


def test_header_is_weeks_label_plus_muted_note():
    block = _playoff_block()
    # Weeks come from the payload's first/last rows, not a hardcoded range.
    assert "Playoff schedule · Weeks ${_psFirst.week}-${_psLast.week}" in block
    # The aggregate is a muted note, not the old big tier-colored block.
    assert 'class="pm-playoff-note"' in block
    assert "${_pmTeamOrd(Number(_ps.rank))} easiest" in block
    assert "pm-playoff-rank-big" not in block
    assert ".pm-playoff-note {" in CSS
    # The dead bar/rank-block styles are gone from the stylesheet.
    assert ".pm-playoff-track" not in CSS
    assert ".pm-playoff-rank-big" not in CSS


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node skipped")
def test_playoff_panel_renders_rows_node():
    """Behavioral: run the real builder in node on the mock's sample weeks
    (W15 at LAR #6 34.6, a no-opponent week, W17 vs NYG #12 34.6) and check
    the panel HTML end to end."""
    if shutil.which("node") is None:
        pytest.skip("Node.js not available")
    block = _playoff_block()
    fn_end = block.index("return playoffCard;")
    builder = block[:block.index("};", fn_end) + 2]
    harness = (
        "var data = { position: 'WR' };\n"
        "function _pmEsc(s) { return String(s == null ? '' : s)"
        ".replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }\n"
        "function _pmTeamOrd(n) { var s = ['th', 'st', 'nd', 'rd'], v = n % 100;"
        " return n + (s[(v - 20) % 10] || s[v] || s[0]); }\n"
        + builder + "\n"
        "var html = _buildPlayoffCard({ rank: 7, weeks: [\n"
        "  { week: 15, opp: 'LAR', at: true, opp_rank: 6, opp_fpts: 34.6 },\n"
        "  { week: 16, opp: null, at: false, opp_rank: null, opp_fpts: null },\n"
        "  { week: 17, opp: 'NYG', at: false, opp_rank: 12, opp_fpts: 34.6 }\n"
        "]});\n"
        "var fails = [];\n"
        "function chk(c, m) { if (!c) fails.push(m); }\n"
        "chk(html.indexOf('Playoff schedule · Weeks 15-17') >= 0, 'header weeks range');\n"
        "chk(html.indexOf('7th easiest for WRs') >= 0, 'easiest note');\n"
        "chk(html.indexOf('Week 15') >= 0 && html.indexOf('at LAR') >= 0, 'week 15 row');\n"
        "chk(html.indexOf('#6') >= 0 && html.indexOf('34.6 pts allowed') >= 0, 'row figures');\n"
        "chk(html.indexOf('pm-rk-easy') >= 0, 'rank tier class');\n"
        "chk(html.indexOf('Week 16') >= 0, 'no-opponent week still a row');\n"
        "chk((html.match(/pm-playoff-row\\\"/g) || []).length === 3, 'three rows');\n"
        "chk(html.indexOf('pm-playoff-track') < 0, 'no bar track');\n"
        "chk(html.indexOf('pm-playoff-rank-big') < 0, 'no big rank block');\n"
        "if (fails.length) { console.error(fails.join('; ')); process.exit(3); }\n"
        "process.exit(0);\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "playoff_panel_check.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True, timeout=15)
        assert res.returncode == 0, (res.stderr or res.stdout)
