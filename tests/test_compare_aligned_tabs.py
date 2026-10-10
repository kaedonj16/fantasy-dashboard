"""Compare alignment (2026-10-09): the /compare page and the compare modal render
the same Overview / Start-Sit / Stats tabs: names in the center column with
every value centered, and full game logs on both surfaces."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_overview_uses_centered_table_on_both_surfaces():
    assert "function _cmpOverviewTable2(p1, p2)" in APP_JS
    body = APP_JS[APP_JS.index("function _compareBodyHTML(p1, p2, opts)"):]
    assert "_cmpOverviewTable2(p1, p2)" in body
    # Names in the middle column, values centered.
    assert "mid-names" in APP_JS
    assert "mid-val" in APP_JS


def test_startsit_two_player_uses_middle_column():
    assert "function _ssMidRow(label, cells, dir)" in APP_JS
    fn = APP_JS[APP_JS.index("function _buildStartSitTabHTML"):]
    fn = fn[: fn.index("// One <style> block for the Start/Sit tab")]
    assert "players.length === 2" in fn
    assert "_ssMidRow" in fn


def test_modal_stats_shows_full_game_logs():
    # The old modal-only summary strip exception is gone.
    assert "summaryOnly: false" in APP_JS
    assert "compare modal's Stats tab shows the season summary strip only" not in APP_JS


def test_mid_table_css_shared():
    for cls in ("table.mid-table", ".mid-col", ".mid-names", ".mid-score", ".mid-lbl", ".mid-val", ".mid-best", ".pv-strip"):
        assert cls in CSS, cls
