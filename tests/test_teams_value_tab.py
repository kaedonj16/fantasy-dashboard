"""Guards for the Teams-page Value tab markup (Mock 4 ranked rows).

The Value tab renders ranked rows: rank badge (1/2/3 highlighted), team name,
period change, and a vs-avg pill, with the 7d/14d/30d/60d window pills and the
header intact. The viewer's row is highlighted.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "teams.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _fn_block(name: str) -> str:
    needle = "function %s(" % name
    start = JS.index(needle)
    line_start = JS.rfind("\n", 0, start) + 1
    indent = JS[line_start:start]
    rest = JS[start:]
    nxt = rest.find("\n" + indent + "function ", 1)
    return rest if nxt < 0 else rest[:nxt]


def test_value_tab_renders_ranked_rows():
    render = _fn_block("renderBtm")
    assert '"rl-head"' in render
    assert '"rl-cols"' in render
    assert '"rl-row' in render
    assert '"rk ' in render
    assert "'t1'" in render and "'t2'" in render and "'t3'" in render
    assert '"chg ' in render
    assert '"vs"' in render
    assert "'pill-pos'" in render and "'pill-neg'" in render
    assert "'pos-chg'" in render and "'neg-chg'" in render
    assert '"vbar"' in render


def test_value_tab_highlights_viewer_row():
    render = _fn_block("renderBtm")
    assert "rl-mine" in render
    assert "rl-you" in render
    assert "String(r.roster_id) === String(_viewerRosterId)" in render


def test_value_tab_keeps_window_pills_and_header():
    render = _fn_block("renderBtm")
    assert "btm-pill" in render
    for days in ("7", "14", "30", "60"):
        assert 'data-days="%s"' % days in render
    assert "Value Tracker" in render
    assert "Which rosters gained the most dynasty value?" in render
    # Column header tracks the active window.
    assert "c-30d" in render
    assert "+ days + 'D" in render


def test_value_tab_escapes_team_names():
    render = _fn_block("renderBtm")
    assert "_sosEsc(r.team_name)" in render


def test_value_tab_css_selectors_exist():
    for sel in (".rl-head", ".rl-cols", ".rl-row", ".rl-mine", ".rk", ".chg", ".vs", ".vbar", ".pill-pos", ".pill-neg"):
        assert sel in CSS, f"missing Value-tab CSS for {sel}"
    rk = re.search(r"\.rl-row\s+\.rk\.t1\s*\{([^}]+)\}", CSS)
    assert rk, "missing .rk.t1 highlight rule"
    assert "#92400e" in rk.group(1)
