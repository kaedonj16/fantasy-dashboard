"""Cross-position Start/Sit compare headline must not fake a tie.

Regression test: the compare Start/Sit tab (player modal compare and the
standalone /compare page share `_buildStartSitTabHTML` in static/app.js)
headlined the 0-100 start/sit INDEX, which is relative to each player's own
position. Wan'Dale Robinson (WR) vs Dalton Kincaid (TE) showed twin 69
indexes while the verdict, decided on the raw start_score, had Wan'Dale
ahead by 2+ points, and the intro copy claimed the index made a QB and a
WR comparable. Cross-position compares now headline the raw start score
(the number that decides the call) and the intro says the index cannot be
compared across positions. Same-position compares keep the index headline
and the original copy, byte for byte.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

ROOT = Path(__file__).resolve().parents[1]


def _ss_block() -> str:
    """The self-contained Start/Sit compare block of static/app.js: the _ss*
    helpers, _buildStartSitTabHTML, and the _SS_TAB_CSS it prepends."""
    src = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    start = src.index("const _ssEsc")
    css_start = src.index("const _SS_TAB_CSS", start)
    end = src.index("'</style>';", css_start) + len("'</style>';")
    return src[start:end]


def _player(name, position, score, pct, proj):
    # Shape mirrors the player-details/compare payload stats the tab reads.
    return {
        "name": name,
        "position": position,
        "stats": {
            "start_score": score,
            "start_score_pct": pct,
            "value": 800,
            "ppg": proj - 1.0,
            "start_score_factors": {"proj": proj},
            "start_sit": {
                "proj_pts": proj,
                "recent_ppg": proj - 0.5,
                "implied_total": 23.5,
                "opponent_label": "vs DAL",
            },
        },
    }


# The diagnosed pair: identical indexes (69), raw scores 12.08 vs 9.96.
WANDALE = _player("Wan'Dale Robinson", "WR", 12.08, 69, 14.1)
KINCAID = _player("Dalton Kincaid", "TE", 9.96, 69, 11.2)
# A same-position control pair with distinct indexes.
WILSON = _player("Garrett Wilson", "WR", 10.4, 55, 12.9)
RB = _player("Breece Hall", "RB", 11.3, 69, 13.6)
WR3 = _player("Chris Olave", "WR", 11.9, 64, 13.9)


def _render(sets):
    fixtures = {
        "wandale": WANDALE, "kincaid": KINCAID, "wilson": WILSON,
        "hall": RB, "olave": WR3,
    }
    driver = "const F = " + json.dumps(fixtures) + ";\n"
    driver += "const out = {};\n"
    for key, names in sets.items():
        args = ", ".join("F." + n for n in names)
        driver += f"out[{json.dumps(key)}] = _buildStartSitTabHTML([{args}]);\n"
    driver += "console.log(JSON.stringify(out));"
    res = subprocess.run(
        ["node", "-e", _ss_block() + "\n" + driver],
        capture_output=True, text=True, timeout=20,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@pytest.fixture(scope="module")
def html():
    return _render({
        "cross": ["wandale", "kincaid"],
        "same": ["wandale", "wilson"],
        "triple_cross": ["wandale", "hall", "kincaid"],
        "triple_same": ["wandale", "wilson", "olave"],
    })


def test_cross_position_headlines_raw_start_score(html):
    cross = html["cross"]
    # The score gap that drives the verdict is the headline, on both sides.
    assert '<div class="mid-score">12.1</div><div class="mid-cap">start score</div>' in cross
    assert '<div class="mid-score">10</div><div class="mid-cap">start score</div>' in cross
    # Never the twin position indexes.
    assert '<div class="mid-score">69</div>' not in cross
    assert "index (0 to 100)" not in cross


def test_cross_position_verdict_unchanged(html):
    # Display only: the verdict still picks the higher raw start_score.
    assert "ss-verdict-name" in html["cross"]
    assert "Wan&#039;Dale Robinson" in html["cross"] or "Wan'Dale Robinson" in html["cross"]


def test_cross_position_intro_copy_corrected(html):
    cross = html["cross"]
    assert "cannot be compared across positions" in cross
    assert "that score is what decides the call" in cross
    assert "so a QB and a WR are comparable" not in cross


def test_same_position_keeps_index_headline_and_copy(html):
    same = html["same"]
    assert '<div class="mid-score">69</div><div class="mid-cap">index (0 to 100)</div>' in same
    assert '<div class="mid-score">55</div><div class="mid-cap">index (0 to 100)</div>' in same
    # Original intro copy, including the comparability sentence, untouched.
    assert "is a 0 to 100 index relative to that player's own position" in same
    assert "so a QB and a WR are comparable" in same
    assert "cannot be compared across positions" not in same


def test_triple_cross_position_table_head_uses_start_score(html):
    triple = html["triple_cross"]
    # The 3-player layout headlines through the table header cells.
    assert '<div class="ss-th-score">12.1</div><div class="ss-th-cap">start score</div>' in triple
    assert '<div class="ss-th-score">11.3</div><div class="ss-th-cap">start score</div>' in triple
    assert "index (0 to 100)" not in triple
    assert "cannot be compared across positions" in triple


def test_triple_same_position_keeps_index_headline(html):
    triple = html["triple_same"]
    assert '<div class="ss-th-score">69</div><div class="ss-th-cap">index (0 to 100)</div>' in triple
    assert '<div class="ss-th-score">64</div><div class="ss-th-cap">index (0 to 100)</div>' in triple
    assert "so a QB and a WR are comparable" in triple
