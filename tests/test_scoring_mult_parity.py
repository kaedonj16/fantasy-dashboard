"""Parity guard: the scoring-format position multipliers are duplicated in the
browser (static/app.js `SCORING_MULTS`, used by getPlayerValue) and the server
(app.py `_SCORING_MULTS`, used by build_side). If they drift, the trade
calculator's live preview values diverge from the submitted server totals - the
same class of bug that made draft grades disagree. This parses both tables from
source and fails if they aren't identical.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _parse_table(text: str, start_marker: str) -> dict:
    i = text.index(start_marker)
    # Grab the outer { ... } block after the marker.
    brace = text.index("{", i)
    depth = 0
    end = brace
    for j in range(brace, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    block = text[brace:end + 1]
    out: dict = {}
    # Each format row:  fmt: { QB: 1.00, RB: ... }
    for fmt, inner in re.findall(r'["\']?(\w+)["\']?\s*:\s*\{([^}]*)\}', block):
        row = {}
        for pos, num in re.findall(r'["\']?(\w+)["\']?\s*:\s*([\d.]+)', inner):
            row[pos] = float(num)
        out[fmt] = row
    return out


def test_scoring_mults_match():
    js = _parse_table((REPO / "static" / "app.js").read_text(), "SCORING_MULTS = {")
    py = _parse_table((REPO / "app.py").read_text(), "_SCORING_MULTS = {")

    assert js, "failed to parse SCORING_MULTS from app.js"
    assert py, "failed to parse _SCORING_MULTS from app.py"
    assert set(js) == set(py), f"scoring formats differ: js={set(js)} py={set(py)}"
    for fmt in js:
        assert js[fmt] == py[fmt], (
            f"multipliers for '{fmt}' drifted: js={js[fmt]} py={py[fmt]}"
        )
