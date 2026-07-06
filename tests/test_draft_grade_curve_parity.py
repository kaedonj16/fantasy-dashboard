"""Parity guard: the browser draft-grade curve (static/draft_grade_curve.js)
and the Python server mirror (utils.draft_grade.dr_apply_field_curve) must
produce identical output. If someone edits one and not the other, this fails.

Skips cleanly when Node isn't available so the rest of the suite still runs.
"""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from utils.draft_grade import dr_apply_field_curve

REPO = Path(__file__).resolve().parents[1]
CURVE_JS = REPO / "static" / "draft_grade_curve.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available"
)


def _js_curve(cases):
    """Run curveFieldScores in Node for a list of (scores, rounds_done)."""
    driver = (
        "const {curveFieldScores} = require(%s);\n"
        "const cases = %s;\n"
        "const out = cases.map(c => curveFieldScores(c[0], c[1]));\n"
        "process.stdout.write(JSON.stringify(out));\n"
        % (json.dumps(str(CURVE_JS)), json.dumps(cases))
    )
    res = subprocess.run(
        ["node", "-e", driver], capture_output=True, text=True, timeout=30
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _build_cases():
    rng = random.Random(1234)
    cases = []
    # Deterministic edge cases: below the 3-team floor, a tie field, extremes.
    cases.append(([90.0, 40.0], 10))                    # < 3 teams -> unchanged
    cases.append(([70.0, 70.0, 70.0], 10))              # zero variance
    cases.append(([95.0, 60.0, 55.0, 50.0], 10))        # one strong team
    cases.append(([79.9, 60.0, 55.0], 10))              # A-band raw floor edge
    cases.append(([88.0, 70.0, 65.0], 0))               # round 0 damping
    # Random fields at several draft stages.
    for _ in range(200):
        n = rng.randint(3, 14)
        scores = [round(rng.uniform(20, 100), rng.choice([0, 1, 2])) for _ in range(n)]
        rounds = rng.randint(0, 20)
        cases.append((scores, rounds))
    return cases


def test_js_and_python_curves_match():
    cases = _build_cases()
    js_out = _js_curve(cases)
    for (scores, rounds), js_row in zip(cases, js_out):
        py_row = dr_apply_field_curve(scores, rounds)
        assert [float(x) for x in js_row] == [float(x) for x in py_row], (
            f"curve drift for scores={scores} rounds={rounds}: "
            f"js={js_row} py={py_row}"
        )
