"""Parity guard: the browser pick-score formula (static/pick_score.js
computePickScore) and the Python server mirror (utils.pick_score.compute_pick_score)
must produce identical grades for identical inputs. This isolates the grade
FORMULA: if this passes, any remaining Draft-Room vs Teams-page difference is
input data, not math.

Skips cleanly when Node isn't available.
"""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from utils.pick_score import compute_pick_score

REPO = Path(__file__).resolve().parents[1]
PICK_JS = REPO / "static" / "pick_score.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available"
)


def _js_scores(cases):
    driver = (
        "const {computePickScore} = require(%s);\n"
        "const cases = %s;\n"
        "process.stdout.write(JSON.stringify(cases.map(computePickScore)));\n"
        % (json.dumps(str(PICK_JS)), json.dumps(cases))
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _build_cases():
    rng = random.Random(99)
    positions = ["QB", "RB", "WR", "TE"]
    types = ["startup", "redraft", "rookie"]
    cases = []
    for _ in range(400):
        pos = rng.choice(positions)
        max_val = rng.choice([1000, 5000, 8000, 12000])
        c = {
            "pos": pos,
            "value": round(rng.uniform(0, max_val), 1),
            "vor": rng.choice([None, round(rng.uniform(-500, max_val), 1)]),
            "tier": rng.choice([None, 0, 1, 2, 3, 5, 9, 12]),
            "age": rng.choice([None, 21, 23, 24, 27, 29, 33]),
            "rankChange7d": rng.choice([None, -15, 0, 8, 25]),
            "avgPick": rng.choice([None, 1.0, 4.5, 8.0, 20.0, 55.0, 150.0]),
            "pickNo": rng.randint(1, 200),
            "maxVal": max_val,
            "draftType": rng.choice(types),
            "isSf": rng.choice([True, False]),
            "needRaw": round(rng.uniform(0, 1), 2),
            "qbCount": rng.randint(0, 3),
            "totalPicks": rng.choice([0, 180, 240, 288]),
            "numTeams": rng.choice([10, 12, 14]),
            "ppgNorm": rng.choice([None, 0.0, 0.3, 0.7, 1.0]),
            "ppr": rng.choice([0.0, 0.5, 1.0]),
            "tep": rng.choice([0.0, 0.5, 1.0]),
            "isTierCliff": rng.choice([True, False]),
        }
        cases.append(c)
    return cases


def _py_score(c):
    return compute_pick_score(
        pos=c["pos"], value=c["value"], vor=c["vor"], tier=c["tier"],
        age=c["age"], rank_change_7d=c["rankChange7d"], avg_pick=c["avgPick"],
        pick_no=c["pickNo"], max_val=c["maxVal"], draft_type=c["draftType"],
        is_sf=c["isSf"], need_raw=c["needRaw"], qb_count=c["qbCount"],
        total_picks=c["totalPicks"], num_teams=c["numTeams"], ppg_norm=c["ppgNorm"],
        ppr=c["ppr"], tep=c["tep"], is_tier_cliff=c["isTierCliff"],
    )


def test_js_and_python_pick_scores_match():
    cases = _build_cases()
    js_out = _js_scores(cases)
    mismatches = []
    for c, js in zip(cases, js_out):
        py = _py_score(c)
        if int(js) != int(py):
            mismatches.append((c, js, py))
    assert not mismatches, (
        f"{len(mismatches)} formula mismatches, first: "
        f"case={mismatches[0][0]} js={mismatches[0][1]} py={mismatches[0][2]}"
    )
