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

from utils.pick_score import compute_pick_score, empirical_slot_allocation

REPO = Path(__file__).resolve().parents[1]
PICK_JS = REPO / "static" / "pick_score.js"
CORE_JS = REPO / "static" / "draft_board_core.js"

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
            "passTd": rng.choice([4.0, 6.0]),
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
        ppr=c["ppr"], tep=c["tep"], pass_td=c["passTd"], is_tier_cliff=c["isTierCliff"],
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


def test_empirical_slot_allocation_match():
    """The browser allocator (draft_board_core.js empiricalSlotAllocation) and the
    Python server mirror (utils.pick_score.empirical_slot_allocation) must derive
    the SAME starters-per-position from the same pool/slots/teams - it now anchors
    VOR and PPG replacement on both surfaces, so any drift re-opens the grade gap
    this parity guards against. Exercises FLEX/SF menus and the empty-slots
    default-roster branch."""
    rng = random.Random(2024)
    positions = ["QB", "RB", "WR", "TE"]
    slot_menus = [
        ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "FLEX"],
        ["QB", "SUPER_FLEX", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        [],  # exercises the default-roster branch on both sides
    ]
    cases = []
    for _ in range(60):
        players = [
            {"position": rng.choice(positions), "value": round(rng.uniform(0, 9000), 1)}
            for _ in range(rng.randint(0, 40))
        ]
        cases.append({
            "players": players,
            "slots": rng.choice(slot_menus),
            "numTeams": rng.choice([8, 10, 12, 14]),
        })
    driver = (
        "global.self=global; global.BRPickScore=require(%s);\n"
        "const C=require(%s);\n"
        "const cases=%s;\n"
        "process.stdout.write(JSON.stringify(cases.map("
        "c=>C.empiricalSlotAllocation(c.players,c.slots,c.numTeams))));\n"
        % (json.dumps(str(PICK_JS)), json.dumps(str(CORE_JS)), json.dumps(cases))
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    js_out = json.loads(res.stdout)
    for c, js in zip(cases, js_out):
        py = empirical_slot_allocation(c["players"], c["slots"], c["numTeams"])
        for pos in ("QB", "RB", "WR", "TE"):
            assert abs(float(js[pos]) - float(py[pos])) < 1e-9, (
                f"empirical mismatch pos={pos} case={c} js={js} py={py}"
            )


def test_starter_counts_match():
    from utils.pick_score import starter_counts

    rng = random.Random(7)
    cases = [
        {"QB": 1, "SF": 0, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1},
        {"QB": 1, "SF": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2},
        {"QB": 2, "SF": 0, "RB": 3, "WR": 3, "TE": 2, "FLEX": 0},
    ]
    for _ in range(50):
        cases.append({k: rng.randint(0, 4) for k in ("QB", "SF", "RB", "WR", "TE", "FLEX")})
    driver = (
        "const {starterCounts} = require(%s);\n"
        "const cases = %s;\n"
        "process.stdout.write(JSON.stringify(cases.map(starterCounts)));\n"
        % (json.dumps(str(PICK_JS)), json.dumps(cases))
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    js_out = json.loads(res.stdout)
    for c, js in zip(cases, js_out):
        py = starter_counts(c)
        assert {k: float(v) for k, v in js.items()} == {k: float(v) for k, v in py.items()}, (
            f"starter_counts mismatch for {c}: js={js} py={py}"
        )
