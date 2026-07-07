"""Parity guard: the browser team-grade composite (static/draft_grade_team.js
teamGradeComposite) and the Python server mirror (utils.draft_grade
dr_team_grade_score) must produce the same raw composite total for identical
inputs. This is the third and last grade mirror (after the curve and the
per-pick formula); pinning it means the Draft Room and Teams-page team letters
can't drift apart.

Skips cleanly when Node isn't available.
"""
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from utils.draft_grade import dr_team_grade_score

REPO = Path(__file__).resolve().parents[1]
TEAM_JS = REPO / "static" / "draft_grade_team.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available"
)

POS = ["QB", "RB", "WR", "TE"]
SLOTS_1QB = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX"]
SLOTS_SF = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SF"]
TARGETS = {"QB": 1.7, "RB": 5.45, "WR": 5.8, "TE": 2.05}


def _js_totals(cases):
    # Cases are large, so pass them via a temp file rather than the command line.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cases, f)
        cases_path = f.name
    driver = (
        "const {teamGradeComposite} = require(%s);\n"
        "const cases = require(%s);\n"
        "const out = cases.map(c => {\n"
        "  const r = teamGradeComposite(c.picks, c.slots, c.targets, c.numTeams,"
        " c.draftType, c.leaguePpg, c.leagueVal);\n"
        "  return r ? r.total : null;\n"
        "});\n"
        "process.stdout.write(JSON.stringify(out));\n"
        % (json.dumps(str(TEAM_JS)), json.dumps(cases_path))
    )
    try:
        res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=30)
    finally:
        Path(cases_path).unlink(missing_ok=True)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _build_cases():
    rng = random.Random(2024)
    cases = []
    for _ in range(300):
        sf = rng.choice([True, False])
        slots = SLOTS_SF if sf else SLOTS_1QB
        n = rng.randint(4, 22)
        picks = []
        for i in range(n):
            picks.append({
                "id": f"p{i}",
                "pos": rng.choice(POS),
                "ps": rng.choice([None, rng.randint(20, 100)]),
                "pn": rng.randint(1, 260),
                "val": round(rng.uniform(0, 9000), 1),
                "ppg": rng.choice([None, round(rng.uniform(0, 24), 1)]),
            })
        league_ppg = [round(rng.uniform(0, 24), 1) for _ in range(rng.randint(0, 60))]
        league_val = [round(rng.uniform(0, 9000), 1) for _ in range(rng.randint(0, 120))]
        cases.append({
            "picks": picks, "slots": slots, "targets": TARGETS,
            "numTeams": rng.choice([10, 12, 14]),
            "draftType": rng.choice(["startup", "redraft"]),
            "leaguePpg": league_ppg, "leagueVal": league_val,
        })
    return cases


def test_team_grade_composites_match():
    cases = _build_cases()
    js_out = _js_totals(cases)
    mismatches = []
    for c, js in zip(cases, js_out):
        py = dr_team_grade_score(
            c["picks"], slots=c["slots"], targets=c["targets"],
            num_teams=c["numTeams"], draft_type=c["draftType"],
            league_ppg_list=c["leaguePpg"], league_val_list=c["leagueVal"],
        )
        if (js is None) != (py is None):
            mismatches.append((js, py))
        elif js is not None and int(js) != int(py):
            mismatches.append((js, py))
    assert not mismatches, f"{len(mismatches)} composite mismatches, first: {mismatches[0]}"
