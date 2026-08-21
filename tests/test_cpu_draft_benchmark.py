import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "benchmark_cpu_drafts.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def _run(seed: int = 42):
    result = subprocess.run(
        ["node", str(SCRIPT), "--drafts", "3", "--type", "redraft",
         "--teams", "12", "--rounds", "15", "--qb", "1", "--rb", "2",
         "--wr", "2", "--te", "1", "--flex", "2", "--seed", str(seed)],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_headless_cpu_benchmark_is_seeded_and_reports_roster_metrics():
    first = _run()
    second = _run()

    assert first["medianRound"] == second["medianRound"]
    assert first["medianFinalCount"] == second["medianFinalCount"]
    assert first["maximumFinalCount"] == second["maximumFinalCount"]
    assert first["maximumFinalCount"]["TE"] <= 3
    assert first["configuration"]["drafts"] == 3
    assert first["model"] == "shared-kernel"
    assert first["playerPool"] >= 180
    assert set(first["medianRound"]) == {"QB1", "QB2", "QB3", "TE1", "TE2", "TE3"}
    assert set(first["selectionRate"]) == set(first["medianRound"])
    assert sum(first["medianFinalCount"].values()) == 15
    assert 99 <= sum(first["phaseShare"]["early"].values()) <= 101
    assert first["invariants"] == {
        "incompleteRosters": 0,
        "wrongRosterSize": 0,
        "specialTeamsOverfill": 0,
    }
    assert set(first["waitingCalibration"]) == {"<50", "50-64", "65-79", "80+"}


def test_required_kicker_and_defense_do_not_become_normal_bench_depth():
    result = subprocess.run(
        ["node", str(SCRIPT), "--drafts", "20", "--type", "redraft",
         "--teams", "12", "--rounds", "15", "--qb", "1", "--rb", "2",
         "--wr", "2", "--te", "1", "--flex", "1", "--k", "1", "--def", "1",
         "--seed", "42"],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    report = json.loads(result.stdout)

    assert report["medianFinalCount"]["K"] == 1
    assert report["medianFinalCount"]["DEF"] == 1
    assert report["maximumFinalCount"]["K"] == 1
    assert report["maximumFinalCount"]["DEF"] == 1
