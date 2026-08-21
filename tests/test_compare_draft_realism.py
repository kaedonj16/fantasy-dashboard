import json

import pytest

from scripts.compare_draft_realism import compare, load_report, main, markdown


def fixtures():
    real = [{"draft_type": "redraft", "format": "1QB", "drafts": 100,
             "qb_rounds": {"QB1": 7}, "te_rounds": {"TE1": 7},
             "position_counts": {"QB": 2, "RB": 5, "WR": 6, "TE": 2, "K": 0, "DEF": 0},
             "phase_shares": {"early (1-4)": {"QB": 5}, "middle (5-9)": {}, "late (10+)": {}}}]
    cpu = {"configuration": {"type": "redraft", "sf": False, "drafts": 1000, "k": 1, "def": 1},
           "medianRound": {"QB1": 6, "TE1": 7},
           "medianFinalCount": {"QB": 2, "RB": 5, "WR": 6, "TE": 2, "K": 1, "DEF": 1},
           "phaseShare": {"early": {"QB": 6}, "middle": {}, "late": {}},
           "invariants": {"incompleteRosters": 0},
           "waitingCalibration": {"80+": {"samples": 10, "predictedPct": 85, "actualPct": 80}}}
    return real, cpu


def test_comparison_matches_cohort_and_calculates_deltas():
    report = compare(*fixtures())
    assert report["timing"]["QB1"]["delta"] == -1
    assert report["finalCounts"]["RB"]["delta"] == 0
    assert "CPU vs Real Draft Comparison" in markdown(report)
    assert any("mixed-roster comparison" in warning for warning in report["warnings"])


def test_cli_writes_reports_and_can_enforce_threshold(tmp_path):
    real, cpu = fixtures()
    real_path, cpu_path = tmp_path / "real.json", tmp_path / "cpu.json"
    out, jout = tmp_path / "report.md", tmp_path / "report.json"
    real_path.write_text(json.dumps(real))
    cpu_path.write_text(json.dumps(cpu))
    assert main(["--real", str(real_path), "--cpu", str(cpu_path), "--output", str(out),
                 "--json", str(jout), "--max-mean-delta", "100"]) == 0
    assert out.exists() and json.loads(jout.read_text())["cohort"]["format"] == "1QB"


def test_missing_render_artifact_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ephemeral across deploys"):
        load_report(tmp_path / "real-redraft.json", "Real-draft audit")
