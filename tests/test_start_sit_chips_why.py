"""Start/Sit display chips + WHY line + Trends share/RZ rows.

Covers audit items 6-8 (display-only):
- every non-injury ``demotion`` value from utils/start_sit_score.py has a chip
  label in the waivers start/sit row renderer;
- the tap-to-expand evidence WHY line names the score factors that moved the
  player (>=1%, sorted by absolute impact, top 4), including the three factors
  wvVerdictReasons omits (oline, expected_plays, role);
- the player modal Trends view has target/carry share and red-zone rows.

The WHY-line logic itself is executed under node: the function is extracted
from the waivers_page.py f-string (undoubling the f-string braces) so the
test exercises the exact shipped code.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WAIVERS = ROOT / "dashboard_services" / "pages" / "waivers_page.py"
SCORE = ROOT / "utils" / "start_sit.py"
MODAL = ROOT / "static" / "player_modal.js"

EXPECTED_DEMOTION_LABELS = {
    "low_total": "Low team total",
    "weather": "Bad weather",
    "oline": "Weak O-line",
    "low_play_volume": "Slow pace",
    "volatile_role": "Volatile role",
}


def _demotion_values():
    """All demotion string values assigned in compute_start_score."""
    return re.findall(r'demotion\s*=\s*(?:demotion\s*or\s*)?"([a-z_]+)"', SCORE.read_text())


def _fstring_region(start_marker, end_marker):
    """Extract a JS region from the waivers_page.py f-string, undoubling braces."""
    src = WAIVERS.read_text(encoding="utf-8")
    start = src.index(start_marker)
    end = src.index(end_marker, start)
    return src[start:end].replace("{{", "{").replace("}}", "}")


def test_all_score_demotions_have_chip_labels():
    values = set(_demotion_values())
    # injury-driven demotions surface through the injury badge, not a chip
    assert {"low_total", "weather", "oline", "low_play_volume", "volatile_role"} <= values
    src = WAIVERS.read_text(encoding="utf-8")
    for demotion, label in EXPECTED_DEMOTION_LABELS.items():
        assert f"{demotion}: '{label}'" in src, f"missing chip label for {demotion}"
    # the row renderer looks the label up instead of hardcoding low_total only
    assert "WV_DEMOTION_LABELS[p.demotion]" in src
    assert "(p.demotion === 'low_total')" not in src


def test_why_line_wired_into_evidence():
    src = WAIVERS.read_text(encoding="utf-8")
    assert "wvSsWhyLine(p)" in src
    for key, label in [("oline", "O-line"), ("expected_plays", "Pace"), ("role", "Role")]:
        assert f"{key}: '{label}'" in src


def _run_why(factors):
    js = _fstring_region("var WV_SS_FACTOR_LABELS", "// Evidence grid for one player")
    driver = (
        js
        + "\nconsole.log(JSON.stringify(wvSsWhyLine({score_factors: "
        + json.dumps(factors)
        + "})));\n"
    )
    proc = subprocess.run(
        ["node", "-e", driver], capture_output=True, text=True, timeout=30, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


def test_why_line_names_movers_as_signed_percentages():
    out = _run_why({
        "form": 1.05, "oline": 0.97, "weather": 0.96, "usage": 1.012,
        "floor": 1.0, "vegas": 1.0, "avail": 1.0,
        "expected_plays": 1.0, "role": 1.0,
    })
    # sorted by absolute impact, factors under 1% excluded
    assert out == "Form +5% \u00b7 Weather -4% \u00b7 O-line -3% \u00b7 Usage +1%"


def test_why_line_caps_at_four_and_names_pace():
    out = _run_why({
        "form": 1.10, "oline": 0.90, "weather": 1.08, "usage": 1.06,
        "vegas": 0.95, "role": 1.04, "floor": 1.03, "avail": 1.02,
        "expected_plays": 0.85,
    })
    parts = out.split(" \u00b7 ")
    assert len(parts) == 4
    assert parts[0] == "Pace -15%"  # expected_plays is named, never anonymous
    assert parts[1] == "Form +10%"


def test_why_line_empty_when_nothing_moved():
    assert _run_why({}) == ""
    assert _run_why({"form": 1.005, "vegas": 0.999}) == ""


def test_trends_has_share_and_red_zone_rows():
    js = MODAL.read_text(encoding="utf-8")
    assert "rowFor('Tgt Share %', 'target_share'" in js
    assert "rowFor('Carry %', 'carry_share'" in js
    assert "rowForComputed('RZ Touches'" in js
    assert "Number(w.rz_targets || 0) + Number(w.rz_carries || 0)" in js


def test_player_modal_js_syntax():
    proc = subprocess.run(
        ["node", "--check", str(MODAL)], capture_output=True, text=True, timeout=30, check=False
    )
    assert proc.returncode == 0, proc.stderr
