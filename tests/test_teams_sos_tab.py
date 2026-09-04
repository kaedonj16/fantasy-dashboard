"""Guards for the Teams-page Schedule (SoS) tab markup and CSS.

The old list used a 140px name column plus bars scaled to max opponent
score, which made preseason rows look identical and all-red. These tests
lock the rebuilt ranking UI and the relative min–max bar math.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "teams.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")

_SOS_SELECTORS = (
    ".sos-panel",
    ".sos-header",
    ".sos-title",
    ".sos-subtitle",
    ".sos-meta",
    ".sos-note",
    ".sos-legend",
    ".sos-list",
    ".sos-row",
    ".sos-name-text",
    ".sos-you",
    ".sos-diff",
    ".sos-track",
    ".sos-fill",
    ".sos-val",
    ".sos-mine",
)


def _fn_block(name: str) -> str:
    start = JS.index("function %s(" % name)
    rest = JS[start:]
    nxt = rest.find("\n      function ", 10)
    return rest if nxt < 0 else rest[:nxt]


def test_sos_css_selectors_exist():
    for sel in _SOS_SELECTORS:
        assert sel in CSS, f"missing Schedule-tab CSS for {sel}"


def test_sos_name_ellipsis_lives_on_inner_text():
    """YOU pill must not eat the last pixels of a long team name."""
    name = re.search(r"\.sos-name-text\s*\{([^}]+)\}", CSS)
    assert name, "missing .sos-name-text rule"
    body = name.group(1)
    assert "ellipsis" in body
    assert "overflow: hidden" in body
    assert "nowrap" in body


def test_sos_bars_use_spread_not_max_only():
    bar = _fn_block("_sosBarPct")
    assert "spread < 0.05" in bar
    assert "22 +" in bar
    assert "* 78" in bar
    # Must not scale solely against the league max (that filled every bar).
    assert "/ maxOpp" not in _fn_block("renderSos")


def test_sos_tiers_collapse_when_schedules_are_even():
    tier = _fn_block("_sosTier")
    assert "if (even) return { key: 'even', label: 'Even' }" in tier
    assert "Hardest" in tier
    assert "Easiest" in tier


def test_sos_render_escapes_team_names_and_marks_viewer():
    render = _fn_block("renderSos")
    assert "_sosEsc(t.team_name)" in render
    assert "sos-mine" in render
    assert "sos-you" in render
    assert "No games played yet" in render
    # Duplicate preseason copy is gone.
    assert "Based on roster strength (no games played yet)" not in render
    assert "analytics-bar-list" not in render
    assert "analytics-empty" not in render


def test_load_sos_resyncs_then_renders():
    load = _fn_block("loadSos")
    assert load.index("_syncLeagueCfg()") < load.index("if (_loaded.sos)")
    assert "renderSos(panel, data)" in load
    assert "/api/schedule-strength" in load


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_sos_helpers_rank_clustered_values():
    """Clustered opponent scores must still produce a spread of bar widths."""
    helpers = _fn_block("_sosTier") + "\n" + _fn_block("_sosBarPct")
    script = helpers + r"""
    var vals = [121.4, 119.8, 118.2, 116.1, 114.0, 112.2];
    var min = Math.min.apply(null, vals);
    var max = Math.max.apply(null, vals);
    var spread = max - min;
    var pcts = vals.map(function(v) { return _sosBarPct(v, min, spread); });
    var unique = Object.create(null);
    pcts.forEach(function(p) { unique[p] = 1; });
    var tiers = vals.map(function(_, i) { return _sosTier(i, vals.length, false); });
    var even = _sosTier(0, 6, true);
    console.log(JSON.stringify({
      pcts: pcts,
      unique: Object.keys(unique).length,
      first: tiers[0],
      last: tiers[tiers.length - 1],
      even: even,
      evenBar: _sosBarPct(100, 100, 0)
    }));
    """
    out = subprocess.check_output(["node", "-e", script], text=True)
    import json
    data = json.loads(out)
    assert data["unique"] == 6
    assert data["pcts"][0] == 100
    assert data["pcts"][-1] == 22
    assert data["first"]["label"] == "Hardest"
    assert data["last"]["label"] == "Easiest"
    assert data["even"]["label"] == "Even"
    assert data["evenBar"] == 48
