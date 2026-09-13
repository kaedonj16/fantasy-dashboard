"""Regression coverage for Redzone's canonical PBP scoring and contributors."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RZ = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")


def _js_between(start: str, end: str) -> str:
    return RZ[RZ.index(start):RZ.index(end, RZ.index(start))]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_redzone_scores_receptions_kicks_and_cumulative_lines_from_league_contract():
    scoring = _js_between("function _n(x)", "function _scoringForPid")
    script = f"""
{scoring}
var ppr = {{rec: 1, rec_yd: .1, rec_td: 6, bonus_rec_te: .5}};
var half = {{rec: .5, rec_yd: .1}};
var standard = {{rec: 0, rec_yd: .1}};
var kicking = {{fgm: 3, fgm_50p: 5, xpm: 1}};
console.log(JSON.stringify({{
  ppr: _lineToPts({{rec: 1, rec_yds: 5}}, ppr, 'WR'),
  half: _lineToPts({{rec: 1, rec_yds: 5}}, half, 'WR'),
  standard: _lineToPts({{rec: 1, rec_yds: 5}}, standard, 'WR'),
  te: _lineToPts({{rec: 1, rec_yds: 5}}, ppr, 'TE'),
  wr: _lineToPts({{rec: 1, rec_yds: 5}}, ppr, 'WR'),
  made56: _lineToPts({{fgm: 1, fg_yds: 56, fgm_50_59: 1}}, kicking, 'K'),
  missed52: _lineToPts({{}}, kicking, 'K'),
  xp: _lineToPts({{xpm: 1}}, kicking, 'K'),
  recIncrement: _lineToPts({{rec: 5, rec_yds: 25}}, ppr, 'WR') - _lineToPts({{rec: 4, rec_yds: 20}}, ppr, 'WR'),
  fgIncrement: _lineToPts({{fgm: 2, fgm_50_59: 1, fgm_40_49: 1}}, kicking, 'K') - _lineToPts({{fgm: 1, fgm_40_49: 1}}, kicking, 'K'),
  noDouble: _lineToPts({{fgm: 1, fgm_50_59: 1}}, kicking, 'K')
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert out == {
        "ppr": 1.5, "half": 1.0, "standard": 0.5, "te": 2.0, "wr": 1.5,
        "made56": 5, "missed52": 0, "xp": 1, "recIncrement": 1.5,
        "fgIncrement": 5, "noDouble": 5,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_secondary_contributors_are_canonical_deduped_and_never_replace_primary():
    helper = _js_between("function _secondaryContributors", "function _eventHtml")
    script = f"""
function _n(x) {{ return parseFloat(x || 0) || 0; }}
{helper}
var ev = {{pid: 'wr', contributions: [
  {{pid: 'wr', pts: 9.3, line: {{rec_td: 1}}}},
  {{pid: 'qb', name: 'Quarterback', pts: 4.9, line: {{pass_td: 1}}}},
  {{pid: 'qb', name: 'Quarterback duplicate', pts: 4.9, line: {{pass_td: 1}}}},
  {{pid: 'k', name: 'Zero', pts: 0}}
]}};
console.log(JSON.stringify(_secondaryContributors(ev)));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert [row["pid"] for row in out] == ["qb"]
    assert out[0]["pts"] == 4.9


def test_redzone_boundary_restamps_normalized_league_scoring_and_renders_context():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "normalize_league_scoring(" in app[app.index("def _redzone_collect"):app.index("def _redzone_fetch")]
    assert "function _secondaryContributors(ev)" in RZ
    assert "rz-event-contributors" in RZ
    # Canonical groups continue to own the single top-level event identity.
    grouped = _js_between("function _eventsFromPbp", "function _detectChanges")
    assert "contributionsByKey" in grouped
    assert "playId: playKey" in grouped
