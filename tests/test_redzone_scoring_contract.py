"""Regression coverage for Redzone's canonical PBP scoring and contributors."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RZ = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
APP = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _js_between(start: str, end: str) -> str:
    return RZ[RZ.index(start):RZ.index(end, RZ.index(start))]


def _app_js_between(start: str, end: str) -> str:
    return APP[APP.index(start):APP.index(end, APP.index(start))]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_redzone_scores_receptions_kicks_and_cumulative_lines_from_league_contract():
    scoring = _js_between("function _n(x)", "function _scoringForPid")
    script = f"""
{scoring}
var ppr = {{rec: 1, rec_yd: .1, rec_td: 6, bonus_rec_te: .5}};
var half = {{rec: .5, rec_yd: .1}};
var standard = {{rec: 0, rec_yd: .1}};
var kicking = {{fgm: 3, fgm_50p: 5, xpm: 1}};
// Per-yard FG league: no flat fgm, no distance bucket -- points per made-FG yard.
var perYd = {{fgm_yds: 0.1, xpm: 1}};
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
  noDouble: _lineToPts({{fgm: 1, fgm_50_59: 1}}, kicking, 'K'),
  // Per-yard league scores a 58-yd FG off its distance (the reported bug: this
  // used to be 0 because no fgm/bucket key matched). Box lines carry only
  // fg_long for a single kick, so that path must score too.
  perYdWithTotal: parseFloat(_lineToPts({{fgm: 1, fg_yds: 58}}, perYd, 'K').toFixed(2)),
  perYdFromLong: parseFloat(_lineToPts({{fgm: 1, fg_long: 58}}, perYd, 'K').toFixed(2)),
  perYdXp: _lineToPts({{xpm: 1}}, perYd, 'K')
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert out == {
        "ppr": 1.5, "half": 1.0, "standard": 0.5, "te": 2.0, "wr": 1.5,
        "made56": 5, "missed52": 0, "xp": 1, "recIncrement": 1.5,
        "fgIncrement": 5, "noDouble": 5,
        "perYdWithTotal": 5.8, "perYdFromLong": 5.8, "perYdXp": 1,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_player_modal_summary_scores_fgs_by_distance_and_per_yard():
    """The player-modal Redzone summary must score field goals the same way the
    feed does -- by distance bucket or per yard -- not at a flat `fgm` rate that
    Sleeper distance/per-yard leagues never define (the reported bug: a made FG
    showed 0 pts and no FGM row in the summary)."""
    fg = _app_js_between("function _rzFgRate(yds, s)", "// ── Redzone live player HTML")
    script = f"""
{fg}
var bucket = {{fgm_50p: 5, xpm: 1}};       // distance-bucket league
var perYd = {{fgm_yds: 0.1, xpm: 1}};      // per-yard league (no flat fgm)
var flat = {{fgm: 3, xpm: 1}};             // flat per-make league
console.log(JSON.stringify({{
  bucket58: _rzFgPts({{fgm: 1, fg_long: 58}}, bucket),
  perYd58: parseFloat(_rzFgPts({{fgm: 1, fg_long: 58}}, perYd).toFixed(2)),
  flat: _rzFgPts({{fgm: 1, fg_long: 58}}, flat),
  noKickerScoring: _rzFgPts({{fgm: 1, fg_long: 58}}, {{xpm: 1}})
}}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert out == {"bucket58": 5, "perYd58": 5.8, "flat": 3, "noKickerScoring": 0}


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
