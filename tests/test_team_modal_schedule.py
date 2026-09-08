"""Team-modal Schedule tab must not invent results before any games are played.

The tab used to default ``lastPlayed`` to week 5 when weekly scores were empty,
so a 0-0 preseason team rendered as 0-5 with fake finals. These tests lock the
API contract and the client helper that decides which weeks are final.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from utils.matchup_schedule import last_finalized_week
from utils.week_proj import (
    flatten_proj_by_week,
    league_player_week_projections,
    team_schedule_from_matchups,
)

_REPO = Path(__file__).parents[1]
_APP_JS = (_REPO / "static" / "app.js").read_text(encoding="utf-8")
_APP_PY = (_REPO / "app.py").read_text(encoding="utf-8")


def _schedule_helpers_js() -> str:
    start = _APP_JS.index("function _tmSeededRng")
    end = _APP_JS.index("\nfunction tmToggleMatchup")
    return _APP_JS[start:end]


def test_api_sends_last_finalized_week_and_points_against():
    start = _APP_PY.index('@app.route("/api/team-details/<roster_id>")')
    end = _APP_PY.index('@app.route("/api/team-trades/<roster_id>")', start)
    body = _APP_PY[start:end]
    assert '"last_finalized_week": last_finalized_week' in body
    assert '"points_against": points_against' in body
    assert '"schedule_projections": schedule_projections' in body
    assert '"schedule_weeks": schedule_weeks' in body
    assert "team_schedule_from_matchups" in body
    assert "flatten_proj_by_week" in body
    assert "ensure_weekly_bits" in body
    assert "build_projections_by_week" in body
    assert "raw_scoring_settings" in body
    assert "league_player_week_projections" in body
    assert "fpts_against" in body
    # Viewed season only — not the graph fallback season.
    assert "get_league_ctx_from_cache(platform, league_id, season)" in body
    assert "the tab doesn't invent a 0-5 record before kickoff" in body


def test_js_does_not_default_last_played_to_week_five():
    block = _schedule_helpers_js()
    assert "function _tmScheduleLastPlayed" in block
    assert "const lastPlayed = _tmScheduleLastPlayed(data);" in block
    assert re.search(r"lastPlayed = ws\.length \? .* : 5", block) is None
    assert "return weeks.length ? Math.max.apply(null, weeks) : 0;" in block
    assert "(data && data.record) ? String(data.record)" in block
    assert "function _tmLineupForWeek" in block
    assert "data.schedule_projections" in block
    assert "data.schedule_weeks" in block
    assert "_tmSidesFromScheduleWeek" in block
    assert "_tmScoreLineup(myLineup, weekProj)" in block
    assert "_tmGenPts(pl.pos, rng)" not in block


def test_last_finalized_week_empty_is_zero():
    assert last_finalized_week(None) == 0
    assert last_finalized_week([]) == 0
    assert last_finalized_week([{"week": 1, "finalized": False}]) == 0
    assert last_finalized_week([{"week": 3, "finalized": True}, {"week": 1, "finalized": True}]) == 3
    assert last_finalized_week([{"week": 2}]) == 2


def test_league_player_week_projections_uses_weekly_map_and_median_gap():
    weeks = {
        1: {"a": 22.4, "b": 10.0},
        2: {"a": 0.0},  # explicit bye — keep 0, do not median-fill
        3: {"a": 20.0, "b": 11.0},
    }

    def load(season, week):
        assert season == 2026
        return weeks.get(week, {})

    out = league_player_week_projections(
        2026, 3, ["a", "b"], scoring_settings={"rec": 1.0}, load_week=load,
    )
    assert out["1"]["a"] == 22.4
    assert out["1"]["b"] == 10.0
    assert out["2"]["a"] == 0.0
    assert out["2"]["b"] == 10.5
    assert out["3"]["a"] == 20.0


def test_league_player_week_projections_empty_without_players():
    assert league_player_week_projections(2026, 17, [], load_week=lambda *_: {"x": 1}) == {}


def test_flatten_proj_by_week_unwraps_matchup_bundles():
    bundles = {
        1: {"projections": {"a": 22.4, "b": 10.0}},
        "2": {"a": 0.0, "b": 11.5},
        "_available": True,
    }
    out = flatten_proj_by_week(bundles)
    assert out["1"]["a"] == 22.4
    assert out["1"]["b"] == 10.0
    assert out["2"]["a"] == 0.0
    assert out["2"]["b"] == 11.5
    assert "_available" not in out


def test_team_schedule_from_matchups_uses_starter_projections():
    matchups = {
        1: [{
            "left": {
                "roster_id": "7",
                "name": "Caleb's Casting Couch",
                "starters": [
                    {"pid": "q", "name": "J. Hurts", "pos": "QB"},
                    {"pid": "s", "name": "S. Barkley", "pos": "RB"},
                ],
            },
            "right": {
                "roster_id": "3",
                "name": "The Audibles",
                "starters": [
                    {"pid": "q2", "name": "J. Allen", "pos": "QB"},
                    {"pid": "s2", "name": "J. Gibbs", "pos": "RB"},
                ],
            },
        }],
    }
    proj = {1: {"projections": {"q": 22.4, "s": 18.2, "q2": 24.1, "s2": 15.0}}}
    out = team_schedule_from_matchups("7", matchups, proj, starter_slots=["QB", "RB"])
    assert len(out) == 1
    week = out[0]
    assert week["week"] == 1
    assert week["opp"]["name"] == "The Audibles"
    assert week["opp"]["roster_id"] == "3"
    assert week["me"]["starters"][0]["name"] == "J. Hurts"
    assert week["me"]["starters"][0]["points"] == 22.4
    assert week["me"]["starters"][1]["points"] == 18.2
    assert week["me"]["total"] == 40.6
    assert week["opp"]["total"] == 39.1
    assert week["me"]["starters"][0]["label"] == "QB"


def test_team_schedule_from_matchups_skips_weeks_without_this_roster():
    matchups = {
        1: [{"left": {"roster_id": "1", "name": "A", "starters": []},
             "right": {"roster_id": "2", "name": "B", "starters": []}}],
    }
    assert team_schedule_from_matchups("7", matchups, {}) == []


def _node_available() -> bool:
    try:
        subprocess.run(["node", "-v"], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


@pytest.mark.skipif(not _node_available(), reason="node is required to eval the schedule helpers")
def test_preseason_schedule_html_is_projected_not_final(tmp_path):
    helpers = _schedule_helpers_js()
    script = tmp_path / "schedule_preseason.js"
    script.write_text(
        helpers
        + r"""
global.window = { _tmRosterId: 7 };
global.location = { pathname: '/sleeper/2026/1312067280816832512/standings' };
const html = _tmBuildScheduleHtml({
  team_name: "Caleb's Casting Couch",
  record: '0-0',
  points_for: 0,
  points_against: 0,
  last_finalized_week: 0,
  playoff_odds: 90,
  graphs: { weekly_scores: [], season_used: 2025 },
  roster: [{ name: 'A. Player', position: 'QB', player_id: '1' }],
});
if (_tmScheduleLastPlayed({ graphs: { weekly_scores: [] } }) !== 0) {
  console.error('empty weekly_scores must be lastPlayed=0');
  process.exit(1);
}
if (_tmScheduleLastPlayed({}) !== 0) {
  console.error('missing graphs must be lastPlayed=0, not a fake week 5');
  process.exit(1);
}
if (_tmScheduleLastPlayed({ last_finalized_week: 0, graphs: { weekly_scores: [{week:17, points:100}] } }) !== 0) {
  console.error('explicit last_finalized_week=0 must win over leftover scores');
  process.exit(1);
}
if (_tmScheduleLastPlayed({ graphs: { weekly_scores: [{week:1},{week:5}], season_used: 2025 } }) !== 0) {
  console.error('prior-season graph fallback must not count as played');
  process.exit(1);
}
if (_tmScheduleLastPlayed({ last_finalized_week: 3, graphs: { weekly_scores: [] } }) !== 3) {
  console.error('in-season last_finalized_week must be used');
  process.exit(1);
}
const failures = [];
if (!html.includes('>0-0<')) failures.push('record tile should be 0-0');
if (html.includes('>0-5<') || html.includes('>L5<')) failures.push('must not invent 0-5 / L5');
if (html.includes('tm-sched-result tm-sched-l') || html.includes('tm-sched-result tm-sched-w')) {
  failures.push('no final W/L badges before kickoff');
}
if (!html.includes('tm-sched-proj')) failures.push('unplayed weeks should show projections');
if (!html.includes('tm-sched-upcoming')) failures.push('weeks should be upcoming, not final');
if (html.includes('>FINAL<')) failures.push('no FINAL tag before kickoff');
if (failures.length) { console.error(failures.join('; ')); process.exit(1); }

const inSeason = _tmBuildScheduleHtml({
  team_name: 'In Season',
  record: '3-2',
  points_for: 612,
  points_against: 580,
  last_finalized_week: 5,
  graphs: { weekly_scores: [{week:1},{week:2},{week:3},{week:4},{week:5}] },
});
if (!inSeason.includes('>3-2<')) { console.error('in-season record tile should use API record'); process.exit(1); }
if (!/tm-sched-result tm-sched-[wlt]/.test(inSeason)) {
  console.error('in-season weeks 1-5 should have W/L/T badges');
  process.exit(1);
}
if (!inSeason.includes('>FINAL<')) { console.error('played weeks should be FINAL'); process.exit(1); }

const scored = _tmScoreLineup([{pos:'QB', id:'1', name:'Hurts'}], {'1': 22.4});
if (scored.total !== 22.4 || scored.starters[0].points !== 22.4) {
  console.error('lineup points must use schedule_projections, got ' + scored.total);
  process.exit(1);
}
const blank = _tmScoreLineup([{pos:'QB', id:'1', name:'Hurts'}], {});
if (blank.total !== 0) {
  console.error('missing projections must be 0, not RNG; got ' + blank.total);
  process.exit(1);
}
const slots = _tmResolveSlots(['QB', 'RB']);
const lined = _tmLineupForWeek(
  [{name:'Backup', position:'RB', player_id:'b'}, {name:'Starter', position:'RB', player_id:'s'}, {name:'QB1', position:'QB', player_id:'q'}],
  'position', slots, {s: 18.2, b: 4.1, q: 22.4}
);
if (lined[0].name !== 'QB1' || lined[1].name !== 'Starter') {
  console.error('slots must fill highest remaining projection, got ' + lined.map(x => x.name).join(','));
  process.exit(1);
}
const htmlProj = _tmBuildScheduleHtml({
  team_name: 'Proj Team',
  record: '0-0',
  points_for: 0,
  points_against: 0,
  last_finalized_week: 0,
  starter_slots: ['QB', 'RB'],
  roster: [{name:'QB1', position:'QB', player_id:'q'}, {name:'RB1', position:'RB', player_id:'s'}],
  schedule_projections: {1: {q: 22.4, s: 18.2}},
});
if (!htmlProj.includes('22.4') || !htmlProj.includes('18.2')) {
  console.error('expanded/projected rows must show actual weekly projections');
  process.exit(1);
}
if (!htmlProj.includes('Proj 40.6')) {
  console.error('week total should be the sum of starter projections, got missing Proj 40.6');
  process.exit(1);
}
const htmlMu = _tmBuildScheduleHtml({
  team_name: 'Proj Team',
  record: '0-0',
  points_for: 0,
  points_against: 0,
  last_finalized_week: 0,
  starter_slots: ['QB', 'RB'],
  roster: [{name:'Wrong QB', position:'QB', player_id:'wrong'}],
  schedule_projections: {1: {q: 99, s: 99, wrong: 50}},
  schedule_weeks: [{
    week: 1,
    me: {
      roster_id: '7',
      name: 'Proj Team',
      total: 40.6,
      starters: [
        {id:'q', name:'J. Hurts', pos:'QB', label:'QB', points:22.4},
        {id:'s', name:'S. Barkley', pos:'RB', label:'RB', points:18.2},
      ],
    },
    opp: {
      roster_id: '3',
      name: 'The Audibles',
      total: 19.5,
      starters: [
        {id:'q2', name:'J. Allen', pos:'QB', label:'QB', points:19.5},
      ],
    },
  }],
});
if (!htmlMu.includes('The Audibles')) {
  console.error('schedule_weeks opponent must be used instead of a shuffled name');
  process.exit(1);
}
if (!htmlMu.includes('J. Hurts') || !htmlMu.includes('22.4') || !htmlMu.includes('S. Barkley')) {
  console.error('schedule_weeks starters and matchups-page points must render');
  process.exit(1);
}
if (!htmlMu.includes('Proj 40.6')) {
  console.error('week total must come from matchup starter projections');
  process.exit(1);
}
const week1 = htmlMu.slice(htmlMu.indexOf('tm-sched-week">W1'), htmlMu.indexOf('tm-sched-week">W2'));
if (week1.includes('Wrong QB')) {
  console.error('week 1 must not greedy-fill from roster when schedule_weeks has that week');
  process.exit(1);
}
if (week1.includes('99.0') || week1.includes('50.0')) {
  console.error('week 1 must not score from leftover schedule_projections');
  process.exit(1);
}
console.log('ok');
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout
