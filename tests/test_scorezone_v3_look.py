"""ScoreZone v3 look treatments + as-of-play score (Kaedon 'build it', 2026-10-25).

Covers:
- the shortened lowercase cume format (treatment 3),
- the one-line hero center status (treatment 4),
- the as-of-play score line (item 8): provider stamp first, then a
  derivation over the game's play rows that is trusted only when it
  reconciles exactly with the game's current score,
- extractor carry-through of ESPN/Tank01 per-play score fields,
- CSS source contracts for the de-chrome / color-job treatments (1, 2, 5,
  6, 7).
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RZ = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node required")


def _extract(src: str, name: str) -> str:
    marker = f"function {name}("
    start = src.index(marker)
    i = src.index("{", start)
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
    raise AssertionError(f"unbalanced braces extracting {name}")


def _run_node(body: str) -> dict:
    fns = [
        "_cumeLine",
        "_normGameStatus",
        "_playerGameState",
        "_sideCounts",
        "_matchupState",
        "_matchupCenterHtml",
        "_nflPlayKey",
        "_playNflPoints",
        "_asOfScoreMap",
        "_scoreStrForEvent",
    ]
    extracted = "\n".join(_extract(RZ, n) for n in fns)
    script = f"""
const assert = require('node:assert/strict');
function _n(x) {{ return parseFloat(x || 0) || 0; }}
const factory = new Function('_state', '_nflGameInfo', `
  function _n(x) {{ return parseFloat(x || 0) || 0; }}
  var _asOfCache = {{}};
  {extracted}
  return {{ _cumeLine, _matchupCenterHtml, _playNflPoints, _scoreStrForEvent }};
`);
{body}
"""
    out = subprocess.check_output([node, "-e", script], text=True)
    return json.loads(out.strip().splitlines()[-1])


# ── Treatment 3: lowercase single-line cume ──────────────────────────────────


def test_cume_line_is_shortened_lowercase():
    res = _run_node(
        """
const api = factory({}, () => null);
const out = {
  rb: api._cumeLine('RB', {carries: 18, rush_yds: 96, rush_td: 2}),
  qb: api._cumeLine('QB', {pass_cmp: 19, pass_att: 28, pass_yds: 212, pass_td: 2, int: 1}),
  wr: api._cumeLine('WR', {rec: 6, targets: 9, rec_yds: 88, rec_td: 1}),
  k: api._cumeLine('K', {fgm: 2, xpm: 3}),
};
console.log(JSON.stringify(out));
"""
    )
    assert res["rb"] == "18 car, 96 yd, 2 td"
    assert res["qb"] == "19/28 cmp, 212 yd, 2 td, 1 int"
    assert res["wr"] == "6/9 rec, 88 yd, 1 td"
    assert res["k"] == "2 fg, 3 xp"


def test_yd_chip_uses_lowercase_yd():
    assert "ydVal + ' yd</span>'" in RZ
    assert "ydVal + ' YD</span>'" not in RZ


# ── Treatment 4: hero center is one muted line ───────────────────────────────


def test_hero_center_line_live_toplay_final():
    res = _run_node(
        """
function build(games, pi) {
  const state = { games, player_info: pi };
  const nflGameInfo = (gid) => (gid && games[gid]) ? games[gid] : null;
  return factory(state, nflGameInfo);
}
// Live: 3 starters live for A, 2 for B.
{
  const games = { L: { status: 'live', game_code: '1' } };
  const pi = {};
  ['a1','a2','a3'].forEach(p => pi[p] = { team: 'NE', game_id: 'L', game_code: '1' });
  ['b1','b2'].forEach(p => pi[p] = { team: 'SEA', game_id: 'L', game_code: '1' });
  var liveHtml = build(games, pi)._matchupCenterHtml(
    { starters: ['a1','a2','a3'] }, { starters: ['b1','b2'] });
}
// Pregame: 4 to play for A, 3 for B.
{
  const games = { U: { status: 'pregame', game_code: '0' } };
  const pi = {};
  ['a1','a2','a3','a4'].forEach(p => pi[p] = { team: 'NE', game_id: 'U', game_code: '0' });
  ['b1','b2','b3'].forEach(p => pi[p] = { team: 'SEA', game_id: 'U', game_code: '0' });
  var toplayHtml = build(games, pi)._matchupCenterHtml(
    { starters: ['a1','a2','a3','a4'] }, { starters: ['b1','b2','b3'] });
}
// Final.
{
  const games = { F: { status: 'final', game_code: '2' } };
  const pi = { a1: { team: 'NE', game_id: 'F', game_code: '2' },
               b1: { team: 'SEA', game_id: 'F', game_code: '2' } };
  var finalHtml = build(games, pi)._matchupCenterHtml(
    { starters: ['a1'] }, { starters: ['b1'] });
}
console.log(JSON.stringify({ liveHtml, toplayHtml, finalHtml }));
"""
    )
    assert res["liveHtml"] == (
        '<div class="rz-mc-line"><span class="rz-mc-dot"></span>'
        "3 playing · 2 for them</div>"
    )
    assert res["toplayHtml"] == (
        '<div class="rz-mc-line">4 to play · 3 for them</div>'
    )
    assert res["finalHtml"] == '<div class="rz-mc-line">Final</div>'


# ── Item 8: score as of the play ─────────────────────────────────────────────

# DET (away) at GB (home), final DET 31-28 GB. Scoring order:
# DET TD, DET PAT, GB TD, GB PAT, DET FG, GB TD, GB PAT, DET TD, DET PAT,
# (GB non-scoring), GB TD, GB PAT, DET TD, DET PAT, GB TD, GB PAT,
# DET TD, DET PAT.
_AS_OF_BODY = """
const GID = '20261025_DET@GB';
function row(seq, playId, team, line, extra) {
  return Object.assign({
    play_id: playId, seq: seq, team: team, stat_line: line || {},
    is_td: false, play_state: 'VALID',
  }, extra || {});
}
const plays = [
  row(1, 'p1', 'DET', {carries: 1, rush_yds: 4}),
  row(2, 'p2', 'DET', {pass_td: 1}, {is_td: true, pid: 'qb1'}),
  row(2, 'p2', 'DET', {rec_td: 1}, {is_td: true, pid: 'wr1'}),
  row(3, 'p3', 'DET', {xpm: 1}),
  row(4, 'p4', 'GB', {pass_td: 1}, {is_td: true}),
  row(4, 'p4', 'GB', {rec_td: 1}, {is_td: true}),
  row(5, 'p5', 'GB', {xpm: 1}),
  // A nullified DET touchdown: on review it never scored.
  row(6, 'p6', 'DET', {pass_td: 1}, {is_td: true, play_state: 'NULLIFIED'}),
  row(7, 'p7', 'DET', {fgm: 1, fg_long: 44}),
  row(8, 'p8', 'GB', {rush_td: 1}, {is_td: true}),
  row(9, 'p9', 'GB', {xpm: 1}),
  row(10, 'p10', 'DET', {pass_td: 1}, {is_td: true}),
  row(11, 'p11', 'DET', {xpm: 1}),
  row(12, 'p12', 'GB', {targets: 1}),
  row(13, 'p13', 'GB', {rec_td: 1}, {is_td: true}),
  row(14, 'p14', 'GB', {xpm: 1}),
  row(15, 'p15', 'DET', {rush_td: 1}, {is_td: true}),
  row(16, 'p16', 'DET', {xpm: 1}),
  row(17, 'p17', 'GB', {pass_td: 1}, {is_td: true}),
  row(18, 'p18', 'GB', {xpm: 1}),
  row(19, 'p19', 'DET', {rec_td: 1}, {is_td: true}),
  row(20, 'p20', 'DET', {xpm: 1}),
];
function buildWith(anchorA, anchorH, pbp) {
  const state = { games: {}, player_info: {}, pbp_by_game: { [GID]: pbp } };
  const info = { away: 'DET', home: 'GB', away_pts: String(anchorA), home_pts: String(anchorH) };
  return factory(state, () => info);
}
const PI = { away: 'DET', home: 'GB' };
function evFor(playId, extra) {
  return Object.assign({
    gameId: GID, playId: GID + ':' + playId,
    scoreHome: null, scoreAway: null, isNullified: false,
  }, extra || {});
}
const api = buildWith(31, 28, plays);
const out = {
  beforeAnyScore: api._scoreStrForEvent(evFor('p1'), PI),
  tdPlayItself: api._scoreStrForEvent(evFor('p2'), PI),
  afterPat: api._scoreStrForEvent(evFor('p3'), PI),
  afterFg: api._scoreStrForEvent(evFor('p7'), PI),
  nonScoringKeeps: api._scoreStrForEvent(evFor('p12'), PI),
  finalPlayEqualsCurrent: api._scoreStrForEvent(evFor('p20'), PI),
  nullifiedIgnoresStamp: api._scoreStrForEvent(
    evFor('p6', { isNullified: true, scoreAway: 14, scoreHome: 7 }), PI),
  stampedWins: api._scoreStrForEvent(
    evFor('p2', { scoreAway: 24, scoreHome: 21 }), PI),
};
// Unreconciled anchor (an unmodeled safety, +2 DET): derivation must refuse.
const bad = buildWith(33, 28, plays);
out.unreconciledOmits = bad._scoreStrForEvent(evFor('p7'), PI);
out.stampStillWorksWhenUnreconciled = bad._scoreStrForEvent(
  evFor('p7', { scoreAway: 10, scoreHome: 7 }), PI);
// Defensive touchdown is credited to the defending team's row.
const dst = api._playNflPoints([
  { team: 'GB', stat_line: { def_td: 1 }, is_td: true, play_state: 'VALID' },
  { team: 'DET', stat_line: { int: 1 }, is_td: false, play_state: 'VALID' },
]);
out.defTd = dst;
// TD + PAT modeled on one provider play sum to 7 for that team.
out.tdPlusPatSamePlay = api._playNflPoints([
  { team: 'DET', stat_line: { pass_td: 1 }, is_td: true, play_state: 'VALID' },
  { team: 'DET', stat_line: { rec_td: 1 }, is_td: true, play_state: 'VALID' },
  { team: 'DET', stat_line: { xpm: 1 }, is_td: false, play_state: 'VALID' },
]);
console.log(JSON.stringify(out));
"""


def test_score_line_shows_score_as_of_each_play():
    res = _run_node(_AS_OF_BODY)
    assert res["beforeAnyScore"] == "DET 0–0 GB"
    # The TD card shows the score the TD produced (before the PAT play).
    assert res["tdPlayItself"] == "DET 6–0 GB"
    assert res["afterPat"] == "DET 7–0 GB"
    assert res["afterFg"] == "DET 10–7 GB"
    # A later non-scoring play keeps the score from the last scoring play.
    assert res["nonScoringKeeps"] == "DET 17–14 GB"
    # The final play's as-of score equals the game's current score.
    assert res["finalPlayEqualsCurrent"] == "DET 31–28 GB"


def test_score_line_fallbacks_are_honest():
    res = _run_node(_AS_OF_BODY)
    # A nullified play never scored: its (pre-review) stamp is ignored and
    # the derived score after the previous valid play is shown instead.
    assert res["nullifiedIgnoresStamp"] == "DET 7–7 GB"
    # Provider stamp beats derivation when present.
    assert res["stampedWins"] == "DET 24–21 GB"
    # When the modeled plays cannot reproduce the current score (e.g. a
    # safety the stat model does not carry), no score line is printed
    # rather than a wrong running score.
    assert res["unreconciledOmits"] == ""
    assert res["stampStillWorksWhenUnreconciled"] == "DET 10–7 GB"
    assert res["defTd"] == {"GB": 6}
    assert res["tdPlusPatSamePlay"] == {"DET": 7}


def test_event_html_no_longer_reads_current_score():
    # The card's score line comes from the as-of resolver, and the old
    # current-score construction is gone from the whole client.
    assert "var scoreStr = _scoreStrForEvent(ev, pi);" in RZ
    assert "(pi.away_pts || '0')" not in RZ


# ── Item 8: extractor carry-through of per-play scores ───────────────────────


def test_espn_extractor_carries_per_play_score_stamp():
    from utils.scorezone import extract_espn_pbp_plays

    payload = {
        "gamepackageJSON": {
            "drives": {
                "previous": [
                    {
                        "team": {"abbreviation": "NE"},
                        "plays": [
                            {
                                "id": "1",
                                "text": "D.Maye pass short right to M.Hollins for 12 yards (E.Jones).",
                                "clock": {"displayValue": "2:00"},
                                "period": {"number": 4},
                                "start": {"down": 4, "distance": 9},
                                "type": {"text": "Pass"},
                            },
                            {
                                "id": "2",
                                "text": "D.Maye pass short left to M.Hollins to SEA 36 for 11 yards (E.Jones).",
                                "clock": {"displayValue": "1:40"},
                                "period": {"number": 4},
                                "start": {"down": 1, "distance": 10},
                                "type": {"text": "Pass"},
                                # ESPN gamepackage per-play score (after the play):
                                # NE is away, SEA is home in this game id.
                                "awayScore": 17,
                                "homeScore": 21,
                            },
                        ],
                    }
                ]
            }
        }
    }
    plays = extract_espn_pbp_plays(
        payload,
        "20260909_NE@SEA",
        name_to_pid={"drake maye": "11564", "malik hollins": "200"},
    )
    stamped = [p for p in plays if p["play_id"] == "2"]
    assert stamped
    assert all(p["score_away"] == 17 and p["score_home"] == 21 for p in stamped)
    unstamped = [p for p in plays if p["play_id"] == "1"]
    assert unstamped
    assert all("score_home" not in p and "score_away" not in p for p in unstamped)


def test_tank01_extractor_carries_per_play_score_stamp_when_sent():
    from utils.scorezone import extract_pbp_plays

    box = {
        "allPlayByPlay": [
            {
                "playId": "42",
                "quarter": "2",
                "clock": "8:42",
                "down": "1",
                "distance": "10",
                "yardLine": "KC 35",
                "play": "Patrick Mahomes pass complete to Travis Kelce for 17 yards",
                "homeScore": "24",
                "awayScore": "17",
                "playerStats": {
                    "a": {
                        "longName": "Patrick Mahomes",
                        "Passing": {"passYds": "17", "passCompletions": "1", "passAttempts": "1"},
                    },
                },
            }
        ]
    }
    plays = extract_pbp_plays(
        box, "20240905_KC@BAL", name_to_pid={"patrick mahomes": "4046"}
    )
    assert plays
    assert all(p["score_home"] == 24 and p["score_away"] == 17 for p in plays)


# ── Treatments 1/2/5/6/7: CSS source contracts ───────────────────────────────


def _block(selector: str) -> str:
    return CSS.split(selector + " {", 1)[1].split("}", 1)[0]


def test_color_jobs_and_dechrome_contracts():
    # MY TEAM tag + RZ tag are navy pills; green/red are data only.
    assert "background: var(--accent)" in _block(".rz-event-tag.mine")
    assert "background: var(--accent)" in _block(".rz-event-rz")
    # LIVE chip: transparent fill/border.
    assert "background: transparent" in _block(".rz-live-chip")
    # Event card: no thick colored left bar, no TD amber gradient.
    ev = _block(".rz-event")
    assert "border-left: 4px" not in ev
    assert "linear-gradient" not in _block(".rz-event.td")
    # Delta column: borderless with one hairline divider.
    delta = _block(".rz-event-delta")
    assert "border-left: 1px solid var(--rz-border)" in delta
    assert "background: transparent" in delta
    # Avatar tile: one neutral token for every position.
    assert "background: var(--rz-card-soft)" in _block(".rz-event-avatar")
    assert ".rz-av-qb" not in CSS and ".rz-av-te" not in CSS
    # TD delta: no amber override.
    assert "#d97706" not in _block(".rz-event.td .rz-event-delta")


def test_selected_pill_header_and_clearance_contracts():
    sel = _block(".rz-game-pill.is-selected")
    assert "background: var(--accent)" in sel
    assert "rz-red" not in sel
    assert "min-width: 68px" in _block(".rz-game-pill-all")
    # Sticky header: 1px hairline, shadow only when stuck.
    assert "height: 1px" in _block(".rz-header::after")
    assert "box-shadow: 0 10px 22px -12px" in _block(".rz-header.is-stuck")
    assert "_syncHeaderStuck" in RZ and "is-stuck" in RZ
    # Feed clearance + ScoreZone-scoped FAB size via the page-shell hook.
    assert "max(110px, calc(var(--dock-safe-bottom) + 66px))" in CSS
    fab = CSS.split('body:has(.page-shell[data-page="scorezone"]) .br-asst-launcher {', 1)[1]
    assert "width: 46px" in fab.split("}", 1)[0]
