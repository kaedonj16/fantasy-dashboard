"""ScoreZone My Team / Opp roster rows: per-player cumulative stat line.

Kaedon (2026-10-25): 'add the stat lines under each player on the my team/opp
tabs'. The line comes from player_info[pid].stat_line -- the Tank01 box-score
cumulative the row's own points are computed from -- formatted per position in
the same lowercase grammar as the feed's _cumeLine. A player with no stat_line
(game not started) or a genuinely empty one renders NO line, never zeros.

Also covers the additive mapper keys (pass_att / pass_cmp / fga / xpa) that
rz_stat_line_from_ps now carries from the box score so QB and K lines can
show attempts. Those keys are unscored: _lineToPts never reads them.
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


_FNS = [
    "_n", "_fmt", "_pos", "_name", "_team", "_statLine",
    "_normGameStatus", "_playerGameState", "_gameStatus", "_gameLine",
    "_posHtml", "_injuryDot", "_defPtsAllowed", "_rowStatLine",
    "_fgRate", "_fgPts", "_lineToPts", "_scoringForPid", "_playerPts",
    "_playerRowHtml", "_rosterCard",
]


def _run_node(body: str) -> dict:
    # A comment in _fgPts quotes `fgm` with backticks; neutralize them so the
    # extracted source can sit inside the factory's template literal.
    extracted = "\n".join(_extract(RZ, n) for n in _FNS).replace("`", "'")
    script = f"""
const assert = require('node:assert/strict');
const factory = new Function('_state', '_nflGameInfo', `
  {extracted}
  return {{ _rowStatLine, _playerRowHtml, _rosterCard }};
`);
{body}
"""
    out = subprocess.check_output([node, "-e", script], text=True)
    return json.loads(out.strip().splitlines()[-1])


_STATE_JS = """
const G_LIVE = '20261025_PIT@NO';
const G_PRE = '20261025_KC@SEA';
const games = {
  [G_LIVE]: { game_id: G_LIVE, away: 'PIT', home: 'NO', away_pts: '16', home_pts: '10', status: 'live', game_code: '1' },
  [G_PRE]: { game_id: G_PRE, away: 'KC', home: 'SEA', away_pts: '', home_pts: '', status: 'pregame', game_code: '0' },
};
function pi(gid, team, name, pos, stat_line) {
  const g = games[gid];
  const o = { game_id: gid, team, name, pos, away: g.away, home: g.home,
    away_pts: g.away_pts, home_pts: g.home_pts, game_code: g.game_code,
    game_status: g.status, injury_status: '' };
  if (stat_line) o.stat_line = stat_line;
  return o;
}
const state = {
  games,
  scoring: { pass_yd: 0.04, pass_td: 4, pass_int: -2, rush_yd: 0.1, rush_td: 6,
             rec: 1, rec_yd: 0.1, rec_td: 6, fum_lost: -2, fgm: 3, xpm: 1,
             sack: 1, def_int: 2, fum_rec: 2, def_td: 6 },
  player_info: {
    purdy:     pi(G_LIVE, 'PIT', 'Brock Purdy', 'QB', { pass_att: 28, pass_cmp: 19, pass_yds: 212, pass_td: 2, int: 1, carries: 5, rush_yds: 31 }),
    oldqb:     pi(G_LIVE, 'PIT', 'Old Payload QB', 'QB', { pass_yds: 180, pass_td: 1 }),
    warren:    pi(G_LIVE, 'PIT', 'Jaylen Warren', 'RB', { carries: 14, rush_yds: 52, rush_td: 1, rec: 3, rec_yds: 25, targets: 4 }),
    fumrb:     pi(G_LIVE, 'PIT', 'Fumble Back', 'RB', { carries: 10, rush_yds: 44, fum_lost: 1 }),
    lamb:      pi(G_LIVE, 'NO', 'CeeDee Lamb', 'WR', { rec: 6, rec_yds: 84, rec_td: 1, targets: 9 }),
    mcbride:   pi(G_LIVE, 'NO', 'Trey McBride', 'TE', { rec: 5, rec_yds: 58, targets: 7 }),
    tgtwr:     pi(G_LIVE, 'NO', 'Target Only', 'WR', { targets: 3 }),
    fairbairn: pi(G_LIVE, 'NO', "Ka'imi Fairbairn", 'K', { fgm: 2, fga: 2, xpm: 3, xpa: 3 }),
    oldk:      pi(G_LIVE, 'NO', 'Old Payload K', 'K', { fgm: 1, xpm: 2 }),
    pitdef:    pi(G_LIVE, 'PIT', 'Steelers', 'DEF', { sacks: 4, def_int: 2, fum_rec: 1, def_td: 0 }),
    predef:    pi(G_PRE, 'KC', 'Chiefs', 'DEF', { sacks: 0, def_int: 0, fum_rec: 0, def_td: 0 }),
    nostat:    pi(G_PRE, 'KC', 'Not Started', 'RB', null),
    zerorb:    pi(G_LIVE, 'PIT', 'Empty Line', 'RB', { carries: 0, rush_yds: 0, rec: 0, rec_yds: 0, targets: 0 }),
  },
};
const api = factory(state, (gid) => state.games[gid] || null);
"""


def test_row_stat_line_per_position():
    res = _run_node(
        _STATE_JS
        + """
const pids = ['purdy','oldqb','warren','fumrb','lamb','mcbride','tgtwr',
              'fairbairn','oldk','pitdef','predef','nostat','zerorb'];
const out = {};
pids.forEach(p => { out[p] = api._rowStatLine(p); });
console.log(JSON.stringify(out));
"""
    )
    # QB: attempts group first, rushing as its own group.
    assert res["purdy"] == "19/28 cmp, 212 yd, 2 td, 1 int · 5 car, 31 yd"
    # Older payload without attempt keys: no cmp segment, no invented numbers.
    assert res["oldqb"] == "180 yd, 1 td"
    # RB: rushing group, receiving group, targets appended.
    assert res["warren"] == "14 car, 52 yd, 1 td · 3 rec, 25 yd · 4 tgt"
    assert res["fumrb"] == "10 car, 44 yd · 1 fum lost"
    # WR / TE.
    assert res["lamb"] == "6 rec, 84 yd, 1 td · 9 tgt"
    assert res["mcbride"] == "5 rec, 58 yd · 7 tgt"
    # Targets are real recorded stats even with no catch yet.
    assert res["tgtwr"] == "3 tgt"
    # K: made/attempted when the box score recorded attempts, plain otherwise.
    assert res["fairbairn"] == "2/2 fg, 3/3 xp"
    assert res["oldk"] == "1 fg, 2 xp"
    # DEF: sacks / turnovers / points allowed from the actual game score
    # (PIT is away at NO; NO has 10, so PIT's defense allowed 10).
    assert res["pitdef"] == "4 sacks, 2 int, 1 fum rec, 10 pts allowed"
    # No line for: pregame DEF with an all-zero line, a player with no
    # stat_line at all, and a player whose line is genuinely all zeros.
    assert res["predef"] == ""
    assert res["nostat"] == ""
    assert res["zerorb"] == ""


def test_roster_card_rows_carry_stat_line_and_keep_game_line():
    res = _run_node(
        _STATE_JS
        + """
const matchup = {
  roster_id: 1,
  starters: ['purdy', 'warren', 'pitdef'],
  bench_slots: ['nostat'],
  players_points: { purdy: 20.8, warren: 16.7, pitdef: 9.0 },
};
const html = api._rosterCard(matchup);
const out = {
  hasStatsDiv: html.includes('rz-player-stats'),
  warrenLine: html.includes('14 car, 52 yd, 1 td · 3 rec, 25 yd · 4 tgt'),
  defLine: html.includes('4 sacks, 2 int, 1 fum rec, 10 pts allowed'),
  gameLineKept: html.includes('PIT 16 @ NO 10'),
  liveGameLine: html.includes('rz-meta-game live'),
  nostatHasNoStats: !html.split('data-pid="nostat"')[1].split('</div></div>')[0].includes('rz-player-stats'),
  ptsKept: html.includes('16.7'),
};
console.log(JSON.stringify(out));
"""
    )
    assert res == {
        "hasStatsDiv": True,
        "warrenLine": True,
        "defLine": True,
        "gameLineKept": True,
        "liveGameLine": True,
        "nostatHasNoStats": True,
        "ptsKept": True,
    }


def test_top_performers_builder_untouched():
    top = _extract(RZ, "_renderTopPerformers")
    assert "rz-player-stats" not in top
    assert "_rowStatLine" not in top


def test_row_stat_line_source_contracts():
    row = _extract(RZ, "_playerRowHtml")
    assert "_rowStatLine(pid)" in row
    assert "rz-player-stats" in row
    # Stat line renders above the game line inside the info block.
    assert row.index("rz-player-stats") < row.index("rz-player-meta")
    assert ".rz-player-stats" in CSS


# ── Mapper: attempts carried additively from the box score ──────────────────


def test_stat_line_maps_attempts():
    from utils.scorezone_stats import rz_stat_line_from_ps

    line = rz_stat_line_from_ps({
        "Passing": {"passYds": "212", "passCompletions": "19", "passAttempts": "28"},
        "Kicking": {"fgMade": 2, "fgAttempted": 3, "xpMade": 3, "xpAttempted": 3},
    })
    assert line["pass_cmp"] == 19.0
    assert line["pass_att"] == 28.0
    assert line["fga"] == 3.0
    assert line["xpa"] == 3.0


def test_stat_line_attempts_default_zero_and_unscored():
    from utils.scorezone_stats import rz_stat_line_from_ps

    line = rz_stat_line_from_ps({})
    assert line["pass_att"] == 0.0
    assert line["pass_cmp"] == 0.0
    assert line["fga"] == 0.0
    assert line["xpa"] == 0.0
