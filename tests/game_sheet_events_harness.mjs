// Behavioral harness for the Matchups game sheet's play derivation.
//
// The sheet's Plays tab renders one game's raw pbp_by_game rows as
// ScoreZone feed events. window._rzGameEvents (app.js) derives them:
// one event per NFL play, headlined by the primary fantasy contributor,
// scored with that contributor's league scoring, newest first, with the
// score as of the play derived the way ScoreZone's _scoreStrForEvent
// derives it. This exercises the REAL shipped helper (extracted by
// source, not reimplemented).
//
// Run: node tests/game_sheet_events_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'app.js'), 'utf8');

// Extract the contiguous helper block that ends just before the stub IIFE.
const start = SRC.indexOf('window._rzScoringForPid = function');
const end = SRC.indexOf('// Default stub for non-ScoreZone pages');
if (start < 0 || end < 0 || end <= start) {
  throw new Error('could not locate the ScoreZone stub helper block in app.js');
}
const block = SRC.slice(start, end);

const window = {};
// eslint-disable-next-line no-eval
eval(block);
assert.equal(typeof window._rzGameEvents, 'function', '_rzGameEvents exported');

const scoring = {
  rush_yd: 0.1, rush_td: 6, rec: 1, rec_yd: 0.1, rec_td: 6,
  pass_yd: 0.04, pass_td: 4, pass_int: -2, fum_lost: -2, xpm: 1,
};
const state = {
  scoring,
  player_info: {
    '1': { name: 'Bryce Young', team: 'CAR', pos: 'QB' },
    '2': { name: 'Tetairoa McMillan', team: 'CAR', pos: 'WR' },
    '3': { name: 'Chuba Hubbard', team: 'CAR', pos: 'RB' },
    '4': { name: 'Jordan Love', team: 'GB', pos: 'QB' },
    '5': { name: 'Tucker Kraft', team: 'GB', pos: 'TE' },
    '6': { name: 'Ryan Fitzgerald', team: 'CAR', pos: 'K' },
    '7': { name: 'Brandon McManus', team: 'GB', pos: 'K' },
  },
  games: {
    'CAR@GB': { away: 'CAR', home: 'GB', away_pts: 7, home_pts: 7 },
    'KC@BUF': { away: 'KC', home: 'BUF', away_pts: 0, home_pts: 0 },
  },
  pbp_by_game: {
    'CAR@GB': [
      { play_id: 'p1', seq: 1, pid: '3', name: 'Chuba Hubbard', team: 'CAR',
        quarter: 'Q1', clock: '9:24', down: '2', distance: 'Goal', yard_line: 'GB 3',
        play_text: 'Chuba Hubbard run up the middle for 3 yards, touchdown',
        stat_line: { carries: 1, rush_yds: 3, rush_td: 1 }, is_td: true },
      { play_id: 'p2', seq: 2, pid: '6', name: 'Ryan Fitzgerald', team: 'CAR',
        quarter: 'Q1', clock: '9:24',
        play_text: 'Ryan Fitzgerald extra point is good',
        stat_line: { xpm: 1 } },
      // One NFL play, two fantasy contributions (passer + receiver).
      { play_id: 'p9', seq: 9, pid: '4', name: 'Jordan Love', team: 'GB',
        quarter: 'Q3', clock: '12:15', down: '1', distance: '10', yard_line: 'CAR 16',
        play_text: 'Jordan Love pass short middle to Tucker Kraft for 16 yards, touchdown',
        stat_line: { pass_yds: 16, pass_td: 1 }, is_td: true },
      { play_id: 'p9', seq: 9, pid: '5', name: 'Tucker Kraft', team: 'GB',
        quarter: 'Q3', clock: '12:15', down: '1', distance: '10', yard_line: 'CAR 16',
        play_text: 'Jordan Love pass short middle to Tucker Kraft for 16 yards, touchdown',
        stat_line: { rec: 1, rec_yds: 16, rec_td: 1, targets: 1 }, is_td: true },
      { play_id: 'p10', seq: 10, pid: '7', name: 'Brandon McManus', team: 'GB',
        quarter: 'Q3', clock: '12:15',
        play_text: 'Brandon McManus extra point is good',
        stat_line: { xpm: 1 } },
      { play_id: 'p12', seq: 12, pid: '3', name: 'Chuba Hubbard', team: 'CAR',
        quarter: 'Q3', clock: '7:50', down: '1', distance: '10', yard_line: 'CAR 32',
        play_text: 'Chuba Hubbard run up the middle for 4 yards',
        stat_line: { carries: 1, rush_yds: 4 } },
    ],
    'KC@BUF': [
      // Unresolvable player in another game: must never leak into CAR@GB.
      { play_id: 'x1', seq: 1, pid: '99', name: 'Patrick Mahomes', team: 'KC',
        quarter: 'Q1', clock: '5:00',
        play_text: 'Patrick Mahomes pass for 9 yards',
        stat_line: { pass_yds: 9 } },
    ],
  },
};

const game = state.games['CAR@GB'];
const events = window._rzGameEvents(state, 'CAR@GB', game);

// 1. One event per NFL play of the tapped game (the TD pass groups its
//    passer + receiver rows into a single event), nothing from KC@BUF.
assert.equal(events.length, 5, 'expected 5 events, got ' + events.length);
assert.ok(events.every((e) => e.gameId === 'CAR@GB'), 'all events belong to the tapped game');

// 2. Newest first by seq.
assert.deepEqual(events.map((e) => e.seq), [12, 10, 9, 2, 1], 'newest-first order');

const bySeq = Object.fromEntries(events.map((e) => [e.seq, e]));

// 3. The grouped TD pass is headlined by the receiver (primary actor),
//    scored with his line: 1 rec + 16 yds + TD = 8.6.
const td = bySeq[9];
assert.equal(td.pid, '5', 'receiver headlines the TD pass');
assert.equal(td.name, 'Tucker Kraft');
assert.equal(td.kind, 'td');
assert.equal(td.pts, 8.6, 'Kraft pts');
assert.equal(td.down, '1');
assert.equal(td.distance, '10');
assert.equal(td.yardLine, 'CAR 16');

// 4. Score as of the play, derived from the reconciled NFL-points walk:
//    Hubbard TD 6-0, PAT 7-0, Kraft TD 7-6, PAT 7-7, last gain still 7-7.
assert.deepEqual([bySeq[1].scoreAway, bySeq[1].scoreHome], [6, 0], 'TD play itself reads 6-0');
assert.deepEqual([bySeq[2].scoreAway, bySeq[2].scoreHome], [7, 0], 'PAT play reads 7-0');
assert.deepEqual([bySeq[9].scoreAway, bySeq[9].scoreHome], [7, 6], 'Kraft TD reads 7-6');
assert.deepEqual([bySeq[12].scoreAway, bySeq[12].scoreHome], [7, 7], 'latest play reads 7-7');

// 5. A plain gain: kind, points off the rusher's line.
assert.equal(bySeq[12].kind, 'gain');
assert.equal(bySeq[12].pts, 0.4, 'Hubbard gain pts');
assert.equal(bySeq[1].pts, 6.3, 'Hubbard TD pts');

// 6. Other game: only its own (unresolvable) play, so no events; and a
//    game with no pbp at all degrades to an empty list.
assert.deepEqual(window._rzGameEvents(state, 'KC@BUF', state.games['KC@BUF']), []);
assert.deepEqual(window._rzGameEvents(state, 'NOPE', {}), []);
assert.deepEqual(window._rzGameEvents({}, 'CAR@GB', game), []);
assert.deepEqual(window._rzGameEvents(null, 'CAR@GB', game), []);

// 7. When the walk cannot reconcile with the game's current score (a
//    scoring play the stat model cannot see), score-as-of is withheld
//    rather than printed wrong.
const skewed = JSON.parse(JSON.stringify(state));
skewed.games['CAR@GB'] = { away: 'CAR', home: 'GB', away_pts: 9, home_pts: 7 };
const noScore = window._rzGameEvents(skewed, 'CAR@GB', skewed.games['CAR@GB']);
assert.equal(noScore.length, 5, 'events still derive when scores cannot reconcile');
assert.ok(noScore.every((e) => e.scoreAway == null && e.scoreHome == null),
  'unreconciled walk yields no score line');

console.log('ALL GAME SHEET EVENT HARNESS CHECKS PASSED');
