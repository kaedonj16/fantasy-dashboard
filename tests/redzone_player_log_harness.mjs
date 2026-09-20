// Behavioral harness for the off-Redzone-page player game-log builder.
//
// The player modal's Redzone tab used to receive an empty event feed on every
// page except /redzone, so its game log was always "No plays recorded yet" even
// when the player had plays. window._rzStubPbpEvents now derives per-player
// events from the redzone-data payload's raw pbp_by_game. This exercises the
// REAL shipped helper (extracted by source, not reimplemented).
//
// Run: node tests/redzone_player_log_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'app.js'), 'utf8');

// Extract the contiguous helper block that ends just before the stub IIFE.
const start = SRC.indexOf('window._rzScoringForPid = function');
const end = SRC.indexOf('// Default stub for non-Redzone pages');
if (start < 0 || end < 0 || end <= start) {
  throw new Error('could not locate _rzStubPbpEvents helper block in app.js');
}
const block = SRC.slice(start, end);

const window = {};
// eslint-disable-next-line no-eval
eval(block);

const scoring = { rush_yd: 0.1, rush_td: 6, rec: 1, rec_yd: 0.1, rec_td: 6, pass_yd: 0.04, pass_td: 4, pass_int: -2 };
const state = {
  scoring,
  player_info: {
    '9509': { name: 'Jeremiyah Love', team: 'ARI', pos: 'RB' },
    '4046': { name: 'Patrick Mahomes', team: 'KC', pos: 'QB' },
  },
  pbp_by_game: {
    'SEA@ARI': [
      // Resolves by explicit pid.
      { pid: '9509', name: 'Jeremiyah Love', play_id: 'p1', seq: 10, quarter: 'Q2', clock: '5:00',
        play_text: 'J.Love rush for 12 yards', stat_line: { carries: 1, rush_yds: 12 }, play_state: 'VALID' },
      // Resolves by name only (pid missing).
      { name: 'Jeremiyah Love', play_id: 'p2', seq: 22, quarter: 'Q3', clock: '11:33',
        play_text: 'J.Love rush for 14 yards', stat_line: { carries: 1, rush_yds: 14 }, play_state: 'VALID' },
      // Different player -- must be excluded.
      { pid: '0000', name: 'Someone Else', play_id: 'p3', seq: 5,
        play_text: 'other', stat_line: { carries: 1, rush_yds: 3 } },
      // Nullified play -- included, zeroed.
      { pid: '9509', name: 'Jeremiyah Love', play_id: 'p4', seq: 30,
        play_text: 'J.Love rush nullified', stat_line: { rush_yds: 20 }, play_state: 'NULLIFIED', is_no_play: true },
    ],
  },
};

// 1. Love's three plays are found; the other player's play is not.
const love = window._rzStubPbpEvents('9509', state);
assert.equal(love.length, 3, 'expected 3 Love events, got ' + love.length);

// 2. Points score off the resolved player's league scoring.
const byId = Object.fromEntries(love.map((e) => [e.playId, e]));
assert.equal(byId.p1.pts, 1.2, 'p1 pts');
assert.equal(byId.p2.pts, 1.4, 'p2 (name-resolved) pts');

// 3. Nullified plays are surfaced as zeroed, flagged rows (so a client can
//    reverse a previously-applied contribution).
assert.equal(byId.p4.pts, 0, 'nullified play must be zeroed');
assert.equal(byId.p4.isNullified, true, 'nullified flag');
assert.equal(byId.p4.kind, 'nullified', 'nullified kind');

// 4. A player with no plays gets an empty list (not an error).
assert.deepEqual(window._rzStubPbpEvents('4046', state), [], 'QB with no plays');

// 5. Missing pbp_by_game degrades to an empty list.
assert.deepEqual(window._rzStubPbpEvents('9509', { player_info: {}, scoring }), []);
assert.deepEqual(window._rzStubPbpEvents('9509', null), []);

console.log('ALL REDZONE PLAYER LOG HARNESS CHECKS PASSED');
