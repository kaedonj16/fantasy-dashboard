// Behavioral harness for the Redzone frontend status resolvers.
//
// redzone.js is a DOM-guarded IIFE, so we can't import it. Instead we extract
// the four pure resolver functions by source and evaluate them against stubbed
// module state. This exercises the REAL shipped logic (not a reimplementation)
// for the game-status / matchup-state matrix in the task's §22 tests.
//
// Run: node tests/redzone_status_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'redzone.js'), 'utf8');

// Extract `function NAME( ... ) { ... }` with brace matching.
function extract(name) {
  const marker = `function ${name}(`;
  const start = SRC.indexOf(marker);
  if (start < 0) throw new Error(`function ${name} not found in redzone.js`);
  let i = SRC.indexOf('{', start);
  let depth = 0;
  for (let j = i; j < SRC.length; j++) {
    const c = SRC[j];
    if (c === '{') depth++;
    else if (c === '}') {
      depth--;
      if (depth === 0) return SRC.slice(start, j + 1);
    }
  }
  throw new Error(`unbalanced braces extracting ${name}`);
}

const fnSrc = ['_normGameStatus', '_playerGameState', '_sideCounts', '_matchupState']
  .map(extract).join('\n');

// Stubbed module scope. _nflGameInfo returns the games map row; _state carries
// player_info + games. _fmt is unused by these fns but referenced defensively.
const factory = new Function('_state', '_nflGameInfo', `
  ${fnSrc}
  return { _normGameStatus, _playerGameState, _sideCounts, _matchupState };
`);

function build(games, playerInfo) {
  const state = { games, player_info: playerInfo };
  const nflGameInfo = (gid) => (gid && games[gid]) ? games[gid] : null;
  return factory(state, nflGameInfo);
}

// ── §1 / §22 A-C: game status normalization ────────────────────────────────
{
  const { _normGameStatus } = build({}, {});
  for (const q of ['1', '2', '3', '4']) {
    assert.equal(_normGameStatus({ status: 'live', game_quarter: q }), 'live', `Q${q} live`);
  }
  assert.equal(_normGameStatus({ status: 'halftime' }), 'halftime');
  assert.equal(_normGameStatus({ status: 'final' }), 'final');
  // No server status: derive from code. code 2 -> final, 1 -> live, 0 -> pregame.
  assert.equal(_normGameStatus({ game_code: '2' }), 'final');
  assert.equal(_normGameStatus({ game_code: '1' }), 'live');
  assert.equal(_normGameStatus({ game_code: '0' }), 'pregame');
  // Blank code never final.
  assert.equal(_normGameStatus({ game_code: '', game_status: '' }), 'unknown');
  // Game-level code wins over stale text.
  assert.equal(_normGameStatus({ game_code: '1', game_status: 'Final' }), 'live');
  console.log('ok - game status normalization (Q1-Q4/half/final/code authority)');
}

// ── §22 C: stale player record says final, game object says live -> live ────
{
  const games = { G1: { away: 'NE', home: 'SEA', status: 'live', game_quarter: '4', game_clock: '2:18' } };
  // player_info carries a stale final code, but game-level status is authoritative.
  const pi = { p1: { team: 'NE', game_id: 'G1', game_code: '2', game_status: 'Final' } };
  const { _playerGameState } = build(games, pi);
  assert.equal(_playerGameState('p1').type, 'live', 'game-level live wins over stale player final');
  console.log('ok - stale player final overridden by live game');
}

// ── §22 G: 5 upcoming vs 6 upcoming, no live -> TO PLAY, 5|6 ────────────────
{
  const games = { UP: { away: 'A', home: 'B', status: 'pregame', game_code: '0' } };
  const mkStarters = (n) => Array.from({ length: n }, (_, i) => 'u' + i);
  const pi = {};
  mkStarters(5).forEach((p) => pi['A_' + p] = { team: 'A', game_id: 'UP', game_code: '0' });
  mkStarters(6).forEach((p) => pi['B_' + p] = { team: 'B', game_id: 'UP', game_code: '0' });
  const A = { starters: mkStarters(5).map((p) => 'A_' + p) };
  const B = { starters: mkStarters(6).map((p) => 'B_' + p) };
  const { _matchupState } = build(games, pi);
  const ms = _matchupState(A, B);
  assert.equal(ms.state, 'toplay');
  assert.equal(ms.toPlayA, 5);
  assert.equal(ms.toPlayB, 6);
  console.log('ok - toplay 5|6');
}

// ── §22 H: live players present -> LIVE ─────────────────────────────────────
{
  const games = { L: { status: 'live', game_code: '1' } };
  const pi = {
    a1: { team: 'A', game_id: 'L', game_code: '1' }, a2: { team: 'A', game_id: 'L', game_code: '1' },
    b1: { team: 'B', game_id: 'L', game_code: '1' },
  };
  const { _matchupState } = build(games, pi);
  const ms = _matchupState({ starters: ['a1', 'a2'] }, { starters: ['b1'] });
  assert.equal(ms.state, 'live');
  assert.equal(ms.liveA, 2);
  assert.equal(ms.liveB, 1);
  console.log('ok - live 2|1');
}

// ── §22 I: one final + one upcoming -> NOT final (toplay) ────────────────────
{
  const games = { F: { status: 'final', game_code: '2' }, U: { status: 'pregame', game_code: '0' } };
  const pi = {
    af: { team: 'SF', game_id: 'F', game_code: '2' }, au: { team: 'BUF', game_id: 'U', game_code: '0' },
    bf: { team: 'LAR', game_id: 'F', game_code: '2' }, bu: { team: 'DAL', game_id: 'U', game_code: '0' },
  };
  const { _matchupState } = build(games, pi);
  const ms = _matchupState({ starters: ['af', 'au'] }, { starters: ['bf', 'bu'] });
  assert.equal(ms.state, 'toplay', 'one final + one upcoming must not be FINAL');
  assert.equal(ms.toPlayA, 1);
  assert.equal(ms.toPlayB, 1);
  console.log('ok - final+upcoming -> toplay (never final)');
}

// ── §22 J: all complete -> FINAL ────────────────────────────────────────────
{
  const games = { F: { status: 'final', game_code: '2' } };
  const pi = { a: { team: 'A', game_id: 'F', game_code: '2' }, b: { team: 'B', game_id: 'F', game_code: '2' } };
  const { _matchupState } = build(games, pi);
  assert.equal(_matchupState({ starters: ['a'] }, { starters: ['b'] }).state, 'final');
  console.log('ok - all complete -> final');
}

// ── §22 K: 0 upcoming vs 4 upcoming -> 0|4 ──────────────────────────────────
{
  const games = { F: { status: 'final', game_code: '2' }, U: { status: 'pregame', game_code: '0' } };
  const pi = {
    a1: { team: 'A', game_id: 'F', game_code: '2' }, a2: { team: 'A', game_id: 'F', game_code: '2' },
  };
  ['b1', 'b2', 'b3', 'b4'].forEach((p) => pi[p] = { team: 'B', game_id: 'U', game_code: '0' });
  const { _matchupState } = build(games, pi);
  const ms = _matchupState({ starters: ['a1', 'a2'] }, { starters: ['b1', 'b2', 'b3', 'b4'] });
  assert.equal(ms.toPlayA, 0);
  assert.equal(ms.toPlayB, 4);
  assert.equal(ms.state, 'toplay');
  console.log('ok - 0|4');
}

// ── §22 L: bye players not counted as to-play ───────────────────────────────
{
  // Player rostered (has team) but no game this week = bye. Not "to play".
  const games = { U: { status: 'pregame', game_code: '0' } };
  const pi = {
    bye1: { team: 'CLE', game_id: '' },          // bye
    up1: { team: 'A', game_id: 'U', game_code: '0' },
  };
  const { _sideCounts, _matchupState } = build(games, pi);
  const c = _sideCounts({ starters: ['bye1', 'up1'] });
  assert.equal(c.bye, 1);
  assert.equal(c.upcoming, 1);
  assert.equal(c.total, 1, 'bye excluded from relevant total');
  const ms = _matchupState({ starters: ['bye1'] }, { starters: ['up1'] });
  assert.equal(ms.toPlayA, 0, 'bye side has 0 to play');
  assert.equal(ms.toPlayB, 1);
  console.log('ok - bye not counted as to-play');
}

// ── §22 M: unknown player game status must not force FINAL ───────────────────
{
  // Player with a game_id but an unrecognized/blank code -> unknown, not final.
  const games = { X: { away: 'A', home: 'B', game_code: '', status: 'unknown' } };
  const pi = { a: { team: 'A', game_id: 'X', game_code: '' }, b: { team: 'B', game_id: 'X', game_code: '' } };
  const { _matchupState } = build(games, pi);
  const ms = _matchupState({ starters: ['a'] }, { starters: ['b'] });
  assert.notEqual(ms.state, 'final', 'unknown must never be FINAL');
  assert.equal(ms.state, 'unknown');
  console.log('ok - unknown never final');
}

// ── §20: both sides have live AND upcoming -> LIVE takes precedence ──────────
{
  const games = { L: { status: 'live', game_code: '1' }, U: { status: 'pregame', game_code: '0' } };
  const pi = {
    al: { team: 'A', game_id: 'L', game_code: '1' }, au: { team: 'A2', game_id: 'U', game_code: '0' },
    bl: { team: 'B', game_id: 'L', game_code: '1' }, bu: { team: 'B2', game_id: 'U', game_code: '0' },
  };
  const { _matchupState } = build(games, pi);
  assert.equal(_matchupState({ starters: ['al', 'au'] }, { starters: ['bl', 'bu'] }).state, 'live');
  console.log('ok - live precedence over upcoming');
}

// ── §20: all-bye matchup -> neutral (unknown), never false FINAL ────────────
{
  const pi = { a: { team: 'CLE', game_id: '' }, b: { team: 'PIT', game_id: '' } };
  const { _matchupState } = build({}, pi);
  assert.equal(_matchupState({ starters: ['a'] }, { starters: ['b'] }).state, 'unknown');
  console.log('ok - all-bye -> neutral, not final');
}

console.log('\nALL REDZONE STATUS HARNESS CHECKS PASSED');
