// Behavioral harness for the Redzone scoreboard-vs-PBP clock freshness fix (§4).
//
// redzone.js is a DOM-guarded IIFE, so we extract the two real functions by
// source and evaluate them against stubbed module state. This exercises the
// SHIPPED _nflGameInfoUncached() / _quarterRank() logic, not a reimplementation.
//
// Run: node tests/redzone_clock_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'redzone.js'), 'utf8');

// Extract `function NAME( ... ) { ... }` (with optional leading `async `) via
// brace matching.
function extract(name) {
  const marker = `function ${name}(`;
  const at = SRC.indexOf(marker);
  if (at < 0) throw new Error(`function ${name} not found in redzone.js`);
  const start = SRC.slice(Math.max(0, at - 6), at).endsWith('async ') ? at - 6 : at;
  let depth = 0;
  for (let j = SRC.indexOf('{', at); j < SRC.length; j++) {
    if (SRC[j] === '{') depth++;
    else if (SRC[j] === '}' && --depth === 0) return SRC.slice(start, j + 1);
  }
  throw new Error(`unbalanced braces extracting ${name}`);
}

const fnSrc = ['_quarterRank', '_nflGameInfoUncached'].map(extract).join('\n');
// _nflGameInfoUncached reads _state and (via _giCache in the cached wrapper)
// nothing else; provide a fresh _state per build.
const factory = new Function('_state', `
  ${fnSrc}
  return { _nflGameInfoUncached, _quarterRank };
`);

function build(games, playerInfo, pbp) {
  return factory({ games: games || {}, player_info: playerInfo || {}, pbp_by_game: pbp || {} });
}

// ── §4 core: a fresh scoreboard clock must survive an older PBP play ─────────
{
  // Board says Q3 8:00 (current). The newest parsed play ran at Q3 9:10.
  const games = { G: { game_id: 'G', away: 'NE', home: 'SEA', status: 'live',
    game_code: '1', game_quarter: '3', game_clock: '8:00' } };
  const pbp = { G: [
    { seq: 10, team: 'NE', quarter: '3', clock: '9:10', down: '2', distance: '7', yard_line: 'SEA 30' },
    { seq: 5,  team: 'NE', quarter: '3', clock: '11:20' },
  ] };
  const { _nflGameInfoUncached } = build(games, {}, pbp);
  const row = _nflGameInfoUncached('G');
  assert.equal(row.game_clock, '8:00', 'fresh scoreboard 8:00 must NOT be replaced by older PBP 9:10');
  assert.equal(row.game_quarter, '3', 'same-quarter PBP must not change the board quarter');
  // Situation fields the board never carries still come from PBP.
  assert.equal(row.possession, 'NE');
  assert.equal(row.down, '2');
  assert.equal(row.distance, '7');
  assert.equal(row.yard_line, 'SEA 30');
  console.log('ok - fresh scoreboard clock preserved over older PBP (8:00 not 9:10)');
}

// ── clock/quarter fill from PBP only when the board lacks them ───────────────
{
  const games = { G: { game_id: 'G', away: 'A', home: 'B', status: 'live',
    game_code: '1', game_quarter: '', game_clock: '' } };
  const pbp = { G: [{ seq: 3, team: 'A', quarter: '2', clock: '4:44' }] };
  const { _nflGameInfoUncached } = build(games, {}, pbp);
  const row = _nflGameInfoUncached('G');
  assert.equal(row.game_clock, '4:44', 'missing board clock filled from PBP');
  assert.equal(row.game_quarter, '2', 'missing board quarter filled from PBP');
  console.log('ok - PBP fills a missing board clock/quarter');
}

// ── a genuine quarter change: PBP has advanced past the board ────────────────
{
  // Board is a poll behind at Q2 1:30; PBP has already moved to Q3.
  const games = { G: { game_id: 'G', away: 'A', home: 'B', status: 'live',
    game_code: '1', game_quarter: '2', game_clock: '1:30' } };
  const pbp = { G: [{ seq: 40, team: 'B', quarter: '3', clock: '14:50' }] };
  const { _nflGameInfoUncached } = build(games, {}, pbp);
  const row = _nflGameInfoUncached('G');
  assert.equal(row.game_quarter, '3', 'later PBP period wins over a stale board quarter');
  assert.equal(row.game_clock, '14:50', 'and its clock comes along with the period change');
  console.log('ok - later PBP period (quarter change / OT) overrides a stale board');
}

// ── decreasing clock within the same quarter is NOT treated as fresher ──────
{
  // Board Q4 2:00 (authoritative current). PBP newest play at Q4 5:30 (older).
  const games = { G: { game_id: 'G', away: 'A', home: 'B', status: 'live',
    game_code: '1', game_quarter: '4', game_clock: '2:00' } };
  const pbp = { G: [{ seq: 99, team: 'A', quarter: '4', clock: '5:30' }] };
  const { _nflGameInfoUncached } = build(games, {}, pbp);
  const row = _nflGameInfoUncached('G');
  assert.equal(row.game_clock, '2:00', 'a smaller board clock is kept; PBP is not fresher just for having a larger clock');
  console.log('ok - same-quarter PBP clock never replaces the board clock');
}

// ── _quarterRank ordering (Q1-4, halftime, OT variants) ─────────────────────
{
  const { _quarterRank } = build({}, {}, {});
  assert.ok(_quarterRank('3') > _quarterRank('2'));
  assert.ok(_quarterRank('Q3') === _quarterRank('3'));
  assert.ok(_quarterRank('OT') > _quarterRank('4'));
  assert.ok(_quarterRank('2OT') > _quarterRank('OT'));
  assert.ok(_quarterRank('Half') > _quarterRank('2') && _quarterRank('Half') < _quarterRank('3'));
  assert.equal(_quarterRank(''), -1);
  assert.equal(_quarterRank('pregame'), -1);
  console.log('ok - _quarterRank orders Q1-Q4 / halftime / OT');
}

// ── OT after Q4: PBP OT must override a stale Q4 board ───────────────────────
{
  const games = { G: { game_id: 'G', away: 'A', home: 'B', status: 'live',
    game_code: '1', game_quarter: '4', game_clock: '0:00' } };
  const pbp = { G: [{ seq: 200, team: 'A', quarter: 'OT', clock: '9:58' }] };
  const { _nflGameInfoUncached } = build(games, {}, pbp);
  const row = _nflGameInfoUncached('G');
  assert.equal(row.game_quarter, 'OT');
  assert.equal(row.game_clock, '9:58');
  console.log('ok - PBP overtime overrides a stale Q4 board');
}

console.log('\nALL REDZONE CLOCK HARNESS CHECKS PASSED');
