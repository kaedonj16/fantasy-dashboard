// Behavioral harness for ScoreZone's roster label resolver `_ownerName`.
//
// Regression: in Kaedon's Yahoo league every ScoreZone matchup card read
// "Sunday alex" on both sides. Yahoo's viewer-personalized payload stamps
// the caller's guid as the manager guid on EVERY team, so all rosters share
// one owner_id and all users share one user_id; the old
// find(user_id === owner_id) landed on the first user (team 1) for every
// roster. The shipped fix: when another roster in the same league scope
// shares the owner_id, the roster's own per-team `team_name` (now carried
// in the collect payload) wins.
//
// scorezone.js is a DOM-guarded IIFE, so we extract the REAL shipped
// `_rosterOf` + `_ownerName` by source and evaluate them against stubbed
// module state, like tests/scorezone_status_harness.mjs.
//
// Run: node tests/scorezone_owner_name_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'scorezone.js'), 'utf8');

function extract(name) {
  const marker = `function ${name}(`;
  const start = SRC.indexOf(marker);
  if (start < 0) throw new Error(`function ${name} not found in scorezone.js`);
  let depth = 0;
  for (let j = SRC.indexOf('{', start); j < SRC.length; j++) {
    const c = SRC[j];
    if (c === '{') depth++;
    else if (c === '}') {
      depth--;
      if (depth === 0) return SRC.slice(start, j + 1);
    }
  }
  throw new Error(`unbalanced braces extracting ${name}`);
}

const fnSrc = ['_rosterOf', '_ownerName'].map(extract).join('\n');
const factory = new Function('_state', `
  ${fnSrc}
  return { _ownerName };
`);

function build(rosters, users) {
  return factory({ rosters, users });
}

// ── Yahoo shape: one guid stamped on every roster and every user ──────
{
  const rosters = [
    { roster_id: 1, owner_id: 'CALLER-GUID', team_name: 'Sunday alex' },
    { roster_id: 4, owner_id: 'CALLER-GUID', team_name: 'Dicker? I hardly know her' },
    { roster_id: 6, owner_id: 'CALLER-GUID', team_name: 'Sixth Sense' },
  ];
  // Worst case: the personalized payload collapses the display name too.
  const users = [
    { user_id: 'CALLER-GUID', display_name: 'Sunday alex' },
    { user_id: 'CALLER-GUID', display_name: 'Sunday alex' },
    { user_id: 'CALLER-GUID', display_name: 'Sunday alex' },
  ];
  const { _ownerName } = build(rosters, users);
  assert.equal(_ownerName(1), 'Sunday alex');
  assert.equal(_ownerName(4), 'Dicker? I hardly know her');
  assert.equal(_ownerName(6), 'Sixth Sense');
  assert.notEqual(_ownerName(4), _ownerName(6));
}

// ── Normal league: distinct owners keep their display names ───────────
{
  const rosters = [
    { roster_id: 1, owner_id: 'u1', team_name: 'Team One' },
    { roster_id: 2, owner_id: 'u2', team_name: 'Team Two' },
  ];
  const users = [
    { user_id: 'u1', display_name: 'Owner One' },
    { user_id: 'u2', display_name: 'Owner Two' },
  ];
  const { _ownerName } = build(rosters, users);
  assert.equal(_ownerName(1), 'Owner One');
  assert.equal(_ownerName(2), 'Owner Two');
}

// ── My Leagues scope: same person owning teams in two leagues ─────────
// Roster ids are namespaced per league ("0:" / "1:"), so a shared
// owner_id across namespaces is NOT ambiguity and must not switch the
// label to the team name.
{
  const rosters = [
    { roster_id: '0:1', owner_id: 'u1', team_name: 'Alpha Squad' },
    { roster_id: '1:3', owner_id: 'u1', team_name: 'Beta Squad' },
  ];
  const users = [{ user_id: 'u1', display_name: 'Same Person' }];
  const { _ownerName } = build(rosters, users);
  assert.equal(_ownerName('0:1'), 'Same Person');
  assert.equal(_ownerName('1:3'), 'Same Person');
}

// ── Fallbacks ──────────────────────────────────────────────────────────
{
  const rosters = [{ roster_id: 7, owner_id: 'gone', team_name: 'Orphan Team' }];
  const { _ownerName } = build(rosters, []);
  assert.equal(_ownerName(7), 'Orphan Team'); // no user row: team name, not ''
  assert.equal(_ownerName(99), '');           // unknown roster stays empty
}

console.log('ALL SCOREZONE OWNER NAME HARNESS CHECKS PASSED');
