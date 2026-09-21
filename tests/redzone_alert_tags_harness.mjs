// Behavioral harness for how Redzone alerts set their `mine` / `opp` flags.
//
// redzone.js is a DOM-guarded IIFE, so we can't import it. Instead we extract
// the REAL shipped `_rosterTags` + `_isMyRid` by source (brace matching) and
// evaluate them against a stubbed module scope, exactly like
// tests/redzone_status_harness.mjs. This proves the tagging that every alert,
// feed row, and history entry keys off (`mine: tags.my.has(rid)`,
// `opp: tags.opp.has(rid)`) classifies each player class correctly.
//
// Run: node tests/redzone_alert_tags_harness.mjs   (exit 0 = pass)

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

const fnSrc = ['_isMyRid', '_rosterTags'].map(extract).join('\n');

// Stubbed module scope: `_myRids` is the viewer's roster-id set, `_heroMid` /
// `_scope` gate the hero-matchup branch. `_isMyRid` closes over `_myRids`.
const factory = new Function('_myRids', '_heroMid', '_scope', `
  ${fnSrc}
  return { _rosterTags };
`);

function build(myRids, { heroMid = null, scope = 'league' } = {}) {
  return factory(new Set(myRids.map(String)), heroMid, scope);
}

// Mirror the shipped call sites (e.g. redzone.js lines ~1175, ~1643): a player's
// rid is looked up via pidToRoster, then flags are set from the tag sets.
function classify(tags, pid) {
  const rid = tags.pidToRoster[pid];
  return {
    mine: rid ? tags.my.has(rid) : false,
    opp: rid ? tags.opp.has(rid) : false,
  };
}

// A standard payload: the viewer owns roster 1, facing roster 2 in matchup m1.
// Roster 3 sits in a different matchup (m2) the viewer isn't part of. Player
// pFree appears in no matchup's roster (unrostered / free agent on the field).
function standardData() {
  return {
    matchups: [
      { matchup_id: 'm1', roster_id: 1, players: ['pMine', 'pMine2'] },
      { matchup_id: 'm1', roster_id: 2, players: ['pOpp'] },
      { matchup_id: 'm2', roster_id: 3, players: ['pOther'] },
    ],
  };
}

// ── A rostered player (viewer's own) -> mine=true, opp=false ────────────────
{
  const { _rosterTags } = build([1]);
  const tags = _rosterTags(standardData());
  assert.equal(tags.my.has('1'), true, 'viewer roster is in the my set');
  assert.deepEqual(classify(tags, 'pMine'), { mine: true, opp: false },
    'a player on the viewer roster is flagged mine, not opp');
  console.log('ok - rostered player -> mine');
}

// ── A non-rostered player (no roster) -> mine=false, opp=false ──────────────
{
  const { _rosterTags } = build([1]);
  const tags = _rosterTags(standardData());
  assert.equal('pFree' in tags.pidToRoster, false, 'unrostered pid maps to no roster');
  assert.deepEqual(classify(tags, 'pFree'), { mine: false, opp: false },
    'a player on no roster is never flagged mine or opp');
  console.log('ok - non-rostered player -> neither');
}

// ── An actual opponent (viewer's matchup, other side) -> opp=true ───────────
{
  const { _rosterTags } = build([1]);
  const tags = _rosterTags(standardData());
  assert.equal(tags.opp.has('2'), true, 'the facing roster is in the opp set');
  assert.deepEqual(classify(tags, 'pOpp'), { mine: false, opp: true },
    'a player on the facing roster is flagged opp, not mine');
  console.log('ok - actual opponent -> opp');
}

// ── A non-opponent (another matchup entirely) -> mine=false, opp=false ──────
{
  const { _rosterTags } = build([1]);
  const tags = _rosterTags(standardData());
  assert.equal(tags.my.has('3'), false, 'unrelated roster is not mine');
  assert.equal(tags.opp.has('3'), false, 'unrelated roster is not an opponent');
  assert.deepEqual(classify(tags, 'pOther'), { mine: false, opp: false },
    'a player in a matchup the viewer is not part of is neither mine nor opp');
  console.log('ok - non-opponent -> neither');
}

// ── mine and opp are mutually exclusive across the whole payload ────────────
{
  const { _rosterTags } = build([1]);
  const tags = _rosterTags(standardData());
  for (const rid of tags.my) {
    assert.equal(tags.opp.has(rid), false, `roster ${rid} cannot be both mine and opp`);
  }
  console.log('ok - mine and opp never overlap');
}

console.log('\nALL REDZONE ALERT TAG HARNESS CHECKS PASSED');
