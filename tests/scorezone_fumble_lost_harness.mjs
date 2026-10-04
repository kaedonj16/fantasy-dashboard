// Behavioral harness for ScoreZone fumble-lost scoring.
//
// Drake Maye's sack+fumble rendered 0 pts in ScoreZone but should be -2
// under standard fum_lost scoring. This exercises the REAL shipped
// _lineToPts (extracted by source, not reimplemented).
//
// Run: node tests/scorezone_fumble_lost_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'scorezone.js'), 'utf8');

const start = SRC.indexOf('function _n(x)');
const end = SRC.indexOf('function _scoringForPid');
if (start < 0 || end < 0 || end <= start) {
  throw new Error('could not locate scoring helpers in scorezone.js');
}
const block = SRC.slice(start, end);

// _fgPts only calls _fgRate when FG makes exist; our lines have none, but
// stub it loudly in case that ever changes.
function _fgRate() { throw new Error('_fgRate should not be called here'); }
// Strict-mode eval keeps function declarations local, so export explicitly.
// eslint-disable-next-line no-eval
eval(block + '\n;globalThis.__lineToPts = _lineToPts;');
const _lineToPts = globalThis.__lineToPts;

// Standard league scoring: fumbles lost are -2.
const SCORING = { fum_lost: -2.0, pass_yd: 0.04 };

// One fumble lost = -2.
assert.equal(_lineToPts({ fum_lost: 1 }, SCORING), -2);

// Combines with other stats: 84 pass yds (3.36) + 1 fumble lost (-2) = 1.36.
assert.ok(Math.abs(_lineToPts({ fum_lost: 1, pass_yds: 84 }, SCORING) - 1.36) < 1e-9);

// Zero / missing fum_lost scores 0, even when the league defines the key.
assert.equal(_lineToPts({ fum_lost: 0 }, SCORING), 0);
assert.equal(_lineToPts({ pass_yds: 84 }, SCORING), 3.36);

// A league that does not define fum_lost scores the fumble at 0.
assert.equal(_lineToPts({ fum_lost: 1 }, { pass_yd: 0.04 }), 0);

console.log('scorezone fumble-lost harness: all assertions passed');
