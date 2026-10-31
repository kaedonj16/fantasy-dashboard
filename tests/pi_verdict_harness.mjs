// Behavioral harness for the Playoff Impact verdict classifier in
// static/app.js (fetchPlayoffImpact's Future Outlook branch).
//
// Extracts the REAL classifier statements (the signal definitions plus
// the verdict if-chain) and drives them with the scenario from Kaedon's
// 2026-10-30 screenshot: DeVonta Smith (27.9) in for Quinshon Judkins
// (23.0) plus a 2027 mid 3rd in Blackedraw. Playoff odds were flat
// (99.x -> 99.9), the value delta banked +66.8, roster age ROSE
// 25.3 -> 25.5 and prime years left fell 7.0 -> 2.1, yet the card said
// "Future Lean / builds for later" because the flat branch fired on
// banking value alone and never read prime years. The fix gates the
// future labels on an aging-cost check (older roster AND a prime-years
// drop of 1.0 or more on the traded players).
//
// Run: node tests/pi_verdict_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'app.js'), 'utf8');

function extractBlock(src, marker) {
  const at = src.indexOf(marker);
  if (at < 0) throw new Error(`marker not found: ${marker}`);
  let depth = 0;
  for (let j = src.indexOf('{', at); j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}' && --depth === 0) return src.slice(at, j + 1);
  }
  throw new Error(`unbalanced braces extracting ${marker}`);
}

const listPhrase = extractBlock(SRC, 'function listPhrase(items)');

const sigStart = SRC.indexOf('const younger');
const verdictAt = SRC.indexOf('const verdict = `');
assert.ok(sigStart >= 0 && verdictAt > sigStart, 'classifier slice markers present');
const CLASSIFIER = SRC.slice(sigStart, verdictAt).trimEnd();

const classify = new Function(
  'isRedraft', 'ageDelta', 'valDelta', 'poD', 'pickD', 'primeInArg', 'primeOutArg',
  `
  const outlook = {
    age_delta: ageDelta,
    value_delta: valDelta,
    incoming: { avg_prime_years: primeInArg },
    outgoing: { avg_prime_years: primeOutArg },
  };
  ${listPhrase}
  ${CLASSIFIER}
  return { vTitle, vSub };
  `
);

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log('ok - ' + name);
}

// 1. Kaedon's exact case (flat odds, banks value, older, prime 7.0 -> 2.1):
// NOT Future Lean, and the sub tells the truth about the trade-off.
check('screenshot case: banking value with an aging cost is not Future Lean', () => {
  const out = classify(false, 0.2, 66.8, 0.4, 0, 2.1, 7.0);
  assert.equal(out.vTitle, 'Balanced');
  assert.match(out.vSub, /older with fewer prime years left/);
});

// 2. The same case in the odds-down branch must not claim long-term value
// as a Building Move gain either: with no other gain it is a downgrade.
check('screenshot case in the down branch: no Building Move on banked value alone', () => {
  const out = classify(false, 0.2, 66.8, -2.0, 0, 2.1, 7.0);
  assert.equal(out.vTitle, 'Win-Now Downgrade');
});

// 3. A genuinely younger return still earns Future Lean.
check('younger core with flat odds is Future Lean', () => {
  const out = classify(false, -0.6, 20, 0.2, 0, 5.5, 4.0);
  assert.equal(out.vTitle, 'Future Lean');
});

// 4. Banking value with an older roster but prime years roughly kept
// (drop under the 1.0 gate) still earns Future Lean.
check('banking value with prime years kept is Future Lean', () => {
  const out = classify(false, 0.3, 80, 0, 0, 4.6, 5.0);
  assert.equal(out.vTitle, 'Future Lean');
});

// 5. Building Move still fires when the long-term value gain is real
// (prime years intact).
check('odds-down banking move with prime years intact is a Building Move', () => {
  const out = classify(false, -0.1, 90, -2.0, 0, 5.5, 6.0);
  assert.equal(out.vTitle, 'Building Move');
  assert.match(out.vSub, /long-term value/);
});

// 6. No prime data: the gate cannot fire, so banking value alone still
// leans future (the gate needs evidence, not absence).
check('missing prime data leaves the old banking behavior in place', () => {
  const out = classify(false, 0.2, 66.8, 0.4, 0, null, null);
  assert.equal(out.vTitle, 'Future Lean');
});

console.log(`\nALL PI VERDICT HARNESS CHECKS PASSED (${passed})`);
