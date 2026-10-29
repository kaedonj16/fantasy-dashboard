// Behavioral harness for the shared player-name matcher (brNameMatch).
//
// Extracts the [br-name-match] block VERBATIM from the shipped bundles and
// runs the 44-case battery from the 2026-10-29 name-matching audit against
// it: punctuation variants, concatenated names, initials, common
// misspellings, out-of-order and partial queries. Also pins the four copies
// of the block (app.js core, rankings.js, draft_room.js, cheat_sheet.js)
// byte-identical, and the ranking pins (score -> value) on common names.
//
// Run: node tests/name_match_harness.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));

function extractBlock(rel) {
  const src = readFileSync(join(here, '..', rel), 'utf8');
  const begin = src.indexOf('// [br-name-match:begin]');
  const endTok = '// [br-name-match:end]';
  const end = src.indexOf(endTok, begin);
  assert.ok(begin >= 0 && end > begin, `br-name-match block missing in ${rel}`);
  return src.slice(begin, end + endTok.length);
}

const appBlock = extractBlock('static/app.js');
for (const rel of ['static/rankings.js', 'static/draft_room.js', 'static/cheat_sheet.js']) {
  assert.equal(extractBlock(rel), appBlock, `br-name-match block in ${rel} differs from app.js`);
}

function build(block) {
  const factory = new Function('window', `
    ${block}
    return { brRankPlayersByName, brScorePlayerName, brNameFields };
  `);
  return factory({});
}

const M = build(appBlock);
const M2 = build(extractBlock('static/rankings.js')); // a port copy, standalone

const P = (name, value) => ({ name, value });
const PLAYERS = [
  P("Ja'Marr Chase", 1050), P('Jamari Thrash', 40), P('Jamar Johnson', 5),
  P('Amon-Ra St. Brown', 980), P('Equanimeous St. Brown', 60), P('Earnest Brown IV', 20),
  P('Jaxon Smith-Njigba', 990),
  P("De'Von Achane", 920), P('DeVonta Smith', 700),
  P('Travis Etienne Jr.', 650), P('Caleb Etienne', 10), P('Trevor Etienne', 30),
  P('Kenneth Walker III', 700), P('Walker Little', 15), P('PJ Walker', 25), P('Tracy Walker III', 30),
  P('Marvin Harrison Jr.', 900), P('Marvin Jones Jr.', 300), P('Marvin Mims Jr.', 200),
  P('Harrison Bryant', 90), P('Harrison Butker', 110),
  P('Bijan Robinson', 1090), P('Brian Robinson Jr.', 420), P('Ty Robinson', 35), P('David Martin-Robinson', 15),
  P('Jahmyr Gibbs', 1010), P('Antonio Gibson', 260),
  P('Justin Jefferson', 1020), P('Jermar Jefferson', 45), P('Van Jefferson', 120), P('Jeffery Simmons', 55),
  P('Christian McCaffrey', 950),
  P('Brian Thomas Jr.', 880),
  P('J.J. McCarthy', 250),
  P('A.J. Brown', 960), P('Sam Brown Jr.', 20),
  P('DJ Moore', 860), P('David Moore', 70), P('A.J. Moore Jr.', 15),
  P('CeeDee Lamb', 1000),
  P('Isiah Pacheco', 380),
  P('Rachaad White', 340),
  P('Hollywood Brown', 410), P('Tank Dell', 300),
  P('Puka Nacua', 970), P('Emeka Egbuka', 620),
  P('Tetairoa McMillan', 560), P('Stetson Bennett IV', 18),
  P('Patrick Mahomes', 780), P('Jaylen Mahoney', 12),
  P('Kyren Williams', 620),
  P('Chris Olave', 610),
  P("D'Andre Swift", 590),
  P('Josh Allen', 940), P('Allen Lazard', 150),
  P('Chase Brown', 540),
];

// The audit battery. Two expectations corrected against the real index:
// the listed name is "DJ Moore" (not "D.J. Moore") and "Hollywood Brown"
// is itself the listed name for that player.
const CASES = [
  ['jamarr', "Ja'Marr Chase"], ["ja'marr", "Ja'Marr Chase"], ['jamar chase', "Ja'Marr Chase"],
  ['amonra', 'Amon-Ra St. Brown'], ['amon-ra', 'Amon-Ra St. Brown'], ['st brown', 'Amon-Ra St. Brown'], ['st. brown', 'Amon-Ra St. Brown'],
  ['jsn', 'Jaxon Smith-Njigba'], ['smith njigba', 'Jaxon Smith-Njigba'], ['njigba', 'Jaxon Smith-Njigba'],
  ['devon achane', "De'Von Achane"], ["de'von", "De'Von Achane"],
  ['etienne jr', 'Travis Etienne Jr.'], ['walker iii', 'Kenneth Walker III'], ['harrison jr', 'Marvin Harrison Jr.'],
  ['bijon', 'Bijan Robinson'], ['gibs', 'Jahmyr Gibbs'], ['jeffeson', 'Justin Jefferson'],
  ['cmc', 'Christian McCaffrey'], ['btj', 'Brian Thomas Jr.'], ['mhj', 'Marvin Harrison Jr.'], ['jj', 'Justin Jefferson'],
  ['a.j. brown', 'A.J. Brown'], ['aj brown', 'A.J. Brown'], ['d.j. moore', 'DJ Moore'], ['dj moore', 'DJ Moore'],
  ['ceedee', 'CeeDee Lamb'], ['cee dee lamb', 'CeeDee Lamb'], ['cd lamb', 'CeeDee Lamb'],
  ['chase jamarr', "Ja'Marr Chase"], ['brown amon', 'Amon-Ra St. Brown'],
  ['isaiah pacheco', 'Isiah Pacheco'], ['rashad white', 'Rachaad White'],
  ['hollywood brown', 'Hollywood Brown'], ['tank dell', 'Tank Dell'],
  ['puka', 'Puka Nacua'], ['tet', 'Tetairoa McMillan'], ['marv', 'Marvin Harrison Jr.'],
  ['mahome', 'Patrick Mahomes'], ['kyran williams', 'Kyren Williams'], ['olavee', 'Chris Olave'],
  ['dandre swift', "D'Andre Swift"], ['jeffer', 'Justin Jefferson'], ['robinson', 'Bijan Robinson'],
];

const RANK_PINS = [
  ['robinson', 'Bijan Robinson'],
  ['st brown', 'Amon-Ra St. Brown'],
  ['marv', 'Marvin Harrison Jr.'],
  ['walker iii', 'Kenneth Walker III'],
];

let checks = 0;
for (const [q, expected] of CASES) {
  const top = M.brRankPlayersByName(PLAYERS, q, p => p.value, 5).map(p => p.name);
  assert.ok(top.includes(expected), `query ${JSON.stringify(q)}: expected ${expected} in top 5, got ${JSON.stringify(top)}`);
  checks++;
}
for (const [q, expected] of RANK_PINS) {
  const top = M.brRankPlayersByName(PLAYERS, q, p => p.value, 5).map(p => p.name);
  assert.equal(top[0], expected, `rank pin ${JSON.stringify(q)}: expected top = ${expected}, got ${JSON.stringify(top)}`);
  checks++;
}
// The rankings.js port copy scores identically on a spot subset.
for (const [q, expected] of [['jamarr', "Ja'Marr Chase"], ['cmc', 'Christian McCaffrey'], ['rashad white', 'Rachaad White'], ['walker iii', 'Kenneth Walker III']]) {
  assert.ok(M2.brScorePlayerName(expected, q) > 0, `port copy missed ${q} -> ${expected}`);
  assert.equal(M2.brScorePlayerName(expected, q), M.brScorePlayerName(expected, q));
  checks++;
}
// No-match sanity: gibberish and suffix-only queries score 0.
assert.equal(M.brScorePlayerName('Bijan Robinson', 'zzzqq'), 0);
assert.equal(M.brScorePlayerName('Bijan Robinson', 'jr'), 0);
checks += 2;

console.log(`name_match_harness: ${checks} checks passed`);
console.log('CHECKS PASSED');
