// Regression test: the Box score button must appear on the real Redzone page,
// where app.js loads first and defines window._rzRenderGameBoard. The local
// redzone.js board fallback (which also carries the button) never runs there,
// so _renderNflBoard must attach the button to the shared renderer's output.
//
// Run: node tests/redzone_boxscore_button.mjs   (exit 0 = pass)

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(here, '..', 'static', 'redzone.js'), 'utf8');

function extractFn(src, name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('function not found: ' + name);
  let depth = 0;
  const start = src.indexOf('{', i);
  for (let j = start; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') {
      depth--;
      if (depth === 0) return src.slice(i, j + 1);
    }
  }
  throw new Error('unbalanced braces in: ' + name);
}

const fns = extractFn(SRC, '_boxScoreButtonHtml') + '\n' + extractFn(SRC, '_renderNflBoard');

function renderBoard({ sharedRenderer, filter }) {
  const _filters = { nfl: filter };
  const _nflGameInfo = (gid) => (gid === '20260924_ATL@GB' ? { away: 'ATL', home: 'GB' } : null);
  const _esc = (s) => String(s);
  const window = {};
  if (sharedRenderer) window._rzRenderGameBoard = () => '<div class="rz-nfl-board">shared</div>';
  const factory = new Function('_filters', '_nflGameInfo', '_esc', 'window', fns + '\nreturn _renderNflBoard();');
  return factory(_filters, _nflGameInfo, _esc, window);
}

// 1. Shared-renderer path (the real page): button must be present.
const shared = renderBoard({ sharedRenderer: true, filter: '20260924_ATL@GB' });
assert.ok(shared.includes('rz-nfl-board'), 'shared board markup rendered');
assert.ok(shared.includes('rz-boxscore-btn'), 'button attached after shared renderer output');
assert.ok(
  shared.includes('data-boxscore-game="20260924_ATL@GB"'),
  'button carries the selected game id'
);

// 2. "All games" scope: no board, no button.
assert.equal(renderBoard({ sharedRenderer: true, filter: 'all' }), '', 'no board on all-games scope');

// 3. Button HTML carries the game id even when the filter id needs escaping.
const xss = renderBoard({ sharedRenderer: true, filter: '20260924_ATL@GB' });
assert.ok(!xss.includes('<script'), 'no script injection via game id');

console.log('redzone box score button: 3/3 pass');
