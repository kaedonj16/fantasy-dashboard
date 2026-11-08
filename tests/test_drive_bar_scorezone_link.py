"""Drive bar -> game sheet (Kaedon, 2026-10-25; sheet, 2026-11-08).

Tapping a live drive bar on a matchup row used to open ScoreZone focused
on that drive's game; it now opens the game sheet in place instead (see
tests/test_matchup_game_sheet.py). The contract pinned here:

- app.js wraps the bar (and the "on Defense" variant) in a button carrying
  the team's abbr; the sheet resolves the game from the carousel's lite
  state. Without a game_id the bar renders unwrapped, byte-for-byte the
  pre-link markup.
- A capture-phase handler stops the tap from also firing the matchup row's
  own click handlers and opens the sheet (preventDefault: no navigation).
- The ScoreZone deep link survives as the sheet's footer link, still built
  by scorezoneGameUrl from the current URL's league scope.
- scorezone.js consumes ?game= lazily and one-shot via _applyGameLink: it
  resolves the id against the canonical pill list (_nflMatchupOptions),
  falling back to the canonical team pair for duplicate-id spellings, and
  sets the single canonical filter _filters.nfl -- the same state a pill tap
  sets. It is wired into _setState (every canonical state replacement) and
  the boot path before the first _render().
- dashboard.css gives the bar an invisible hit-area expansion and resets
  link/button chrome without touching the approved treatment-B resting look.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(rel):
    with open(os.path.join(_ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def test_drive_bar_opens_game_sheet_with_game_id():
    js = _read("static/app.js")
    assert "function scorezoneGameUrl(game)" in js
    assert "game.game_id" in js
    assert "'/scorezone?game=' + encodeURIComponent(gid)" in js
    # Both live variants (ball carrier's bar + defense indicator) open the sheet.
    assert js.count('class="mb-fld-link"') == 2
    assert js.count('<button type="button" class="mb-fld-link"') == 2
    assert js.count('aria-label="Open this game"') == 2
    assert '<a class="mb-fld-link"' not in js
    # Unwrapped fallback preserved when there is no game_id / league scope.
    assert "return gid ? '<button type=\"button\" class=\"mb-fld-link\"" in js


def test_drive_bar_click_opens_sheet_not_row_handlers():
    js = _read("static/app.js")
    start = js.index("var link = e.target && e.target.closest")
    handler = js[js.rindex("document.addEventListener('click'", 0, start):]
    handler = handler[: handler.index("}, true);")]
    assert "closest('.mb-fld-link')" in handler
    assert "e.stopPropagation()" in handler
    assert "e.preventDefault()" in handler
    assert "openGameSheet(game)" in handler
    assert "gameForTeam(lastState, team)" in handler


def test_scorezone_consumes_game_param_into_canonical_filter():
    js = _read("static/scorezone.js")
    assert "function _applyGameLink()" in js
    assert "new URLSearchParams(window.location.search).get('game')" in js
    assert "_gameLinkDone" in js  # one-shot: polls/scope switches never re-force it
    assert "_nflMatchupOptions()" in js
    assert "_filters.nfl = hit.id" in js
    # Duplicate game_id spellings of one matchup resolve by canonical pair.
    assert "_canonTeam(info.away)" in js and "_canonTeam(info.home)" in js
    # Wired into the canonical state choke point and the boot path.
    assert re.search(r"function _setState\(data\) \{\s*_state = data \|\| \{\};\s*"
                     r"_myRids = _myRidSet\(_state, _scope\);", js)
    assert "if (typeof _applyGameLink === 'function') _applyGameLink();" in js
    assert re.search(r"_applyGameLink\(\);[^\n]*\n\s*_render\(\);", js)


def test_drive_bar_link_css_hit_area_without_resting_visuals():
    css = _read("static/dashboard.css")
    link = re.search(r"\.mb-fld-link\s*\{([^}]+)\}", css)
    assert link, ".mb-fld-link rule missing"
    assert "display: block" in link.group(1)
    assert "text-decoration: none" in link.group(1)
    assert "cursor: pointer" in link.group(1)
    hit = re.search(r"\.mb-fld-link::after\s*\{([^}]+)\}", css)
    assert hit, ".mb-fld-link::after hit-area rule missing"
    assert "position: absolute" in hit.group(1)
    assert "inset: -6px 0" in hit.group(1)
    assert "background" not in hit.group(1)  # invisible: no resting visual change
    # As a <button> now, the bar also needs its native chrome reset.
    assert "button.mb-fld-link {" in css
