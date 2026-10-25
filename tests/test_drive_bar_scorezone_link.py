"""Drive bar -> ScoreZone deep link (Kaedon, 2026-10-25).

Tapping a live drive bar on a matchup row opens ScoreZone focused on that
drive's game. The contract pinned here:

- app.js wraps the bar (and the "on Defense" variant) in a real anchor whose
  href is /<platform>/<season>/<league_id>/scorezone?game=<game_id> -- the
  game_id the shared scorezone-data payload already keys state.games by, and
  the league scope parsed from the current URL exactly like the widget's own
  data fetch. Without a game_id the bar renders unlinked, byte-for-byte the
  pre-link markup.
- A capture-phase handler stops the tap from also firing the matchup row's
  own click handlers, without preventDefault (navigation proceeds).
- scorezone.js consumes ?game= lazily and one-shot via _applyGameLink: it
  resolves the id against the canonical pill list (_nflMatchupOptions),
  falling back to the canonical team pair for duplicate-id spellings, and
  sets the single canonical filter _filters.nfl -- the same state a pill tap
  sets. It is wired into _setState (every canonical state replacement) and
  the boot path before the first _render().
- dashboard.css gives the anchor an invisible hit-area expansion and resets
  link chrome without touching the approved treatment-B resting look.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(rel):
    with open(os.path.join(_ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def test_drive_bar_links_to_scorezone_with_game_id():
    js = _read("static/app.js")
    assert "function scorezoneGameUrl(game)" in js
    assert "game.game_id" in js
    assert "'/scorezone?game=' + encodeURIComponent(gid)" in js
    # Both live variants (ball carrier's bar + defense indicator) are links.
    assert js.count('class="mb-fld-link"') == 2
    assert 'aria-label="Open this game in ScoreZone"' in js
    # Unlinked fallback preserved when there is no game_id / league scope.
    assert "return href ? '<a class=\"mb-fld-link\"" in js


def test_drive_bar_click_does_not_fire_row_handlers():
    js = _read("static/app.js")
    m = re.search(
        r"document\.addEventListener\('click', function \(e\) \{\s*"
        r"if \(e\.target && e\.target\.closest && e\.target\.closest\('\.mb-fld-link'\)\) "
        r"e\.stopPropagation\(\);\s*\}, true\);",
        js,
    )
    assert m, "capture-phase stopPropagation guard for .mb-fld-link missing"


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
