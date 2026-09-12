"""Source contracts + behavioral harness for the Redzone NFL game strip,
selected-game scoreboard, and fantasy matchup "To Play" carousel.

The frontend logic (redzone.js) is a DOM-guarded IIFE; the repo's convention is
to assert on its source. The heavier status/matchup matrix (task §22 G-M) runs
in tests/redzone_status_harness.mjs, executed here via Node when available.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REDZONE_JS = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


# ── One canonical selected-NFL-game filter (§5, §8, §9, §23) ─────────────────
def test_game_pills_bind_to_the_single_nfl_filter():
    assert "function _renderGameStrip()" in REDZONE_JS
    assert "function _wireGameStrip(" in REDZONE_JS
    # The pill click mutates the SAME _filters.nfl the Filter panel uses -- no
    # second selection state.
    strip = REDZONE_JS.split("function _wireGameStrip(", 1)[1].split(
        "function _wireHeroCards", 1
    )[0]
    assert "_filters.nfl = next" in strip
    assert "data-nfl-gid" in strip
    # There must be exactly one place that assigns _filters.nfl from a game pill.
    assert REDZONE_JS.count("_filters.nfl = next") == 1


def test_game_strip_sits_below_chip_bar_and_above_feed_header():
    render = REDZONE_JS.split("root.innerHTML =", 1)[1]
    # Order in the main card: filter chips (.rz-chip-bar) -> game strip -> board.
    i_chips = render.index("_renderFilterChips()")
    i_strip = render.index("_renderGameStrip()")
    i_board = render.index("_renderNflBoard()")
    i_panels = render.index("+ panels")
    assert i_chips < i_strip < i_board < i_panels
    # #rz-feed-hdr lives inside panels, so the strip is above it by construction.
    assert 'id="rz-feed-hdr"' in REDZONE_JS


def test_filter_panel_still_contains_matchup_options():
    # The Filter must keep the NFL matchup selector (§5, §23: do not remove it).
    assert "_nflMatchupOptions()" in REDZONE_JS
    chips = REDZONE_JS.split("function _renderFilterChips()", 1)[1].split(
        "function _posHtml", 1
    )[0]
    assert "fpRow('Matchup', 'nfl', nOpts)" in chips


def test_all_pill_present_and_selected_state_reflects_filter():
    strip = REDZONE_JS.split("function _renderGameStrip()", 1)[1].split(
        "function _renderFilterChips", 1
    )[0]
    assert 'data-nfl-gid="all"' in strip
    assert "rz-game-pill" in strip
    assert "is-selected" in strip
    assert "_filters.nfl === o.id" in strip


def test_game_pills_are_real_buttons_with_aria():
    strip = REDZONE_JS.split("function _renderGameStrip()", 1)[1].split(
        "function _renderFilterChips", 1
    )[0]
    assert 'type="button"' in strip
    assert "aria-pressed=" in strip
    assert "aria-label=" in strip


def test_selected_pill_scrolls_into_view():
    wire = REDZONE_JS.split("function _wireGameStrip(", 1)[1].split(
        "function _wireHeroCards", 1
    )[0]
    assert "scrollIntoView" in wire
    assert "inline: 'nearest'" in wire


# ── Deterministic slate ordering (§7) ────────────────────────────────────────
def test_pill_order_is_stable_by_kickoff_not_score():
    opts = REDZONE_JS.split("function _nflMatchupOptions()", 1)[1].split(
        "function _nflGameInfo", 1
    )[0]
    # Ranked live -> upcoming -> final, then kickoff epoch, then label.
    assert "rank(a.code) - rank(b.code)" in opts
    assert "a.epoch - b.epoch" in opts


# ── No down-and-distance / possession on a FINAL game (§2) ───────────────────
def test_no_down_and_distance_when_final():
    board = REDZONE_JS.split("function _nflBoardSitLine(", 1)[1].split(
        "function _renderNflBoard", 1
    )[0]
    # The situation (D&D + yard line) line only renders for a live game.
    assert "norm !== 'live'" in board

    render = REDZONE_JS.split("function _renderNflBoard()", 1)[1].split(
        "function _gamePillStatus", 1
    )[0]
    # Possession is gated on a live game only.
    assert "norm === 'live' && poss" in render


def test_scoreboard_uses_normalized_status_not_raw_final_text():
    render = REDZONE_JS.split("function _renderNflBoard()", 1)[1].split(
        "function _gamePillStatus", 1
    )[0]
    assert "_normGameStatus(g)" in render
    # The old fragile `code === '2' || ...includes('final')` player-text check
    # must be gone from the board clock line.
    clock = REDZONE_JS.split("function _nflBoardClockLine(", 1)[1].split(
        "function _renderNflBoard", 1
    )[0]
    assert "norm === 'final'" in clock
    assert "includes('final')" not in clock


# ── Matchup state is not decided by a single player's game (§10, §15) ────────
def test_matchup_state_resolver_exists_and_cards_use_it():
    assert "function _matchupState(" in REDZONE_JS
    assert "function _sideCounts(" in REDZONE_JS
    assert "function _matchupCenterHtml(" in REDZONE_JS
    # The carousel cards no longer label FINAL from anyFinal && !anyLive.
    hero = REDZONE_JS.split("function _renderHeroCards()", 1)[1].split(
        "function _renderLeaguesSummary", 1
    )[0]
    assert "_matchupCenterHtml(" in hero
    assert "anyFinal" not in hero


def test_to_play_counts_do_not_depend_on_fantasy_points():
    side = REDZONE_JS.split("function _sideCounts(", 1)[1].split(
        "function _matchupState", 1
    )[0]
    # Counts come from _playerGameState only -- never from points / scores.
    assert "_playerGameState(pid)" in side
    for banned in ("points", "_playerPts", "players_points"):
        assert banned not in side


def test_players_to_play_excludes_live_players():
    fn = REDZONE_JS.split("function _playersToPlay(", 1)[1].split("}", 1)[0]
    assert ".upcoming" in fn  # to-play == upcoming, not live


# ── CSS surface (§3, §6, §13, §16, §17, §19) ─────────────────────────────────
@pytest.mark.parametrize(
    "selector",
    [
        ".rz-game-strip",
        ".rz-game-strip-scroll",
        ".rz-game-pill",
        ".rz-game-pill.is-selected",
        ".rz-game-pill.is-live",
        ".rz-game-pill.is-final",
        ".rz-gp-status",
        ".rz-mc-center",
        ".rz-mc-state",
        ".rz-mc-to-play-counts",
        ".rz-mc-to-play-divider",
        ".rz-mc-playing-counts",
        ".rz-nfl-live-dot",
    ],
)
def test_css_classes_exist(selector):
    # Present either as its own rule or as part of a compound/descendant rule.
    assert selector in DASHBOARD_CSS


def test_matchup_card_is_three_column_grid():
    block = DASHBOARD_CSS.split(".rz-mch-matchup {", 1)[1].split("}", 1)[0]
    assert "grid-template-columns" in block
    assert "minmax(0, 1fr)" in block


def test_strip_is_single_row_horizontal_scroll_on_mobile():
    block = DASHBOARD_CSS.split(".rz-game-strip-scroll {", 1)[1].split("}", 1)[0]
    assert "overflow-x: auto" in block
    assert "scroll-snap-type: x proximity" in block
    assert "overscroll-behavior-inline: contain" in block
    assert "-webkit-overflow-scrolling: touch" in block
    # No wrap: it's a flex row with fixed-basis pills, never wrapping.
    assert "flex-wrap" not in block


def test_strip_uses_shared_real_gutters_at_both_scroll_edges():
    page = DASHBOARD_CSS.split(".rz-page {", 1)[1].split("}", 1)[0]
    strip = DASHBOARD_CSS.split(".rz-game-strip-scroll {", 1)[1].split("}", 1)[0]
    assert "--rz-content-gutter: 14px" in page
    assert "padding: 3px var(--rz-content-gutter) 7px" in strip
    assert "scroll-padding-inline: var(--rz-content-gutter)" in strip
    assert "transform" not in strip and "margin-left" not in strip
    for selector in (".rz-chip-bar {", ".rz-mt-list {", ".rz-feed-hdr {"):
        block = DASHBOARD_CSS.rsplit(selector, 1)[1].split("}", 1)[0]
        assert "var(--rz-content-gutter)" in block


def test_pregame_hides_scores_by_status_but_live_zero_scores_remain():
    strip = REDZONE_JS.split("function _renderGameStrip()", 1)[1].split(
        "function _renderFilterChips", 1
    )[0]
    assert "showScore = norm !== 'pregame'" in strip
    assert "showScore ?" in strip
    assert "aPts === 0" not in strip and "hPts === 0" not in strip
    assert "_gamePillTeamRow" in strip
    schedule = REDZONE_JS.split("function _pregameScheduleHtml()", 1)[1].split(
        "function _syncFeed", 1
    )[0]
    assert "rz-game-pill rz-pregame-pill" in schedule
    assert "_gamePillTeamRow" in schedule


def test_hero_arrows_use_scroll_tolerance_hidden_and_resize_recalculation():
    arrows = REDZONE_JS.split("function _updateHeroArrows()", 1)[1].split(
        "function _wireHeroScroll", 1
    )[0]
    assert "x <= 3" in arrows
    assert "x >= maxScroll - 3" in arrows
    assert "leftBtn.hidden = leftOff" in arrows
    assert "rightBtn.hidden = rightOff" in arrows
    assert "window.addEventListener('resize', _updateHeroArrows)" in REDZONE_JS
    off = DASHBOARD_CSS.split(".rz-hero-arrow.rz-arrow-off {", 1)[1].split("}", 1)[0]
    assert "display: none" in off


def test_selected_matchup_style_is_not_loud_green(selector=".rz-mc-hero.selected"):
    block = DASHBOARD_CSS.split(selector + " {", 1)[1].split("}", 1)[0]
    # Selection uses the neutral accent, not a green (win-state) color.
    assert "--rz-accent" in block
    assert "var(--rz-green)" not in block


# ── Behavioral matrix via Node (task §22 G-M) ────────────────────────────────
def test_frontend_status_harness_passes():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS status harness")
    harness = ROOT / "tests" / "redzone_status_harness.mjs"
    result = subprocess.run(
        [node, str(harness)], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL REDZONE STATUS HARNESS CHECKS PASSED" in result.stdout
