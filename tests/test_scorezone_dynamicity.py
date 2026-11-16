"""ScoreZone dynamicity set (2026-11-15): static contract checks.

The set is presentation-layer: motion hooks in scorezone.js / app.js and
rz CSS in dashboard.css. These tests pin the hooks by content so a future
cleanup cannot silently drop a package.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SZ = (ROOT / "static" / "scorezone.js").read_text()
APP = (ROOT / "static" / "app.js").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()
CHANGELOG = (ROOT / "dashboard_services" / "changelog.py").read_text()


def test_package1_score_countup_and_team_flash():
    assert "_dynAfterRender" in SZ
    assert "rz-score-flash-team" in SZ and "rz-score-flash-team" in CSS
    assert "--rz-flash" in SZ
    assert "rz-changed-flash" in SZ  # clock / situation highlight


def test_package1_feed_td_enter_no_wash():
    assert "rz-td-enter" in SZ and "rz-td-enter" in CSS
    assert "rz-delta-pop" in CSS


def test_package2_field_depth_and_fd_pulse():
    assert "rz-fb-rzdeep" in APP and "rz-fb-rzdeep" in CSS
    assert 'data-depth="' in APP
    assert "rz-fb-fd-pulse" in APP and "rz-fb-fd-pulse" in CSS
    assert "rz-board-heat-" in APP


def test_package3_game_state_classes():
    assert "rz-board-nailbiter" in APP and "is-nailbiter" in SZ
    assert "rz-board-blowout" in APP and "is-blowout-final" in SZ
    assert "rz-badge-pop" in APP
    assert "--rz-acc-a" in APP  # ambient wash vars ride the board


def test_package4_lead_flip_hooks():
    assert "rz-lead-flip" in SZ and "rz-lead-flip" in CSS
    assert "rz-card-flash" in SZ
    assert "_dynFlashHero" in SZ


def test_change5_sticky_bar():
    assert "_stickyBarHtml" in SZ and "rz-sticky-bar" in CSS
    # The bar's field is the Matchups drive bar (mb-fld-* markup fed with
    # the game's drive state), not the old rz-sticky-mini custom field.
    assert "mb-fld-track" in SZ and "mb-fld-fill" in SZ and "mb-fld-ball" in SZ
    assert "mb-fld-fd" in SZ  # yellow first-down tick
    assert "rz-sticky-mini" not in SZ and "rz-sticky-mini" not in CSS
    # Fixed under the ScoreZone header (nav + measured header height) and
    # shown only while the hero field board is out of view: an
    # IntersectionObserver toggles .is-on; it never sits in flow.
    # The base rule is the one opening with position: fixed (a 640px
    # width override and a reduced-motion rule share the selector).
    assert ".rz-sticky-bar {\n    position: fixed" in CSS
    bar_css = CSS.split(".rz-sticky-bar {\n    position: fixed", 1)[1].split("}", 1)[0]
    assert "--rz-header-h" in bar_css and "--rz-header-h" in SZ
    assert "IntersectionObserver" in SZ and "_bindStickyBar" in SZ
    assert ".rz-sticky-bar.is-on" in CSS
    # Animation: entry/exit rides the glide token at 250ms; drive
    # updates FLIP from the previous render's values (the bar element
    # is rebuilt every render, so transitions alone never run) and a
    # changed score takes the hero board's team-colour flash. The
    # glide transitions stay scoped to the sticky mount -- the shared
    # .mb-fld-* component itself gains no transition.
    assert "var(--ease-glide)" in bar_css and ".25s" in bar_css
    assert "_animStickyBar" in SZ and "_stickyPrev" in SZ and "_stickyCur" in SZ
    assert ".rz-sticky-bar .mb-fld-fill" in CSS and ".rz-sticky-bar .mb-fld-ball" in CSS
    assert "\n.mb-fld-fill { transition" not in CSS and "\n.mb-fld-ball { transition" not in CSS


def test_moments_and_boxscore_share_board_gutter():
    # Inside the main card the board card's side inset is 12px; the
    # Moments wrap (label + rail + clip panel) takes the same inset
    # instead of running edge to edge.
    wrap = CSS.rsplit(".rz-moments-rail-wrap {", 1)[1].split("}", 1)[0]
    assert "margin: 10px 12px 4px" in wrap
    # Shared-renderer path: the Box score button is the board's next
    # sibling and takes the board's inset too (the base rule keeps
    # width: 100% for the fallback renderer's nested button).
    sib = CSS.split(".rz-nfl-board + .rz-boxscore-btn {", 1)[1].split("}", 1)[0]
    assert "width: calc(100% - 24px)" in sib
    assert "margin-left: 12px" in sib and "margin-right: 12px" in sib


def test_change6_live_dark_borrows_real_site_theme():
    # The scoped imitation palette is gone: no #rz-root.rz-live-dark block
    # redefining rz tokens in either file.
    assert "#rz-root.rz-live-dark" not in CSS
    assert "#rz-root.rz-live-dark" not in SZ
    assert "--rz-card: #1e293b" not in CSS  # the old hardcoded card value
    # ScoreZone's toggle drives the REAL theme through the borrow helper,
    # which applies data-theme without persisting the global preference.
    assert "brBorrowTheme" in APP and "brBorrowTheme" in SZ
    borrow = APP.split("window.brBorrowTheme")[1].split("};")[0]
    assert "setAttribute('data-theme', 'dark')" in borrow
    assert "localStorage.setItem('theme'" not in borrow
    # ScoreZone's own choice stays in its own key; restore-on-exit exists
    # for both exits (soft-nav detach observer + tick backstop).
    assert "rzLiveDark" in SZ
    assert "_syncLiveDark" in SZ and "_restoreBorrowedTheme" in SZ
    assert "root.isConnected === false" in SZ
    assert "rz-theme-fade" in SZ


def test_change7_moments_rail_uses_existing_clips():
    assert "_momentsRailHtml" in SZ
    assert "rz-moments-rail" in CSS
    assert "data-rail-clip" in SZ  # the rail's own clip surface key
    assert "clipPosterUrl" in SZ    # poster only when the payload carries one
    # The rail toggles its own panel (_toggleRailClip, via the root click
    # delegation); it must never forward to the feed card's toggle, which
    # is paginated out of the DOM for most moments (2026-11-15 fix).
    assert "_toggleRailClip" in SZ
    assert "scrollIntoView({ block: 'center', behavior: 'smooth' }); t.click();" not in SZ


def test_fix2_moments_rail_requires_clip_or_poster():
    rail = SZ.split("function _momentsRailHtml")[1].split("function _dynAfterRender")[0]
    # A card requires clip fields (the feed's clip contract) or a poster;
    # clipless plays never reach the rail (they stay in the feed).
    assert "ev.clipTweetId || ev.clipVideoUrl || ev.clipPosterUrl" in rail
    # Clip-but-no-poster falls back to the player's headshot over the
    # initials, resolved like the feed avatar — not a black box.
    assert "_momentMediaHtml" in SZ
    assert "sleepercdn.com/content/nfl/players/thumb/" in SZ
    assert "rz-moment-headshot" in SZ and "rz-moment-headshot" in CSS
    # Watch only where the feed card carries the same toggle (TD clips).
    assert "ev.kind === 'td' && ev.clipTweetId && ev.playId" in rail


def test_fix3_moments_rail_respects_feed_scope():
    rail = SZ.split("function _momentsRailHtml")[1].split("function _dynAfterRender")[0]
    # The rail applies the feed's own filter predicate: My Team (ev.mine),
    # Big Plays, game / position / stat filters all match the feed.
    assert "_eventMatches(ev)" in rail


def test_change8_ambient_wash_board_only():
    assert ".rz-nfl-board::before" in CSS
    assert "--rz-acc-a" in CSS and "--rz-acc-h" in CSS


def test_theme_transition_site_wide():
    assert "theme-transition" in APP and "theme-transition" in CSS
    assert "__themeTransitionTimer" in APP  # class is removed after the switch
    # No permanent universal transition rule was introduced.
    assert "* { transition: all" not in CSS


def test_reduced_motion_and_changelog():
    assert "prefers-reduced-motion: reduce" in CSS
    assert "ScoreZone now feels live" in CHANGELOG
