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


def test_package2_fd_pulse_and_no_field_washes():
    # The yellow FD line and its arrival pulse stay (the mock has them).
    assert "rz-fb-fd-pulse" in APP and "rz-fb-fd-pulse" in CSS
    # The prod-only washes are gone from the field board in both renderers:
    # no possession shade, no attack-depth wash on the attacked end zone,
    # no board heat border -- the mock's bands and EZ tints stay clean.
    assert "rz-fb-shade" not in APP and "rz-fb-shade" not in SZ
    assert "rz-fb-shade" not in CSS
    assert "rz-fb-rzdeep" not in APP and "rz-fb-rzdeep" not in CSS
    assert "rz-board-heat-" not in APP and "rz-board-heat-" not in CSS
    assert 'data-depth="' not in APP


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
    assert "data-rail-card" in SZ  # the rail's own clip surface key
    assert "clipPosterUrl" in SZ    # poster only when the payload carries one
    # The whole card is the rail's clip surface (_toggleRailCard, via
    # the root click delegation); it must never forward to the feed
    # card's toggle, which is paginated out of the DOM for most moments
    # (2026-11-15 fix).
    assert "_toggleRailCard" in SZ
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
    # No Watch pill anywhere: the whole card is the tap target and grows
    # in place (Kaedon's locked design). Compact cards carry only the
    # poster, the name and the points -- no kind label.
    assert "data-rail-card" in rail and "data-rail-close" in rail
    assert "rz-moment-watch" not in SZ and "rz-moment-watch" not in CSS
    assert "rz-moment-meta" not in rail


def test_fix3_moments_rail_respects_feed_scope():
    rail = SZ.split("function _momentsRailHtml")[1].split("function _dynAfterRender")[0]
    # The rail applies the feed's own filter predicate: My Team (ev.mine),
    # Big Plays, game / position / stat filters all match the feed.
    assert "_eventMatches(ev)" in rail


def test_moments_card_grows_in_place():
    # The tapped card grows to the rail's full width over 450ms on the
    # calm ease (no overshoot); siblings slide away meanwhile.
    card_css = CSS.split(".rz-moment-card {", 1)[1].split("}", 1)[0]
    assert ".45s" in card_css and "--ease-soft" in card_css
    assert ".rz-moment-card.is-grown" in CSS and "flex-basis: 100%" in CSS
    # Siblings are never collapsed away: they slide along the scroll
    # row at full size so a switch tap can always reach them.
    assert ".rz-moment-card.is-away" not in CSS
    assert ".rz-moments-rail.has-grown" in CSS and "scroll-snap-type: none" in CSS
    # The clip panel lives inside the grown card now, not under the
    # rail, and a rebuild restores the grown card from the open key.
    rail = SZ.split("function _momentsRailHtml")[1].split("function _dynAfterRender")[0]
    assert "rz-clip-panel" in rail and "is-grown" in rail
    assert "_openClipSurface === 'rail'" in rail
    # Grow/shrink machinery with the settle guard for rapid taps.
    assert "_growRailCard" in SZ and "_shrinkRailCardEl" in SZ
    assert "_railSettle" in SZ and "_closeRailClip" in SZ


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
