"""My Leagues portfolio card rendering guards."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _portfolio_fn() -> str:
    source = (ROOT / "app.py").read_text()
    return source.split("def build_portfolio_body")[1].split("\ndef ")[0]


def test_my_leagues_cards_show_a_platform_script():
    fn = _portfolio_fn()
    assert "def _plat_script(_plat):" in fn
    assert "def _lg_id(name_inner, _plat, extra=\"\", arch=\"\", team=\"\"):" in fn
    assert "class='pf-lg-plat pf-lg-plat-" in fn
    assert ".pf-lg-plat-sleeper{color:#6C4BF0;}" in fn
    assert ".pf-lg-plat-espn{color:#D33A46;}" in fn
    assert ".pf-lg-plat-yahoo{color:#12A4A0;}" in fn
    assert ".pf-lg-plat-mfl{color:#3B7DD8;}" in fn
    assert ".pf-lg-plat-fleaflicker{color:#E08A1E;}" in fn
    assert fn.count("_lg_id(") >= 4
    assert '"sleeper": "Sleeper"' in fn
    assert '"espn": "ESPN"' in fn
    assert '"yahoo": "Yahoo"' in fn
    assert '"mfl": "MFL"' in fn
    assert '"fleaflicker": "Fleaflicker"' in fn


def test_predraft_league_cards_use_draft_status_pill():
    fn = _portfolio_fn()
    assert "pf-status-pill--draft" in fn
    assert "pf-lg-pending--draft" in fn
    assert "pf-pending-cta--draft" in fn
    assert "Mock draft &rarr;" in fn


def test_unlinked_team_cards_offer_link_my_team_for_every_platform():
    fn = _portfolio_fn()
    assert "linkMyTeam(" in fn
    assert "Link my team →" in fn
    modal = (ROOT / "app.py").read_text()
    js = modal.split("window.linkMyTeam=function")[1].split("function linkSetMsg")[0]
    assert "/api/link/espn/preview" in js
    assert 'id="linkEspnResult"' in modal
    assert "linkYahooPreview" in js
    assert "linkMflConnect" in js
    assert "linkFleaConnect" in js


def test_positional_strength_card_is_rank_heatmap_not_signed_deltas():
    # v25: Positional Strength is a leagues x positions rank heatmap built
    # from the summaries' in-league ranks, not percentile bars.
    fn = _portfolio_fn()
    assert "heatmap_rows" in fn
    assert "pf-hm-chip" in fn
    assert "in-league rank by position" in fn
    assert "avg percentile across your leagues" not in fn
    assert "vs. league averages" not in fn
    # Old signed % vs median must not come back — that is what painted
    # mixed portfolios all-negative.
    assert "(ratio - 1.0) * 100" not in fn
    assert "max_ratio" not in fn


def test_undrafted_league_cards_show_draft_countdown():
    fn = _portfolio_fn()
    client = (ROOT / "static" / "app.js").read_text()
    assert "pf-draft-cd" in fn
    assert "data-draft-ts" in fn
    assert "draft_countdown_copy" in fn
    assert "Join Draft Room →" in fn
    assert "Mock draft →" in fn or "Mock draft &rarr;" in fn
    assert "draft_countdown_copy" in fn
    assert "owner.countdownTimer = setInterval(tick, 1000)" in client
    # Positional rank chips must not be the predraft card body.
    pending = fn.split("if lg.get(\"pending\")")[1].split("if lg.get(\"error\")")[0]
    assert "rank_chips" not in pending
    assert "pos_user_rank" not in pending


def test_my_leagues_cards_do_not_overflow_on_mobile():
    fn = _portfolio_fn()
    assert "min-width:0" in fn
    assert "max-width:100%" in fn
    # Phone grid is a horizontal snap-scroll row (speed fold-in), so cards
    # scroll sideways instead of stacking into one tall column.
    assert "scroll-snap-type:x mandatory" in fn
    assert ".pf-lg-card{flex:0 0 85%;scroll-snap-align:start" in fn
    assert ".pf-lg-foot .pf-arch" in fn
    assert "flex-shrink:0" in fn
    assert ".pf-lg-open" in fn
    assert "overflow:hidden;box-sizing:border-box;" in fn


def test_my_leagues_card_footer_wraps_on_mobile():
    """Footer regression (2026-10-26): the nowrap footer fit with one
    posture chip, but the window chip + shape tag (#2816) beside the
    updated stamp and the Front Office / Open links overflowed the card
    at phone widths - the stamp crushed into a ~70px multi-line column
    and the links clipped at the card edge. The phone block must wrap
    the footer and give the stamp its own full-width line."""
    fn = _portfolio_fn()
    phone = fn.split("@media(max-width:640px){")[1].split('".pf-lg-foot .pf-arch')[0]
    assert ".pf-lg-foot{gap:6px;padding-top:5px;flex-wrap:wrap;row-gap:5px;}" in phone
    assert ".pf-lg-foot [data-summary-updated]{flex:1 1 100%;}" in phone


def test_my_leagues_fav_and_arch_share_a_tools_group():
    """Favorite star stays in tools; archetype badge sits in the action row."""
    fn = _portfolio_fn()
    assert "def _lg_tools(" in fn
    assert "class='pf-lg-tools'" in fn
    assert ".pf-lg-tools{display:flex;align-items:center;gap:4px;flex:0 0 auto;}" in fn
    assert "width:24px;height:24px;display:inline-flex" in fn
    assert "class='pf-lg-meta'" in fn
    # Archetype badge moved out of the identity meta line into the foot so it
    # can't collide with the platform script / owner name.
    assert ".pf-lg-foot .pf-arch{" in fn
    # Star and arch are no longer competing for the same top-right slot.
    assert fn.count("_lg_tools(") >= 3
    assert "_lg_id(name_link, plat, off_note, '', lg.get('team_name') or '')" in fn
    assert "pf-lg-fav' aria-label='Favorite league'" not in fn.split("def _lg_tools")[0]


def test_my_leagues_favorites_are_server_backed_and_pinned_without_pager():
    fn = _portfolio_fn()
    assert "data-favorite=" in fn
    assert "all_rows.sort(key=lambda lg: not bool(lg.get(\"is_favorite\")))" in fn
    assert "/api/my-leagues/favorite" in fn
    assert "data-favorite',on?'true':'false'" in fn
    # v25: the 4-per-page pager is gone; every league renders and the sort
    # control (Urgency / Record / Standings) drives the order instead.
    assert "pfLgPager" not in fn
    assert "var PAGE=4" not in fn
    assert "pfSortSel" in fn
    assert "data-rank=" in fn
    assert "data-odds" not in fn


def test_my_leagues_cards_carry_weakest_room_and_heatmap_ranks():
    """v25: cards slim down to a weakest-room line; the full rank grid moves
    to the Positional Strength heatmap in the insights band."""
    fn = _portfolio_fn()
    assert "def _pos_tier(" in fn
    assert "data-weakest" in fn
    assert "Weakest room: " in fn
    assert "pf-hm-chip q-" in fn
    assert ".q-good{" in fn and ".q-mid{" in fn and ".q-weak{" in fn
    # No "best" crown — it misread as a #1 badge on merely top-third positions.
    assert "pc-crown" not in fn
    assert "&#9650; best" not in fn
    # The old per-card strength strip must not come back on cards.
    assert "class='pf-lg-strength'" not in fn.split("league_rows += (")[-1]
    assert "class='pf-pos-chips'" not in fn.split("league_rows += (")[-1]


def test_my_leagues_standing_is_an_ordinal_place_with_a_flag():
    fn = _portfolio_fn()
    assert "from utils.format import ord_suffix" in fn
    assert "pf-lg-v--weak" in fn
    assert "class='pf-lg-l'>Standing</span>" in fn
    # A bottom-third standing is flagged via the tier helper, not hard-coded.
    assert "_pos_tier(_rank_i, _total_i)" in fn


def test_my_leagues_streak_uses_win_loss_pills_not_empty_dots():
    fn = _portfolio_fn()
    assert "pf-s-pill" in fn
    assert ".pf-s-w{background:var(--win);}" in fn
    assert "pf-streak-empty" in fn
    # The old always-on three grey dots are gone from the played-team card.
    assert "range(3 - len(streak))" not in fn


def test_my_leagues_cards_show_resolved_team_and_standings_rank():
    fn = _portfolio_fn()
    assert "class='pf-lg-team'" in fn
    assert "Regular-season standings: wins, then points for" in fn
    assert "pf-lg-l'>Standing</span>" in fn


def test_my_leagues_cards_are_compact():
    """League cards should stay dense so several fit on a phone screen."""
    fn = _portfolio_fn()
    assert "padding:10px 12px" in fn
    assert "gap:7px" in fn
    assert "width:28px;height:28px;border-radius:8px" in fn
    assert "class='pf-lg-stat'" in fn
    assert ".pf-lg-mid{display:flex;align-items:center;flex-wrap:wrap;" in fn
    assert "padding:9px 10px;gap:6px;" in fn
    assert "overflow:hidden;box-sizing:border-box;" in fn


def test_my_leagues_does_not_poll_and_reload_the_whole_page():
    """The old reconciliation loop fetched the whole account every five
    seconds and called location.reload whenever an async/stale response differed.
    Card summaries and isolated live scores may refresh; the league list may not.
    """
    fn = _portfolio_fn()
    assert "__pfLeagueReconcileTimer" not in fn
    assert "server!==shown" not in fn
    assert "location.reload()" not in fn
    assert "},5000);" not in fn
    # Existing summary cards stay mounted while their local payload hydrates.
    assert "data-summary-card" in fn
    client = (ROOT / "static" / "app.js").read_text()
    assert "owner.queued.has(key) || owner.inflight.has(key)" in client
    # Live score polling remains isolated to live slots and is non-destructive.
    assert "owner.pollTimer = setInterval" in client
    assert "if (!owner.alive() || document.hidden) return" in client
