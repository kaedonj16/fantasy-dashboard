"""My Leagues portfolio card rendering guards."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _portfolio_fn() -> str:
    source = (ROOT / "app.py").read_text()
    return source.split("def build_portfolio_body")[1].split("\ndef ")[0]


def test_my_leagues_cards_show_platform_label_in_meta_line():
    # Direction A (2026-10-29): the platform is a plain label on the
    # header meta line (league, platform, posture), not a colored
    # script beside the name.
    fn = _portfolio_fn()
    assert "def _plat_label(_plat):" in fn
    assert "def _lg_header(" in fn
    assert "class='pf-lg-platname'" in fn
    assert fn.count("_lg_header(") >= 4
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
    assert "flex-shrink:0" in fn
    assert "overflow:hidden;box-sizing:border-box;" in fn


def test_my_leagues_cards_have_no_footer():
    """Direction A (2026-10-29): the card footer is gone entirely. No
    Front Office link, no Open button, no per-card Updated stamp (the
    #2816/#2821 footer overflow class dies with it); one Updated line
    sits above the grid instead."""
    fn = _portfolio_fn()
    assert "pf-lg-foot" not in fn
    assert "class='pf-lg-fo'" not in fn
    assert "pf-lg-open" not in fn
    assert "data-summary-updated" not in fn
    assert "data-portfolio-updated" in fn


def test_my_leagues_tools_hold_star_and_kebab_menu():
    """Direction A header: star + kebab on the top row, Remove league
    inside the kebab menu, and the meta line is a full-width block
    below the top row so it can wrap underneath the tools."""
    fn = _portfolio_fn()
    assert "def _lg_tools(" in fn
    assert "class='pf-lg-tools'" in fn
    assert ".pf-lg-tools{display:flex;align-items:center;gap:4px;flex:0 0 auto;}" in fn
    assert "width:24px;height:24px;display:inline-flex" in fn
    assert "class='pf-lg-meta'" in fn
    assert "pf-lg-menu-btn" in fn
    assert "pf-lg-menu" in fn
    assert ">Remove league</button>" in fn
    # The archetype badge and its shape tag are gone from the card.
    assert "data-posture-shape" not in fn
    assert fn.count("_lg_tools(") >= 3
    assert "_lg_header(_headline" in fn
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


def test_my_leagues_cards_carry_fix_best_rooms_and_heatmap_ranks():
    """Direction A: rooms are two columns, Fix (weakest) left and Best
    (strongest) right, both from the same pos_user_rank dict; the full
    rank grid stays in the Positional Strength heatmap."""
    fn = _portfolio_fn()
    assert "def _pos_tier(" in fn
    assert "data-rooms" in fn
    assert "Fix: " in fn and "Best: " in fn
    assert "strongest_room as _sr_fn" in fn
    assert "weakest_room as _wr_fn" in fn
    assert "Weakest room: " not in fn
    assert "pf-hm-chip q-" in fn
    assert ".q-good{" in fn and ".q-mid{" in fn and ".q-weak{" in fn
    # The old per-card strength strip must not come back on cards.
    assert "class='pf-lg-strength'" not in fn.split("league_rows += (")[-1]
    assert "class='pf-pos-chips'" not in fn.split("league_rows += (")[-1]


def test_my_leagues_standing_is_an_ordinal_place_with_a_flag():
    fn = _portfolio_fn()
    assert "from utils.format import ord_suffix" in fn
    assert "pf-lg-v--weak" in fn and "pf-lg-v--top" in fn
    assert "class='pf-lg-standing" in fn
    assert "Regular-season standings: wins, then points for" in fn
    # A bottom-third standing is flagged red via the tier helper, not
    # hard-coded; a top-3 standing is flagged green.
    assert "_pos_tier(_rank_i, _total_i)" in fn
    assert "_rank_i <= 3" in fn


def test_my_leagues_streak_is_plain_text_not_pills():
    # Direction A: the streak is plain text (L-W-W) in the record band;
    # the W/L square pills are gone.
    fn = _portfolio_fn()
    assert "pf-s-pill" not in fn
    assert "class='pf-lg-streak'" in fn
    assert '"-".join(' in fn
    assert "range(3 - len(streak))" not in fn


def test_my_leagues_cards_show_resolved_team_and_standings_rank():
    # The resolved team name headlines the card (H4); the standing in
    # the record band still carries its explanatory title.
    fn = _portfolio_fn()
    assert "_team_name = str(lg.get(\"team_name\") or \"\").strip()" in fn
    assert "html.escape(_team_name)" in fn
    assert "Regular-season standings: wins, then points for" in fn
    assert "class='pf-lg-standing" in fn


def test_my_leagues_cards_are_compact():
    """League cards should stay dense so several fit on a phone screen."""
    fn = _portfolio_fn()
    assert "padding:10px 12px" in fn
    assert "gap:7px" in fn
    # Tier-1 record in the tinted band; no crest box.
    assert ".pf-lg-rec{font-size:50px" in fn
    assert "pf-lg-crest" not in fn
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
