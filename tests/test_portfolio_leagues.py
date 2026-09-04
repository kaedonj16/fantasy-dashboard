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


def test_positional_strength_card_renders_percentiles_not_signed_deltas():
    fn = _portfolio_fn()
    assert "avg percentile across your leagues" in fn
    assert "vs. league averages" not in fn
    assert "from utils.format import ordinal" in fn
    assert "d_str = ordinal(pct_i)" in fn
    assert "left:50%" in fn
    assert "pct >= 67" in fn
    assert "pct <= 33" in fn
    # Old signed % vs median must not come back — that is what painted
    # mixed portfolios all-negative.
    assert "(ratio - 1.0) * 100" not in fn
    assert "max_ratio" not in fn


def test_undrafted_league_cards_show_draft_countdown():
    fn = _portfolio_fn()
    assert "pf-draft-cd" in fn
    assert "data-draft-ts" in fn
    assert "draft_countdown_copy" in fn
    assert "Join Draft Room →" in fn
    assert "Mock draft →" in fn or "Mock draft &rarr;" in fn
    assert "draft_countdown_copy" in fn
    assert "setInterval(tick,1000)" in fn
    # Positional rank chips must not be the predraft card body.
    pending = fn.split("if lg.get(\"pending\")")[1].split("if lg.get(\"error\")")[0]
    assert "rank_chips" not in pending
    assert "pos_user_rank" not in pending


def test_my_leagues_cards_do_not_overflow_on_mobile():
    fn = _portfolio_fn()
    assert "min-width:0" in fn
    assert "max-width:100%" in fn
    assert "grid-template-columns:minmax(0,1fr)" in fn
    assert ".pf-lg-foot .pf-arch" in fn
    assert "flex-shrink:0" in fn
    assert ".pf-lg-open" in fn
    assert "overflow:hidden;box-sizing:border-box;" in fn


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


def test_my_leagues_cards_promote_position_strength_strip():
    """Positional rank is the signature data: a quality-tinted strength strip,
    not a grey footer line. Best position is crowned only when top-third."""
    fn = _portfolio_fn()
    assert "def _pos_tier(" in fn
    assert "class='pf-lg-strength'" in fn
    assert "Position strength" in fn
    assert "class='pf-strbar'" in fn
    assert "pf-pos-chip q-" in fn
    assert ".q-good{" in fn and ".q-mid{" in fn and ".q-weak{" in fn
    assert "pc-crown" in fn
    assert "_crown_ok" in fn
    # Old grey inline footer chips must not come back.
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
