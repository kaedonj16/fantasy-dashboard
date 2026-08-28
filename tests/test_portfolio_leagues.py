"""My Leagues portfolio card rendering guards."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _portfolio_fn() -> str:
    source = (ROOT / "app.py").read_text()
    return source.split("def build_portfolio_body")[1].split("\ndef ")[0]


def test_my_leagues_cards_show_a_platform_script():
    fn = _portfolio_fn()
    assert "def _plat_script(_plat):" in fn
    assert "def _lg_id(name_inner, _plat, extra=\"\"):" in fn
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
