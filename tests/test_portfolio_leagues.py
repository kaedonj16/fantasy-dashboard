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
