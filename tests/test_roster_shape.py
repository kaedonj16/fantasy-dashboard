"""Unit tests for the teams-page roster-shape classifier.

Pure logic (positional value shares -> a descriptive build label), so these run
without the app / DB. The label is descriptive, not predictive, so the tests pin
the recognizable shapes rather than any outcome correlation.
"""
from dashboard_services.pages.teams_page import roster_shape_label


def _pv(qb=None, rb=None, wr=None, te=None):
    return {"QB": qb or [], "RB": rb or [], "WR": wr or [], "TE": te or []}


def test_empty_roster_is_unlabeled():
    assert roster_shape_label(_pv(), is_sf=False) == ""


def test_wr_factory_when_wr_value_dominates():
    # WR-dominant but with a normal RB room (not Zero RB, no single elite back).
    v = _pv(qb=[1000], rb=[2500, 1500], wr=[3500, 3000, 2500], te=[500])
    assert roster_shape_label(v, is_sf=False) == "WR Factory"


def test_robust_rb_when_rb_value_leads():
    v = _pv(qb=[1000], rb=[3500, 3000, 2000], wr=[2000, 1000], te=[500])
    assert roster_shape_label(v, is_sf=False) == "Robust RB"


def test_zero_rb_when_rb_share_tiny_and_wr_loaded():
    v = _pv(qb=[1500], rb=[400], wr=[3000, 2800, 2500], te=[900])
    assert roster_shape_label(v, is_sf=False) == "Zero RB"


def test_hero_rb_one_elite_back_then_wr():
    # One dominant RB (>=55% of the RB room), WR share >= RB share, RB not heavy.
    v = _pv(qb=[1000], rb=[4000, 300], wr=[3000, 2500], te=[600])
    assert roster_shape_label(v, is_sf=False) == "Hero RB"


def test_te_premium_when_te_invested():
    # TE is a genuine strength: high TE share AND at least matching the RB room
    # (the tightened rule stops an RB-heavy team with one good TE from qualifying).
    v = _pv(qb=[1200], rb=[1500], wr=[2500, 1500], te=[3000])
    assert roster_shape_label(v, is_sf=False) == "TE Premium"


def test_rb_heavy_with_one_elite_te_is_not_te_premium():
    # RB share exceeds TE share -> Robust RB build, not "TE Premium".
    v = _pv(qb=[1000], rb=[3500, 2500], wr=[2000, 1500], te=[2800])
    assert roster_shape_label(v, is_sf=False) == "Robust RB"


def test_konami_code_only_in_superflex():
    v = _pv(qb=[3000, 2800], rb=[2500, 1000], wr=[2000, 1500], te=[500])
    assert roster_shape_label(v, is_sf=True) == "Konami Code"
    # Same QB weight in a 1QB league is not a Konami build.
    assert roster_shape_label(v, is_sf=False) != "Konami Code"


def test_balanced_when_no_shape_dominates():
    v = _pv(qb=[1600], rb=[1900, 1500], wr=[2000, 1600], te=[1000])
    assert roster_shape_label(v, is_sf=False) == "Balanced"
