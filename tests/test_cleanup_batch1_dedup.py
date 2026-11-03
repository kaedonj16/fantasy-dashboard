"""Cleanup Batch 1 dedup: private copies now resolve to utils/core, and the
archetype_engine ordinal bug (21th/22th/23th) is fixed by the canonical."""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

from utils.core import clamp, esc, ordinal, safe_float_finite, safe_float_or_none


def test_core_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
    assert clamp(2, 0, 3) == 2


def test_core_esc():
    assert esc(None) == ""
    assert esc("a<b") == "a&lt;b"
    assert esc('a"b') == "a&quot;b"
    assert esc(0) == "0"


def test_core_safe_float_finite():
    assert safe_float_finite("3.5") == 3.5
    assert safe_float_finite(None) == 0.0
    assert safe_float_finite(None, 1.5) == 1.5
    assert safe_float_finite("") == 0.0
    assert safe_float_finite("nope", 9.0) == 9.0
    assert safe_float_finite(float("nan")) == 0.0
    assert safe_float_finite(float("inf"), 7.0) == 7.0
    assert safe_float_finite(float("-inf")) == 0.0


def test_core_safe_float_or_none():
    assert safe_float_or_none(2) == 2.0
    assert safe_float_or_none("2.5") == 2.5
    assert safe_float_or_none(None) is None
    assert safe_float_or_none("nope") is None
    assert safe_float_or_none("") is None
    assert safe_float_or_none(float("nan")) is None
    # The drifted _f contract: Inf passes through.
    assert safe_float_or_none(float("inf")) == float("inf")


def test_clamp_sites_use_core():
    from dashboard_services.assistant.evidence import _clamp as c1
    from data_building.breakout_engine.components import _clamp as c2
    from data_building.breakout_engine.weekly_breakout import _clamp as c3
    assert c1 is clamp and c2 is clamp and c3 is clamp


def test_ordinal_sites_use_core():
    from dashboard_services.pages.graphs_page import _ordinal as o1
    from dashboard_services.rookie_api import _ordinal as o2
    from utils.standings import _ordinal as o3
    from dashboard_services.landing_demos import _ordinal as o4
    assert o1 is ordinal and o2 is ordinal and o3 is ordinal and o4 is ordinal
    for fn in (o1, o2, o3, o4):
        assert fn(1) == "1st" and fn(11) == "11th" and fn(111) == "111th"
        assert fn(21) == "21st" and fn(22) == "22nd" and fn(23) == "23rd"


def test_archetype_ordinal_bug_fixed():
    # Regression: archetype_engine's private copy was {1,2,3} + f"{n}th",
    # rendering 21th / 22th / 23th. Canonical core.ordinal handles them.
    from dashboard_services.archetype_engine import _ordinal
    assert _ordinal is ordinal
    assert _ordinal(21) == "21st"
    assert _ordinal(22) == "22nd"
    assert _ordinal(23) == "23rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(12) == "12th"


def test_esc_sites_use_core():
    from dashboard_services.pages.draft_recap_page import _esc as e1
    from dashboard_services.pages.optimal_page import _esc as e2
    from utils.standings import _esc as e3
    assert e1 is esc and e2 is esc and e3 is esc


def test_f_pair_uses_core():
    from data_building.oline_ratings import _f as f1
    from data_building.oline_talent_prior import _f as f2
    assert f1 is safe_float_or_none and f2 is safe_float_or_none
    assert f1(None) is None and f1(float("nan")) is None and f1("4.5") == 4.5


def test_num_pair_uses_core():
    from dashboard_services.market_intelligence.normalize import _num as n1
    from dashboard_services.market_intelligence.team import _num as n2
    assert n1 is safe_float_or_none and n2 is safe_float_or_none
    assert n1("3") == 3.0 and n1(None) is None and n2("x") is None
