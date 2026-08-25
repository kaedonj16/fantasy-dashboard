"""Guards that advanced-metrics sorts keep missing values off the top."""
from functools import cmp_to_key
from pathlib import Path

_AM_PAGE = Path("dashboard_services/pages/advanced_metrics_page.py").read_text(encoding="utf-8")
_ADV = Path("data_building/advanced_metrics.py").read_text(encoding="utf-8")


def test_page_sorts_with_nulls_last_helper():
    assert "function _amCmpVal" in _AM_PAGE
    assert "function _amMissing" in _AM_PAGE
    assert "ed.byId[String(a.player_id)] ?? 0" not in _AM_PAGE


def test_leaderboard_sql_orders_nulls_last():
    assert "ORDER BY t.value DESC NULLS LAST" in _ADV
    assert "ORDER BY t.value {order} NULLS LAST" in _ADV


def _am_missing(v):
    if v is None or v == "":
        return True
    try:
        n = float(v)
    except (TypeError, ValueError):
        return True
    return n != n


def _am_cmp_val(av, bv, desc):
    a_miss, b_miss = _am_missing(av), _am_missing(bv)
    if a_miss != b_miss:
        return 1 if a_miss else -1
    if a_miss:
        return 0
    diff = float(av) - float(bv)
    return -diff if desc else diff


def test_nulls_sort_after_negative_and_positive_values():
    from functools import cmp_to_key
    rows = [None, -0.4, 1.2, 0.0, None]
    desc = sorted(rows, key=cmp_to_key(lambda a, b: _am_cmp_val(a, b, True)))
    asc = sorted(rows, key=cmp_to_key(lambda a, b: _am_cmp_val(a, b, False)))
    assert desc[:3] == [1.2, 0.0, -0.4]
    assert desc[3:] == [None, None]
    assert asc[:3] == [-0.4, 0.0, 1.2]
    assert asc[3:] == [None, None]
