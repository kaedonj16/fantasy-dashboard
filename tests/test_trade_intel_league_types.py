"""Regression tests for true-redraft/keeper/dynasty market isolation."""
from pathlib import Path
from unittest.mock import patch

import pytest

from data_building.trade_intel.league_types import LeagueType, calibration_mode


def _import_crawler():
    """Import the crawler only when its optional HTTP/web stack is installed."""
    pytest.importorskip("requests")
    pytest.importorskip("flask")  # keeps this module in the full-stack CI job
    from data_building.trade_intel.trade_crawler import _leagues_to_crawl
    return _leagues_to_crawl


def _import_value_model():
    """Import the scientific-stack model only in the full-stack CI job."""
    for dependency in ("numpy", "pandas", "psycopg"):
        pytest.importorskip(dependency)
    from data_building.trade_intel.trade_value_model import (
        _build_normal_equations,
        _col_names,
        _load_pick_keys,
        _write_redraft_values,
    )
    return _col_names, _load_pick_keys, _build_normal_equations, _write_redraft_values


class _FakeConn:
    def __init__(self):
        self.query = ""
        self.params = ()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.query = query
        self.params = params
        return self

    def fetchall(self):
        return []


@pytest.mark.parametrize(
    ("league_type", "expected"),
    [(LeagueType.REDRAFT, "redraft"), (LeagueType.DYNASTY, "dynasty")],
)
def test_calibration_mode_uses_sleeper_contract(league_type, expected):
    assert calibration_mode(league_type) == expected


def test_keeper_cannot_be_calibrated_as_redraft():
    with pytest.raises(ValueError, match="Keeper"):
        calibration_mode(LeagueType.KEEPER)
    col_names, _, _, _ = _import_value_model()
    with pytest.raises(ValueError, match="Keeper"):
        col_names(LeagueType.KEEPER, 10)


def test_redraft_columns_are_selected_by_type_zero():
    col_names, _, _, _ = _import_value_model()
    assert col_names(LeagueType.REDRAFT, 10) == (
        "redraft_value_1qb", "redraft_value_sf"
    )
    assert col_names(LeagueType.DYNASTY, 10) == (
        "calibrated_value_1qb", "calibrated_value_sf"
    )


def test_redraft_does_not_create_pick_unknowns():
    _, load_pick_keys, _, _ = _import_value_model()
    assert load_pick_keys(2026, LeagueType.REDRAFT, 10) == set()


def _player(player_id, side):
    return {"asset_type": "player", "player_id": player_id, "side": side}


def _pick(side):
    return {
        "asset_type": "pick", "player_id": None, "side": side,
        "pick_season": 2027, "pick_round": 1, "pick_order": None,
        "pick_slot": None,
    }


def test_redraft_rejects_entire_player_plus_pick_package():
    _, _, build, _ = _import_value_model()
    trades = [{
        "assets": [_player("a", "a"), _pick("a"), _player("b", "b")],
        "decay_weight": 1.0,
    }]
    matrix, _, stats = build(
        trades, {"a": 0, "b": 1}, 2, 2026, LeagueType.REDRAFT,
    )
    assert stats == {
        "accepted": 0, "rejected_pick": 1,
        "rejected_empty": 0, "rejected_one_sided": 0,
    }
    assert not matrix.any()


def test_dynasty_keeps_player_plus_pick_package():
    _, _, build, _ = _import_value_model()
    trades = [{
        "assets": [_player("a", "a"), _pick("a"), _player("b", "b")],
        "decay_weight": 1.0,
    }]
    index = {"a": 0, "b": 1, "pick_2027_1": 2}
    matrix, _, stats = build(
        trades, index, 3, 2026, LeagueType.DYNASTY,
    )
    assert stats["accepted"] == 1
    assert matrix.any()


@pytest.mark.parametrize(
    ("write_1qb", "write_sf", "included", "excluded"),
    [
        (True, False, "redraft_value_1qb", "redraft_value_sf"),
        (False, True, "redraft_value_sf", "redraft_value_1qb"),
    ],
)
def test_redraft_writer_updates_only_qualified_format(
    write_1qb, write_sf, included, excluded,
):
    _, _, _, write_values = _import_value_model()
    conn = _FakeConn()
    rows = [{
        "player_id": "a", "redraft_value_1qb": 100,
        "redraft_value_sf": 200,
    }]
    with patch("data_building.trade_intel.trade_value_model.get_conn", return_value=conn):
        assert write_values(
            rows, write_1qb=write_1qb, write_sf=write_sf,
        ) == 1
    compact = " ".join(conn.query.split())
    assert included in compact
    assert excluded not in compact


def test_trade_summary_keeps_legacy_count_and_exposes_total():
    """Guard the result contract used by cron/log parsers during deployment."""
    source = Path("data_building/trade_intel/trade_value_model.py").read_text()
    assert '"trades_used": M_1qb' in source
    assert '"trades_used_total": M' in source


@pytest.mark.parametrize("crawl_mode", ["new", "existing", "both"])
def test_crawler_selects_redraft_and_dynasty_but_not_keeper(crawl_mode):
    leagues_to_crawl = _import_crawler()
    conn = _FakeConn()
    with patch("data_building.trade_intel.trade_crawler.get_conn", return_value=conn):
        assert leagues_to_crawl(50, crawl_mode, 7) == []
    compact = " ".join(conn.query.split())
    assert "league_type IN (0, 2)" in compact
    assert "league_type IN (1, 2)" not in compact
