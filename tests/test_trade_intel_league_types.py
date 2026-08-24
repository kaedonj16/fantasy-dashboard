"""Regression tests for true-redraft/keeper/dynasty market isolation."""
from unittest.mock import patch

import pytest

from data_building.trade_intel.league_types import LeagueType, calibration_mode
from data_building.trade_intel.trade_crawler import _leagues_to_crawl
from data_building.trade_intel.trade_value_model import _col_names, _load_pick_keys


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
    with pytest.raises(ValueError, match="Keeper"):
        _col_names(LeagueType.KEEPER, 10)


def test_redraft_columns_are_selected_by_type_zero():
    assert _col_names(LeagueType.REDRAFT, 10) == (
        "redraft_value_1qb", "redraft_value_sf"
    )
    assert _col_names(LeagueType.DYNASTY, 10) == (
        "calibrated_value_1qb", "calibrated_value_sf"
    )


def test_redraft_does_not_create_pick_unknowns():
    assert _load_pick_keys(2026, LeagueType.REDRAFT, 10) == set()


@pytest.mark.parametrize("crawl_mode", ["new", "existing", "both"])
def test_crawler_selects_redraft_and_dynasty_but_not_keeper(crawl_mode):
    conn = _FakeConn()
    with patch("data_building.trade_intel.trade_crawler.get_conn", return_value=conn):
        assert _leagues_to_crawl(50, crawl_mode, 7) == []
    compact = " ".join(conn.query.split())
    assert "league_type IN (0, 2)" in compact
    assert "league_type IN (1, 2)" not in compact
