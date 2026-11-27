"""Dashboard hero "Your record" card: division record + division-aware rank.

The hero used to show a plain Wins-Losses record and a global Wins/PF rank,
while the standings rail on the same page shows "2-1 (2-0)" and seeds
division winners ahead of wild cards. The hero now uses the same helpers.
"""
from dashboard_services.pages.dashboard_page import hero_record_and_seed


_ROWS = [
    {"owner": "Alpha", "Wins": 10, "Losses": 2, "Ties": 0, "PF": 1200, "PA": 900},
    {"owner": "Bravo", "Wins": 9, "Losses": 3, "Ties": 0, "PF": 1100, "PA": 950},
    {"owner": "Charlie", "Wins": 8, "Losses": 4, "Ties": 0, "PF": 1000, "PA": 980},
    {"owner": "Delta", "Wins": 2, "Losses": 10, "Ties": 0, "PF": 600, "PA": 1300},
]
_OWNER_TO_RID = {"Alpha": "1", "Bravo": "2", "Charlie": "3", "Delta": "4"}
_BY_RID = {1: 1, 2: 1, 3: 2, 4: 2}
_DIV_RECORDS = {1: (4, 0, 0), 2: (2, 2, 0), 3: (3, 1, 0), 4: (0, 4, 0)}


def test_hero_record_shows_division_record():
    result = hero_record_and_seed(
        _ROWS, "Bravo", owner_to_rid=_OWNER_TO_RID,
        by_rid=_BY_RID, div_records=_DIV_RECORDS,
    )
    assert result is not None
    record, _seed = result
    assert record == "9-3 (2-2)"


def test_hero_rank_seeds_division_winners_first():
    # Bravo has the league's second-best record but is a wild card: both
    # division winners seed ahead, so the hero rank is 3rd, not 2nd.
    _record, seed = hero_record_and_seed(
        _ROWS, "Bravo", owner_to_rid=_OWNER_TO_RID,
        by_rid=_BY_RID, div_records=_DIV_RECORDS,
    )
    assert seed == 3
    # Charlie won division 2 at 8-4 and seeds 2nd despite fewer wins.
    _record, seed = hero_record_and_seed(
        _ROWS, "Charlie", owner_to_rid=_OWNER_TO_RID,
        by_rid=_BY_RID, div_records=_DIV_RECORDS,
    )
    assert seed == 2


def test_hero_flat_league_unchanged():
    record, seed = hero_record_and_seed(_ROWS, "Bravo", owner_to_rid=_OWNER_TO_RID)
    assert record == "9-3"
    assert seed == 2


def test_hero_single_division_stays_flat():
    by_rid = {1: 1, 2: 1, 3: 1, 4: 1}
    record, seed = hero_record_and_seed(
        _ROWS, "Bravo", owner_to_rid=_OWNER_TO_RID,
        by_rid=by_rid, div_records=_DIV_RECORDS,
    )
    assert record == "9-3"
    assert seed == 2


def test_hero_ties_and_missing_viewer():
    rows = [
        {"owner": "Alpha", "Wins": 5, "Losses": 5, "Ties": 1, "PF": 900, "PA": 900},
        {"owner": "Bravo", "Wins": 6, "Losses": 5, "Ties": 0, "PF": 800, "PA": 900},
    ]
    record, seed = hero_record_and_seed(rows, "Alpha", owner_to_rid=_OWNER_TO_RID)
    assert record == "5-5-1"
    assert seed == 2
    assert hero_record_and_seed(rows, "Nobody", owner_to_rid=_OWNER_TO_RID) is None
    assert hero_record_and_seed([], "Alpha") is None
