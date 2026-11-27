"""Dashboard hero "Your record" card: division record + place in division.

The hero used to show a plain Wins-Losses record and a global Wins/PF rank,
while the standings rail on the same page shows "2-1 (2-0)" and groups
teams by division. The hero now shows the division record too, and the
rank under it is the viewer's place inside his own division.
"""
from dashboard_services.pages.dashboard_page import hero_record_and_rank


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
    result = hero_record_and_rank(
        _ROWS, "Bravo", owner_to_rid=_OWNER_TO_RID,
        by_rid=_BY_RID, div_records=_DIV_RECORDS,
    )
    assert result is not None
    record, _rank = result
    assert record == "9-3 (2-2)"


def test_hero_rank_is_place_in_own_division():
    # Bravo is 2nd in division 1: the rank reads 2nd even though his
    # playoff seed is 3rd (both division winners seed ahead of wild cards).
    _record, rank = hero_record_and_rank(
        _ROWS, "Bravo", owner_to_rid=_OWNER_TO_RID,
        by_rid=_BY_RID, div_records=_DIV_RECORDS,
    )
    assert rank == 2
    # Charlie leads division 2 at 8-4: 1st in his division, not his seed.
    _record, rank = hero_record_and_rank(
        _ROWS, "Charlie", owner_to_rid=_OWNER_TO_RID,
        by_rid=_BY_RID, div_records=_DIV_RECORDS,
    )
    assert rank == 1


def test_hero_within_division_tiebreak_uses_division_record_before_pf():
    # Tied on overall wins inside a division: the team that won the
    # division head-to-head places first even with fewer PF, matching
    # the standings table's within-division order.
    rows = [
        {"owner": "Alpha", "Wins": 5, "Losses": 5, "Ties": 0, "PF": 900, "PA": 900},
        {"owner": "Bravo", "Wins": 5, "Losses": 5, "Ties": 0, "PF": 1000, "PA": 900},
        {"owner": "Charlie", "Wins": 6, "Losses": 4, "Ties": 0, "PF": 950, "PA": 900},
        {"owner": "Delta", "Wins": 3, "Losses": 7, "Ties": 0, "PF": 800, "PA": 900},
    ]
    div_records = {1: (1, 0, 0), 2: (0, 1, 0), 3: (1, 0, 0), 4: (0, 1, 0)}
    _record, rank = hero_record_and_rank(
        rows, "Alpha", owner_to_rid=_OWNER_TO_RID,
        by_rid=_BY_RID, div_records=div_records,
    )
    assert rank == 1
    _record, rank = hero_record_and_rank(
        rows, "Bravo", owner_to_rid=_OWNER_TO_RID,
        by_rid=_BY_RID, div_records=div_records,
    )
    assert rank == 2


def test_hero_flat_league_unchanged():
    record, rank = hero_record_and_rank(_ROWS, "Bravo", owner_to_rid=_OWNER_TO_RID)
    assert record == "9-3"
    assert rank == 2


def test_hero_single_division_stays_flat():
    by_rid = {1: 1, 2: 1, 3: 1, 4: 1}
    record, rank = hero_record_and_rank(
        _ROWS, "Bravo", owner_to_rid=_OWNER_TO_RID,
        by_rid=by_rid, div_records=_DIV_RECORDS,
    )
    assert record == "9-3"
    assert rank == 2


def test_hero_ties_and_missing_viewer():
    rows = [
        {"owner": "Alpha", "Wins": 5, "Losses": 5, "Ties": 1, "PF": 900, "PA": 900},
        {"owner": "Bravo", "Wins": 6, "Losses": 5, "Ties": 0, "PF": 800, "PA": 900},
    ]
    record, rank = hero_record_and_rank(rows, "Alpha", owner_to_rid=_OWNER_TO_RID)
    assert record == "5-5-1"
    assert rank == 2
    assert hero_record_and_rank(rows, "Nobody", owner_to_rid=_OWNER_TO_RID) is None
    assert hero_record_and_rank([], "Alpha") is None
