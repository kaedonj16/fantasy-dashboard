"""Yahoo co-manager guids: rosters must surface managers beyond the first.

The Yahoo roster owner_id is the FIRST listed manager's guid
(_yahoo_primary_manager). A subscriber who co-manages a team and is listed
second could never match per-owner notifications (injury flip, TD alerts)
unless the remaining guids ride along as co_owners. Fixtures mirror the
format=json team shape _yahoo_manager_entries parses: a team attr list
whose `managers` value is a list of {"manager": {...}} nodes.
"""
from dashboard_services.providers import yahoo_api as Y


def _team(managers):
    return [{"managers": [{"manager": m} for m in managers]}]


def test_co_owner_ids_returns_managers_beyond_primary():
    team = _team([{"guid": "g-first", "nickname": "A"},
                  {"guid": "g-second", "nickname": "B"},
                  {"guid": "g-third", "nickname": "C"}])
    assert Y._yahoo_owner_id(team, 7) == "g-first"
    assert Y._yahoo_co_owner_ids(team) == ["g-second", "g-third"]


def test_co_owner_ids_empty_for_single_manager():
    team = _team([{"guid": "g-only", "nickname": "Solo"}])
    assert Y._yahoo_co_owner_ids(team) == []


def test_co_owner_ids_empty_without_managers():
    assert Y._yahoo_co_owner_ids([{"team_id": "7"}]) == []
