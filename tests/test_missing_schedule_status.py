"""When NFL schedule data is missing (e.g., preseason or Tank01 down),
players must default to NOT_STARTED so projections display instead of
wall-to-wall 0.0 actuals.

This test imports from ``utils.utils`` which pulls in optional heavyweight
dependencies (BeautifulSoup) at module-import time. The repo's CI runs a
lightweight unit-test job that installs only pytest, so gate the module to
avoid import-time collection failures there.
"""

import pytest

pytest.importorskip("bs4")

from utils.utils import build_status_by_pid, STATUS_NOT_STARTED, STATUS_FINAL


_PLAYERS = {
    "4046": {"team": "KC"},
    "6794": {"team": "MIN"},
    "4034": {"team": "SF"},
}
_TEAMS = {
    "KC": {"byeWeek": 6},
    "MIN": {"byeWeek": 9},
    "SF": {"byeWeek": 9},
}


def test_empty_schedule_gives_not_started():
    statuses = build_status_by_pid(_PLAYERS, {}, _TEAMS, 1)
    for pid in _PLAYERS:
        assert statuses[pid] == STATUS_NOT_STARTED


def test_empty_schedule_defenses_not_started():
    statuses = build_status_by_pid({}, {}, _TEAMS, 1)
    for team in _TEAMS:
        assert statuses[team] == STATUS_NOT_STARTED


def test_partial_schedule_missing_team_is_final():
    games = {"KC": {"status": "pre"}}
    statuses = build_status_by_pid(_PLAYERS, games, _TEAMS, 1)
    assert statuses["4046"] == STATUS_NOT_STARTED
    assert statuses["6794"] == STATUS_FINAL
    assert statuses["4034"] == STATUS_FINAL


def test_partial_schedule_defense_bye():
    games = {"KC": {"status": "pre"}}
    statuses = build_status_by_pid({}, games, _TEAMS, 9)
    assert statuses["KC"] == STATUS_NOT_STARTED
    assert statuses["MIN"] == "BYE"
    assert statuses["SF"] == "BYE"


def test_full_schedule_pre_status():
    games = {
        "KC": {"status": "pre"},
        "MIN": {"status": "pre"},
        "SF": {"status": "in"},
    }
    statuses = build_status_by_pid(_PLAYERS, games, _TEAMS, 1)
    assert statuses["4046"] == STATUS_NOT_STARTED
    assert statuses["6794"] == STATUS_NOT_STARTED
    assert statuses["4034"] == "in_progress"
