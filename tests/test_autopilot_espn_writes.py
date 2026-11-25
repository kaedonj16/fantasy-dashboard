"""Fake-transport tests for the Autopilot ESPN write client.

Payload and response shapes mirror the Phase 0 probe's proven traffic
(envelope type ROSTER, items type LINEUP, success = HTTP 200 + status
EXECUTED, lock rejection 409 TRAN_LINEUP_LOCKED) and the roster shapes
espn_api.py parses (mRoster/mTeam entries, owners, lineupLocked,
eligibleSlots). No live calls.
"""
import copy

import pytest

from dashboard_services.providers import espn_writes

SWID = "{11111111-2222-3333-4444-555555555555}"
CREDS = {"espn_s2": "s2-value", "swid": SWID}
ACCOUNT = 42
LEAGUE = "887776065"
SEASON = 2026


def make_payload(players, *, period=13):
    """players: [{pid, name, slot, pos_id, eligible, locked}] for team 2."""
    entries = []
    for p in players:
        entries.append({
            "playerId": p["pid"],
            "lineupSlotId": p["slot"],
            "lineupLocked": bool(p.get("locked")),
            "playerPoolEntry": {
                "id": p["pid"],
                "lineupSlotId": p["slot"],
                "lineupLocked": bool(p.get("locked")),
                "player": {
                    "id": p["pid"],
                    "fullName": p["name"],
                    "eligibleSlots": p.get("eligible", []),
                    "defaultPositionId": p.get("pos_id"),
                    "injuryStatus": p.get("injury", "ACTIVE"),
                    "proTeamId": 1,
                },
            },
        })
    return {
        "id": int(LEAGUE),
        "scoringPeriodId": period,
        "teams": [{
            "id": 2,
            "name": "Cook and Ball Torture",
            "owners": [SWID],
            "roster": {"entries": entries},
        }],
        "settings": {},
    }


BASE_PLAYERS = [
    {"pid": 1, "name": "Starter RB", "slot": 2, "pos_id": 2, "eligible": [2, 23, 20]},
    {"pid": 2, "name": "Bench RB", "slot": 20, "pos_id": 2, "eligible": [2, 23, 20]},
    {"pid": 3, "name": "Flex WR", "slot": 23, "pos_id": 3, "eligible": [4, 23, 20]},
]


class FakeTransport:
    def __init__(self, payload, *, reject=None):
        self.payload = payload
        self.reject = reject
        self.posts = []
        self.gets = []

    def get_league(self, season, league_id, params):
        self.gets.append(list(params))
        return copy.deepcopy(self.payload)

    def post_transaction(self, season, league_id, payload):
        self.posts.append(payload)
        if self.reject is not None:
            return self.reject
        team = self.payload["teams"][0]
        by_pid = {e["playerId"]: e for e in team["roster"]["entries"]}
        for item in payload["items"]:
            entry = by_pid[item["playerId"]]
            entry["lineupSlotId"] = item["toLineupSlotId"]
            entry["playerPoolEntry"]["lineupSlotId"] = item["toLineupSlotId"]
        return 200, {"status": "EXECUTED", "id": "txn-9"}


@pytest.fixture
def creds(monkeypatch):
    calls = []

    def league_creds(account_id, league_id, season):
        calls.append(("league", account_id, league_id, season))
        return dict(CREDS)

    def any_creds(account_id):
        calls.append(("any", account_id))
        return dict(CREDS)

    monkeypatch.setattr(
        "dashboard_services.accounts.get_espn_league_credentials", league_creds)
    monkeypatch.setattr(
        "dashboard_services.accounts.get_any_espn_account_credentials", any_creds)
    return calls


def test_payload_builder_emits_proven_envelope():
    payload = espn_writes.build_lineup_payload(2, SWID, 13, [
        {"player_id": 1, "from_slot": 2, "to_slot": 20},
        {"player_id": 2, "from_slot": 20, "to_slot": 2},
    ])
    assert payload["type"] == "ROSTER"
    assert payload["executionType"] == "EXECUTE"
    assert payload["teamId"] == 2
    assert payload["scoringPeriodId"] == 13
    assert payload["memberId"] == SWID
    assert payload["items"] == [
        {"playerId": 1, "type": "LINEUP", "fromLineupSlotId": 2, "toLineupSlotId": 20},
        {"playerId": 2, "type": "LINEUP", "fromLineupSlotId": 20, "toLineupSlotId": 2},
    ]


def test_submit_executed_parses_as_success(creds):
    transport = FakeTransport(make_payload(BASE_PLAYERS))
    out = espn_writes.submit_lineup(
        ACCOUNT, LEAGUE, SEASON, 2, 13,
        [{"player_id": 2, "from_slot": 20, "to_slot": 2},
         {"player_id": 1, "from_slot": 2, "to_slot": 20}],
        transport=transport,
    )
    assert out["status"] == "EXECUTED"
    assert out["transaction_id"] == "txn-9"
    assert len(transport.posts) == 1
    # Credentials were resolved for the acting account only.
    assert creds == [("league", ACCOUNT, LEAGUE, SEASON)]


def test_rejection_stops_with_verbatim_message_and_no_retry(creds):
    body = {
        "status": "FAILED",
        "messages": ["TRAN_LINEUP_LOCKED: player 2 is locked"],
    }
    transport = FakeTransport(make_payload(BASE_PLAYERS), reject=(409, body))
    with pytest.raises(espn_writes.EspnWriteRejected) as excinfo:
        espn_writes.submit_lineup(
            ACCOUNT, LEAGUE, SEASON, 2, 13,
            [{"player_id": 2, "from_slot": 20, "to_slot": 2}],
            transport=transport,
        )
    assert excinfo.value.http_status == 409
    assert "TRAN_LINEUP_LOCKED" in str(excinfo.value)
    assert len(transport.posts) == 1  # no retry, ever


def test_credential_fallback_is_same_account_only(monkeypatch):
    calls = []

    def league_creds(account_id, league_id, season):
        calls.append(("league", account_id))
        return {}

    def any_creds(account_id):
        calls.append(("any", account_id))
        return dict(CREDS)

    monkeypatch.setattr(
        "dashboard_services.accounts.get_espn_league_credentials", league_creds)
    monkeypatch.setattr(
        "dashboard_services.accounts.get_any_espn_account_credentials", any_creds)
    out = espn_writes.resolve_account_credentials(ACCOUNT, LEAGUE, SEASON)
    assert out["swid"] == SWID
    assert calls == [("league", ACCOUNT), ("any", ACCOUNT)]


def test_missing_credentials_hard_fail_before_any_write(monkeypatch):
    monkeypatch.setattr(
        "dashboard_services.accounts.get_espn_league_credentials",
        lambda *a: None)
    monkeypatch.setattr(
        "dashboard_services.accounts.get_any_espn_account_credentials",
        lambda *a: None)
    transport = FakeTransport(make_payload(BASE_PLAYERS))
    with pytest.raises(espn_writes.AutopilotCredentialError):
        espn_writes.submit_lineup(
            ACCOUNT, LEAGUE, SEASON, 2, 13,
            [{"player_id": 2, "from_slot": 20, "to_slot": 2}],
            transport=transport,
        )
    assert transport.posts == []


def test_read_lineup_state_resolves_team_and_snapshot(creds):
    transport = FakeTransport(make_payload(BASE_PLAYERS))
    state = espn_writes.read_lineup_state(ACCOUNT, LEAGUE, SEASON, transport=transport)
    assert state["team_id"] == 2
    assert state["team_name"] == "Cook and Ball Torture"
    assert state["scoring_period_id"] == 13
    assert state["snapshot"] == {1: 2, 2: 20, 3: 23}
    entries = {e["player_id"]: e for e in state["entries"]}
    assert entries[1]["position"] == "RB"
    assert entries[3]["eligible_slots"] == [4, 23, 20]
    # The second read is week-scoped so slot ids are the real ones.
    assert any(("scoringPeriodId", 13) in params for params in transport.gets)


def test_verify_lineup_reports_match_and_mismatch(creds):
    transport = FakeTransport(make_payload(BASE_PLAYERS))
    espn_writes.submit_lineup(
        ACCOUNT, LEAGUE, SEASON, 2, 13,
        [{"player_id": 2, "from_slot": 20, "to_slot": 2},
         {"player_id": 1, "from_slot": 2, "to_slot": 20}],
        transport=transport,
    )
    ok, actual = espn_writes.verify_lineup(
        ACCOUNT, LEAGUE, SEASON, {1: 20, 2: 2, 3: 23}, transport=transport)
    assert ok is True
    assert actual == {1: 20, 2: 2, 3: 23}
    ok, _ = espn_writes.verify_lineup(
        ACCOUNT, LEAGUE, SEASON, {1: 2, 2: 20, 3: 23}, transport=transport)
    assert ok is False
