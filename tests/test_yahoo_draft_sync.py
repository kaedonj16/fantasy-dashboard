"""Yahoo draft normalize + relay helpers."""
from __future__ import annotations

from dashboard_services.draft_sync import (
    get_draft_sync_provider,
    make_yahoo_draft_id,
    normalize_yahoo_picks,
    normalize_yahoo_relay_payload,
    parse_yahoo_draft_id,
    yahoo_status_from_label,
)
from dashboard_services.yahoo_draft_relay import (
    clear_relay_snapshot,
    get_relay_snapshot,
    merge_live_with_relay,
    put_relay_snapshot,
)


def test_yahoo_draft_id_roundtrip():
    assert make_yahoo_draft_id("12345", 2026) == "yahoo_12345_2026"
    assert parse_yahoo_draft_id("yahoo_12345_2026") == ("12345", 2026)
    assert parse_yahoo_draft_id("espn_1_2026") is None


def test_yahoo_status_from_label():
    assert yahoo_status_from_label("postdraft") == "complete"
    assert yahoo_status_from_label("draft") == "drafting"
    assert yahoo_status_from_label("predraft") == "pre_draft"
    assert yahoo_status_from_label("predraft", pick_count=3) == "drafting"


def test_normalize_yahoo_picks_maps_ids():
    rows = [
        {"pick": 1, "round": 1, "player_id": "5", "team_id": "2"},
        {"pick": 2, "round": 1, "player_id": "0", "team_id": "3"},  # dropped
        {"overallPickNumber": 3, "playerId": "6", "teamId": "1", "roundId": 1},
    ]
    picks = normalize_yahoo_picks(
        rows,
        yahoo_to_canon={"5": "111", "6": "222"},
        player_lookup=lambda cid: {
            "111": {"name": "A", "position": "RB", "team": "DAL"},
            "222": {"name": "B", "position": "WR", "team": "BUF"},
        }.get(cid) or {},
        team_slot_map={"2": 1, "1": 2},
        n_teams=2,
        source="yahoo",
    )
    assert [p.overall_pick for p in picks] == [1, 3]
    assert picks[0].canonical_player_id == "111"
    assert picks[0].draft_slot == 1
    assert picks[1].canonical_player_id == "222"


def test_normalize_yahoo_relay_payload():
    out = normalize_yahoo_relay_payload(
        {
            "leagueId": "99",
            "season": 2026,
            "inProgress": True,
            "picks": [
                {"overallPickNumber": 1, "playerId": "5", "teamId": "1"},
                {"overallPickNumber": 2, "player_key": "nfl.p.6", "team_key": "461.l.99.t.2"},
            ],
        },
        yahoo_to_canon={"5": "111", "6": "222"},
        player_lookup=lambda cid: {"name": "X", "position": "QB", "team": "KC"},
    )
    assert out["source"] == "yahoo-relay"
    assert out["status"] == "drafting"
    assert len(out["picks"]) == 2
    assert out["picks"][0]["player_id"] == "111"
    assert out["picks"][1]["external_player_id"] == "6"


def test_yahoo_relay_store_isolated_from_espn():
    clear_relay_snapshot("99", 2026)
    from dashboard_services.espn_draft_relay import (
        clear_relay_snapshot as clear_espn,
        put_relay_snapshot as put_espn,
        get_relay_snapshot as get_espn,
    )
    clear_espn("99", 2026)
    put_relay_snapshot("99", 2026, {"picks": [{"pick_no": 1, "player_id": "y"}]}, source="ext")
    put_espn("99", 2026, {"picks": [{"pick_no": 1, "player_id": "e"}]}, source="ext")
    y = get_relay_snapshot("99", 2026)
    e = get_espn("99", 2026)
    assert y["payload"]["picks"][0]["player_id"] == "y"
    assert e["payload"]["picks"][0]["player_id"] == "e"
    merged = merge_live_with_relay({"picks": []}, y)
    assert merged["picks"][0]["player_id"] == "y"


def test_registry_yahoo():
    provider = get_draft_sync_provider("yahoo")
    assert provider.source == "yahoo"
