"""ESPN draft relay store and merge helpers (desktop extension)."""
from __future__ import annotations

from dashboard_services.espn_draft_relay import (
    get_relay_snapshot,
    merge_live_with_relay,
    put_relay_snapshot,
)


def test_put_get_relay_snapshot():
    payload = {"picks": [{"pick_no": 1, "player_id": "5938"}], "source": "espn-relay"}
    put_relay_snapshot("99", 2026, payload, source="extension")
    entry = get_relay_snapshot("99", 2026)
    assert entry is not None
    assert entry["payload"]["picks"][0]["player_id"] == "5938"
    assert entry["source"] == "extension"


def test_merge_prefers_longer_relay():
    live = {"picks": [], "status": "drafting", "fingerprint": "a"}
    relay_entry = {
        "updated_at": 1,
        "source": "extension",
        "payload": {
            "picks": [
                {"pick_no": 1, "player_id": "5938", "external_player_id": "4039057"},
                {"pick_no": 2, "player_id": "6794", "external_player_id": "4241479"},
            ],
            "status": "drafting",
            "source": "espn-relay",
        },
    }
    merged = merge_live_with_relay(live, relay_entry)
    assert len(merged["picks"]) == 2
    assert merged["relay_source"] == "espn-relay"
    assert merged["picks_observed"] is True


def test_no_mobile_token_or_bookmarklet_helpers():
    import dashboard_services.espn_draft_relay as mod

    assert not hasattr(mod, "mint_relay_token")
    assert not hasattr(mod, "build_bookmarklet")
    assert not hasattr(mod, "shortcut_javascript")
    assert not hasattr(mod, "verify_relay_token")
