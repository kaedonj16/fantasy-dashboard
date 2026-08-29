"""ESPN draft relay tokens, store, and merge helpers."""
from __future__ import annotations

import time

from dashboard_services.espn_draft_relay import (
    build_bookmarklet,
    get_relay_snapshot,
    merge_live_with_relay,
    mint_relay_token,
    put_relay_snapshot,
    shortcut_javascript,
    verify_relay_token,
)


def test_mint_and_verify_token_roundtrip():
    minted = mint_relay_token(league_id="99", season=2026, account_id="7")
    claims = verify_relay_token(minted["token"])
    assert claims is not None
    assert claims["league_id"] == "99"
    assert claims["season"] == 2026
    assert claims["account_id"] == "7"


def test_verify_rejects_tampered_or_expired(monkeypatch):
    minted = mint_relay_token(league_id="99", season=2026)
    bad = minted["token"][:-4] + "dead"
    assert verify_relay_token(bad) is None
    # Force expiry by rewriting exp in the past.
    parts = minted["token"].split(".")
    parts[2] = str(int(time.time()) - 10)
    # Signature no longer matches → invalid
    assert verify_relay_token(".".join(parts)) is None


def test_put_get_relay_snapshot():
    payload = {"picks": [{"pick_no": 1, "player_id": "5938"}], "source": "espn-relay"}
    put_relay_snapshot("99", 2026, payload, source="bookmarklet")
    entry = get_relay_snapshot("99", 2026)
    assert entry is not None
    assert entry["payload"]["picks"][0]["player_id"] == "5938"
    assert entry["source"] == "bookmarklet"


def test_merge_prefers_longer_relay():
    live = {"picks": [], "status": "drafting", "fingerprint": "a"}
    relay_entry = {
        "updated_at": 1,
        "source": "bookmarklet",
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


def test_bookmarklet_contains_origin_and_token():
    bm = build_bookmarklet("https://www.brfantasyfootball.com", "TOK.EN")
    assert bm.startswith("javascript:")
    assert "brfantasyfootball.com" in bm
    assert "TOK.EN" in bm
    assert "/api/draft/espn-relay" in bm
    assert "source:\"bookmarklet\"" in bm.replace(" ", "") or "'bookmarklet'" in bm or '"bookmarklet"' in bm


def test_shortcut_javascript_calls_completion_not_bookmarklet_prefix():
    js = shortcut_javascript("https://www.brfantasyfootball.com", "TOK.EN")
    assert not js.startswith("javascript:")
    assert "completion(" in js
    assert "function finish(" in js
    assert "brfantasyfootball.com" in js
    assert "TOK.EN" in js
    assert "ios-shortcut" in js
    assert "/api/draft/espn-relay" in js
    assert "getOwnPropertyNames" in js
    assert "apiScan" in js
    assert "scanReact" in js
    assert "selectedPid" in js


def test_bookmarklet_also_uses_hardened_scanner():
    bm = build_bookmarklet("https://www.brfantasyfootball.com", "TOK.EN")
    assert "getOwnPropertyNames" in bm
    assert "apiScan" in bm
    assert "No picks found in page state" in bm
