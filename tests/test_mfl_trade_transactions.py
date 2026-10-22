"""MFL get_transactions must parse the real export trade shape.

Fake payload is the shape captured live 2026-10-22 from
``/2023/export?TYPE=transactions&L=43049&TRANS_TYPE=TRADE&JSON=1``:
``franchise``/``franchise2`` are the two sides, ``franchiseN_gave_up`` is a
comma-separated list (trailing comma) of MFL player ids, ``FP_<orig>_<year>_
<round>`` future picks, ``DP_<rr>_<pp>`` current-year picks (round/pick one
less than actual) and ``BB_<amount>`` blind-bid dollars. Timestamps are
epoch SECONDS; the unified shape carries epoch ms.
"""
from __future__ import annotations

import pytest

from dashboard_services.providers.mfl_api import _CACHE, MFLProvider


@pytest.fixture(autouse=True)
def clear_cache():
    _CACHE.clear()


CAPTURED_TRADE = {
    "franchise": "0004", "franchise2": "0003", "type": "TRADE",
    "franchise1_gave_up": "15751,15293,FP_0004_2024_2,",
    "franchise2_gave_up": "14845,16188,FP_0003_2024_2,",
    "timestamp": "1701392281", "expires": "1701993600", "comments": "",
}

XWALK = {"15751": "canon-a", "15293": "canon-b",
         "14845": "canon-c", "16188": "canon-d"}


def _provider(monkeypatch, transactions, xwalk=XWALK):
    provider = MFLProvider()
    payload = {"transactions": {"transaction": transactions}}
    monkeypatch.setattr(provider, "_export", lambda *a, **k: payload)
    monkeypatch.setattr(provider, "_canonical_map", lambda *a, **k: dict(xwalk))
    return provider


def test_captured_trade_shape_parsed(monkeypatch):
    provider = _provider(monkeypatch, [CAPTURED_TRADE])
    rows = provider.get_transactions("43049", 2023, 1)

    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "trade"
    assert row["status"] == "complete"
    # Epoch seconds on the wire -> epoch ms in the unified shape.
    assert row["created"] == 1701392281000
    # A player given up by one side is received by the OTHER side.
    assert row["adds"] == {
        "canon-a": "0003", "canon-b": "0003",   # 0004 gave up -> 0003 receives
        "canon-c": "0004", "canon-d": "0004",   # 0003 gave up -> 0004 receives
    }
    assert row["drops"] == {
        "canon-a": "0004", "canon-b": "0004",
        "canon-c": "0003", "canon-d": "0003",
    }
    assert set(row["roster_ids"]) == {"0004", "0003"}
    # Raw gave_up strings are kept in metadata for debugging.
    assert row["metadata"]["franchise1_gave_up"] == "15751,15293,FP_0004_2024_2,"


def test_future_pick_parsed_with_actual_round(monkeypatch):
    provider = _provider(monkeypatch, [CAPTURED_TRADE])
    row = provider.get_transactions("43049", 2023, 1)[0]

    picks = {(p["roster_id"], p["owner_id"]): p for p in row["draft_picks"]}
    assert len(picks) == 2
    pick = picks[("0004", "0003")]  # 0004's pick, now owned by 0003
    assert pick["season"] == 2024
    assert pick["round"] == 2
    assert pick["previous_owner_id"] == "0004"


def test_current_year_pick_and_blind_bid_tokens(monkeypatch):
    trade = {
        "franchise": "0001", "franchise2": "0002", "type": "TRADE",
        "franchise1_gave_up": "11111,DP_0_4,BB_25,99999,",
        "franchise2_gave_up": "22222,",
        "timestamp": "1695000000",
    }
    xwalk = {"11111": "canon-x", "22222": "canon-y"}  # 99999 unmapped
    provider = _provider(monkeypatch, [trade], xwalk=xwalk)
    row = provider.get_transactions("43049", 2023, 1)[0]

    # BB dollars skipped; unmapped MFL ids never surface as canonical ids.
    assert row["adds"] == {"canon-x": "0002", "canon-y": "0001"}
    assert row["drops"] == {"canon-x": "0001", "canon-y": "0002"}
    # DP_0_4 = round 1, pick 5 (both stored one less than actual).
    assert row["draft_picks"] == [{
        "season": 2023, "round": 1, "order": 5,
        "roster_id": "0001", "owner_id": "0002", "previous_owner_id": "0001",
    }]


def test_trade_without_gave_up_fields_tolerated(monkeypatch):
    # Older fakes / partial exports carry no gave_up strings: the row must
    # still normalize (empty assets) rather than crash.
    trade = {"id": "tx1", "type": "TRADE", "franchise": "0001,0002",
             "timestamp": "10"}
    provider = _provider(monkeypatch, [trade])
    row = provider.get_transactions("123", 2026, 1)[0]

    assert row["type"] == "trade"
    assert row["adds"] == {}
    assert row["drops"] == {}
    assert row["draft_picks"] == []
