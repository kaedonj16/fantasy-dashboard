"""Multi-platform connected-league trade contribution (contrib.py).

Covers the account path added alongside the original Sleeper user path:
per-week platforms (ESPN/Fleaflicker) vs season-wide platforms (Yahoo/MFL),
provider-namespaced league/transaction ids, skip rules, and the account
opt-in gate. The DB is an in-memory fake capturing executed inserts
(psycopg3-style: ``conn.execute`` returns an object with fetchone()/rowcount;
there is no executemany on a psycopg3 Connection and contrib never uses one).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dashboard_services import accounts, platform_api
from data_building.trade_intel import contrib, trade_crawler

# ---------------------------------------------------------------------------
# In-memory fake DB
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, row=None, rowcount=0):
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, db):
        self._db = db

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        return self._db.execute(sql, params)


class _FakeDB:
    """Captures contrib's inserts, honoring ON CONFLICT DO NOTHING."""

    def __init__(self):
        self.leagues = {}          # stored league_id -> insert params
        self.trades = {}           # transaction_id -> (insert params, db id)
        self.asset_rows = []       # flat param lists, 10 values per asset
        self.account_optins = {}   # account_id -> bool
        self._next_trade_id = 1

    def execute(self, sql, params=None):
        params = params or ()
        s = " ".join(str(sql).split()).upper()
        if s.startswith("CREATE TABLE"):
            return _Result(rowcount=0)
        if "INTO TRADE_INTEL_LEAGUES" in s:
            if params[0] in self.leagues:
                return _Result(rowcount=0)
            self.leagues[params[0]] = params
            return _Result(rowcount=1)
        if "INTO TRADE_INTEL_TRADES" in s:
            if params[1] in self.trades:
                return _Result(row=None)  # conflict: no RETURNING row
            trade_id = self._next_trade_id
            self._next_trade_id += 1
            self.trades[params[1]] = (params, trade_id)
            return _Result(row={"id": trade_id}, rowcount=1)
        if "INTO TRADE_INTEL_ASSETS" in s:
            self.asset_rows.append(list(params))
            return _Result(rowcount=len(params) // 10)
        if "FROM TRADE_INTEL_CONTRIB_ACCOUNTS" in s:
            opted = self.account_optins.get(params[0])
            return _Result(row={"opted_in": opted} if opted is not None else None)
        if "INTO TRADE_INTEL_CONTRIB_ACCOUNTS" in s:
            self.account_optins[params[0]] = params[1]
            return _Result(rowcount=1)
        raise AssertionError(f"unexpected SQL: {s[:100]}")

    def assets(self):
        """All inserted assets as dicts (10-value groups, insert order)."""
        out = []
        for flat in self.asset_rows:
            for i in range(0, len(flat), 10):
                g = flat[i:i + 10]
                out.append({
                    "trade_id": g[0], "side": g[1], "asset_type": g[2],
                    "player_id": g[3], "pick_season": g[4], "pick_round": g[5],
                    "pick_order": g[6], "pick_roster_id": g[7],
                    "pick_slot": g[8], "provider": g[9],
                })
        return out


@pytest.fixture
def db(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(contrib, "get_conn", lambda *a, **k: _FakeConn(fake))
    monkeypatch.setattr(contrib, "_check_player_identity", lambda *a, **k: None)
    monkeypatch.setattr(contrib, "_ACCOUNT_TABLE_READY", False)
    monkeypatch.setattr(trade_crawler, "_current_nfl_week", lambda: 18)
    return fake


def _patch_platform(monkeypatch, txns_for, league_info=None):
    """Patch platform_api fetches. txns_for(platform, league_id, week, season)."""
    calls = []

    def fake_get_transactions(platform, league_id, week, season):
        calls.append((platform, str(league_id), week, season))
        return txns_for(platform, str(league_id), week, season)

    monkeypatch.setattr(platform_api, "get_transactions", fake_get_transactions)
    monkeypatch.setattr(
        platform_api, "get_league",
        lambda *a, **k: league_info if league_info is not None
        else {"total_rosters": 12, "settings": {"type": 2}},
    )
    return calls


def _ms(year, month, day):
    return int(datetime(year, month, day, 12, tzinfo=timezone.utc).timestamp() * 1000)


# 2025 season: Labor Day = Sep 1, week 1 starts Tue Sep 2.
WEEK1_MS = _ms(2025, 9, 4)
WEEK6_MS = _ms(2025, 10, 9)


def _txn(tid, created, adds=None, picks=None, status="complete", ttype="trade"):
    adds = adds if adds is not None else {"p1": 1, "p2": 2}
    return {
        "transaction_id": tid, "type": ttype, "status": status,
        "created": created, "roster_ids": [1, 2],
        "adds": adds,
        "drops": {pid: (2 if rid == 1 else 1) for pid, rid in adds.items()},
        "draft_picks": picks or [],
    }


# ---------------------------------------------------------------------------
# (a) ESPN: per-week fetch, namespaced ids, provider-tagged assets
# ---------------------------------------------------------------------------

def test_espn_per_week_trade_contributed_with_namespaced_ids(db, monkeypatch):
    def txns_for(platform, league_id, week, season):
        return [_txn("espn-txn-1", WEEK1_MS)] if week == 3 else []

    calls = _patch_platform(monkeypatch, txns_for)
    n = contrib._contribute_platform_league("espn", "12345", 2025)

    assert n == 1
    assert len(calls) == 18  # looped weeks 1..current week
    assert "espn:12345" in db.leagues
    assert "espn:espn-txn-1" in db.trades
    trade_params, _ = db.trades["espn:espn-txn-1"]
    assert trade_params[0] == "espn:12345"   # stored league id
    assert trade_params[3] == 3              # week from the per-week fetch
    assets = db.assets()
    assert {a["player_id"] for a in assets} == {"p1", "p2"}
    assert all(a["provider"] == "espn" for a in assets)


# ---------------------------------------------------------------------------
# (b) Yahoo: season-wide fetch ONCE, weeks derived from timestamps
# ---------------------------------------------------------------------------

def test_yahoo_season_wide_single_fetch_timestamp_weeks(db, monkeypatch):
    def txns_for(platform, league_id, week, season):
        # Yahoo returns the whole season on every call, stamped leg=week.
        return [_txn("y-txn-w1", WEEK1_MS), _txn("y-txn-w6", WEEK6_MS)]

    calls = _patch_platform(monkeypatch, txns_for)
    n = contrib._contribute_platform_league("yahoo", "999", 2025)

    assert n == 2
    assert len(calls) == 1  # NOT looped per week
    weeks = {db.trades[f"yahoo:{tid}"][0][3] for tid in ("y-txn-w1", "y-txn-w6")}
    assert weeks == {1, 6}


def test_seconds_created_normalized_to_ms(db, monkeypatch):
    created_s = int(datetime(2025, 9, 4, 12, tzinfo=timezone.utc).timestamp())

    def txns_for(platform, league_id, week, season):
        return [_txn("y-sec", created_s)]

    _patch_platform(monkeypatch, txns_for)
    n = contrib._contribute_platform_league("yahoo", "998", 2025)

    assert n == 1
    trade_params, _ = db.trades["yahoo:y-sec"]
    assert trade_params[3] == 1  # week derived after seconds->ms scaling
    assert trade_params[5] == datetime(2025, 9, 4, 12, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# (c) Non-complete / non-trade / zero-asset rows are skipped
# ---------------------------------------------------------------------------

def test_non_complete_and_zero_asset_trades_skipped(db, monkeypatch):
    def txns_for(platform, league_id, week, season):
        if week != 1:
            return []
        return [
            _txn("vetoed", WEEK1_MS, status="vetoed"),
            _txn("waiver-add", WEEK1_MS, ttype="waiver"),
            _txn("empty", WEEK1_MS, adds={}),
        ]

    _patch_platform(monkeypatch, txns_for)
    n = contrib._contribute_platform_league("espn", "777", 2025)

    assert n == 0
    assert db.trades == {}


# ---------------------------------------------------------------------------
# (d) Account flow: opt-in gate, then linked-league contribution
# ---------------------------------------------------------------------------

def test_account_flow_requires_opt_in(db, monkeypatch):
    calls = _patch_platform(monkeypatch, lambda *a: [_txn("x", WEEK1_MS)])
    monkeypatch.setattr(
        accounts, "list_user_leagues",
        lambda account_id: [
            {"platform": "espn", "league_id": "555", "season": 2025},
        ],
    )

    result = contrib.contribute_account_leagues(7, season=2025)
    assert result["opted_in"] is False
    assert result["trades"] == 0
    assert calls == []          # nothing fetched
    assert db.leagues == {}     # nothing stored

    contrib.set_account_opt_in(7, True)
    assert contrib.is_account_opted_in(7) is True

    result = contrib.contribute_account_leagues(7, season=2025)
    assert result["opted_in"] is True
    assert result["leagues"] == 1
    assert result["trades"] == 1
    assert "espn:555" in db.leagues


# ---------------------------------------------------------------------------
# (e) MFL: pick assets keep season/round through the unified shape
# ---------------------------------------------------------------------------

def test_mfl_contributed_trade_stores_pick_asset(db, monkeypatch):
    pick = {"season": 2026, "round": 1, "roster_id": "0004",
            "owner_id": 2, "previous_owner_id": "0004"}

    def txns_for(platform, league_id, week, season):
        return [_txn("mfl-txn-1", _ms(2025, 9, 11), picks=[pick])]

    calls = _patch_platform(monkeypatch, txns_for)
    n = contrib._contribute_platform_league("mfl", "43049", 2025)

    assert n == 1
    assert len(calls) == 1  # MFL is season-wide too
    assert "mfl:43049" in db.leagues
    pick_assets = [a for a in db.assets() if a["asset_type"] == "pick"]
    assert len(pick_assets) == 1
    assert pick_assets[0]["pick_season"] == 2026
    assert pick_assets[0]["pick_round"] == 1
    assert pick_assets[0]["provider"] == "mfl"
