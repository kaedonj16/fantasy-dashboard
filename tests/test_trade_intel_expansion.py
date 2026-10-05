"""Focused tests for the trade-intel expansion (feat/trade-intel-expansion).

Covers: keeper-league inclusion mapping, trade-context side mapping,
previous_league_id chain walking, velocity-ordered crawl queries, and
env-var crawler caps. Pure unit tests — no DB, no network.
"""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_building.trade_intel.league_types import (
    LeagueType,
    calibration_mode,
    format_name,
    league_format_sql_param,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Item 1: keeper inclusion
# ---------------------------------------------------------------------------

def test_keeper_format_mapping():
    assert league_format_sql_param("keeper") == int(LeagueType.KEEPER) == 1
    assert league_format_sql_param("dynasty") == 2
    assert league_format_sql_param("redraft") == 0
    assert league_format_sql_param("all") is None
    assert league_format_sql_param("bogus") is None


def test_keeper_still_rejected_by_calibration():
    with pytest.raises(ValueError):
        calibration_mode(1)
    with pytest.raises(ValueError):
        calibration_mode(int(LeagueType.KEEPER))
    assert calibration_mode(0) == "redraft"
    assert calibration_mode(2) == "dynasty"


def test_format_name_buckets():
    assert format_name(0) == "redraft"
    assert format_name(1) == "keeper"
    assert format_name(2) == "dynasty"
    assert format_name(99) == "all"  # unknown folds to the aggregate bucket
    assert format_name(None) == "all"


def test_crawl_queries_include_keeper():
    """_leagues_to_crawl SQL must accept league_type 1 in every mode."""
    pytest.importorskip("requests")
    from data_building.trade_intel import trade_crawler

    captured = {}

    def fake_execute(query, params=None):
        captured["query"] = query
        captured["params"] = params
        fake = MagicMock()
        fake.fetchall.return_value = []
        return fake

    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.execute.side_effect = fake_execute

    with patch.object(trade_crawler, "get_conn", return_value=fake_conn):
        for mode in ("new", "existing", "both"):
            trade_crawler._leagues_to_crawl(batch_size=10, crawl_mode=mode, recrawl_days=7)
            assert "IN (0, 1, 2)" in captured["query"], f"mode={mode}: keeper missing"


def test_velocity_ordering_and_dead_league_predicate():
    """Existing/both modes order by trade velocity; dead leagues back off."""
    pytest.importorskip("requests")
    from data_building.trade_intel import trade_crawler

    captured = {}

    def fake_execute(query, params=None):
        captured["query"] = query
        fake = MagicMock()
        fake.fetchall.return_value = []
        return fake

    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.execute.side_effect = fake_execute

    with patch.object(trade_crawler, "get_conn", return_value=fake_conn):
        trade_crawler._leagues_to_crawl(batch_size=10, crawl_mode="existing", recrawl_days=2)
        q = captured["query"]
        assert "last_trade_at DESC NULLS LAST" in q
        assert "last_crawled_at ASC" in q
        # dead leagues (0 trades, fully crawled) re-crawl at most every 14 days
        assert "total_trades > 0" in q
        assert "INTERVAL '14 days'" in q

        trade_crawler._leagues_to_crawl(batch_size=10, crawl_mode="both", recrawl_days=2)
        q = captured["query"]
        order_by = q[q.index("ORDER BY"):]
        assert order_by.index("last_trade_at DESC NULLS LAST") < order_by.index("last_crawled_at")


# ---------------------------------------------------------------------------
# Items 2/7: chain walking + trade context (pure logic)
# ---------------------------------------------------------------------------

def _fake_meta(lid, season, league_type=2, prev=None):
    return {
        "league_id": lid,
        "season": str(season),
        "settings": {"type": league_type},
        "total_rosters": 10,
        "roster_positions": [],
        "previous_league_id": prev,
    }


def test_walk_history_chains_inserts_ancestors_with_own_seasons():
    from data_building.trade_intel import league_discovery as ld

    rows = [
        {"league_id": "100", "previous_league_id": "90"},
        {"league_id": "200", "previous_league_id": None},
    ]
    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.execute.return_value.fetchall.return_value = rows

    metas = {
        "90": _fake_meta("90", 2025, prev="80"),
        "80": _fake_meta("80", 2024, league_type=1, prev=None),
    }
    saved = []

    with patch.object(ld, "get_conn", return_value=fake_conn), \
         patch.object(ld, "_league_meta", side_effect=lambda lid: metas.get(lid)), \
         patch.object(ld, "_save_leagues", side_effect=lambda leagues: saved.extend(leagues) or len(leagues)), \
         patch.object(ld.time, "sleep", lambda s: None):
        n = ld.walk_history_chains(max_depth=3, season=2026)

    assert n == 2
    by_id = {lg["league_id"]: lg for lg in saved}
    assert by_id["90"]["season"] == 2025
    assert by_id["80"]["season"] == 2024
    assert by_id["90"]["previous_league_id"] == "80"
    assert by_id["80"]["previous_league_id"] is None


def test_walk_history_chains_respects_max_depth():
    from data_building.trade_intel import league_discovery as ld

    rows = [{"league_id": "100", "previous_league_id": "90"}]
    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.execute.return_value.fetchall.return_value = rows

    # 90 -> 80 -> 70 -> 60, but max_depth=2 stops after 90, 80
    metas = {
        str(lid): _fake_meta(str(lid), 2026 - (90 - lid), prev=str(lid - 10) if lid > 70 else None)
        for lid in (90, 80, 70, 60)
    }
    saved = []

    with patch.object(ld, "get_conn", return_value=fake_conn), \
         patch.object(ld, "_league_meta", side_effect=lambda lid: metas.get(lid)), \
         patch.object(ld, "_save_leagues", side_effect=lambda leagues: saved.extend(leagues) or len(leagues)), \
         patch.object(ld.time, "sleep", lambda s: None):
        n = ld.walk_history_chains(max_depth=2, season=2026)

    assert n == 2
    assert {lg["league_id"] for lg in saved} == {"90", "80"}


def test_side_map_for_txn():
    pytest.importorskip("requests")
    from data_building.trade_intel.trade_crawler import _side_map_for_txn

    # adds/drops: roster ids sorted, first -> "a"
    txn = {"adds": {"p1": 5, "p2": 3}, "drops": {"p3": 5}, "draft_picks": []}
    assert _side_map_for_txn(txn) == {"3": "a", "5": "b"}

    # pick-only trade falls back to owner_id
    txn = {"adds": {}, "drops": {},
           "draft_picks": [{"owner_id": 9, "season": "2027", "round": 1}]}
    assert _side_map_for_txn(txn) == {"9": "a"}


def test_trade_context_maps_records_to_sides_and_omits_missing():
    pytest.importorskip("requests")
    from data_building.trade_intel.trade_crawler import _trade_context_for_txn

    txn = {"adds": {"p1": 5, "p2": 3}, "drops": {}, "draft_picks": []}
    roster_map = {
        "3": {"wins": 8, "losses": 2, "ties": 0, "fpts": 1234.5},
        # roster "5" missing -> side "b" omitted
    }
    ctx = _trade_context_for_txn(txn, roster_map)
    assert ctx == {"a": {"wins": 8, "losses": 2, "ties": 0, "fpts": 1234.5}}

    import json
    json.dumps(ctx)  # must be JSON-serializable for the JSONB column


# ---------------------------------------------------------------------------
# Item 8: env-var caps
# ---------------------------------------------------------------------------

def test_env_var_caps_parsed():
    env = dict(os.environ)
    env.update({
        "TRADE_INTEL_DISCOVERY_WORKERS": "6",
        "TRADE_INTEL_DISCOVERY_IN_FLIGHT": "9",
        "TRADE_INTEL_MAX_SEEDS": "120",
        "TRADE_INTEL_FRONTIER_CAP": "2000",
    })
    code = (
        "import data_building.trade_intel.league_discovery as ld; "
        "print(ld._DISCOVERY_WORKERS, ld._DISCOVERY_IN_FLIGHT, "
        "ld._MAX_SEEDS_PER_RUN, ld._FRONTIER_CAP)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "6 9 120 2000"


def test_default_caps_unchanged():
    """Without env vars the Render-512Mi-safe defaults hold."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TRADE_INTEL_")}
    code = (
        "import data_building.trade_intel.league_discovery as ld; "
        "print(ld._DISCOVERY_WORKERS, ld._DISCOVERY_IN_FLIGHT, "
        "ld._MAX_SEEDS_PER_RUN, ld._FRONTIER_CAP)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "2 4 80 1500"
