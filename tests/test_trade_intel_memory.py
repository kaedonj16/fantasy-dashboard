"""Guards that trade-intel discovery/crawl stay inside the 512Mi cron cap.

The trade-recrawl starter job OOM'd while BFS-expanding 2000 seed leagues with
10 workers, each holding full Sleeper JSON, then (had it survived) would have
cached every crawled week's transactions in dashboard_services.api.ttl_cache.
"""
from pathlib import Path
from unittest.mock import patch
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_discovery_no_longer_expands_two_thousand_seeds_up_front():
    source = (ROOT / "data_building/trade_intel/league_discovery.py").read_text()
    assert "LIMIT 2000" not in source
    assert "max_workers=10" not in source
    assert "_expand_seeds_into_frontier" in source
    assert "_MAX_SEEDS_PER_RUN" in source
    assert "_ids_from_stream" in source
    assert "read=False" in source


def test_crawler_does_not_use_process_wide_transaction_cache():
    source = (ROOT / "data_building/trade_intel/trade_crawler.py").read_text()
    assert "from dashboard_services.api import" not in source
    assert "get_transactions(" not in source
    assert '_get(f"/league/{league_id}/transactions/{week}")' in source


def test_refresh_script_isolates_stages_in_subprocesses():
    source = (ROOT / "scripts/refresh_trade_intel.py").read_text()
    assert "subprocess.run" in source
    assert '"discovery", "crawl", "analytics"' in source
    assert "--stage" in source


def test_trade_recrawl_cron_stays_on_starter_without_playwright():
    source = (ROOT / "render.yaml").read_text()
    recrawl = source.split("name: trade-recrawl", 1)[1].split("\n  - type:", 1)[0]
    assert "plan: starter" in recrawl
    assert "playwright" not in recrawl
    assert "--batch-size 1000" in recrawl


def test_league_discovery_cron_caps_target_on_starter():
    source = (ROOT / "render.yaml").read_text()
    disc = source.split("name: league-discovery", 1)[1]
    assert "plan: starter" in disc
    assert "--target 1000" in disc
    assert "--target 2000" not in disc.split("databases:", 1)[0]


def test_orchestrator_spawns_each_stage_as_a_subprocess(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "refresh_trade_intel", ROOT / "scripts/refresh_trade_intel.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    def fake_run(cmd, cwd=None, env=None):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "refresh_trade_intel.py",
        "--discover-target", "500",
        "--batch-size", "1000",
        "--workers", "4",
        "--crawl-mode", "both",
        "--recrawl-days", "2",
        "--analytics",
    ])
    assert mod.main() == 0
    stages = [c[c.index("--stage") + 1] for c in calls]
    assert stages == ["discovery", "crawl", "analytics"]
    assert any("--discover-target" in c for c in calls)
    assert any("--batch-size" in c for c in calls)


@pytest.mark.integration
def test_seed_and_frontier_caps_are_well_under_the_old_limits():
    pytest.importorskip("requests")
    from data_building.trade_intel.league_discovery import (
        _MAX_SEEDS_PER_RUN,
        _FRONTIER_CAP,
        _frontier_cap_for_target,
        _seed_limit_for_target,
    )
    assert _MAX_SEEDS_PER_RUN <= 80
    assert _seed_limit_for_target(500) <= _MAX_SEEDS_PER_RUN
    assert _seed_limit_for_target(1000) <= _MAX_SEEDS_PER_RUN
    assert _seed_limit_for_target(500) < 2000
    assert _frontier_cap_for_target(500) <= _FRONTIER_CAP
    assert _frontier_cap_for_target(500) < 5000


@pytest.mark.integration
def test_league_ids_extracted_without_keeping_payload_fields():
    pytest.importorskip("requests")
    from data_building.trade_intel.league_discovery import (
        _MAX_LEAGUES_PER_USER,
        _league_ids_from_payload,
    )
    payload = [
        {"league_id": "1", "scoring_settings": {"rec": 1} | {f"k{i}": i for i in range(40)}},
        {"league_id": "1"},  # dup
        {"league_id": "2"},
        "skip-me",
        {},
    ]
    assert _league_ids_from_payload(payload) == ["1", "2"]
    assert _league_ids_from_payload(None) == []
    huge = [{"league_id": str(i), "settings": {"x": "y" * 20}} for i in range(500)]
    ids = _league_ids_from_payload(huge)
    assert len(ids) == _MAX_LEAGUES_PER_USER
    assert ids[0] == "0"


@pytest.mark.integration
def test_ids_from_chunks_does_not_parse_json_and_handles_split_ids():
    """Regression for league-discovery OOM: never build the full object tree."""
    pytest.importorskip("requests")
    from data_building.trade_intel.league_discovery import (
        _LEAGUE_ID_RE,
        _MAX_LEAGUES_PER_USER,
        _OWNER_ID_RE,
        _ids_from_chunks,
    )
    raw = (
        b'[{"league_id": "111111111111111111", "scoring_settings": {"rec": 1}},'
        b'{"league_id": "222222222222222222"}]'
    )
    assert _ids_from_chunks([raw], _LEAGUE_ID_RE, 10) == [
        "111111111111111111", "222222222222222222",
    ]
    # id split across chunk boundary
    split_at = raw.index(b"222222") + 3
    assert _ids_from_chunks(
        [raw[:split_at], raw[split_at:]], _LEAGUE_ID_RE, 10,
    ) == ["111111111111111111", "222222222222222222"]
    owners = b'[{"owner_id": "999999999999999999", "players": ["a"] * 1}]'
    assert _ids_from_chunks([owners], _OWNER_ID_RE, 8) == ["999999999999999999"]
    # cap
    many = b",".join(b'{"league_id": "%d"}' % i for i in range(100000, 100000 + 200))
    assert len(_ids_from_chunks([many], _LEAGUE_ID_RE, _MAX_LEAGUES_PER_USER)) == _MAX_LEAGUES_PER_USER
    first = b'{"league_id": "123456789012345678"}'
    second = b'{"league_id": "999999999999999999"}'
    assert _ids_from_chunks(
        [first, second], _LEAGUE_ID_RE, 10, max_bytes=len(first),
    ) == ["123456789012345678"]


@pytest.mark.integration
def test_seed_expand_stops_once_frontier_is_full():
    """Regression: submitting every seed up front is what OOM'd the cron."""
    pytest.importorskip("requests")
    from data_building.trade_intel.league_discovery import _expand_seeds_into_frontier

    calls: list[str] = []

    def expand(lid: str):
        calls.append(lid)
        return [f"u-{lid}"], [f"new-{lid}-{i}" for i in range(20)]

    seeds = [str(i) for i in range(200)]
    frontier, owners, n_expanded = _expand_seeds_into_frontier(
        seeds,
        known=set(),
        frontier_cap=50,
        expand_league=expand,
        workers=2,
        in_flight_cap=4,
    )
    assert len(frontier) <= 50
    assert n_expanded == len(calls)
    assert 1 <= len(calls) < 40
    assert len(calls) < len(seeds)
    assert owners


@pytest.mark.integration
def test_seed_expand_skips_already_known_ids():
    pytest.importorskip("requests")
    from data_building.trade_intel.league_discovery import _expand_seeds_into_frontier

    known = {"already"}

    def expand(_lid: str):
        return ["u1"], ["already", "fresh"]

    frontier, _, n = _expand_seeds_into_frontier(
        ["s1", "s2", "s3", "s4"],
        known=known,
        frontier_cap=10,
        expand_league=expand,
        workers=1,
        in_flight_cap=1,
    )
    assert frontier == {"fresh"}
    assert n == 4


@pytest.mark.integration
def test_fetch_week_uses_crawler_get_not_ttl_cache():
    pytest.importorskip("requests")
    from data_building.trade_intel import trade_crawler as tc

    captured: list[str] = []

    def fake_get(path: str):
        captured.append(path)
        return [
            {"type": "trade", "status": "complete", "transaction_id": "t1"},
            {"type": "waiver", "status": "complete"},
        ]

    with patch.object(tc, "_get", fake_get):
        week, trades = tc._fetch_week("123", 7)
    assert week == 7
    assert captured == ["/league/123/transactions/7"]
    assert len(trades) == 1
    assert trades[0]["transaction_id"] == "t1"
