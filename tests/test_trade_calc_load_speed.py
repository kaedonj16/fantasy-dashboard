"""Trade-calculator load-speed regressions (2026-11-23).

Kaedon: "the trade calc is slow and players often take forever to load
their values into the fair bar." Profiling (live + local) indicted three
things, each pinned here:

1. A self-busting cache loop: when a global ADP source's snapshot map
   stays empty, every league-players payload build re-fetched and
   REWROTE that snapshot; write_adp_snapshot always rewrote the file
   (new collected_at -> new mtime), and that mtime is
   global_adp_snapshot_signature() -- the freshness input for the payload
   memo, the overlay cache, the serialized-body caches, and the ETag.
   So every /api/league-players hit paid a full rebuild and 304s never
   engaged. Fix: an identical rewrite is skipped.
2. The trade view ran the full projection/Market-vs-ADP/historical/ROS
   overlay plus the league PPG overlay on every hit, although
   slim_trade_payload keeps none of those fields (the body built from the
   base payload is byte-identical), and the PPG overlay forced the
   signed-in trade response to no-store. Fix: a warm-path shortcut serves
   the serialized trade body, the overlay is skipped for view=trade, and
   the PPG overlay no longer applies to the trade view.
3. The client's format-toggle reload blanked the player pool for the
   whole fetch (forever on failure). Fix: the pool is swapped atomically
   when the new feed lands.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


# ── 1. Identical ADP snapshot rewrites are skipped ───────────────────────────

@pytest.fixture()
def adp_tmp(monkeypatch, tmp_path):
    from dashboard_services import adp_service

    monkeypatch.setattr(adp_service, "_snapshot_dir", lambda: tmp_path)
    monkeypatch.setattr(adp_service, "_persist_snapshot_db", lambda record: None)
    return adp_service, tmp_path


def test_identical_snapshot_rewrite_is_skipped(adp_tmp):
    adp_service, tmp_path = adp_tmp
    payload = {"adp": {"101": 12.5, "102": 30.0},
               "extra": {"101": {"min_pick": 4}}}
    assert adp_service.write_adp_snapshot("mfl", "redraft", 2026, payload) is True
    path = tmp_path / "mfl_redraft_2026.json"
    before_bytes = path.read_bytes()
    before_mtime = os.stat(path).st_mtime_ns
    time.sleep(0.02)
    # Same data again (a re-fetch that changed nothing): no rewrite, so the
    # signature downstream caches key on cannot move.
    assert adp_service.write_adp_snapshot("mfl", "redraft", 2026, payload) is False
    assert path.read_bytes() == before_bytes
    assert os.stat(path).st_mtime_ns == before_mtime


def test_changed_snapshot_still_writes(adp_tmp):
    adp_service, tmp_path = adp_tmp
    assert adp_service.write_adp_snapshot(
        "mfl", "redraft", 2026, {"adp": {"101": 12.5}}) is True
    assert adp_service.write_adp_snapshot(
        "mfl", "redraft", 2026, {"adp": {"101": 9.0}}) is True
    snap = adp_service.load_adp_snapshot("mfl", "redraft", 2026)
    assert snap["adp"] == {"101": 9.0}


def test_first_snapshot_write_is_not_skipped(adp_tmp):
    adp_service, _tmp_path = adp_tmp
    assert adp_service.write_adp_snapshot(
        "espn", "redraft", 2026, {"adp": {"7": 3.0}}) is True


def test_empty_snapshot_rewrite_is_skipped(adp_tmp):
    # A source whose feed keeps coming back empty (the live MFL case):
    # the first empty record writes, later empty fetches must not keep
    # bumping the file's mtime and busting the downstream caches.
    adp_service, tmp_path = adp_tmp
    assert adp_service.write_adp_snapshot(
        "mfl", "redraft", 2026, {"adp": {}}) is True
    path = tmp_path / "mfl_redraft_2026.json"
    before = (path.read_bytes(), os.stat(path).st_mtime_ns)
    time.sleep(0.02)
    assert adp_service.write_adp_snapshot(
        "mfl", "redraft", 2026, {"adp": {}}) is False
    assert (path.read_bytes(), os.stat(path).st_mtime_ns) == before


# ── 2. Trade body cannot be changed by the overlays it now skips ────────────

def test_trade_slim_ignores_overlay_only_fields():
    from dashboard_services.league_players_board import slim_trade_payload

    base_row = {
        "id": "6770", "name": "Joe Burrow", "team": "CIN", "position": "QB",
        "age": 29.9, "value": 428.0, "sf_value": 700.0,
        "pos_rank_label": "QB2", "espnHeadshot": "https://x/y.png",
        "tier_marker": None,
    }
    overlaid_row = dict(base_row)
    overlaid_row.update({
        # Everything the skipped work stamps: projections, Market vs ADP,
        # historical, ROS, and the league-PPG overlay's fields.
        "projection": {"ppg": 22.1}, "proj_ppg": 22.1,
        "market_vs_adp": 3.4, "sf_market_vs_adp": 1.2,
        "historical": {"2025": {"ppg": 20.0}},
        "ros_ppg": 21.0, "ros_score": 99.0,
        "ppg": 23.4, "total_pts": 400.0, "ppg_rank": 2,
    })
    base = {"players": [base_row], "tier_thresholds": {"1qb": {}}}
    overlaid = {"players": [overlaid_row], "tier_thresholds": {"1qb": {}}}
    assert slim_trade_payload(base) == slim_trade_payload(overlaid)


def test_trade_view_warm_path_and_overlay_skip_are_wired():
    # Warm path: a cached serialized trade body is served before the
    # overlay / PPG work, through the ETag-aware response helper.
    assert '("trade", overlay_key, bool(_mi_is_sf),' in APP_PY
    assert "return _lp_cacheable_response(_trade_body, _lp_version_key)" in APP_PY
    # The overlay only runs for non-trade views now.
    assert "if not _view_trade:\n        payload = _overlay_league_players_payload(" in APP_PY
    # The league PPG overlay is gated off for the trade view, so a
    # signed-in league context can no longer force it to no-store.
    ppg_gate = APP_PY.index('if _view_trade:\n        # The trade slim keeps no PPG')
    assert '_ppg_lg = ""' in APP_PY[ppg_gate:ppg_gate + 800]


# ── 3. Format-toggle reload keeps the pool until the new feed lands ──────────

def test_league_type_change_does_not_blank_the_pool():
    start = APP_JS.index("async function onLeagueTypeChange()")
    body = APP_JS[start:start + 900]
    assert "allPlayers = [];" not in body
    assert "allPlayers = buildPlayerPool(data);" in body
