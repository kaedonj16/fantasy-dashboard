"""Regression tests for the 2026-09-26 CPU/memory perf fixes.

Covers: durable cron backfill markers (P1), gunicorn recycle threshold (P2),
ETag caching on /api/league-players (P3), bounded caches + _PLAYERS_GLOBAL
refresh (P4), missing DB indexes (P5), and redzone shared-collect + 304 polls
(P8). Source-structure style: importing app.py is intentionally avoided.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
RZJS = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
CRON = (ROOT / "cron_daily.py").read_text(encoding="utf-8")


# ── P1: durable cron backfill markers ─────────────────────────────────────────

def test_pipeline_markers_use_postgres_not_dotfiles():
    src = (ROOT / "data_building" / "pipeline_markers.py").read_text(encoding="utf-8")
    assert "app_state" in src
    assert "backfill_done" in src and "mark_backfill_done" in src
    # No dotfile marker paths (the docstring mentions the old pattern only
    # to explain what it replaces).
    assert 'Path(' not in src and 'os.path.join("cache"' not in src


def test_pipeline_markers_upsert_and_check(monkeypatch):
    mod = types.ModuleType("dashboard_services.db")
    calls = []

    class FakeConn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None):
            calls.append((sql, params))
            return self
        def fetchone(self):
            return None
        def commit(self): pass

    def get_conn():
        class CM:
            def __enter__(self): return FakeConn()
            def __exit__(self, *a): return False
        return CM()

    mod.get_conn = get_conn
    pkg = types.ModuleType("dashboard_services"); pkg.__path__ = []
    # monkeypatch restores the ORIGINAL module objects after the test. A
    # manual sys.modules set/pop leaves a fresh namespace-package object
    # behind, which broke later tests' monkeypatch.setattr on
    # dashboard_services.<submodule> (same failure mode as the 2026-09-25
    # utils-stub incident).
    monkeypatch.setitem(sys.modules, "dashboard_services", pkg)
    monkeypatch.setitem(sys.modules, "dashboard_services.db", mod)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pipeline_markers", ROOT / "data_building" / "pipeline_markers.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    assert pm.backfill_done("backfill:x") is False
    pm.mark_backfill_done("backfill:x")
    upserts = [c for c in calls if "ON CONFLICT" in c[0]]
    assert len(upserts) == 1 and upserts[0][1] == ("backfill:x", "done")


def test_cron_uses_db_markers_for_both_backfills():
    assert "backfill:weekly_metrics_perweek_team_v4" in CRON
    assert "backfill:redzone_snapshots_v1" in CRON
    assert 'cache/.redzone_snapshots_v1.done' not in CRON
    assert ".weekly_metrics_perweek_team_v4.done" not in CRON
    # Redzone repair still only marks done on a successful, available run.
    assert 'if result["updated"] and not result["unavailable"]:' in CRON


# ── P2: gunicorn recycle threshold ────────────────────────────────────────────

def test_gunicorn_max_requests_raised():
    src = (ROOT / "data_building" / "updates" / "startup.py").read_text(encoding="utf-8")
    assert '"--max-requests", "10000"' in src
    assert '"--max-requests", "1000"' not in src


# ── P3: /api/league-players ETag caching ──────────────────────────────────────

def test_league_players_version_key_covers_all_inputs():
    assert "def _lp_response_version_key" in APP
    assert "def _lp_etag_for" in APP
    assert "def _lp_not_modified_response" in APP
    assert "def _lp_cacheable_response" in APP
    # Overlay key (model_ts + adp_sig) + superflex + view + historical mtimes.
    assert "profile_aggregates_version" in APP
    assert 'Cache-Control"] = "public, max-age=60"' in APP
    assert "status=304" in APP


def test_league_players_nondefault_paths_stay_no_store():
    # Historical-season and explicit-ADP overlays bypass the version key.
    assert "_finish(payload, board_cacheable=False)" in APP


def test_league_players_caches_are_bounded():
    assert "_prune_ttl_cache(_LP_BOARD_JSON_CACHE, 64)" in APP
    assert "_prune_ttl_cache(_LP_ETAG_CACHE, 128)" in APP


# ── P4: bounded caches + _PLAYERS_GLOBAL refresh ───────────────────────────────

def test_players_global_has_ttl_refresh():
    assert "_PLAYERS_GLOBAL_TTL" in APP
    assert "_PLAYERS_GLOBAL_TS" in APP
    # Refresh path replaces the global instead of pinning the first fetch.
    assert "fresh = get_nfl_players()" in APP


def test_unbounded_caches_pruned_on_write():
    assert "_prune_ttl_cache(_WEEKLY_PTS_CACHE, 32)" in APP
    assert "_prune_ttl_cache(_TEAM_PAYLOAD_CACHE, 64)" in APP
    assert "_prune_ttl_cache(_NFL_TEAM_DETAILS_CACHE, 64)" in APP
    subs = (ROOT / "dashboard_services" / "subscriptions.py").read_text(encoding="utf-8")
    assert "_MEMBER_CACHE" in subs and "2048" in subs


# ── P5: missing DB indexes ────────────────────────────────────────────────────

def test_perf_indexes_exist():
    mig = (ROOT / "migrations" / "040_perf_indexes.sql").read_text(encoding="utf-8")
    assert "idx_pwm_season_player_week" in mig
    assert "player_weekly_metrics (season, player_id, week)" in mig
    wd = (ROOT / "dashboard_services" / "waiver_discoveries.py").read_text(encoding="utf-8")
    assert "idx_wbgd_season_week_cat" in wd
    pbp = (ROOT / "routes" / "push_bp.py").read_text(encoding="utf-8")
    assert "push_subscriptions_league_owner_idx" in pbp


# ── P8: redzone shared collect + conditional polls ────────────────────────────

def test_redzone_shared_collect_cache():
    assert "def _rz_cached_collect" in APP
    assert "_RZ_COLLECT_TTL = 20.0" in APP
    assert "_prune_ttl_cache(_RZ_COLLECT_CACHE, 32)" in APP
    # Per-viewer fields stamped onto a copy; cached payload never mutated.
    assert "d = dict(_rz_d)" in APP
    # Honest clock: data age is the collect's age.
    assert '"updated_at": _rz_collected_at' in APP


def test_redzone_api_answers_304():
    assert 'request.headers.get("If-None-Match") == etag' in APP
    assert 'response.headers["ETag"] = etag' in APP


def test_redzone_client_polls_conditionally():
    assert "_rzEtagByScope" in RZJS
    assert "If-None-Match" in RZJS
    assert "resp.status === 304" in RZJS
    assert "refresh-304" in RZJS
