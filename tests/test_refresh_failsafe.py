"""Refresh failsafe: /api/refresh-league must not destroy the stale fallback.

Regression test for the fail-unsafe refresh. The endpoint used to zero the
cache entry's ts, which made get_league_ctx_from_cache's stale-fallback check
(time.time() - old_ts <= stale_window) always false, turning rebuild failures
into HTTP 500s. Now the endpoint sets a force_refresh marker instead, and
_league_ctx_cache_valid treats the marker as "rebuild but keep the old ts".
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _load_cache_valid_fn():
    """Exec the real _league_ctx_cache_valid from app.py with stubbed deps."""
    src = _read("app.py")
    tree = ast.parse(src)
    fn_node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_league_ctx_cache_valid"
    )
    fn_src = ast.get_source_segment(src, fn_node)
    assert fn_src, "could not extract _league_ctx_cache_valid from app.py"
    ns = {
        "time": time,
        "CACHE_TTL": 43200,
        "_league_bust_mtime": lambda *a, **k: 0,
        "_league_ctx_effective_ttl": lambda: 43200,
    }
    exec(compile(ast.parse(fn_src), "app_fn", "exec"), ns)  # noqa: S102
    return ns["_league_ctx_cache_valid"]


@pytest.fixture(scope="module")
def cache_valid():
    return _load_cache_valid_fn()


def test_force_refresh_marker_forces_rebuild(cache_valid):
    entry = {"ctx": {"a": 1}, "ts": time.time(), "page_html": {}, "force_refresh": True}
    assert cache_valid(entry, "sleeper", 2026, "123") is False


def test_fresh_entry_without_marker_stays_valid(cache_valid):
    entry = {"ctx": {"a": 1}, "ts": time.time(), "page_html": {}}
    assert cache_valid(entry, "sleeper", 2026, "123") is True


def test_stale_entry_still_invalid(cache_valid):
    entry = {"ctx": {"a": 1}, "ts": time.time() - 99999, "page_html": {}}
    assert cache_valid(entry, "sleeper", 2026, "123") is False


def test_empty_entry_invalid(cache_valid):
    assert cache_valid(None, "sleeper", 2026, "123") is False
    assert cache_valid({}, "sleeper", 2026, "123") is False


def test_stale_fallback_arithmetic_with_preserved_ts():
    """With the old ts preserved (not zeroed), a 1h-old entry is within the
    6h stale window, so a failed rebuild serves last-known-good data."""
    stale_window = 21600
    old_ts = time.time() - 3600
    assert time.time() - old_ts <= stale_window


def test_stale_fallback_arithmetic_with_zeroed_ts():
    """Documents the old bug: zeroing ts made the fallback check always false."""
    stale_window = 21600
    old_ts = 0
    assert not (time.time() - old_ts <= stale_window)


def test_refresh_endpoint_sets_marker_not_zero_ts():
    src = _read("routes/internal_bp.py")
    # Find the api_refresh_league function body.
    start = src.index("def api_refresh_league")
    end = src.index("@internal_bp.route", start)
    body = src[start:end]
    assert '["force_refresh"] = True' in body
    assert '["ts"] = 0' not in body


def test_build_failure_clears_force_marker():
    src = _read("app.py")
    assert 'pop("force_refresh", None)' in src


def test_client_refresh_timeout_is_60s():
    src = _read("static/app.js")
    assert "brFetchWithTimeout(location.href, opts, 60000)" in src
    assert "brFetchWithTimeout(location.href, opts, 30000)" not in src
