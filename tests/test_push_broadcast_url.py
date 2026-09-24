"""Tests for broadcast push URL resolution (routes/push_bp.py).

Regression: tapping the changelog push notification opened the entry's raw
link (e.g. "/waivers?tab=startsit"), which 404s because league-scoped pages
live under /<platform>/<season>/<league_id>/. The in-app changelog dropdown
prepends that prefix; _resolve_broadcast_url applies the same policy per
recipient from their newest subscription row.
"""
from __future__ import annotations

import datetime

import pytest

pytest.importorskip("flask")

from routes.push_bp import _broadcast_season, _resolve_broadcast_url


def test_league_scoped_path_gets_prefix():
    out = _resolve_broadcast_url("/waivers?tab=startsit", "sleeper", 2026, "abc123")
    assert out == "/sleeper/2026/abc123/waivers?tab=startsit"


def test_dashboard_path_gets_prefix():
    out = _resolve_broadcast_url("/dashboard", "sleeper", 2026, "abc123")
    assert out == "/sleeper/2026/abc123/dashboard"


def test_espn_platform_uses_platform_in_prefix():
    out = _resolve_broadcast_url("/teams", "espn", 2026, "999")
    assert out == "/espn/2026/999/teams"


def test_empty_platform_defaults_to_sleeper():
    out = _resolve_broadcast_url("/waivers", "", 2026, "abc123")
    assert out == "/sleeper/2026/abc123/waivers"


def test_home_is_never_prefixed():
    assert _resolve_broadcast_url("/", "sleeper", 2026, "abc123") == "/"


def test_portfolio_is_global_only():
    assert _resolve_broadcast_url("/portfolio", "sleeper", 2026, "abc123") == "/portfolio"


def test_top_movers_is_global_only():
    assert _resolve_broadcast_url("/top-movers", "sleeper", 2026, "abc123") == "/top-movers"


def test_global_path_with_query_string_stays_global():
    assert _resolve_broadcast_url("/portfolio?x=1", "sleeper", 2026, "abc123") == "/portfolio?x=1"


def test_no_league_context_falls_back_to_home_not_404():
    # No league_id: prefixing is impossible, so "/" beats a guaranteed 404.
    assert _resolve_broadcast_url("/waivers?tab=startsit", "sleeper", 2026, "") == "/"
    assert _resolve_broadcast_url("/waivers?tab=startsit", "sleeper", 2026, None) == "/"


def test_no_season_falls_back_to_home_not_404():
    assert _resolve_broadcast_url("/waivers", "sleeper", None, "abc123") == "/"


def test_absolute_url_passes_through():
    u = "https://example.com/x"
    assert _resolve_broadcast_url(u, "sleeper", 2026, "abc123") == u


def test_broadcast_season_uses_nfl_state(monkeypatch):
    import dashboard_services.api as api

    monkeypatch.setattr(api, "get_nfl_state", lambda: {"season": 2025, "season_type": "regular"})
    assert _broadcast_season() == 2025


def test_broadcast_season_falls_back_to_current_year(monkeypatch):
    import dashboard_services.api as api

    def _boom():
        raise RuntimeError("sleeper down")

    monkeypatch.setattr(api, "get_nfl_state", _boom)
    assert _broadcast_season() == datetime.datetime.now().year
