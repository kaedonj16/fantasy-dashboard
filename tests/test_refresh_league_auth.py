"""Auth paths for POST /api/refresh-league.

A Google-signed-in caller often has no Sleeper identity in the session (it is
only stored after explicitly linking a team). The endpoint must fall back to
the Sleeper identities linked to their Google account before returning 403 --
otherwise the dashboard "Refresh Data" button fails for legitimate members.

The auth decision lives in ``_refresh_league_authorized`` (pure function of
the session values) so it can be tested without importing all of app.py.
"""
from __future__ import annotations

import sys
import types

import pytest


def _load(monkeypatch):
    """Import routes.admin_api_bp with heavy deps stubbed; return helper + stubs."""
    app_stub = types.ModuleType("app")
    app_stub.DASHBOARD_CACHE = {}
    app_stub.CACHE_TTL = 43200
    monkeypatch.setitem(sys.modules, "app", app_stub)

    ext_stub = types.ModuleType("extensions")

    class _Limiter:
        def limit(self, *a, **k):
            def deco(fn):
                return fn

            return deco

    ext_stub.limiter = _Limiter()
    monkeypatch.setitem(sys.modules, "extensions", ext_stub)

    pkg = types.ModuleType("dashboard_services")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "dashboard_services", pkg)
    subs_stub = types.ModuleType("dashboard_services.subscriptions")
    monkeypatch.setitem(sys.modules, "dashboard_services.subscriptions", subs_stub)
    accts_stub = types.ModuleType("dashboard_services.accounts")
    accts_stub.list_account_platform_ids = lambda account_id, platform: []
    accts_stub.list_user_leagues = lambda account_id: []
    monkeypatch.setitem(sys.modules, "dashboard_services.accounts", accts_stub)

    sys.modules.pop("routes.admin_api_bp", None)
    import routes.admin_api_bp as bp

    return bp._refresh_league_authorized, subs_stub, accts_stub


def _base_kwargs(**over):
    kw = {
        "platform": "sleeper",
        "season": 2026,
        "league_id": "1312067280816832512",
        "provided_secret": "",
        "last_league_id": "",
        "member_id": None,
        "account_id": None,
    }
    kw.update(over)
    return kw


def test_viewing_this_league_is_allowed(monkeypatch):
    auth, _, _ = _load(monkeypatch)
    assert auth(**_base_kwargs(last_league_id="1312067280816832512")) is True


def test_stranger_is_rejected(monkeypatch):
    auth, subs, _ = _load(monkeypatch)
    subs.viewer_is_league_member = lambda *a, **k: False
    assert auth(**_base_kwargs()) is False


def test_session_member_is_allowed(monkeypatch):
    auth, subs, _ = _load(monkeypatch)
    subs.viewer_is_league_member = lambda uid, *a, **k: uid == "hoodiekj1"
    assert auth(**_base_kwargs(member_id="hoodiekj1")) is True


def test_account_linked_sleeper_identity_is_allowed(monkeypatch):
    """No session identity, but the Google account has a linked Sleeper id
    that belongs to the league -> allowed (the reported 403 case)."""
    auth, subs, accts = _load(monkeypatch)
    accts.list_user_leagues = lambda account_id: []
    accts.list_account_platform_ids = lambda account_id, platform: ["1013151967931162624"]
    seen = []
    def is_member(uid, league_id, platform, season):
        seen.append(uid)
        return uid == "1013151967931162624" and league_id == "1312067280816832512"

    subs.viewer_is_league_member = is_member
    assert auth(**_base_kwargs(account_id=42)) is True
    assert seen == ["1013151967931162624"]


def test_account_linked_league_is_allowed(monkeypatch):
    """The league itself is linked to the Google account -> allowed directly,
    no platform identity needed."""
    auth, subs, accts = _load(monkeypatch)
    accts.list_user_leagues = lambda account_id: [
        {"platform": "sleeper", "league_id": "1312067280816832512", "season": 2026}
    ]
    subs.viewer_is_league_member = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not reach live membership check")
    )
    assert auth(**_base_kwargs(account_id=42)) is True


def test_account_linked_league_season_mismatch_rejected(monkeypatch):
    auth, subs, accts = _load(monkeypatch)
    accts.list_user_leagues = lambda account_id: [
        {"platform": "sleeper", "league_id": "1312067280816832512", "season": 2025}
    ]
    accts.list_account_platform_ids = lambda account_id, platform: []
    subs.viewer_is_league_member = lambda *a, **k: False
    assert auth(**_base_kwargs(account_id=42)) is False


def test_account_linked_league_other_platform(monkeypatch):
    """user_leagues fallback is platform-aware: an ESPN-linked league does not
    authorize a Sleeper refresh for a different league id."""
    auth, subs, accts = _load(monkeypatch)
    accts.list_user_leagues = lambda account_id: [
        {"platform": "espn", "league_id": "12345", "season": 2026}
    ]
    accts.list_account_platform_ids = lambda account_id, platform: []
    subs.viewer_is_league_member = lambda *a, **k: False
    assert auth(**_base_kwargs(account_id=42)) is False


def test_account_linked_nonmember_is_rejected(monkeypatch):
    auth, subs, accts = _load(monkeypatch)
    accts.list_user_leagues = lambda account_id: []
    accts.list_account_platform_ids = lambda account_id, platform: ["999"]
    subs.viewer_is_league_member = lambda *a, **k: False
    assert auth(**_base_kwargs(account_id=42)) is False


def test_account_fallback_checks_all_platforms(monkeypatch):
    """The user_leagues fallback is not Sleeper-only: an ESPN league linked
    to the account authorizes its own refresh."""
    auth, subs, accts = _load(monkeypatch)
    accts.list_user_leagues = lambda account_id: [
        {"platform": "espn", "league_id": "12345", "season": 2026}
    ]
    subs.viewer_is_league_member = lambda *a, **k: False
    assert (
        auth(
            **_base_kwargs(
                platform="espn", league_id="12345", account_id=42
            )
        )
        is True
    )
    # ...but a different league is still rejected.
    assert (
        auth(**_base_kwargs(platform="espn", league_id="99999", account_id=42))
        is False
    )


def test_ops_secret_bypass(monkeypatch):
    auth, subs, _ = _load(monkeypatch)
    subs.viewer_is_league_member = lambda *a, **k: False
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    assert auth(**_base_kwargs(provided_secret="s3cret")) is True
    assert auth(**_base_kwargs(provided_secret="wrong")) is False
