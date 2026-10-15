"""Regression: every league-ctx serve path must re-resolve the viewer.

The dashboard actions card flips to "Link your team to see this week's
actions" whenever the rendered ctx carries no viewer roster. The ctx is
cached per league and shared across viewers, and build_league_context bakes
a "viewer" into it at build time -- routinely from a background warm thread
running with an empty test session, which bakes an all-None viewer.
get_league_ctx_from_cache re-stamped the viewer for the current request on
its cache-hit paths (and the Redis load path does too; the Redis slice
excludes "viewer" for exactly this reason), but several serve paths handed
back the baked viewer verbatim:

- the fresh-build return,
- the post-cross-lock cache recheck,
- the generation-bump stale return,
- the ContextLockBusy / LeagueBuildBusy / build-failure stale degrades.

A signed-in, already-linked manager (Google sign-in, league linked through
user_leagues, no session Sleeper identity) whose request landed on one of
those paths was rendered as unlinked, while surfaces that always resolve
per request (portfolio, the BR Assistant) kept finding her team. These
tests drive each path with the account resolver stubbed to her viewer and
assert the baked build-time viewer never reaches the caller.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import pytest

# app.py imports pandas (and flask) at module load. The fast "lint" CI shard
# has flask but not pandas, so guard on pandas first -- otherwise importing
# app here raises at COLLECTION and aborts the whole run.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import app  # noqa: E402
from dashboard_services import league_singleflight  # noqa: E402

PLATFORM, SEASON, LEAGUE_ID = "sleeper", 2026, "1312067280816832512"

# The account-based resolution for a Google-only manager whose league is
# linked via user_leagues: resolve_account_viewer_for_league's output shape.
HER_VIEWER = {
    "viewer_username": "hoodiekj1",
    "viewer_user_id": "sleeper-user-1",
    "viewer_roster_id": "3",
    "viewer_team_name": "Caleb's Casting Couch",
}
# What a session-less background build bakes into the ctx.
BAKED_VIEWER = {
    "viewer_username": None,
    "viewer_user_id": None,
    "viewer_roster_id": None,
    "viewer_team_name": None,
}


def _ctx(viewer):
    return {
        "platform": PLATFORM,
        "league_id": LEAGUE_ID,
        "season": SEASON,
        "users": [{"user_id": "sleeper-user-1", "username": "hoodiekj1"}],
        "rosters": [{"roster_id": 3, "owner_id": "sleeper-user-1"}],
        "viewer": dict(viewer),
    }


class _OkLock:
    def __enter__(self):
        return 0.0

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def serve_env(monkeypatch):
    """Stub every dependency of get_league_ctx_from_cache except the serve
    logic itself. The viewer resolver is a holder so a test can switch the
    "current requester" between calls."""
    key = app._cache_key(PLATFORM, SEASON, LEAGUE_ID)
    app.DASHBOARD_CACHE.pop(key, None)
    holder = {"viewer": dict(HER_VIEWER)}
    monkeypatch.setattr(
        app, "get_viewer_session_for_league",
        lambda users, rosters, *a, **k: dict(holder["viewer"]),
    )
    monkeypatch.setattr(app, "_kick_player_modal_warm", lambda *a, **k: None)
    monkeypatch.setattr(app, "_kick_modal_league_warm", lambda *a, **k: None)
    monkeypatch.setattr(app, "_maybe_check_roster_freshness", lambda *a, **k: None)
    monkeypatch.setattr(app, "_warm_league_ctx_async", lambda *a, **k: None)
    monkeypatch.setattr(app, "_league_bust_mtime", lambda *a, **k: 0.0)
    monkeypatch.setattr(app, "_league_ctx_redis_load", lambda *a, **k: None)
    monkeypatch.setattr(app, "_league_ctx_redis_store", lambda *a, **k: None)
    monkeypatch.setattr(
        app, "_acquire_context_lock", lambda key, timeout=None: (object(), 0.0)
    )
    monkeypatch.setattr(app, "_release_context_lock", lambda *a, **k: None)
    monkeypatch.setattr(
        league_singleflight, "read_generation",
        lambda *a: {"generation": 0, "completed_at": 0.0},
    )
    monkeypatch.setattr(
        league_singleflight, "mark_success",
        lambda *a: {"generation": 1, "completed_at": 0.0},
    )
    monkeypatch.setattr(
        league_singleflight, "league_build_lock", lambda *a, **k: _OkLock()
    )
    import utils.utils as _uu

    monkeypatch.setattr(
        _uu, "clear_league_provider_cache_for_league", lambda *a, **k: None
    )
    yield holder
    app.DASHBOARD_CACHE.pop(key, None)


def _seed_stale(monkeypatch, baked=BAKED_VIEWER):
    """A last-known-good entry that is not cache-valid (force refresh) but
    well inside the stale window: the degrade paths' input."""
    key = app._cache_key(PLATFORM, SEASON, LEAGUE_ID)
    old = _ctx(baked)
    app.DASHBOARD_CACHE[key] = {
        "ctx": old, "ts": time.time() - 5, "page_html": {}, "force_refresh": True,
    }
    return old


def test_fresh_build_restamps_request_viewer(serve_env, monkeypatch):
    # The build ran in a session-less warm context and baked an empty
    # viewer; the requesting manager must still get HER viewer back.
    monkeypatch.setattr(
        app, "build_league_context", lambda *a, **k: _ctx(BAKED_VIEWER)
    )
    ctx = app.get_league_ctx_from_cache(PLATFORM, LEAGUE_ID, SEASON)
    assert ctx["viewer"] == HER_VIEWER
    assert ctx["viewer"]["viewer_roster_id"] == "3"


def test_build_failure_degrade_restamps_stale_viewer(serve_env, monkeypatch):
    old = _seed_stale(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(app, "build_league_context", _boom)
    ctx = app.get_league_ctx_from_cache(PLATFORM, LEAGUE_ID, SEASON)
    assert ctx is old
    assert ctx["viewer"] == HER_VIEWER


def test_league_build_busy_degrade_restamps_stale_viewer(serve_env, monkeypatch):
    old = _seed_stale(monkeypatch)

    class _BusyLock:
        def __enter__(self):
            raise league_singleflight.LeagueBuildBusy("locked")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        league_singleflight, "league_build_lock", lambda *a, **k: _BusyLock()
    )
    ctx = app.get_league_ctx_from_cache(PLATFORM, LEAGUE_ID, SEASON)
    assert ctx is old
    assert ctx["viewer"] == HER_VIEWER


def test_generation_bump_degrade_restamps_stale_viewer(serve_env, monkeypatch):
    old = _seed_stale(monkeypatch)
    generations = iter([0, 5])
    monkeypatch.setattr(
        league_singleflight, "read_generation",
        lambda *a: {"generation": next(generations), "completed_at": 0.0},
    )
    ctx = app.get_league_ctx_from_cache(PLATFORM, LEAGUE_ID, SEASON)
    assert ctx is old
    assert ctx["viewer"] == HER_VIEWER


def test_post_cross_lock_valid_entry_restamps_viewer(serve_env, monkeypatch):
    # Entry starts invalid (force refresh); while this request waits on the
    # cross-worker lock a sibling completes and the entry becomes valid.
    old = _seed_stale(monkeypatch)
    key = app._cache_key(PLATFORM, SEASON, LEAGUE_ID)

    class _SiblingDoneLock:
        def __enter__(self):
            entry = app.DASHBOARD_CACHE[key]
            entry.pop("force_refresh", None)
            entry["ts"] = time.time()
            return 0.0

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        league_singleflight, "league_build_lock",
        lambda *a, **k: _SiblingDoneLock(),
    )
    ctx = app.get_league_ctx_from_cache(PLATFORM, LEAGUE_ID, SEASON)
    assert ctx is old
    assert ctx["viewer"] == HER_VIEWER


def test_viewer_restamped_per_request_no_cross_viewer_leak(serve_env, monkeypatch):
    # The cached ctx is shared: each request must get its OWN viewer, never
    # the previous requester's (and never the baked one).
    def _boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(app, "build_league_context", _boom)
    other = dict(HER_VIEWER, viewer_roster_id="8", viewer_team_name="Other Team")

    _seed_stale(monkeypatch)
    first = app.get_league_ctx_from_cache(PLATFORM, LEAGUE_ID, SEASON)
    assert first["viewer"]["viewer_roster_id"] == "3"

    serve_env["viewer"] = other
    _seed_stale(monkeypatch)
    second = app.get_league_ctx_from_cache(PLATFORM, LEAGUE_ID, SEASON)
    assert second["viewer"] == other


# ── get_viewer_session_for_league composition (the resolver chain the card
# ── ultimately depends on), with the account resolver stubbed at its module.


@pytest.fixture()
def _account_resolver(monkeypatch):
    import dashboard_services.accounts as accounts

    state = {"viewer": None}
    monkeypatch.setattr(
        accounts, "resolve_account_viewer_for_league",
        lambda *a, **k: state["viewer"],
    )
    return state


def test_google_only_session_uses_account_resolution(_account_resolver):
    # (a) Google sign-in: account_id in session, NO Sleeper identity at all.
    # The account resolver finds her roster from the user_leagues link, and
    # that viewer (with a roster id) is what the dashboard renders from.
    _account_resolver["viewer"] = dict(HER_VIEWER)
    with app.app.test_request_context():
        from flask import session

        session["account_id"] = 7
        viewer = app.get_viewer_session_for_league(
            [{"user_id": "sleeper-user-1"}],
            [{"roster_id": 3, "owner_id": "sleeper-user-1"}],
            "sleeper", LEAGUE_ID, SEASON,
        )
    assert viewer["viewer_roster_id"] == "3"
    assert viewer["viewer_team_name"] == "Caleb's Casting Couch"


def test_session_sleeper_identity_still_resolves(monkeypatch):
    # (b) No account: a session Sleeper identity resolves per-league exactly
    # as before the account path existed.
    monkeypatch.setattr(
        app, "resolve_viewer_for_league",
        lambda users, rosters, username, user_id=None: dict(HER_VIEWER),
    )
    with app.app.test_request_context():
        from flask import session

        session["viewer_username"] = "hoodiekj1"
        session["viewer_user_id"] = "sleeper-user-1"
        viewer = app.get_viewer_session_for_league(
            [{"user_id": "sleeper-user-1"}],
            [{"roster_id": 3, "owner_id": "sleeper-user-1"}],
            "sleeper", LEAGUE_ID, SEASON,
        )
    assert viewer["viewer_roster_id"] == "3"


def test_unlinked_account_gets_no_roster(_account_resolver):
    # (c) An account with no link to this league must NOT be matched to any
    # team: the account resolver returns None and no roster is attached, so
    # the link prompt remains the correct state for them.
    _account_resolver["viewer"] = None
    with app.app.test_request_context():
        from flask import session

        session["account_id"] = 99
        viewer = app.get_viewer_session_for_league(
            [{"user_id": "sleeper-user-1"}],
            [{"roster_id": 3, "owner_id": "sleeper-user-1"}],
            "sleeper", LEAGUE_ID, SEASON,
        )
    assert not viewer.get("viewer_roster_id")


def test_refresh_section_restamps_viewer_source_contract():
    src = Path(app.__file__).read_text(encoding="utf-8")
    fn = src[src.index("def refresh_league_ctx_section"):src.index("def _owner_to_rid_map")]
    assert 'ctx["viewer"] = get_viewer_session_for_league(' in fn


def test_every_serve_path_stamps_viewer_source_contract():
    src = Path(app.__file__).read_text(encoding="utf-8")
    fn = src[src.index("def get_league_ctx_from_cache"):src.index("def page_dashboard")]
    assert "def _stamp_viewer(c):" in fn
    # Fresh build, post-cross-lock recheck, generation bump, both busy
    # degrades, build-failure degrade.
    assert fn.count("return _stamp_viewer(") >= 6
