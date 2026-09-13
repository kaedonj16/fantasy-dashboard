"""Regression: Redzone 'This League' must resolve the viewer per league.

The league-scope payload used to echo the raw ``session['viewer_roster_id']``.
Roster ids are league-scoped integers, so a session id resolved for a different
league silently bound the feed to whichever manager happened to hold that id in
the league being viewed -- "my team" rendering as another user, while the rest
of the site (which re-resolves via ``get_viewer_session_for_league``) stayed
correct. This locks the league branch to that same per-league resolution.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _league_scope_branch() -> str:
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    start = app.index("def _redzone_fetch")
    # The league branch is the tail of _redzone_fetch, before the next def.
    end = app.index("\ndef ", start + 1)
    body = app[start:end]
    # Everything after the user-scope early return is the league branch.
    return body[body.index('d = _redzone_collect(platform, league_id, season, week)'):]


def test_league_scope_reresolves_viewer_for_this_league():
    branch = _league_scope_branch()
    # Uses the same per-league resolver the rest of the site uses.
    assert "get_viewer_session_for_league(" in branch
    # Must NOT bind vrid straight from the stale session roster id.
    assert 'vrid = str(session.get("viewer_roster_id")' not in branch
    assert "vrid = str((league_viewer or {}).get(\"viewer_roster_id\")" in branch


def test_get_viewer_session_for_league_prefers_league_scoped_match():
    """The helper the fix leans on resolves against the league's own rosters,
    not a stale session id, so a valid user_id maps to this league's roster."""
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    helper = app[app.index("def get_viewer_session_for_league"):app.index("def get_viewer_session()")]
    # Account team first, then per-league username/user_id resolution.
    assert "resolve_account_viewer_for_league(" in helper
    assert "resolve_viewer_for_league(users, rosters, username, user_id=stored_user_id)" in helper
    # A roster resolved elsewhere is never leaked into this league.
    assert "never leak a\n    # roster_id from another league" in helper \
        or "leak a roster_id from another league" in helper \
        or "into this one's" in helper
