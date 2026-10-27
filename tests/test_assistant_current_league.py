"""resolve_league must align with the user's CURRENT league.

Regression (2026-10-27): on the My Leagues page the header pill showed
blackedraw (the session's last league, which league-less pages render
their chrome from), but the assistant answered for League of Greatness
and said so ("I'm assuming League of Greatness"). With no explicit
league, resolve_league picked list_user_leagues()[0] - favorite-first,
i.e. the most recently favorited league - and flagged it uncertain.

The fix: an owned league matching the session's last league wins, and
it is not an assumption (assumed_league_uncertain stays False). The
favorite-first default only applies when there is no current league.
"""
from __future__ import annotations

import pytest

flask = pytest.importorskip("flask")

from dashboard_services import accounts
from dashboard_services.assistant import context as ctx_mod

_LEAGUES = [
    {"platform": "sleeper", "league_id": "GREATNESS", "season": 2026,
     "name": "League of Greatness", "is_favorite": True},
    {"platform": "sleeper", "league_id": "BLACKEDRAW", "season": 2026,
     "name": "blackedraw", "is_favorite": True},
    {"platform": "espn", "league_id": "999", "season": 2026,
     "name": "Josh Allen's Twitter", "is_favorite": True},
]


@pytest.fixture()
def app_ctx(monkeypatch):
    app = flask.Flask(__name__)
    app.secret_key = "test"
    monkeypatch.setattr(accounts, "list_user_leagues", lambda account_id: list(_LEAGUES))
    monkeypatch.setattr(
        ctx_mod, "get_league_ctx_from_cache",
        lambda platform, league_id, season: {"league": {"name": "ctx-name"}, "rosters": [], "users": [],
                                             "viewer": {"roster_id": 1}},
    )
    with app.test_request_context():
        flask.session["account_id"] = 7
        yield flask.session


def test_current_session_league_wins_over_favorite_first(app_ctx):
    app_ctx["last_league_id"] = "BLACKEDRAW"
    app_ctx["last_platform"] = "sleeper"
    app_ctx["last_season"] = 2026
    out = ctx_mod.resolve_league("acct:7")
    assert out["league_id"] == "BLACKEDRAW"
    assert out["league_name"] == "blackedraw"
    assert out["assumed_league_uncertain"] is False


def test_no_current_league_falls_back_to_favorite_first_uncertain(app_ctx):
    out = ctx_mod.resolve_league("acct:7")
    assert out["league_id"] == "GREATNESS"
    assert out["assumed_league_uncertain"] is True


def test_unowned_session_league_is_ignored(app_ctx):
    app_ctx["last_league_id"] = "NOT-MINE"
    app_ctx["last_platform"] = "sleeper"
    app_ctx["last_season"] = 2026
    out = ctx_mod.resolve_league("acct:7")
    assert out["league_id"] == "GREATNESS"
    assert out["assumed_league_uncertain"] is True


def test_session_league_from_another_platform_does_not_match(app_ctx):
    # Same league id on a different platform is a different league.
    app_ctx["last_league_id"] = "999"
    app_ctx["last_platform"] = "sleeper"
    app_ctx["last_season"] = 2026
    out = ctx_mod.resolve_league("acct:7")
    assert out["league_id"] == "GREATNESS"
    assert out["assumed_league_uncertain"] is True


def test_changelog_and_sw_for_assistant_league_fix():
    from pathlib import Path

    from dashboard_services.changelog import CHANGELOG

    top = CHANGELOG[5]  # prepended above it: graphs overhaul, holdings sort + insights speed, the rooting/bye cleanup, the Front Office scan fix, the trade fair-band fix
    assert top["date"] == "2026-10-27"
    assert top["tag"] == "fix"
    assert "BR Assistant" in top["text"]
    assert "—" not in top["text"]  # no em dashes in UI copy
    sw = (Path(__file__).resolve().parents[1] / "static" / "sw.js").read_text(encoding="utf-8")
    assert "br-fantasy-v119" in sw


def test_explicit_league_still_wins(app_ctx):
    app_ctx["last_league_id"] = "BLACKEDRAW"
    app_ctx["last_platform"] = "sleeper"
    app_ctx["last_season"] = 2026
    out = ctx_mod.resolve_league("acct:7", platform="espn", league_id="999", season=2026)
    assert out["league_id"] == "999"
    assert out["platform"] == "espn"
    assert out["assumed_league_uncertain"] is False
