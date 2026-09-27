"""Stale-while-revalidate for /portfolio.

Account users are served from the Postgres portfolio_summary_cache when the
entry is fresh; a miss or stale entry renders a hydratable shell and kicks a
bounded background refresh instead of blocking the page on summary
computation. Sleeper-only sessions (no account_id) keep the inline path.
"""
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("flask")


def _summary(**overrides):
    base = {
        "platform": "sleeper", "league_id": "L1", "season": 2026,
        "name": "Cached League", "state": "ready",
        "wins": 7, "losses": 4, "ties": 0, "record": "7-4", "rank": 2,
        "total_teams": 12, "pf": 1234.5,
        "streak": ["W", "L", "W"],
        "pos_user_rank": {"QB": 3, "RB": 5, "WR": 2, "TE": 8},
        "pos_user_pctile": {"QB": 70.0, "RB": 55.0, "WR": 85.0, "TE": 30.0},
        "all_players": {
            "p1": {"name": "Player One", "position": "WR", "value": 321.0,
                   "pos_rank": "WR12", "nfl_team": "BUF"},
        },
        "total_value": 321.0, "offseason": False, "urgency": 2.8,
        "team_name": "My Team",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    base.update(overrides)
    return base


def _membership(**overrides):
    base = {
        "league_id": "L1", "platform": "sleeper", "season": 2026,
        "name": "Cached League", "is_favorite": True,
    }
    base.update(overrides)
    return base


def _setup(monkeypatch, leagues, cached_by_key, game_day=False):
    """Wire /portfolio with a canned portfolio and cache contents.

    ``cached_by_key`` maps (platform, league_id, season) -> summary dict (or
    None for a miss). Returns the pages module and a list capturing
    _kick_summary_refresh calls.
    """
    import routes.user_pages_bp as pages

    monkeypatch.setattr(pages, "get_nfl_state", lambda: {"season": 2026, "week": 3})
    monkeypatch.setattr(pages, "_games_scheduled_today",
                        lambda season, week: game_day)

    def resolve(viewer_user_id, account_id, current_season, **kw):
        return (list(leagues), 2026)

    monkeypatch.setattr(
        "dashboard_services.accounts.resolve_my_leagues", resolve)
    monkeypatch.setattr(
        "dashboard_services.accounts.schedule_account_league_reconciliation",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        pages, "get_cached_summary",
        lambda account_id, platform, league_id, season: cached_by_key.get(
            (str(platform).lower(), str(league_id), int(season))),
    )
    kicked = []
    monkeypatch.setattr(
        pages, "_kick_summary_refresh",
        lambda account_id, membership: kicked.append(
            (account_id, str(membership.get("league_id")))),
    )
    return pages, kicked


def _signed_in(offline_client, **session_values):
    with offline_client.session_transaction() as sess:
        for key, value in session_values.items():
            sess[key] = value


def test_fresh_cached_summary_used_without_recompute(offline_client, monkeypatch):
    """A fresh cache hit renders the full card and never touches the league
    context or the background refresher."""
    pages, kicked = _setup(
        monkeypatch, [_membership()], {("sleeper", "L1", 2026): _summary()})

    def _no_ctx(*a, **k):
        raise AssertionError("league context must not be read on a fresh hit")

    monkeypatch.setattr(pages, "get_league_ctx_from_cache", _no_ctx)
    _signed_in(offline_client, account_id=7, account_email="a@example.com")

    resp = offline_client.get("/portfolio")
    assert resp.status_code == 200
    assert b"Cached League" in resp.data
    assert b"7-4" in resp.data
    assert b"Player One" in resp.data
    assert b"Record loading" not in resp.data
    assert kicked == []


def test_cache_miss_renders_shell_and_kicks_refresh(offline_client, monkeypatch):
    pages, kicked = _setup(monkeypatch, [_membership()], {})
    _signed_in(offline_client, account_id=7, account_email="a@example.com")

    resp = offline_client.get("/portfolio")
    assert resp.status_code == 200
    # Hydratable skeleton: the client fills it via /api/portfolio/card.
    assert b"Record loading" in resp.data
    assert b'data-summary-card' in resp.data
    assert kicked == [(7, "L1")]


def test_stale_cache_renders_shell_and_kicks_refresh(offline_client, monkeypatch):
    old = _summary(generated_at=(
        datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())
    pages, kicked = _setup(
        monkeypatch, [_membership()], {("sleeper", "L1", 2026): old})
    _signed_in(offline_client, account_id=7, account_email="a@example.com")

    resp = offline_client.get("/portfolio")
    assert resp.status_code == 200
    assert b"Record loading" in resp.data
    assert b"7-4" not in resp.data
    assert kicked == [(7, "L1")]


def test_game_day_uses_shorter_ttl(offline_client, monkeypatch):
    """15-minute TTL on game days vs 1 hour otherwise."""
    thirty_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    key = ("sleeper", "L1", 2026)

    # Game day: 30-minute-old entry is stale -> shell + refresh.
    pages, kicked = _setup(
        monkeypatch, [_membership()],
        {key: _summary(generated_at=thirty_min_ago)}, game_day=True)
    _signed_in(offline_client, account_id=7, account_email="a@example.com")
    resp = offline_client.get("/portfolio")
    assert b"Record loading" in resp.data
    assert kicked == [(7, "L1")]

    # Not a game day: same entry is fresh -> full card, no refresh.
    pages, kicked = _setup(
        monkeypatch, [_membership()],
        {key: _summary(generated_at=thirty_min_ago)}, game_day=False)
    _signed_in(offline_client, account_id=7, account_email="a@example.com")
    resp = offline_client.get("/portfolio")
    assert b"7-4" in resp.data
    assert b"Record loading" not in resp.data
    assert kicked == []


def test_mixed_fresh_and_stale_leagues(offline_client, monkeypatch):
    """Cold cache (all shells), warm cache (full cards), and mixed portfolios
    all render correctly in one pass."""
    leagues = [_membership(league_id="L1", name="Fresh League"),
               _membership(league_id="L2", name="Stale League")]
    cached = {("sleeper", "L1", 2026): _summary(league_id="L1", name="Fresh League")}
    pages, kicked = _setup(monkeypatch, leagues, cached)
    _signed_in(offline_client, account_id=7, account_email="a@example.com")

    resp = offline_client.get("/portfolio")
    assert resp.status_code == 200
    assert b"Fresh League" in resp.data
    assert b"7-4" in resp.data  # fresh league's full card
    assert b"Stale League" in resp.data
    assert b"Record loading" in resp.data  # stale league's shell
    assert kicked == [(7, "L2")]


def test_cached_team_not_linked_renders_pending_card(offline_client, monkeypatch):
    cached = _summary(state="team_not_linked",
                      generated_at=datetime.now(timezone.utc).isoformat())
    # A pending card carries no record/players; strip them like the builder does.
    for key in ("wins", "losses", "ties", "record", "rank", "all_players"):
        cached.pop(key, None)
    pages, kicked = _setup(
        monkeypatch, [_membership()], {("sleeper", "L1", 2026): cached})
    _signed_in(offline_client, account_id=7, account_email="a@example.com")

    resp = offline_client.get("/portfolio")
    assert resp.status_code == 200
    assert b"Team not linked yet" in resp.data
    assert b"Record loading" not in resp.data
    assert kicked == []


def test_sleeper_only_session_keeps_inline_path(offline_client, monkeypatch):
    """No account_id -> no Postgres cache; the inline _league_summary path
    still renders full cards for Sleeper-only sessions."""
    import routes.user_pages_bp as pages

    monkeypatch.setattr(pages, "get_nfl_state", lambda: {"season": 2026, "week": 3})
    monkeypatch.setattr(pages, "_games_scheduled_today", lambda season, week: False)
    monkeypatch.setattr(
        "dashboard_services.accounts.resolve_my_leagues",
        lambda viewer_user_id, account_id, current_season, **kw: ([_membership()], 2026),
    )

    def _no_cache(*a):
        raise AssertionError("no account_id means no cache reads")

    monkeypatch.setattr(pages, "get_cached_summary", _no_cache)
    monkeypatch.setattr(pages, "get_league_ctx_from_cache", lambda *a, **k: {
        "league": {"name": "Sleeper League"},
        "rosters": [{"roster_id": 9, "owner_id": "u1",
                     "players": ["p1", "p2", "p3", "p4", "p5"]}],
        "users": [{"user_id": "u1", "display_name": "My Team"}],
        "players_index": {"p1": {"name": "Player One", "pos": "WR", "team": "BUF"}},
        "total_rosters": 1, "roster_positions": ["QB", "RB", "WR", "TE"],
        "latest_draft": {"status": "complete"},
    })
    monkeypatch.setattr("dashboard_services.accounts.resolve_account_viewer_for_league",
                        lambda *a, **k: {"viewer_roster_id": "9"})
    monkeypatch.setattr("dashboard_services.ai.context_builders.league_format_value_lookup",
                        lambda ctx, _cache=None: {"p1": {"name": "Player One", "position": "WR",
                                              "team": "BUF", "value": 321,
                                              "pos_rank_label": "WR12"}})
    monkeypatch.setattr("dashboard_services.ai.context_builders.portfolio_record_and_rank",
                        lambda *a: (7, 4, 0, 1234.5, 2))
    _signed_in(offline_client, viewer_username="sleeperuser", viewer_user_id="u1")

    resp = offline_client.get("/portfolio")
    assert resp.status_code == 200
    assert b"7-4" in resp.data
    assert b"Record loading" not in resp.data
