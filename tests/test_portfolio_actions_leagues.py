"""The /api/portfolio-actions payload ships a `leagues` array: the
per-league state (board_meta) the scan computes for EVERY scanned
league, not only the leagues holding board actions. The deep
dive's Brief narrates the leagues themselves (posture split,
flagship and deep-end records), which is impossible from board
groups alone: group_actions only creates a group for a league
holding a top-24 action, so a quiet league used to be invisible
to the report. These tests pin that the array covers a league
with no actions at all, carries the free facts the scan already
had in hand (points for, trade deadline week, playoff start
week), and survives the per-league fragment path (#2836) so a
fragment-built payload keeps the full league layer."""
import time

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _ctx(name, viewer_settings, others, settings):
    rosters = [{
        "roster_id": 1, "owner_id": "owner-1", "starters": [],
        "players": [], "reserve": [], "taxi": [],
        "settings": viewer_settings,
    }]
    for i, (oid, st) in enumerate(others, start=2):
        rosters.append({
            "roster_id": i, "owner_id": oid, "starters": [],
            "players": [], "reserve": [], "taxi": [], "settings": st,
        })
    return {
        "rosters": rosters,
        "league": {"name": name, "settings": settings},
        "users": [{"user_id": "owner-1", "display_name": "tester"}],
        "roster_positions": ["QB", "BN"],
        "proj_by_week": {},
        "players_index": {},
        "model_value_table": [],
    }


_CTX = {
    "LA": _ctx("Alpha", {"wins": 6, "losses": 1, "fpts": 812.5},
               [("o2", {"wins": 3, "losses": 4, "fpts": 600.0}),
                ("o3", {"wins": 1, "losses": 6, "fpts": 500.0})],
               {"trade_deadline": 10, "playoff_week_start": 15}),
    "LB": _ctx("Beta", {"wins": 2, "losses": 5, "fpts": 601.25},
               [("o2", {"wins": 5, "losses": 2, "fpts": 700.0}),
                ("o3", {"wins": 4, "losses": 3, "fpts": 650.0})],
               {}),
}


def _client(monkeypatch, provider):
    from flask import Flask

    import dashboard_services.accounts as accounts
    import dashboard_services.api as ds_api
    import dashboard_services.portfolio_board as pb
    import dashboard_services.subscriptions as subs
    import dashboard_services.ai.context_builders as cb
    import routes.user_pages_bp as bp_mod
    import utils.utils as uu

    bp_mod._ACTIONS_CACHE.clear()
    bp_mod._ACTIONS_INFLIGHT.clear()
    bp_mod._ACTIONS_PENDING.clear()
    bp_mod._ACTIONS_FRAGMENTS.clear()
    bp_mod._ACTIONS_LATEST.clear()

    monkeypatch.setattr(
        bp_mod, "get_nfl_state",
        lambda *a, **k: {"season": 2026, "week": 7,
                         "season_type": "reg"})
    monkeypatch.setattr(bp_mod, "get_league_ctx_from_cache", provider)
    monkeypatch.setattr(
        bp_mod, "get_model_value_table_cached", lambda *a, **k: [])
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda uid, aid, season: ([
            {"league_id": "LA", "platform": "sleeper", "season": 2026,
             "name": "Alpha"},
            {"league_id": "LB", "platform": "sleeper", "season": 2026,
             "name": "Beta"},
        ], 2026))
    monkeypatch.setattr(subs, "has_premium_for_viewer",
                        lambda *a, **k: True)
    monkeypatch.setattr(ds_api, "get_nfl_players", lambda *a, **k: {})
    monkeypatch.setattr(uu, "load_week_schedule",
                        lambda season, week: [])
    # Deterministic board meta: the sim-backed odds and the value-
    # table posture are not what this test is about.
    monkeypatch.setattr(pb, "odds_for_ctx",
                        lambda *a, **k: {"pct": 62.0, "delta": None})
    monkeypatch.setattr(pb, "window_posture_for",
                        lambda *a, **k: {"label": "Contender",
                                         "cls": "win"})
    monkeypatch.setattr(cb, "league_format_value_lookup",
                        lambda ctx, _cache=None: (lambda pid: 0.0))
    monkeypatch.setattr(cb, "ctx_scoring_type",
                        lambda ctx: "dynasty")
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(bp_mod.user_pages_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "owner-1"
    return client, bp_mod


def _wait_for_idle(bp_mod, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not bp_mod._ACTIONS_PENDING and not bp_mod._ACTIONS_INFLIGHT:
            return True
        time.sleep(0.05)
    return False


def _completed_payload(client, bp_mod):
    client.get("/api/portfolio-actions")
    assert _wait_for_idle(bp_mod)
    data = client.get("/api/portfolio-actions").get_json()
    assert data["scan"]["incomplete"] is False
    return data


def test_leagues_array_covers_leagues_without_actions(monkeypatch):
    client, bp_mod = _client(
        monkeypatch, lambda plat, lid, season, allow_build=True:
        _CTX.get(lid, {}))
    data = _completed_payload(client, bp_mod)
    # Neither league produced a single action (empty rosters, and
    # Alpha's deadline is three weeks out), so the board has no
    # groups at all; the leagues array still covers both.
    assert data["board"]["groups"] == []
    by_key = {e["key"]: e for e in data["leagues"]}
    assert set(by_key) == {"sleeper|LA", "sleeper|LB"}
    alpha = by_key["sleeper|LA"]
    assert alpha["league_name"] == "Alpha"
    assert alpha["record_line"] == "6-1 · 1st of 3"
    assert alpha["posture"] == "Contender"
    assert alpha["wins"] == 6 and alpha["losses"] == 1
    assert alpha["rank"] == 1 and alpha["team_count"] == 3
    assert alpha["odds_pct"] == 62.0
    # The free riders: points for off the roster settings the
    # standing sort already read, calendar weeks off the league
    # settings already in the loop.
    assert alpha["points_for"] == 812.5
    assert alpha["trade_deadline_week"] == 10
    assert alpha["playoff_week_start"] == 15
    beta = by_key["sleeper|LB"]
    assert beta["record_line"] == "2-5 · 3rd of 3"
    assert beta["points_for"] == 601.25
    # Beta's league settings carry no calendar weeks: the keys are
    # absent, not zero-filled or invented.
    assert "trade_deadline_week" not in beta
    assert "playoff_week_start" not in beta


def test_leagues_array_survives_the_fragment_path(monkeypatch):
    state = {"cold": False}

    def provider(plat, lid, season, allow_build=True):
        if state["cold"]:
            return {}
        return _CTX.get(lid, {})

    client, bp_mod = _client(monkeypatch, provider)
    first = _completed_payload(client, bp_mod)
    assert len(first["leagues"]) == 2
    assert bp_mod._ACTIONS_FRAGMENTS

    # Expire the assembled payloads but keep the per-league
    # fragments; go cold so no context can be re-read. The next
    # completed payload is built entirely from fragments and must
    # carry the identical league layer.
    bp_mod._ACTIONS_CACHE.clear()
    bp_mod._ACTIONS_LATEST.clear()
    state["cold"] = True
    second = _completed_payload(client, bp_mod)
    assert second["leagues"] == first["leagues"]
