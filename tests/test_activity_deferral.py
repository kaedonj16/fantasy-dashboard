"""Activity/injury deferral out of the first league-context build.

build_league_context used to pay ~18 get_transactions calls (plus the
injury report's players scan) on every cold build, although only the
Activity page, since-last-visit, Season Wrapped, and the front-office
memo read that data. It now leaves activity_df / injury_df as None
(explicit pending) and the sections fill lazily via ensure_*_bits.
These tests pin: no activity work on first build, fill-on-first-use,
explicit pending states (never a fake-empty), and the conservative
transactions sweep cap.
"""
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")


# ── First build performs no activity work ────────────────────────────────────

def _stub_build_dependencies(appmod, monkeypatch, calls):
    """Stub every provider/global build_league_context touches, offline."""
    monkeypatch.setattr(appmod, "get_league", lambda *a, **k: {
        "league_id": "L1", "settings": {}, "scoring_settings": {},
        "roster_positions": [], "total_rosters": 10,
    })
    monkeypatch.setattr(appmod, "get_users", lambda *a, **k: [])
    monkeypatch.setattr(appmod, "get_rosters", lambda *a, **k: [])
    monkeypatch.setattr(appmod, "get_drafts", lambda *a, **k: [])
    monkeypatch.setattr(appmod, "get_nfl_state", lambda *a, **k: {
        "season": 2026, "season_type": "regular", "week": 4, "leg": 4,
    })
    monkeypatch.setattr(appmod, "_regular_season_week_with_games", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "sync_league_globals", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "get_players_global", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "load_players_index", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "load_teams_index", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "get_players_map", lambda players: {})
    monkeypatch.setattr(
        appmod, "build_tables",
        lambda **k: (pd.DataFrame(), pd.DataFrame(), {}),
    )
    monkeypatch.setattr(appmod, "get_nfl_scores_for_date", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "build_team_game_lookup", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "get_model_value_table_cached", lambda: [])
    monkeypatch.setattr(appmod, "get_viewer_session_for_league", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_rookie_rankings_for_ctx", lambda: [])
    monkeypatch.setattr(
        "utils.draft_capital.provider_exposes_draft_capital", lambda *a, **k: False)
    monkeypatch.setattr(
        "utils.draft_capital.has_future_draft_capital", lambda *a, **k: False)

    def _activity(*a, **k):
        calls.append("activity")
        return pd.DataFrame(columns=["kind", "week", "ts", "data"])

    def _injury(*a, **k):
        calls.append("injury")
        return pd.DataFrame()

    monkeypatch.setattr(appmod, "build_week_activity", _activity)
    monkeypatch.setattr(appmod, "build_injury_report", _injury)


def test_first_build_makes_no_activity_or_injury_calls(monkeypatch):
    import app as appmod

    calls: list = []
    _stub_build_dependencies(appmod, monkeypatch, calls)

    ctx = appmod.build_league_context("sleeper", "L1", 2026)

    assert calls == []  # no build_week_activity / build_injury_report
    # Keys present, explicitly pending -- not empty frames posing as data.
    assert ctx["activity_df"] is None
    assert ctx["injury_df"] is None


# ── Lazy fill ────────────────────────────────────────────────────────────────

def _pending_ctx():
    return {
        "league_id": "L1",
        "resolved_league_id": "L1",
        "platform": "sleeper",
        "season": 2026,
        "activity_df": None,
        "injury_df": None,
        "players_map": {"p1": {"name": "Added Player"}},
        "users": [],
        "rosters": [],
        "players": {},
        "roster_map": {},
    }


def test_ensure_activity_bits_fills_once_from_ctx(monkeypatch):
    import app as appmod

    calls: list = []
    activity_frame = pd.DataFrame([{
        "kind": "waiver", "week": 4,
        "ts": datetime.now(timezone.utc),
        "data": {"rid": "9", "name": "Other", "adds": [{"name": "Added Player"}]},
    }])
    injury_frame = pd.DataFrame([{"PlayerID": "p1", "Injury": "Questionable"}])

    def _activity(league_id, platform, season, players_map, users=None, rosters=None):
        calls.append(("activity", league_id, platform, season))
        return activity_frame

    def _injury(*a, **k):
        calls.append(("injury",))
        return injury_frame

    monkeypatch.setattr(appmod, "build_week_activity", _activity)
    monkeypatch.setattr(appmod, "build_injury_report", _injury)

    ctx = _pending_ctx()
    appmod.ensure_activity_bits(ctx)

    assert ctx["activity_df"] is activity_frame
    assert ctx["injury_df"] is injury_frame
    assert calls == [("injury",), ("activity", "L1", "sleeper", 2026)]

    # Second call is a no-op: the fill happens exactly once.
    appmod.ensure_activity_bits(ctx)
    assert len(calls) == 2


def test_activity_page_body_fills_pending_then_renders_real_data(monkeypatch):
    import app as appmod
    from dashboard_services.pages.activity_page import build_activity_body

    activity_frame = pd.DataFrame([{
        "kind": "waiver", "week": 4,
        "ts": datetime.now(timezone.utc),
        "data": {
            "rid": "3", "name": "Team Three",
            "adds": [{"name": "Added Player", "pid": "p1", "pos": "WR", "team": "BUF"}],
        },
    }])
    monkeypatch.setattr(appmod, "build_week_activity", lambda *a, **k: activity_frame)
    monkeypatch.setattr(appmod, "build_injury_report", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(appmod, "load_pick_value_table", lambda: {})

    ctx = _pending_ctx()
    ctx.update({"standings_map": {}, "model_value_table": [], "scoring_settings": {}})

    body = build_activity_body(ctx)

    # The consumer triggered the fill and rendered the real rows, not the
    # "No recent activity yet" empty state.
    assert ctx["activity_df"] is activity_frame
    assert "Added Player" in body
    assert "No recent activity yet" not in body


# ── Activity page route: pending ctx gets the explicit loading state ──────

class _FakeThread:
    started: list = []

    def __init__(self, target=None, args=(), daemon=None, name=None):
        self.target = target
        self.args = args

    def start(self):
        _FakeThread.started.append(self)


def _seed_activity_route(appmod, monkeypatch, ctx):
    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False)
    key = appmod._cache_key("sleeper", 2026, "L1")
    appmod.DASHBOARD_CACHE[key] = {"ctx": ctx, "ts": 0, "page_html": {}}
    monkeypatch.setattr(appmod, "get_page_html_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "store_page_html", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_league_ctx_cache_valid", lambda *a, **k: True)
    monkeypatch.setattr(
        appmod, "render_page", lambda *a, **k: a[3] if len(a) > 3 else "")
    _FakeThread.started = []
    monkeypatch.setattr(appmod.threading, "Thread", _FakeThread)
    appmod.app.config["TESTING"] = True
    return key


def test_activity_route_pending_ctx_shows_loading_skeleton(monkeypatch):
    import app as appmod

    key = _seed_activity_route(appmod, monkeypatch, _pending_ctx())
    try:
        with appmod.app.test_client() as client:
            response = client.get("/sleeper/2026/L1/activity")
    finally:
        appmod.DASHBOARD_CACHE.pop(key, None)

    body = response.get_data(as_text=True)
    # Explicit loading state + a background build was kicked off -- the
    # pending section is never rendered as an empty feed.
    assert "Loading league activity" in body
    assert _FakeThread.started


def test_activity_route_built_ctx_renders_synchronously(monkeypatch):
    import app as appmod

    ctx = _pending_ctx()
    ctx["activity_df"] = pd.DataFrame(columns=["kind", "week", "ts", "data"])
    ctx["injury_df"] = pd.DataFrame()
    key = _seed_activity_route(appmod, monkeypatch, ctx)
    monkeypatch.setattr(appmod, "build_activity_body", lambda c: "SYNC BODY")
    try:
        with appmod.app.test_client() as client:
            response = client.get("/sleeper/2026/L1/activity")
    finally:
        appmod.DASHBOARD_CACHE.pop(key, None)

    assert response.get_data(as_text=True) == "SYNC BODY"
    assert _FakeThread.started == []


# ── Since-last-visit: pending is explicit and non-destructive ───────────────

def _waiver_frame(minutes_ago=3):
    return pd.DataFrame([{
        "kind": "waiver", "week": 4,
        "ts": datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        "data": {"rid": "9", "name": "Other", "adds": [{"name": "B"}]},
    }])


def test_since_last_visit_pending_is_explicit_and_preserves_baseline(monkeypatch):
    import app as appmod
    import dashboard_services.accounts as accounts

    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False)
    ctx = {"rosters": [], "activity_df": None, "injury_df": None}
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a, **k: ctx)

    fills: list = []
    monkeypatch.setattr(
        appmod, "_fill_activity_section_async", lambda *a, **k: fills.append(a))
    consumed: list = []
    monkeypatch.setattr(
        accounts, "consume_league_visit", lambda *a, **k: consumed.append(a))
    appmod.app.config["TESTING"] = True

    with appmod.app.test_client() as client:
        with client.session_transaction() as signed_in:
            signed_in["account_id"] = 42
        response = client.get(
            "/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7&since=1"
        )

    payload = response.get_json()
    # Explicit pending marker (not a fake "0 trades / 0 waivers" answer),
    # background fill kicked off, and the one-time visit was NOT consumed.
    assert payload["activity_pending"] is True
    assert fills and fills[0][1] == "L"
    assert consumed == []

    # Once the fill lands, the same request reports the real activity.
    ctx["activity_df"] = _waiver_frame()
    with appmod.app.test_client() as client:
        response = client.get(
            "/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7&since=1"
        )
    payload = response.get_json()
    assert not payload.get("activity_pending")
    assert payload["waivers"] == 1
    assert [item["text"] for item in payload["items"]] == ["Other added B"]


def test_since_last_visit_without_baseline_skips_pending(monkeypatch):
    """No since + no account => activity is never consulted, so a pending
    section must not block the roster snapshot path."""
    import app as appmod

    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False)
    ctx = {"rosters": [], "activity_df": None, "injury_df": None}
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a, **k: ctx)
    fills: list = []
    monkeypatch.setattr(
        appmod, "_fill_activity_section_async", lambda *a, **k: fills.append(a))
    appmod.app.config["TESTING"] = True

    with appmod.app.test_client() as client:
        response = client.get(
            "/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id="
        )

    payload = response.get_json()
    assert not payload.get("activity_pending")
    assert fills == []


# ── Transactions sweep cap ───────────────────────────────────────────────────

def _live_state(**overrides):
    state = {
        "season": 2026,
        "season_type": "regular",
        "week": 4,
        "freshness": {"classification": "live", "stale": False},
    }
    state.update(overrides)
    return state


def test_activity_sweep_weeks_boundaries(monkeypatch):
    from dashboard_services import service as svc

    full = list(range(1, 19))

    def sweep(state, season=2026):
        if isinstance(state, Exception):
            def _raise():
                raise state
            monkeypatch.setattr(svc, "get_nfl_state", _raise)
        else:
            monkeypatch.setattr(svc, "get_nfl_state", lambda: state)
        return svc._activity_sweep_weeks(season)

    # Plain in-season: only weeks up to the current one can hold transactions.
    assert sweep(_live_state()) == [1, 2, 3, 4]
    assert sweep(_live_state(week=1)) == [1]
    assert sweep(_live_state(week=18)) == full
    assert sweep(_live_state(week=17)) == list(range(1, 18))

    # Anything ambiguous keeps the full sweep.
    assert sweep(_live_state(week=0)) == full
    assert sweep(_live_state(week=19)) == full
    assert sweep(_live_state(season_type="pre")) == full
    assert sweep(_live_state(season_type="post")) == full
    assert sweep(_live_state(season_type="off")) == full
    assert sweep(_live_state(season=2025)) == full          # not the live season
    assert sweep(_live_state(), season=2025) == full        # past-season league
    assert sweep(_live_state(freshness={"classification": "live", "stale": True})) == full
    assert sweep(_live_state(freshness={"classification": "fallback"})) == full
    assert sweep(_live_state(freshness={})) == full
    assert sweep({}) == full
    assert sweep(RuntimeError("state down")) == full


def test_build_week_activity_uses_capped_sweep(monkeypatch):
    from dashboard_services import service as svc

    monkeypatch.setattr(svc, "get_nfl_state", lambda: _live_state())
    recorded: list = []

    def _tx(league_id, season_weeks, platform=None, season=0):
        recorded.append(list(season_weeks))
        return {}

    monkeypatch.setattr(svc, "get_transactions_by_week", _tx)
    frame = svc.build_week_activity(
        "L1", "sleeper", 2026, {}, users=[], rosters=[])
    assert recorded == [[1, 2, 3, 4]]
    assert frame.empty
