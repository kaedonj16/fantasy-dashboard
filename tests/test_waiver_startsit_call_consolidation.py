"""Call-consolidation tests for the Waivers / Start-Sit PR.

Performance refactor only: candidates, verdicts and Lab numbers are
unchanged; what changes is how often the expensive builds run.

1. /api/waiver-candidates serves an identical repeat request from a
   bounded result cache (the pipeline is counted once), misses when a
   roster changes, and the three free-agent endpoints share ONE
   model-value index build keyed on the table's identity.
2. The player-modal Start/Sit strip distills a second roster player from
   the cached roster options payload (one build for the whole roster),
   while guest (non-roster) players keep their per-player build.
3. The Lineup Lab payload cache falls back to the shared (Redis) tier
   when the in-process tier is cold, and the waivers page warms the Lab
   only on toggle intent, never on page load.
"""

import copy

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


# ── 1a. Candidates result cache: K/DEF path counts the pipeline ─────────────

_K_ROWS = [
    {"player_id": "k2", "name": "Elite Kicker", "position": "K",
     "team": "BUF", "opponent": "MIA", "matchup": "vs MIA",
     "own_implied": 27.5, "stream_score": 134.0},
    {"player_id": "k3", "name": "Mid Kicker", "position": "K",
     "team": "MIA", "opponent": "BUF", "matchup": "@ BUF",
     "own_implied": 20.0, "stream_score": 74.0},
]
_DEF_ROWS = [
    {"player_id": "KC", "name": "KC D/ST", "position": "DEF", "team": "KC",
     "opponent": "LV", "matchup": "vs LV", "opp_implied": 16.5,
     "stream_score": 126.0},
]
_KDEF_CTX = {
    "current_week": 4,
    "league": {"settings": {"waiver_type": 0}},
    "rosters": [{
        "roster_id": 1,
        "players": ["k1", "qb1", "rb1", "wr1", "te1", "dst1", "b1", "b2"],
        "reserve": [], "taxi": [], "settings": {},
    }],
    "users": [],
    "players_index": {
        "k1": {"name": "My Kicker", "pos": "K", "team": "KC"},
        "qb1": {"name": "My QB", "pos": "QB", "team": "BUF"},
        "rb1": {"name": "My RB", "pos": "RB", "team": "DAL"},
        "wr1": {"name": "My WR", "pos": "WR", "team": "PHI"},
        "te1": {"name": "My TE", "pos": "TE", "team": "SF"},
        "dst1": {"name": "My D/ST", "pos": "DEF", "team": "BAL"},
        "b1": {"name": "Bench One", "pos": "RB", "team": "GB"},
        "b2": {"name": "Bench Two", "pos": "WR", "team": "DET"},
    },
    "roster_positions": ["QB", "RB", "WR", "TE", "K", "DEF", "BN", "BN"],
    "raw_scoring_settings": {},
}
_FWD = {"k1": 5.0, "k2": 9.0, "k3": 4.0, "qb1": 20.0, "rb1": 12.0,
        "wr1": 11.0, "te1": 8.0, "dst1": 7.0, "b1": 3.0, "b2": 2.0}
_VTABLE = [{"id": p, "value": 1000, "sf_value": 1000}
           for p in ("k1", "k2", "k3", "qb1", "rb1", "wr1", "te1",
                     "dst1", "b1", "b2", "KC")]


def _kdef_env(monkeypatch):
    from flask import Flask
    import routes.waiver_bp as waiver

    ctx = copy.deepcopy(_KDEF_CTX)
    calls = {"streaming": 0}

    def _stream(c, season, players_index=None):
        calls["streaming"] += 1
        return {"defense": [dict(r) for r in _DEF_ROWS],
                "kicker": [dict(r) for r in _K_ROWS],
                "in_season": True, "uses_k": True, "uses_def": True}

    app = Flask(__name__)
    app.register_blueprint(waiver.waiver_bp)
    monkeypatch.setattr(waiver, "get_nfl_state", lambda: {
        "season": 2026, "week": 4, "display_week": 4,
        "season_type": "reg"})
    monkeypatch.setattr(waiver, "get_league_ctx_from_cache", lambda *a: ctx)
    monkeypatch.setattr(waiver, "_streaming_targets", _stream)
    monkeypatch.setattr(waiver, "_forward_ppg_map",
                        lambda season, raw, entries: dict(_FWD))
    monkeypatch.setattr(waiver, "_sleeper_trending_adds", lambda limit=50: [])
    monkeypatch.setattr(waiver, "get_model_value_table_cached",
                        lambda *a, **k: _VTABLE)
    waiver._WAIVER_CANDIDATES_CACHE.clear()
    waiver._VALUE_INDEX_MEMO["key"] = None
    waiver._VALUE_INDEX_MEMO["index"] = None
    return app.test_client(), calls, ctx


_K_URL = ("/api/waiver-candidates?platform=sleeper&league_id=L"
          "&season=2026&position=K&rid=1")


def test_candidates_repeat_request_served_from_cache(monkeypatch):
    client, calls, _ctx = _kdef_env(monkeypatch)
    r1 = client.get(_K_URL)
    assert r1.status_code == 200
    assert calls["streaming"] == 1
    r2 = client.get(_K_URL)
    assert r2.status_code == 200
    # The repeat was served from the result cache: no second pipeline run,
    # and the payload is byte-identical.
    assert calls["streaming"] == 1
    assert r2.get_json() == r1.get_json()
    # A different position tab is a different cache key and computes once.
    r3 = client.get(_K_URL.replace("position=K", "position=DEF"))
    assert r3.status_code == 200
    assert calls["streaming"] == 2


def test_candidates_roster_change_misses_cache(monkeypatch):
    client, calls, ctx = _kdef_env(monkeypatch)
    assert client.get(_K_URL).status_code == 200
    assert calls["streaming"] == 1
    # Moving a bench player to reserve changes the roster signature: the
    # cached payload must NOT be served for the new roster state.
    ctx["rosters"][0]["players"].remove("b2")
    ctx["rosters"][0]["reserve"] = ["b2"]
    assert client.get(_K_URL).status_code == 200
    assert calls["streaming"] == 2
    # And the new state caches in turn.
    assert client.get(_K_URL).status_code == 200
    assert calls["streaming"] == 2


# ── 1b. Candidates result cache: skill path counts one pipeline seam ────────

_SS_POS = {"will": "QB", "nix": "QB", "purdy": "QB",
           "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB",
           "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR", "wr5": "WR",
           "te1": "TE", "te2": "TE",
           "mayf": "QB", "wrb": "RB", "wwr": "WR"}
_SS_NAMES = {"will": "Caleb Williams", "nix": "Bo Nix",
             "purdy": "Brock Purdy", "mayf": "Baker Mayfield",
             "wrb": "Rising Back", "wwr": "Wire Wideout"}
_SS_TEAMS = {"will": "CHI", "nix": "DEN", "purdy": "SF", "mayf": "TB"}
_SS_VAL = {"will": 2400, "nix": 2100, "purdy": 1500,
           "rb1": 4000, "rb2": 2600, "rb3": 1500, "rb4": 800,
           "wr1": 4200, "wr2": 3400, "wr3": 2600, "wr4": 1400, "wr5": 700,
           "te1": 2400, "te2": 600,
           "mayf": 2600, "wrb": 2100, "wwr": 900}
_SS_PROJ = {"will": 19.5, "nix": 18.0, "purdy": 16.5,
            "rb1": 15.0, "rb2": 12.0, "rb3": 9.0, "rb4": 6.0,
            "wr1": 15.0, "wr2": 13.0, "wr3": 11.0, "wr4": 8.0, "wr5": 5.0,
            "te1": 10.0, "te2": 4.0, "wrb": 13.5, "wwr": 7.0}
_SS_ROSTER = ["will", "nix", "purdy", "rb1", "rb2", "rb3", "rb4",
              "wr1", "wr2", "wr3", "wr4", "wr5", "te1", "te2"]
_SS_RP = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
          "BN", "BN", "BN", "BN", "BN", "BN"]


def test_candidates_skill_path_repeat_served_from_cache(monkeypatch):
    import data_building.weekly_metrics as wm_mod
    from flask import Flask
    import routes.waiver_bp as waiver

    ctx = {
        "current_week": 7, "current_season": 2026,
        "league": {"type": 2, "settings": {"type": 2, "waiver_type": 0}},
        "settings": {"type": 2, "waiver_type": 0},
        "scoring_settings": {}, "raw_scoring_settings": {"rec": 0.5},
        "rosters": [
            {"roster_id": 1, "players": list(_SS_ROSTER),
             "reserve": [], "taxi": []},
            {"roster_id": 2, "players": ["o1", "o2", "o3"],
             "reserve": [], "taxi": []},
        ],
        "users": [],
        "players_index": {
            **{p: {"name": _SS_NAMES.get(p, p), "pos": _SS_POS[p],
                   "team": _SS_TEAMS.get(p, "BUF")} for p in _SS_POS},
            **{p: {"name": p, "pos": "RB", "team": "GB"}
               for p in ("o1", "o2", "o3")},
        },
        "roster_positions": list(_SS_RP),
    }
    rows = [{"id": p, "name": _SS_NAMES.get(p, p), "position": _SS_POS[p],
             "team": _SS_TEAMS.get(p, "BUF"), "value": _SS_VAL[p],
             "sf_value": _SS_VAL[p], "age": 25, "pos_rank_label": ""}
            for p in _SS_POS]
    fwd = {**_SS_PROJ, "mayf": 21.5}
    counts = {"trends": 0}

    def _trends(*a, **k):
        counts["trends"] += 1
        return {}

    app = Flask(__name__)
    app.register_blueprint(waiver.waiver_bp)
    monkeypatch.setattr(waiver, "get_nfl_state",
                        lambda: {"season": 2026, "week": 7,
                                 "season_type": "reg"})
    monkeypatch.setattr(waiver, "get_league_ctx_from_cache", lambda *a: ctx)
    monkeypatch.setattr(waiver, "get_model_value_table_cached",
                        lambda *a, **k: rows)
    monkeypatch.setattr(waiver, "get_players_global", lambda *a, **k: {
        p: {"injury_status": "", "team": _SS_TEAMS.get(p, "BUF")}
        for p in _SS_POS})
    monkeypatch.setattr(waiver, "_forward_ppg_map",
                        lambda season, raw, entries: dict(fwd))
    monkeypatch.setattr(waiver, "_sleeper_trending_adds", lambda limit=50: [])
    monkeypatch.setattr(wm_mod, "get_usage_trends", _trends)
    monkeypatch.setattr("utils.utils.load_week_projection",
                        lambda season, week: {})
    _teams = ["CHI", "DEN", "SF", "TB", "BUF", "GB"]
    games = [{"away": _teams[i], "home": _teams[i + 1]}
             for i in range(0, len(_teams) - 1, 2)]
    monkeypatch.setattr("utils.utils.load_week_schedule",
                        lambda season, week: games)
    monkeypatch.setattr("utils.utils.load_usage_table", lambda *a, **k: [
        {"player_id": p, "ppr_ppg": fwd[p]} for p in fwd])
    waiver._WAIVER_CANDIDATES_CACHE.clear()

    client = app.test_client()
    url = ("/api/waiver-candidates?platform=sleeper&league_id=L1"
           "&season=2026&rid=1")
    r1 = client.get(url)
    assert r1.status_code == 200
    assert counts["trends"] == 1  # the pipeline ran exactly once
    r2 = client.get(url)
    assert r2.status_code == 200
    assert counts["trends"] == 1  # repeat served from the result cache
    assert r2.get_json() == r1.get_json()
    # A horizon toggle is a different key: one more compute, then cached.
    r3 = client.get(url + "&horizon=this_week")
    assert r3.status_code == 200
    assert counts["trends"] == 2
    assert client.get(url + "&horizon=this_week").status_code == 200
    assert counts["trends"] == 2


# ── 1c. One shared value index across the three endpoints ───────────────────

def test_value_index_built_once_per_table_identity(monkeypatch):
    import routes.waiver_bp as waiver

    rows = [{"id": "a", "value": 1}, {"id": "b", "value": 2},
            {"name": "no-id row"}, "not-a-dict"]
    monkeypatch.setattr(waiver, "get_model_value_table_cached",
                        lambda *a, **k: rows)
    waiver._VALUE_INDEX_MEMO["key"] = None
    waiver._VALUE_INDEX_MEMO["index"] = None
    i1 = waiver._waiver_value_index()
    i2 = waiver._waiver_value_index()
    assert i1 is i2  # one build, shared by every caller
    assert set(i1) == {"a", "b"}  # same membership rule as the old dicts
    # A rebuilt table (new list object) invalidates the memo exactly once.
    rows2 = [dict(r) if isinstance(r, dict) else r for r in rows]
    monkeypatch.setattr(waiver, "get_model_value_table_cached",
                        lambda *a, **k: rows2)
    i3 = waiver._waiver_value_index()
    assert i3 is not i1
    assert i3 == i1
    assert waiver._waiver_value_index() is i3


# ── 2. Modal strip: one roster build serves every roster player ─────────────

_WEEK = 5
_PROJ = {"rb1": 20.0, "rb2": 14.0, "rb3": 11.0, "rb4": 8.0, "qb1": 22.0,
         "rbR": 17.0, "rbF": 9.0, "k1": 7.0}
_NAMES = {"rb1": "Runner One", "rb2": "Runner Two", "rb3": "Runner Three",
          "rb4": "Runner Four", "qb1": "Quarter Back", "rbR": "Rival Runner",
          "rbF": "Free Runner", "k1": "Kicker One"}
_TEAMS = {"rb1": "KC", "rb2": "BUF", "rb3": "MIA", "rb4": "DEN", "qb1": "KC",
          "rbR": "DAL", "rbF": "NYJ", "k1": "KC"}
_POS = {p: ("QB" if p == "qb1" else "K" if p == "k1" else "RB")
        for p in _NAMES}
_SCHED = [{"home": "KC", "away": "BUF"}, {"home": "MIA", "away": "DEN"},
          {"home": "DAL", "away": "NYJ"}]
_STRIP = ("/api/player-startsit-strip?platform=sleeper&league_id=L1"
          "&season=2026&player_id=")


def _strip_ctx():
    idx = {p: {"pos": _POS[p], "position": _POS[p], "team": _TEAMS[p],
               "full_name": _NAMES[p], "name": _NAMES[p], "injury_status": ""}
           for p in _NAMES}
    return {
        "viewer": {"viewer_roster_id": 1},
        "current_week": _WEEK,
        "rosters": [
            {"roster_id": 1, "owner_id": "me-user",
             "players": ["rb1", "rb2", "rb3", "rb4", "qb1"],
             "reserve": [], "taxi": [], "metadata": {"team_name": "My Team"}},
            {"roster_id": 2, "owner_id": "rival-user", "players": ["rbR"],
             "reserve": [], "taxi": [], "metadata": {"team_name": "Rival"}},
        ],
        "users": [{"user_id": "me-user", "display_name": "Kaedon"},
                  {"user_id": "rival-user", "display_name": "Rival Rick"}],
        "players_index": idx,
        "players": idx,
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                             "BN", "BN", "BN"],
        "scoring_settings": {},
        "raw_scoring_settings": {},
        "proj_by_week": {_WEEK: {"projections": dict(_PROJ)}},
    }


@pytest.fixture
def strip_env(offline_client, monkeypatch):
    import app as appmod
    import data_building.start_score_bundle as bundle_mod
    import data_building.weekly_metrics as wm_mod
    import utils.game_conditions as gc_mod
    import utils.utils as uu_mod

    appmod._SS_STRIP_CACHE.clear()
    appmod._SS_OPTIONS_CACHE.clear()
    ctx = _strip_ctx()  # ONE shared ctx object, as production serves
    monkeypatch.setattr(appmod, "_session_signed_in", lambda: True)
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: ctx)
    rows = [{"id": p, "name": _NAMES[p], "position": _POS[p],
             "team": _TEAMS[p], "value": 5000, "sf_value": 5000,
             "pos_rank_label": "RB1"} for p in _NAMES]
    monkeypatch.setattr(appmod, "get_model_value_table_cached",
                        lambda *a, **k: rows)
    monkeypatch.setattr(appmod, "_matchup_rank_table",
                        lambda *a, **k: ({}, 32, None, None))
    monkeypatch.setattr(appmod, "_fpts_against_effective", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_team_play_volume", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_season_weekstat_points",
                        lambda *a, **k: ((), ()))
    monkeypatch.setattr(appmod, "_load_season_weekly_points",
                        lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_season_snap_totals", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "build_projections_by_week", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "start_score_pos_anchors", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_ss_game_env", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_ss_qb_situation", lambda *a, **k: None)
    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda *a, **k: None)
    monkeypatch.setattr(bundle_mod, "load_start_score_bundles",
                        lambda *a, **k: {})
    monkeypatch.setattr(wm_mod, "get_usage_trends", lambda *a, **k: {})
    monkeypatch.setattr(gc_mod, "build_week_conditions", lambda *a, **k: {})
    monkeypatch.setattr(uu_mod, "load_week_projection", lambda *a, **k: {})
    monkeypatch.setattr(uu_mod, "load_week_sched", lambda *a, **k: list(_SCHED))
    builds = {"n": 0}
    _orig_build = appmod._build_start_sit_options

    def _counting(*a, **k):
        builds["n"] += 1
        return _orig_build(*a, **k)

    monkeypatch.setattr(appmod, "_build_start_sit_options", _counting)
    yield offline_client, builds
    appmod._SS_STRIP_CACHE.clear()
    appmod._SS_OPTIONS_CACHE.clear()


def test_strip_second_roster_player_distills_from_cached_build(strip_env):
    client, builds = strip_env
    d1 = client.get(_STRIP + "rb1").get_json()
    assert d1["state"] == "ok"
    assert builds["n"] == 1
    # A second roster player in the same league distills from the cached
    # roster options payload: no second full build.
    d2 = client.get(_STRIP + "rb2").get_json()
    assert d2["state"] == "ok"
    assert builds["n"] == 1
    # A guest (rival roster) player keeps the per-player guest build.
    dg = client.get(_STRIP + "rbR").get_json()
    assert dg["state"] == "ok"
    assert builds["n"] == 2
    # Parity: with both caches cold, rb2's strip is identical to the one
    # distilled from the shared build.
    import app as appmod
    appmod._SS_STRIP_CACHE.clear()
    appmod._SS_OPTIONS_CACHE.clear()
    d2_cold = client.get(_STRIP + "rb2").get_json()
    assert builds["n"] == 3
    assert d2_cold == d2


# ── 3a. Lab payload: shared tier serves a cold in-process cache ─────────────

class _FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value


_LAB_URL = "/api/lineup-lab?platform=sleeper&league_id=L1&season=2026&week=4"


def test_lab_payload_served_from_shared_tier_when_in_process_cold(
        offline_client, monkeypatch):
    import app as appmod
    import data_building.lineup_lab as lab_mod

    appmod._LAB_PAYLOAD_CACHE.clear()
    monkeypatch.setattr(appmod, "_session_signed_in", lambda: True)
    monkeypatch.setattr(
        appmod, "get_league_ctx_from_cache",
        lambda *a, **k: {"viewer": {"viewer_roster_id": 7},
                         "current_week": 4})
    calls = {"n": 0}

    def fake_build(**kwargs):
        calls["n"] += 1
        return {"week": kwargs["week"],
                "you": {"roster_id": kwargs["viewer_roster_id"]},
                "n_sims": 2000}

    monkeypatch.setattr(lab_mod, "build_lineup_lab_payload", fake_build)
    fake_redis = _FakeRedis()
    monkeypatch.setattr(appmod, "_lab_redis_client", lambda: fake_redis)
    try:
        r1 = offline_client.get(_LAB_URL)
        assert r1.status_code == 200
        assert calls["n"] == 1
        assert fake_redis.store  # the build was mirrored to the shared tier
        # Simulate a worker recycle: the in-process tier is wiped, the
        # shared tier still holds the payload inside its TTL window.
        appmod._LAB_PAYLOAD_CACHE.clear()
        r2 = offline_client.get(_LAB_URL)
        assert r2.status_code == 200
        assert calls["n"] == 1  # no rebuild
        assert r2.get_json() == r1.get_json()
    finally:
        appmod._LAB_PAYLOAD_CACHE.clear()


# ── Changelog announcement ──────────────────────────────────────────────────

def test_changelog_announces_call_consolidation():
    from dashboard_services.changelog import CHANGELOG

    top = CHANGELOG[5]
    assert top["date"] == "2026-10-27"
    assert top["tag"] == "fix"
    assert "Waivers and Start/Sit" in top["text"]
    assert "Lineup Lab" in top["text"]
    assert "—" not in top["text"]  # no em dashes in UI copy


# ── 3b. Lab prefetch: intent only, never on page load ───────────────────────

def test_lab_prefetch_fires_on_intent_not_on_load():
    from dashboard_services.pages.waivers_page import build_waivers_body

    body = build_waivers_body("sleeper", 2026, "123", {})
    # The Start/Sit success path wires intent listeners; it must NOT fire
    # the Lab fetch itself anymore.
    start = body.index("function wvFetchStartSit(")
    ss_body = body[start:body.index("\nfunction ", start + 1)]
    assert "wvPrefetchLab();" not in ss_body
    assert "wvWireLabIntent();" in ss_body
    assert ss_body.index("wvStartSitData = d;") < ss_body.index(
        "wvWireLabIntent();")
    # The intent wiring targets the Lab toggle and warms on first contact.
    start = body.index("function wvWireLabIntent(")
    intent_body = body[start:body.index("\nfunction ", start + 1)]
    assert "getElementById('wvSsModeLab')" in intent_body
    assert "pointerenter" in intent_body
    assert "focus" in intent_body
    assert "wvPrefetchLab" in intent_body
