"""PR 3 of Site Knowledge Wave 1 (spec §7): Waiver & Start/Sit truth.

- Start/Sit identity fallback chain (spec §4): a DEF pid absent from the
  model value table and the players index still resolves -- name and
  position from the full players payload, or a synthesized DEF identity
  from the teams index -- instead of dropping the row ("No roster data")
  or rendering the literal "Player NO" for the Saints.
- Waiver freshness (spec §5): the transactions delta nets drops/adds
  newer than the ctx build into the derived pool; a recently dropped
  player whose current-week projection reaches the viewer's worst
  starter bypasses the value floor; rows carry dropped_at /
  waiver_process for the "claim by" chip.
- The Next Steps shared selector path applies the same delta.
"""
import pytest

pd = pytest.importorskip("pandas")

import app as app_module  # noqa: E402


# ── Start/Sit identity chain (spec §4.2) ───────────────────────────────────

TEAMS_INDEX = {"NO": {"teamId": "22", "byeWeek": 12},
               "KC": {"teamId": "16", "byeWeek": 10}}
GROUPS = ["QB", "RB", "WR", "TE", "K", "DEF"]


def test_def_identity_from_full_players_payload():
    full = {"NO": {"full_name": "New Orleans Saints", "position": "DEF",
                   "team": "NO"}}
    pos, name, team = app_module._ss_resolve_identity(
        "NO", {}, {}, full["NO"], TEAMS_INDEX, GROUPS)
    assert pos == "DEF"
    assert name == "New Orleans Saints"
    assert name != "Player NO"
    assert team == "NO"


def test_def_identity_synthesized_from_teams_index():
    # No value row, no index meta, no full-payload entry at all: the pid
    # is an NFL team code in a league with a DEF group, so synthesize.
    pos, name, team = app_module._ss_resolve_identity(
        "NO", {}, {}, {}, TEAMS_INDEX, GROUPS)
    assert pos == "DEF"
    assert name == "New Orleans Saints"
    assert team == "NO"


def test_def_not_synthesized_without_def_group():
    pos, _name, _team = app_module._ss_resolve_identity(
        "NO", {}, {}, {}, TEAMS_INDEX, ["QB", "RB", "WR", "TE"])
    assert pos == ""


def test_idp_player_never_relabelled_def():
    # An IDP defensive back whose pid happens to be a team code must not
    # be synthesized into a team defense: a source DID carry a position.
    full = {"NO": {"full_name": "Some Safety", "position": "DB",
                   "team": "NO"}}
    pos, name, _team = app_module._ss_resolve_identity(
        "NO", {}, {}, full["NO"], TEAMS_INDEX, GROUPS)
    assert pos == ""
    assert name == "Some Safety"


def test_partial_meta_def_gets_real_name():
    # The "Player NO" shape: meta carries the position but no name.
    pos, name, _team = app_module._ss_resolve_identity(
        "NO", {}, {"pos": "DEF"}, {}, TEAMS_INDEX, GROUPS)
    assert pos == "DEF"
    assert name == "New Orleans Saints"


def test_kicker_identity_unchanged():
    row = {"name": "Harrison Butker", "position": "K", "team": "KC"}
    pos, name, team = app_module._ss_resolve_identity(
        "9901", row, {}, {}, TEAMS_INDEX, GROUPS)
    assert (pos, name, team) == ("K", "Harrison Butker", "KC")


def test_value_row_still_wins_over_full_payload():
    row = {"name": "Value Name", "position": "WR", "team": "BUF"}
    full = {"full_name": "Payload Name", "position": "WR", "team": "BUF"}
    pos, name, _team = app_module._ss_resolve_identity(
        "1234", row, {}, full, TEAMS_INDEX, GROUPS)
    assert name == "Value Name"


def test_start_sit_row_builder_uses_fallback_chain():
    from pathlib import Path
    src = Path(app_module.__file__).read_text(encoding="utf-8")
    start = src.index("def _ss_build_row")
    body = src[start:start + 1200]
    assert "_ss_resolve_identity" in body
    # DEF rows are forced down the live path, never the bundle path.
    assert "_ss_use_bundles" in src[src.index("DEF rows always take the live path") - 200:
                               src.index("DEF rows always take the live path") + 200]


# ── Transactions delta (spec §5.2A) ────────────────────────────────────────

from datetime import datetime, timedelta, timezone  # noqa: E402

from utils.waiver_freshness import (  # noqa: E402
    apply_delta_to_rostered,
    build_delta,
    dropped_within,
    get_transactions_delta,
    waiver_process_info,
)

CTX_TS = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def _ms(dt):
    return int(dt.timestamp() * 1000)


def _tx(pid_add=None, pid_drop=None, when=None, status="complete"):
    tx = {"status": status, "created": _ms(when or CTX_TS)}
    if pid_add:
        tx["adds"] = {pid_add: 2}
    if pid_drop:
        tx["drops"] = {pid_drop: 2}
    return tx


def test_delta_keeps_only_transactions_newer_than_ctx_build():
    from datetime import timedelta
    txs = [
        _tx(pid_drop="old", when=CTX_TS - timedelta(hours=1)),
        _tx(pid_drop="bris", when=CTX_TS + timedelta(hours=3)),
        _tx(pid_add="lock", when=CTX_TS + timedelta(hours=2)),
    ]
    delta = build_delta(txs, CTX_TS)
    assert set(delta["drops"]) == {"bris"}
    assert delta["adds"] == {"lock"}
    assert delta["hash"]


def test_delta_net_state_wins_in_timestamp_order():
    from datetime import timedelta
    txs = [
        _tx(pid_add="p1", when=CTX_TS + timedelta(hours=3)),
        _tx(pid_drop="p1", when=CTX_TS + timedelta(hours=1)),
        _tx(pid_drop="p2", when=CTX_TS + timedelta(hours=1)),
        _tx(pid_add="p2", when=CTX_TS + timedelta(hours=2)),
    ]
    delta = build_delta(txs, CTX_TS)
    # Both were dropped at +1h and (re-)added later: net state is
    # rostered for both, and neither remains in the drops map.
    assert delta["adds"] == {"p1", "p2"}
    assert delta["drops"] == {}


def test_delta_ignores_failed_transactions_and_missing_baseline():
    from datetime import timedelta
    failed = _tx(pid_drop="p9", when=CTX_TS + timedelta(hours=1),
                 status="failed")
    assert build_delta([failed], CTX_TS)["drops"] == {}
    good = _tx(pid_drop="p9", when=CTX_TS + timedelta(hours=1))
    assert build_delta([good], None)["drops"] == {}


def test_apply_delta_and_dropped_within():
    delta = {"adds": {"a1"},
             "drops": {"d1": "2026-11-04T15:00:00+00:00",
                       "d2": "2026-11-01T15:00:00+00:00"},
             "hash": "x"}
    rostered = apply_delta_to_rostered({"d1", "d2", "keep"}, delta)
    assert rostered == {"keep", "a1"}
    recent = dropped_within(delta, hours=48.0,
                            now=datetime(2026, 11, 4, 16, 0,
                                         tzinfo=timezone.utc))
    assert set(recent) == {"d1"}


def test_get_transactions_delta_caches_and_fails_soft(monkeypatch):
    import utils.waiver_freshness as wf
    wf._DELTA_CACHE.clear()
    calls = []

    def fake_fetch(platform, league_id, season, weeks):
        calls.append(weeks)
        from datetime import timedelta
        return {9: [_tx(pid_drop="bris",
                        when=CTX_TS + timedelta(hours=2))]}

    ctx = {"_cache_synced_at": CTX_TS.isoformat(), "current_week": 9}
    d1 = get_transactions_delta("fleaflicker", "92916", 2026, ctx,
                                fetch_fn=fake_fetch)
    d2 = get_transactions_delta("fleaflicker", "92916", 2026, ctx,
                                fetch_fn=fake_fetch)
    assert set(d1["drops"]) == {"bris"} and d1 == d2
    assert len(calls) == 1  # second call served by the 120s delta cache

    def boom(platform, league_id, season, weeks):
        raise RuntimeError("provider down")

    wf._DELTA_CACHE.clear()
    assert get_transactions_delta("fleaflicker", "92916", 2026, ctx,
                                  fetch_fn=boom) == {
        "adds": set(), "drops": {}, "hash": ""}


# ── Waiver process / "claim by" (spec §5.2C) ───────────────────────────────

def test_waiver_process_next_time_rolling_daily():
    settings = {"waiver_type": "rolling", "waiver_process_time": "03:00",
                "waiver_process_days": "daily",
                "waiver_process_tz": "America/New_York"}
    # 2026-11-04 12:00 UTC = 7:00 AM ET: today's 3 AM already passed.
    info = waiver_process_info(settings, now=CTX_TS)
    assert info["type"] == "rolling"
    assert info["next_process_at"].startswith("2026-11-05T03:00")
    # 2026-11-04 06:30 UTC = 1:30 AM ET: next process is today 3 AM ET.
    early = datetime(2026, 11, 4, 6, 30, tzinfo=timezone.utc)
    info2 = waiver_process_info(settings, now=early)
    assert info2["next_process_at"].startswith("2026-11-04T03:00")


def test_waiver_process_faab_and_missing_keys():
    faab = waiver_process_info(
        {"waiver_type": "faab", "waiver_process_time": "3:00 AM"},
        now=CTX_TS)
    assert faab["type"] == "faab" and faab["next_process_at"]
    # No process time carried: never guessed.
    assert waiver_process_info({"waiver_type": "rolling"}, now=CTX_TS) is None
    assert waiver_process_info({}, now=CTX_TS) is None
    assert waiver_process_info(None, now=CTX_TS) is None


# ── /api/waiver-candidates integration (spec §5.2A/B/C) ────────────────────

# The candidates endpoint judges "dropped within 48h" against the real
# wall clock (dropped_within's default now), so this must stay a recent
# timestamp: the fixed 2026-11-04 date it replaced aged out of the 48h
# window on 2026-11-06 and the drop-path tests started failing on time
# alone, on every branch.
DROP_ISO = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()

_WV_TEAMS = {"q1": "BUF", "rb1": "DET", "rb2": "SF", "wr1": "CIN",
             "wr2": "MIA", "te1": "KC", "b1": "DAL", "dropqb": "ARI",
             "addedrb": "HOU", "lowrb": "ATL", "freerb": "GB",
             "o1": "GB", "o2": "MIN"}
_WV_POS = {"q1": "QB", "rb1": "RB", "rb2": "RB", "wr1": "WR", "wr2": "WR",
           "te1": "TE", "b1": "RB", "dropqb": "QB", "addedrb": "RB",
           "lowrb": "RB", "freerb": "RB", "o1": "RB", "o2": "WR"}
_WV_VAL = {"q1": 5000, "rb1": 4000, "rb2": 3000, "wr1": 4000, "wr2": 3000,
           "te1": 3000, "b1": 500, "dropqb": 10, "addedrb": 4000,
           "lowrb": 10, "freerb": 2000, "o1": 3000, "o2": 3000}
_WV_PROJ = {"q1": 20.0, "rb1": 14.0, "rb2": 10.0, "wr1": 13.0, "wr2": 10.0,
            "te1": 9.0, "b1": 4.0, "dropqb": 22.0, "addedrb": 15.0,
            "lowrb": 3.0, "freerb": 12.0, "o1": 8.0, "o2": 8.0}
_WV_ROSTER1 = ["q1", "rb1", "rb2", "wr1", "wr2", "te1", "b1"]
_WV_ROSTER2 = ["o1", "o2", "dropqb", "lowrb"]


def _wv_client(monkeypatch):
    from flask import Flask
    import routes.waiver_bp as waiver

    waiver._WAIVER_CANDIDATES_CACHE.clear()
    rows = [{"id": pid, "name": f"Name {pid}", "position": _WV_POS[pid],
             "team": _WV_TEAMS[pid], "value": _WV_VAL[pid], "age": 26,
             "pos_rank_label": "", "rank_change_7d": 0}
            for pid in _WV_TEAMS]
    pidx = {pid: {"name": f"Name {pid}", "pos": _WV_POS[pid],
                  "team": _WV_TEAMS[pid]} for pid in _WV_TEAMS}
    ctx = {
        "current_week": 9, "current_season": 2026, "season": 2026,
        "_cache_synced_at": CTX_TS.isoformat(),
        "league": {"settings": {}},
        "league_settings": {"waiver_type": "rolling",
                            "waiver_process_time": "03:00",
                            "waiver_process_days": "daily",
                            "waiver_process_tz": "America/New_York"},
        "scoring_settings": {}, "raw_scoring_settings": {},
        "players_index": pidx,
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                             "BN", "BN", "BN"],
        "rosters": [
            {"roster_id": 1, "players": list(_WV_ROSTER1),
             "reserve": [], "taxi": [], "settings": {}},
            {"roster_id": 2, "players": list(_WV_ROSTER2),
             "reserve": [], "taxi": [], "settings": {}},
        ],
        "users": [],
    }
    app = Flask(__name__)
    app.register_blueprint(waiver.waiver_bp)
    monkeypatch.setattr(waiver, "get_nfl_state",
                        lambda: {"season": 2026, "week": 9,
                                 "season_type": "regular"})
    monkeypatch.setattr(waiver, "get_league_ctx_from_cache",
                        lambda *a, **k: ctx)
    monkeypatch.setattr(waiver, "get_model_value_table_cached",
                        lambda *a, **k: rows)
    monkeypatch.setattr(waiver, "get_players_global", lambda *a, **k: {})
    monkeypatch.setattr(
        waiver, "get_viewer_session_for_league",
        lambda users, rosters: {"viewer_roster_id": "1"})
    monkeypatch.setattr(waiver, "_sleeper_trending_adds",
                        lambda limit=50: [])
    monkeypatch.setattr(waiver, "load_do_not_drop",
                        lambda *a, **k: set())
    import utils.waiver_freshness as wf

    # The drop fixture sits at DROP_ISO (2026-11-04T15:00Z) and the route
    # admits it through dropped_within's 48h window, which reads the real
    # clock. Pin the freshness clock to an hour after the drop so the
    # test can never age out of the window.
    class _PinnedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 11, 4, 16, 0, tzinfo=timezone.utc)
            return base.astimezone(tz) if tz else base

    monkeypatch.setattr(wf, "datetime", _PinnedDateTime)
    monkeypatch.setattr(wf, "get_transactions_delta",
                        lambda *a, **k: {
                            "adds": {"addedrb"},
                            "drops": {"dropqb": DROP_ISO, "lowrb": DROP_ISO},
                            "hash": "delta-test-hash"})
    monkeypatch.setattr("data_building.weekly_metrics.get_usage_trends",
                        lambda season: {})
    monkeypatch.setattr(
        "dashboard_services.waiver_discoveries.get_week_discoveries",
        lambda season, week: [])
    monkeypatch.setattr("utils.utils.load_usage_table", lambda: [])
    monkeypatch.setattr("utils.utils.load_week_projection",
                        lambda season, week: dict(_WV_PROJ))
    teams = sorted(set(_WV_TEAMS.values()))
    sched = [{"away": teams[i], "home": teams[i + 1]}
             for i in range(0, len(teams) - 1, 2)]
    monkeypatch.setattr("utils.utils.load_week_schedule",
                        lambda season, week: sched)
    return app.test_client()


def test_waiver_candidates_delta_drop_add_and_floor_bypass(monkeypatch):
    client = _wv_client(monkeypatch)
    resp = client.get("/api/waiver-candidates?platform=fleaflicker"
                      "&league_id=92916&season=2026&rid=1")
    assert resp.status_code == 200
    cands = {c["player_id"]: c for c in resp.json["candidates"]}
    # Dropped after the ctx build + starter-level projection: in the
    # pool despite a value (10) far below the 25.0 floor.
    assert "dropqb" in cands
    assert cands["dropqb"]["dropped_at"] == DROP_ISO
    assert cands["dropqb"]["waiver_process"]["type"] == "rolling"
    assert cands["dropqb"]["waiver_process"]["next_process_at"]
    # Added by another team after the build: out of the pool.
    assert "addedrb" not in cands
    # Dropped but with no starter case (3.0 proj vs a 10.0 worst RB
    # starter / 4.0 FLEX starter): the floor still holds.
    assert "lowrb" not in cands
    # Ordinary above-floor free agent unaffected, no freshness fields.
    assert "freerb" in cands
    assert cands["freerb"]["dropped_at"] is None
    assert cands["freerb"]["waiver_process"] is None


# ── Next Steps shared selector path (spec §5.2A) ───────────────────────────

_NS_TEAMS = {
    "q1": "BUF", "q2": "NYJ", "q3": "CLE",
    "rb1": "DET", "rb2": "SF", "rb3": "SEA", "rb4": "DAL",
    "wr1": "CIN", "wr2": "MIA", "wr3": "LAR", "wr4": "DEN", "wr5": "PHI",
    "te1": "KC", "te2": "BAL",
    "mayf": "TB", "wrb": "HOU", "wwr": "ATL",
    "o1": "GB", "o2": "MIN", "o3": "CHI",
}
_NS_POS = {
    "q1": "QB", "q2": "QB", "q3": "QB",
    "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB",
    "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR", "wr5": "WR",
    "te1": "TE", "te2": "TE",
    "mayf": "QB", "wrb": "RB", "wwr": "WR",
    "o1": "RB", "o2": "WR", "o3": "TE",
}
_NS_VAL = {
    "q1": 5200, "q2": 1800, "q3": 950,
    "rb1": 4600, "rb2": 2400, "rb3": 1200, "rb4": 700,
    "wr1": 4200, "wr2": 3600, "wr3": 2800, "wr4": 1500, "wr5": 650,
    "te1": 2600, "te2": 500,
    "mayf": 2600, "wrb": 2100, "wwr": 900,
    "o1": 3000, "o2": 3000, "o3": 3000,
}
_NS_PROJ = {
    "q1": 24.0, "q2": 19.0, "q3": 15.0,
    "rb1": 16.0, "rb2": 11.0, "rb3": 8.0, "rb4": 5.0,
    "wr1": 15.0, "wr2": 14.0, "wr3": 12.5, "wr4": 9.0, "wr5": 5.0,
    "te1": 10.0, "te2": 4.0,
    "mayf": 20.5, "wrb": 13.5, "wwr": 8.0,
    "o1": 7.0, "o2": 7.0, "o3": 7.0,
}
_NS_ROSTER = ["q1", "q2", "rb1", "rb2", "rb3", "rb4",
              "wr1", "wr2", "wr3", "wr4", "wr5", "te1", "te2"]


def _ns_fixture(monkeypatch, delta):
    pidx = {
        pid: {"position": _NS_POS[pid], "pos": _NS_POS[pid],
              "team": _NS_TEAMS[pid], "full_name": f"Name {pid}",
              "injury_status": "Out" if pid == "q2" else ""}
        for pid in _NS_TEAMS
    }
    rows = [{"id": pid, "name": f"Name {pid}", "position": _NS_POS[pid],
             "team": _NS_TEAMS[pid], "value": _NS_VAL[pid], "age": 25,
             "pos_rank_label": "", "rank_change_7d": 0}
            for pid in _NS_TEAMS]
    ctx = {
        "platform": "sleeper", "league_id": "L1", "season": 2026,
        "current_season": 2026, "current_week": 7,
        "league": {"type": 2}, "settings": {"type": 2},
        "scoring_settings": {}, "raw_scoring_settings": {},
        "players_index": pidx,
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "WR", "TE",
                             "FLEX", "BN", "BN", "BN", "BN", "BN", "BN"],
        "rosters": [
            {"roster_id": 1, "players": list(_NS_ROSTER)},
            # mayf sits on the other roster in the (stale) ctx: without
            # the delta he is owned and can never be a claim.
            {"roster_id": 2, "players": ["o1", "o2", "o3", "mayf"]},
        ],
        "viewer": {"viewer_roster_id": 1},
    }

    def fake_proj(season, week):
        out = dict(_NS_PROJ)
        if week == 7:
            out.pop("q1", None)  # starter's team on bye in Week 7
            out.pop("q2", None)  # backup hurt: no projection at all
        return out

    def fake_sched(season, week):
        teams = sorted(set(_NS_TEAMS.values()))
        if week == 7:
            teams = [t for t in teams if t != "BUF"]
        return [{"away": teams[i], "home": teams[i + 1]}
                for i in range(0, len(teams) - 1, 2)]

    monkeypatch.setattr("utils.utils.load_week_projection", fake_proj)
    monkeypatch.setattr("utils.utils.load_week_schedule", fake_sched)
    import utils.waiver_freshness as wf
    monkeypatch.setattr(wf, "get_transactions_delta",
                        lambda *a, **k: delta)
    return ctx, rows


def test_next_steps_delta_drop_makes_claim_possible(monkeypatch):
    # QB hole (starter on bye, backup Out) + mayf dropped after the ctx
    # build: the shared selector can now judge him and the claim lands.
    delta = {"adds": set(), "drops": {"mayf": DROP_ISO}, "hash": "h1"}
    ctx, rows = _ns_fixture(monkeypatch, delta)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    qb_claims = [a for a in actions if a["player_id"] == "mayf"]
    assert len(qb_claims) == 1
    assert qb_claims[0]["outcome"] in ("add", "add_drop")


def test_next_steps_delta_add_removes_claim(monkeypatch):
    # wrb is a genuine lineup gain in the base fixture, but another team
    # added him after the ctx build: no claim may surface.
    delta = {"adds": {"wrb"}, "drops": {"mayf": DROP_ISO}, "hash": "h2"}
    ctx, rows = _ns_fixture(monkeypatch, delta)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    assert not any(a["player_id"] == "wrb" for a in actions)


def test_next_steps_no_delta_no_claim_for_rostered_qb(monkeypatch):
    # Control: with an empty delta the stale ctx still shows mayf as
    # rostered, so the QB-hole claim cannot surface.
    delta = {"adds": set(), "drops": {}, "hash": ""}
    ctx, rows = _ns_fixture(monkeypatch, delta)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    assert not any(a["player_id"] == "mayf" for a in actions)
