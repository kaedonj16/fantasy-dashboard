"""Iteration 3 (Nov 19 Fleaflicker check) pins:

1. Next Steps: an unowned player who projects into the lineup (Joe
   Milton, dropped in the user's own Wednesday batch) breaks the
   all-set state when bench spots are open, and a starter trending
   out (Kincaid, DNP Wed + Thu) gets an action even with his game
   more than 48h away.
2. Waivers: just-dropped players enter the pool even when the ctx
   carries no build timestamp (the Fleaflicker case); drop
   suggestions respect forward coverage (Drew Lock, the Week 12
   starting QB, is never the suggested drop); the K/DEF drop pool
   only ever contains K/DEF players (the Jets D/ST add arrived with
   "DROP J.J. McCarthy" attached).
4. Trade gate: in Superflex, a package that leaves the partner
   without a startable QB2 never surfaces, even through the archetype
   engine's flagged-structural branch (the Caleb pitches: Kyren
   alone, Kyren+Kincaid, Watson alone, Kyren+Moore).
5. GET-side keeper legality: a designated keeper on the partner's
   roster (Mahomes at FLASHOVERS) can never be a suggestion target,
   at the shared gate and in the archetype engine.
6. Waiver list determinism: identical inputs produce an identical
   list, including at the shown-30 boundary, regardless of the
   input order the pool was built in (Mason Taylor vanished between
   loads).
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("pandas")

from dashboard_services import trade_partner_checks as tpc
from dashboard_services.trade_partner_checks import gate_package

ET = ZoneInfo("America/New_York")

SF_REQ = {"QB": 2, "RB": 2, "WR": 2, "TE": 1}
BARS = {"QB": 400.0, "RB": 300.0, "WR": 300.0, "TE": 180.0}


def _p(pid, pos, value):
    return {"player_id": pid, "id": pid, "position": pos,
            "value": float(value), "name": pid}


# --- 5 (gate level): receive-side keeper ban -------------------------

_PARTNER = [
    _p("caleb", "QB", 800), _p("maye", "QB", 850),
    _p("ward", "QB", 500), _p("mendoza", "QB", 480),
    _p("p-rb1", "RB", 600), _p("p-rb2", "RB", 400),
    _p("p-wr1", "WR", 700), _p("p-wr2", "WR", 500),
    _p("p-te1", "TE", 300),
]
_VIEWER = [
    _p("lock", "QB", 700), _p("mccarthy", "QB", 430),
    _p("kyren", "RB", 780), _p("v-rb2", "RB", 500),
    _p("watson", "WR", 560), _p("kincaid", "TE", 350),
]
_PROJ = {"maye": 23.0, "caleb": 20.6, "mccarthy": 17.9, "lock": 19.9}


def test_gate_rejects_designated_keeper_target():
    ok, _pkg, ev = gate_package(
        partner_players=_PARTNER, viewer_players=_VIEWER,
        receive=[_p("caleb", "QB", 800)],
        package=[_p("watson", "WR", 560), _p("mccarthy", "QB", 430)],
        requirements=SF_REQ, value_bars=BARS, projections=_PROJ,
        receive_locked_ids={"caleb"})
    assert ok is False
    assert ev is None  # legality verdict: no evaluation ever runs


def test_gate_allows_same_package_when_target_not_locked():
    ok, _pkg, ev = gate_package(
        partner_players=_PARTNER, viewer_players=_VIEWER,
        receive=[_p("caleb", "QB", 800)],
        package=[_p("watson", "WR", 560), _p("mccarthy", "QB", 430)],
        requirements=SF_REQ, value_bars=BARS, projections=_PROJ,
        receive_locked_ids={"maye"})
    assert ok is True
    assert ev is not None and ev["viable"] is True


# --- 6: deterministic waiver ordering ---------------------------------

def test_deterministic_candidate_order_ignores_input_order():
    from utils.waivers import deterministic_candidate_order

    scores = {}
    cands = []
    for i in range(35):
        pid = f"p{i:02d}"
        # Heavy ties, including a tie straddling the shown-30 boundary.
        scores[pid] = 50.0 if i >= 5 else 90.0
        cands.append({"player_id": pid, "name": f"Player {i:02d}",
                      "value": 100.0})
    fwd = deterministic_candidate_order(cands, lambda c: scores[c["player_id"]])
    rev = deterministic_candidate_order(
        list(reversed(cands)), lambda c: scores[c["player_id"]])
    assert [c["player_id"] for c in fwd] == [c["player_id"] for c in rev]
    assert len(fwd) == 35
    # The boundary is stable: positions 29/30 are the same players
    # on every load.
    assert fwd[29]["player_id"] == rev[29]["player_id"]
    assert fwd[30]["player_id"] == rev[30]["player_id"]


def test_deterministic_candidate_order_breaks_ties_by_value_name_id():
    from utils.waivers import deterministic_candidate_order

    cands = [
        {"player_id": "b", "name": "Beta", "value": 100.0},
        {"player_id": "a", "name": "Alpha", "value": 100.0},
        {"player_id": "c", "name": "Gamma", "value": 300.0},
    ]
    out = deterministic_candidate_order(cands, lambda c: 10.0)
    assert [c["player_id"] for c in out] == ["c", "a", "b"]


# --- 2 (coverage): forward starters are never the suggested drop -----

def test_forward_coverage_protects_future_week_starters():
    from utils.waivers import forward_coverage_protected_ids

    positions = {"lock": "QB", "mccarthy": "QB", "sanders": "QB",
                 "cmc": "RB", "dobbins": "RB"}
    # Week 11: Lock on bye (0). Week 12: Lock + McCarthy are the two
    # startable QBs in a Superflex league; Sanders never projects.
    weekly = [
        {"mccarthy": 18.0, "lock": 0.0, "sanders": 0.0,
         "cmc": 15.0, "dobbins": 6.0},
        {"mccarthy": 18.0, "lock": 19.0, "sanders": 0.0,
         "cmc": 15.0, "dobbins": 6.0},
    ]
    rp = ["QB", "RB", "RB", "WR", "WR", "TE", "SUPER_FLEX", "FLEX", "BN"]
    protected = forward_coverage_protected_ids(
        list(positions), positions, weekly, rp)
    assert "lock" in protected       # Week 12 starting QB
    assert "mccarthy" in protected
    assert "sanders" not in protected  # never projects: a legal drop


# --- 2 (freshness): drops land in the pool without a ctx timestamp ---

def test_delta_fallback_baseline_without_ctx_timestamp(monkeypatch):
    import utils.waiver_freshness as wf

    wf._DELTA_CACHE.clear()
    now = datetime.now(timezone.utc)
    dropped_at = now - timedelta(hours=2)

    def _ms(dt):
        return int(dt.timestamp() * 1000)

    def fake_fetch(platform, league_id, season, weeks):
        tx = {"status": "complete", "created": _ms(dropped_at),
              "drops": {"milton": 1}}
        return {weeks[0]: [tx]}

    ctx = {"current_week": 11}  # no _cache_synced_at: the Fleaflicker shape
    delta = wf.get_transactions_delta(
        "fleaflicker", "92916", 2026, ctx, fetch_fn=fake_fetch)
    assert set(delta["drops"]) == {"milton"}
    rostered = wf.apply_delta_to_rostered({"milton", "keep"}, delta)
    assert rostered == {"keep"}
    wf._DELTA_CACHE.clear()


# --- 3: true roster capacity + deterministic K/DEF drop pool ----------

def test_effective_roster_positions_restores_bench_capacity():
    from utils.waivers import effective_roster_positions

    starters_only = ["QB", "RB", "RB", "WR", "WR", "TE",
                     "SUPER_FLEX", "FLEX", "K", "DEF"]
    # Fleaflicker shape: roster_positions are starters only; the
    # bench (8 slots, 2 empty) rides on the roster as bench_slots.
    roster = {"bench_slots": ["p1", "p2", "p3", "p4", "p5", "p6", "0", "0"]}
    eff = effective_roster_positions(roster, starters_only)
    assert len(eff) == 18
    assert eff.count("BN") == 8
    # Sleeper shape (bench already in the list) passes through.
    sleeper = starters_only + ["BN"] * 8
    assert effective_roster_positions({"bench_slots": []}, sleeper) == sleeper
    # No bench information anywhere: unchanged.
    assert effective_roster_positions({}, starters_only) == starters_only


def test_kdef_drop_pool_is_deterministic_at_zero_ties():
    from routes.waiver_bp import _kdef_drop_pool

    viewer_active = ["mccarthy", "cmc", "nyj_def", "little"]
    viewer_pos = {"mccarthy": "QB", "cmc": "RB",
                  "nyj_def": "DEF", "little": "K"}
    # Forward projections exist only for K/DEF; skill players tie at
    # 0.0. The pool order must not depend on input iteration order.
    fwd = {"nyj_def": 8.0, "little": 11.0}
    a = _kdef_drop_pool(viewer_active, viewer_pos, fwd)
    b = _kdef_drop_pool(list(reversed(viewer_active)), viewer_pos, fwd)
    assert [d["player_id"] for d in a] == [d["player_id"] for d in b]
    assert a[0]["ppg"] == 0.0 and a[-1]["player_id"] == "little"


# --- 1: Next Steps attention (FA upgrade + trending out) ---------------

def _ts(day, hour):
    return datetime(2026, 11, day, hour, tzinfo=ET).timestamp()


WED_1200 = _ts(11, 12)          # Wednesday Nov 11, noon ET
SUN_1300 = _ts(15, 13)          # Sunday Nov 15, 1 PM ET

NS_PLAYERS = {
    "mccarthy": ("J.J. McCarthy", "MIN", "QB", ""),
    "sanders": ("Shedeur Sanders", "CLE", "QB", ""),
    "kincaid": ("Dalton Kincaid", "BUF", "TE", "Questionable"),
    "cmc": ("Christian McCaffrey", "SF", "RB", ""),
    "jsn": ("Jaxon Smith-Njigba", "SEA", "WR", ""),
    "sadiq": ("Theo Sadiq", "BUF", "TE", ""),
    "milton": ("Joe Milton", "PHI", "QB", ""),
}
NS_GAMES = [
    {"home": "MIN", "away": "WAS", "gameTime_epoch": SUN_1300},
    {"home": "BUF", "away": "MIA", "gameTime_epoch": SUN_1300},
    {"home": "SF", "away": "ARI", "gameTime_epoch": SUN_1300},
    {"home": "SEA", "away": "LV", "gameTime_epoch": SUN_1300},
    {"home": "CLE", "away": "NYJ", "gameTime_epoch": SUN_1300},
    {"home": "PHI", "away": "PIT", "gameTime_epoch": SUN_1300},
]
NS_STARTERS = ["mccarthy", "sanders", "kincaid", "cmc", "jsn"]
NS_BENCH = ["sadiq"]


def _ns_ctx(projections):
    positions = ["QB", "QB", "TE", "RB", "WR"] + ["BN"] * 3
    return {
        "platform": "fleaflicker",
        "league_id": "92916",
        "current_season": 2026,
        "current_week": 11,
        "rosters": [{
            "roster_id": 1,
            "starters": list(NS_STARTERS),
            "players": list(NS_STARTERS) + list(NS_BENCH),
            "reserve": [], "taxi": [],
        }],
        "players_map": {
            pid: {"name": name, "team": team, "pos": pos}
            for pid, (name, team, pos, _inj) in NS_PLAYERS.items()
        },
        "roster_positions": positions,
        "proj_by_week": {11: {"projections": dict(projections)}},
        "league": {"settings": {}},
    }


def _ns_attention(monkeypatch, projections, extra_full=None):
    import app as app_module

    full = {
        pid: {"full_name": name, "team": team, "position": pos,
              "injury_status": inj}
        for pid, (name, team, pos, inj) in NS_PLAYERS.items()
    }
    for pid, extra in (extra_full or {}).items():
        full[pid].update(extra)
    monkeypatch.setattr(app_module, "get_players_global", lambda: full)
    monkeypatch.setattr(app_module, "load_week_schedule",
                        lambda season, week: list(NS_GAMES))
    with app_module.app.test_request_context("/"):
        return app_module._next_steps_attention_actions(
            _ns_ctx(projections), 1, now_ts=WED_1200)


def test_unowned_qb_projecting_into_lineup_breaks_all_set(monkeypatch):
    actions = _ns_attention(monkeypatch, {
        "mccarthy": 17.9, "sanders": 8.0, "milton": 15.6,
        "kincaid": 6.4, "cmc": 15.3, "jsn": 16.7, "sadiq": 2.9})
    adds = [a for a in actions if a["action"] == "Add Joe Milton"]
    assert adds, f"expected a Milton add action, got {[a['action'] for a in actions]}"
    assert "Shedeur Sanders" in adds[0]["why"]


def test_trending_out_starter_breaks_all_set_beyond_48h(monkeypatch):
    # Kincaid: Questionable with DNP on consecutive practice days;
    # his Sunday game is ~97h out on Wednesday, far past the 48h
    # questionable window.
    actions = _ns_attention(
        monkeypatch,
        {"mccarthy": 17.9, "sanders": 12.0, "kincaid": 6.4,
         "cmc": 15.3, "jsn": 16.7, "sadiq": 2.9},
        extra_full={"kincaid": {"dnp_streak": 2}})
    trend = [a for a in actions if a["action"] == "Dalton Kincaid is trending out"]
    assert trend, f"expected a trending-out action, got {[a['action'] for a in actions]}"
    assert "Theo Sadiq is the plug" in trend[0]["why"]


def test_questionable_without_dnp_streak_stays_quiet_beyond_48h(monkeypatch):
    actions = _ns_attention(
        monkeypatch,
        {"mccarthy": 17.9, "sanders": 12.0, "kincaid": 6.4,
         "cmc": 15.3, "jsn": 16.7, "sadiq": 2.9})
    assert not any("trending out" in a["action"] for a in actions)
    assert not any(a["action"].startswith("Add ") for a in actions)


# --- 4 + 5: archetype engine (Superflex QB2 + GET-side keeper) --------

@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2026", "week": 0, "leg": 0,
                    "season_type": "off", "display_week": 1,
                    "season_start_date": "2026-09-10"}
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)


@pytest.fixture(autouse=True)
def _clear_result_cache():
    from dashboard_services import archetype_engine as ae
    ae._RESULT_CACHE.clear()
    yield
    ae._RESULT_CACHE.clear()


def _ae_p(pid, name, pos, val, age, team="FA"):
    return {"id": pid, "name": name, "position": pos, "team": team,
            "age": age, "value": val, "sf_value": val,
            "redraft_value_1qb": val * 0.8, "redraft_value_sf": val * 0.8,
            "pos_rank_label": f"{pos}1", "rank_change_7d": 0}


def _caleb_no_spare_ctx():
    """Goodfellas shape, but the viewer's QB room has exactly ONE
    startable QB this week: Lock and Cousins carry no projection
    (bye week), so neither is a spare the gate can send back. The
    old flagged-structural branch resurfaced Caleb packages here;
    in Superflex they must drop instead."""
    table = [
        _ae_p("maye", "Drake Maye", "QB", 867, 24, "NE"),
        _ae_p("caleb", "Caleb Williams", "QB", 805, 25, "CHI"),
        _ae_p("ward", "Cam Ward", "QB", 502, 24, "TEN"),
        _ae_p("mendoza", "Fernando Mendoza", "QB", 419, 23, "LV"),
        _ae_p("g_rb1", "G RB1", "RB", 600, 26, "NE"),
        _ae_p("g_rb2", "G RB2", "RB", 450, 27, "NE"),
        _ae_p("g_wr1", "G WR1", "WR", 650, 26, "NE"),
        _ae_p("g_wr2", "G WR2", "WR", 500, 28, "NE"),
        _ae_p("g_te1", "G TE1", "TE", 320, 27, "NE"),
        _ae_p("lock", "Drew Lock", "QB", 720, 29, "SEA"),
        _ae_p("mccarthy", "J.J. McCarthy", "QB", 613, 23, "MIN"),
        _ae_p("cousins", "Kirk Cousins", "QB", 640, 38, "ATL"),
        _ae_p("kyren", "Kyren Williams", "RB", 780, 26, "LAR"),
        _ae_p("v_rb2", "Viewer RB2", "RB", 500, 27, "PHI"),
        _ae_p("watson", "Christian Watson", "WR", 615, 27, "GB"),
        _ae_p("jsn", "Jaxon Smith-Njigba", "WR", 900, 24, "SEA"),
        _ae_p("v_wr3", "Viewer WR3", "WR", 500, 26, "PHI"),
        _ae_p("kincaid", "Dalton Kincaid", "TE", 350, 27, "BUF"),
        _ae_p("x_rb", "X RB", "RB", 700, 25, "DET"),
        _ae_p("x_wr", "X WR", "WR", 700, 25, "DET"),
    ]
    rosters = [
        {"roster_id": 1, "players": ["lock", "mccarthy", "cousins", "kyren",
                                     "v_rb2", "watson", "jsn", "v_wr3",
                                     "kincaid"],
         "keepers": ["jsn"]},
        {"roster_id": 2, "players": ["maye", "caleb", "ward", "mendoza",
                                     "g_rb1", "g_rb2", "g_wr1", "g_wr2",
                                     "g_te1"],
         "keepers": ["maye"]},
        {"roster_id": 3, "players": ["x_rb", "x_wr"], "keepers": []},
    ]
    return {
        "rosters": rosters,
        "roster_map": {1: "Viewer", 2: "Goodfellas", 3: "Team Three"},
        "standings_map": {1: 6, 2: 1, 3: 2},
        "model_value_table": table,
        "picks_by_roster": {},
        "settings": {"playoff_week_start": 15},
        "league_settings": {"max_keepers": 5},
        # Only Maye, Caleb and McCarthy project: Ward/Mendoza fail
        # closed, and Lock/Cousins are not startable this week, so
        # the viewer has no QB to send back.
        "projections": {"maye": 23.0, "caleb": 20.6, "mccarthy": 17.9},
    }


@pytest.mark.parametrize("archetype", ["contending", "consolidate"])
def test_superflex_caleb_never_surfaces_without_qb_back(archetype):
    from dashboard_services import archetype_engine as ae

    out = ae.get_archetype_suggestions(
        archetype=archetype, platform="sleeper", league_id="iter3",
        season=2026, viewer_roster_id="1", league_type="superflex",
        league_size=10, ctx=_caleb_no_spare_ctx())
    caleb_rows = [s for s in (out.get("suggestions") or [])
                  if str(s.get("player_id") or "") == "caleb"]
    assert caleb_rows == [], (
        f"Caleb surfaced without a QB returning in Superflex: {caleb_rows}")


def test_partner_designated_keeper_is_never_a_target():
    from dashboard_services import archetype_engine as ae

    ctx = _caleb_no_spare_ctx()
    # Maye becomes the (keeper-locked) target shape: he is already a
    # designated keeper at Goodfellas, so no suggestion may target
    # him, in any archetype.
    for archetype in ("contending", "consolidate"):
        out = ae.get_archetype_suggestions(
            archetype=archetype, platform="sleeper", league_id="iter3",
            season=2026, viewer_roster_id="1", league_type="superflex",
            league_size=10, ctx=ctx)
        ids = {str(s.get("player_id") or "")
               for s in (out.get("suggestions") or [])}
        assert "maye" not in ids, (
            f"{archetype} pitched GET on designated keeper Maye")
