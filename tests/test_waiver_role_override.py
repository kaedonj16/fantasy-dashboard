"""Wave 2 PR 2 (spec §2.3 / §5 PR 2): waiver starting-role override.

The 92916 fixture: a Superflex viewer whose QBs are McCarthy/Fagnano-grade.
Drew Lock and Jacoby Brissett are promoted free agents with starter
projections but backup-grade model values: both must be admitted past
the WEIGHTS.min_value floor and pass the positional gate on projection.
Spencer Rattler is the anti-test: a healthy backup with no promotion
signal must produce no claim and no Next Steps action in 1QB *and*
Superflex, even with a starter-level projection: the override requires
the role state; projection alone never fires it.

Role states are injected through the ctx (``ctx["_role_states"]``), the
fail-soft access path that works before Wave 2 PR 1's
``utils/role_state.py`` lands on this stack.
"""
import pytest

pd = pytest.importorskip("pandas")

import app as app_module  # noqa: E402

from utils.digest import select_waiver_add, waiver_claim_beats_roster  # noqa: E402
from utils.waiver_role import (  # noqa: E402
    build_starting_role_signals,
    passes_projection_gate,
    resolve_role_states,
)

WEEK = 9

# ── Pure gate / helper units ───────────────────────────────────────────────


def test_projection_gate_mode_of_waiver_claim_beats_roster():
    # Value says no (10 vs a 100 bar), projection says yes.
    assert waiver_claim_beats_roster(
        10.0, 100.0, 0.0, candidate_projection=20.0,
        pos_bar_projection=15.0, starting_role=True)
    # Projection below the worst rostered projection: no pass.
    assert not waiver_claim_beats_roster(
        10.0, 100.0, 0.0, candidate_projection=14.0,
        pos_bar_projection=15.0, starting_role=True)
    # A role-based starter hole passes regardless of the bar.
    assert waiver_claim_beats_roster(
        10.0, 100.0, 1.0, candidate_projection=5.0,
        pos_bar_projection=15.0, starting_role=True)
    # The override requires a projection to judge.
    assert not waiver_claim_beats_roster(
        10.0, 100.0, 0.0, starting_role=True)
    # Legacy value mode is untouched.
    assert waiver_claim_beats_roster(106.0, 100.0, 0.0)
    assert not waiver_claim_beats_roster(100.0, 100.0, 0.0)


def _signals(candidates, states, proj, roster_proj=None, sf=True):
    positions = {"mcc": "QB", "fag": "QB"}
    positions.update({pid: "QB" for pid in candidates})
    proj_all = {"mcc": 15.0, "fag": 13.37}
    proj_all.update(roster_proj or {})
    proj_all.update(proj)
    return build_starting_role_signals(
        candidate_pids=list(candidates),
        positions_by_pid=positions,
        proj_by_pid=proj_all,
        roster_pids=["mcc", "fag"],
        roster_positions=(["QB", "SUPER_FLEX", "RB", "WR", "TE", "BN"]
                          if sf else ["QB", "RB", "WR", "TE", "BN"]),
        role_states=states,
        teams_by_pid={pid: "SEA" for pid in candidates},
    )


def test_signal_admission_requires_role_and_projection():
    promoted = {"lock": {"state": "promoted",
                         "reason": "Starting for SEA this week"}}
    sig = _signals(["lock"], promoted, {"lock": 20.67})
    assert set(sig) == {"lock"}
    assert sig["lock"]["signal_kind"] == "starting_role"
    assert sig["lock"]["reason"] == "Starting for SEA this week"
    assert passes_projection_gate(sig["lock"])
    # Returning into a starting job also qualifies.
    returning = {"lock": {"state": "returning", "reason": "Back from IR"}}
    assert set(_signals(["lock"], returning, {"lock": 20.67})) == {"lock"}
    # No role state: projection alone never fires (the Rattler case).
    assert _signals(["lock"], {}, {"lock": 20.67}) == {}
    assert _signals(["lock"], {"lock": {"state": "unknown"}},
                    {"lock": 20.67}) == {}
    # Role state but no projection: no admission.
    assert _signals(["lock"], promoted, {}) == {}
    # Role state but projection below the worst starter (13.37 in SF,
    # where both QBs start): no admission.
    assert _signals(["lock"], promoted, {"lock": 12.0}) == {}


def test_resolve_role_states_ctx_first_and_fail_soft():
    ctx = {"_role_states": {"lock": {"state": "promoted", "reason": "R"}}}
    assert resolve_role_states(ctx)["lock"]["state"] == "promoted"
    ctx2 = {"role_states": {"lock": {"state": "promoted"}}}
    assert resolve_role_states(ctx2)["lock"]["state"] == "promoted"
    assert isinstance(resolve_role_states({}), dict)
    assert isinstance(resolve_role_states(None), dict)


# ── 92916-shaped Next Steps fixture (Superflex) ────────────────────────────

_TEAMS = {
    "mcc": "MIN", "fag": "UConn", "rb1": "DET", "rb2": "SF", "rb3": "SEA",
    "wr1": "CIN", "wr2": "MIA", "wr3": "LAR", "wr4": "DEN", "te1": "KC",
    "te2": "BAL", "lock": "SEA", "bris": "BAL", "rat": "NO",
    "o1": "GB", "o2": "CHI", "o3": "NYJ",
}
# "UConn" is not an NFL team; keep every team in the schedule builder by
# using a real code for the second QB instead.
_TEAMS["fag"] = "NYG"
_NAMES = {
    "mcc": "J.J. McCarthy", "fag": "Joe Fagnano",
    "rb1": "Back One", "rb2": "Back Two", "rb3": "Back Three",
    "wr1": "Wide One", "wr2": "Wide Two", "wr3": "Wide Three",
    "wr4": "Wide Four", "te1": "Tight One", "te2": "Tight Two",
    "lock": "Drew Lock", "bris": "Jacoby Brissett",
    "rat": "Spencer Rattler",
    "o1": "Opp One", "o2": "Opp Two", "o3": "Opp Three",
}
_POS = {
    "mcc": "QB", "fag": "QB", "rb1": "RB", "rb2": "RB", "rb3": "RB",
    "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR", "te1": "TE",
    "te2": "TE", "lock": "QB", "bris": "QB", "rat": "QB",
    "o1": "RB", "o2": "WR", "o3": "TE",
}
_VALUES = {
    "mcc": 900, "fag": 700, "rb1": 4600, "rb2": 2400, "rb3": 1200,
    "wr1": 4200, "wr2": 3600, "wr3": 2800, "wr4": 1500, "te1": 2600,
    "te2": 500,
    # The candidates carry literal below-floor values (the spec case:
    # backup-grade model value, starter's projection).
    "lock": 15, "bris": 12, "rat": 10,
    "o1": 3000, "o2": 3000, "o3": 3000,
}
_PROJ = {
    "mcc": 15.0, "fag": 13.37, "rb1": 16.0, "rb2": 11.0, "rb3": 8.0,
    "wr1": 15.0, "wr2": 14.0, "wr3": 12.5, "wr4": 9.0, "te1": 10.0,
    "te2": 4.0, "lock": 20.67, "bris": 19.5,
    # Rattler projects like a starter in the SF anti-test: projection
    # alone must still not fire the override.
    "rat": 18.0,
    "o1": 7.0, "o2": 7.0, "o3": 7.0,
}
_ROSTER_SF = ["mcc", "fag", "rb1", "rb2", "rb3",
              "wr1", "wr2", "wr3", "wr4", "te1", "te2"]
_POSITIONS_SF = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                 "SUPER_FLEX", "BN", "BN"]
_ROLE_STATES = {
    "lock": {"state": "promoted", "reason": "Starting for SEA this week",
             "signal": "provider"},
    "bris": {"state": "promoted", "reason": "Starting for BAL this week",
             "signal": "provider"},
}


def _schedule():
    teams = sorted(set(_TEAMS.values()))
    return [{"away": teams[i], "home": teams[i + 1]}
            for i in range(0, len(teams) - 1, 2)]


def _ns_fixture(monkeypatch, *, superflex=True, role_states=None,
                roster=None, roster_positions=None, proj=None):
    proj_map = dict(_PROJ if proj is None else proj)
    pidx = {
        pid: {"position": _POS[pid], "pos": _POS[pid], "team": _TEAMS[pid],
              "full_name": _NAMES[pid], "injury_status": ""}
        for pid in _TEAMS
    }
    rows = [{
        "id": pid, "name": _NAMES[pid], "position": _POS[pid],
        "team": _TEAMS[pid], "value": _VALUES[pid], "age": 27,
        "pos_rank_label": "QB31" if pid == "lock" else
        "QB32" if pid == "bris" else "QB35" if pid == "rat" else "",
        "rank_change_7d": 0,
    } for pid in _TEAMS]
    ctx = {
        "platform": "fleaflicker", "league_id": "92916", "season": 2026,
        "current_season": 2026, "current_week": WEEK,
        "league": {"type": 2}, "settings": {"type": 2},
        "scoring_settings": {}, "raw_scoring_settings": {},
        "players_index": pidx,
        "roster_positions": list(roster_positions if roster_positions is not None
                                 else _POSITIONS_SF),
        "rosters": [
            {"roster_id": 1, "players": list(roster or _ROSTER_SF)},
            {"roster_id": 2, "players": ["o1", "o2", "o3"]},
        ],
        "viewer": {"viewer_roster_id": 1},
    }
    if role_states is not None:
        ctx["_role_states"] = role_states
    monkeypatch.setattr("utils.utils.load_week_projection",
                        lambda season, week: dict(proj_map))
    monkeypatch.setattr("utils.utils.load_week_schedule",
                        lambda season, week: _schedule())
    import utils.waiver_freshness as wf
    monkeypatch.setattr(wf, "get_transactions_delta",
                        lambda *a, **k: {"adds": set(), "drops": {},
                                         "hash": ""})
    return ctx, rows


def test_next_steps_sf_promoted_qbs_claimed_past_floor(monkeypatch):
    ctx, rows = _ns_fixture(monkeypatch, role_states=dict(_ROLE_STATES))
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    by_pid = {a["player_id"]: a for a in actions}
    assert "lock" in by_pid and "bris" in by_pid
    for pid, reason in (("lock", "Starting for SEA this week"),
                        ("bris", "Starting for BAL this week")):
        claim = by_pid[pid]
        assert claim["action"].startswith("Claim ")
        assert claim["outcome"] in ("add", "add_drop")
        assert claim["starting_role"] is True
        assert claim["signal_kind"] == "starting_role"
        assert reason in claim["why"]


def test_next_steps_sf_rattler_no_claim_despite_projection(monkeypatch):
    # Same fixture: Rattler projects 18.0 (above the worst starter's
    # 13.37) but has no promotion signal and was not dropped.
    ctx, rows = _ns_fixture(monkeypatch, role_states=dict(_ROLE_STATES))
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    assert not any(a["player_id"] == "rat" for a in actions)
    blob = " ".join(a["action"] for a in actions)
    assert "Rattler" not in blob


def test_next_steps_1qb_rattler_no_claim(monkeypatch):
    # 1QB: a covered QB room (Allen-grade starter plus two backups) and
    # Rattler on the wire with a starter projection, no role state.
    roster = ["mcc", "fag", "rb1", "rb2", "rb3",
              "wr1", "wr2", "wr3", "wr4", "te1", "te2"]
    positions = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                 "BN", "BN", "BN"]
    proj = dict(_PROJ)
    proj["mcc"] = 24.0
    proj["fag"] = 19.0
    ctx, rows = _ns_fixture(
        monkeypatch, superflex=False, role_states={},
        roster=roster, roster_positions=positions, proj=proj)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    assert not any(a["player_id"] == "rat" for a in actions)
    assert not any("Rattler" in a["action"] for a in actions)


def test_next_steps_no_role_states_behaves_as_before(monkeypatch):
    # Fail-soft: with no role layer at all, Lock/Brissett stay floored
    # out (their values are below WEIGHTS.min_value and nothing else
    # admits them): exactly today's behavior.
    ctx, rows = _ns_fixture(monkeypatch, role_states=None)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    assert not any(a["player_id"] in ("lock", "bris", "rat")
                   for a in actions)


def test_next_steps_dropped_path_joins_same_signal(monkeypatch):
    # The Wave 1 dropped-starter admission is now a path into the same
    # starting_role signal: Lock dropped within 48h, no role state at
    # all, still admitted and claimed with the signal kind set.
    ctx, rows = _ns_fixture(monkeypatch, role_states={})
    import utils.waiver_freshness as wf
    # Clock-relative drop: freshness is judged against wall-clock now
    # (dropped_within, 48h), so a hardcoded date time-bombs this fixture.
    from datetime import datetime, timedelta, timezone
    recent_drop = (datetime.now(timezone.utc)
                   - timedelta(hours=2)).isoformat()
    monkeypatch.setattr(wf, "get_transactions_delta",
                        lambda *a, **k: {
                            "adds": set(),
                            "drops": {"lock": recent_drop},
                            "hash": "drop-hash"})
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    lock_claims = [a for a in actions if a["player_id"] == "lock"]
    assert len(lock_claims) == 1
    assert lock_claims[0]["signal_kind"] == "starting_role"
    assert lock_claims[0]["starting_role"] is True


# ── digest select_waiver_add: projection gate + Rattler anti-test ──────────

_SEL_TEAMS = {"dak": "DAL", "tua": "MIA", "rb1": "DET", "rb2": "SF",
              "rb3": "SEA", "wr1": "CIN", "wr2": "MIA", "wr3": "LAR",
              "wr4": "DEN", "te1": "KC", "lock": "SEA", "rat": "NO"}
_SEL_POS = {"dak": "QB", "tua": "QB", "rb1": "RB", "rb2": "RB",
            "rb3": "RB", "wr1": "WR", "wr2": "WR", "wr3": "WR",
            "wr4": "WR", "te1": "TE", "lock": "QB", "rat": "QB"}
_SEL_VAL = {"dak": 100.0, "tua": 95.0, "rb1": 100.0, "rb2": 90.0,
            "rb3": 70.0, "wr1": 100.0, "wr2": 90.0, "wr3": 80.0,
            "wr4": 60.0, "te1": 90.0, "lock": 15.0, "rat": 10.0}
_SEL_PROJ = {"dak": 20.0, "tua": 19.0, "rb1": 16.0, "rb2": 11.0,
             "rb3": 8.0, "wr1": 15.0, "wr2": 14.0, "wr3": 12.0,
             "wr4": 9.0, "te1": 10.0, "lock": 22.0, "rat": 21.0}
_SEL_ROSTER = ["dak", "tua", "rb1", "rb2", "rb3",
               "wr1", "wr2", "wr3", "wr4", "te1"]
_SEL_POSITIONS = ["QB", "SUPER_FLEX", "RB", "RB", "WR", "WR", "WR",
                  "TE", "FLEX", "BN", "BN", "BN", "BN", "BN"]


def _sel_call(candidate, **kw):
    pidx = {pid: {"position": _SEL_POS[pid], "pos": _SEL_POS[pid],
                  "team": _SEL_TEAMS[pid], "full_name": f"Name {pid}",
                  "injury_status": ""}
            for pid in _SEL_TEAMS}
    rows = [{"id": pid, "name": f"Name {pid}", "position": _SEL_POS[pid],
             "team": _SEL_TEAMS[pid], "value": _SEL_VAL[pid],
             "pos_rank": 30, "age": 27}
            for pid in _SEL_TEAMS if pid in _SEL_ROSTER or pid == candidate]
    return select_waiver_add(
        rows, set(_SEL_ROSTER), value_key="value", fallback_key="value",
        is_redraft=False, is_sf=True, n_teams=12,
        roster_players=list(_SEL_ROSTER),
        roster_positions=list(_SEL_POSITIONS), pidx=pidx,
        min_value=25.0, forward_pts_by_pid=dict(_SEL_PROJ),
        projection_by_pid=dict(_SEL_PROJ), **kw)


def test_select_waiver_add_promoted_qb_passes_on_projection():
    hit = _sel_call("lock", role_states={
        "lock": {"state": "promoted",
                 "reason": "Starting for SEA this week"}})
    assert hit is not None
    assert hit["player_id"] == "lock"
    assert hit["starting_role"] is True
    assert hit["signal_kind"] == "starting_role"
    assert "Starting for SEA" in hit["reason"]


def test_select_waiver_add_rattler_anti_superflex():
    # Rattler projects 21.0 (above both rostered QBs) but carries no
    # promotion signal, so no claim in Superflex either.
    hit = _sel_call("rat")
    assert hit is None
    hit2 = _sel_call("rat", role_states={"rat": {"state": "unknown"}})
    assert hit2 is None


# ── /api/waiver-candidates: admission + role chips ─────────────────────────

_WV_TEAMS = {"mcc": "MIN", "fag": "NYG", "rb1": "DET", "rb2": "SF",
             "wr1": "CIN", "wr2": "MIA", "te1": "KC", "b1": "DAL",
             "lock": "SEA", "bris": "BAL", "rat": "NO", "freerb": "GB",
             "o1": "GB", "o2": "CHI"}
_WV_POS = {"mcc": "QB", "fag": "QB", "rb1": "RB", "rb2": "RB",
           "wr1": "WR", "wr2": "WR", "te1": "TE", "b1": "RB",
           "lock": "QB", "bris": "QB", "rat": "QB", "freerb": "RB",
           "o1": "RB", "o2": "WR"}
_WV_VAL = {"mcc": 900, "fag": 700, "rb1": 4000, "rb2": 3000, "wr1": 4000,
           "wr2": 3000, "te1": 3000, "b1": 500, "lock": 15, "bris": 12,
           "rat": 10, "freerb": 2000, "o1": 3000, "o2": 3000}
_WV_PROJ = {"mcc": 15.0, "fag": 13.37, "rb1": 14.0, "rb2": 10.0,
            "wr1": 13.0, "wr2": 10.0, "te1": 9.0, "b1": 4.0,
            "lock": 20.67, "bris": 19.5, "rat": 18.0, "freerb": 12.0,
            "o1": 8.0, "o2": 8.0}
_WV_ROSTER1 = ["mcc", "fag", "rb1", "rb2", "wr1", "wr2", "te1", "b1"]


def _wv_client(monkeypatch):
    from datetime import datetime, timezone
    from flask import Flask
    import routes.waiver_bp as waiver

    waiver._WAIVER_CANDIDATES_CACHE.clear()
    rows = [{"id": pid, "name": f"Name {pid}", "position": _WV_POS[pid],
             "team": _WV_TEAMS[pid], "value": _WV_VAL[pid], "age": 26,
             "pos_rank_label": "", "rank_change_7d": 0}
            for pid in _WV_TEAMS]
    pidx = {pid: {"name": f"Name {pid}", "pos": _WV_POS[pid],
                  "team": _WV_TEAMS[pid]} for pid in _WV_TEAMS}
    ctx_ts = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    ctx = {
        "current_week": WEEK, "current_season": 2026, "season": 2026,
        "_cache_synced_at": ctx_ts.isoformat(),
        "league": {"settings": {}},
        "league_settings": {},
        "scoring_settings": {}, "raw_scoring_settings": {},
        "players_index": pidx,
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                             "SUPER_FLEX", "BN", "BN", "BN"],
        "rosters": [
            {"roster_id": 1, "players": list(_WV_ROSTER1),
             "reserve": [], "taxi": [], "settings": {}},
            {"roster_id": 2, "players": ["o1", "o2"],
             "reserve": [], "taxi": [], "settings": {}},
        ],
        "users": [],
        "_role_states": dict(_ROLE_STATES),
    }
    app = Flask(__name__)
    app.register_blueprint(waiver.waiver_bp)
    monkeypatch.setattr(waiver, "get_nfl_state",
                        lambda: {"season": 2026, "week": WEEK,
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
    monkeypatch.setattr(wf, "get_transactions_delta",
                        lambda *a, **k: {"adds": set(), "drops": {},
                                         "hash": "role-test"})
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


def test_waiver_candidates_role_admission_and_chips(monkeypatch):
    client = _wv_client(monkeypatch)
    resp = client.get("/api/waiver-candidates?platform=fleaflicker"
                      "&league_id=92916&season=2026&rid=1")
    assert resp.status_code == 200
    cands = {c["player_id"]: c for c in resp.json["candidates"]}
    # Both promoted QBs admitted past the floor, carrying the role
    # reason for the chip/why-line.
    for pid, reason in (("lock", "Starting for SEA this week"),
                        ("bris", "Starting for BAL this week")):
        assert pid in cands, f"{pid} missing from {sorted(cands)}"
        assert cands[pid]["starting_role"] is True
        assert cands[pid]["signal_kind"] == "starting_role"
        assert cands[pid]["role_reason"] == reason
    # Rattler: starter projection, no promotion signal -> floored out.
    assert "rat" not in cands
    # Ordinary above-floor free agent unaffected, no role fields set.
    assert "freerb" in cands
    assert cands["freerb"]["starting_role"] is False
    assert cands["freerb"]["role_reason"] is None
