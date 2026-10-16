"""Covered-QB-room regression (2026-10-16).

Kaedon (Blackedraw, dynasty 1QB) rosters Caleb Williams, Bo Nix and
Brock Purdy and was still told to pick up Baker Mayfield, the top
available QB by value. Three earlier fixes (#2619 shared selector,
#2634 assistant/Next Steps bypass, #2642 Waiver Wire Targets + digest
parity) each missed a way he still passed:

* the selector's QB coverage rule only fired when the candidate ALSO
  failed the x1.05 drop bar, a case the drop-bar gate had already
  handled, so it never suppressed anything: Mayfield's value (2600)
  beat the worst rostered QB (Purdy, 1500) x 1.05 and he passed;
* with full projection maps, one spot-week projection edge over the
  starter (21.5 vs 19.5) classified him add_drop, and nothing checked
  QB-room coverage once the value bar passed;
* with no projection maps at all, the selector failed open and emitted
  him on value alone (the email digest's no-scoring path);
* the assistant's build_waivers never called the selector: its gate is
  value vs worst-at-position x 1.05, full stop;
* the Waivers page outcome came straight from evaluate_pickup, with no
  coverage check on its add/add_drop verdicts.

The rule now: a covered QB room (healthy QBs at/above the QB +
SUPER_FLEX slots, no same-week calendar hole) yields no QB claim on
any surface, and team-fit claims whose lineup-gain classification
cannot run are not surfaced on value alone (a real starter hole is
the only value-only pass). The legitimate hole (tw4's shape: starter
on bye, backup Out, two QBs rostered) still surfaces.

Fixture values use the site's dynasty scale (thousands). 2026 byes in
the Week 7-10 window: none for CHI/DEN/SF, and all three QBs healthy,
which is the realistic state that makes any Mayfield pass indefensible.
"""
import pytest

from utils.waivers import (
    qb_coverage_detail,
    rank_team_waiver_suggestions,
)

RP = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
      "BN", "BN", "BN", "BN", "BN", "BN"]
POS = {"will": "QB", "nix": "QB", "purdy": "QB",
       "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB",
       "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR", "wr5": "WR",
       "te1": "TE", "te2": "TE",
       "mayf": "QB", "wrb": "RB", "wwr": "WR"}
NAMES = {"will": "Caleb Williams", "nix": "Bo Nix", "purdy": "Brock Purdy",
         "mayf": "Baker Mayfield", "wrb": "Rising Back", "wwr": "Wire Wideout"}
TEAMS = {"will": "CHI", "nix": "DEN", "purdy": "SF", "mayf": "TB"}
VAL = {"will": 2400, "nix": 2100, "purdy": 1500,
       "rb1": 4000, "rb2": 2600, "rb3": 1500, "rb4": 800,
       "wr1": 4200, "wr2": 3400, "wr3": 2600, "wr4": 1400, "wr5": 700,
       "te1": 2400, "te2": 600,
       "mayf": 2600, "wrb": 2100, "wwr": 900}
PROJ = {"will": 19.5, "nix": 18.0, "purdy": 16.5,
        "rb1": 15.0, "rb2": 12.0, "rb3": 9.0, "rb4": 6.0,
        "wr1": 15.0, "wr2": 13.0, "wr3": 11.0, "wr4": 8.0, "wr5": 5.0,
        "te1": 10.0, "te2": 4.0, "wrb": 13.5, "wwr": 7.0}
ROSTER = ["will", "nix", "purdy", "rb1", "rb2", "rb3", "rb4",
          "wr1", "wr2", "wr3", "wr4", "wr5", "te1", "te2"]


def _cand(pid, weekly=None):
    c = {"player_id": pid, "name": NAMES.get(pid, pid), "position": POS[pid],
         "team": TEAMS.get(pid, "BUF"), "value": VAL[pid],
         "age": 30 if pid == "mayf" else 25,
         "pos_rank": 6 if pid == "mayf" else 20}
    if weekly is not None:
        c["weekly_pts"] = weekly
    return c


def _maps(roster_pts):
    return [{"pts_map": dict(roster_pts), "covered": True} for _ in range(4)]


def _selector(cands, roster=ROSTER, **kw):
    base = dict(owned_ids=set(roster), roster_pids=list(roster),
                roster_positions=list(RP), positions_by_pid=dict(POS),
                roster_values_by_pid={p: VAL[p] for p in roster},
                is_redraft=False, min_value=25.0)
    base.update(kw)
    return rank_team_waiver_suggestions(cands, **base)


# ---------------------------------------------------------------------------
# The shared coverage helper
# ---------------------------------------------------------------------------

def test_coverage_detail_her_room_is_covered():
    d = qb_coverage_detail(ROSTER, POS, {}, RP)
    assert d == {"covered": True, "healthy_qbs": 3, "qb_slots": 1, "hole": False}


def test_coverage_detail_bye_plus_out_is_a_hole():
    roster = [p for p in ROSTER if p != "purdy"]
    avail = [{"week": 7, "available_by_pos": {"QB": 0}},
             {"week": 8, "available_by_pos": {"QB": 1}}]
    d = qb_coverage_detail(roster, POS, {"nix": "Out"}, RP,
                           weekly_availability=avail)
    assert d["hole"] is True and d["covered"] is False


def test_coverage_detail_multi_qb_formats_never_covered():
    # The absolute rule is a 1QB rule: in superflex/2QB, depth past the
    # slots is startable, so coverage never hard-suppresses and the
    # value bar + classification govern (the stacked-room great-claim
    # case in test_waiver_roster_gate.py is pinned behavior there).
    d = qb_coverage_detail(["will"], POS, {}, ["QB", "SUPER_FLEX", "RB"])
    assert d["qb_slots"] == 2 and d["covered"] is False
    d2 = qb_coverage_detail(["will", "nix", "purdy"], POS, {},
                            ["QB", "SUPER_FLEX", "RB"])
    assert d2["healthy_qbs"] == 3 and d2["covered"] is False


# ---------------------------------------------------------------------------
# Selector: her shape, three input modes
# ---------------------------------------------------------------------------

def test_selector_bare_mode_suppresses_covered_qb():
    # No projection maps at all (the degraded caller mode): Mayfield's
    # value beats Purdy x 1.05, but a team-fit claim may not pass on
    # value alone, and the QB room is covered besides.
    out = _selector([_cand("mayf"), _cand("wrb"), _cand("wwr")])
    assert "mayf" not in {r["player_id"] for r in out}


def test_selector_full_maps_spot_week_edge_still_suppressed():
    # Mayfield out-projects Williams in Week 7 only (21.5 vs 19.5): the
    # classification calls it add_drop, but coverage suppresses it: a
    # fourth QB in 1QB is bench weight, not a lineup upgrade.
    cands = [_cand("mayf", [21.5, 18.5, 18.5, 18.5]),
             _cand("wrb", [13.5] * 4), _cand("wwr", [7.0] * 4)]
    out = _selector(cands, weekly_maps=_maps({p: PROJ[p] for p in ROSTER}))
    ids = {r["player_id"] for r in out}
    assert "mayf" not in ids
    assert "wrb" in ids  # the genuine flex gain still surfaces


def test_selector_full_maps_no_edge_suppressed_control():
    cands = [_cand("mayf", [18.5] * 4),
             _cand("wrb", [13.5] * 4), _cand("wwr", [7.0] * 4)]
    out = _selector(cands, weekly_maps=_maps({p: PROJ[p] for p in ROSTER}))
    assert "mayf" not in {r["player_id"] for r in out}


def test_selector_calendar_hole_still_surfaces_qb():
    # tw4's shape through the bare path: two QBs rostered, starter on
    # bye in Week 7 and the backup Out, so the calendar registers a QB
    # hole and the claim survives on the hole alone.
    from utils.digest import recommend_waivers
    roster = [p for p in ROSTER if p != "purdy"]
    rows = [{"id": p, "name": NAMES.get(p, p), "pos": POS[p], "value": VAL[p],
             "age": 30 if p == "mayf" else 25,
             "pos_rank": 6 if p == "mayf" else 20,
             "pos_rank_label": "QB6" if p == "mayf" else ""}
            for p in POS]
    full = {"QB": 1, "RB": 4, "WR": 5, "TE": 2}
    avail = [{"week": 7, "available_by_pos": {**full, "QB": 0}},
             {"week": 8, "available_by_pos": dict(full)},
             {"week": 9, "available_by_pos": dict(full)},
             {"week": 10, "available_by_pos": dict(full)}]
    hits = recommend_waivers(
        rows, set(roster), roster_players=roster, roster_positions=list(RP),
        pidx={p: {"position": POS[p]} for p in POS},
        fmt={"is_dynasty": True, "is_superflex": False},
        limit=5, min_score=0.0, weekly_availability=avail)
    by_pid = {t["player_id"]: t for t in hits}
    assert "mayf" in by_pid
    assert by_pid["mayf"]["starter_gap"] >= 1


def test_digest_bare_call_suppresses_covered_qb():
    # The parity gap, closed: recommend_waivers with no opportunity
    # inputs used to wave Mayfield in on value gates alone.
    from utils.digest import recommend_waivers
    rows = [{"id": p, "name": NAMES.get(p, p), "pos": POS[p], "value": VAL[p],
             "age": 30 if p == "mayf" else 25,
             "pos_rank": 6 if p == "mayf" else 20,
             "pos_rank_label": "QB6" if p == "mayf" else ""}
            for p in POS]
    hits = recommend_waivers(
        rows, set(ROSTER), roster_players=list(ROSTER),
        roster_positions=list(RP),
        pidx={p: {"position": POS[p]} for p in POS},
        fmt={"is_dynasty": True, "is_superflex": False},
        limit=5, min_score=0.0)
    assert "mayf" not in {t["player_id"] for t in hits}


# ---------------------------------------------------------------------------
# Assistant evidence pack: build_waivers on her exact shape
# ---------------------------------------------------------------------------

def _assistant_bundle(monkeypatch):
    pytest.importorskip("openai")
    from dashboard_services.assistant import evidence
    import routes.waiver_bp as _wb
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: {
        p: {"name": NAMES.get(p, p), "team": TEAMS.get(p, "BUF"),
            "position": POS[p]} for p in POS})
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda *a, **k: [{"id": p, "value": VAL[p]} for p in POS])
    monkeypatch.setattr(_wb, "load_do_not_drop", lambda *a, **k: set())
    return {"platform": "sleeper", "league_id": "L1", "season": 2026,
            "league_name": "Blackedraw", "team_name": "Caleb's Casting Couch",
            "week": 7, "league": {"is_dynasty": True},
            "ctx": {"raw_scoring_settings": {"rec": 0.5},
                    "rosters": [{"roster_id": 1, "players": list(ROSTER)}],
                    "roster_positions": list(RP)},
            "roster": {"players": list(ROSTER), "starters": []},
            "roster_player_ids": set(ROSTER)}


def test_assistant_waivers_pack_suppresses_covered_qb(monkeypatch):
    from dashboard_services.assistant import evidence
    pack, _tools = evidence.build_waivers(_assistant_bundle(monkeypatch), {})
    ids = {c["player_id"] for c in pack["candidates"]}
    assert "mayf" not in ids
    assert "wrb" in ids  # non-QB claims are unaffected
    assert pack["qb_coverage"]["covered"] is True
    assert pack["qb_coverage"]["healthy_qbs"] == 3


def test_assistant_stash_pool_suppresses_covered_qb(monkeypatch):
    from dashboard_services.assistant import evidence
    pack, _tools = evidence.build_waivers(
        _assistant_bundle(monkeypatch), {"waiver_mode": "stash"})
    assert "mayf" not in {c["player_id"] for c in pack["stash_candidates"]}


# ---------------------------------------------------------------------------
# Waivers page API: the outcome downgrade
# ---------------------------------------------------------------------------

def test_waiver_candidates_api_downgrades_covered_qb(monkeypatch):
    pytest.importorskip("flask")
    pytest.importorskip("pandas")
    from flask import Flask
    import routes.waiver_bp as waiver

    ctx = {
        "current_week": 7, "current_season": 2026,
        "league": {"type": 2, "settings": {"type": 2, "waiver_type": 0}},
        "settings": {"type": 2, "waiver_type": 0},
        "scoring_settings": {}, "raw_scoring_settings": {"rec": 0.5},
        "rosters": [
            {"roster_id": 1, "players": list(ROSTER), "reserve": [], "taxi": []},
            {"roster_id": 2, "players": ["o1", "o2", "o3"],
             "reserve": [], "taxi": []},
        ],
        "users": [],
        "players_index": {
            **{p: {"name": NAMES.get(p, p), "pos": POS[p],
                    "team": TEAMS.get(p, "BUF")} for p in POS},
            **{p: {"name": p, "pos": "RB", "team": "GB"} for p in ("o1", "o2", "o3")},
        },
        "roster_positions": list(RP),
    }
    rows = [{"id": p, "name": NAMES.get(p, p), "position": POS[p],
             "team": TEAMS.get(p, "BUF"), "value": VAL[p], "sf_value": VAL[p],
             "age": 30 if p == "mayf" else 25,
             "pos_rank_label": "QB6" if p == "mayf" else ""}
            for p in POS]
    fwd = {**PROJ, "mayf": 21.5}  # Mayfield out-projects the room ROS-wide

    app = Flask(__name__)
    app.register_blueprint(waiver.waiver_bp)
    monkeypatch.setattr(waiver, "get_nfl_state",
                        lambda: {"season": 2026, "week": 7, "season_type": "reg"})
    monkeypatch.setattr(waiver, "get_league_ctx_from_cache", lambda *a: ctx)
    monkeypatch.setattr(waiver, "get_model_value_table_cached", lambda *a, **k: rows)
    monkeypatch.setattr(waiver, "get_players_global", lambda *a, **k: {
        p: {"injury_status": "", "team": TEAMS.get(p, "BUF")} for p in POS})
    monkeypatch.setattr(waiver, "_forward_ppg_map",
                        lambda season, raw, entries: dict(fwd))
    monkeypatch.setattr(waiver, "_sleeper_trending_adds", lambda limit=50: [])
    monkeypatch.setattr("utils.utils.load_week_projection",
                        lambda season, week: {})
    all_teams = ["CHI", "DEN", "SF", "TB", "BUF", "GB"]
    games = [{"away": all_teams[i], "home": all_teams[i + 1]}
             for i in range(0, len(all_teams) - 1, 2)]
    monkeypatch.setattr("utils.utils.load_week_schedule",
                        lambda season, week: games)
    monkeypatch.setattr("utils.utils.load_usage_table", lambda *a, **k: [
        {"player_id": p, "ppr_ppg": fwd[p]} for p in fwd])

    client = app.test_client()
    resp = client.get("/api/waiver-candidates?platform=sleeper&league_id=L1"
                      "&season=2026&rid=1")
    assert resp.status_code == 200
    cands = {c["player_id"]: c for c in resp.json["candidates"]}
    assert "mayf" in cands  # the row stays browsable
    mayf = cands["mayf"]
    assert mayf["outcome"] == "hold"  # but the claim is gone
    assert mayf["lineup_gain"] is None
    assert mayf["drop"] is None
    wrb = cands["wrb"]
    assert wrb["outcome"] in ("add", "add_drop")  # real gains unaffected
    assert wrb["lineup_gain"] == 4.5
