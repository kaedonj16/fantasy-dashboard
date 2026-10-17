"""RB room-fit gates in the shared waiver selector (2026-10-17).

Regression for the live report: Kaedon's Next Steps queue (Blackedraw,
dynasty contender, Week 6) pitched "Claim Phil Mafah" and "Claim Kareem
Hunt" - the same family as the Mayfield QB claim fixed 2026-10-16. Three
ungated paths in rank_team_waiver_suggestions produced them:

  * Mafah type: a deep-stash RB with no projection rows anywhere. The
    evaluator returned cannot_evaluate and the selector emitted it as a
    claim; only the no-maps case (outcome None) had been failed closed.
  * Hunt type: a veteran whose window projections beat the lineup by
    >= 0.5, so evaluate_pickup said add_drop - but the drop it costed on
    points alone was a far more valuable starter who was out for the
    window (projects ~0, loses every points tie-break). The candidate's
    value was never confirmed against the ACTUAL drop; the x1.05 bar
    only compares against the worst rostered body.
  * Stash type: a breakout-driven stash outcome surfaced with no tie to
    the roster - no hole, no startable-count deficit, not a handcuff to
    her own starter, and a value below her weakest starting RB.

These tests freeze that shape: a contender roster deep and healthy at
RB where (a) and (b) must not surface, plus the positive controls - a
real value-confirmed upgrade, a handcuff stash, a future-starter stash,
a deficit-room stash, and the attested-hole exception - still do.
"""
import pytest

pd = pytest.importorskip("pandas")

import app as app_module  # noqa: E402
from utils.waivers import (  # noqa: E402
    position_room_detail,
    rank_team_waiver_suggestions,
)


# --------------------------------------------------------------------------
# position_room_detail
# --------------------------------------------------------------------------

def test_room_detail_counts_dedicated_plus_flex_requirement():
    room = position_room_detail(
        ["rb1", "rb2", "rb3"],
        {"rb1": "RB", "rb2": "RB", "rb3": "RB"},
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        position="RB",
        values_by_pid={"rb1": 4600.0, "rb2": 2400.0, "rb3": 700.0},
        teams_by_pid={"rb1": "DET", "rb2": "SF", "rb3": "SEA"},
    )
    assert room["dedicated_slots"] == 2
    assert room["flex_slots"] == 1
    assert room["required"] == 3
    assert room["healthy"] == 3
    assert room["deficit"] is False
    # The weakest starter is the 2nd-best value, not the worst body.
    assert room["weakest_starter_value"] == 2400.0
    assert room["starter_team"] == "DET"


def test_room_detail_deficit_ignores_seriously_hurt_bodies():
    room = position_room_detail(
        ["rb1", "rb2", "rb3"],
        {"rb1": "RB", "rb2": "RB", "rb3": "RB"},
        injured_by_pid={"rb2": "Out", "rb3": "IR"},
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        position="RB",
        values_by_pid={"rb1": 900.0, "rb2": 5000.0, "rb3": 4000.0},
    )
    assert room["healthy"] == 1
    assert room["deficit"] is True
    # Starters cannot be filled from healthy bodies: no starter line.
    assert room["weakest_starter_value"] is None


# --------------------------------------------------------------------------
# (a) Mafah type: no projections -> cannot_evaluate must fail closed
# --------------------------------------------------------------------------

_NS_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                 "BN", "BN", "BN", "BN", "BN", "BN"]
_NS_ROSTER = ["q1", "rb1", "rb2", "rb3", "rb4",
              "wr1", "wr2", "wr3", "te1", "te2"]
_NS_POS = {"q1": "QB", "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB",
           "wr1": "WR", "wr2": "WR", "wr3": "WR", "te1": "TE", "te2": "TE",
           "mafah": "RB"}
_NS_VALS = {"q1": 5200.0, "rb1": 4600.0, "rb2": 2400.0, "rb3": 1200.0,
           "rb4": 700.0, "wr1": 4200.0, "wr2": 3600.0, "wr3": 2800.0,
           "te1": 2600.0, "te2": 500.0}
_NS_PTS = {"q1": 24.0, "rb1": 16.0, "rb2": 11.0, "rb3": 8.0, "rb4": 5.0,
           "wr1": 15.0, "wr2": 14.0, "wr3": 12.5, "te1": 10.0, "te2": 4.0}


def _mafah_selector(**overrides):
    kwargs = dict(
        owned_ids=set(_NS_ROSTER), roster_pids=_NS_ROSTER,
        roster_positions=_NS_POSITIONS, positions_by_pid=_NS_POS,
        roster_values_by_pid=_NS_VALS,
        is_redraft=False, min_value=25.0,
        weekly_maps=[{"pts_map": dict(_NS_PTS), "covered": True}
                     for _ in range(4)],
        forward_pts_by_pid=dict(_NS_PTS),
    )
    kwargs.update(overrides)
    return rank_team_waiver_suggestions(
        [{"player_id": "mafah", "name": "Deep Stash Back", "position": "RB",
          "team": "DAL", "value": 800.0, "age": 22, "pos_rank": 45}],
        **kwargs)


def test_no_projection_candidate_suppressed_in_covered_room():
    # Value 800 clears the worst-body bar (700 x 1.05) but the candidate
    # has no points in any map: pre-fix this surfaced as a claim with
    # outcome cannot_evaluate.
    assert _mafah_selector() == []


def test_no_projection_candidate_still_surfaces_with_attested_hole():
    # The 2026-10-16 exception is unchanged: a caller/calendar-attested
    # starter hole (gap > 0) is the one value-only pass.
    out = _mafah_selector(starter_gaps={"RB": 1.0})
    assert [r["player_id"] for r in out] == ["mafah"]


# --------------------------------------------------------------------------
# (b) Hunt type: projection-edge add_drop must confirm value vs the drop
# --------------------------------------------------------------------------

_HUNT_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                   "BN", "BN", "BN", "BN"]
_HUNT_ROSTER = ["q1", "rb1", "rb2", "rb3", "rb4",
                "wr1", "wr2", "wr3", "wr4", "te1", "te2"]
_HUNT_POS = {"q1": "QB", "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB",
             "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR",
             "te1": "TE", "te2": "TE", "hunt": "RB"}
_HUNT_VALS = {"q1": 3000.0, "rb1": 5000.0, "rb2": 2600.0, "rb3": 4200.0,
              "rb4": 700.0, "wr1": 4000.0, "wr2": 3000.0, "wr3": 1500.0,
              "wr4": 900.0, "te1": 2000.0, "te2": 800.0}
_HUNT_PTS = {"q1": 22.0, "rb1": 16.0, "rb2": 11.0, "rb3": 0.0, "rb4": 1.0,
             "wr1": 15.0, "wr2": 13.0, "wr3": 6.0, "wr4": 3.0,
             "te1": 10.0, "te2": 2.0, "hunt": 12.0}


def _hunt_selector(cand_value=900.0):
    roster_pts = {k: v for k, v in _HUNT_PTS.items() if k != "hunt"}
    return rank_team_waiver_suggestions(
        [{"player_id": "hunt", "name": "Aging Vet Back", "position": "RB",
          "team": "KC", "value": cand_value, "age": 31, "pos_rank": 38,
          "weekly_pts": [12.0, 12.0, 12.0, 12.0]}],
        owned_ids=set(_HUNT_ROSTER), roster_pids=_HUNT_ROSTER,
        roster_positions=_HUNT_POSITIONS, positions_by_pid=_HUNT_POS,
        roster_values_by_pid=_HUNT_VALS,
        injured_by_pid={"rb3": "Out"},
        is_redraft=False, min_value=25.0,
        weekly_maps=[{"pts_map": dict(roster_pts), "covered": True}
                     for _ in range(4)],
        forward_pts_by_pid=dict(_HUNT_PTS),
    )


def test_projection_edge_add_drop_blocked_when_drop_worth_more():
    # The points-costed drop is rb3 (4200, out all window, projects 0).
    # A 900-value candidate cutting him is a downgrade: suppressed.
    assert _hunt_selector() == []


def test_projection_edge_add_drop_surfaces_when_candidate_outvalues_drop():
    out = _hunt_selector(cand_value=5000.0)
    assert len(out) == 1
    assert out[0]["outcome"] == "add_drop"
    assert out[0]["drop_pid"] == "rb3"


# --------------------------------------------------------------------------
# Stash rationale
# --------------------------------------------------------------------------

def _stash_selector(cand_value=800.0, cand_team="DAL", teams=None,
                    roster=None, injured=None, forward=None):
    roster = roster or ["rb1", "rb2", "rb3", "rb4"]
    pos = {p: "RB" for p in roster}
    pos["stashrb"] = "RB"
    return rank_team_waiver_suggestions(
        [{"player_id": "stashrb", "name": "Speculative Back",
          "position": "RB", "team": cand_team, "value": cand_value,
          "age": 22, "pos_rank": 40, "ros_ppg": 3.0}],
        owned_ids=set(), roster_pids=roster,
        roster_positions=["RB", "RB", "FLEX", "BN", "BN"],
        positions_by_pid=pos,
        roster_values_by_pid={"rb1": 4600.0, "rb2": 2400.0, "rb3": 1200.0,
                              "rb4": 700.0},
        teams_by_pid=teams,
        injured_by_pid=injured,
        is_redraft=False, min_value=25.0,
        breakout_by_pid={"stashrb": 80.0},
        forward_pts_by_pid=forward or {"rb1": 16.0, "rb2": 11.0, "rb3": 8.0,
                                        "rb4": 5.0, "stashrb": 3.0},
    )


def test_stash_without_roster_rationale_suppressed():
    # Deep healthy room (4 RBs for RB/RB/FLEX), candidate is not her
    # starter's handcuff and his value sits below her weakest starter.
    assert _stash_selector(teams={"rb1": "DET", "rb2": "SF",
                                  "rb3": "SEA", "rb4": "DAL"}) == []


def test_stash_handcuff_to_own_starter_surfaces():
    out = _stash_selector(cand_team="DET",
                          teams={"rb1": "DET", "rb2": "SF",
                                 "rb3": "SEA", "rb4": "DAL"})
    assert len(out) == 1
    assert out[0]["outcome"] == "stash"


def test_stash_above_weakest_starter_value_surfaces():
    # 2600 clears the weakest starter (rb2, 2400) x 1.05: a genuine
    # future-starter stash even with no hole, deficit, or handcuff tie.
    out = _stash_selector(cand_value=2600.0,
                          teams={"rb1": "DET", "rb2": "SF",
                                 "rb3": "SEA", "rb4": "DAL"})
    assert len(out) == 1
    assert out[0]["outcome"] == "stash"


def test_stash_in_deficit_room_surfaces():
    # One healthy RB for RB/RB/FLEX: a startable-count deficit is its
    # own rationale (here the candidate also cracks the gutted lineup,
    # so any non-hold outcome proves the room let him through).
    out = rank_team_waiver_suggestions(
        [{"player_id": "stashrb", "name": "Depth Back", "position": "RB",
          "team": "DAL", "value": 1000.0, "age": 22, "pos_rank": 40,
          "ros_ppg": 3.0}],
        owned_ids=set(), roster_pids=["rb1", "rb2", "rb3"],
        roster_positions=["RB", "RB", "FLEX", "BN", "BN"],
        positions_by_pid={"rb1": "RB", "rb2": "RB", "rb3": "RB",
                          "stashrb": "RB"},
        roster_values_by_pid={"rb1": 900.0, "rb2": 5000.0, "rb3": 4000.0},
        injured_by_pid={"rb2": "Out", "rb3": "IR"},
        is_redraft=False, min_value=25.0,
        breakout_by_pid={"stashrb": 80.0},
        forward_pts_by_pid={"rb1": 16.0, "rb2": 0.0, "rb3": 0.0,
                            "stashrb": 3.0},
    )
    assert [r["player_id"] for r in out] == ["stashrb"]


# --------------------------------------------------------------------------
# Next Steps surface: her shape end to end through _next_steps_waiver_actions
# --------------------------------------------------------------------------

TEAMS = {
    "q1": "BUF",
    "rb1": "DET", "rb2": "SF", "rb3": "SEA", "rb4": "DAL", "rb5": "NYJ",
    "wr1": "CIN", "wr2": "MIA", "wr3": "LAR", "wr4": "DEN", "wr5": "PHI",
    "wr6": "CLE",
    "te1": "KC", "te2": "BAL",
    "mafah": "DAL", "hunt": "KC",
    "o1": "GB", "o2": "MIN", "o3": "CHI",
}
NAMES = {
    "q1": "Josh Allen",
    "rb1": "Jahmyr Gibbs", "rb2": "Breece Hall", "rb3": "Star Back",
    "rb4": "Tony Pollard", "rb5": "Deep Bench Back",
    "wr1": "Ja'Marr Chase", "wr2": "Tyreek Hill", "wr3": "Puka Nacua",
    "wr4": "Courtland Sutton", "wr5": "Diontae Johnson",
    "wr6": "Bench Wideout",
    "te1": "Travis Kelce", "te2": "David Njoku",
    "mafah": "Deep Stash", "hunt": "Aging Vet",
    "o1": "Opp One", "o2": "Opp Two", "o3": "Opp Three",
}
POS = {
    "q1": "QB",
    "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB", "rb5": "RB",
    "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR", "wr5": "WR",
    "wr6": "WR",
    "te1": "TE", "te2": "TE",
    "mafah": "RB", "hunt": "RB",
    "o1": "RB", "o2": "WR", "o3": "TE",
}
VALUES = {
    "q1": 5200,
    "rb1": 4600, "rb2": 2400, "rb3": 4200, "rb4": 700, "rb5": 300,
    "wr1": 4200, "wr2": 3600, "wr3": 2800, "wr4": 900, "wr5": 650,
    "wr6": 200,
    "te1": 2600, "te2": 500,
    # Mafah (800) clears the healthy worst-body bar (rb5, 300) and the
    # rank-less quality value check, but has no projection at all.
    "mafah": 800,
    # Hunt (900) also clears the bar; his projection edge is real, but
    # the drop the points math picks is rb3 (4200, Out all window).
    "hunt": 900,
    "o1": 3000, "o2": 3000, "o3": 3000,
}
PROJ = {
    "q1": 24.0,
    "rb1": 16.0, "rb2": 11.0, "rb4": 5.0, "rb5": 1.0,
    # rb3 (Star Back) is Out: no projection in any week.
    "wr1": 15.0, "wr2": 14.0, "wr3": 12.5, "wr4": 3.0, "wr5": 2.0,
    "wr6": 1.0,
    "te1": 10.0, "te2": 4.0,
    # mafah: no projection rows anywhere.
    "hunt": 12.0,
}
ROSTER = ["q1", "rb1", "rb2", "rb3", "rb4", "rb5",
          "wr1", "wr2", "wr3", "wr4", "wr5", "wr6", "te1", "te2"]
ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                    "BN", "BN", "BN", "BN", "BN", "BN"]


def _schedule():
    teams = sorted(set(TEAMS.values()))
    return [{"away": teams[i], "home": teams[i + 1]}
            for i in range(0, len(teams) - 1, 2)]


def _fixture(monkeypatch):
    pidx = {
        pid: {"position": POS[pid], "pos": POS[pid], "team": TEAMS[pid],
              "full_name": NAMES[pid],
              "injury_status": "Out" if pid == "rb3" else ""}
        for pid in TEAMS
    }
    rows = [{
        "id": pid, "name": NAMES[pid], "position": POS[pid],
        "team": TEAMS[pid], "value": VALUES[pid],
        "age": 31 if pid == "hunt" else 25,
        "pos_rank_label": "", "rank_change_7d": 0,
    } for pid in TEAMS]
    ctx = {
        "platform": "sleeper", "league_id": "L1", "season": 2026,
        "current_season": 2026, "current_week": 7,
        "league": {"type": 2}, "settings": {"type": 2},
        "scoring_settings": {}, "raw_scoring_settings": {},
        "players_index": pidx,
        "roster_positions": list(ROSTER_POSITIONS),
        "rosters": [
            {"roster_id": 1, "players": list(ROSTER)},
            {"roster_id": 2, "players": ["o1", "o2", "o3"]},
        ],
        "viewer": {"viewer_roster_id": 1},
    }

    def fake_proj(season, week):
        return {pid: pts for pid, pts in PROJ.items()}

    monkeypatch.setattr("utils.utils.load_week_projection", fake_proj)
    monkeypatch.setattr("utils.utils.load_week_schedule",
                        lambda season, week: _schedule())
    return ctx, rows


def test_next_steps_pitches_neither_bad_rb_claim(monkeypatch):
    ctx, rows = _fixture(monkeypatch)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    pids = {a["player_id"] for a in actions}
    assert "mafah" not in pids
    assert "hunt" not in pids
    blob = " ".join(a["action"] for a in actions)
    assert "Deep Stash" not in blob
    assert "Aging Vet" not in blob


def test_next_steps_rb_claim_surfaces_when_it_outvalues_the_drop(monkeypatch):
    # Positive control at the surface: same projection edge, but the
    # candidate is worth more than the injured starter he would cut.
    monkeypatch.setitem(VALUES, "hunt", 5000)
    ctx, rows = _fixture(monkeypatch)
    actions = app_module._next_steps_waiver_actions(ctx, 1, rows)
    claims = [a for a in actions if a["player_id"] == "hunt"]
    assert len(claims) == 1
    assert claims[0]["action"] == "Claim Aging Vet"
    assert claims[0]["outcome"] == "add_drop"
    # The no-projection stash stays suppressed in the same queue.
    assert "mafah" not in {a["player_id"] for a in actions}
