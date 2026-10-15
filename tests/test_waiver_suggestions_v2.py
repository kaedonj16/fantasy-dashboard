"""Waiver suggestions v2 (2026-10-14): lineup gain, calendar need, horizon,
claim economics, opportunity inputs, and the shared selector.

The Rattler regression leads: a stacked superflex QB room must never be
pitched a non-starting backup QB, even with calendar/horizon inputs on.
"""
from utils.digest import recommend_waivers, select_waiver_add
from utils.waivers import (
    calendar_need_by_position,
    choose_suggestion_horizon,
    claim_guidance_for,
    evaluate_pickup,
    faab_recommendation,
    rank_team_waiver_suggestions,
    weekly_availability_from_roster,
    weekly_maps_from_per_week_points,
)


# ---------------------------------------------------------------- (a) Rattler
def _sf_qb_room(candidate_value=60.0, candidate_rank=22):
    pidx = {}
    rows = []
    roster_values = {"dak": 100.0, "tua": 95.0, "darnold": 90.0, "lawrence": 85.0}
    for pid, val in roster_values.items():
        pidx[pid] = {"position": "QB", "injury_status": ""}
        rows.append({"id": pid, "name": f"QB {pid}", "position": "QB",
                     "team": "DAL", "value": val, "pos_rank": 10, "age": 28})
    pidx["rattler"] = {"position": "QB", "injury_status": ""}
    rows.append({"id": "rattler", "name": "Spencer Rattler", "position": "QB",
                 "team": "NO", "value": candidate_value,
                 "pos_rank": candidate_rank, "age": 24})
    positions = ["QB", "SUPER_FLEX", "RB", "RB", "WR", "WR", "WR", "TE",
                 "FLEX", "BN", "BN", "BN", "BN", "BN", "BN", "BN"]
    return rows, pidx, positions, list(roster_values)


def test_rattler_never_surfaces_with_calendar_and_horizon_inputs():
    rows, pidx, positions, roster = _sf_qb_room()
    hit = select_waiver_add(
        rows, set(roster), value_key="value", is_redraft=False, is_sf=True,
        roster_players=roster, roster_positions=positions, pidx=pidx,
        min_value=10.0, direction="contender", playoff_pct=60,
        calendar_need={"QB": {"gap_weeks": [], "starter_gap": 0.0, "need_mult": 1.0}},
    )
    assert hit is None


def test_rattler_also_filtered_by_shared_selector_directly():
    rows, pidx, positions, roster = _sf_qb_room()
    out = rank_team_waiver_suggestions(
        [{"player_id": "rattler", "name": "Spencer Rattler", "position": "QB",
          "team": "NO", "value": 60.0, "age": 24, "pos_rank": 22}],
        owned_ids=set(roster), roster_pids=roster, roster_positions=positions,
        positions_by_pid={k: v["position"] for k, v in pidx.items()},
        roster_values_by_pid={"dak": 100.0, "tua": 95.0, "darnold": 90.0,
                              "lawrence": 85.0},
        is_redraft=False, is_sf=True, min_value=10.0,
    )
    assert out == []


# ------------------------------------------------------- (b) lineup gain gate
def _wr_room_selector(cand_extra=None, breakout=None):
    cand = {"player_id": "cand", "name": "Value Win", "position": "WR",
            "team": "BUF", "value": 500, "age": 24, "pos_rank": 20,
            "ros_ppg": 5.0}
    cand.update(cand_extra or {})
    return rank_team_waiver_suggestions(
        [cand],
        owned_ids=set(), roster_pids=["w1", "w2", "f1"],
        roster_positions=["WR", "WR", "FLEX", "BN", "BN", "BN"],
        positions_by_pid={"w1": "WR", "w2": "WR", "f1": "WR"},
        roster_values_by_pid={"w1": 300.0, "w2": 280.0, "f1": 260.0},
        is_redraft=True, min_value=25.0, breakout_by_pid=breakout,
        forward_pts_by_pid={"w1": 15.0, "w2": 14.0, "f1": 13.0, "cand": 5.0},
    )


def test_value_win_with_zero_lineup_gain_is_suppressed():
    assert _wr_room_selector() == []


def test_value_win_with_zero_gain_but_upside_is_stash_never_add():
    out = _wr_room_selector(breakout={"cand": 80.0})
    assert len(out) == 1
    assert out[0]["outcome"] == "stash"
    assert out[0]["lineup_gain"] == 0.0


def test_lineup_gain_add_carries_drop_and_replaces():
    out = rank_team_waiver_suggestions(
        [{"player_id": "cand", "name": "Real Upgrade", "position": "WR",
          "team": "BUF", "value": 500, "age": 24, "pos_rank": 20,
          "ros_ppg": 20.0}],
        owned_ids=set(), roster_pids=["w1", "w2", "f1"],
        roster_positions=["WR", "WR", "FLEX", "BN", "BN", "BN"],
        positions_by_pid={"w1": "WR", "w2": "WR", "f1": "WR"},
        roster_values_by_pid={"w1": 300.0, "w2": 280.0, "f1": 260.0},
        is_redraft=True, min_value=25.0,
        forward_pts_by_pid={"w1": 15.0, "w2": 14.0, "f1": 13.0, "cand": 20.0},
    )
    assert len(out) == 1
    assert out[0]["outcome"] in ("add", "add_drop")
    assert out[0]["lineup_gain"] >= 0.5
    assert out[0]["replaces_pid"] in ("w1", "w2", "f1")


# ------------------------------------------------------------- (c) calendar
def test_calendar_need_helper_finds_bye_gap():
    need = calendar_need_by_position(
        [{"week": 8, "available_by_pos": {"TE": 1}},
         {"week": 9, "available_by_pos": {"TE": 0}},
         {"week": 10, "available_by_pos": {"TE": 1}}],
        {"TE": 1},
    )
    assert need["TE"]["gap_weeks"] == [9]
    assert need["TE"]["starter_gap"] == 1.0
    assert need["TE"]["need_mult"] > 1.0


def test_weekly_availability_builder_counts_bye_as_unavailable():
    avail = weekly_availability_from_roster(
        ["te1"], {"te1": {"position": "TE", "team": "KC"}},
        current_week=8, weeks=3, bye_week_by_team={"KC": 9},
    )
    assert avail[0]["available_by_pos"].get("TE") == 1
    assert avail[1]["available_by_pos"].get("TE", 0) == 0
    assert avail[2]["available_by_pos"].get("TE") == 1


def _te_rows():
    rows = [
        {"id": "qb1", "name": "QB One", "position": "QB", "team": "DAL",
         "value": 200, "pos_rank": 8, "age": 27},
        {"id": "rb1", "name": "RB One", "position": "RB", "team": "DAL",
         "value": 200, "pos_rank": 10, "age": 25},
        {"id": "rb2", "name": "RB Two", "position": "RB", "team": "DAL",
         "value": 180, "pos_rank": 14, "age": 25},
        {"id": "wr1", "name": "WR One", "position": "WR", "team": "DAL",
         "value": 200, "pos_rank": 12, "age": 26},
        {"id": "wr2", "name": "WR Two", "position": "WR", "team": "DAL",
         "value": 180, "pos_rank": 16, "age": 26},
        {"id": "te1", "name": "TE One", "position": "TE", "team": "KC",
         "value": 200, "pos_rank": 5, "age": 28},
        {"id": "tefa", "name": "Wire TE", "position": "TE", "team": "BUF",
         "value": 90, "pos_rank": 11, "age": 26},
    ]
    pidx = {r["id"]: {"position": r["position"]} for r in rows}
    roster = ["qb1", "rb1", "rb2", "wr1", "wr2", "te1"]
    positions = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN"]
    return rows, pidx, roster, positions


def test_calendar_bye_hole_surfaces_te_that_drop_bar_blocks_otherwise():
    rows, pidx, roster, positions = _te_rows()
    base = select_waiver_add(
        rows, set(roster), value_key="value", is_redraft=True,
        roster_players=roster, roster_positions=positions, pidx=pidx,
    )
    assert base is None  # 90 does not beat TE1's 200 x 1.05, and no hole today
    full = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    avail = [
        {"week": 8, "available_by_pos": dict(full)},
        {"week": 9, "available_by_pos": dict(full)},
        {"week": 10, "available_by_pos": {**full, "TE": 0}},
    ]
    hit = select_waiver_add(
        rows, set(roster), value_key="value", is_redraft=True,
        roster_players=roster, roster_positions=positions, pidx=pidx,
        weekly_availability=avail,
    )
    assert hit is not None
    assert hit["player_id"] == "tefa"
    assert hit["starter_gap"] > 0


# -------------------------------------------------------------- (d) horizon
def test_choose_suggestion_horizon_rules():
    assert choose_suggestion_horizon({"is_dynasty": True}, direction="rebuild") == "stash"
    assert choose_suggestion_horizon({"is_dynasty": True}, playoff_pct=10) == "stash"
    assert choose_suggestion_horizon({"is_dynasty": True}, direction="contender") == "four_week"
    assert choose_suggestion_horizon({"is_redraft": True}, playoff_pct=50) == "this_week"
    assert choose_suggestion_horizon({"is_redraft": True}, playoff_pct=95) == "four_week"
    assert choose_suggestion_horizon() == "four_week"


def _horizon_pair(**sel_kwargs):
    cands = [
        {"player_id": "young", "name": "Young Stash", "position": "WR",
         "team": "BUF", "value": 300, "age": 21, "pos_rank": 25, "ros_ppg": 4.0},
        {"player_id": "vet", "name": "Aging Vet", "position": "WR",
         "team": "BUF", "value": 280, "age": 31, "pos_rank": 22, "ros_ppg": 12.0},
    ]
    out = rank_team_waiver_suggestions(
        cands, owned_ids=set(), min_value=25.0, **sel_kwargs)
    return [r["player_id"] for r in out]


def test_horizon_flips_young_stash_vs_aging_vet():
    dynasty = _horizon_pair(is_redraft=False, direction="rebuild")
    assert dynasty[0] == "young"
    redraft = _horizon_pair(is_redraft=True, playoff_pct=50)
    assert redraft[0] == "vet"


# ------------------------------------------------------- (e) claim economics
def test_claim_guidance_labels():
    top = claim_guidance_for(180, adds_count=1_500_000)
    assert top["label"].startswith("Top claim")
    low = claim_guidance_for(50, adds_count=120)
    assert low["label"].startswith("Likely clears")
    mid = claim_guidance_for(120, adds_count=100)
    assert mid["label"] == "Worth a claim"


def test_faab_recommendation_fills_guidance_and_caps_at_remaining():
    rec = faab_recommendation(190.0, budget_total=100, budget_remaining=60)
    assert rec["claim_guidance"]
    assert rec["high"] <= 60
    assert rec["low"] <= rec["target"] <= rec["high"]


def test_selector_attaches_claim_guidance_and_faab():
    out = rank_team_waiver_suggestions(
        [{"player_id": "hot", "name": "Hot Add", "position": "RB",
          "team": "BUF", "value": 900, "age": 25, "pos_rank": 10,
          "ros_ppg": 15.0, "adds_count": 2_000_000}],
        owned_ids=set(), min_value=25.0, is_redraft=True,
        budget_total=100, budget_remaining=40,
    )
    assert len(out) == 1
    assert out[0]["claim_guidance"].startswith("Top claim")
    assert out[0]["faab"]["high"] <= 40


# ------------------------------------------------------------ (f) opportunity
def test_low_value_credible_opportunity_surfaces_noise_does_not():
    rows = [
        {"id": "cred", "name": "Next Man", "pos": "RB", "value": 30, "age": 23,
         "pos_rank": 24, "pos_rank_label": "RB24",
         "injured_ahead": ["OUT"], "healthy_ahead": 0,
         "usage_stat": "targets", "usage_delta": 3.0},
        {"id": "noise", "name": "Noise Mover", "pos": "RB", "value": 30, "age": 23,
         "pos_rank": 24, "pos_rank_label": "RB24", "rank_change_7d": 20},
    ]
    hits = recommend_waivers(
        rows, set(), roster_players=[],
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        pidx={}, fmt={"is_redraft": True}, limit=3, min_score=1.0,
    )
    ids = [h["player_id"] for h in hits]
    assert "cred" in ids
    assert "noise" not in ids


def test_breakout_map_is_credible_admission_for_select():
    out = rank_team_waiver_suggestions(
        [{"player_id": "bo", "name": "Breakout", "position": "WR",
          "team": "BUF", "value": 20, "age": 22, "pos_rank": 30}],
        owned_ids=set(), min_value=25.0, breakout_by_pid={"bo": 70.0},
    )
    assert [r["player_id"] for r in out] == ["bo"]
    out2 = rank_team_waiver_suggestions(
        [{"player_id": "bo", "name": "Breakout", "position": "WR",
          "team": "BUF", "value": 20, "age": 22, "pos_rank": 30}],
        owned_ids=set(), min_value=25.0, breakout_by_pid={},
    )
    assert out2 == []


# ------------------------------------------------- (g) per-week lineup maps
def test_horizon_gain_uses_distinct_weekly_maps():
    weekly = weekly_maps_from_per_week_points(
        ["a", "b"], "c",
        [{"a": 10.0, "b": 9.0}, {"a": 10.0, "b": 9.0}],
        [20.0, 18.0],
    )
    ev = evaluate_pickup(
        "c", "WR", ["a", "b"], weekly,
        {"a": "WR", "b": "WR", "c": "WR"}, ["WR", "FLEX", "BN", "BN", "BN"],
        droppable_pids=["a", "b"],
    )
    assert ev.week_gain == 11.0
    assert ev.horizon_gain == 20.0
    assert ev.horizon_gain != ev.week_gain


def test_bye_week_is_skipped_in_horizon_gain():
    weekly = weekly_maps_from_per_week_points(
        ["a", "b"], "c",
        [{"a": 10.0, "b": 9.0}, {"a": 10.0, "b": 9.0}],
        [20.0, 18.0],
        candidate_bye_weeks=[False, True],
    )
    ev = evaluate_pickup(
        "c", "WR", ["a", "b"], weekly,
        {"a": "WR", "b": "WR", "c": "WR"}, ["WR", "FLEX", "BN", "BN", "BN"],
        droppable_pids=["a", "b"],
    )
    assert ev.horizon_gain == ev.week_gain
