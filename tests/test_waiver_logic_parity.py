"""Waiver-logic parity: the three leftovers from the #2634 gate fix.

1. The Waiver Wire Targets card ranked by the old _waiver_pickup_score
   ("up N spots this week") behind a value-only gate; it now renders
   rows from the shared v2 selector, like Next Steps.
2. The digest adapters (portfolio select_waiver_add, email
   recommend_waivers) called the selector without weekly availability
   or per-week maps, so calendar need never activated there; callers
   now build the shared opportunity inputs.
3. evaluate_pickup costed a drop on week-0 points only, so a bye-week
   cover claim could name the bye-week starter as the drop. Drop
   costing is horizon-aware now.
"""
import pytest

pd = pytest.importorskip("pandas")

import app as app_module  # noqa: E402
from utils.digest import recommend_waivers, select_waiver_add  # noqa: E402
from utils.waiver_lineup import evaluate_pickup  # noqa: E402
from utils.waivers import selector_opportunity_inputs  # noqa: E402


# ---------------------------------------------------------------------------
# Item 3: horizon-aware drop costing (pure evaluator)
# ---------------------------------------------------------------------------

_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN", "BN"]
_POS = {
    "allen": "QB", "backup": "QB", "cover": "QB",
    "rb1": "RB", "rb2": "RB", "flexrb": "RB", "scrb": "RB",
    "wr1": "WR", "wr2": "WR", "scwr": "WR", "te1": "TE",
}
_ROSTER10 = ["allen", "backup", "rb1", "rb2", "wr1", "wr2",
             "te1", "flexrb", "scrb", "scwr"]
_WK0 = {"allen": 0.0, "backup": 18.0, "rb1": 14.0, "rb2": 12.0,
        "wr1": 15.0, "wr2": 13.0, "te1": 9.0, "flexrb": 10.0,
        "scrb": 2.0, "scwr": 1.0, "cover": 20.0}
_WK_LATER = {**_WK0, "allen": 25.0}


def _wk(pts):
    return {"pts_map": dict(pts), "covered": True}


def test_bye_week_starter_is_not_the_named_drop():
    # Allen is on bye in week 0 (0 pts, the lowest on the roster) and is
    # the starter every other week (25). The claim covers his bye week.
    # Week-0-only costing names Allen as the drop (he loses the week-0
    # tie-break at 0 points); horizon costing must cut a bench scrub.
    weekly = [_wk(_WK0), _wk(_WK_LATER), _wk(_WK_LATER), _wk(_WK_LATER)]
    ev = evaluate_pickup("cover", "QB", _ROSTER10, weekly, dict(_POS),
                         _SLOTS, droppable_pids=list(_ROSTER10))
    assert ev.outcome == "add_drop"
    assert ev.drop_pid != "allen"
    assert ev.drop_pid == "scwr"  # never starts, cheapest across the window
    assert ev.drop_points == pytest.approx(1.0)
    assert ev.week_gain == pytest.approx(2.0)   # 93 vs 91 in the bye week
    assert ev.horizon_gain == pytest.approx(2.0)  # no phantom later losses
    assert ev.horizon_weeks_covered == 4


def test_horizon_drop_can_differ_from_week0_drop():
    # Same shape, but the scrub is startable in week 0 only (flexrb is
    # the week-0 FLEX and a zero after): the week-0-cheapest cut that
    # also stays cheapest across the window still wins.
    wk0 = {**_WK0, "scwr": 11.0}   # scwr starts at FLEX in week 0
    later = {**_WK_LATER, "scwr": 0.0}
    weekly = [_wk(wk0), _wk(later), _wk(later), _wk(later)]
    ev = evaluate_pickup("cover", "QB", _ROSTER10, weekly, dict(_POS),
                         _SLOTS, droppable_pids=list(_ROSTER10))
    assert ev.outcome == "add_drop"
    assert ev.drop_pid != "allen"
    # scrb (2 pts, never starts) is the cheapest cut across the window
    # even though scwr out-points him in week 0.
    assert ev.drop_pid == "scrb"


# ---------------------------------------------------------------------------
# Shared league fixture (mirrors tests/test_next_steps_waiver_gate.py)
# ---------------------------------------------------------------------------

TEAMS = {
    "q1": "BUF", "q2": "NYJ", "q3": "CLE",
    "rb1": "DET", "rb2": "SF", "rb3": "SEA", "rb4": "DAL",
    "wr1": "CIN", "wr2": "MIA", "wr3": "LAR", "wr4": "DEN", "wr5": "PHI",
    "te1": "KC", "te2": "BAL",
    "mayf": "TB", "wrb": "HOU", "wwr": "ATL",
    "o1": "GB", "o2": "MIN", "o3": "CHI",
}
NAMES = {
    "q1": "Josh Allen", "q2": "Justin Fields", "q3": "Sam Howell",
    "rb1": "Jahmyr Gibbs", "rb2": "Breece Hall", "rb3": "Kenneth Walker",
    "rb4": "Tony Pollard",
    "wr1": "Ja'Marr Chase", "wr2": "Tyreek Hill", "wr3": "Puka Nacua",
    "wr4": "Courtland Sutton", "wr5": "Diontae Johnson",
    "te1": "Travis Kelce", "te2": "David Njoku",
    "mayf": "Baker Mayfield", "wrb": "Rising Back", "wwr": "Wire Wideout",
    "o1": "Opp One", "o2": "Opp Two", "o3": "Opp Three",
}
POS = {
    "q1": "QB", "q2": "QB", "q3": "QB",
    "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB",
    "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR", "wr5": "WR",
    "te1": "TE", "te2": "TE",
    "mayf": "QB", "wrb": "RB", "wwr": "WR",
    "o1": "RB", "o2": "WR", "o3": "TE",
}
VALUES = {
    "q1": 5200, "q2": 1800, "q3": 950,
    "rb1": 4600, "rb2": 2400, "rb3": 1200, "rb4": 700,
    "wr1": 4200, "wr2": 3600, "wr3": 2800, "wr4": 1500, "wr5": 650,
    "te1": 2600, "te2": 500,
    "mayf": 2600, "wrb": 2100, "wwr": 900,
    "o1": 3000, "o2": 3000, "o3": 3000,
}
LABELS = {"mayf": "QB8", "wrb": "RB18", "wwr": "WR48",
          "q1": "QB1", "q2": "QB14", "q3": "QB27"}
PROJ = {
    "q1": 24.0, "q2": 19.0, "q3": 15.0,
    "rb1": 16.0, "rb2": 11.0, "rb3": 8.0, "rb4": 5.0,
    "wr1": 15.0, "wr2": 14.0, "wr3": 12.5, "wr4": 9.0, "wr5": 5.0,
    "te1": 10.0, "te2": 4.0,
    "mayf": 20.5, "wrb": 13.5, "wwr": 8.0,
}
ROSTER = ["q1", "q2", "q3", "rb1", "rb2", "rb3", "rb4",
          "wr1", "wr2", "wr3", "wr4", "wr5", "te1", "te2"]
ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                    "BN", "BN", "BN", "BN", "BN", "BN"]
OWNED = set(ROSTER) | {"o1", "o2", "o3"}


def _schedule(bye_teams=()):
    teams = sorted({t for t in TEAMS.values() if t not in bye_teams})
    return [{"away": teams[i], "home": teams[i + 1]}
            for i in range(0, len(teams) - 1, 2)]


def _fixture(monkeypatch, *, starter_bye=False, backup_out=False,
             roster=None, wire=("mayf", "wrb", "wwr")):
    pidx = {
        pid: {"position": POS[pid], "pos": POS[pid], "team": TEAMS[pid],
              "full_name": NAMES[pid],
              "injury_status": "Out" if (backup_out and pid == "q2") else ""}
        for pid in TEAMS
    }
    rows = [{
        "id": pid, "name": NAMES[pid], "position": POS[pid],
        "team": TEAMS[pid], "value": VALUES[pid], "age": 31 if pid == "mayf" else 25,
        "pos_rank_label": LABELS.get(pid, ""),
        "rank_change_7d": 34 if pid == "mayf" else 112 if pid == "wrb" else 0,
    } for pid in TEAMS if pid in (roster or ROSTER) or pid in wire]
    ctx = {
        "platform": "sleeper", "league_id": "L1", "season": 2026,
        "current_season": 2026, "current_week": 7,
        "league": {"type": 2}, "settings": {"type": 2},
        "scoring_settings": {}, "raw_scoring_settings": {},
        "players_index": pidx,
        "roster_positions": list(ROSTER_POSITIONS),
        "rosters": [
            {"roster_id": 1, "players": list(roster or ROSTER)},
            {"roster_id": 2, "players": ["o1", "o2", "o3"]},
        ],
        "viewer": {"viewer_roster_id": 1},
    }

    def fake_proj(season, week):
        out = {}
        for pid, pts in PROJ.items():
            if starter_bye and pid == "q1" and week == 7:
                continue
            if backup_out and pid == "q2":
                continue
            out[pid] = pts
        return out

    def fake_sched(season, week):
        if starter_bye and week == 7:
            return _schedule(bye_teams={"BUF"})
        return _schedule()

    monkeypatch.setattr("utils.utils.load_week_projection", fake_proj)
    monkeypatch.setattr("utils.utils.load_week_schedule", fake_sched)
    return ctx, rows


def _opportunity(ctx, roster):
    pidx = ctx["players_index"]
    return selector_opportunity_inputs(
        season=2026, week=7, roster_pids=list(roster),
        candidate_pids=["mayf", "wrb", "wwr"],
        positions_by_pid={pid: POS[pid] for pid in TEAMS},
        players_index=pidx, raw_scoring_settings={},
    )


# ---------------------------------------------------------------------------
# Item 1: the Waiver Wire Targets card runs the v2 gate
# ---------------------------------------------------------------------------

def test_card_rows_come_from_the_gate_not_the_risers(monkeypatch):
    ctx, rows = _fixture(monkeypatch)
    html_rows, gate_emptied = app_module._build_waiver_targets_rows(ctx, rows)
    assert gate_emptied is False
    assert "Rising Back" in html_rows
    assert "lineup pts" in html_rows
    # The covered QB room yields no QB row, and value movement is
    # never the reason on the card.
    assert "Mayfield" not in html_rows
    assert "spots" not in html_rows


def test_card_gate_emptied_hides_even_with_draft_links(monkeypatch):
    # Wire holds only the covered-room QB: pool exists, gate clears all.
    ctx, rows = _fixture(monkeypatch, wire=("mayf",))
    html_rows, gate_emptied = app_module._build_waiver_targets_rows(ctx, rows)
    assert html_rows == ""
    assert gate_emptied is True
    card = app_module._render_do_next_waiver_card(
        ctx, rows, platform="sleeper", season=2026, league_id="L1",
        draft_prep_hrefs=[("Draft Room", "/draft-room")],
    )
    assert card == ""


def test_card_no_viewer_no_rows(monkeypatch):
    ctx, rows = _fixture(monkeypatch)
    ctx["viewer"] = {}
    html_rows, gate_emptied = app_module._build_waiver_targets_rows(ctx, rows)
    assert html_rows == ""
    assert gate_emptied is False


# ---------------------------------------------------------------------------
# Item 2: digest adapters finally see availability + per-week maps
# ---------------------------------------------------------------------------

def test_recommend_waivers_calendar_need_activates(monkeypatch):
    # QB hole: starter on bye in Week 7, backup Out, only two QBs kept.
    roster = [p for p in ROSTER if p != "q3"]
    ctx, rows = _fixture(monkeypatch, starter_bye=True, backup_out=True,
                         roster=roster)
    opp = _opportunity(ctx, roster)
    assert opp["weekly_availability"]  # schedules + injuries were read
    targets = recommend_waivers(
        rows, OWNED, roster_players=roster,
        roster_positions=ROSTER_POSITIONS, pidx=ctx["players_index"],
        fmt={}, limit=3, min_score=0.0,
        weekly_availability=opp["weekly_availability"],
        lineup_weekly=opp["weekly_maps"],
        weekly_pts_by_pid=opp["weekly_pts_by_pid"],
    )
    by_pid = {t["player_id"]: t for t in targets}
    assert "mayf" in by_pid
    assert by_pid["mayf"]["starter_gap"] >= 1  # the calendar hole registered
    assert by_pid["mayf"]["outcome"] in ("add", "add_drop")


def test_recommend_waivers_covered_room_stays_quiet_with_inputs(monkeypatch):
    ctx, rows = _fixture(monkeypatch)
    opp = _opportunity(ctx, ROSTER)
    targets = recommend_waivers(
        rows, OWNED, roster_players=ROSTER,
        roster_positions=ROSTER_POSITIONS, pidx=ctx["players_index"],
        fmt={}, limit=3, min_score=0.0,
        weekly_availability=opp["weekly_availability"],
        lineup_weekly=opp["weekly_maps"],
        weekly_pts_by_pid=opp["weekly_pts_by_pid"],
    )
    ids = {t["player_id"] for t in targets}
    assert "mayf" not in ids
    assert "wrb" in ids
    # Control: the same call WITHOUT the opportunity inputs still waves
    # the covered-room QB in on value gates alone (the old parity gap).
    bare = recommend_waivers(
        rows, OWNED, roster_players=ROSTER,
        roster_positions=ROSTER_POSITIONS, pidx=ctx["players_index"],
        fmt={}, limit=3, min_score=0.0,
    )
    assert "mayf" in {t["player_id"] for t in bare}


def test_gather_digest_actions_builds_the_inputs(monkeypatch):
    # End-to-end through the email digest's action gatherer: with the
    # league scoring supplied, gather builds the opportunity inputs
    # itself, so the covered-room QB never reaches the digest targets.
    from utils.digest import gather_digest_action_items
    ctx, rows = _fixture(monkeypatch)
    items = gather_digest_action_items(
        platform="sleeper", season=2026, league_id="L1",
        roster={"players": list(ROSTER), "starters": []},
        pidx=ctx["players_index"], base_url="https://example.test",
        fmt={}, owned_ids=set(OWNED), model_rows=rows,
        nfl_state={"season": 2026, "week": 7, "season_type": "reg"},
        nfl_players={}, teams_playing=set(TEAMS.values()),
        proj_map={}, roster_positions=ROSTER_POSITIONS,
        raw_scoring_settings={"rec": 1.0, "pass_td": 4.0},
    )
    waiver_items = [i for i in items if i.get("kind") == "waiver"]
    assert waiver_items
    ids = {t["player_id"] for t in waiver_items[0]["targets"]}
    assert "mayf" not in ids
    assert "wrb" in ids


def test_select_waiver_add_uses_the_same_gate(monkeypatch):
    ctx, rows = _fixture(monkeypatch)
    opp = _opportunity(ctx, ROSTER)
    hit = select_waiver_add(
        rows, OWNED, value_key="value",
        roster_players=ROSTER, roster_positions=ROSTER_POSITIONS,
        pidx=ctx["players_index"],
        weekly_availability=opp["weekly_availability"],
        lineup_weekly=opp["weekly_maps"],
        weekly_pts_by_pid=opp["weekly_pts_by_pid"],
        forward_pts_by_pid=opp["forward_pts_by_pid"],
    )
    assert hit is not None
    assert hit["player_id"] == "wrb"  # the real lineup gain, not the QB

    roster = [p for p in ROSTER if p != "q3"]
    ctx2, rows2 = _fixture(monkeypatch, starter_bye=True, backup_out=True,
                           roster=roster, wire=("mayf",))
    opp2 = _opportunity(ctx2, roster)
    hit2 = select_waiver_add(
        rows2, OWNED, value_key="value",
        roster_players=roster, roster_positions=ROSTER_POSITIONS,
        pidx=ctx2["players_index"],
        weekly_availability=opp2["weekly_availability"],
        lineup_weekly=opp2["weekly_maps"],
        weekly_pts_by_pid=opp2["weekly_pts_by_pid"],
        forward_pts_by_pid=opp2["forward_pts_by_pid"],
    )
    assert hit2 is not None
    assert hit2["player_id"] == "mayf"
    assert hit2["starter_gap"] >= 1  # calendar hole, not raw value
