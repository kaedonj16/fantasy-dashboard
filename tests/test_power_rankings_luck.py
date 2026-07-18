"""Guards the power-ranking accuracy upgrades in build_power_rankings_context:

  1. The win term is luck-adjusted (blended toward all-play), so a team that
     scores well but lost the schedule lottery isn't punished as if it were bad,
     and a team that won on a soft schedule isn't over-credited.
  2. The value term is the top-8 win-now (redraft) starter total, not a
     whole-roster dynasty average.

The heavy roster-summary/grade helpers are stubbed so this stays a pure-math
test with no Flask/pandas-provider dependencies (pandas itself is fine).
"""
import pytest

# pandas isn't installed in the lightweight CI base suite; skip there.
pd = pytest.importorskip("pandas")

import dashboard_services.ai.context_builders as cb


@pytest.fixture(autouse=True)
def _stub_heavy_helpers(monkeypatch):
    monkeypatch.setattr(cb, "summarize_roster_players", lambda **k: [])
    monkeypatch.setattr(cb, "detect_team_direction", lambda a, b: "balanced")
    monkeypatch.setattr(cb, "group_position_strength", lambda x: {})
    monkeypatch.setattr(cb, "calculate_roster_grade", lambda *a, **k: {"win_window": "balanced"})
    monkeypatch.setattr(cb, "build_model_value_lookup", lambda tbl, is_sf=False: {r["player_id"]: r for r in tbl})


def _mv(pid, pos, val, redraft):
    return {"player_id": pid, "name": pid, "position": pos, "value": val, "redraft_value_1qb": redraft}


def _ctx():
    # Three teams. Unlucky outscores everyone every week but is 0-3; Lucky is the
    # mid scorer but 3-0; Cellar is last. all-play uses only the scores.
    mvt = [_mv("u1", "RB", 2000, 1800), _mv("l1", "RB", 2000, 1800), _mv("c1", "RB", 500, 400)]
    rows = []
    for wk, (pu, pl, pc) in enumerate([(130, 100, 60), (128, 101, 62), (126, 99, 58)], start=1):
        rows.append({"week": wk, "roster_id": "U", "points": pu, "finalized": True})
        rows.append({"week": wk, "roster_id": "L", "points": pl, "finalized": True})
        rows.append({"week": wk, "roster_id": "C", "points": pc, "finalized": True})
    return {
        "rosters": [
            {"roster_id": "U", "players": ["u1"], "settings": {"wins": 0, "losses": 3, "fpts": 384}},
            {"roster_id": "L", "players": ["l1"], "settings": {"wins": 3, "losses": 0, "fpts": 300}},
            {"roster_id": "C", "players": ["c1"], "settings": {"wins": 0, "losses": 3, "fpts": 180}},
        ],
        "standings_map": {"U": {"PF": 384}, "L": {"PF": 300}, "C": {"PF": 180}},
        "roster_map": {"U": "Unlucky", "L": "Lucky", "C": "Cellar"},
        "model_value_table": mvt,
        "picks_by_roster": {},
        "df_weekly": pd.DataFrame(rows),
        "league_type": "1qb",
    }


def _by_name(res):
    return {t["team_name"]: t for t in res["teams"]}


def test_all_play_replaces_raw_record():
    teams = _by_name(cb.build_power_rankings_context(_ctx()))
    unlucky, lucky = teams["Unlucky"], teams["Lucky"]
    # Raw record says Lucky is perfect and Unlucky winless...
    assert lucky["win_pct"] == 1.0
    assert unlucky["win_pct"] == 0.0
    # ...but all-play knows Unlucky is the strongest scorer every week.
    assert unlucky["all_play_pct"] == 1.0
    assert lucky["all_play_pct"] == 0.5
    # The luck-adjusted win term flips them: the high scorer outranks the
    # schedule-lucky team on the win dimension.
    assert unlucky["luck_adj_win"] > lucky["luck_adj_win"]


def test_luckier_team_does_not_top_the_board():
    teams = cb.build_power_rankings_context(_ctx())["teams"]  # sorted best-first
    # The genuinely strongest team (Unlucky) should rank #1 despite its 0-3
    # record, not the 3-0 schedule beneficiary.
    assert teams[0]["team_name"] == "Unlucky"


def test_starter_value_is_top8_redraft():
    teams = _by_name(cb.build_power_rankings_context(_ctx()))
    # Single-player rosters: starter value is just that player's redraft value.
    assert teams["Unlucky"]["starter_value"] == 1800.0
    assert teams["Cellar"]["starter_value"] == 400.0


def _momentum_ctx():
    """Two teams with identical season résumés (same PF, record, value) but
    mirror-image recent form: HotNow was cold early and is red-hot lately, ColdNow
    the reverse. A cellar team anchors the field."""
    mvt = [
        {"player_id": "a", "position": "RB", "value": 1500, "redraft_value_1qb": 1500},
        {"player_id": "b", "position": "RB", "value": 1500, "redraft_value_1qb": 1500},
    ]
    a = [80, 80, 80, 120, 120, 120]
    b = [120, 120, 120, 80, 80, 80]
    c = [70, 70, 70, 70, 70, 70]
    rows = []
    for wk in range(6):
        rows.append({"week": wk + 1, "roster_id": "A", "points": a[wk], "finalized": True})
        rows.append({"week": wk + 1, "roster_id": "B", "points": b[wk], "finalized": True})
        rows.append({"week": wk + 1, "roster_id": "C", "points": c[wk], "finalized": True})
    return {
        "rosters": [
            {"roster_id": "A", "players": ["a"], "settings": {"wins": 3, "losses": 3, "fpts": 600}},
            {"roster_id": "B", "players": ["b"], "settings": {"wins": 3, "losses": 3, "fpts": 600}},
            {"roster_id": "C", "players": [], "settings": {"wins": 0, "losses": 6, "fpts": 420}},
        ],
        "standings_map": {"A": {"PF": 600}, "B": {"PF": 600}, "C": {"PF": 420}},
        "roster_map": {"A": "HotNow", "B": "ColdNow", "C": "Cellar"},
        "model_value_table": mvt,
        "picks_by_roster": {},
        "df_weekly": pd.DataFrame(rows),
        "league_type": "1qb",
    }


def test_momentum_breaks_a_tie_between_identical_resumes():
    teams = _by_name(cb.build_power_rankings_context(_momentum_ctx()))
    hot, cold = teams["HotNow"], teams["ColdNow"]
    # Same PF/record/value components...
    assert hot["power_components"]["pf"] == cold["power_components"]["pf"]
    assert hot["power_components"]["record"] == cold["power_components"]["record"]
    # ...but recent form separates them.
    assert hot["momentum"] > 0 and cold["momentum"] < 0
    assert hot["momentum_label"] == "Heating up"
    assert cold["momentum_label"] == "Cooling off"
    assert hot["power_score"] > cold["power_score"]


def test_power_components_are_exposed():
    teams = cb.build_power_rankings_context(_momentum_ctx())["teams"]
    for t in teams:
        assert set(t["power_components"]) == {"pf", "record", "value", "momentum", "consistency", "sos"}


def _sos_ctx():
    """Four teams, four weeks, fixed schedule: TeamTough always plays the Strong
    team, TeamCake always plays the Cellar. Tough and Cake post identical scores,
    so only strength of schedule separates them."""
    mvt = [{"player_id": p, "position": "RB", "value": 1500, "redraft_value_1qb": 1500}
           for p in ("t", "k", "s", "c")]
    sc = {"T": [100] * 4, "K": [100] * 4, "S": [130] * 4, "C": [60] * 4}
    rows = []
    for wk in range(4):
        for mid, (x, y) in enumerate([("T", "S"), ("K", "C")]):
            rows.append({"week": wk + 1, "matchup_id": mid, "roster_id": x, "points": sc[x][wk], "finalized": True})
            rows.append({"week": wk + 1, "matchup_id": mid, "roster_id": y, "points": sc[y][wk], "finalized": True})
    names = {"T": "TeamTough", "K": "TeamCake", "S": "Strong", "C": "Cellar"}
    return {
        "rosters": [{"roster_id": t, "players": ["t"], "settings": {"wins": 2, "losses": 2, "fpts": sum(sc[t])}}
                    for t in ("T", "K", "S", "C")],
        "standings_map": {t: {"PF": sum(sc[t])} for t in sc},
        "roster_map": names,
        "model_value_table": mvt,
        "picks_by_roster": {},
        "df_weekly": pd.DataFrame(rows),
        "league_type": "1qb",
    }


def test_strength_of_schedule_breaks_a_tie():
    teams = _by_name(cb.build_power_rankings_context(_sos_ctx()))
    tough, cake = teams["TeamTough"], teams["TeamCake"]
    # Identical scoring résumé...
    assert tough["power_components"]["pf"] == cake["power_components"]["pf"]
    # ...but the tougher schedule is recognized and rewarded.
    assert tough["sos"] > cake["sos"]
    assert tough["sos_label"] == "Brutal"
    assert cake["sos_label"] == "Soft"
    assert tough["power_score"] > cake["power_score"]
