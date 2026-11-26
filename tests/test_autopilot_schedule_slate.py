"""Schedule-aware Autopilot: slate-derived windows, due-gating,
schedule locks in planning and execution.

Fixture slates are the real 2026 nflverse shapes (games.csv columns
season/game_type/week/gameday/gametime/away_team/home_team):

- Week 12 (Thanksgiving): Wed 20:00 game, Thu 13:00/16:30/20:20,
  Fri 15:00 Black Friday game, Sun slate, Mon night.
- Week 14 (a normal week): Thu 20:15, Sun 13:00 wave + afternoon wave
  + 20:20, Mon 20:15.
- Week 5 (international Sunday): Sun 09:30 London game + the usual
  Sunday waves.
- Week 15 (Saturday week): Sat 17:00 + 20:20 doubleheader.
"""
import copy
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("flask")
# The schedule/plan/execute import chains pull pandas.
pytest.importorskip("pandas")

from dashboard_services import autopilot_schedule as sched
from dashboard_services import autopilot_store as store

ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def _row(week, gameday, gametime, away, home):
    return {
        "season": "2026", "game_type": "REG", "week": str(week),
        "gameday": gameday, "gametime": gametime,
        "away_team": away, "home_team": home,
    }


THANKSGIVING = [
    _row(12, "2026-11-25", "20:00", "GB", "LA"),
    _row(12, "2026-11-26", "13:00", "CHI", "DET"),
    _row(12, "2026-11-26", "16:30", "PHI", "DAL"),
    _row(12, "2026-11-26", "20:20", "KC", "BUF"),
    _row(12, "2026-11-27", "15:00", "DEN", "PIT"),
    _row(12, "2026-11-29", "13:00", "NO", "CIN"),
    _row(12, "2026-11-29", "13:00", "LV", "CLE"),
    _row(12, "2026-11-29", "13:00", "BAL", "HOU"),
    _row(12, "2026-11-29", "13:00", "NYG", "IND"),
    _row(12, "2026-11-29", "13:00", "NYJ", "MIA"),
    _row(12, "2026-11-29", "13:00", "ATL", "MIN"),
    _row(12, "2026-11-29", "16:05", "TEN", "JAX"),
    _row(12, "2026-11-29", "16:25", "WAS", "ARI"),
    _row(12, "2026-11-29", "16:25", "SEA", "SF"),
    _row(12, "2026-11-29", "20:20", "NE", "LAC"),
    _row(12, "2026-11-30", "20:15", "CAR", "TB"),
]

NORMAL_WEEK = [
    _row(14, "2026-12-10", "20:15", "MIN", "NE"),
    _row(14, "2026-12-13", "13:00", "TB", "BAL"),
    _row(14, "2026-12-13", "13:00", "NO", "CAR"),
    _row(14, "2026-12-13", "13:00", "ATL", "CLE"),
    _row(14, "2026-12-13", "13:00", "TEN", "DET"),
    _row(14, "2026-12-13", "13:00", "CHI", "MIA"),
    _row(14, "2026-12-13", "13:00", "DEN", "NYJ"),
    _row(14, "2026-12-13", "13:00", "IND", "PHI"),
    _row(14, "2026-12-13", "13:00", "HOU", "WAS"),
    _row(14, "2026-12-13", "16:05", "LAC", "LV"),
    _row(14, "2026-12-13", "16:25", "KC", "CIN"),
    _row(14, "2026-12-13", "16:25", "NYG", "SEA"),
    _row(14, "2026-12-13", "16:25", "LA", "SF"),
    _row(14, "2026-12-13", "20:20", "BUF", "GB"),
    _row(14, "2026-12-14", "20:15", "PIT", "JAX"),
]

INTERNATIONAL_WEEK = [
    _row(5, "2026-10-08", "20:15", "TB", "DAL"),
    _row(5, "2026-10-11", "09:30", "PHI", "JAX"),
    _row(5, "2026-10-11", "13:00", "CHI", "GB"),
    _row(5, "2026-10-11", "13:00", "CIN", "MIA"),
    _row(5, "2026-10-11", "13:00", "LV", "NE"),
    _row(5, "2026-10-11", "13:00", "MIN", "NO"),
    _row(5, "2026-10-11", "13:00", "CLE", "NYJ"),
    _row(5, "2026-10-11", "13:00", "IND", "PIT"),
    _row(5, "2026-10-11", "13:00", "HOU", "TEN"),
    _row(5, "2026-10-11", "13:00", "NYG", "WAS"),
    _row(5, "2026-10-11", "16:05", "DEN", "LAC"),
    _row(5, "2026-10-11", "16:25", "DET", "ARI"),
    _row(5, "2026-10-11", "16:25", "SF", "SEA"),
    _row(5, "2026-10-11", "20:20", "BAL", "ATL"),
    _row(5, "2026-10-12", "20:15", "BUF", "LA"),
]

SATURDAY_WEEK = [
    _row(15, "2026-12-17", "20:15", "SF", "LAC"),
    _row(15, "2026-12-19", "17:00", "SEA", "PHI"),
    _row(15, "2026-12-19", "20:20", "CHI", "BUF"),
    _row(15, "2026-12-20", "13:00", "CIN", "CAR"),
    _row(15, "2026-12-20", "13:00", "MIA", "GB"),
    _row(15, "2026-12-20", "13:00", "JAX", "HOU"),
    _row(15, "2026-12-20", "16:05", "NYJ", "ARI"),
    _row(15, "2026-12-20", "16:25", "DAL", "LA"),
    _row(15, "2026-12-20", "20:20", "DET", "MIN"),
    _row(15, "2026-12-21", "20:15", "NE", "KC"),
]


def _starts(windows):
    return [w["start"] for w in windows]


# ── Due windows per slate ──────────────────────────────────────────────

def test_due_windows_normal_week_is_the_familiar_two():
    wins = sched.due_windows(2026, 14, rows=NORMAL_WEEK)
    assert _starts(wins) == [_et(2026, 12, 10, 19, 0), _et(2026, 12, 13, 11, 30)]
    # The Sunday window serves the 13:00 slate first.
    assert wins[1]["kickoff"] == _et(2026, 12, 13, 13, 0)


def test_due_windows_thanksgiving_week():
    wins = sched.due_windows(2026, 12, rows=THANKSGIVING)
    assert _starts(wins) == [
        _et(2026, 11, 25, 19, 0),   # before the Wednesday 20:00 game
        _et(2026, 11, 26, 12, 0),   # before the Thu 13:00 kickoff
        _et(2026, 11, 26, 19, 0),   # familiar Thu-evening pass (20:20 game)
        _et(2026, 11, 27, 14, 0),   # before the Black Friday 15:00 game
        _et(2026, 11, 29, 11, 30),  # Sunday late-morning pass
    ]
    by_start = {w["start"]: w["kickoff"] for w in wins}
    assert by_start[_et(2026, 11, 26, 12, 0)] == _et(2026, 11, 26, 13, 0)
    assert by_start[_et(2026, 11, 26, 19, 0)] == _et(2026, 11, 26, 20, 20)
    assert by_start[_et(2026, 11, 27, 14, 0)] == _et(2026, 11, 27, 15, 0)


def test_due_windows_international_sunday_gets_its_own_early_window():
    wins = sched.due_windows(2026, 5, rows=INTERNATIONAL_WEEK)
    assert _starts(wins) == [
        _et(2026, 10, 8, 20, 0),    # Thu-evening pass (EDT anchor)
        _et(2026, 10, 11, 8, 30),   # before the 09:30 London kickoff
        _et(2026, 10, 11, 12, 30),  # main slate keeps its pass (EDT)
    ]


def test_due_windows_saturday_week():
    wins = sched.due_windows(2026, 15, rows=SATURDAY_WEEK)
    assert _starts(wins) == [
        _et(2026, 12, 17, 19, 0),   # Thursday pass
        _et(2026, 12, 19, 16, 0),   # before the Saturday 17:00 game
        _et(2026, 12, 20, 11, 30),  # Sunday pass
    ]


def test_due_windows_empty_slate_is_empty():
    assert sched.due_windows(2026, 12, rows=[]) == []


# ── Kickoff map + schedule locks ───────────────────────────────────────

def test_kickoffs_by_team_normalizes_nflverse_codes():
    by_team = sched.kickoffs_by_team(2026, 12, rows=THANKSGIVING)
    assert by_team["DET"] == _et(2026, 11, 26, 13, 0)
    assert by_team["LAR"] == _et(2026, 11, 25, 20, 0)  # nflverse "LA"
    assert by_team["PIT"] == _et(2026, 11, 27, 15, 0)
    assert by_team["JAX"] == _et(2026, 11, 29, 16, 5)  # nflverse "JAX"


def test_schedule_locked_teams_thanksgiving_morning():
    # Thu 03:15 ET: only the Wednesday game's teams have kicked off.
    locked = sched.schedule_locked_teams(
        2026, 12, _et(2026, 11, 26, 3, 15), rows=THANKSGIVING)
    assert locked == {"GB", "LAR"}
    # Thu 14:00 ET: the 13:00 game has kicked off too.
    locked = sched.schedule_locked_teams(
        2026, 12, _et(2026, 11, 26, 14, 0), rows=THANKSGIVING)
    assert locked == {"GB", "LAR", "CHI", "DET"}


# ── Due-gating in run_scheduled_runs ───────────────────────────────────

ACCOUNT = 42
LEAGUE = "887776065"
SEASON = 2026


@pytest.fixture
def gate(monkeypatch):
    g = {
        "passes": [],
        "settings": {"mode": "auto"},
        "plan_created_at": None,  # ET datetime of a stored plan, if any
        "exists_calls": [],
    }
    monkeypatch.setattr(sched, "_schedule_rows", lambda: list(THANKSGIVING))
    monkeypatch.setattr(sched, "_candidate_leagues", lambda: [
        {"account_id": ACCOUNT, "platform": "espn",
         "league_id": LEAGUE, "season": SEASON},
    ])

    def fake_pass(account_id, platform, league_id, season, *, transport=None):
        g["passes"].append(league_id)
        return {"outcome": "executed", "league_id": str(league_id)}

    monkeypatch.setattr(sched, "run_league_pass", fake_pass)
    monkeypatch.setattr(store, "get_settings", lambda *a: dict(g["settings"]))

    def fake_exists(account_id, platform, league_id, season, period, since,
                    *, statuses=None):
        g["exists_calls"].append({"period": period, "since": since,
                                  "statuses": statuses})
        created = g["plan_created_at"]
        return bool(created is not None and created >= since)

    monkeypatch.setattr(store, "plan_exists_since", fake_exists)
    return g


def test_gate_runs_when_window_due_and_unserved(gate):
    # Thu 12:05 EST: inside the noon window, nothing served yet.
    summary = sched.run_scheduled_runs(now=_utc(2026, 11, 26, 17, 5))
    assert gate["passes"] == [LEAGUE]
    assert summary["executed"] == 1
    call = gate["exists_calls"][0]
    assert call["period"] == 12
    assert call["since"] == _et(2026, 11, 26, 12, 0)
    # Auto leagues: only an executed/attempted plan counts as served.
    assert call["statuses"] == ("executed", "partial", "failed",
                                "already_applied")


def test_gate_skips_when_nothing_due(gate):
    # Thu 10:00 EST: the noon window has not opened.
    summary = sched.run_scheduled_runs(now=_utc(2026, 11, 26, 15, 0))
    assert gate["passes"] == []
    assert summary["not_due"] == 1
    assert summary["details"][0]["outcome"] == "not_due"


def test_gate_skips_after_kickoff_until_next_window(gate):
    # Thu 13:05 EST: the 13:00 game has kicked off; the noon window is
    # over and the evening window has not opened.
    summary = sched.run_scheduled_runs(now=_utc(2026, 11, 26, 18, 5))
    assert gate["passes"] == []
    assert summary["not_due"] == 1


def test_gate_skips_window_already_served(gate):
    gate["plan_created_at"] = _et(2026, 11, 26, 12, 1)
    summary = sched.run_scheduled_runs(now=_utc(2026, 11, 26, 17, 5))
    assert gate["passes"] == []
    assert summary["already_served"] == 1


def test_gate_second_window_not_served_by_first_windows_plan(gate):
    # A plan built in the noon window does not serve the 19:00 window.
    gate["plan_created_at"] = _et(2026, 11, 26, 12, 1)
    summary = sched.run_scheduled_runs(now=_utc(2026, 11, 27, 0, 5))
    assert gate["passes"] == [LEAGUE]
    assert summary["executed"] == 1


def test_gate_approve_league_any_plan_in_window_serves(gate):
    gate["settings"] = {"mode": "approve"}
    gate["plan_created_at"] = _et(2026, 11, 26, 12, 30)
    summary = sched.run_scheduled_runs(now=_utc(2026, 11, 26, 17, 45))
    assert gate["passes"] == []
    assert summary["already_served"] == 1
    assert gate["exists_calls"][0]["statuses"] is None


def test_gate_fails_open_without_slate(gate, monkeypatch):
    monkeypatch.setattr(sched, "_schedule_rows", lambda: [])
    # A plainly not-due clock time still runs: no slate, no gate.
    summary = sched.run_scheduled_runs(now=_utc(2026, 11, 26, 15, 0))
    assert gate["passes"] == [LEAGUE]
    assert summary["executed"] == 1


# ── Store: plan_exists_since ───────────────────────────────────────────

class _Cursor:
    def __init__(self, one=None):
        self._one = one

    def fetchone(self):
        return self._one


class _Ctx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


def test_plan_exists_since_sql_shape(monkeypatch):
    calls = []

    class Conn:
        def execute(self, sql, params=None):
            calls.append((" ".join(str(sql).split()), list(params or [])))
            return _Cursor(one=(1,))

        def commit(self):
            pass

    monkeypatch.setattr(store, "init_autopilot_tables", lambda: None)
    monkeypatch.setattr(store, "_conn", lambda: _Ctx(Conn()))
    since = _et(2026, 11, 26, 12, 0)
    assert store.plan_exists_since(
        ACCOUNT, "espn", LEAGUE, SEASON, 12, since,
        statuses=("executed", "partial")) is True
    sql, params = calls[0]
    assert "FROM autopilot_plans" in sql
    assert "scoring_period = %s" in sql and "created_at >= %s" in sql
    assert "status IN (%s, %s)" in sql
    assert params[4] == 12 and params[5] == since
    assert params[6:] == ["executed", "partial"]


# ── Plan builder: schedule locks ───────────────────────────────────────

from dashboard_services import autopilot_plan as plan_mod

SWID = "{11111111-2222-3333-4444-555555555555}"
P_ACCOUNT = 42
P_LEAGUE = "887776065"
P_SEASON = 2026

CANON = {
    "101": "q1", "102": "rb1", "103": "rb2", "104": "wr1", "105": "wr2",
    "106": "te1", "107": "wr3", "108": "k1", "-16007": "KC",
    "110": "rb3", "111": "wr4", "112": "qb2", "113": "rb4", "114": "te2",
    "115": "wr5", "116": "wr6",
}
POS = {
    "q1": "QB", "rb1": "RB", "rb2": "RB", "rb3": "RB", "rb4": "RB",
    "wr1": "WR", "wr2": "WR", "wr3": "WR", "wr4": "WR", "wr5": "WR",
    "wr6": "WR", "te1": "TE", "te2": "TE", "k1": "K", "KC": "DEF",
    "qb2": "QB",
}
POS_ID = {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "K": 5, "DEF": 16}
PRO_TEAM = {"ATL": 1, "DET": 8, "KC": 12}
ELIGIBLE = {
    "QB": [0, 20], "RB": [2, 23, 20], "WR": [4, 23, 20], "TE": [6, 23, 20],
    "K": [17, 20], "DEF": [16, 20],
}
SCORES = {
    "q1": 20, "rb1": 8, "rb2": 15, "wr1": 14, "wr2": 13, "te1": 10,
    "wr3": 12, "k1": 8, "KC": 7, "rb3": 18, "wr4": 9, "qb2": 18,
    "rb4": 5, "te2": 6, "wr5": 11,
}
BASE_SLOTS = {
    101: 0, 102: 2, 103: 2, 104: 4, 105: 4, 106: 6, 107: 23, 108: 17,
    -16007: 16, 110: 20, 111: 20, 112: 20, 113: 20, 114: 20, 115: 20,
    116: 21,
}
SLOT_COUNTS = {"0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "17": 1, "16": 1,
               "20": 6, "21": 1}

# Thanksgiving-shaped kickoffs: DET kicked off Thu 13:00, ATL plays
# Sunday 13:00. KC (the DEF canon's team label) is irrelevant: entries
# resolve their team from proTeamId, and the DEF entry uses KC's id.
KICKOFFS = {"DET": _et(2026, 11, 26, 13, 0), "ATL": _et(2026, 11, 29, 13, 0),
            "KC": _et(2026, 11, 26, 20, 20)}


def _payload(pro_teams):
    entries = []
    for pid, slot in BASE_SLOTS.items():
        canon = CANON[str(pid)]
        pos = POS[canon]
        entries.append({
            "playerId": pid,
            "lineupSlotId": slot,
            "lineupLocked": False,
            "playerPoolEntry": {
                "id": pid,
                "lineupSlotId": slot,
                "lineupLocked": False,
                "player": {
                    "id": pid,
                    "fullName": f"Player {canon}",
                    "eligibleSlots": ELIGIBLE[pos],
                    "defaultPositionId": POS_ID[pos],
                    "injuryStatus": "ACTIVE",
                    "proTeamId": PRO_TEAM[pro_teams.get(pid, "ATL")],
                },
            },
        })
    return {
        "id": int(P_LEAGUE),
        "scoringPeriodId": 12,
        "teams": [{
            "id": 2, "name": "Cook and Ball Torture", "owners": [SWID],
            "roster": {"entries": entries},
        }],
        "settings": {
            "scoringSettings": {"scoringItems": []},
            "rosterSettings": {"lineupSlotCounts": SLOT_COUNTS},
        },
    }


def _bundles(scores=None):
    scores = dict(SCORES if scores is None else scores)
    return {
        canon: {
            "pos": POS[canon], "proj_pts": float(score),
            "recent_ppg": float(score), "season_ppg": float(score),
            "last4_ppg": float(score), "played_games": 8,
            "on_bye": False, "injury_status": None,
            "team": "ATL", "opponent": "MIN",
        }
        for canon, score in scores.items()
    }


class _PlanTransport:
    def __init__(self, payload):
        self.payload = payload

    def get_league(self, season, league_id, params):
        return copy.deepcopy(self.payload)


@pytest.fixture
def plan_harness(monkeypatch):
    h = {"settings": dict(store.DEFAULT_SETTINGS, mode="approve"),
         "saved_plans": [], "bundles": _bundles()}
    monkeypatch.setattr(
        "dashboard_services.accounts.get_espn_league_credentials",
        lambda *a: {"espn_s2": "s2", "swid": SWID})
    monkeypatch.setattr(
        "dashboard_services.accounts.get_any_espn_account_credentials",
        lambda *a: {"espn_s2": "s2", "swid": SWID})
    monkeypatch.setattr(
        "dashboard_services.providers.espn_api._espn_to_canon_cached",
        lambda: {})
    monkeypatch.setattr(
        "dashboard_services.providers.espn_api.resolve_espn_player_id",
        lambda pid, m, player=None: CANON.get(str(pid)))
    monkeypatch.setattr(
        "data_building.start_score_bundle.load_start_score_bundles_for_scoring",
        lambda season, week, scoring, pids: {
            p: h["bundles"][p] for p in pids if p in h["bundles"]})
    monkeypatch.setattr(plan_mod, "_def_rank", lambda *a: (None, 32))
    monkeypatch.setattr(
        "dashboard_services.autopilot_schedule.kickoffs_by_team",
        lambda *a, **k: dict(KICKOFFS))
    monkeypatch.setattr(store, "get_settings", lambda *a: dict(h["settings"]))
    monkeypatch.setattr(store, "get_kill_switch", lambda *a: False)

    def fake_save_plan(account_id, platform, league_id, season, plan):
        saved = dict(plan, id=100 + len(h["saved_plans"]), status="open",
                     account_id=account_id, platform=platform,
                     league_id=league_id, season=season)
        h["saved_plans"].append(saved)
        return saved

    monkeypatch.setattr(store, "save_plan", fake_save_plan)
    monkeypatch.setattr(store, "record_audit", lambda *a, **k: None)
    return h


def _build_plan(h, payload, now):
    return plan_mod.build_plan(
        P_ACCOUNT, "espn", P_LEAGUE, P_SEASON,
        transport=_PlanTransport(payload), now=now)


def test_plan_never_promotes_a_player_whose_game_kicked_off(plan_harness):
    # rb3 (110) is the standard +10 promotion, but he plays for DET,
    # whose Thanksgiving game kicked off at 13:00. At 14:00 he is
    # schedule-locked: no plan may touch him, even with ESPN's flag off.
    payload = _payload({110: "DET"})
    out = _build_plan(plan_harness, payload, _et(2026, 11, 26, 14, 0))
    assert out["status"] == "no_moves"
    assert out["plan"]["swaps"] == []
    assert all(i["player_id"] != 110 for i in out["plan"]["lineup_items"])


def test_plan_promotes_him_before_his_kickoff(plan_harness):
    # Same roster at 12:00, an hour before DET kicks off: the normal
    # promotion is planned.
    payload = _payload({110: "DET"})
    out = _build_plan(plan_harness, payload, _et(2026, 11, 26, 12, 0))
    assert out["status"] == "planned"
    swap = out["plan"]["swaps"][0]
    assert [e["player_id"] for e in swap["in"]] == [110]
    assert [e["player_id"] for e in swap["out"]] == [102]


def test_plan_pins_a_schedule_locked_starter(plan_harness):
    # rb2 (103) is weak here and would normally be benched for rb3;
    # as a DET player after the 13:00 kickoff he is pinned instead.
    # The upgrade still happens through rb1's slot.
    plan_harness["bundles"] = _bundles({**SCORES, "rb2": 3})
    payload = _payload({103: "DET"})
    out = _build_plan(plan_harness, payload, _et(2026, 11, 26, 14, 0))
    plan = out["plan"]
    assert all(i["player_id"] != 103 for i in plan["lineup_items"])
    assert any(i["player_id"] == 110 for i in plan["lineup_items"])


# ── Execution: a newly locked group is skipped, the rest execute ───────

from dashboard_services import autopilot_execute as exec_mod
from dashboard_services import autopilot_preview

E_BASE = {1: 2, 2: 20, 3: 23, 4: 20}
E_TEAMS = {1: "ATL", 2: "ATL", 3: "DET", 4: "DET"}


def _swap_group(in_pid, in_from, in_to, out_pid):
    items = [
        {"player_id": in_pid, "from_slot": in_from, "to_slot": in_to},
        {"player_id": out_pid, "from_slot": in_to, "to_slot": in_from},
    ]
    return {
        "items": items,
        "in": [{"player_id": in_pid, "name": f"Player {in_pid}",
                "from_slot": in_from, "to_slot": in_to, "score": 18.0}],
        "out": [{"player_id": out_pid, "name": f"Player {out_pid}",
                 "from_slot": in_to, "to_slot": in_from, "score": 8.0}],
        "delta": 10.0,
    }


GROUP_A = _swap_group(2, 20, 2, 1)    # ATL players (Sunday game)
GROUP_B = _swap_group(4, 20, 23, 3)   # DET players (Thu 13:00 game)


def _exec_payload():
    entries = []
    for pid, slot in E_BASE.items():
        entries.append({
            "playerId": pid,
            "lineupSlotId": slot,
            "lineupLocked": False,
            "playerPoolEntry": {
                "id": pid,
                "lineupSlotId": slot,
                "lineupLocked": False,
                "player": {
                    "id": pid,
                    "fullName": f"Player {pid}",
                    "eligibleSlots": [2, 4, 23, 20],
                    "defaultPositionId": 2,
                    "injuryStatus": "ACTIVE",
                    "proTeamId": PRO_TEAM[E_TEAMS[pid]],
                },
            },
        })
    return {
        "id": int(P_LEAGUE),
        "scoringPeriodId": 12,
        "teams": [{
            "id": 2, "name": "Team", "owners": [SWID],
            "roster": {"entries": entries},
        }],
        "settings": {},
    }


class _ExecTransport:
    def __init__(self, payload):
        self.payload = payload
        self.posts = []

    def get_league(self, season, league_id, params):
        return copy.deepcopy(self.payload)

    def post_transaction(self, season, league_id, payload):
        self.posts.append(payload)
        team = self.payload["teams"][0]
        by_pid = {e["playerId"]: e for e in team["roster"]["entries"]}
        for item in payload["items"]:
            entry = by_pid[item["playerId"]]
            entry["lineupSlotId"] = item["toLineupSlotId"]
            entry["playerPoolEntry"]["lineupSlotId"] = item["toLineupSlotId"]
        return 200, {"status": "EXECUTED", "id": "txn"}


@pytest.fixture
def exec_harness(monkeypatch):
    h = {"settings": dict(store.DEFAULT_SETTINGS, mode="approve"),
         "plan_status": {}, "audit": []}
    plan = {
        "id": 7, "account_id": P_ACCOUNT, "platform": "espn",
        "league_id": P_LEAGUE, "season": P_SEASON, "team_id": 2,
        "scoring_period": 12, "mode": "approve", "status": "open",
        "snapshot": dict(E_BASE), "swaps": [GROUP_A, GROUP_B],
        "lineup_items": [], "issues": [],
    }
    h["plan"] = plan
    monkeypatch.setattr(
        "dashboard_services.accounts.get_espn_league_credentials",
        lambda *a: {"espn_s2": "s2", "swid": SWID})
    monkeypatch.setattr(
        "dashboard_services.accounts.get_any_espn_account_credentials",
        lambda *a: {"espn_s2": "s2", "swid": SWID})
    monkeypatch.setattr(autopilot_preview, "autopilot_access",
                        lambda *a: True)
    monkeypatch.setattr(
        "dashboard_services.autopilot_schedule.kickoffs_by_team",
        lambda *a, **k: dict(KICKOFFS))
    monkeypatch.setattr(store, "get_settings", lambda *a: dict(h["settings"]))
    monkeypatch.setattr(store, "get_kill_switch", lambda *a: False)
    monkeypatch.setattr(store, "get_plan", lambda pid: h["plan"])
    monkeypatch.setattr(
        store, "set_plan_status",
        lambda pid, status: h["plan_status"].__setitem__(pid, status))
    monkeypatch.setattr(store, "set_paused", lambda *a: None)
    monkeypatch.setattr(
        store, "record_audit",
        lambda *a, **k: h["audit"].append((a, k)))
    return h


def test_execute_skips_newly_locked_group_and_executes_the_rest(exec_harness):
    # Thu 14:00 ET: DET (group B) kicked off at 13:00, ATL (group A)
    # plays Sunday. ESPN's lock flags are all still False; the slate
    # alone must skip group B without superseding the plan.
    transport = _ExecTransport(_exec_payload())
    out = exec_mod.execute_plan(
        P_ACCOUNT, "espn", P_LEAGUE, P_SEASON, 7,
        accepted_groups=[0, 1], transport=transport,
        now=_et(2026, 11, 26, 14, 0))
    assert out["status"] == "partial"
    assert len(transport.posts) == 1
    posted = {(i["playerId"], i["toLineupSlotId"])
              for i in transport.posts[0]["items"]}
    assert posted == {(2, 2), (1, 20)}  # group A only
    by_index = {g["index"]: g for g in out["groups"]}
    assert by_index[0]["outcome"] == "executed"
    assert by_index[1]["outcome"] == "failed"
    assert "kicked off" in by_index[1]["detail"]
    assert exec_harness["plan_status"][7] == "partial"
    actions = [a[4] for a, _k in exec_harness["audit"]]
    assert "executed" in actions and "failed" in actions


def test_execute_before_kickoff_runs_both_groups(exec_harness):
    transport = _ExecTransport(_exec_payload())
    out = exec_mod.execute_plan(
        P_ACCOUNT, "espn", P_LEAGUE, P_SEASON, 7,
        accepted_groups=[0, 1], transport=transport,
        now=_et(2026, 11, 26, 12, 0))
    assert out["status"] == "executed"
    posted = {(i["playerId"], i["toLineupSlotId"])
              for i in transport.posts[0]["items"]}
    assert posted == {(2, 2), (1, 20), (4, 23), (3, 20)}
