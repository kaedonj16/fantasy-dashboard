"""Iteration 2, item 4: Next Steps must not read "all set" over
attention states. The Nov 12 roster (league 92916) had a +0.6 D/ST
swap on its own Weekly Hub, two Questionable starters (Lock,
Kincaid), two empty bench spots, and a starter (McCarthy) whose
slot locked that Thursday night — and the card said "You are all
set for Week 10."

Pins:
- a Thursday-night starter inside 24h of kickoff gets a lock action;
- Questionable starters with a game inside 48h get a watch action
  naming the healthy same-position plug;
- open bench spots in the regular season get a fill action;
- a genuinely quiet roster still gets NO actions (all-set intact);
- the lineup collector's sub-threshold pass surfaces the Hub's
  +0.6 D/ST swap as one quiet action instead of an all-clear.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("pandas")

import app as app_module  # noqa: E402

ET = ZoneInfo("America/New_York")


def _ts(day, hour, minute=0):
    return datetime(2026, 11, day, hour, minute, tzinfo=ET).timestamp()


THU_1600 = _ts(12, 16)          # Thursday Nov 12, 4:00 PM ET
SUN_0900 = _ts(15, 9)           # Sunday Nov 15, 9:00 AM ET
THU_GAME = _ts(12, 20, 15)      # McCarthy, Thu 8:15 PM ET
SUN_1300 = _ts(15, 13)          # Kincaid / CMC / JSN window
SUN_1605 = _ts(15, 16, 5)       # Lock @ LV, Sun 4:05 PM ET

PLAYERS = {
    # pid: (name, team, pos, injury_status)
    "mccarthy": ("J.J. McCarthy", "MIN", "QB", ""),
    "lock": ("Drew Lock", "SEA", "QB", "Questionable"),
    "kincaid": ("Dalton Kincaid", "BUF", "TE", "Questionable"),
    "cmc": ("Christian McCaffrey", "SF", "RB", ""),
    "jsn": ("Jaxon Smith-Njigba", "SEA", "WR", ""),
    "sanders": ("Shedeur Sanders", "CLE", "QB", ""),
    "sadiq": ("Theo Sadiq", "BUF", "TE", ""),
    "hill": ("Justice Hill", "BAL", "RB", ""),
}
STARTERS = ["mccarthy", "lock", "kincaid", "cmc", "jsn"]
BENCH = ["sanders", "sadiq", "hill"]

GAMES = [
    {"home": "MIN", "away": "WAS", "gameTime_epoch": THU_GAME},
    {"home": "BUF", "away": "MIA", "gameTime_epoch": SUN_1300},
    {"home": "SF", "away": "ARI", "gameTime_epoch": SUN_1300},
    {"home": "LV", "away": "SEA", "gameTime_epoch": SUN_1605},
    {"home": "CLE", "away": "NYJ", "gameTime_epoch": SUN_1300},
    {"home": "BAL", "away": "PIT", "gameTime_epoch": SUN_1300},
]


def _ctx(open_spots=2):
    positions = ["QB", "QB", "RB", "WR", "TE"] + ["BN"] * (3 + open_spots)
    return {
        "platform": "sleeper",
        "league_id": "L-iter2",
        "current_season": 2026,
        "current_week": 10,
        "rosters": [{
            "roster_id": 1,
            "starters": list(STARTERS),
            "players": list(STARTERS) + list(BENCH),
            "reserve": [], "taxi": [],
        }],
        "players_map": {
            pid: {"name": name, "team": team, "pos": pos}
            for pid, (name, team, pos, _inj) in PLAYERS.items()
        },
        "roster_positions": positions,
        "proj_by_week": {10: {"projections": {"sanders": 12.0, "sadiq": 3.1}}},
        "league": {"settings": {}},
    }


def _fixture(monkeypatch):
    full = {
        pid: {"full_name": name, "team": team, "position": pos,
              "injury_status": inj}
        for pid, (name, team, pos, inj) in PLAYERS.items()
    }
    monkeypatch.setattr(app_module, "get_players_global", lambda: full)
    monkeypatch.setattr(app_module, "load_week_schedule",
                        lambda season, week: list(GAMES))


def _attention(monkeypatch, now_ts, **ctx_kw):
    _fixture(monkeypatch)
    with app_module.app.test_request_context("/"):
        return app_module._next_steps_attention_actions(
            _ctx(**ctx_kw), 1, now_ts=now_ts)


def test_thursday_lock_and_open_bench_break_all_set(monkeypatch):
    actions = _attention(monkeypatch, THU_1600)
    texts = [a["action"] for a in actions]
    assert any("McCarthy" in t and "locks at 8:15 PM ET" in t for t in texts)
    assert any(t == "Fill 2 open bench spots" for t in texts)
    # Sunday's questionable starters are >48h out on Thursday: the
    # watch actions must not fire early.
    assert not any("questionable" in t for t in texts)


def test_sunday_questionable_starters_get_watch_actions(monkeypatch):
    actions = _attention(monkeypatch, SUN_0900)
    by_text = {a["action"]: a for a in actions}
    lock_a = by_text.get("Drew Lock is questionable")
    kin_a = by_text.get("Dalton Kincaid is questionable")
    assert lock_a is not None and "Shedeur Sanders is the plug" in lock_a["why"]
    assert kin_a is not None and "Theo Sadiq is the plug" in kin_a["why"]
    # Thursday's lock already kicked off: no lock action on Sunday.
    assert not any("locks at" in t for t in by_text)


def test_quiet_roster_still_all_set(monkeypatch):
    # Full bench, no questionable starters near a game, and the only
    # early-window starter is more than 24h out.
    actions = _attention(monkeypatch, _ts(11, 12), open_spots=0)
    assert actions == []


# --- sub-threshold D/ST swap (the Hub's +0.6) -----------------------------

DST_PLAYERS = {
    "qb1": ("QB One", "BUF", "QB", ""),
    "rb1": ("RB One", "BUF", "RB", ""),
    "wr1": ("WR One", "BUF", "WR", ""),
    "te1": ("TE One", "BUF", "TE", ""),
    "sea_def": ("Seattle Seahawks", "SEA", "DEF", ""),
    "no_def": ("New Orleans Saints", "NO", "DEF", ""),
}
DST_PROJ = {"qb1": 20.0, "rb1": 15.0, "wr1": 14.0, "te1": 8.0,
            "sea_def": 10.2, "no_def": 10.8}
DST_GAMES = [
    {"home": "BUF", "away": "MIA", "gameTime_epoch": SUN_1300},
    {"home": "SEA", "away": "LV", "gameTime_epoch": SUN_1605},
    {"home": "NO", "away": "ATL", "gameTime_epoch": SUN_1300},
]


def test_sub_threshold_dst_swap_surfaces_as_quiet_action(monkeypatch):
    full = {
        pid: {"full_name": name, "team": team, "position": pos,
              "injury_status": inj}
        for pid, (name, team, pos, inj) in DST_PLAYERS.items()
    }
    monkeypatch.setattr(app_module, "get_players_global", lambda: full)
    monkeypatch.setattr(app_module, "load_week_schedule",
                        lambda season, week: list(DST_GAMES))
    ctx = {
        "platform": "sleeper",
        "league_id": "L-iter2-dst",
        "current_season": 2026,
        "current_week": 10,
        "rosters": [{
            "roster_id": 1,
            "starters": ["qb1", "rb1", "wr1", "te1", "sea_def"],
            "players": ["qb1", "rb1", "wr1", "te1", "sea_def", "no_def"],
            "reserve": [], "taxi": [],
        }],
        "players_map": {
            pid: {"name": name, "team": team, "pos": pos}
            for pid, (name, team, pos, _inj) in DST_PLAYERS.items()
        },
        "roster_positions": ["QB", "RB", "WR", "TE", "DEF", "BN"],
        "proj_by_week": {10: {"projections": dict(DST_PROJ)}},
        "league": {"settings": {}},
    }
    with app_module.app.test_request_context("/"):
        actions = app_module._next_steps_lineup_actions(ctx, 1)
    swaps = [a for a in actions if a["action"].startswith("Start ")]
    assert len(swaps) == 1
    assert swaps[0]["action"] == "Start New Orleans Saints over Seattle Seahawks"
    assert swaps[0]["impact"] == "+0.6 projected points"
    assert swaps[0]["priority"] == "low"
