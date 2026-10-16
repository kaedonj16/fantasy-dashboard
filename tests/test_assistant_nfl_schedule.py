"""NFL schedule intent tests.

"Who plays tonight?" with no possessive or league cue must answer the
real NFL game (Kaedon's live report: the assistant answered with her
fantasy matchup's win probability and projections instead). The pack
reads the site's own week-schedule store (utils.data_cache.
load_week_schedule over the ESPN-fed week files), stubbed here at the
evidence module's shim with both row shapes the store carries: the
Tank01-shaped disk rows and the ESPN-normalized rows. Days resolve
against a stubbed America/New_York clock (Thursday, October 15, 2026,
the real Week 6 Thursday of the 2026 season).
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from datetime import datetime
from zoneinfo import ZoneInfo

from dashboard_services import nfl_game_data
from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence, router, tools
from dashboard_services.assistant.router import _keyword_route


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "blackedraw",
              "season": 2026, "league_name": "Blackedraw",
              "team_name": "Caleb's Casting Couch", "week": 6,
              "ctx": {}, "roster": {"roster_id": 3, "players": [],
                                    "starters": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


# -- router: keyword path -----------------------------------------------------

@pytest.mark.parametrize("message,day,team", [
    ("Who plays tonight?", "tonight", None),
    ("who plays tonight", "tonight", None),
    ("Who's playing tonight?", "tonight", None),
    ("What games are on Sunday?", "sunday", None),
    ("What games are on tonight?", "tonight", None),
    ("Who plays this week?", "this_week", None),
    ("Who plays on Monday night?", "monday", None),
    ("Monday night football: who is playing?", "monday", None),
    ("Are there any games tomorrow?", "tomorrow", None),
    ("What is the NFL schedule this week?", "this_week", None),
    ("When do the Chiefs play?", None, "KC"),
    ("What time is the Seahawks game?", None, "SEA"),
    ("When do the Chiefs play this week?", "this_week", "KC"),
    ("Do the Bills play on Sunday?", "sunday", "BUF"),
    ("Who do the Cowboys play this week?", "this_week", "DAL"),
])
def test_keyword_route_nfl_schedule(message, day, team):
    out = _keyword_route(message)
    assert out["intent"] == "nfl_schedule"
    assert out["day_hint"] == day
    assert out["team"] == team


@pytest.mark.parametrize("message,intent", [
    # Possessive fantasy forms stay on their own paths.
    ("Who am I playing this week?", "this_week"),
    ("Who does my team play this week?", "this_week"),
    ("Will I win this week?", "matchup"),
    ("Who is winning my game right now?", "live"),
    ("What do I need tonight to win?", "live"),
    # Lineup anchors are never stolen by the schedule rule.
    ("Should I start Josh Allen or Jalen Hurts?", "start_sit"),
    ("Who should I flex this week?", "start_sit"),
    # Schedule strength is the fantasy schedule intent, not fixtures.
    ("Who has the easiest remaining schedule?", "schedule"),
])
def test_keyword_route_neighbors_keep_intents(message, intent):
    assert _keyword_route(message)["intent"] == intent


def test_registered_intents_have_builders():
    assert "nfl_schedule" in router.INTENTS
    assert "nfl_schedule" in evidence.BUILDERS


def test_model_route_parses_day_hint(monkeypatch):
    captured = {}

    def fake(system, user, schema, model=None, timeout=30):
        captured["system"] = system
        return {"intent": "nfl_schedule", "team": "KC",
                "day_hint": "sunday"}

    monkeypatch.setattr(router, "call_model_json", fake)
    out = router.classify("When do the Chiefs play on Sunday?",
                          _bundle())
    assert out["intent"] == "nfl_schedule"
    assert out["entities"]["team"] == "KC"
    assert out["entities"]["day_hint"] == "sunday"
    assert "nfl_schedule" in captured["system"]


# -- evidence: row normalization ----------------------------------------------

def _tank_row(away, home, date, time, epoch, week="Week 6", **over):
    row = {"gameID": f"{date}_{away}@{home}", "gameWeek": week,
           "gameDate": date, "gameTime": time,
           "gameTime_epoch": str(epoch), "away": away, "home": home,
           "gameStatus": "Scheduled", "gameStatusCode": "0"}
    row.update(over)
    return row


WEEK6_TANK = [
    _tank_row("SEA", "DEN", "20261015", "8:15p", 1792109700),
    _tank_row("HOU", "JAX", "20261018", "9:30a", 1792330200),
    _tank_row("LAC", "KC", "20261018", "4:25p", 1792355100),
    _tank_row("DAL", "GB", "20261018", "8:20p", 1792369200),
    _tank_row("WAS", "SF", "20261019", "8:15p", 1792455300),
]
WEEK7_TANK = [
    _tank_row("KC", "SEA", "20261025", "8:20p", 1792974000, week="Week 7"),
]


def _stub_schedule(monkeypatch, weeks, now=datetime(2026, 10, 15, 12, 0,
                                                   tzinfo=ZoneInfo("America/New_York"))):
    monkeypatch.setattr(
        evidence, "load_week_schedule",
        lambda season, week: weeks.get((season, week), []))
    monkeypatch.setattr(evidence, "_et_now", lambda: now)


def _build(monkeypatch, entities, weeks=None, **bundle_over):
    _stub_schedule(monkeypatch,
                   weeks if weeks is not None
                   else {(2026, 6): WEEK6_TANK, (2026, 7): WEEK7_TANK})
    pack, used = evidence.build_nfl_schedule(_bundle(**bundle_over),
                                             entities, [])
    return pack, used


def test_tonight_pack_from_tank_rows(monkeypatch):
    pack, used = _build(monkeypatch, {"day_hint": "tonight"})
    assert pack["available"] is True
    assert pack["scope"] == "tonight"
    assert pack["day_label"] == "Thursday, October 15"
    assert pack["game_count"] == 1
    g = pack["games"][0]
    assert (g["away"], g["home"]) == ("SEA", "DEN")
    assert g["away_name"] == "Seattle Seahawks"
    assert g["kickoff_label"] == "8:15 PM ET"
    assert g["status"] == "pregame"
    assert g["network"] is None  # Tank01 rows carry no network: omitted
    assert "data_cache.load_week_schedule" in used


def test_sunday_slate_and_team_scope(monkeypatch):
    pack, _ = _build(monkeypatch, {"day_hint": "sunday"})
    assert pack["scope"] == "day"
    assert pack["game_count"] == 3
    assert [(g["away"], g["home"]) for g in pack["games"]] == [
        ("HOU", "JAX"), ("LAC", "KC"), ("DAL", "GB")]
    pack, _ = _build(monkeypatch, {"team": "KC"})
    assert pack["scope"] == "team"
    assert pack["bye"] is False
    g = pack["games"][0]
    assert (g["away_name"], g["home_name"]) == (
        "Los Angeles Chargers", "Kansas City Chiefs")
    assert g["kickoff_label"] == "4:25 PM ET"
    assert g["date_label"] == "Sun, Oct 18"


def test_bye_team_points_at_next_week(monkeypatch):
    # Denver plays Thursday of week 6, so ask on the week-7 view: no
    # week-7 game is loaded for them beyond the stub, KC plays Sunday.
    pack, _ = _build(monkeypatch, {"team": "KC", "week": 7},
                     week=7)
    assert pack["scope"] == "team" and pack["bye"] is False
    pack, _ = _build(monkeypatch, {"team": "DEN", "week": 7}, week=7)
    assert pack["bye"] is True
    assert pack["games"] == []


def test_week_scope_and_empty_day_next_game(monkeypatch):
    pack, _ = _build(monkeypatch, {"day_hint": "this_week"})
    assert pack["scope"] == "week"
    assert pack["game_count"] == 5  # week 7 rows stay out of week scope
    pack, _ = _build(monkeypatch, {"day_hint": "friday"})
    assert pack["scope"] == "day"
    assert pack["games"] == []
    # No Friday game in either loaded week: the next game is pointed
    # at instead (tonight's, asked on the Thursday).
    assert pack["next_game"]["away"] == "SEA"


def test_espn_normalized_rows_and_network(monkeypatch):
    espn_rows = [{
        "gameID": "20261015_SEA@DEN", "gameTime": "2026-10-16T00:15Z",
        "gameTime_epoch": 1792109700, "home": "DEN", "away": "SEA",
        "homePts": "", "awayPts": "", "week": 6,
        "normalized_status": "pregame", "network": "Prime Video",
    }]
    pack, _ = _build(monkeypatch, {"day_hint": "tonight"},
                     weeks={(2026, 6): espn_rows, (2026, 7): []})
    g = pack["games"][0]
    assert g["kickoff_label"] == "8:15 PM ET"
    assert g["network"] == "Prime Video"
    assert g["status"] == "pregame"


def test_no_schedule_rows_is_unavailable(monkeypatch):
    pack, _ = _build(monkeypatch, {"day_hint": "tonight"},
                     weeks={(2026, 6): [], (2026, 7): []})
    assert pack["available"] is False
    assert "not available" in pack["missing"]


# -- compose: deterministic answers ---------------------------------------------

def test_compose_tonight_answer(monkeypatch):
    pack, _ = _build(monkeypatch, {"day_hint": "tonight"})
    ans = compose_mod.compose_answer("nfl_schedule",
                                     "Who plays tonight?", pack)
    assert ans["verdict"] == "Seattle Seahawks at Denver Broncos tonight"
    assert "8:15 PM ET" in ans["text"]
    assert ans["act_link"]["label"] == "Open ScoreZone"
    assert "/scorezone" in ans["act_link"]["href"]
    # The bug being fixed: no fantasy projections or win probability.
    blob = (ans["verdict"] + ans["text"]).lower()
    assert "win probability" not in blob
    assert "projected" not in blob
    assert "casting couch" not in blob


def test_compose_team_and_bye_answers(monkeypatch):
    pack, _ = _build(monkeypatch, {"team": "KC"})
    ans = compose_mod.compose_answer("nfl_schedule",
                                     "When do the Chiefs play?", pack)
    assert ans["verdict"] == "Kansas City Chiefs play Sun, Oct 18"
    assert "4:25 PM ET" in ans["text"]
    pack, _ = _build(monkeypatch, {"team": "DEN", "week": 7}, week=7)
    ans = compose_mod.compose_answer("nfl_schedule",
                                     "When do the Broncos play?", pack)
    assert "bye" in ans["verdict"].lower()


def test_compose_sunday_groups_buckets(monkeypatch):
    pack, _ = _build(monkeypatch, {"day_hint": "sunday"})
    ans = compose_mod.compose_answer("nfl_schedule",
                                     "What games are on Sunday?", pack)
    assert ans["verdict"] == "3 games on Sunday"
    assert "Morning:" in ans["text"] and "Primetime:" in ans["text"]


# -- open-layer tool ---------------------------------------------------------------

def test_tool_nfl_schedule(monkeypatch):
    _stub_schedule(monkeypatch, {(2026, 6): WEEK6_TANK,
                                 (2026, 7): WEEK7_TANK})
    out = tools._tool_nfl_schedule(_bundle(), {"day": "tonight"})
    assert out["ok"] is True
    assert out["games"][0]["away_name"] == "Seattle Seahawks"
    assert out["games"][0]["kickoff_label"] == "8:15 PM ET"
    assert out["link"]["label"] == "Open ScoreZone"
    assert "nfl_schedule" in tools.TOOL_FUNCS
    assert "nfl_schedule" in {t["name"] for t in tools.TOOL_CATALOG}


# -- scoreboard normalizer: network pass-through -----------------------------------

def test_normalize_scoreboard_carries_network():
    payload = {"events": [{
        "id": "1", "date": "2026-10-16T00:15Z",
        "season": {"year": 2026, "type": 2}, "week": {"number": 6},
        "status": {"type": {"state": "pre", "completed": False}},
        "competitions": [{
            "broadcasts": [{"names": ["Prime Video"]}],
            "competitors": [
                {"homeAway": "home", "team": {"abbreviation": "DEN"},
                 "score": ""},
                {"homeAway": "away", "team": {"abbreviation": "SEA"},
                 "score": ""}]}]}]}
    rows = nfl_game_data.normalize_scoreboard(payload)
    assert rows[0]["network"] == "Prime Video"
    payload["events"][0]["competitions"][0]["broadcasts"] = []
    rows = nfl_game_data.normalize_scoreboard(payload)
    assert rows[0]["network"] is None
