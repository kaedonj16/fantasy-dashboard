"""Buy-low / sell-high candidate boards: pool-wide buy/sell questions.

"Who should I buy low?" names no player, so the single-player
buy_sell_hold path declined it with the leaked internal string "That is
not available: no resolved player." (live bug, 2026-10-15). These tests
pin the fix: pool-wide buy/sell phrasings route to leaderboard boards
(board_kind buy_low / sell_high) in BOTH routers, the evidence builder
screens the site's value movers by whether production supports the move,
the deterministic board composer renders them, named-player controls
stay buy_sell_hold, and an unresolvable player question declines in a
natural sentence that asks which player they mean.
Everything external is monkeypatched at the evidence module's lazy shims.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence
from dashboard_services.assistant.router import _keyword_route


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": {"raw_scoring_settings": {"rec": 1.0},
                      "rosters": [], "roster_positions": []},
              "roster": {"players": [], "starters": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


# -- router keyword fallback ------------------------------------------------

@pytest.mark.parametrize("message,kind", [
    ("Who should I buy low?", "buy_low"),
    ("Who should I sell high?", "sell_high"),
    ("Buy-low candidates", "buy_low"),
    ("Sell-high candidates", "sell_high"),
    ("Who is a buy low right now?", "buy_low"),
    ("Who is undervalued right now?", "buy_low"),
    ("Who is overvalued right now?", "sell_high"),
    ("Who should I buy?", "buy_low"),
    ("Top 5 buy-low candidates", "buy_low"),
])
def test_keyword_pool_buy_sell_is_a_value_board(message, kind):
    r = _keyword_route(message)
    assert r["intent"] == "leaderboard"
    assert r["board_kind"] == kind


@pytest.mark.parametrize("message", [
    "Is Bijan Robinson a buy low?",
    "Should I buy Bijan Robinson?",
    "Should I sell Bijan Robinson?",
    "Buy, sell, or hold on Bijan Robinson?",
])
def test_keyword_named_player_stays_buy_sell_hold(message):
    r = _keyword_route(message)
    assert r["intent"] == "buy_sell_hold"


# -- evidence builder ---------------------------------------------------------

def _mover(pid, name, pos, delta, old, new):
    return {"player_id": pid, "name": name, "position": pos, "team": "T",
            "delta": delta, "old_value": old, "new_value": new}


MOVERS = {
    "used_days": 7,
    "risers": [
        _mover("r1", "Riser One", "RB", 440.0, 1540.0, 1980.0),
        _mover("r2", "Riser Two", "WR", 365.0, 3045.0, 3410.0),
    ],
    "fallers": [
        _mover("f1", "Faller One", "WR", -450.0, 3660.0, 3210.0),
        _mover("f2", "Faller Two", "WR", -500.0, 2900.0, 2400.0),
        _mover("f3", "Faller Three", "RB", -300.0, 2200.0, 1900.0),
        _mover("f4", "Faller Four", "TE", -200.0, 1800.0, 1600.0),
    ],
}

# ppg / expected_ppr_per_game per player: f2's production fell with his
# value (not a buy-low), r2 produces above expectation (not a sell-high),
# f3 has no production line at all (unverifiable, never a candidate).
PRODUCTION = {
    "f1": {"ppg": 13.8, "expected_ppr_per_game": 12.9, "games": 6},
    "f2": {"ppg": 8.0, "expected_ppr_per_game": 10.5, "games": 6},
    "f3": {},
    "f4": {"ppg": 12.0, "expected_ppr_per_game": 12.0, "games": 5},
    "r1": {"ppg": 9.8, "expected_ppr_per_game": 11.6, "games": 6},
    "r2": {"ppg": 14.1, "expected_ppr_per_game": 12.4, "games": 6},
}


@pytest.fixture()
def value_env(monkeypatch):
    calls = []

    def fake_movers(**kwargs):
        calls.append(kwargs)
        return dict(MOVERS)

    monkeypatch.setattr(evidence, "get_top_movers", fake_movers)
    monkeypatch.setattr(evidence, "get_model_value_table_cached", list)
    monkeypatch.setattr(
        evidence, "_metrics_for",
        lambda pid, season=None: (dict(PRODUCTION.get(pid, {})), []))
    return calls


def test_buy_low_board_screens_fallers_by_production(value_env):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low"})
    assert pack["available"] is True
    assert pack["label"] == "Buy-low candidates"
    # f2 dropped (production fell too), f3 dropped (no production line).
    assert [r["name"] for r in pack["board"]] == ["Faller One", "Faller Four"]
    top = pack["board"][0]
    assert top["delta"] == -450.0 and top["value"] == 3210.0
    assert top["baseline_value"] == 3660.0
    assert top["ppg"] == 13.8 and top["expected_ppg"] == 12.9
    assert top["over_expected"] == 0.9
    assert pack["pool_size"] == 2 and pack["movers_examined"] == 4
    assert "2 of 4 fallers qualify" in pack["qualification_text"]
    # The movers call mirrors the /api/value-movers route's arguments.
    call = value_env[0]
    assert call["days"] == 7 and call["min_baseline_value"] == 10
    assert call["min_current_value"] == 20.0 and call["league_type"] == "1qb"
    assert call["current_values"] == {}


def test_sell_high_board_screens_risers_by_production(value_env):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "sell_high"})
    assert pack["available"] is True
    assert pack["label"] == "Sell-high candidates"
    # r2 rose WITH production (over expectation): not a sell-high.
    assert [r["name"] for r in pack["board"]] == ["Riser One"]
    top = pack["board"][0]
    assert top["delta"] == 440.0 and top["over_expected"] == -1.8
    assert "1 of 2 risers qualify" in pack["qualification_text"]


def test_value_board_position_filter(value_env):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low", "position": "TE"})
    assert [r["name"] for r in pack["board"]] == ["Faller Four"]


def test_value_board_no_candidates_is_no_data(value_env, monkeypatch):
    monkeypatch.setattr(evidence, "_metrics_for",
                        lambda pid, season=None: ({}, []))
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low"})
    assert pack["available"] is False
    assert pack["missing"] == "no buy-low candidates right now"


# -- compose -------------------------------------------------------------------

def _value_pack():
    return {"type": "leaderboard", "available": True,
            "league": {"platform": "sleeper", "league_id": "L1",
                       "season": 2026, "league_name": "Blackedraw",
                       "team_name": "Couch"},
            "week": 7, "board_kind": "buy_low", "label": "Buy-low candidates",
            "direction": "fallers", "value_days": 7, "pool_size": 2,
            "movers_examined": 4, "board_count": 2, "requested_n": 10,
            "position_filter": None, "team_filter": None,
            "qualification_text": (
                "Value movement over the last 7 days from the site's value "
                "history; value fell while production held at or above "
                "expectation (PPG vs expected PPG). 2 of 4 fallers qualify."),
            "board": [
                {"rank": 1, "player_id": "f1", "name": "Faller One",
                 "team": "T1", "position": "WR", "value": 3210.0, "games": 6,
                 "on_my_roster": False, "delta": -450.0,
                 "baseline_value": 3660.0, "ppg": 13.8,
                 "expected_ppg": 12.9, "over_expected": 0.9},
                {"rank": 2, "player_id": "f4", "name": "Faller Four",
                 "team": "T4", "position": "TE", "value": 1600.0, "games": 5,
                 "on_my_roster": False, "delta": -200.0,
                 "baseline_value": 1800.0, "ppg": 12.0,
                 "expected_ppg": 12.0, "over_expected": 0.0}]}


def _canned(text):
    def fake(system, payload, schema, name):
        return {"verdict": "MODEL VERDICT", "text": text, "receipts": [],
                "caveat": None, "followups": [], "act_link": None}
    return fake


def test_compose_buy_low_board(monkeypatch):
    monkeypatch.setattr(compose_mod, "call_model_json",
                        _canned("A quiet intro that names nobody."))
    ans = compose_mod.compose_answer("leaderboard", "Who should I buy low?",
                                     _value_pack())
    assert ans["verdict"] == "Faller One is the top buy-low"
    assert len(ans["receipts"]) == 2  # one per board row
    assert ans["receipts"][0] == {
        "label": "1. Faller One",
        "value": "value -450 (now 3210); 13.8 PPG (+0.9 vs expected)"}
    assert ("1. Faller One (T1, WR): value -450 to 3210 (was 3660), "
            "13.8 PPG vs 12.9 expected") in ans["text"]
    assert ans["act_link"] == {"href": "/top-movers",
                               "label": "View on Top Movers"}


# -- the decline string ---------------------------------------------------------

@pytest.mark.parametrize("intent,missing", [
    ("buy_sell_hold", "no resolved player"),
    ("player", "no resolved player"),
    ("start_sit", "no resolved players"),
    ("breakout_history", "no resolved player"),
])
def test_unresolved_player_decline_asks_who(intent, missing):
    ans = compose_mod.no_data_answer(
        {"type": intent, "available": False, "missing": missing})
    assert "no resolved player" not in ans["text"]
    assert "not available" not in ans["text"]
    assert "which player" in ans["text"]
    assert ans["verdict"] is None


def test_unresolved_buy_sell_hold_decline_via_compose_answer():
    pack = {"type": "buy_sell_hold", "available": False,
            "missing": "no resolved player", "league": {}}
    ans = compose_mod.compose_answer("buy_sell_hold", "Buy, sell, or hold?",
                                     pack)
    assert "buy, sell, or hold call" in ans["text"]
    assert "no resolved player" not in ans["text"]
