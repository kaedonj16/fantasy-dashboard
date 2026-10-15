"""Watchlist pack: builder, routing, and compose guarantees.

The read path is the Watchlist tool's own: rows come from
routes.watchlist_bp._rows_for under the acct:<id> key from its _user_key,
7-day deltas from the movers board /api/watchlist-alerts reads, and the
alert call from utils.watchlist_alerts.is_value_alert. All external sources
are monkeypatched at the evidence module's lazy shims.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose, evidence, router

ROWS = [
    {"player_id": "b2", "name": "Woody Marks", "position": "RB", "team": "HOU",
     "note": "", "added_at": "2026-10-01T00:00:00"},
    {"player_id": "a5", "name": "Zay Flowers", "position": "WR", "team": "BAL",
     "note": "", "added_at": "2026-10-02T00:00:00"},
    {"player_id": "i1", "name": "Drake London", "position": "WR", "team": "ATL",
     "note": "", "added_at": "2026-10-03T00:00:00"},
    {"player_id": "w1", "name": "Kaleb Johnson", "position": "RB", "team": "PIT",
     "note": "stash target", "added_at": "2026-10-04T00:00:00"},
]
VALUES = [{"id": "a5", "value": 4680.0}, {"id": "b2", "value": 3400.0},
          {"id": "i1", "value": 3900.0}, {"id": "w1", "value": 900.0}]
MOVERS = {"risers": [{"player_id": "a5", "delta": 620.0},
                     {"player_id": "w1", "delta": 240.0},
                     {"player_id": "i1", "delta": 120.0}],
          "fallers": [{"player_id": "b2", "delta": -180.0}]}
INDEX = {"i1": {"full_name": "Drake London", "position": "WR", "team": "ATL",
                "injury_status": "Out"}}


def _bundle():
    return {"platform": "sleeper", "league_id": "blackedraw", "season": 2026,
            "league_name": "Blackedraw", "team_name": "Caleb's Casting Couch",
            "week": 7, "ctx": {"rosters": [{"roster_id": 1, "players": ["a5", "i1"]},
                                           {"roster_id": 2, "players": ["b2"]}]},
            "roster": {"roster_id": 2}, "roster_player_ids": {"b2"}}


@pytest.fixture()
def wired(monkeypatch):
    monkeypatch.setattr(evidence, "watchlist_user_key", lambda *a, **k: "acct:7")
    monkeypatch.setattr(evidence, "watchlist_rows", lambda key: ROWS)
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: VALUES)
    monkeypatch.setattr(evidence, "watchlist_top_movers", lambda **k: MOVERS)
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: INDEX)


def test_watchlist_builder_alerts_and_order(wired):
    pack, tools = evidence.build_watchlist(_bundle(), {}, [])
    assert pack["available"] is True
    assert "watchlist_bp._rows_for" in tools
    assert pack["count"] == 4
    # Alert-first by |delta7|: Flowers (620, clears 10% of 4680), Johnson
    # (240, clears 10% of 900), London (injury designation), then Marks
    # (180 does not clear 10% of 3400, no designation).
    assert [p["name"] for p in pack["players"]] == [
        "Zay Flowers", "Kaleb Johnson", "Drake London", "Woody Marks"]
    assert pack["alerts"] == ["Zay Flowers", "Kaleb Johnson", "Drake London"]
    assert pack["top_mover"]["name"] == "Zay Flowers"
    by_name = {p["name"]: p for p in pack["players"]}
    assert by_name["Drake London"]["injury_status"] == "Out"
    assert by_name["Woody Marks"]["alert"] is False
    assert by_name["Woody Marks"]["on_my_roster"] is True
    assert by_name["Kaleb Johnson"]["rostered"] is False


def test_watchlist_builder_no_account_is_no_data(monkeypatch):
    monkeypatch.setattr(evidence, "watchlist_user_key", lambda *a, **k: None)
    pack, _tools = evidence.build_watchlist(_bundle(), {}, [])
    assert pack["available"] is False


def test_watchlist_registered():
    assert evidence.BUILDERS["watchlist"] is evidence.build_watchlist
    assert "watchlist" in router.INTENTS


# -- routing ---------------------------------------------------------------------

@pytest.fixture()
def keyword(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(router, "call_model_json", _boom)
    return router.classify


def _intent(classify, message):
    return classify(message, {"league_name": "Blackedraw", "week": 7, "season": 2026}, None)


@pytest.mark.parametrize("message", [
    "What's happening with my watchlist?",
    "Who is on my watchlist?",
    "Show my watch list",
    "Any alerts on my starred players?",
])
def test_keyword_watchlist(keyword, message):
    assert _intent(keyword, message)["intent"] == "watchlist"


@pytest.mark.parametrize("message,intent", [
    ("Who should I add from waivers?", "waivers"),
    ("What should I do with my team this week?", "this_week"),
])
def test_keyword_watchlist_does_not_steal(keyword, message, intent):
    assert _intent(keyword, message)["intent"] == intent


# -- compose guarantees ------------------------------------------------------------

def _golden_pack():
    import json, pathlib
    data = json.loads(pathlib.Path("assistant_eval/questions.json").read_text())
    return next(q["evidence_pack"] for q in data["questions"] if q["id"] == "wl1")


def test_watchlist_required_receipts_cite_top_mover():
    rows = compose._required_receipts("watchlist", _golden_pack())
    assert rows == [{"label": "Zay Flowers value", "value": "4680"},
                    {"label": "Zay Flowers 7-day move", "value": "620"}]


def test_watchlist_required_mentions_name_top_mover():
    assert compose._required_mentions("watchlist", _golden_pack()) == [("Zay Flowers", "watchlist")]


def test_watchlist_mention_injected_when_draft_skips_it():
    data = {"verdict": "Three alerts", "text": "A few of your players moved.",
            "caveat": None, "receipts": [], "followups": []}
    out = compose._ensure_required_mentions("watchlist", _golden_pack(), data, [])
    assert "Zay Flowers" in out["text"]


def test_watchlist_act_link():
    assert compose._resolve_act_link("watchlist", _golden_pack()) == {
        "href": "/watchlist", "label": "View Watchlist"}


def test_watchlist_empty_model_verdict_falls_back_to_top_mover(monkeypatch):
    # Live eval round 3 wl1: the model returned no verdict at all. The
    # deterministic fallback names the pack's top mover, the same
    # contract the player intent got in round 2.
    def _draft():
        return {"verdict": None,
                "text": "Zay Flowers is up 620 over the last 7 days to 4680.",
                "receipts": [], "caveat": None, "followups": []}
    monkeypatch.setattr(compose, "call_model_json", lambda *a, **k: _draft())
    answer = compose.compose_answer(
        "watchlist", "What's happening with my watchlist?", _golden_pack())
    assert answer["verdict"] == "Zay Flowers"
