"""League-settings intent tests.

Kaedon's live report (2026-11-05): the BR Assistant answered "when is
trade deadline?" in Blackedraw with the canned out-of-scope decline,
even though the league settings the bundle already carries hold the
deadline (Sleeper trade_deadline week), the playoff start / field, and
the waiver system. The router used to send deadline questions to
out_of_scope on the premise that no pack could answer them; the
league_settings pack now answers all three topics deterministically
from those settings. Shapes here are Blackedraw's real settings
(verified against the Sleeper API 2026-11-05: trade_deadline 11,
playoff_week_start 15, playoff_teams 6, waiver_type 0 rolling with a
100 budget field the FAAB gate correctly rejects).
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence, router
from dashboard_services.assistant.router import _keyword_route


def _bundle(ctx, **over):
    bundle = {"platform": "sleeper", "league_id": "1312067280816832512",
              "season": 2026, "league_name": "blackedraw",
              "team_name": "Caleb's Casting Couch", "week": 9,
              "ctx": ctx, "roster": {"roster_id": 1, "players": [],
                                     "starters": [], "settings": {}},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


def _blackedraw_ctx():
    return {"league": {"name": "blackedraw", "settings": {
        "trade_deadline": 11, "playoff_week_start": 15,
        "playoff_teams": 6, "waiver_type": 0, "waiver_budget": 100,
        "waiver_clear_days": 2}}}


def _no_faab(monkeypatch, enabled=False, total=None):
    monkeypatch.setattr(
        evidence, "waiver_faab_context",
        lambda ctx: {"enabled": enabled, "total": total,
                     "waiver_type": "faab" if enabled else "priority",
                     "phase": "mid"})


# -- router: keyword path -----------------------------------------------------

@pytest.mark.parametrize("message,topic", [
    ("when is trade deadline?", "deadline"),
    ("When is the trade deadline?", "deadline"),
    ("What date is the trade deadline?", "deadline"),
    ("When do trades stop?", "deadline"),
    ("When do playoffs start?", "playoffs"),
    ("What week do playoffs start?", "playoffs"),
    ("How many teams make the playoffs?", "playoffs"),
    ("How do waivers work in my league?", "waivers"),
    ("What's my waiver budget?", "waivers"),
    ("When do waivers process?", "waivers"),
    ("Is my league a FAAB league?", "waivers"),
    ("What are my league settings?", "all"),
])
def test_keyword_route_league_settings(message, topic):
    out = _keyword_route(message)
    assert out["intent"] == "league_settings"
    assert out["settings_topic"] == topic


@pytest.mark.parametrize("message,intent", [
    # Advice questions that merely mention the settings keep their
    # own intents.
    ("Who should I target before the trade deadline?", "trade_finding"),
    ("What are my chances of making the playoffs?", "playoffs"),
    ("Who should I add from waivers this week?", "waivers"),
    ("How much FAAB should I bid on Justin Jefferson?", "waivers"),
    ("Who has the easiest playoff stretch?", "schedule"),
])
def test_keyword_route_settings_neighbors_keep_intents(message, intent):
    assert _keyword_route(message)["intent"] == intent


def test_league_settings_in_intents():
    assert "league_settings" in router.INTENTS
    assert "league_settings" in evidence.BUILDERS


# -- builder + compose: Blackedraw shapes -------------------------------------

def test_blackedraw_deadline_answers_week_11(monkeypatch):
    _no_faab(monkeypatch)
    pack, _tools = evidence.build_league_settings(
        _bundle(_blackedraw_ctx()), {"settings_topic": "deadline"}, [])
    assert pack["deadline"] == {"kind": "week", "week": 11,
                                "weeks_remaining": 2, "passed": False}
    answer = compose_mod.compose_answer(
        "league_settings", "when is trade deadline?", pack)
    assert "Week 11" in answer["text"]
    assert "2 weeks left" in answer["text"]
    assert answer["receipts"][0] == {"label": "Trade deadline",
                                     "value": "Week 11"}


def test_blackedraw_playoffs_start_week_15_six_teams(monkeypatch):
    _no_faab(monkeypatch)
    pack, _tools = evidence.build_league_settings(
        _bundle(_blackedraw_ctx()), {"settings_topic": "playoffs"}, [])
    assert pack["playoffs"]["week_start"] == 15
    assert pack["playoffs"]["teams"] == 6
    answer = compose_mod.compose_answer(
        "league_settings", "When do playoffs start?", pack)
    assert "Playoffs start in Week 15" in answer["text"]
    assert "6 teams make it" in answer["text"]


def test_blackedraw_waivers_are_priority_not_faab(monkeypatch):
    # Sleeper waiver_type 0 with a budget field is rolling priority:
    # the site's FAAB gate (stubbed here to its real verdict for this
    # shape) says not enabled, so the answer must not claim FAAB.
    _no_faab(monkeypatch, enabled=False)
    pack, _tools = evidence.build_league_settings(
        _bundle(_blackedraw_ctx()), {"settings_topic": "waivers"}, [])
    assert pack["waivers"]["system"] == "priority"
    assert pack["waivers"]["faab_enabled"] is False
    answer = compose_mod.compose_answer(
        "league_settings", "How do waivers work in my league?", pack)
    assert "waiver priority, not FAAB" in answer["text"]


def test_faab_league_reports_budget_and_remaining(monkeypatch):
    _no_faab(monkeypatch, enabled=True, total=100)
    ctx = {"league": {"settings": {"trade_deadline": 10,
                                   "playoff_week_start": 14,
                                   "playoff_teams": 4,
                                   "waiver_type": 2,
                                   "waiver_budget": 100}}}
    bundle = _bundle(ctx, roster={"roster_id": 1, "players": [],
                                  "starters": [],
                                  "settings": {"waiver_budget_used": 30}})
    pack, _tools = evidence.build_league_settings(
        bundle, {"settings_topic": "waivers"}, [])
    assert pack["waivers"]["system"] == "faab"
    assert pack["waivers"]["budget_total"] == 100
    assert pack["waivers"]["budget_remaining"] == 70
    answer = compose_mod.compose_answer(
        "league_settings", "What's my waiver budget?", pack)
    assert "FAAB" in answer["text"]
    assert "$100" in answer["text"]
    assert "$70 left" in answer["text"]


def test_all_topics_render_together(monkeypatch):
    _no_faab(monkeypatch)
    pack, _tools = evidence.build_league_settings(
        _bundle(_blackedraw_ctx()), {"settings_topic": "all"}, [])
    answer = compose_mod.compose_answer(
        "league_settings", "What are my league settings?", pack)
    assert "Week 11" in answer["text"]
    assert "Week 15" in answer["text"]
    assert "waiver priority" in answer["text"]


def test_deadline_passed_and_unset_shapes(monkeypatch):
    _no_faab(monkeypatch)
    ctx = {"league": {"settings": {"trade_deadline": 8}}}
    pack, _ = evidence.build_league_settings(
        _bundle(ctx), {"settings_topic": "deadline"}, [])
    assert pack["deadline"]["passed"] is True
    answer = compose_mod.compose_answer(
        "league_settings", "When is the trade deadline?", pack)
    assert "trading is closed" in answer["text"]

    ctx = {"league": {"settings": {"trade_deadline": 0}}}
    pack, _ = evidence.build_league_settings(
        _bundle(ctx), {"settings_topic": "deadline"}, [])
    assert pack["deadline"]["kind"] == "none"
    answer = compose_mod.compose_answer(
        "league_settings", "When is the trade deadline?", pack)
    assert "no trade deadline set" in answer["text"]


def test_espn_deadline_timestamp_shape(monkeypatch):
    _no_faab(monkeypatch)
    import time as _time
    ts = int(_time.time()) + 14 * 86400
    ctx = {"league": {"settings": {}},
           "league_settings": {"trade_deadline_ts": ts}}
    pack, _ = evidence.build_league_settings(
        _bundle(ctx), {"settings_topic": "deadline"}, [])
    assert pack["deadline"]["kind"] == "date"
    assert pack["deadline"]["passed"] is False
    answer = compose_mod.compose_answer(
        "league_settings", "When is the trade deadline?", pack)
    assert pack["deadline"]["date_label"] in answer["text"]


def test_no_settings_loaded_declines_honestly(monkeypatch):
    _no_faab(monkeypatch)
    pack, _ = evidence.build_league_settings(
        _bundle({}), {"settings_topic": "deadline"}, [])
    assert pack["available"] is False
    answer = compose_mod.compose_answer(
        "league_settings", "When is the trade deadline?", pack)
    assert "not" in answer["text"]
    assert "Week" not in answer["text"]
