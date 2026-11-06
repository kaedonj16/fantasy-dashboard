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


# -- scoring + format topics (PR B) -------------------------------------------
#
# Kaedon's follow-up (2026-11-06): the deadline fix landed, but the same
# class of settings questions still declined (PPR or half PPR, passing
# TD points, superflex, team count) or misrouted (bench spots went to
# start_sit on the bare "bench" anchor). Scoring values live on the
# ctx's scoring-settings keys and format facts on roster_positions /
# roster counts; both topics extend the league_settings pack.

@pytest.mark.parametrize("message,topic", [
    ("Is my league PPR or half PPR?", "scoring"),
    ("How many points is a passing TD in my league?", "scoring"),
    ("What is a passing TD worth?", "scoring"),
    ("How many points per reception do we get?", "scoring"),
    ("Is this a superflex league?", "format"),
    ("Are we superflex or 1QB?", "format"),
    ("How many teams are in my league?", "format"),
    ("How many bench spots do we have?", "format"),
    ("How many roster spots do we have?", "format"),
    ("What is our roster size?", "format"),
])
def test_keyword_route_scoring_and_format(message, topic):
    out = _keyword_route(message)
    assert out["intent"] == "league_settings"
    assert out["settings_topic"] == topic


@pytest.mark.parametrize("message,intent", [
    # Benching A PLAYER is start/sit advice, not a settings question:
    # only the spots/size/count phrasing is a settings shape.
    ("Should I bench Gibbs?", "start_sit"),
    ("Should I bench Josh Allen this week?", "start_sit"),
    # A metric question that mentions points per reception is not
    # asking for the league's scoring settings.
    ("Who leads the league in points per reception?", "metric"),
])
def test_keyword_route_scoring_format_neighbors(message, intent):
    assert _keyword_route(message)["intent"] == intent


def _sf_ctx():
    positions = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX",
                 "BN", "BN", "BN", "BN", "BN", "BN", "BN", "IR"]
    return {"league": {"name": "KC fantasy-yearly", "total_rosters": 10,
                       "settings": {"trade_deadline": 11,
                                    "playoff_week_start": 15,
                                    "playoff_teams": 6, "waiver_type": 0}},
            "raw_scoring_settings": {"rec": 0.5, "pass_td": 4,
                                     "rush_td": 6, "rec_td": 6,
                                     "bonus_rec_te": 0.5},
            "roster_positions": positions,
            "rosters": [{"roster_id": i} for i in range(1, 11)],
            "total_rosters": 10}


def test_scoring_answers_half_ppr_and_td_points(monkeypatch):
    _no_faab(monkeypatch)
    pack, _tools = evidence.build_league_settings(
        _bundle(_sf_ctx()), {"settings_topic": "scoring"}, [])
    assert pack["scoring"]["format"] == "half"
    assert pack["scoring"]["points_per_reception"] == 0.5
    assert pack["scoring"]["pass_td"] == 4.0
    answer = compose_mod.compose_answer(
        "league_settings", "Is my league PPR or half PPR?", pack)
    assert "Half PPR" in answer["text"]
    assert "0.5" in answer["text"]
    answer = compose_mod.compose_answer(
        "league_settings", "How many points is a passing TD?", pack)
    assert "passing TD is worth 4 points" in answer["text"]
    assert {"label": "Passing TD points", "value": "4"} in answer["receipts"]


def test_format_answers_superflex_teams_and_bench(monkeypatch):
    _no_faab(monkeypatch)
    pack, _tools = evidence.build_league_settings(
        _bundle(_sf_ctx()), {"settings_topic": "format"}, [])
    assert pack["format"] == {"superflex": True, "teams": 10,
                              "bench_spots": 7, "roster_size": 16}
    answer = compose_mod.compose_answer(
        "league_settings", "Is this a superflex league?", pack)
    assert "superflex" in answer["text"]
    assert "10 teams" in answer["text"]
    answer = compose_mod.compose_answer(
        "league_settings", "How many bench spots do we have?", pack)
    assert {"label": "Bench spots", "value": "7"} in answer["receipts"]


def test_format_answers_1qb_without_superflex_slot(monkeypatch):
    _no_faab(monkeypatch)
    ctx = _sf_ctx()
    ctx["roster_positions"] = ["QB", "RB", "WR", "TE", "FLEX", "BN", "BN"]
    pack, _tools = evidence.build_league_settings(
        _bundle(ctx), {"settings_topic": "format"}, [])
    assert pack["format"]["superflex"] is False
    assert pack["format"]["bench_spots"] == 2
    answer = compose_mod.compose_answer(
        "league_settings", "Is this a superflex league?", pack)
    assert "1QB" in answer["text"]


def test_scoring_and_format_unknown_without_the_feed_fields(monkeypatch):
    # Blackedraw's settings carry no scoring-settings dict and no
    # roster_positions: both topics answer honestly unknown instead of
    # defaulting to PPR / a stock league shape.
    _no_faab(monkeypatch)
    pack, _ = evidence.build_league_settings(
        _bundle(_blackedraw_ctx()), {"settings_topic": "scoring"}, [])
    assert pack["scoring"] is None
    answer = compose_mod.compose_answer(
        "league_settings", "Is my league PPR or half PPR?", pack)
    assert "don't have your league's scoring settings" in answer["text"]

    pack, _ = evidence.build_league_settings(
        _bundle(_blackedraw_ctx()), {"settings_topic": "format"}, [])
    assert pack["format"] is None
    answer = compose_mod.compose_answer(
        "league_settings", "How many teams are in my league?", pack)
    assert "don't have your league's format details" in answer["text"]


# -- draft start time (PR B) ----------------------------------------------------

@pytest.mark.parametrize("message", [
    "When is our draft?",
    "When is the draft?",
    "When does the draft start?",
])
def test_keyword_route_when_is_the_draft(message):
    assert _keyword_route(message)["intent"] == "draft"


def _stub_predraft(monkeypatch, start_ms=1754006400000):
    monkeypatch.setattr(evidence, "league_drafts", lambda *a: [
        {"draft_id": "D1", "status": "pre_draft", "season": 2026,
         "type": "snake", "start_time": start_ms}])
    monkeypatch.setattr(evidence, "draft_detail", lambda draft_id: {
        "status": "pre_draft", "season": 2026, "type": "snake",
        "start_time": start_ms,
        "settings": {"rounds": 15, "teams": 10},
        "slot_to_roster_id": {}})
    monkeypatch.setattr(evidence, "draft_picks_live", lambda draft_id: [])
    monkeypatch.setattr(evidence, "league_traded_picks",
                        lambda league_id: [])
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda: [])
    monkeypatch.setattr(evidence, "_index", lambda: {})
    monkeypatch.setattr(evidence, "active_rookie_class", lambda: 2027)
    monkeypatch.setattr(evidence, "rookie_rankings", lambda year: [])
    ctx = {"rosters": [{"roster_id": i, "owner_id": str(i)}
                       for i in range(1, 11)],
           "users": [],
           "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "BN"]}
    return _bundle(ctx)


def test_draft_pack_carries_scheduled_start(monkeypatch):
    pack, _tools = evidence.build_draft(_stub_predraft(monkeypatch), {}, [])
    assert pack["start_time"] == 1754006400000
    assert pack["start_label"] == "Thursday, July 31 at 8:00 PM EDT"
    receipts = compose_mod._required_receipts("draft", pack)
    assert {"label": "Draft starts",
            "value": "Thursday, July 31 at 8:00 PM EDT"} in receipts


def test_draft_pack_without_start_time_stays_unknown(monkeypatch):
    monkeypatch.setattr(evidence, "league_drafts", lambda *a: [
        {"draft_id": "D1", "status": "pre_draft", "season": 2026,
         "type": "snake"}])
    monkeypatch.setattr(evidence, "draft_detail", lambda draft_id: {
        "status": "pre_draft", "season": 2026, "type": "snake",
        "settings": {"rounds": 15, "teams": 10},
        "slot_to_roster_id": {}})
    monkeypatch.setattr(evidence, "draft_picks_live", lambda draft_id: [])
    monkeypatch.setattr(evidence, "league_traded_picks",
                        lambda league_id: [])
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda: [])
    monkeypatch.setattr(evidence, "_index", lambda: {})
    monkeypatch.setattr(evidence, "active_rookie_class", lambda: 2027)
    monkeypatch.setattr(evidence, "rookie_rankings", lambda year: [])
    pack, _tools = evidence.build_draft(_bundle({}), {}, [])
    assert "start_time" not in pack
    labels = [r["label"] for r in
              compose_mod._required_receipts("draft", pack)]
    assert "Draft starts" not in labels


# -- router catch-all trims (PR B) ----------------------------------------------

@pytest.mark.parametrize("message,intent", [
    # Trade news about a named player is a player question, not a deal.
    ("Any trade rumors about Jonathan Taylor?", "player"),
    ("Is Jonathan Taylor on the trade block?", "player"),
    # Deal-shaped questions keep trade_eval, rumors or not.
    ("Should I trade Jonathan Taylor for Gibbs?", "trade_eval"),
    ("Should I trade for Jonathan Taylor after the trade rumors?",
     "trade_eval"),
    # A score question asks for the number (matchup pack); the digest
    # shapes keep this_week.
    ("How many points did my team score this week?", "matchup"),
    ("What did my team score this week?", "matchup"),
    ("How did my team do this week?", "this_week"),
])
def test_keyword_route_catchall_trims(message, intent):
    assert _keyword_route(message)["intent"] == intent
